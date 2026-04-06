"""
Ensembles adicionales calculados a partir de datos ya existentes:
1. Ensemble MV (Majority Voting): Votación por mayoría de prompts individuales
2. Ensemble Entropy Filtering: Filtra por baja entropía y evalúa MEAN, MV, MAX

Usa el mismo formato de columnas que calibration_analysis.py
"""

import os
import json
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIGURACIÓN
# =========================
RESULTS_DIR = "../results/resultados_paper"
INPUT_JSONL_BLIP = os.path.join(RESULTS_DIR, "h1_blip2_unified_all_samples.jsonl")
INPUT_JSONL_CLIP = os.path.join(RESULTS_DIR, "h1_clip_unified_all_samples.jsonl")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "img")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FUNCIONES AUXILIARES (desde calibration_analysis.py)
# =========================

ECE_BINS = 15

def safe_log(x, eps=1e-12):
    return math.log(max(x, eps))

def calculate_entropy(probs, eps=1e-12):
    """Calcula la entropía de una distribución de probabilidades."""
    probs = np.array(probs)
    probs = np.clip(probs, eps, 1.0)
    return float(-(probs * np.log(probs)).sum())

def multiclass_brier(probs, true_index):
    """Brier score multiclase."""
    y = np.zeros_like(probs, dtype=float)
    y[true_index] = 1.0
    return float(np.mean((probs - y) ** 2))

def nll(probs, true_index, eps=1e-12):
    """Negative log-likelihood."""
    return float(-safe_log(float(probs[true_index]), eps=eps))

def ece_score(confidences, correct, n_bins=15):
    """
    Expected Calibration Error.
    confidences: array shape (N,) = max prob
    correct: array shape (N,) in {0,1}
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(confidences)
    if N == 0:
        return float("nan")

    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        nk = int(mask.sum())
        if nk == 0:
            continue

        acc_k = float(correct[mask].mean())
        conf_k = float(confidences[mask].mean())
        ece += (nk / N) * abs(acc_k - conf_k)

    return float(ece)

def get_true_index(row):
    """Mapea true_label a índice usando label_order."""
    label_order = row["label_order"]
    true_label = row["true_label"]
    
    if not isinstance(label_order, list):
        return None
    
    try:
        return int(label_order.index(true_label))
    except ValueError:
        return None

def load_samples_from_jsonl(jsonl_path):
    """Carga todas las muestras de un archivo JSONL."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

# =========================
# ENSEMBLE MAJORITY VOTING
# =========================

def compute_majority_voting_ensemble(samples_df, model_name):
    """
    Calcula ensemble por votación mayoritaria.
    Para cada muestra, cuenta qué clase predicen más prompts individuales.
    FORMATO: mismo que calibration_analysis.py
    """
    print(f"\n{'='*80}")
    print(f"ENSEMBLE MAJORITY VOTING - {model_name.upper()}")
    print(f"{'='*80}")
    
    results = []
    
    # Agrupar por dataset, mode, y prompt_set base
    for (dataset, mode), dataset_group in samples_df.groupby(['dataset', 'mode']):
        # Procesar solo prompts individuales
        individual_prompts = dataset_group[~dataset_group['prompt_set'].str.contains('ensemble')]
        
        if len(individual_prompts) == 0:
            continue
        
        # Identificar tipos base (generic/specific)
        base_types = set()
        for ps in individual_prompts['prompt_set'].unique():
            if 'generic' in ps:
                base_types.add('generic')
            elif 'specific' in ps:
                base_types.add('specific')
        
        for base_type in base_types:
            # Obtener todos los prompts del tipo base
            all_prompts = individual_prompts[
                individual_prompts['prompt_set'].str.contains(base_type)
            ]
            
            if len(all_prompts) == 0:
                continue
            
            # Agrupar por sample_id
            sample_ids = sorted(all_prompts['sample_id'].unique())
            
            sample_results = []
            
            for sample_id in sample_ids:
                sample_prompts = all_prompts[all_prompts['sample_id'] == sample_id]
                
                if len(sample_prompts) == 0:
                    continue
                
                # Votación
                votes = sample_prompts['pred_label'].tolist()
                vote_counts = Counter(votes)
                mv_pred = vote_counts.most_common(1)[0][0]
                mv_confidence = vote_counts[mv_pred] / len(votes)
                
                # Obtener datos de la primera muestra
                first_sample = sample_prompts.iloc[0]
                true_label = first_sample['true_label']
                label_order = first_sample['label_order']
                
                # Para calcular probs MV: basadas en proporción de votos
                mv_probs = np.zeros(len(label_order))
                for label, count in vote_counts.items():
                    if label in label_order:
                        idx = label_order.index(label)
                        mv_probs[idx] = count / len(votes)
                
                sample_results.append({
                    'sample_id': sample_id,
                    'dataset': dataset,
                    'mode': mode,
                    'prompt_set': f"{base_type}_ensemble_mv",
                    'true_label': true_label,
                    'pred_label': mv_pred,
                    'confidence': mv_confidence,
                    'probs': mv_probs.tolist(),
                    'label_order': label_order
                })
            
            if len(sample_results) == 0:
                continue
            
            # Crear DataFrame y calcular métricas agregadas
            mv_df = pd.DataFrame(sample_results)
            
            # Calcular métricas por muestra
            mv_df['probs_np'] = mv_df['probs'].apply(lambda x: np.array(x, dtype=float))
            mv_df['true_index'] = mv_df.apply(get_true_index, axis=1)
            mv_df = mv_df.dropna(subset=['true_index']).copy()
            mv_df['true_index'] = mv_df['true_index'].astype(int)
            mv_df['correct'] = (mv_df['pred_label'] == mv_df['true_label']).astype(int)
            
            mv_df['entropy'] = mv_df['probs_np'].apply(calculate_entropy)
            mv_df['brier'] = mv_df.apply(lambda r: multiclass_brier(r['probs_np'], r['true_index']), axis=1)
            mv_df['nll'] = mv_df.apply(lambda r: nll(r['probs_np'], r['true_index']), axis=1)
            
            # Métricas agregadas
            y_true = mv_df['true_label'].tolist()
            y_pred = mv_df['pred_label'].tolist()
            
            acc = accuracy_score(y_true, y_pred)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            ece = ece_score(mv_df['confidence'].values, mv_df['correct'].values, n_bins=ECE_BINS)
            
            results.append({
                'dataset': dataset,
                'mode': mode,
                'prompt_set': f"{base_type}_ensemble_mv",
                'n_samples': int(len(mv_df)),
                'accuracy': float(acc),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                'brier_mean': float(mv_df['brier'].mean()),
                'nll_mean': float(mv_df['nll'].mean()),
                'ece': float(ece),
                'entropy_mean': float(mv_df['entropy'].mean()),
                'entropy_std': float(mv_df['entropy'].std(ddof=0)) if len(mv_df) > 1 else 0.0,
                'confidence_mean': float(mv_df['confidence'].mean()),
                'confidence_std': float(mv_df['confidence'].std(ddof=0)) if len(mv_df) > 1 else 0.0,
                'accuracy_by_conf_bin_bins': ECE_BINS,
            })
            
            print(f"\n{dataset} - {mode} - {base_type}_ensemble_mv:")
            print(f"  Acc: {acc:.4f} | F1-M: {f1_macro:.4f} | F1-W: {f1_weighted:.4f}")
            print(f"  Brier: {mv_df['brier'].mean():.4f} | ECE: {ece:.4f} | NLL: {mv_df['nll'].mean():.4f}")
            
            # Comparar con MEAN y MAX existentes
            existing_mean = samples_df[
                (samples_df['dataset'] == dataset) & 
                (samples_df['mode'] == mode) &
                (samples_df['prompt_set'] == f"{base_type}_ensemble_mean")
            ]
            existing_max = samples_df[
                (samples_df['dataset'] == dataset) & 
                (samples_df['mode'] == mode) &
                (samples_df['prompt_set'] == f"{base_type}_ensemble_max")
            ]
            
            if len(existing_mean) > 0:
                mean_acc = (existing_mean['true_label'] == existing_mean['pred_label']).mean()
                print(f"  Comparación MEAN: {mean_acc:.4f} {'(MV mejor)' if acc > mean_acc else '(MEAN mejor)'}")
            
            if len(existing_max) > 0:
                max_acc = (existing_max['true_label'] == existing_max['pred_label']).mean()
                print(f"  Comparación MAX:  {max_acc:.4f} {'(MV mejor)' if acc > max_acc else '(MAX mejor)'}")
    
    return pd.DataFrame(results)

# =========================
# ENSEMBLE ENTROPY FILTERING
# =========================

def compute_entropy_filtering_ensemble(samples_df, model_name, entropy_percentile=50):
    """
    Filtra muestras por baja entropía del ensemble MEAN y evalúa con MEAN, MV, MAX.
    FORMATO: mismo que calibration_analysis.py
    
    Args:
        entropy_percentile: Percentil de entropía para filtrar (por defecto 50 = mediana)
    """
    print(f"\n{'='*80}")
    print(f"ENSEMBLE ENTROPY FILTERING - {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Filtrando por entropía <= percentil {entropy_percentile}")
    
    results = []
    
    # Procesar solo ensembles MEAN ya calculados
    ensemble_mean_samples = samples_df[samples_df['prompt_set'].str.contains('ensemble_mean')]
    
    for (dataset, mode, prompt_set), group in ensemble_mean_samples.groupby(['dataset', 'mode', 'prompt_set']):
        # Extraer tipo base
        if 'generic' in prompt_set:
            base_type = 'generic'
        elif 'specific' in prompt_set:
            base_type = 'specific'
        else:
            continue
        
        # Calcular entropía para cada muestra
        group = group.copy()
        group['entropy'] = group['probs'].apply(calculate_entropy)
        
        # Calcular umbral
        entropy_threshold = np.percentile(group['entropy'].values, entropy_percentile)
        
        # Filtrar muestras con baja entropía
        low_entropy_group = group[group['entropy'] <= entropy_threshold].copy()
        
        print(f"\n{dataset} - {mode} - {base_type}:")
        print(f"  Total muestras: {len(group)}")
        print(f"  Entropía umbral (p{entropy_percentile}): {entropy_threshold:.4f}")
        print(f"  Muestras filtradas: {len(low_entropy_group)} ({len(low_entropy_group)/len(group)*100:.1f}%)")
        
        if len(low_entropy_group) == 0:
            print("  ⚠️  Sin muestras después del filtrado")
            continue
        
        # ===== MEAN con filtrado de entropía =====
        low_entropy_group['probs_np'] = low_entropy_group['probs'].apply(lambda x: np.array(x, dtype=float))
        low_entropy_group['true_index'] = low_entropy_group.apply(get_true_index, axis=1)
        low_entropy_group = low_entropy_group.dropna(subset=['true_index']).copy()
        low_entropy_group['true_index'] = low_entropy_group['true_index'].astype(int)
        low_entropy_group['correct'] = (low_entropy_group['pred_label'] == low_entropy_group['true_label']).astype(int)
        
        low_entropy_group['brier'] = low_entropy_group.apply(
            lambda r: multiclass_brier(r['probs_np'], r['true_index']), axis=1
        )
        low_entropy_group['nll'] = low_entropy_group.apply(
            lambda r: nll(r['probs_np'], r['true_index']), axis=1
        )
        
        y_true = low_entropy_group['true_label'].tolist()
        y_pred = low_entropy_group['pred_label'].tolist()
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        ece = ece_score(low_entropy_group['confidence'].values, low_entropy_group['correct'].values, n_bins=ECE_BINS)
        
        results.append({
            'dataset': dataset,
            'mode': mode,
            'prompt_set': f"{base_type}_ensemble_mean_entropy_p{entropy_percentile}",
            'n_samples': int(len(low_entropy_group)),
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'brier_mean': float(low_entropy_group['brier'].mean()),
            'nll_mean': float(low_entropy_group['nll'].mean()),
            'ece': float(ece),
            'entropy_mean': float(low_entropy_group['entropy'].mean()),
            'entropy_std': float(low_entropy_group['entropy'].std(ddof=0)) if len(low_entropy_group) > 1 else 0.0,
            'confidence_mean': float(low_entropy_group['confidence'].mean()),
            'confidence_std': float(low_entropy_group['confidence'].std(ddof=0)) if len(low_entropy_group) > 1 else 0.0,
            'accuracy_by_conf_bin_bins': ECE_BINS,
        })
        
        print(f"\n  MEAN (filtrado por entropía p{entropy_percentile}):")
        print(f"    Acc: {acc:.4f} | F1-M: {f1_macro:.4f} | F1-W: {f1_weighted:.4f}")
        print(f"    Brier: {low_entropy_group['brier'].mean():.4f} | ECE: {ece:.4f} | NLL: {low_entropy_group['nll'].mean():.4f}")
        
        # ===== MV con filtrado de entropía =====
        sample_ids_filtered = low_entropy_group['sample_id'].tolist()
        all_prompts = samples_df[
            (samples_df['dataset'] == dataset) & 
            (samples_df['mode'] == mode) &
            (samples_df['prompt_set'].str.contains(base_type)) &
            (~samples_df['prompt_set'].str.contains('ensemble')) &
            (samples_df['sample_id'].isin(sample_ids_filtered))
        ]
        
        if len(all_prompts) > 0:
            sample_results_mv = []
            
            for sample_id in sample_ids_filtered:
                sample_prompts = all_prompts[all_prompts['sample_id'] == sample_id]
                if len(sample_prompts) == 0:
                    continue
                
                votes = sample_prompts['pred_label'].tolist()
                vote_counts = Counter(votes)
                mv_pred = vote_counts.most_common(1)[0][0]
                mv_confidence = vote_counts[mv_pred] / len(votes)
                
                first_sample = sample_prompts.iloc[0]
                true_label = first_sample['true_label']
                label_order = first_sample['label_order']
                
                # Para calcular probs MV: basadas en proporción de votos
                mv_probs = np.zeros(len(label_order))
                for label, count in vote_counts.items():
                    if label in label_order:
                        idx = label_order.index(label)
                        mv_probs[idx] = count / len(votes)
                
                sample_results_mv.append({
                    'sample_id': sample_id,
                    'dataset': dataset,
                    'mode': mode,
                    'prompt_set': f"{base_type}_ensemble_mv_entropy_p{entropy_percentile}",
                    'true_label': true_label,
                    'pred_label': mv_pred,
                    'confidence': mv_confidence,
                    'probs': mv_probs.tolist(),
                    'label_order': label_order
                })
            
            if len(sample_results_mv) > 0:
                mv_df = pd.DataFrame(sample_results_mv)
                mv_df['probs_np'] = mv_df['probs'].apply(lambda x: np.array(x, dtype=float))
                mv_df['true_index'] = mv_df.apply(get_true_index, axis=1)
                mv_df = mv_df.dropna(subset=['true_index']).copy()
                mv_df['true_index'] = mv_df['true_index'].astype(int)
                mv_df['correct'] = (mv_df['pred_label'] == mv_df['true_label']).astype(int)
                
                mv_df['entropy'] = mv_df['probs_np'].apply(calculate_entropy)
                mv_df['brier'] = mv_df.apply(lambda r: multiclass_brier(r['probs_np'], r['true_index']), axis=1)
                mv_df['nll'] = mv_df.apply(lambda r: nll(r['probs_np'], r['true_index']), axis=1)
                
                y_true_mv = mv_df['true_label'].tolist()
                y_pred_mv = mv_df['pred_label'].tolist()
                
                acc_mv = accuracy_score(y_true_mv, y_pred_mv)
                f1_macro_mv = f1_score(y_true_mv, y_pred_mv, average='macro', zero_division=0)
                f1_weighted_mv = f1_score(y_true_mv, y_pred_mv, average='weighted', zero_division=0)
                ece_mv = ece_score(mv_df['confidence'].values, mv_df['correct'].values, n_bins=ECE_BINS)
                
                results.append({
                    'dataset': dataset,
                    'mode': mode,
                    'prompt_set': f"{base_type}_ensemble_mv_entropy_p{entropy_percentile}",
                    'n_samples': int(len(mv_df)),
                    'accuracy': float(acc_mv),
                    'f1_macro': float(f1_macro_mv),
                    'f1_weighted': float(f1_weighted_mv),
                    'brier_mean': float(mv_df['brier'].mean()),
                    'nll_mean': float(mv_df['nll'].mean()),
                    'ece': float(ece_mv),
                    'entropy_mean': float(mv_df['entropy'].mean()),
                    'entropy_std': float(mv_df['entropy'].std(ddof=0)) if len(mv_df) > 1 else 0.0,
                    'confidence_mean': float(mv_df['confidence'].mean()),
                    'confidence_std': float(mv_df['confidence'].std(ddof=0)) if len(mv_df) > 1 else 0.0,
                    'accuracy_by_conf_bin_bins': ECE_BINS,
                })
                
                print(f"\n  MV (filtrado por entropía p{entropy_percentile}):")
                print(f"    Acc: {acc_mv:.4f} | F1-M: {f1_macro_mv:.4f} | F1-W: {f1_weighted_mv:.4f}")
                print(f"    Brier: {mv_df['brier'].mean():.4f} | ECE: {ece_mv:.4f} | NLL: {mv_df['nll'].mean():.4f}")
        
        # ===== MAX con filtrado de entropía =====
        existing_max = samples_df[
            (samples_df['dataset'] == dataset) & 
            (samples_df['mode'] == mode) &
            (samples_df['prompt_set'] == f"{base_type}_ensemble_max") &
            (samples_df['sample_id'].isin(sample_ids_filtered))
        ].copy()
        
        if len(existing_max) > 0:
            existing_max['probs_np'] = existing_max['probs'].apply(lambda x: np.array(x, dtype=float))
            existing_max['true_index'] = existing_max.apply(get_true_index, axis=1)
            existing_max = existing_max.dropna(subset=['true_index']).copy()
            existing_max['true_index'] = existing_max['true_index'].astype(int)
            existing_max['correct'] = (existing_max['pred_label'] == existing_max['true_label']).astype(int)
            
            existing_max['entropy'] = existing_max['probs_np'].apply(calculate_entropy)
            existing_max['brier'] = existing_max.apply(
                lambda r: multiclass_brier(r['probs_np'], r['true_index']), axis=1
            )
            existing_max['nll'] = existing_max.apply(
                lambda r: nll(r['probs_np'], r['true_index']), axis=1
            )
            
            y_true_max = existing_max['true_label'].tolist()
            y_pred_max = existing_max['pred_label'].tolist()
            
            acc_max = accuracy_score(y_true_max, y_pred_max)
            f1_macro_max = f1_score(y_true_max, y_pred_max, average='macro', zero_division=0)
            f1_weighted_max = f1_score(y_true_max, y_pred_max, average='weighted', zero_division=0)
            ece_max = ece_score(existing_max['confidence'].values, existing_max['correct'].values, n_bins=ECE_BINS)
            
            results.append({
                'dataset': dataset,
                'mode': mode,
                'prompt_set': f"{base_type}_ensemble_max_entropy_p{entropy_percentile}",
                'n_samples': int(len(existing_max)),
                'accuracy': float(acc_max),
                'f1_macro': float(f1_macro_max),
                'f1_weighted': float(f1_weighted_max),
                'brier_mean': float(existing_max['brier'].mean()),
                'nll_mean': float(existing_max['nll'].mean()),
                'ece': float(ece_max),
                'entropy_mean': float(existing_max['entropy'].mean()),
                'entropy_std': float(existing_max['entropy'].std(ddof=0)) if len(existing_max) > 1 else 0.0,
                'confidence_mean': float(existing_max['confidence'].mean()),
                'confidence_std': float(existing_max['confidence'].std(ddof=0)) if len(existing_max) > 1 else 0.0,
                'accuracy_by_conf_bin_bins': ECE_BINS,
            })
            
            print(f"\n  MAX (filtrado por entropía p{entropy_percentile}):")
            print(f"    Acc: {acc_max:.4f} | F1-M: {f1_macro_max:.4f} | F1-W: {f1_weighted_max:.4f}")
            print(f"    Brier: {existing_max['brier'].mean():.4f} | ECE: {ece_max:.4f} | NLL: {existing_max['nll'].mean():.4f}")
    
    return pd.DataFrame(results)

# =========================
# VISUALIZACIÓN
# =========================

def plot_ensemble_comparison(results_df, model_name):
    """Visualiza comparación de todos los métodos de ensemble."""
    
    # Separar por método
    methods = results_df['method'].unique()
    datasets = results_df['dataset'].unique()
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    fig.suptitle(f'Comparación de Ensembles - {model_name.upper()}', fontsize=16, fontweight='bold')
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = results_df[results_df['dataset'] == dataset]
        
        # Agrupar por método y ensemble_type
        plot_data = []
        labels = []
        
        for ensemble_type in ['generic', 'specific']:
            type_data = data[data['ensemble_type'] == ensemble_type]
            if len(type_data) == 0:
                continue
            
            for method in type_data['method'].unique():
                method_data = type_data[type_data['method'] == method]
                if len(method_data) > 0:
                    plot_data.append(method_data['accuracy'].values[0])
                    labels.append(f"{ensemble_type[:3]}_{method[:6]}")
        
        if len(plot_data) > 0:
            x = np.arange(len(plot_data))
            bars = ax.bar(x, plot_data, edgecolor='black', linewidth=1.5)
            
            # Colorear por tipo
            for i, label in enumerate(labels):
                if 'gen' in label:
                    bars[i].set_color('#3498db')
                else:
                    bars[i].set_color('#e74c3c')
            
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(dataset.capitalize(), fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])
            
            # Añadir valores
            for i, (bar, val) in enumerate(zip(bars, plot_data)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{model_name}_additional_ensembles.png'),
                dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico guardado: {OUTPUT_DIR}/{model_name}_additional_ensembles.png")
    plt.close()

# =========================
# MAIN
# =========================

def main():
    print("="*80)
    print("ENSEMBLES ADICIONALES: MAJORITY VOTING Y ENTROPY FILTERING")
    print("="*80)
    
    all_results = []
    
    # Procesar CLIP
    if os.path.exists(INPUT_JSONL_CLIP):
        print(f"\n{'#'*80}")
        print("# PROCESANDO CLIP")
        print(f"{'#'*80}")
        
        clip_samples = load_samples_from_jsonl(INPUT_JSONL_CLIP)
        clip_df = pd.DataFrame(clip_samples)
        
        # Majority Voting
        mv_results_clip = compute_majority_voting_ensemble(clip_df, 'clip')
        all_results.append(mv_results_clip)
        
        # Entropy Filtering
        entropy_results_clip = compute_entropy_filtering_ensemble(clip_df, 'clip', entropy_percentile=50)
        all_results.append(entropy_results_clip)
        
        # Guardar resultados CLIP
        if len(mv_results_clip) > 0 or len(entropy_results_clip) > 0:
            combined_clip = pd.concat([mv_results_clip, entropy_results_clip], ignore_index=True)
            combined_clip = combined_clip.sort_values(['dataset', 'mode', 'prompt_set'])
            
            output_clip_csv = os.path.join(RESULTS_DIR, "offline_summary_clip_additional_ensembles.csv")
            combined_clip.to_csv(output_clip_csv, index=False)
            print(f"\n✅ Resultados CLIP guardados en: {output_clip_csv}")
            print(f"\nPreview CLIP:")
            print(combined_clip.head(20).to_string(index=False))
            
            # Mejor por dataset
            best_clip = combined_clip.sort_values(['dataset', 'accuracy'], ascending=[True, False]).groupby('dataset').head(1)
            best_clip_path = os.path.join(RESULTS_DIR, "offline_summary_clip_additional_best_by_dataset.csv")
            best_clip.to_csv(best_clip_path, index=False)
            print(f"✅ Best-by-dataset CLIP guardado en: {best_clip_path}")
    
    # Procesar BLIP2
    if os.path.exists(INPUT_JSONL_BLIP):
        print(f"\n{'#'*80}")
        print("# PROCESANDO BLIP2")
        print(f"{'#'*80}")
        
        blip_samples = load_samples_from_jsonl(INPUT_JSONL_BLIP)
        blip_df = pd.DataFrame(blip_samples)
        
        # Majority Voting
        mv_results_blip = compute_majority_voting_ensemble(blip_df, 'blip2')
        all_results.append(mv_results_blip)
        
        # Entropy Filtering
        entropy_results_blip = compute_entropy_filtering_ensemble(blip_df, 'blip2', entropy_percentile=50)
        all_results.append(entropy_results_blip)
        
        # Guardar resultados BLIP2
        if len(mv_results_blip) > 0 or len(entropy_results_blip) > 0:
            combined_blip = pd.concat([mv_results_blip, entropy_results_blip], ignore_index=True)
            combined_blip = combined_blip.sort_values(['dataset', 'mode', 'prompt_set'])
            
            output_blip_csv = os.path.join(RESULTS_DIR, "offline_summary_blip2_additional_ensembles.csv")
            combined_blip.to_csv(output_blip_csv, index=False)
            print(f"\n✅ Resultados BLIP2 guardados en: {output_blip_csv}")
            print(f"\nPreview BLIP2:")
            print(combined_blip.head(20).to_string(index=False))
            
            # Mejor por dataset
            best_blip = combined_blip.sort_values(['dataset', 'accuracy'], ascending=[True, False]).groupby('dataset').head(1)
            best_blip_path = os.path.join(RESULTS_DIR, "offline_summary_blip2_additional_best_by_dataset.csv")
            best_blip.to_csv(best_blip_path, index=False)
            print(f"✅ Best-by-dataset BLIP2 guardado en: {best_blip_path}")
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)

if __name__ == "__main__":
    main()
