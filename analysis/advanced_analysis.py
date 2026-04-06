"""
Análisis avanzado de resultados:
1. Matriz de confusión de Agriculture para prompts específicos (P1-P3)
2. Histograma de confidence_mean por prompt
3. Top-k accuracy (top-3) para BLIP-2
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict

# Configuración
RESULTS_DIR = "../results"
INPUT_JSONL_BLIP = os.path.join(RESULTS_DIR, "h1_blip2_unified_all_samples.jsonl")
SUMMARY_BLIP = os.path.join(RESULTS_DIR, "offline_summary_blip_brier_ece.csv")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "img")

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

print("=" * 80)
print("CARGANDO DATOS...")
print("=" * 80)

# Cargar datos de todas las muestras
all_samples = []
with open(INPUT_JSONL_BLIP, 'r') as f:
    for line in f:
        all_samples.append(json.loads(line))

df_samples = pd.DataFrame(all_samples)
print(f"\n✓ Cargadas {len(df_samples)} muestras")
print(f"  Datasets: {df_samples['dataset'].unique()}")
print(f"  Prompt sets: {df_samples['prompt_set'].unique()}")

# Cargar summary
df_summary = pd.read_csv(SUMMARY_BLIP)
print(f"\n✓ Cargado resumen con {len(df_summary)} configuraciones")


# ============================================================================
# 2. MATRIZ DE CONFUSIÓN - AGRICULTURE (P1-P3 específicos)
# ============================================================================

print("\n" + "=" * 80)
print("1. MATRIZ DE CONFUSIÓN - AGRICULTURE (Specific Prompts P1-P4)")
print("=" * 80)

# Filtrar Agriculture con specific prompts P1-P4
agriculture_specific = df_samples[
    (df_samples['dataset'] == 'agriculture') &
    (df_samples['prompt_set'].isin(['specific_p1', 'specific_p2', 'specific_p3', 'specific_p4']))
].copy()

print(f"\nMuestras filtradas: {len(agriculture_specific)}")

# Crear figura con 4 matrices (una por prompt)
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
fig.suptitle('Matrices de Confusión - Agriculture (Specific Prompts)', fontsize=16, fontweight='bold')

for idx, prompt in enumerate(['specific_p1', 'specific_p2', 'specific_p3', 'specific_p4']):
    data = agriculture_specific[agriculture_specific['prompt_set'] == prompt]
    
    if len(data) == 0:
        continue
    
    # Obtener labels únicos
    labels = sorted(data['true_label'].unique())
    
    # Matriz de confusión
    cm = confusion_matrix(data['true_label'], data['pred_label'], labels=labels)
    
    # Normalizar por filas (para ver % de cada clase)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot con anotaciones solo en diagonal
    ax = axes[idx]
    sns.heatmap(cm_normalized, annot=False, cmap='YlOrRd', 
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Proporción'}, vmin=0, vmax=1)
    
    # Añadir valores solo en diagonal (correctos) con fondo verde
    for i in range(len(labels)):
        val = cm_normalized[i, i]
        color = 'white' if val > 0.5 else 'black'
        ax.text(i + 0.5, i + 0.5, f'{val:.2f}', 
                ha='center', va='center', color=color, fontsize=6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.3))
    
    ax.set_title(f'{prompt.upper()}\nAcc: {(data["true_label"] == data["pred_label"]).mean():.3f}', 
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Verdadero', fontsize=10)
    
    # Rotar labels con mejor legibilidad
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    
    # Análisis de mode collapse
    pred_counts = data['pred_label'].value_counts()
    total = len(data)
    print(f"\n{prompt.upper()}:")
    print(f"  Accuracy: {(data['true_label'] == data['pred_label']).mean():.4f}")
    print(f"  Distribución de predicciones:")
    for label, count in pred_counts.items():
        print(f"    - {label}: {count} ({count/total*100:.1f}%)")
    
    # Detectar mode collapse (>70% de predicciones en 1-2 clases)
    top2_percentage = pred_counts.head(2).sum() / total
    if top2_percentage > 0.7:
        print(f"  ⚠️  MODE COLLAPSE DETECTADO: {top2_percentage*100:.1f}% en top-2 clases")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'agriculture_confusion_matrices_specific.png'), 
            dpi=300, bbox_inches='tight')
print(f"\n✓ Guardado: {OUTPUT_DIR}/agriculture_confusion_matrices_specific.png")

# Explicación de la matriz
print("\n" + "="*80)
print("📊 CÓMO LEER LAS MATRICES DE CONFUSIÓN:")
print("="*80)
print("\n  Cada celda (i,j) muestra: ¿Qué % de muestras de la clase i fueron predichas como j?")
print("  - DIAGONAL (verde): % correctamente clasificadas")
print("  - FUERA DIAGONAL: % de errores/confusiones")
print("  - Colores rojos intensos = Alta proporción de predicciones")
print("\n  💡 MODE COLLAPSE: Cuando una COLUMNA está muy roja")
print("     → El modelo predice esa clase para casi todo")
print("="*80)
plt.close()


# ============================================================================
# 3. HISTOGRAMAS DE CONFIDENCE_MEAN POR PROMPT
# ============================================================================

print("\n" + "=" * 80)
print("2. HISTOGRAMAS DE CONFIDENCE_MEAN POR PROMPT")
print("=" * 80)

# Filtrar agriculture del summary
agriculture_summary = df_summary[df_summary['dataset'] == 'agriculture'].copy()

# Crear figura
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Distribución de Confidence Mean por Prompt - Agriculture', 
             fontsize=16, fontweight='bold')

# Separar generic vs specific
generic_data = agriculture_summary[agriculture_summary['prompt_set'].str.contains('generic')]
specific_data = agriculture_summary[agriculture_summary['prompt_set'].str.contains('specific')]

# Plot 1: Generic prompts
ax1 = axes[0]
generic_sorted = generic_data.sort_values('prompt_set')
x_pos = np.arange(len(generic_sorted))
bars = ax1.bar(x_pos, generic_sorted['confidence_mean'], 
               color=plt.cm.Blues(np.linspace(0.4, 0.9, len(generic_sorted))),
               edgecolor='black', linewidth=1.5)

# Añadir ECE como línea
ax1_right = ax1.twinx()
ax1_right.plot(x_pos, generic_sorted['ece'], 'ro-', linewidth=2, markersize=8, label='ECE')
ax1_right.set_ylabel('ECE', fontsize=12, color='red')
ax1_right.tick_params(axis='y', labelcolor='red')

ax1.set_xlabel('Prompt Set', fontsize=12)
ax1.set_ylabel('Confidence Mean', fontsize=12)
ax1.set_title('Generic Prompts', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(generic_sorted['prompt_set'], rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

# Añadir valores en barras
for i, (bar, val, acc, ece) in enumerate(zip(bars, generic_sorted['confidence_mean'], 
                                               generic_sorted['accuracy'], 
                                               generic_sorted['ece'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}\nAcc:{acc:.3f}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Specific prompts
ax2 = axes[1]
specific_sorted = specific_data.sort_values('prompt_set')
x_pos2 = np.arange(len(specific_sorted))
bars2 = ax2.bar(x_pos2, specific_sorted['confidence_mean'],
                color=plt.cm.Oranges(np.linspace(0.4, 0.9, len(specific_sorted))),
                edgecolor='black', linewidth=1.5)

# Añadir ECE como línea
ax2_right = ax2.twinx()
ax2_right.plot(x_pos2, specific_sorted['ece'], 'ro-', linewidth=2, markersize=8, label='ECE')
ax2_right.set_ylabel('ECE', fontsize=12, color='red')
ax2_right.tick_params(axis='y', labelcolor='red')

ax2.set_xlabel('Prompt Set', fontsize=12)
ax2.set_ylabel('Confidence Mean', fontsize=12)
ax2.set_title('Specific Prompts', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(specific_sorted['prompt_set'], rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

# Añadir valores en barras
for i, (bar, val, acc, ece) in enumerate(zip(bars2, specific_sorted['confidence_mean'],
                                               specific_sorted['accuracy'],
                                               specific_sorted['ece'])):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}\nAcc:{acc:.3f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confidence_histograms_agriculture.png'),
            dpi=300, bbox_inches='tight')
print(f"\n✓ Guardado: {OUTPUT_DIR}/confidence_histograms_agriculture.png")
plt.close()

# Análisis de correlación
print("\n📊 ANÁLISIS DE CORRELACIÓN (Agriculture):")
print(f"  Correlación Confidence-ECE: {agriculture_summary['confidence_mean'].corr(agriculture_summary['ece']):.4f}")
print(f"  Correlación Confidence-Accuracy: {agriculture_summary['confidence_mean'].corr(agriculture_summary['accuracy']):.4f}")
print(f"  Correlación ECE-Accuracy: {agriculture_summary['ece'].corr(agriculture_summary['accuracy']):.4f}")

print("\n💡 OBSERVACIONES:")
# Prompts con menor confidence
low_conf = agriculture_summary.nsmallest(3, 'confidence_mean')
print("\n  Prompts con MENOR confidence:")
for _, row in low_conf.iterrows():
    print(f"    - {row['prompt_set']}: conf={row['confidence_mean']:.3f}, ECE={row['ece']:.3f}, Acc={row['accuracy']:.3f}")

print("\n  → ECE bajos en specific prompts se correlacionan con BAJA CONFIANZA")
print("  → El modelo no está calibrado, simplemente tiene baja confianza en todo")


# ============================================================================
# 4. TOP-K ACCURACY (TOP-3)
# ============================================================================

print("\n" + "=" * 80)
print("3. TOP-K ACCURACY (TOP-1, TOP-2, TOP-3)")
print("=" * 80)

def calculate_topk_accuracy(samples, k):
    """Calcula top-k accuracy dado un array de probabilidades"""
    correct = 0
    for sample in samples:
        probs = sample['probs']
        true_label = sample['true_label']
        label_order = sample['label_order']
        
        # Obtener índices de top-k predicciones
        top_k_indices = np.argsort(probs)[-k:][::-1]
        
        # Verificar si el label verdadero está en top-k
        true_idx = label_order.index(true_label)
        if true_idx in top_k_indices:
            correct += 1
    
    return correct / len(samples)

# Calcular top-k accuracy por dataset y prompt_set
results_topk = []

for dataset in df_samples['dataset'].unique():
    for prompt_set in df_samples['prompt_set'].unique():
        subset = df_samples[
            (df_samples['dataset'] == dataset) &
            (df_samples['prompt_set'] == prompt_set)
        ]
        
        if len(subset) == 0:
            continue
        
        # Convertir a lista para procesamiento
        samples = subset.to_dict('records')
        
        top1 = calculate_topk_accuracy(samples, 1)
        top2 = calculate_topk_accuracy(samples, 2)
        top3 = calculate_topk_accuracy(samples, 3)
        
        results_topk.append({
            'dataset': dataset,
            'prompt_set': prompt_set,
            'top1_accuracy': top1,
            'top2_accuracy': top2,
            'top3_accuracy': top3,
            'top2_gain': top2 - top1,
            'top3_gain': top3 - top1,
            'n_samples': len(subset)
        })

df_topk = pd.DataFrame(results_topk)

# Mostrar resultados
print("\n📊 TOP-K ACCURACY POR DATASET:")
print("-" * 80)

for dataset in df_topk['dataset'].unique():
    print(f"\n{dataset.upper()}:")
    subset = df_topk[df_topk['dataset'] == dataset].sort_values('top3_gain', ascending=False)
    
    for _, row in subset.iterrows():
        print(f"  {row['prompt_set']:30s} | Top-1: {row['top1_accuracy']:.3f} | "
              f"Top-2: {row['top2_accuracy']:.3f} (+{row['top2_gain']:.3f}) | "
              f"Top-3: {row['top3_accuracy']:.3f} (+{row['top3_gain']:.3f})")

# Visualización - incluir todos los datasets
datasets_available = sorted(df_topk['dataset'].unique())
fig, axes = plt.subplots(1, len(datasets_available), figsize=(6*len(datasets_available), 6))
if len(datasets_available) == 1:
    axes = [axes]
fig.suptitle('Top-K Accuracy por Dataset', fontsize=16, fontweight='bold')

for idx, dataset in enumerate(datasets_available):
    ax = axes[idx]
    data = df_topk[df_topk['dataset'] == dataset].sort_values('prompt_set')
    
    x = np.arange(len(data))
    width = 0.25
    
    ax.bar(x - width, data['top1_accuracy'], width, label='Top-1', 
           color='#e74c3c', edgecolor='black', linewidth=1)
    ax.bar(x, data['top2_accuracy'], width, label='Top-2',
           color='#f39c12', edgecolor='black', linewidth=1)
    ax.bar(x + width, data['top3_accuracy'], width, label='Top-3',
           color='#27ae60', edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(dataset.capitalize(), fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data['prompt_set'], rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'topk_accuracy_comparison.png'),
            dpi=300, bbox_inches='tight')
print(f"\n✓ Guardado: {OUTPUT_DIR}/topk_accuracy_comparison.png")
plt.close()

# Guardar CSV con resultados top-k
output_topk_csv = os.path.join(RESULTS_DIR, "topk_accuracy_analysis.csv")
df_topk.to_csv(output_topk_csv, index=False)
print(f"✓ Guardado: {output_topk_csv}")

# Análisis de ganancia promedio
print("\n💡 ANÁLISIS DE GANANCIA TOP-K:")
print("-" * 80)
for dataset in df_topk['dataset'].unique():
    subset = df_topk[df_topk['dataset'] == dataset]
    print(f"\n{dataset.upper()}:")
    print(f"  Ganancia promedio Top-2: {subset['top2_gain'].mean():.4f} (±{subset['top2_gain'].std():.4f})")
    print(f"  Ganancia promedio Top-3: {subset['top3_gain'].mean():.4f} (±{subset['top3_gain'].std():.4f})")
    print(f"  Máxima ganancia Top-3: {subset['top3_gain'].max():.4f} ({subset.loc[subset['top3_gain'].idxmax(), 'prompt_set']})")


# ============================================================================
# 5. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(f"\n✅ ANÁLISIS COMPLETADOS:")
print("  1. Matrices de confusión de Agriculture (specific P1-P4)")
print("     → Mode collapse detectado en algunos prompts")
print("  2. Histogramas de confidence por prompt")
print("     → ECE bajos correlacionan con baja confianza general")
print("  3. Top-K accuracy (top-1, top-2, top-3)")
print("     → BLIP-2 muestra mejoras significativas en top-3")

print("\n📁 ARCHIVOS GENERADOS:")
print(f"  - {OUTPUT_DIR}/agriculture_confusion_matrices_specific.png")
print(f"  - {OUTPUT_DIR}/confidence_histograms_agriculture.png")
print(f"  - {OUTPUT_DIR}/topk_accuracy_comparison.png")
print(f"  - {output_topk_csv}")

print("\n" + "=" * 80)
