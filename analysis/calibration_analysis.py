# offline_calibration_analysis.py
# Lee results/h1_clip_all_samples.jsonl y h1_blip2_all_samples.jsonl y calcula:
# - Accuracy, F1 macro/weighted
# - Brier score (multiclase)
# - ECE (Expected Calibration Error)
# - NLL (log loss) opcional
# - Predictive entropy (incertidumbre)
# Y guarda un CSV resumen.

import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = "../results"
INPUT_JSONL_CLIP = os.path.join(RESULTS_DIR, "h1_clip_unified_all_samples.jsonl")
INPUT_JSONL_BLIP = os.path.join(RESULTS_DIR, "h1_blip2_unified_all_samples.jsonl")
# INPUT_JSONL_SIGCLIP = os.path.join(RESULTS_DIR, "h1_sigclip_all_samples.jsonl")

OUTPUT_SUMMARY_CLIP_CSV = os.path.join(RESULTS_DIR, "offline_summary_clip_brier_ece.csv")
OUTPUT_SUMMARY_BLIP_CSV = os.path.join(RESULTS_DIR, "offline_summary_blip_brier_ece.csv")
# OUTPUT_SUMMARY_SIGCLIP_CSV = os.path.join(RESULTS_DIR, "offline_summary_sigclip_brier_ece.csv")

ECE_BINS = 15  # típico: 10-20

# -----------------------------
# Utils
# -----------------------------
def safe_log(x, eps=1e-12):
    return math.log(max(x, eps))

def predictive_entropy(probs, eps=1e-12):
    # probs: array shape (C,)
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log(p)).sum())

def multiclass_brier(probs, true_index):
    # probs: array shape (C,), true_index int
    y = np.zeros_like(probs, dtype=float)
    y[true_index] = 1.0
    return float(np.mean((probs - y) ** 2))

def nll(probs, true_index, eps=1e-12):
    return float(-safe_log(float(probs[true_index]), eps=eps))

def ece_score(confidences, correct, n_bins=15):
    """
    confidences: array shape (N,) = max prob
    correct: array shape (N,) in {0,1}
    ECE = sum_k (|Bk|/N) * |acc(Bk) - conf(Bk)|
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
        # incluir hi solo en el último bin
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

def parse_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON inválido en línea {line_no}: {e}")
    return rows

def get_true_index(row):
    """
    Necesitamos mapear true_label -> índice de clase para Brier/NLL.
    Usamos label_order guardado en cada fila.
    """
    label_order = row["label_order"]
    true_label = row["true_label"]
    
    # Verificar que label_order sea una lista
    if not isinstance(label_order, list):
        return None
    
    try:
        return int(label_order.index(true_label))
    except ValueError:
        # si por algún motivo no está, devolvemos None
        return None

def process_and_save_results(input_jsonl, output_csv, model_name):
    """
    Procesa un archivo JSONL y guarda los resultados en un CSV.
    """
    if not os.path.exists(input_jsonl):
        print(f"⚠️ Archivo no encontrado, omitiendo: {input_jsonl}")
        return None
    
    print(f"\n{'='*60}")
    print(f"📂 Procesando {model_name}: {input_jsonl}")
    print(f"{'='*60}")
    
    rows = parse_jsonl(input_jsonl)
    if not rows:
        print(f"⚠️ No hay datos en {input_jsonl}")
        return None
    
    df = pd.DataFrame(rows)
    
    # Normalizar tipos
    df["confidence"] = df["confidence"].astype(float)
    
    # Convertir probs a numpy arrays
    df["probs_np"] = df["probs"].apply(lambda x: np.array(x, dtype=float))
    
    # true_index
    df["true_index"] = df.apply(get_true_index, axis=1)
    df = df.dropna(subset=["true_index"]).copy()
    df["true_index"] = df["true_index"].astype(int)
    
    # correct (0/1)
    df["correct"] = (df["pred_label"] == df["true_label"]).astype(int)
    
    # per-sample metrics
    df["entropy"] = df["probs_np"].apply(predictive_entropy)
    df["brier"] = df.apply(lambda r: multiclass_brier(r["probs_np"], r["true_index"]), axis=1)
    df["nll"] = df.apply(lambda r: nll(r["probs_np"], r["true_index"]), axis=1)
    
    # Agregado por (dataset, mode, prompt_set)
    group_cols = ["dataset", "mode", "prompt_set"]
    
    summary_rows = []
    for key, g in df.groupby(group_cols, sort=True):
        dataset, mode, prompt_set = key
        y_true = g["true_label"].tolist()
        y_pred = g["pred_label"].tolist()
    
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
        # ECE con conf=max prob
        ece = ece_score(g["confidence"].values, g["correct"].values, n_bins=ECE_BINS)
    
        summary_rows.append({
            "dataset": dataset,
            "mode": mode,
            "prompt_set": prompt_set,
            "n_samples": int(len(g)),
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "brier_mean": float(g["brier"].mean()),
            "nll_mean": float(g["nll"].mean()),
            "ece": float(ece),
            "entropy_mean": float(g["entropy"].mean()),
            "entropy_std": float(g["entropy"].std(ddof=0)) if len(g) > 1 else 0.0,
            "confidence_mean": float(g["confidence"].mean()),
            "confidence_std": float(g["confidence"].std(ddof=0)) if len(g) > 1 else 0.0,
            "accuracy_by_conf_bin_bins": ECE_BINS,
        })
    
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "mode", "prompt_set"])
    summary.to_csv(output_csv, index=False)
    
    print(f"✅ Resumen {model_name} guardado en: {output_csv}")
    print(f"\nTop filas (preview {model_name}):")
    print(summary.head(20).to_string(index=False))
    
    # Mejor por dataset
    best = summary.sort_values(["dataset", "accuracy"], ascending=[True, False]).groupby("dataset").head(1)
    best_path = output_csv.replace("_brier_ece.csv", "_best_by_dataset.csv")
    best.to_csv(best_path, index=False)
    print(f"✅ Best-by-dataset {model_name} guardado en: {best_path}")
    
    return summary

# -----------------------------
# Main
# -----------------------------
# Procesar CLIP
summary_clip = process_and_save_results(INPUT_JSONL_CLIP, OUTPUT_SUMMARY_CLIP_CSV, "CLIP")

# Procesar BLIP
summary_blip = process_and_save_results(INPUT_JSONL_BLIP, OUTPUT_SUMMARY_BLIP_CSV, "BLIP2")

# Procesar SigCLIP
# summary_sigclip = process_and_save_results(INPUT_JSONL_SIGCLIP, OUTPUT_SUMMARY_SIGCLIP_CSV, "SigCLIP")

print("\n" + "="*60)
print("✅ ANÁLISIS COMPLETADO")
print("="*60)
if summary_clip is not None:
    print(f"CLIP: {OUTPUT_SUMMARY_CLIP_CSV}")
if summary_blip is not None:
    print(f"BLIP2: {OUTPUT_SUMMARY_BLIP_CSV}")
