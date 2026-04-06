import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = "../results"
INPUT_JSONL = os.path.join(RESULTS_DIR, "h1_clip_unified_all_samples.jsonl")  # <-- archivo con muestras individuales
PLOTS_DIR = "../results/img"
ECE_BINS = 15

GROUP_COLS = ["dataset", "mode", "prompt_set"]  # ajusta si tu CSV usa otros nombres


# -----------------------------
# Curva de calibración
# -----------------------------
def reliability_curve(confidences, correct, n_bins=15):
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_conf, bin_acc, bin_count = [], [], []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        if b == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        nk = int(mask.sum())
        if nk == 0:
            continue

        bin_conf.append(float(confidences[mask].mean()))
        bin_acc.append(float(correct[mask].mean()))
        bin_count.append(nk)

    return np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def plot_reliability_diagram(confidences, correct, n_bins=15, title="", save_path=None):
    x, y, counts = reliability_curve(confidences, correct, n_bins=n_bins)

    # Configurar estilo profesional
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Diagonal perfecta (calibración ideal)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect calibration', alpha=0.7)
    
    # Curva real con mejor estilo
    ax.plot(x, y, marker='o', markersize=12, linewidth=3.5, 
            color='#2E86AB', markerfacecolor='#A23B72', 
            markeredgewidth=3, markeredgecolor='white',
            label='Model calibration')
    
    # Área entre curva y diagonal
    ax.fill_between(x, y, x, alpha=0.2, color='#2E86AB')

    # Configuración de ejes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=20, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=16, pad=20)
    
    # Aumentar tamaño de los números en los ejes
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Grid más sutil
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Anotar nº muestras por bin con mejor formato
    for xi, yi, ci in zip(x, y, counts):
        ax.annotate(f'n={ci}', (xi, yi), 
                   textcoords="offset points", 
                   xytext=(0, 12), 
                   fontsize=13,
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', 
                           facecolor='white', 
                           edgecolor='gray',
                           alpha=0.8))
    
    # Leyenda
    ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
    
    # Ajustar layout
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# -----------------------------
# Main
# -----------------------------
# Cargar JSONL
records = []
with open(INPUT_JSONL, "r") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# 1) Crear "correct" si no existe
if "correct" not in df.columns:
    if "pred_label" not in df.columns or "true_label" not in df.columns:
        raise ValueError("Tu JSONL no tiene 'correct' ni ('pred_label' y 'true_label').")
    df["correct"] = (df["pred_label"].astype(str) == df["true_label"].astype(str)).astype(int)

# 2) Asegurar "confidence"
# Si tu CSV ya tiene confidence, genial. Si no, necesitaríamos probs para calcularla.
if "confidence" not in df.columns:
    raise ValueError("Tu CSV no tiene columna 'confidence'. Si tienes 'probs', dime el formato y la calculamos.")

df["confidence"] = df["confidence"].astype(float)

# 3) Plots por grupo
missing = [c for c in GROUP_COLS if c not in df.columns]
if missing:
    raise ValueError(f"Faltan columnas de agrupación en tu CSV: {missing}. "
                     f"Columnas disponibles: {list(df.columns)}")

for key, g in df.groupby(GROUP_COLS, sort=True):
    dataset, mode, prompt_set = key
    title = f"{dataset} | {mode} | {prompt_set} | bins={ECE_BINS} | n={len(g)}"
    fname = f"reliability__{dataset}__{mode}__{prompt_set}.png".replace("/", "_")
    save_path = os.path.join(PLOTS_DIR, fname)

    plot_reliability_diagram(
        confidences=g["confidence"].values,
        correct=g["correct"].values,
        n_bins=ECE_BINS,
        title=title,
        save_path=save_path
    )

print("✅ Reliability diagrams guardados en:", PLOTS_DIR)
