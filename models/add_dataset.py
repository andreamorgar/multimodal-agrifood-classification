"""
Script para añadir nuevos datasets sin re-ejecutar los existentes.

Reutiliza las funciones de unified_experiments.py y hace APPEND a los archivos.
Útil para añadir plant_seedlings u otros datasets sin borrar resultados previos.

Uso:
    python add_dataset.py --datasets plant_seedlings
    python add_dataset.py --datasets plant_seedlings fruitveg
"""

import os
import sys
import json
import random
import torch
import numpy as np
import kagglehub
import pandas as pd
import argparse
from PIL import Image
from collections import Counter
from datasets import load_dataset

# Importar funciones del script principal
from unified_experiments import (
    SEED, RESULTS_DIR,
    OUTPUT_CLIP_JSONL, OUTPUT_BLIP2_JSONL, OUTPUT_SUMMARY_CSV,
    DATASETS_CONFIG, SPECIFIC_PROMPTS,
    load_beans, load_food101, load_kaggle_folder_balanced, load_fruitveg,
    eval_clip_dataset, eval_blip2_dataset
)

from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration

# =========================
# DATASET PLANT SEEDLINGS
# =========================

def load_crops():
    """Carga dataset 140 Most Popular Crops estratificado - SUBCONJUNTO MÍNIMO."""
    print("\n→ Cargando Crops Dataset (subconjunto reducido)...")
    path = kagglehub.dataset_download("omrathod2003/140-most-popular-crops-image-dataset")
    
    # Este dataset tiene estructura: Raw/Raw/<categorías>
    data_root = os.path.join(path, "Raw", "Raw")
    
    # Buscar carpetas de categorías
    cats = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('.')])
    
    samples_by_cat = {cat: [] for cat in cats}
    for cat in cats:
        cat_path = os.path.join(data_root, cat)
        # Las imágenes están en un subdirectorio 'raw' dentro de cada categoría
        raw_subdir = os.path.join(cat_path, "raw")
        if os.path.exists(raw_subdir):
            cat_path = raw_subdir
        
        files = [f for f in os.listdir(cat_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fn in files[:30]:  # REDUCCIÓN: tomar hasta 30 imágenes por clase
            fp = os.path.join(cat_path, fn)
            try:
                img = Image.open(fp).convert("RGB")
                samples_by_cat[cat].append(img)
            except:
                pass
    
    # Estratificar - tomar hasta 30 muestras por clase
    k = min(min(len(lst) for lst in samples_by_cat.values()), 30)
    samples = []
    random.seed(SEED)
    for idx, cat in enumerate(cats):
        imgs = random.sample(samples_by_cat[cat], k)
        for img in imgs:
            samples.append({
                "image": img,
                "label": idx,
                "label_text": cat
            })
    
    label_mapping = {i: cat for i, cat in enumerate(cats)}
    label_order = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases [REDUCIDO]")
    return samples, label_mapping, label_order


# Añadir configuración de crops
DATASETS_CONFIG_EXTENDED = DATASETS_CONFIG.copy()
DATASETS_CONFIG_EXTENDED["crops"] = {
    "name": "140 Popular Crops",
    "source": "kaggle",
    "kaggle_id": "omrathod2003/140-most-popular-crops-image-dataset",
    "stratify": True,
    "samples_per_class": None,
    "label_mapping": None
}

# Añadir prompts específicos
SPECIFIC_PROMPTS_EXTENDED = SPECIFIC_PROMPTS.copy()
SPECIFIC_PROMPTS_EXTENDED["crops"] = [
    "a photo of {label} crop",
    "an image of {label} plant",
    "agricultural crop showing {label}",
    "a field of {label}"
]


def load_dataset_by_name(dataset_name):
    """Carga un dataset por nombre (incluyendo crops)."""
    if dataset_name == "beans":
        return load_beans()
    elif dataset_name == "food101":
        return load_food101()
    elif dataset_name == "food11":
        return load_kaggle_folder_balanced("trolukovich/food11-image-dataset", "food11")
    elif dataset_name == "agriculture":
        return load_kaggle_folder_balanced("mdwaquarazam/agricultural-crops-image-classification", "agriculture")
    elif dataset_name == "fruitveg":
        return load_fruitveg()
    elif dataset_name == "crops":
        return load_crops()
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Añadir datasets sin re-ejecutar existentes")
    parser.add_argument("--datasets", nargs="+", required=True,
                       help="Datasets a añadir (ej: --datasets plant_seedlings)")
    args = parser.parse_args()
    
    print("="*80)
    print("AÑADIR DATASETS - CLIP Y BLIP2")
    print("="*80)
    
    datasets_to_process = args.datasets
    print(f"\n→ Procesando: {', '.join(datasets_to_process)}")
    
    # Validar datasets
    available = list(DATASETS_CONFIG_EXTENDED.keys())
    for ds in datasets_to_process:
        if ds not in available:
            print(f"⚠️  Dataset '{ds}' no encontrado")
            print(f"   Disponibles: {', '.join(available)}")
            return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    print("\n→ Modo APPEND: manteniendo resultados existentes")
    
    # Cargar modelos CLIP
    print("\n→ Cargando CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    
    # Cargar modelos BLIP2
    print("\n→ Cargando BLIP2...")
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", use_fast=True)
    blip2_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=torch.float16
    ).to(device)
    
    all_summary = []
    
    # Procesar cada dataset
    for dataset_name in datasets_to_process:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        # Cargar dataset
        samples, label_mapping, label_order = load_dataset_by_name(dataset_name)
        
        # Evaluar con ambos tipos de ensemble
        for ensemble_type in ["generic", "specific"]:
            # CLIP
            clip_results = eval_clip_dataset(
                dataset_name, samples, label_mapping, label_order,
                ensemble_type, clip_processor, clip_model, device
            )
            all_summary.extend(clip_results)
            
            # BLIP2
            blip2_results = eval_blip2_dataset(
                dataset_name, samples, label_mapping, label_order,
                ensemble_type, blip2_processor, blip2_model, device
            )
            all_summary.extend(blip2_results)
    
    # Combinar con resultados existentes
    print(f"\n{'='*80}")
    print("ACTUALIZANDO RESUMEN...")
    print(f"{'='*80}")
    
    df_new = pd.DataFrame(all_summary)
    
    if os.path.exists(OUTPUT_SUMMARY_CSV):
        df_existing = pd.read_csv(OUTPUT_SUMMARY_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Eliminar duplicados (mismo dataset, model, ensemble_type, prompt_set)
        df_combined = df_combined.drop_duplicates(
            subset=['dataset', 'model', 'ensemble_type', 'prompt_set'],
            keep='last'
        )
        df_combined.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"\n→ Resumen actualizado (combinado): {OUTPUT_SUMMARY_CSV}")
    else:
        df_new.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"\n→ Resumen guardado en: {OUTPUT_SUMMARY_CSV}")
    
    print(f"→ CLIP samples añadidos a: {OUTPUT_CLIP_JSONL}")
    print(f"→ BLIP2 samples añadidos a: {OUTPUT_BLIP2_JSONL}")
    print("\n✓ DATASETS AÑADIDOS CORRECTAMENTE")


if __name__ == "__main__":
    main()
