"""
Script unificado para experimentos CLIP y BLIP2.

Incluye:
- Configuración de prompts (4 genéricos + 4 específicos por dataset)
- Carga de 5 datasets: beans, food101, food11, agriculture, fruitveg
- Evaluación con CLIP y BLIP2
- 2 tipos de ensemble: generic (4 prompts genéricos) y specific (4 prompts específicos)
- Guardado de resultados en JSONL para calibration_analysis
"""

import os
import json
import random
import torch
import numpy as np
import kagglehub
import pandas as pd
from PIL import Image
from collections import Counter
from datasets import load_dataset, concatenate_datasets
from transformers import CLIPProcessor, CLIPModel, Blip2Processor, Blip2ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import re
from difflib import SequenceMatcher
import argparse

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_CLIP_JSONL = os.path.join(RESULTS_DIR, "h1_clip_unified_all_samples.jsonl")
OUTPUT_BLIP2_JSONL = os.path.join(RESULTS_DIR, "h1_blip2_unified_all_samples.jsonl")
OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "h1_unified_summary.csv")

# =========================
# PROMPTS GENÉRICOS
# =========================
GENERIC_PROMPTS = [
    "a photo of a {label}",
    "this is a {label}",
    "an image showing {label}",
    "a picture of {label}"
]

# =========================
# PROMPTS ESPECÍFICOS POR DATASET
# =========================
SPECIFIC_PROMPTS = {
    "beans": [
        "a close-up image of a bean leaf showing {label}",
        "a close-up image of a bean leaf with {label}",
        "a bean leaf with visible symptoms of {label} beans",
        "an agricultural image used to identify {label} on a bean leaf"
    ],
    "food101": [
        "a close-up photo of a prepared {label}",
        "an image of a prepared {label} dish",
        "a serving of {label}",
        "a photo of a plate with {label} food"
    ],
    "food11": [
        "a photo of a dish of {label}",
        "a close-up photo of {label} food",
        "a dish that belongs to the category {label}",
        "an image showing {label} as a type of food"
    ],
    "agriculture": [
        "agricultural crop of {label}",
        "an agricultural image of {label} crops",
        "farm produce showing {label}",
        "a photo of {label} crops in a field"
    ],
    "fruitveg": [
        "fresh {label}",
        "produce photo of {label}"
    ],
    "plant_seedlings": [
        "a photo of a {label} seedling",
        "an image of {label} plant seedling",
        "agricultural image showing {label} seedling",
        "top view of a {label} plant seedling"
    ],
    "crops": [
        "a photo of {label} crop",
        "an image of {label} plant",
        "agricultural crop showing {label}",
        "a field of {label}"
    ]
}

# =========================
# CONFIGURACIÓN DE DATASETS
# =========================
DATASETS_CONFIG = {
    "beans": {
        "name": "Beans",
        "source": "huggingface",
        "hf_name": "beans",
        "split": "test",
        "label_mapping": {
            0: "healthy",
            1: "angular_leaf_spot",
            2: "bean_rust"
        },
        "label_key": "labels",
        "stratify": True,
        "samples_per_class": None
    },
    "food101": {
        "name": "Food101",
        "source": "huggingface",
        "hf_name": "food101",
        "split": "validation",
        "top_k": 5,
        "stratify": True,
        "samples_per_class": 100,
        "others_category": True,
        "label_key": "label"
    },
    "food11": {
        "name": "Food11",
        "source": "kaggle",
        "kaggle_id": "trolukovich/food11-image-dataset",
        "stratify": True,
        "samples_per_class": None,
        "label_key": "label",
        "label_mapping": None
    },
    "agriculture": {
        "name": "Agriculture",
        "source": "kaggle",
        "kaggle_id": "mdwaquarazam/agricultural-crops-image-classification",
        "stratify": True,
        "samples_per_class": None,
        "label_mapping": None
    },
    "fruitveg": {
        "name": "Fruit and Vegetables",
        "source": "huggingface",
        "hf_name": "lansinuote/gen.image.class.dataset.fruits_and_vegetables",
        "split": "train",
        "stratify": True,
        "samples_per_class": None,
        "label_key": "label",
        "label_mapping": None
    },
    "plant_seedlings": {
        "name": "Plant Seedlings",
        "source": "kaggle",
        "kaggle_id": "c/plant-seedlings-classification",
        "stratify": True,
        "samples_per_class": None,
        "label_mapping": None
    }
}

# =========================
# FUNCIONES AUXILIARES
# =========================

def resize_image_if_needed(img, max_size=512):
    """
    Redimensiona la imagen si es muy grande manteniendo aspect ratio.
    Esto acelera significativamente el procesamiento en BLIP2.
    """
    width, height = img.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        return img.resize((new_width, new_height), Image.LANCZOS)
    return img


def get_prompts_for_dataset(dataset_name, ensemble_type="generic"):
    """Obtiene los prompts para un dataset y tipo de ensemble."""
    if ensemble_type == "generic":
        return GENERIC_PROMPTS.copy()
    elif ensemble_type == "specific":
        if dataset_name in SPECIFIC_PROMPTS:
            return SPECIFIC_PROMPTS[dataset_name].copy()
        else:
            # Si no hay prompts específicos, usar genéricos
            return GENERIC_PROMPTS.copy()
    else:
        raise ValueError(f"Tipo de ensemble desconocido: {ensemble_type}")


def append_samples_to_jsonl(sample_rows, output_jsonl):
    """Guarda samples en formato JSONL."""
    with open(output_jsonl, "a", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# CARGA DE DATASETS
# =========================

def load_beans():
    """Carga dataset Beans estratificado."""
    print("\n→ Cargando Beans...")
    beans = load_dataset("beans", split="test")
    label_mapping = {0: "healthy", 1: "angular_leaf_spot", 2: "bean_rust"}
    
    counts = Counter(ex["labels"] for ex in beans)
    k = min(counts.values())
    
    samples = []
    for cls, name in label_mapping.items():
        subset = beans.filter(lambda x, c=cls: x["labels"] == c).shuffle(seed=SEED).select(range(k))
        for ex in subset:
            samples.append({
                "image": ex["image"].convert("RGB"),
                "label": cls,
                "label_text": name
            })
    
    label_order = [label_mapping[i] for i in sorted(label_mapping.keys())]
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases")
    return samples, label_mapping, label_order


def load_food101():
    """Carga dataset Food101 top-5 + others estratificado."""
    print("\n→ Cargando Food101 (top-5 + others)...")
    food_raw = load_dataset("food101", split="validation")
    orig_names = food_raw.features["label"].names
    all_labels = [orig_names[l] for l in food_raw["label"]]
    freqs = Counter(all_labels)
    top5 = [lab for lab, _ in freqs.most_common(5)]
    
    def map_to_top5_others(example):
        name = orig_names[example["label"]]
        example["new_label"] = name if name in top5 else "others"
        return example
    
    food_mapped = food_raw.map(map_to_top5_others)
    food_mapped = food_mapped.class_encode_column("new_label")
    food_mapped = food_mapped.remove_columns("label")
    food_mapped = food_mapped.rename_column("new_label", "label")
    new_labels = food_mapped.features["label"].names
    
    parts = []
    for cid, name in enumerate(new_labels):
        subset = food_mapped.filter(lambda x, c=cid: x["label"] == c).shuffle(seed=SEED)
        parts.append(subset.select(range(min(100, len(subset)))))
    food_bal = concatenate_datasets(parts)
    
    label_mapping = {i: name for i, name in enumerate(new_labels)}
    samples = []
    for ex in food_bal:
        samples.append({
            "image": ex["image"].convert("RGB"),
            "label": ex["label"],
            "label_text": label_mapping[ex["label"]]
        })
    
    label_order = [label_mapping[i] for i in sorted(label_mapping.keys())]
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases")
    return samples, label_mapping, label_order


def load_kaggle_folder_balanced(kaggle_id, dataset_tag):
    """Carga dataset de Kaggle desde estructura de carpetas."""
    print(f"\n→ Cargando {dataset_tag} desde Kaggle...")
    path = kagglehub.dataset_download(kaggle_id)
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    data_root = os.path.join(path, subdirs[0]) if subdirs else path
    
    # Buscar carpetas de categorías
    cats = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    
    samples_by_cat = {cat: [] for cat in cats}
    for cat in cats:
        cat_path = os.path.join(data_root, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fn in files:
            fp = os.path.join(cat_path, fn)
            try:
                img = Image.open(fp).convert("RGB")
                samples_by_cat[cat].append(img)
            except:
                pass
    
    # Estratificar
    k = min(len(lst) for lst in samples_by_cat.values())
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
    
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases")
    return samples, label_mapping, label_order


def load_fruitveg():
    """Carga dataset Fruits and Vegetables estratificado."""
    print("\n→ Cargando Fruits and Vegetables...")
    ds = load_dataset("lansinuote/gen.image.class.dataset.fruits_and_vegetables", split="train")
    
    # Obtener las clases
    if hasattr(ds.features["label"], "names"):
        label_names = ds.features["label"].names
    else:
        label_names = sorted(list(set(ds["label"])))
    
    # Estratificar
    counts = Counter(ds["label"])
    k = min(counts.values())
    
    samples = []
    for cls_idx in range(len(label_names)):
        subset = ds.filter(lambda x, c=cls_idx: x["label"] == c).shuffle(seed=SEED).select(range(k))
        for ex in subset:
            samples.append({
                "image": ex["image"].convert("RGB"),
                "label": cls_idx,
                "label_text": label_names[cls_idx] if isinstance(label_names[cls_idx], str) else str(label_names[cls_idx])
            })
    
    label_mapping = {i: label_names[i] for i in range(len(label_names))}
    label_order = [label_mapping[i] for i in sorted(label_mapping.keys())]
    
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases")
    return samples, label_mapping, label_order


def load_plant_seedlings():
    """Carga dataset Plant Seedlings Classification estratificado."""
    print("\n→ Cargando Plant Seedlings...")
    path = kagglehub.dataset_download("c/plant-seedlings-classification")
    
    # Buscar carpeta train
    train_path = os.path.join(path, "train")
    if not os.path.exists(train_path):
        # Buscar en subdirectorios
        subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        for subdir in subdirs:
            potential_train = os.path.join(path, subdir, "train")
            if os.path.exists(potential_train):
                train_path = potential_train
                break
    
    # Buscar carpetas de categorías
    cats = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    samples_by_cat = {cat: [] for cat in cats}
    for cat in cats:
        cat_path = os.path.join(train_path, cat)
        files = [f for f in os.listdir(cat_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fn in files:
            fp = os.path.join(cat_path, fn)
            try:
                img = Image.open(fp).convert("RGB")
                samples_by_cat[cat].append(img)
            except:
                pass
    
    # Estratificar
    k = min(len(lst) for lst in samples_by_cat.values())
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
    
    print(f"  Cargadas {len(samples)} muestras, {len(label_order)} clases")
    return samples, label_mapping, label_order


def load_dataset_by_name(dataset_name):
    """Carga un dataset por nombre."""
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
    elif dataset_name == "plant_seedlings":
        return load_plant_seedlings()
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")


# =========================
# EVALUACIÓN CON CLIP
# =========================

def eval_clip_single_prompt(samples, label_order, prompt_template, processor, model, device):
    """Evalúa CLIP con un solo prompt y retorna matriz de probabilidades."""
    n = len(samples)
    num_labels = len(label_order)
    probs_mat = np.zeros((n, num_labels), dtype=np.float32)
    
    # Formatear prompts con labels
    text_prompts = [prompt_template.format(label=lbl) for lbl in label_order]
    
    batch_size = 32
    for i in range(0, n, batch_size):
        batch = samples[i:i+batch_size]
        images = [s["image"] for s in batch]
        
        inputs = processor(text=text_prompts, images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # [batch_size, num_labels]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        
        for j in range(len(images)):
            if i + j < n:
                probs_mat[i + j, :] = probs[j].astype(np.float32)
    
    return probs_mat


def eval_clip_dataset(dataset_name, samples, label_mapping, label_order, ensemble_type, processor, model, device):
    """Evalúa CLIP en un dataset con un tipo de ensemble."""
    print(f"\n{'='*60}")
    print(f"CLIP - {dataset_name.upper()} - Ensemble: {ensemble_type}")
    print(f"{'='*60}")
    
    prompts = get_prompts_for_dataset(dataset_name, ensemble_type)
    n = len(samples)
    num_labels = len(label_order)
    true_labels = [s["label_text"] for s in samples]
    
    all_results = []
    
    # Evaluar cada prompt individual
    prompt_probs = []
    for prompt_idx, prompt_template in enumerate(prompts):
        print(f"\n  Prompt {prompt_idx+1}/{len(prompts)}: '{prompt_template}'")
        probs_mat = eval_clip_single_prompt(samples, label_order, prompt_template, processor, model, device)
        prompt_probs.append(probs_mat)
        
        # Evaluar este prompt individual
        pred_idx = probs_mat.argmax(axis=1)
        pred_labels = [label_order[k] for k in pred_idx]
        conf = probs_mat.max(axis=1).astype(float)
        
        acc = accuracy_score(true_labels, pred_labels)
        f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
        f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
        
        print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
        
        # Guardar en JSONL
        rows = []
        for i in range(n):
            rows.append({
                "dataset": dataset_name,
                "mode": "single_clip",
                "prompt_set": f"{ensemble_type}_p{prompt_idx+1}",
                "sample_id": i,
                "true_label": true_labels[i],
                "pred_label": pred_labels[i],
                "pred_index": int(pred_idx[i]),
                "confidence": float(conf[i]),
                "probs": [float(x) for x in probs_mat[i].tolist()],
                "generated_text": "",
                "label_order": label_order,
            })
        append_samples_to_jsonl(rows, OUTPUT_CLIP_JSONL)
        
        all_results.append({
            "dataset": dataset_name,
            "model": "clip",
            "ensemble_type": ensemble_type,
            "prompt_set": f"{ensemble_type}_p{prompt_idx+1}",
            "mode": "single",
            "n_samples": n,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m
        })
    
    # Evaluar ensemble MEAN (promedio de probabilidades)
    print(f"\n  Ensemble MEAN (promedio de {len(prompts)} prompts)")
    ensemble_mean_probs = np.mean(prompt_probs, axis=0)
    pred_idx = ensemble_mean_probs.argmax(axis=1)
    pred_labels = [label_order[k] for k in pred_idx]
    conf = ensemble_mean_probs.max(axis=1).astype(float)
    
    acc = accuracy_score(true_labels, pred_labels)
    f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
    f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
    
    print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
    
    # Guardar en JSONL
    rows = []
    for i in range(n):
        rows.append({
            "dataset": dataset_name,
            "mode": "ensemble_clip",
            "prompt_set": f"{ensemble_type}_ensemble_mean",
            "sample_id": i,
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "pred_index": int(pred_idx[i]),
            "confidence": float(conf[i]),
            "probs": [float(x) for x in ensemble_mean_probs[i].tolist()],
            "generated_text": "",
            "label_order": label_order,
        })
    append_samples_to_jsonl(rows, OUTPUT_CLIP_JSONL)
    
    all_results.append({
        "dataset": dataset_name,
        "model": "clip",
        "ensemble_type": ensemble_type,
        "prompt_set": f"{ensemble_type}_ensemble_mean",
        "mode": "ensemble_mean",
        "n_samples": n,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m
    })
    
    # Evaluar ensemble MAX (máximo de probabilidades)
    print(f"\n  Ensemble MAX (máximo de {len(prompts)} prompts)")
    ensemble_max_probs = np.max(prompt_probs, axis=0)
    pred_idx = ensemble_max_probs.argmax(axis=1)
    pred_labels = [label_order[k] for k in pred_idx]
    conf = ensemble_max_probs.max(axis=1).astype(float)
    
    acc = accuracy_score(true_labels, pred_labels)
    f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
    f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
    
    print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
    
    # Guardar en JSONL
    rows = []
    for i in range(n):
        rows.append({
            "dataset": dataset_name,
            "mode": "ensemble_clip",
            "prompt_set": f"{ensemble_type}_ensemble_max",
            "sample_id": i,
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "pred_index": int(pred_idx[i]),
            "confidence": float(conf[i]),
            "probs": [float(x) for x in ensemble_max_probs[i].tolist()],
            "generated_text": "",
            "label_order": label_order,
        })
    append_samples_to_jsonl(rows, OUTPUT_CLIP_JSONL)
    
    all_results.append({
        "dataset": dataset_name,
        "model": "clip",
        "ensemble_type": ensemble_type,
        "prompt_set": f"{ensemble_type}_ensemble_max",
        "mode": "ensemble_max",
        "n_samples": n,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m
    })
    
    return all_results


# =========================
# EVALUACIÓN CON BLIP2
# =========================

def clean_text(s: str) -> str:
    """Limpia texto generado por BLIP2."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the|this|is|looks like)\b", "", s)
    return s.strip()


def best_match(text: str, classes: list) -> str:
    """Encuentra la mejor coincidencia entre texto generado y clases."""
    txt = clean_text(text)
    best, best_score = classes[0], 0.0
    for c in classes:
        score = SequenceMatcher(None, txt, c.lower()).ratio()
        if score > best_score:
            best_score, best = score, c
    return best


def eval_blip2_single_prompt(samples, label_order, prompt_template, processor, model, device):
    """
    Evalúa BLIP2 con un solo prompt usando método yes/no.
    Convierte cada template de CLIP a formato de pregunta yes/no preservando la estructura.
    
    Args:
        prompt_template: Template de CLIP con {label}, ej: "a photo of a {label}"
    """
    n = len(samples)
    num_labels = len(label_order)
    probs_mat = np.zeros((n, num_labels), dtype=np.float32)
    
    # Tokenizar yes/no una sola vez
    yes_ids = processor.tokenizer("yes", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    no_ids = processor.tokenizer("no", return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    for i in tqdm(range(n), desc=f"BLIP2"):
        img = samples[i]["image"]
        # Redimensionar si la imagen es muy grande para acelerar procesamiento
        img = resize_image_if_needed(img, max_size=512)
        scores = []
        
        with torch.no_grad():
            for label in label_order:
                # Convertir el template de CLIP a pregunta yes/no incluyendo el label
                # "a photo of a {label}" -> "Question: Is this a photo of a {label}? Answer:"
                # Esto preserva la estructura del prompt original
                question = prompt_template.replace("{label}", label)
                prompt = f"Question: Is this {question}? Answer:"
                
                inputs = processor(
                    images=img, 
                    text=prompt,
                    return_tensors="pt",
                    padding=True
                ).to(device)
                
                # Calcular losses para yes y no
                loss_yes = model(**inputs, labels=yes_ids).loss.item()
                loss_no = model(**inputs, labels=no_ids).loss.item()
                
                # Score: menor loss_yes = más probable que sea esta clase
                # Equivalente a: loss_no - loss_yes (cuanto más prefiere "yes" sobre "no")
                score = loss_no - loss_yes
                scores.append(score)
        
        # Convertir scores a probabilidades con softmax
        scores_np = np.array(scores, dtype=np.float64)
        scores_np -= scores_np.max()  # Estabilidad numérica
        probs = np.exp(scores_np)
        probs /= probs.sum()
        
        probs_mat[i, :] = probs.astype(np.float32)
    
    return probs_mat


def eval_blip2_dataset(dataset_name, samples, label_mapping, label_order, ensemble_type, processor, model, device):
    """Evalúa BLIP2 en un dataset con un tipo de ensemble usando método likelihood."""
    print(f"\n{'='*60}")
    print(f"BLIP2 - {dataset_name.upper()} - Ensemble: {ensemble_type}")
    print(f"{'='*60}")
    
    prompts = get_prompts_for_dataset(dataset_name, ensemble_type)
    n = len(samples)
    num_labels = len(label_order)
    true_labels = [s["label_text"] for s in samples]
    
    all_results = []
    prompt_probs = []  # Para ensemble por promedio de probabilidades
    
    # Mantener los prompts originales para BLIP2
    # Los usaremos con {label} igual que CLIP, pero en el cálculo de likelihood
    blip2_prompts = prompts.copy()
    
    # Evaluar cada prompt individual
    for prompt_idx, prompt_text in enumerate(blip2_prompts):
        print(f"\n  Prompt {prompt_idx+1}/{len(blip2_prompts)}: '{prompt_text}'")
        
        # Obtener probabilidades con método likelihood
        probs_mat = eval_blip2_single_prompt(
            samples, label_order, prompt_text, processor, model, device
        )
        prompt_probs.append(probs_mat)
        
        # Predicciones
        pred_idx = probs_mat.argmax(axis=1)
        pred_labels = [label_order[k] for k in pred_idx]
        conf = probs_mat.max(axis=1).astype(float)
        
        acc = accuracy_score(true_labels, pred_labels)
        f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
        f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
        
        print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
        
        # Guardar en JSONL con probabilidades reales
        rows = []
        for i in range(n):
            rows.append({
                "dataset": dataset_name,
                "mode": "single_blip2",
                "prompt_set": f"{ensemble_type}_p{prompt_idx+1}",
                "sample_id": i,
                "true_label": true_labels[i],
                "pred_label": pred_labels[i],
                "pred_index": int(pred_idx[i]),
                "confidence": float(conf[i]),
                "probs": [float(x) for x in probs_mat[i].tolist()],
                "generated_text": "",  # No generamos texto en modo likelihood
                "label_order": label_order,
            })
        append_samples_to_jsonl(rows, OUTPUT_BLIP2_JSONL)
        
        all_results.append({
            "dataset": dataset_name,
            "model": "blip2",
            "ensemble_type": ensemble_type,
            "prompt_set": f"{ensemble_type}_p{prompt_idx+1}",
            "mode": "single",
            "n_samples": n,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_m
        })
    
    # Ensemble MEAN (promedio de probabilidades)
    print(f"\n  Ensemble MEAN (promedio de {len(blip2_prompts)} prompts)")
    ensemble_mean_probs = np.mean(prompt_probs, axis=0)
    pred_idx = ensemble_mean_probs.argmax(axis=1)
    pred_labels = [label_order[k] for k in pred_idx]
    conf = ensemble_mean_probs.max(axis=1).astype(float)
    
    acc = accuracy_score(true_labels, pred_labels)
    f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
    f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
    
    print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
    
    # Guardar en JSONL
    rows = []
    for i in range(n):
        rows.append({
            "dataset": dataset_name,
            "mode": "ensemble_blip2",
            "prompt_set": f"{ensemble_type}_ensemble_mean",
            "sample_id": i,
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "pred_index": int(pred_idx[i]),
            "confidence": float(conf[i]),
            "probs": [float(x) for x in ensemble_mean_probs[i].tolist()],
            "generated_text": "",
            "label_order": label_order,
        })
    append_samples_to_jsonl(rows, OUTPUT_BLIP2_JSONL)
    
    all_results.append({
        "dataset": dataset_name,
        "model": "blip2",
        "ensemble_type": ensemble_type,
        "prompt_set": f"{ensemble_type}_ensemble_mean",
        "mode": "ensemble_mean",
        "n_samples": n,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m
    })
    
    # Ensemble MAX (máximo de probabilidades)
    print(f"\n  Ensemble MAX (máximo de {len(blip2_prompts)} prompts)")
    ensemble_max_probs = np.max(prompt_probs, axis=0)
    pred_idx = ensemble_max_probs.argmax(axis=1)
    pred_labels = [label_order[k] for k in pred_idx]
    conf = ensemble_max_probs.max(axis=1).astype(float)
    
    acc = accuracy_score(true_labels, pred_labels)
    f1_w = precision_recall_fscore_support(true_labels, pred_labels, average="weighted", zero_division=0)[2]
    f1_m = precision_recall_fscore_support(true_labels, pred_labels, average="macro", zero_division=0)[2]
    
    print(f"    Acc: {acc:.4f} | F1-W: {f1_w:.4f} | F1-M: {f1_m:.4f}")
    
    # Guardar en JSONL
    rows = []
    for i in range(n):
        rows.append({
            "dataset": dataset_name,
            "mode": "ensemble_blip2",
            "prompt_set": f"{ensemble_type}_ensemble_max",
            "sample_id": i,
            "true_label": true_labels[i],
            "pred_label": pred_labels[i],
            "pred_index": int(pred_idx[i]),
            "confidence": float(conf[i]),
            "probs": [float(x) for x in ensemble_max_probs[i].tolist()],
            "generated_text": "",
            "label_order": label_order,
        })
    append_samples_to_jsonl(rows, OUTPUT_BLIP2_JSONL)
    
    all_results.append({
        "dataset": dataset_name,
        "model": "blip2",
        "ensemble_type": ensemble_type,
        "prompt_set": f"{ensemble_type}_ensemble_max",
        "mode": "ensemble_max",
        "n_samples": n,
        "accuracy": acc,
        "f1_weighted": f1_w,
        "f1_macro": f1_m
    })
    
    return all_results


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Experimentos CLIP y BLIP2")
    parser.add_argument("--datasets", nargs="+", default=None,
                       help="Datasets a procesar (ej: --datasets plant_seedlings). Por defecto: todos")
    args = parser.parse_args()
    
    print("="*80)
    print("EXPERIMENTOS UNIFICADOS CLIP Y BLIP2")
    print("="*80)
    
    # Determinar datasets a procesar
    if args.datasets:
        datasets_to_process = args.datasets
        print(f"\n→ Procesando: {', '.join(datasets_to_process)}")
        # Validar
        for ds in datasets_to_process:
            if ds not in DATASETS_CONFIG:
                print(f"⚠️  Dataset '{ds}' no encontrado")
                print(f"   Disponibles: {', '.join(DATASETS_CONFIG.keys())}")
                return
    else:
        datasets_to_process = list(DATASETS_CONFIG.keys())
        print("\n→ Procesando todos los datasets")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDispositivo: {device}")
    
    # Solo borrar archivos si procesamos TODOS los datasets
    if len(datasets_to_process) == len(DATASETS_CONFIG):
        print("\n→ Borrando archivos anteriores (procesando todos)")
        for f in [OUTPUT_CLIP_JSONL, OUTPUT_BLIP2_JSONL]:
            if os.path.exists(f):
                os.remove(f)
    else:
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
    
    # Iterar sobre datasets seleccionados
    for dataset_name in datasets_to_process:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")
        
        # Cargar dataset
        samples, label_mapping, label_order = load_dataset_by_name(dataset_name)
        
        # Iterar sobre tipos de ensemble
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
    
    # Guardar resumen en CSV
    print(f"\n{'='*80}")
    print("GUARDANDO RESUMEN...")
    print(f"{'='*80}")
    df_new = pd.DataFrame(all_summary)
    
    # Si procesamos solo algunos datasets, combinar con existentes
    if os.path.exists(OUTPUT_SUMMARY_CSV) and len(datasets_to_process) < len(DATASETS_CONFIG):
        df_existing = pd.read_csv(OUTPUT_SUMMARY_CSV)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Eliminar duplicados (mismo dataset, model, ensemble_type, prompt_set)
        df_combined = df_combined.drop_duplicates(
            subset=['dataset', 'model', 'ensemble_type', 'prompt_set'],
            keep='last'
        )
        df_combined.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"\n→ Resumen actualizado (combinado con existente): {OUTPUT_SUMMARY_CSV}")
    else:
        df_new.to_csv(OUTPUT_SUMMARY_CSV, index=False)
        print(f"\n→ Resumen guardado en: {OUTPUT_SUMMARY_CSV}")
    print(f"→ CLIP samples guardados en: {OUTPUT_CLIP_JSONL}")
    print(f"→ BLIP2 samples guardados en: {OUTPUT_BLIP2_JSONL}")
    print("\n✓ EXPERIMENTOS COMPLETADOS")


if __name__ == "__main__":
    main()
