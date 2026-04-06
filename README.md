# Zero-shot multimodal classification for agri-food applications

Comparative study of **CLIP** (contrastive) and **BLIP-2** (generative) multimodal models applied to image classification tasks in the **agriculture and food domains**.

---

## Project description

The project conducts a **systematic comparison** between **contrastive (CLIP)** and **generative (BLIP-2)** multimodal classification approaches in the **agrifood domain**, evaluating both models in *zero-shot* scenarios across multiple datasets.


## Repository structure

```
multimodal_experiments/
├── models/                    # Main experimentation scripts
│   ├── unified_experiments.py              # Main unified experiment (CLIP + BLIP2)
│   ├── unified_experiments_safe.py         # Utility for safe parallel writing
│   ├── add_dataset.py                      # Add datasets without re-running everything
│   ├── add_dataset_single.py               # Worker for individual processing
│   ├── add_dataset_parallel.py             # Parallel executor on 2 GPUs
│   └── add_dataset_parallel_clases_reducidas.py  # Variant with reduced classes
│
├── analysis/                  # Results analysis scripts
│   ├── calibration_analysis.py             # Calibration analysis (Brier, ECE)
│   ├── advanced_analysis.py                # Advanced metrics analysis
│   ├── additional_ensembles.py             # Additional ensembles analysis
│   └── plot.py                             # Graphics generation
│
├── results/                   # Generated results
│   ├── h1_blip2_unified_all_samples.jsonl  # BLIP2 results (149 MB, 25,200 samples)
│   ├── h1_clip_unified_all_samples.jsonl   # CLIP results (152 MB, 25,200 samples)
│   ├── h1_unified_summary.csv              # Metrics summary
│   ├── offline_summary_*.csv               # Ensemble summaries
│   └── img/                                # Graphics and visualizations
│
├── datasets/                  # Dataset files (see Datasets section below)
│
├── comparison_confusions/     # Comparative images between classes
│
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

---

## Models

| Model | Type | Description | Use Case |
|--------|------|-------------|----------|
| **CLIP** | Contrastive | OpenAI's vision-language model using contrastive learning for image-text alignment | Zero-shot classification via similarity scoring |
| **BLIP-2** | Generative | Salesforce's vision-language model with Q-Former and LLM for conditional generation | Zero-shot classification via likelihood estimation |

---

## Datasets

The experiments use agrifood image datasets from various sources. Most datasets are automatically downloaded from **Hugging Face** or **Kaggle** when running the scripts.

### Datasets from Kaggle/Hugging Face

The following datasets are loaded automatically during execution:
- **Beans** (Hugging Face): Bean leaf disease classification
- **Food101** (Hugging Face): 101 food dishes
- **Food11** (Kaggle): 11 food categories
- **Agriculture** (Kaggle): 30 agricultural crops
- **FruitVeg** (Kaggle): 36 fruits and vegetables

**Note**: For Kaggle datasets, you need to configure Kaggle API credentials:
```bash
# Place your kaggle.json in ~/.kaggle/
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Manually-introduced datasets

**MealRec**: Download from [GitHub](https://github.com/WUT-IDEA/MealRec) and place CSV files in the `datasets/` directory:
```
datasets/
├── meal.csv
├── recipe.csv
├── user_meal.csv
└── user_recipe.csv
```

---

## Installation

### Requirements
- Python 3.8+
- CUDA 11+ (for GPUs)
- ~10 GB disk space for datasets
- ~2 GB VRAM minimum

### Quick installation

```bash
# Clone repository
cd multimodal_experiments

# Install dependencies
pip install -r requirements.txt
```

---

## Execution

### Zero-shot experiment

Run CLIP and BLIP2 on all datasets with generic and specific prompts:

```bash
cd models
python unified_experiments.py
```

**Generated outputs:**
- `results/h1_blip2_unified_all_samples.jsonl` - Detailed BLIP2 results
- `results/h1_clip_unified_all_samples.jsonl` - Detailed CLIP results  
- `results/h1_unified_summary.csv` - Metrics summary

### Adding new datasets

To add a dataset without re-running everything:

```bash
cd models

# Sequential
python add_dataset.py --datasets crops

# Parallel (2 GPUs: generic on GPU0, specific on GPU1)
python add_dataset_parallel.py --datasets crops
```

### Experiment analysis

```bash
cd analysis

# Calibration analysis
python calibration_analysis.py

# Advanced metrics analysis
python advanced_analysis.py

# Additional ensembles analysis
python additional_ensembles.py

# Generate graphics
python plot.py
```

---

## Results

Results include:

- **Classification metrics**: Accuracy, Precision, Recall, F1-Score (top-1, top-3, top-5)
- **Calibration**: Brier Score, ECE (Expected Calibration Error)
- **Ensembles**: Averaging of generic vs specific prompts
- **Per-dataset analysis**: Comparative performance in each domain

JSONL files contain for each sample:
- Source dataset
- Evaluation mode (single_blip2, single_clip)
- Prompt set used
- True and predicted label
- Complete probability distribution

---

## Advanced configuration

### Prompts

The experiment uses **8 prompts per dataset**:

**Generic (4):**
- `"a photo of a {label}"`
- `"this is a {label}"`
- `"an image showing {label}"`
- `"a picture of {label}"`

**Specific (4 per dataset):**
- Adapted to the domain (e.g., beans → "bean leaf with {label}")

### GPUs

To use specific GPUs:

```bash
# GPU 1 only
CUDA_VISIBLE_DEVICES=1 python unified_experiments.py

# Parallel on 2 GPUs (automatic in add_dataset_parallel.py)
python add_dataset_parallel.py --datasets crops
```

---

## License

This project is distributed under the MIT License.
