# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diabetic Retinopathy (DR) Detection pipeline using PyTorch. Trains four CNN backbones (EfficientNet-B0, ResNet-50, VGG-16, InceptionV3) on the APTOS 2019 fundus image dataset to classify DR severity into 5 classes (No DR → Proliferative DR). Includes a full explainability suite (Grad-CAM, SHAP, LIME).

Designed to run in a Kaggle notebook environment; all paths are hardcoded to `/kaggle/input/` and `/kaggle/working/`.

## Running the Pipeline

```bash
# Full pipeline (preprocess → train → evaluate → XAI)
python main.py

# Skip preprocessing (images already preprocessed)
python main.py --skip-preprocess

# Skip training (checkpoints already saved)
python main.py --skip-train

# Evaluation + XAI only
python main.py --eval-only

# XAI only
python main.py --xai-only

# Train specific models only (valid names: efficientnet_b0, resnet50, vgg16, inceptionv3)
python main.py --models efficientnet_b0 resnet50
```

Every script is also runnable standalone with its own `--help`. Key overrides:

```bash
python train.py --models efficientnet_b0 --batch-size 16 --max-epochs 10
python evaluate.py --models efficientnet_b0 resnet50
python xai_shap.py --n-background 20    # reduce if GPU OOM (default 50)
python xai_lime.py
python xai_compare.py
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

## Architecture

### Pipeline (main.py)

Runs tasks 1–9 in sequence. Each task maps to one module:

| Task | File | Role |
|------|------|------|
| 1 | `preprocess.py` | Black border crop + CLAHE + resize to 224×224 |
| 2–3 | `dataset.py`, `model.py` | Data splits, loaders, model summary |
| 4 | `train.py` | Training loop with early stopping |
| 5 | `evaluate.py` | Metrics, confusion matrices, ROC curves |
| 6 | `xai_gradcam.py` | Grad-CAM heatmaps |
| 7 | `xai_shap.py` | SHAP GradientExplainer |
| 8 | `xai_lime.py` | LIME superpixel explanations |
| 9 | `xai_compare.py` | Assembles publication-ready XAI comparison grid |

### Data flow

```
/kaggle/input/aptos2019-blindness-detection/
  train.csv + train_images/
        ↓ preprocess.py
/kaggle/working/data/aptos_preprocessed/
        ↓ dataset.py (70/15/15 stratified split, seed=42)
        ↓ train.py
/kaggle/working/checkpoints/{model_name}_best.pth
        ↓ evaluate.py
/kaggle/working/outputs/
  results_comparison.csv
  xai_sample_indices.json      ← one test image per DR class for XAI
        ↓ xai_gradcam, xai_shap, xai_lime → xai/panels/
        ↓ xai_compare.py
  xai/xai_comparison_grid.png
```

### Key design decisions

- **`APTOSModel` (model.py):** timm backbone (`num_classes=0, global_pool='avg'`) + shared classifier head `Dropout(0.3) → Linear(→512) → ReLU → Dropout(0.2) → Linear(→5)`. In-features are detected at runtime via a **dummy forward pass** (timm's `backbone.num_features` is unreliable for VGG-16 — reports 512 but the backbone outputs 4096), so no per-model hard-coding.
- **`MODEL_REGISTRY` (model.py):** maps the four short keys (`efficientnet_b0`, `resnet50`, `vgg16`, `inceptionv3`) to timm identifiers. Use these exact keys everywhere (`--models`, `--checkpoints`, etc.).
- **Class imbalance:** Inverse-frequency class weights (normalised to sum to `num_classes`) passed to `CrossEntropyLoss`.
- **Training config (`train.py`):** All hyperparameters live in the `TRAIN_CONFIG` dict — `lr=1e-4`, `weight_decay=1e-4`, `max_epochs=30`, `patience=7`, `batch_size=32`. Edit that dict to change them.
- **Training augmentation (`dataset.py`):** Train split gets `RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(15) + ColorJitter(0.2, 0.2)`; val/test get only resize + normalize (ImageNet stats).
- **XAI sample selection (`evaluate.py`):** After evaluation, `xai_sample_indices.json` is written using **EfficientNet-B0's** predictions only (line 175 checks `model_name == "efficientnet_b0"`). The three XAI scripts all read this file — if it is missing or EfficientNet-B0 was not evaluated, XAI steps will fail.
- **Grad-CAM target layers:** EfficientNet-B0 → `conv_head`; ResNet-50 → `layer4[-1]`; VGG-16 → `features[-3]`; InceptionV3 → `Mixed_7c`. Defined in `GRADCAM_LAYERS` dict in `xai_gradcam.py`.
- **SHAP memory:** `xai_shap.py` uses `GradientExplainer` with 50 background samples by default. Pass `--n-background 20` (or lower) if you hit GPU OOM.
- **xai_compare.py dependencies:** Reads `panels/original_class_{i}.png`, `panels/gradcam_class_{i}.png`, `panels/shap_class_{i}.png`, `panels/lime_class_{i}.png`. These are written by the three XAI scripts; running `xai_compare.py` before them will raise `FileNotFoundError`.

### Environment note

Working-directory paths (`/kaggle/working/...` for checkpoints, outputs, preprocessed images) are hardcoded per-module. The **dataset input path is the exception**: `paths.py` resolves it at import time — every script does `from paths import DATASET_CSV` / `DATASET_ROOT` instead of hardcoding `/kaggle/input/...`. `paths.py` probes both `/kaggle/input/aptos2019-blindness-detection` and `/kaggle/input/competitions/aptos2019-blindness-detection` (Kaggle mounts competition data at either, depending on how it's attached), then falls back to a recursive search. To run locally, add your own path to the `_CANDIDATES` list in `paths.py` and replace the `/kaggle/working/` constants in each module.
