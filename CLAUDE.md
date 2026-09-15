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
- **Training config (`train.py`):** All hyperparameters live in the `TRAIN_CONFIG` dict — `lr=1e-4`, `weight_decay=1e-4`, `max_epochs=30`, `patience=7`, `t_max=10`, `batch_size=32`. Edit that dict to change them (but see "Revision experiments" below: they are frozen for the paper).
- **Training augmentation (`dataset.py`):** Train split gets `RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(15) + ColorJitter(0.2, 0.2)`; val/test get only resize + normalize (ImageNet stats).
- **XAI sample selection (`evaluate.py`):** After evaluation, `xai_sample_indices.json` is written using **EfficientNet-B0's** predictions only (line 175 checks `model_name == "efficientnet_b0"`). The three XAI scripts all read this file — if it is missing or EfficientNet-B0 was not evaluated, XAI steps will fail.
- **Grad-CAM target layers:** EfficientNet-B0 → `conv_head`; ResNet-50 → `layer4[-1]`; VGG-16 → `features[-3]`; InceptionV3 → `Mixed_7c`. Defined in `GRADCAM_LAYERS` dict in `xai_gradcam.py`.
- **SHAP memory:** `xai_shap.py` uses `GradientExplainer` with 50 background samples by default. Pass `--n-background 20` (or lower) if you hit GPU OOM.
- **xai_compare.py dependencies:** Reads `panels/original_class_{i}.png`, `panels/gradcam_class_{i}.png`, `panels/shap_class_{i}.png`, `panels/lime_class_{i}.png`. These are written by the three XAI scripts; running `xai_compare.py` before them will raise `FileNotFoundError`.

## Revision experiments (multi-seed suite)

Added for the paper's Major Revision. Kaggle workflow: `KAGGLE_STEPS.md`; results: `FINDINGS.md`.

```bash
python run_suite.py --smoke              # 2-epoch b0_baseline seed 42 → /kaggle/working/runs_smoke, verifies artefacts
python run_suite.py --repro-only         # b0_baseline seed 42, must match the original run exactly (repro_check.py)
python run_suite.py --resume-from auto   # 5 configs × SEEDS, resumable across Kaggle sessions
python analyze.py                        # metrics from saved arrays only, never trains
python benchmark.py                      # params, thop MACs, latency, memory → efficiency.json
python verify_lr_and_bn.py               # lr_schedule.csv, batchnorm_check.json
python environment_info.py --probe       # environment.json + use_deterministic_algorithms probe
```

- `experiment_config.py` holds the shared constants: `SPLIT_SEED` (always 42), `CONFIGS` / `CONFIG_ORDER` (b0_baseline, b0_clahe, b0_clahe_matched, resnet50, vgg16, inceptionv3), `XAI_SAMPLE_INDICES`, `ORIGINAL_RESULTS`. The training seeds are `SEEDS` in `run_suite.py`.
- `run_experiment.py` (one subprocess per run) writes `runs/{config}__seed{seed}/`: test-set `probs/logits/labels/indices/preds.npy`, `val_predictions.npz`, `best_model.pth`, `epoch_log.csv`, `config.json`, and `metrics.json` last (its presence marks the run complete).
- **Seed 42 must keep reproducing the original runs**, so the RNG path matches the original `train.py` / `train_effnet.py`: seeding → loaders → model init in the same order, DataLoader shuffling from the global torch RNG (no seeded `generator`), cuDNN defaults, `num_workers` 2 (train.py configs) or 4 (b0_clahe). `--seeded-generator` / `--cudnn-deterministic` exist but change the numbers. Hyperparameters, `T_max`, augmentation, preprocessing, class weights and the split are frozen.
- `b0_clahe` trains through `preprocess_effnet.py` (on-the-fly LAB-CLAHE from raw images, RandomRotation(20), ColorJitter(0.1, 0.1)), not `dataset.py`.
- `b0_clahe_matched` (preprocessing key `lab_clahe_cache`) reads the LAB-CLAHE PNG cache (`python preprocess_effnet.py --skip-visualize` → `/kaggle/working/data/aptos_lab`) through `dataset.py` with b0_baseline's transforms and 2 workers, so only the preprocessing differs from b0_baseline; `run_experiment.py` verifies the cache before training. `analyze.py` pairs it with b0_baseline and b0_clahe; the QWK ranking (`RANKING_CONFIGS`) stays on the original five.
- `train_one_model` returns a dict and logs val kappa / QWK, the optimizer lr and per-epoch seconds; with `out_paths=None` it writes to the original locations.
- `DR_WORK_DIR` overrides `/kaggle/working` for the new scripts.

### Environment note

Working-directory paths (`/kaggle/working/...` for checkpoints, outputs, preprocessed images) are hardcoded per-module. The **dataset input path is the exception**: `paths.py` resolves it at import time — every script does `from paths import DATASET_CSV` / `DATASET_ROOT` instead of hardcoding `/kaggle/input/...`. `paths.py` probes both `/kaggle/input/aptos2019-blindness-detection` and `/kaggle/input/competitions/aptos2019-blindness-detection` (Kaggle mounts competition data at either, depending on how it's attached), then falls back to a recursive search. To run locally, add your own path to the `_CANDIDATES` list in `paths.py` and replace the `/kaggle/working/` constants in each module.
