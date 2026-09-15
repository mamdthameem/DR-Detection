# FINDINGS

Status 2026-09-15. Numbers only, no interpretation. Sections marked **PENDING** are filled
from the Kaggle outputs named in them; `analysis/findings_numbers.md` (written by
`analyze.py`) holds the run-derived tables in the same order.

## 1. Seed-42 reproduction of b0_baseline — PENDING

Source: `repro_check.json`, printed by `python run_suite.py --repro-only`.
The comparison is `==` on the seven original metrics recomputed from `labels.npy`,
`preds.npy`, `probs.npy` with the original `evaluate.compute_metrics` formulas, plus the
confusion matrix, the selected epoch (4) and the XAI indices the original rule selects.

Checked without a GPU: from the original confusion matrix alone, those formulas give exactly
the original accuracy 0.7509090909090909, cohen_kappa 0.633963877311104,
mcc 0.6388268576698753, precision_macro 0.5919383474782948,
recall_macro 0.6300522427481328 and f1_macro 0.6001622560049165. auc_macro_ovr
(0.9208481694618698) needs the rerun's probabilities.

## 2. Kappa per configuration (mean ± SD over seeds) — PENDING

Source: `analysis/findings_numbers.md` §2, `analysis/summary_aggregated_long.csv`
(`kappa_unweighted`, `kappa_quadratic_weighted`).

## 3. b0_baseline vs b0_clahe relative to seed variance — PENDING

Source: `analysis/paired_b0_baseline_vs_b0_clahe_summary.csv`. "Exceeds seed variance"
is defined there as non-overlapping observed seed ranges; the same file also gives the sign
of every paired per-seed difference and |mean difference| against the larger seed SD.

Differences between the two configurations as implemented (unchanged, recorded in each
`config.json`):

| | b0_baseline | b0_clahe |
|---|---|---|
| preprocessing | crop → resize 224 → CLAHE on each BGR channel (`preprocess.py`) | crop → CLAHE on LAB L-channel → resize 224 (`preprocess_effnet.py`) |
| RandomRotation | 15° | 20° |
| ColorJitter brightness / contrast | 0.2 / 0.2 | 0.1 / 0.1 |
| DataLoader workers | 2 | 4 |

## 4. b0_baseline vs resnet50 relative to seed variance — PENDING

Source: `analysis/paired_b0_baseline_vs_resnet50_summary.csv`.

## 5. Per-seed QWK ranking of the five configurations — PENDING

Source: `analysis/qwk_ranking_per_seed.csv`, `analysis/qwk_ranking_stability.json`.

## 6. Learning-rate schedule

`Adam(lr=1e-4, weight_decay=1e-4)` with `CosineAnnealingLR(T_max=10, eta_min=0)`, one
`scheduler.step()` per epoch, read from `optimizer.param_groups[0]["lr"]` before the step
(the rate used by that epoch's updates, and the value `epoch_log.csv` now records).
Verified with torch 2.12.1 (CPU):

| epoch | learning rate (repr) |
|---|---|
| 1 | 0.0001 |
| 10 | 2.447174185242323e-06 |
| 11 | 0.0 (exactly) |
| 12 | 2.4471741852423237e-06 |
| 16 | 5.0000000000000226e-05 |
| 20 | 9.755282581475812e-05 |
| 21 | 0.00010000000000000044 (not bit-identical to 1e-4; difference 4.3e-19) |
| 26 | 5.0000000000000226e-05 |
| 30 | 2.44717418524234e-06 |

All 30 values: `lr_schedule.csv`. PENDING: the same file from Kaggle
(`python verify_lr_and_bn.py`) for the torch version used in training.

## 7. BatchNorm layers per architecture — PENDING

Source: `batchnorm_check.json` from `python verify_lr_and_bn.py` on Kaggle. It counts
`torch.nn.modules.batchnorm._BatchNorm` instances in the timm models the pipeline trains
(`model.py` builds every backbone with `timm.create_model`, not torchvision) and in the
torchvision `efficientnet_b0`, `resnet50`, `vgg16`, `inception_v3`.

## 8. EfficientNet-B0 parameters: 4.62M in the paper vs 4,665,985

Hand arithmetic from the architecture definitions. `benchmark.py` measures every figure
below; the measured values in `efficiency.json` are PENDING.

- 4,665,985 = backbone 4,007,548 + head 658,437
  (Linear(1280→512) 655,872 + Linear(512→5) 2,565). All parameters are trainable.
- EfficientNet-B0 has 49 BatchNorm layers with 21,008 channels in total, i.e. 42,016 BatchNorm
  weight + bias parameters.
- 4,665,985 − 42,016 = **4,623,969 → 4.62M**. Of the candidates below, it is the only one
  that rounds to 4.62M.

| candidate | parameters | millions |
|---|---|---|
| trained model, total = trainable | 4,665,985 | 4.67 |
| total minus BatchNorm weight and bias | 4,623,969 | **4.62** |
| backbone only | 4,007,548 | 4.01 |
| head only (trainable part of a frozen-backbone variant) | 658,437 | 0.66 |
| backbone + single Linear(1280→5) | 4,013,953 | 4.01 |
| timm / torchvision efficientnet_b0 with 1000-class classifier | 5,288,548 | 5.29 |
| state_dict incl. BatchNorm running mean/var and num_batches_tracked | 4,708,050 | 4.71 |

- Input configuration: EfficientNet-B0's pooled feature size is 1280 at any input resolution,
  so its parameter count does not change with input size (`benchmark.py` records the feature
  size at 224, 240 and 299).
- A tool that produces 4,623,969 for this model: `thop.profile` counts parameters only for
  module types in its rule table. timm's EfficientNet uses `BatchNormAct2d` (a `BatchNorm2d`
  subclass with child modules), which has no rule, so its weights and biases are not counted.
  `efficiency.json` records thop's own parameter count (PENDING).
- This pipeline never ran thop (`model_summary.txt`: `FLOPs (est.) : N/A (install thop)`),
  so these files do not show which tool produced the 4.62M in the paper.
