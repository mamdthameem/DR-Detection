# FINDINGS

Status 2026-09-15: all 25 runs complete (5 configurations × train seeds 42, 1337, 2024, 3407, 7).
Numbers only, no interpretation. Sources: `analysis/findings_numbers.md`,
`analysis/paired_*_summary.csv`, `analysis/per_run/*/confusion_matrix.csv`, `efficiency.json`,
`lr_schedule.csv`, `batchnorm_check.json`, and the run log where marked (4 printed decimals).

Environment of all Kaggle runs and measurements (`environment.json`): Tesla T4 (GPU 0 of 2,
14.56 GiB), driver 580.159.04, CUDA 12.8, cuDNN 91002, torch 2.10.0+cu128, torchvision
0.25.0+cu128, timm 1.0.26, Python 3.12.13, NumPy 2.0.2, OpenCV 4.13.0, thop 0.1.1-2209072238;
Intel Xeon @ 2.00 GHz, 4 logical / 2 physical cores, 31.35 GiB RAM.

## 1. Seed-42 reproduction of b0_baseline — exact

Comparison with `==`, metrics recomputed from the saved arrays with the original
`evaluate.compute_metrics` formulas:

| metric | original | seed-42 rerun | exact |
|---|---|---|---|
| accuracy | 0.7509090909090909 | 0.7509090909090909 | yes |
| cohen_kappa | 0.633963877311104 | 0.633963877311104 | yes |
| mcc | 0.6388268576698753 | 0.6388268576698753 | yes |
| precision_macro | 0.5919383474782948 | 0.5919383474782948 | yes |
| recall_macro | 0.6300522427481328 | 0.6300522427481328 | yes |
| f1_macro | 0.6001622560049165 | 0.6001622560049165 | yes |
| auc_macro_ovr | 0.9208481694618698 | 0.9208481694618698 | yes |
| confusion matrix | [[260,11,0,0,0],[7,35,14,0,0],[1,30,79,19,21],[0,3,4,13,9],[0,6,8,4,26]] | identical | yes |
| selected epoch | 4 | 4 (early stop at epoch 11) | yes |
| XAI indices, original selection rule | {0:263, 1:269, 2:285, 3:352, 4:311} | identical | yes |

Settings of every run: cudnn.benchmark False, cudnn.deterministic False, DataLoader shuffling
from the global torch RNG (no seeded generator), torch.use_deterministic_algorithms off.
The 2-epoch smoke run and the reproduction run, in separate processes, logged identical
epoch-1 and epoch-2 train/val loss, accuracy and QWK.

`torch.use_deterministic_algorithms(True)` probe (`deterministic_algorithms_probe.json`;
CUBLAS_WORKSPACE_CONFIG=:4096:8; one forward + backward pass, batch 4, Tesla T4): no op raised
for efficientnet_b0, resnet50, vgg16 or inceptionv3. Not enabled in the training runs.

### Seed-42 reruns of the other configurations (information only)

Original accuracy is computed from the original confusion matrix; rerun accuracies of vgg16
and inceptionv3 across seeds are from the run log.

| configuration | original: accuracy, selected epoch | seed-42 rerun: accuracy, selected epoch | confusion matrix identical | rerun accuracy, 5 seeds (min–max) |
|---|---|---|---|---|
| b0_clahe | 0.7673, 4 | 0.7673, 4 | PENDING (matrix file not yet received) | 0.7273–0.7673 |
| resnet50 | 0.7691, 20 | 0.7200, 18 | no | 0.7200–0.7418 |
| vgg16 | 0.7327, 6 | 0.7418, 6 | no | 0.7418–0.7764 |
| inceptionv3 | 0.7636, 3 | 0.7273, 2 | no | 0.7200–0.7673 |

Seed-42 rerun confusion matrices (rows = true grade):
- resnet50: [[260,10,1,0,0],[7,28,20,0,1],[3,21,72,32,22],[0,0,6,15,8],[0,6,7,10,21]]
  (original [[261,8,2,0,0],[9,36,11,0,0],[4,23,88,21,14],[0,2,5,15,7],[0,8,6,7,23]])
- vgg16: [[257,13,1,0,0],[2,43,11,0,0],[1,38,72,31,8],[0,4,2,16,7],[0,6,8,10,20]]
  (original [[261,9,0,0,1],[3,34,17,1,1],[0,23,69,49,9],[0,1,5,17,6],[0,4,4,14,22]])
- inceptionv3: [[260,9,2,0,0],[9,35,11,0,1],[1,41,73,2,33],[0,2,9,4,14],[0,9,7,0,28]]
  (original [[259,10,2,0,0],[9,28,17,1,1],[5,20,97,11,17],[0,1,8,11,9],[0,7,5,7,25]])

## 2. Kappa per configuration (test set, 5 seeds)

| configuration | unweighted kappa, mean ± SD | range | quadratic-weighted kappa (QWK), mean ± SD | range |
|---|---|---|---|---|
| b0_baseline | 0.6538 ± 0.0226 | 0.6340–0.6845 | 0.8531 ± 0.0126 | 0.8426–0.8745 |
| b0_clahe | 0.6359 ± 0.0186 | 0.6050–0.6531 | 0.8485 ± 0.0087 | 0.8412–0.8628 |
| resnet50 | 0.6024 ± 0.0130 | 0.5885–0.6199 | 0.8437 ± 0.0100 | 0.8328–0.8599 |
| vgg16 | 0.6476 ± 0.0171 | 0.6262–0.6701 | 0.8615 ± 0.0029 | 0.8581–0.8651 |
| inceptionv3 | 0.6257 ± 0.0279 | 0.5937–0.6519 | 0.8395 ± 0.0201 | 0.8170–0.8593 |

SD: sample SD across seeds (ddof = 1).

## Definitions for sections 3 and 4

- Paired by training seed; delta = other configuration − b0_baseline.
- **Exceeds seed variance**: the observed seed ranges (min–max) of the two configurations do
  not overlap. Also reported: the number of seeds with a positive / negative delta.
- 34 metrics per comparison: accuracy, macro precision / recall / F1, MCC, unweighted kappa,
  QWK, macro one-vs-rest AUC, mean absolute grade error, per-grade precision, recall, F1 and
  one-vs-rest AUC, and counts of predictions off by 0–4 grades.

## 3. b0_baseline vs b0_clahe

| metric | b0_baseline mean ± SD | b0_clahe mean ± SD | delta mean ± SD | seeds +/− | exceeds seed variance |
|---|---|---|---|---|---|
| accuracy | 0.7655 ± 0.0175 | 0.7527 ± 0.0152 | −0.0127 ± 0.0302 | 2/3 | no |
| precision_macro | 0.6086 ± 0.0176 | 0.6036 ± 0.0073 | −0.0050 ± 0.0194 | 2/3 | no |
| recall_macro | 0.6415 ± 0.0076 | 0.6344 ± 0.0158 | −0.0071 ± 0.0146 | 1/4 | no |
| f1_macro | 0.6160 ± 0.0167 | 0.6022 ± 0.0111 | −0.0137 ± 0.0231 | 1/4 | no |
| mcc | 0.6570 ± 0.0204 | 0.6407 ± 0.0151 | −0.0163 ± 0.0317 | 1/4 | no |
| kappa_unweighted | 0.6538 ± 0.0226 | 0.6359 ± 0.0186 | −0.0179 ± 0.0376 | 2/3 | no |
| kappa_quadratic_weighted | 0.8531 ± 0.0126 | 0.8485 ± 0.0087 | −0.0046 ± 0.0173 | 2/3 | no |
| auc_macro_ovr | 0.9184 ± 0.0032 | 0.9182 ± 0.0023 | −0.0001 ± 0.0046 | 3/2 | no |
| mean_abs_grade_error | 0.3171 ± 0.0260 | 0.3258 ± 0.0178 | 0.0087 ± 0.0364 | 1/3 (1 zero) | no |
| recall_grade0 | 0.9565 ± 0.0071 | 0.9587 ± 0.0040 | 0.0022 ± 0.0033 | 2/0 (3 zero) | no |
| recall_grade1 | 0.6000 ± 0.0710 | 0.5964 ± 0.0530 | −0.0036 ± 0.0869 | 1/4 | no |
| recall_grade2 | 0.5973 ± 0.0783 | 0.5533 ± 0.0889 | −0.0440 ± 0.1655 | 3/2 | no |
| recall_grade3 | 0.5172 ± 0.0809 | 0.6000 ± 0.1023 | 0.0828 ± 0.1304 | 2/1 (2 zero) | no |
| recall_grade4 | 0.5364 ± 0.0655 | 0.4636 ± 0.0614 | −0.0727 ± 0.0566 | 0/5 | no |

- Metrics exceeding seed variance: 0 of 34.
- Metrics with the same delta sign in all 5 seeds: recall_grade4 (5 negative),
  precision_grade4 (5 positive; 0.4903 ± 0.0602 vs 0.5320 ± 0.0574), off_by_2 (5 negative;
  31.0 ± 6.7 vs 27.0 ± 5.2).
- The two configurations also differ in augmentation and DataLoader workers (unchanged from the
  original runs, recorded in each `config.json`):

| | b0_baseline | b0_clahe |
|---|---|---|
| preprocessing | crop → resize 224 → CLAHE on each BGR channel (`preprocess.py`) | crop → CLAHE on LAB L-channel → resize 224 (`preprocess_effnet.py`) |
| RandomRotation | 15° | 20° |
| ColorJitter brightness / contrast | 0.2 / 0.2 | 0.1 / 0.1 |
| DataLoader workers | 2 | 4 |

### 3b. Matched control: b0_baseline vs b0_clahe_matched — PENDING

b0_clahe_matched reads b0_clahe's LAB-CLAHE images (PNG cache written by
`preprocess_effnet.py`; each run checks that all 3,662 images are present and that 20 test
images are pixel-identical to on-the-fly preprocessing) through b0_baseline's data pipeline.

Identical to b0_baseline: model and head, hyperparameters, split, class weights, the `dataset.py`
loader, train transforms (Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15),
ColorJitter(0.2, 0.2), in that order), eval transforms, 2 DataLoader workers, training seeds.
No random draw depends on image content, so each seed also has the same shuffle order and
augmentation draws as the b0_baseline run with that seed.

Remaining difference, the preprocessing variant:

| | b0_baseline | b0_clahe_matched |
|---|---|---|
| CLAHE applied to | each BGR channel | LAB L-channel |
| CLAHE applied at | 224 × 224, after resizing | the cropped full-resolution image, before resizing |
| border crop, CLAHE clip limit / tiles, resize interpolation | tolerance 7; 2.0 / 8×8; INTER_AREA | same |

Sources when complete: `analysis/paired_b0_baseline_vs_b0_clahe_matched_summary.csv`, and
`analysis/paired_b0_clahe_vs_b0_clahe_matched_summary.csv` (same images; differing only in
augmentation and DataLoader workers).

## 4. b0_baseline vs resnet50

| metric | b0_baseline mean ± SD | resnet50 mean ± SD | delta mean ± SD | seeds +/− | exceeds seed variance |
|---|---|---|---|---|---|
| accuracy | 0.7655 ± 0.0175 | 0.7298 ± 0.0085 | −0.0356 ± 0.0188 | 0/5 | yes |
| precision_macro | 0.6086 ± 0.0176 | 0.5626 ± 0.0148 | −0.0460 ± 0.0216 | 0/5 | yes |
| recall_macro | 0.6415 ± 0.0076 | 0.5921 ± 0.0182 | −0.0494 ± 0.0146 | 0/5 | yes |
| f1_macro | 0.6160 ± 0.0167 | 0.5610 ± 0.0161 | −0.0550 ± 0.0242 | 0/5 | yes |
| mcc | 0.6570 ± 0.0204 | 0.6074 ± 0.0136 | −0.0496 ± 0.0237 | 0/5 | yes |
| kappa_unweighted | 0.6538 ± 0.0226 | 0.6024 ± 0.0130 | −0.0514 ± 0.0259 | 0/5 | yes |
| kappa_quadratic_weighted | 0.8531 ± 0.0126 | 0.8437 ± 0.0100 | −0.0094 ± 0.0212 | 1/4 | no |
| auc_macro_ovr | 0.9184 ± 0.0032 | 0.9110 ± 0.0039 | −0.0073 ± 0.0041 | 0/5 | no |
| mean_abs_grade_error | 0.3171 ± 0.0260 | 0.3476 ± 0.0123 | 0.0305 ± 0.0308 | 4/1 | no |
| recall_grade0 | 0.9565 ± 0.0071 | 0.9601 ± 0.0071 | 0.0037 ± 0.0074 | 2/1 (2 zero) | no |
| recall_grade1 | 0.6000 ± 0.0710 | 0.5500 ± 0.0832 | −0.0500 ± 0.1368 | 1/4 | no |
| recall_grade2 | 0.5973 ± 0.0783 | 0.5147 ± 0.0369 | −0.0827 ± 0.0934 | 0/5 | no |
| recall_grade3 | 0.5172 ± 0.0809 | 0.5310 ± 0.0189 | 0.0138 ± 0.0932 | 3/2 | no |
| recall_grade4 | 0.5364 ± 0.0655 | 0.4045 ± 0.1009 | −0.1318 ± 0.0929 | 0/5 | no |

- Metrics exceeding seed variance: 11 of 34 — the six marked above and
  precision_grade3 (0.3699 ± 0.0401 vs 0.2652 ± 0.0165; delta −0.1047 ± 0.0351),
  f1_grade3 (0.4262 ± 0.0203 vs 0.3535 ± 0.0172; −0.0727 ± 0.0218),
  auc_ovr_grade2 (0.9213 ± 0.0063 vs 0.9028 ± 0.0055; −0.0185 ± 0.0098),
  off_by_0 (421.0 ± 9.6 vs 401.4 ± 4.7; −19.6 ± 10.4),
  off_by_1 (90.8 ± 9.1 vs 115.0 ± 4.6; +24.2 ± 8.8).
- Metrics with the same delta sign in all 5 seeds but overlapping ranges: auc_macro_ovr,
  recall_grade2, recall_grade4, f1_grade2, f1_grade4, auc_ovr_grade0, auc_ovr_grade1
  (all 5 negative).

## 5. Per-seed QWK ranking of the five configurations

Test QWK per run (run log, 4 decimals):

| configuration | seed 42 | seed 1337 | seed 2024 | seed 3407 | seed 7 |
|---|---|---|---|---|---|
| b0_baseline | 0.8532 | 0.8494 | 0.8745 | 0.8457 | 0.8426 |
| b0_clahe | 0.8442 | 0.8503 | 0.8440 | 0.8628 | 0.8412 |
| resnet50 | 0.8438 | 0.8394 | 0.8328 | 0.8426 | 0.8599 |
| vgg16 | 0.8591 | 0.8623 | 0.8651 | 0.8581 | 0.8629 |
| inceptionv3 | 0.8170 | 0.8186 | 0.8536 | 0.8593 | 0.8491 |

| seed | ordering by test QWK (highest first) |
|---|---|
| 42 | vgg16 > b0_baseline > b0_clahe > resnet50 > inceptionv3 |
| 1337 | vgg16 > b0_clahe > b0_baseline > resnet50 > inceptionv3 |
| 2024 | b0_baseline > vgg16 > inceptionv3 > b0_clahe > resnet50 |
| 3407 | b0_clahe > inceptionv3 > vgg16 > b0_baseline > resnet50 |
| 7 | vgg16 > resnet50 > inceptionv3 > b0_baseline > b0_clahe |

- **The ranking is not stable across seeds**: 5 distinct orderings in 5 seeds; no ties.
- Top configuration: vgg16 (seeds 42, 1337, 7), b0_baseline (2024), b0_clahe (3407).
- Rank range: b0_baseline 1–4, b0_clahe 1–5, resnet50 2–5, vgg16 1–3, inceptionv3 2–5.

## 6. Learning-rate schedule

`Adam(lr=1e-4, weight_decay=1e-4)` with `CosineAnnealingLR(T_max=10, eta_min=0)`, one
`scheduler.step()` per epoch, read from `optimizer.param_groups[0]["lr"]` before the step
(the rate used by that epoch's updates, as recorded in `epoch_log.csv`).
`lr_schedule.csv` from Kaggle (torch 2.10.0+cu128) has the same 30 values as torch 2.12.1:

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

vgg16 validation at epochs 10 and 11 (run log): identical in all five seeds.

| seed | val loss epoch 10 / 11 | val acc epoch 10 / 11 | val QWK epoch 10 / 11 |
|---|---|---|---|
| 42 | 0.8462 / 0.8462 | 0.7855 / 0.7855 | 0.9027 / 0.9027 |
| 1337 | 0.7479 / 0.7479 | 0.8036 / 0.8036 | 0.9064 / 0.9064 |
| 2024 | 0.8475 / 0.8475 | 0.8109 / 0.8109 | 0.9112 / 0.9112 |
| 3407 | 0.7696 / 0.7696 | 0.7891 / 0.7891 | 0.8952 / 0.8952 |
| 7 | 0.8140 / 0.8140 | 0.7909 / 0.7909 | 0.8990 / 0.8990 |

Examples for BatchNorm models, val loss epoch 10 / 11: resnet50 seed 42 0.8860 / 0.8830;
b0_baseline seed 2024 0.9250 / 0.9148; inceptionv3 seed 3407 1.0934 / 1.1131.

Selected epoch / epochs run (early stopping, patience 7; no run reached 30):

| configuration | seed 42 | seed 1337 | seed 2024 | seed 3407 | seed 7 |
|---|---|---|---|---|---|
| b0_baseline | 4 / 11 | 3 / 10 | 4 / 11 | 5 / 12 | 3 / 10 |
| b0_clahe | 4 / 11 | 3 / 10 | 3 / 10 | 3 / 10 | 3 / 10 |
| resnet50 | 18 / 25 | 9 / 16 | 19 / 26 | 16 / 23 | 17 / 24 |
| vgg16 | 6 / 13 | 13 / 20 | 6 / 13 | 9 / 16 | 8 / 15 |
| inceptionv3 | 2 / 9 | 3 / 10 | 3 / 10 | 4 / 11 | 3 / 10 |

## 7. BatchNorm layers per architecture

`batchnorm_check.json` (instances of `torch.nn.modules.batchnorm._BatchNorm`). The pipeline
builds every backbone with timm (`model.py`).

| architecture | timm: layers (type) | timm: BatchNorm weight + bias | torchvision: layers (type) | torchvision: BatchNorm weight + bias |
|---|---|---|---|---|
| efficientnet_b0 | 49 (BatchNormAct2d) | 42,016 | 49 (BatchNorm2d) | 42,016 |
| resnet50 | 53 (BatchNorm2d) | 53,120 | 53 (BatchNorm2d) | 53,120 |
| vgg16 | 0 | 0 | 0 | 0 |
| inceptionv3 | 94 (BatchNormAct2d) | 34,432 | 94 (BatchNorm2d) | 34,432 |

vgg16 has no BatchNorm layers in timm or torchvision; the other three have BatchNorm in both.

## 8. EfficientNet-B0 parameters: 4.62M in the paper vs 4,665,985

Measured by `benchmark.py` (`efficiency.json`):

| candidate | parameters | millions |
|---|---|---|
| trained model, total | 4,665,985 | 4.67 |
| trained model, trainable | 4,665,985 | 4.67 |
| total minus BatchNorm weight and bias | 4,623,969 | **4.62** |
| thop.profile parameter count | 4,623,969 | **4.62** |
| state_dict incl. BatchNorm running statistics | 4,708,050 | 4.71 |
| backbone only (timm efficientnet_b0, num_classes=0) | 4,007,548 | 4.01 |
| head only (trainable part of a frozen-backbone variant) | 658,437 | 0.66 |
| backbone + single Linear(1280→5) | 4,013,953 | 4.01 |
| timm efficientnet_b0 with 1000-class classifier | 5,288,548 | 5.29 |
| torchvision efficientnet_b0 with 1000-class classifier | 5,288,548 | 5.29 |
| torchvision efficientnet_b0 features + the pipeline head | 4,665,985 | 4.67 |

- 4,665,985 = backbone 4,007,548 + head 658,437. The two candidates that round to 4.62M are
  both 4,623,969 = 4,665,985 − 42,016 BatchNorm weights and biases.
- thop lists `timm.layers.norm_act.BatchNormAct2d` (49 modules, 42,016 parameters) among the
  module types without a thop rule; their parameters are not in thop's count.
- The same applies to InceptionV3: thop reports 22,802,789 parameters, total 22,837,221
  (94 BatchNormAct2d modules, 34,432 parameters). ResNet-50 and VGG-16: thop count equals the total.
- Pooled feature size of timm efficientnet_b0 at input 224, 240 and 299: 1280, 1280, 1280.
- This pipeline never ran thop before (`model_summary.txt`: `FLOPs (est.) : N/A (install thop)`),
  so these files do not show which tool produced the 4.62M in the paper.

## 9. Efficiency (`efficiency.json`)

Batch size 1, 3×224×224, float32, eval mode, 20 warmup passes discarded, 550 test images;
GPU Tesla T4; CPU Intel Xeon @ 2.00 GHz with 2 torch threads. Parameters are totals (all trainable).

| architecture | parameters | thop MACs | torch flop counter | GPU ms mean / p95 | CPU ms mean / p95 | GPU peak allocated / reserved MB | CPU peak RSS MB (process start) |
|---|---|---|---|---|---|---|---|
| efficientnet_b0 | 4,665,985 | 385,256,672 | 770,385,344 | 7.92 / 8.40 | 35.97 / 37.91 | 38.8 / 58 | 897.6 (831.9) |
| resnet50 | 24,559,685 | 4,132,745,728 | 8,176,374,784 | 6.14 / 6.94 | 79.24 / 84.99 | 130.5 / 150 | 971.5 (832.4) |
| vgg16 | 136,360,773 | 15,468,276,224 | 30,936,536,064 | 10.92 / 11.09 | 406.55 / 426.32 | 560.3 / 594 | 1791.5 (831.8) |
| inceptionv3 | 22,837,221 | 2,836,977,504 | 5,673,848,512 | 12.24 / 13.61 | 90.34 / 95.18 | 104.5 / 122 | 970.2 (832.4) |

Counting conventions:
- thop: MACs (multiply-accumulate operations) of modules with a thop rule; BatchNormAct2d,
  SiLU and Sigmoid modules have no rule and contribute 0.
- torch.utils.flop_counter: convolution and matrix multiplication counted as
  2 × the number of multiplications (PyTorch source: `... * c_out * c_in * 2`, `m * n * 2 * k`),
  i.e. FLOPs under the FLOPs = 2 × MACs convention. The `torch_flop_counter` convention sentence
  in the first `efficiency.json` ("no factor 2") is wrong; the numbers are unaffected.
