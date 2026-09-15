# Kaggle steps for the revision experiments

All training and measurement runs on Kaggle. On your own computer you only push the code to
GitHub. Session 1 does the checks, the smoke run and the seed-42 reproduction; the full
suite waits until we have looked at those results together.

---

## 0. Before anything else

1. **Back up the original results.** Open the original baseline notebook and the LAB-CLAHE
   notebook on Kaggle → **Output** → download everything. Nothing below deletes them; this
   is only a safety copy.
2. **Note the GPU the original run used.** In the baseline notebook, open the log of the
   version that produced the published numbers and find the line printed when training
   started, e.g. `GPU: Tesla P100-PCIE-16GB` or `GPU: Tesla T4`. A different GPU type can
   change the numbers.

## 1. Push the new code to GitHub (on your computer)

```
cd D:\dr_detection
git add -A
git commit -m "Multi-seed instrumentation: run artefacts, reproduction check, analysis, benchmark"
git push
```

---

## 2. Session 1: checks, smoke run, seed-42 reproduction

### 2.1 Open the notebook and check the settings

Open the **original baseline notebook** and click **Edit**. Using the same notebook keeps
the same software environment. Saving a new version later does not remove the earlier
versions or their outputs; they stay under *Version history*.

In the right-hand **Settings** panel:

| setting | value |
|---|---|
| Accelerator | the same GPU type as in step 0.2 |
| Environment | **Pin to original environment** (not "Always use latest environment") |
| Internet | On (needed for `git clone`, pretrained weights and `pip`) |

### 2.2 Stop the old cells from running

Delete the old cells (for example `!python main.py`). They remain in the earlier versions.
The original pipeline does not need to run again.

### 2.3 Add these six cells, in this order

**Cell 1: get the code**
```
%%bash
set -e
cd /kaggle/working
rm -rf DR-Detection
git clone https://github.com/mamdthameem/DR-Detection.git
cd DR-Detection
git log --oneline -1
pip install -q thop --no-deps
```

**Cell 2: preprocessed images for the baseline configurations**
```
%%bash
set -e
cd /kaggle/working/DR-Detection
python preprocess.py --outputs-dir /kaggle/working/preprocess_outputs
```

**Cell 3: checks that need no training** (environment, learning-rate schedule, BatchNorm,
parameters / MACs / latency / memory; the CPU latency part takes the longest)
```
%%bash
set -e
cd /kaggle/working/DR-Detection
python environment_info.py --probe
python verify_lr_and_bn.py
python benchmark.py
```

**Cell 4: smoke run (2 epochs) with the artefact listing**
```
%%bash
set -e
cd /kaggle/working/DR-Detection
python run_suite.py --smoke
```

**Cell 5: seed-42 reproduction run** (a full early-stopping run of b0_baseline)
```
%%bash
cd /kaggle/working/DR-Detection
python run_suite.py --repro-only || echo "NOT REPRODUCED: see the report above"
```

**Cell 6: analysis on what exists so far** (tests analyze.py on real data)
```
%%bash
cd /kaggle/working/DR-Detection
python analyze.py
```

### 2.4 Run it

Top right: **Save Version** → **Save & Run All (Commit)** → **Save**. It runs in the
background, so you can close the browser. When it has finished, open the new version from
*Version history*.

### 2.5 Send me

1. From the log of Cell 4: everything from `Artefact listing` down to
   `SMOKE TEST PASSED` / `FAILED` (also saved as `runs_smoke/smoke_report.txt`).
2. From the log of Cell 5: the block `Reproduction check: b0_baseline__seed42 vs the
   original run` (also saved as `repro_check.json`).
3. These output files: `environment.json`, `deterministic_algorithms_probe.json`,
   `lr_schedule.csv`, `batchnorm_check.json`, `efficiency.json`.

**Do not start the full suite yet.**

---

## 3. Sessions 2, 3, …: the full suite (only after we agree to go ahead)

The 25 runs need more than one 12-hour session. Each session continues where the previous
one stopped.

1. Open the notebook → **Edit**.
2. **Add Input** (right panel) → **Your Work** → **Notebooks** → choose this notebook. Its
   latest saved output appears under `/kaggle/input/...`. In later sessions make sure the
   attached input is the newest version (Kaggle marks inputs that have a newer version).
   If Kaggle does not let you attach the notebook to itself, use **Output** → **New Dataset**
   on the latest version and attach that dataset instead.
3. Keep Cells 1 and 2. Replace Cells 3–6 with this one cell:
   ```
   %%bash
   set -e
   cd /kaggle/working/DR-Detection
   python run_suite.py --resume-from auto
   python analyze.py
   ```
4. **Save Version** → **Save & Run All (Commit)**.
5. Repeat steps 1–4 until the log says `Suite status: 25/25 run(s) complete`.

What happens in each session:

- `--resume-from auto` copies every finished run (a run folder containing `metrics.json`)
  from the attached previous output into `/kaggle/working/runs`, so every new version's
  output again contains all finished runs. The seed-42 run from Session 1 is reused, not
  repeated.
- A run that was cut off restarts from the beginning.
- No new run starts after 10 hours, so the version finishes and saves its output before
  Kaggle's 12-hour limit.
- Order: all seeds of b0_baseline, then b0_clahe, resnet50, vgg16, inceptionv3.
- Nothing except b0_baseline seed 42 runs unless that run reproduced the original numbers.
- Keep the same GPU type and the pinned environment in every session. If they change, the log
  prints `WARNING: this session's environment differs ...`; stop and tell me.

---

## 4. Where the outputs are (`/kaggle/working`)

| path | contents |
|---|---|
| `runs/<config>__seed<seed>/` | `probs.npy`, `logits.npy`, `labels.npy`, `indices.npy`, `preds.npy`, `val_predictions.npz`, `best_model.pth`, `epoch_log.csv`, `config.json`, `metrics.json` |
| `runs_smoke/` | the 2-epoch smoke run and `smoke_report.txt` |
| `analysis/` | `summary_all.csv`, `summary_aggregated.csv`, paired comparisons, QWK ranking, bootstrap intervals, `thresholds/`, `findings_numbers.md` |
| `environment.json`, `deterministic_algorithms_probe.json` | software and hardware versions |
| `lr_schedule.csv`, `batchnorm_check.json` | learning-rate schedule and BatchNorm check |
| `efficiency.json` | parameters, MACs, latency, memory |
| `repro_check.json` | seed-42 reproduction result |
