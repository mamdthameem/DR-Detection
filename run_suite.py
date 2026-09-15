"""
Resumable multi-seed suite. Every run executes run_experiment.py in its own process.

    python run_suite.py --smoke              # 2-epoch b0_baseline seed 42 → runs_smoke/, verify + list
    python run_suite.py --repro-only         # full b0_baseline seed 42, then the reproduction check
    python run_suite.py                      # every configuration × SEEDS
    python run_suite.py --resume-from auto   # first copy finished runs from a previous notebook
                                             # version attached under /kaggle/input

Order: all seeds of each configuration in CONFIG_ORDER (b0_baseline, b0_clahe,
b0_clahe_matched, resnet50, vgg16, inceptionv3). b0_clahe_matched needs the LAB-CLAHE cache
(python preprocess_effnet.py --skip-visualize). A run whose
directory already holds metrics.json is skipped, so the command can be re-run until
everything is complete; an unfinished run restarts from scratch. No run other than
b0_baseline seed 42 starts until that run reproduces the original numbers exactly
(--skip-repro-gate overrides).
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse
import subprocess

import numpy as np
import pandas as pd

from experiment_config import (RUNS_ROOT, SMOKE_ROOT, CONFIG_ORDER, REPRO_RUN, RUN_FILES,
                               EPOCH_LOG_COLUMNS, N_TEST, NUM_CLASSES, SPLIT_SEED,
                               run_name, write_json)
from repro_check import check_reproduction, print_report, report_path

SEEDS = [42, 1337, 2024, 3407, 7]   # training seeds, run in this order; trim as needed

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
# Top-level outputs carried over by --resume-from (existing local files are never overwritten)
CARRY_OVER_FILES = ["environment.json", "efficiency.json", "lr_schedule.csv",
                    "batchnorm_check.json", "deterministic_algorithms_probe.json",
                    "repro_check.json"]


def is_complete(run_dir: str) -> bool:
    return os.path.isfile(os.path.join(run_dir, "metrics.json"))


# ── Resuming across Kaggle sessions ────────────────────────────────────────────

def find_resume_source(search_root: str = "/kaggle/input"):
    """The attached runs/ directory holding the most finished runs, or None."""
    counts = {}
    pattern = os.path.join(search_root, "**", "runs", "*", "metrics.json")
    for hit in glob.glob(pattern, recursive=True):
        root = os.path.dirname(os.path.dirname(hit))
        counts[root] = counts.get(root, 0) + 1
    return max(counts, key=counts.get) if counts else None


def resume_from(src_runs_root: str, runs_root: str) -> None:
    copied = 0
    for src in sorted(glob.glob(os.path.join(src_runs_root, "*__seed*"))):
        dst = os.path.join(runs_root, os.path.basename(src))
        if not is_complete(src) or is_complete(dst):
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)          # unfinished run from this session
        shutil.copytree(src, dst)
        copied += 1
    src_work = os.path.dirname(os.path.normpath(src_runs_root))
    dst_work = os.path.dirname(os.path.normpath(runs_root))
    carried = []
    for fname in CARRY_OVER_FILES:
        src, dst = os.path.join(src_work, fname), os.path.join(dst_work, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            carried.append(fname)
    print(f"Resumed from {src_runs_root}: copied {copied} finished run(s); "
          f"carried over: {', '.join(carried) or 'nothing'}")


# ── One run ────────────────────────────────────────────────────────────────────

def run_one(config_name: str, seed: int, runs_root: str, args, max_epochs=None) -> int:
    cmd = [sys.executable, "-u", os.path.join(CODE_DIR, "run_experiment.py"),
           "--config", config_name, "--seed", str(seed), "--runs-root", runs_root]
    if max_epochs is not None:
        cmd += ["--max-epochs", str(max_epochs)]
    for flag, value in (("--img-dir", args.img_dir), ("--raw-dir", args.raw_dir),
                        ("--lab-dir", args.lab_dir), ("--csv-path", args.csv_path)):
        if value:
            cmd += [flag, value]
    if args.cudnn_deterministic:
        cmd.append("--cudnn-deterministic")
    if args.seeded_generator:
        cmd.append("--seeded-generator")
    return subprocess.run(cmd).returncode


# ── Smoke test: artefact verification + listing ────────────────────────────────

def verify_artefacts(run_dir: str) -> list:
    """[(check, passed, detail)] for one run directory."""
    checks = []

    def check(name, passed, detail=""):
        checks.append((name, bool(passed), str(detail)))

    for fname in RUN_FILES:
        check(f"present: {fname}", os.path.isfile(os.path.join(run_dir, fname)))
    if not all(passed for _, passed, _ in checks):
        return checks

    probs, logits, labels, indices, preds = (
        np.load(os.path.join(run_dir, f"{key}.npy"))
        for key in ("probs", "logits", "labels", "indices", "preds"))
    check(f"probs.npy float ({N_TEST}, {NUM_CLASSES})",
          probs.shape == (N_TEST, NUM_CLASSES) and np.issubdtype(probs.dtype, np.floating),
          f"{probs.shape} {probs.dtype}")
    check("probs rows sum to 1", np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))
    check(f"logits.npy float ({N_TEST}, {NUM_CLASSES})",
          logits.shape == (N_TEST, NUM_CLASSES) and np.issubdtype(logits.dtype, np.floating),
          f"{logits.shape} {logits.dtype}")
    z = logits.astype(np.float64)
    softmax = np.exp(z - z.max(axis=1, keepdims=True))
    softmax /= softmax.sum(axis=1, keepdims=True)
    check("softmax(logits) == probs", np.allclose(softmax, probs, atol=1e-5))
    check(f"labels.npy int ({N_TEST},)",
          labels.shape == (N_TEST,) and np.issubdtype(labels.dtype, np.integer),
          f"{labels.shape} {labels.dtype}")
    check("indices.npy == 0..N-1", np.array_equal(indices, np.arange(N_TEST)))
    check("preds.npy == argmax(logits)",
          preds.shape == (N_TEST,) and np.array_equal(preds, logits.argmax(axis=1)))
    with np.load(os.path.join(run_dir, "val_predictions.npz")) as val:
        shapes = {k: val[k].shape for k in val.files}
        check("val_predictions.npz arrays",
              set(val.files) == {"probs", "logits", "labels", "preds", "indices"}
              and val["probs"].shape == (len(val["labels"]), NUM_CLASSES), shapes)
    check("best_model.pth non-empty",
          os.path.getsize(os.path.join(run_dir, "best_model.pth")) > 0)

    log = pd.read_csv(os.path.join(run_dir, "epoch_log.csv"))
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "metrics.json")) as f:
        met = json.load(f)
    check("epoch_log.csv columns", list(log.columns) == EPOCH_LOG_COLUMNS, list(log.columns))
    selected = log.loc[log["is_selected_epoch"].astype(str).str.lower() == "true", "epoch"].tolist()
    check("one selected epoch, same as metrics.json",
          len(selected) == 1 and selected[0] == met["training"]["selected_epoch"], selected)
    check("epochs logged == epochs_run", len(log) == met["training"]["epochs_run"], len(log))
    check("epoch 1 learning_rate == configured lr",
          log["learning_rate"].iloc[0] == cfg["hyperparameters"]["lr"],
          log["learning_rate"].iloc[0])
    check("config: split_seed and test size",
          cfg["split_seed"] == SPLIT_SEED and cfg["split"]["n_test"] == N_TEST,
          f"split_seed {cfg['split_seed']}, n_test {cfg['split']['n_test']}")
    check("config: XAI indices have their grades",
          cfg["xai_sample_indices"]["grades_verified"],
          {g: d["grade_at_index"] for g, d in cfg["xai_sample_indices"]["detail"].items()})
    check("metrics: legacy metrics == evaluate.compute_metrics",
          met["legacy_metrics_match_evaluate_py"])
    return checks


def _describe(path: str) -> str:
    try:
        if path.endswith(".npy"):
            arr = np.load(path, mmap_mode="r")
            return f"shape {arr.shape}, {arr.dtype}"
        if path.endswith(".npz"):
            with np.load(path) as z:
                return ", ".join(f"{k} {z[k].shape}" for k in sorted(z.files))
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return f"{len(df)} rows: {', '.join(df.columns)}"
        if path.endswith(".json"):
            with open(path) as f:
                return "keys: " + ", ".join(json.load(f))
    except Exception as exc:
        return f"unreadable ({exc})"
    return ""


def listing(run_dir: str) -> list:
    lines = [f"{run_dir}/"]
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        lines.append(f"  {name:<22} {os.path.getsize(path):>13,} bytes   {_describe(path)}")
    return lines


def smoke(runs_root: str, args) -> int:
    if os.path.normpath(runs_root) == os.path.normpath(RUNS_ROOT):
        print(f"Refusing to smoke-test inside {RUNS_ROOT}; use another --runs-root.")
        return 2
    config, seed = REPRO_RUN
    run_dir = os.path.join(runs_root, run_name(config, seed))
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)                   # smoke output is disposable
    print(f"[smoke] {run_name(config, seed)} for {args.smoke_epochs} epochs → {run_dir}")
    rc = run_one(config, seed, runs_root, args, max_epochs=args.smoke_epochs)
    if rc != 0:
        print(f"[smoke] run_experiment.py failed with exit code {rc}")
        return rc

    checks = verify_artefacts(run_dir)
    lines = ["", "Artefact listing"] + listing(run_dir) + ["", "Checks"]
    for name, passed, detail in checks:
        detail = f"  ({detail[:160]})" if detail else ""
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}{detail}")
    passed_all = all(passed for _, passed, _ in checks)
    lines.append(f"\nSMOKE TEST {'PASSED' if passed_all else 'FAILED'}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(runs_root, "smoke_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return 0 if passed_all else 1


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Resumable multi-seed suite")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="Short b0_baseline seed 42 run into runs_smoke/, then verify and list")
    mode.add_argument("--repro-only", action="store_true",
                      help="Only b0_baseline seed 42, then the reproduction check")
    ap.add_argument("--configs", nargs="+", choices=CONFIG_ORDER,
                    help="Subset of configurations (run order stays fixed)")
    ap.add_argument("--seeds", nargs="+", type=int, help=f"Override SEEDS {SEEDS}")
    ap.add_argument("--runs-root", default=None,
                    help=f"Default {RUNS_ROOT} ({SMOKE_ROOT} with --smoke)")
    ap.add_argument("--resume-from", default=None,
                    help="runs/ directory from a previous session, or 'auto' to search /kaggle/input")
    ap.add_argument("--stop-after-hours", type=float, default=10.0,
                    help="Start no new run after this many hours (Kaggle sessions end at 12 h); 0 disables")
    ap.add_argument("--skip-repro-gate", action="store_true",
                    help="Start other runs even if seed 42 did not reproduce the original")
    ap.add_argument("--smoke-epochs", type=int, default=2)
    ap.add_argument("--img-dir",  help="Passed to run_experiment.py")
    ap.add_argument("--raw-dir",  help="Passed to run_experiment.py")
    ap.add_argument("--lab-dir",  help="Passed to run_experiment.py")
    ap.add_argument("--csv-path", help="Passed to run_experiment.py")
    ap.add_argument("--cudnn-deterministic", action="store_true", help="Passed to run_experiment.py")
    ap.add_argument("--seeded-generator",    action="store_true", help="Passed to run_experiment.py")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # keep order with the children's output

    runs_root = args.runs_root or (SMOKE_ROOT if args.smoke else RUNS_ROOT)
    os.makedirs(runs_root, exist_ok=True)

    if args.resume_from:
        source = find_resume_source() if args.resume_from == "auto" else args.resume_from
        if source:
            resume_from(source, runs_root)
        else:
            print("--resume-from: no attached runs/ directory with finished runs was found.")

    # A child process writes environment.json, so this process never initialises CUDA.
    subprocess.run([sys.executable, os.path.join(CODE_DIR, "environment_info.py")], check=True)

    if args.smoke:
        return smoke(runs_root, args)

    seeds   = args.seeds or SEEDS
    configs = [c for c in CONFIG_ORDER if not args.configs or c in args.configs]
    plan    = [REPRO_RUN] if args.repro_only else [(c, s) for c in configs for s in seeds]
    n_done  = sum(is_complete(os.path.join(runs_root, run_name(*p))) for p in plan)
    print(f"Suite: {len(plan)} run(s), {n_done} already complete  →  {runs_root}")

    t_suite = time.time()
    gate_passed = args.skip_repro_gate
    report = None
    for i, (config, seed) in enumerate(plan, 1):
        name    = run_name(config, seed)
        run_dir = os.path.join(runs_root, name)
        tag     = f"[{i}/{len(plan)}] {name}"

        if is_complete(run_dir):
            print(f"{tag}: already complete, skipped")
        else:
            if (config, seed) != REPRO_RUN and not gate_passed:
                if report is None:
                    report = check_reproduction(runs_root)
                    print_report(report)
                if not report["all_exact"]:
                    print(f"STOP before {name}: b0_baseline seed 42 has not reproduced the "
                          "original run, so nothing else is started. Report the check above. "
                          "--skip-repro-gate overrides once you have decided how to proceed.")
                    return 2
                gate_passed = True

            hours = (time.time() - t_suite) / 3600
            if args.stop_after_hours and hours >= args.stop_after_hours:
                print(f"{tag}: not started, {hours:.2f} h elapsed (--stop-after-hours "
                      f"{args.stop_after_hours}). Continue in a new session with --resume-from auto.")
                break

            print(f"{tag}: starting ({hours:.2f} h into this session)")
            t_run = time.time()
            rc = run_one(config, seed, runs_root, args)
            if rc != 0 or not is_complete(run_dir):
                print(f"{tag}: FAILED (exit code {rc}). Stopping; re-running the suite "
                      "restarts this run from scratch.")
                return rc or 1
            with open(os.path.join(run_dir, "metrics.json")) as f:
                m = json.load(f)
            print(f"{tag}: done in {(time.time() - t_run) / 60:.1f} min, selected epoch "
                  f"{m['training']['selected_epoch']}/{m['training']['epochs_run']}, "
                  f"test acc {m['test']['accuracy']:.4f}, "
                  f"QWK {m['test']['kappa_quadratic_weighted']:.4f}")

        if (config, seed) == REPRO_RUN:
            report = check_reproduction(runs_root)
            print_report(report)
            write_json(report_path(runs_root), report)

    n_done = sum(is_complete(os.path.join(runs_root, run_name(*p))) for p in plan)
    print(f"Suite status: {n_done}/{len(plan)} run(s) complete in {runs_root}")
    if args.repro_only:
        return 0 if report and report["all_exact"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
