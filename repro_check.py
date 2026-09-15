"""
Does b0_baseline at train_seed 42 reproduce the original published run exactly?

    python repro_check.py
    python repro_check.py --runs-root /kaggle/working/runs

The seven original metrics are recomputed from the saved arrays with the original
formulas and compared with == (no tolerance), together with the confusion matrix, the
selected epoch and the XAI indices the original selection rule picks from these
predictions. Seed-42 runs of the other configurations, where present, are compared on
confusion matrix and selected epoch for information only.

Exit code: 0 exact, 1 differs, 2 run not found.
"""

import os
import sys
import json
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix

from analyze import legacy_metrics
from experiment_config import (RUNS_ROOT, ORIGINAL_RESULTS, REPRO_RUN, NUM_CLASSES,
                               run_name, write_json)


def _load(run_dir):
    arrays = {k: np.load(os.path.join(run_dir, f"{k}.npy")) for k in ("labels", "preds", "probs")}
    with open(os.path.join(run_dir, "metrics.json")) as f:
        return arrays, json.load(f)


def _complete(run_dir):
    return os.path.isfile(os.path.join(run_dir, "metrics.json"))


def check_reproduction(runs_root: str = RUNS_ROOT) -> dict:
    config, seed = REPRO_RUN
    original = ORIGINAL_RESULTS[config]
    run_dir  = os.path.join(runs_root, run_name(config, seed))
    report   = {"run": run_name(config, seed), "run_dir": run_dir,
                "available": _complete(run_dir), "all_exact": False}
    if not report["available"]:
        return report

    arrays, saved = _load(run_dir)
    observed = legacy_metrics(arrays["labels"], arrays["preds"], arrays["probs"])
    rows = []
    for key, expected in original["metrics"].items():
        value = float(observed[key])
        rows.append({"metric": key, "expected": expected, "observed": value,
                     "exact": value == expected, "abs_diff": abs(value - expected)})

    cm = confusion_matrix(arrays["labels"], arrays["preds"],
                          labels=list(range(NUM_CLASSES))).tolist()
    xai_expected = {str(k): v for k, v in original["xai_sample_indices"].items()}
    xai_observed = saved.get("xai_samples_by_original_rule")
    epoch_observed = saved["training"]["selected_epoch"]

    report.update({
        "metrics": rows,
        "confusion_matrix": {"expected": original["confusion_matrix"], "observed": cm,
                             "exact": cm == original["confusion_matrix"]},
        "selected_epoch": {"expected": original["selected_epoch"], "observed": epoch_observed,
                           "exact": epoch_observed == original["selected_epoch"]},
        "xai_sample_indices": {"expected": xai_expected, "observed": xai_observed,
                               "exact": xai_observed == xai_expected},
    })
    report["all_exact"] = (all(r["exact"] for r in rows)
                           and report["confusion_matrix"]["exact"]
                           and report["selected_epoch"]["exact"]
                           and report["xai_sample_indices"]["exact"])

    others = {}
    for other, orig in ORIGINAL_RESULTS.items():
        other_dir = os.path.join(runs_root, run_name(other, seed))
        if other == config or not _complete(other_dir):
            continue
        arr, met = _load(other_dir)
        other_cm = confusion_matrix(arr["labels"], arr["preds"],
                                    labels=list(range(NUM_CLASSES))).tolist()
        others[other] = {
            "confusion_matrix": {"expected": orig["confusion_matrix"], "observed": other_cm,
                                 "exact": other_cm == orig["confusion_matrix"]},
            "selected_epoch":   {"expected": orig["selected_epoch"],
                                 "observed": met["training"]["selected_epoch"]},
        }
    report["other_configs_seed42_informational"] = others
    return report


def report_path(runs_root: str) -> str:
    root = os.path.normpath(runs_root)
    if root == os.path.normpath(RUNS_ROOT):
        return os.path.join(os.path.dirname(root), "repro_check.json")
    return os.path.join(root, "repro_check.json")


def print_report(report: dict) -> None:
    print("=" * 78)
    print(f"Reproduction check: {report['run']} vs the original run")
    print("=" * 78)
    if not report["available"]:
        print(f"  Not found: {report['run_dir']}/metrics.json — run it first "
              "(python run_suite.py --repro-only).")
        return
    print(f"  {'metric':<16} {'expected':<22} {'observed':<22} {'exact':<6} |diff|")
    for r in report["metrics"]:
        print(f"  {r['metric']:<16} {r['expected']!r:<22} {r['observed']!r:<22} "
              f"{'yes' if r['exact'] else 'NO':<6} {r['abs_diff']:.3g}")
    cm = report["confusion_matrix"]
    print(f"  confusion matrix exact: {'yes' if cm['exact'] else 'NO'}")
    if not cm["exact"]:
        print(f"    expected {cm['expected']}\n    observed {cm['observed']}")
    ep = report["selected_epoch"]
    print(f"  selected epoch: expected {ep['expected']}, observed {ep['observed']}")
    xai = report["xai_sample_indices"]
    print(f"  XAI indices (original selection rule) exact: {'yes' if xai['exact'] else 'NO'}"
          f"  expected {xai['expected']}  observed {xai['observed']}")
    for other, info in report.get("other_configs_seed42_informational", {}).items():
        c = info["confusion_matrix"]
        print(f"  [info] {other} seed 42 confusion matrix exact: {'yes' if c['exact'] else 'no'}"
              f"  (selected epoch {info['selected_epoch']['observed']}, "
              f"original {info['selected_epoch']['expected']})")
    print("-" * 78)
    if report["all_exact"]:
        print("  RESULT: exact reproduction of the original b0_baseline run.")
    else:
        print("  RESULT: NOT reproduced. Stop here and report this output before "
              "running anything else.")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare b0_baseline seed 42 with the original run")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    args = ap.parse_args()

    report = check_reproduction(args.runs_root)
    print_report(report)
    if not report["available"]:
        return 2
    write_json(report_path(args.runs_root), report)
    print(f"Report → {report_path(args.runs_root)}")
    return 0 if report["all_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
