"""
Post-hoc analysis of saved run artefacts. Never trains and never writes into runs/.

    python analyze.py
    python analyze.py --runs-root /kaggle/working/runs --out /kaggle/working/analysis

Reads every {runs_root}/{config}__seed{seed}/ containing metrics.json and recomputes all
metrics from the saved arrays. Writes to --out:

  summary_all.csv              one row per (config, seed), one column per metric
  summary_aggregated.csv       one row per config: {metric}_mean, _sd (ddof=1), _min, _max
  summary_aggregated_long.csv  one row per (config, metric), with a "mean ± SD" column
  per_run/<run>/confusion_matrix.csv, .npy    rows = true grade, columns = predicted grade
  per_class_all.csv            per-class precision, recall, F1, support, one-vs-rest AUC
  error_distance_all.csv       predictions off by 0..4 grades; mean absolute grade error
  row_squared_cost_all.csv     per true grade i: sum_j n_ij (i-j)^2, the disagreement that
                               quadratic kappa penalises, and its chance-expected value
  bootstrap_ci_per_run.csv     95% percentile intervals from resampling the test images
  bootstrap_ci_by_config.csv   those intervals averaged over seeds
                               (both: TEST-SET SAMPLING variation for a fixed trained model,
                                a different source of variation from training seeds)
  paired_<a>_vs_<b>.csv, paired_<a>_vs_<b>_summary.csv
  qwk_ranking_per_seed.csv, qwk_rank_matrix.csv, qwk_pairwise_order.csv,
  qwk_ranking_stability.json
  thresholds/<run>__grade4_curve.csv, thresholds/grade4_operating_points.csv
  findings_numbers.md          the numbers FINDINGS.md needs, without interpretation
  plots/                       plain inspection plots (skipped if matplotlib is missing)

Variation across training seeds is summarised as mean, SD and observed range only; there
are no significance tests. Requires numpy, pandas and scikit-learn.
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, cohen_kappa_score, matthews_corrcoef,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

from experiment_config import (RUNS_ROOT, ANALYSIS_DIR, CONFIG_ORDER, NUM_CLASSES,
                               CLASS_NAMES, parse_run_name, write_json)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED      = 12345
BOOTSTRAP_CHUNK     = 500
PAIRS               = [("b0_baseline", "b0_clahe"), ("b0_baseline", "resnet50")]
THRESHOLD_CONFIGS   = ["b0_baseline", "b0_clahe"]
TARGET_GRADE        = 4
RECALL_TARGETS      = [0.70, 0.80, 0.90]

HEADLINE_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc",
                    "kappa_unweighted", "kappa_quadratic_weighted", "auc_macro_ovr",
                    "mean_abs_grade_error"]
PER_CLASS_STATS  = ["precision", "recall", "f1", "auc_ovr"]
ID_COLUMNS       = ["config", "seed", "run"]
TRAINING_COLUMNS = ["selected_epoch", "epochs_run"]
TEST_RESAMPLING  = "test-set resampling (bootstrap over test images; trained model fixed)"


# ── Per-run metrics ────────────────────────────────────────────────────────────

def legacy_metrics(labels, preds, probs) -> dict:
    """Verbatim copy of evaluate.compute_metrics: the formulas behind the original results."""
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    return {
        "accuracy":         accuracy_score(labels, preds),
        "precision_macro":  precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":     recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro":         f1_score(labels, preds, average="macro", zero_division=0),
        "auc_macro_ovr":    roc_auc_score(labels_bin, probs,
                                          multi_class="ovr", average="macro"),
        "cohen_kappa":      cohen_kappa_score(labels, preds),
        "mcc":              matthews_corrcoef(labels, preds),
    }


def row_squared_cost(cm) -> dict:
    """Per true-grade row: observed and chance-expected sum_j n_ij (i-j)^2."""
    cm = np.asarray(cm, dtype=float)
    idx = np.arange(cm.shape[0])
    weights = (idx[:, None] - idx[None, :]) ** 2
    row_n = cm.sum(axis=1)
    expected = np.outer(row_n, cm.sum(axis=0)) / cm.sum()
    observed_rows = (weights * cm).sum(axis=1)
    expected_rows = (weights * expected).sum(axis=1)
    total = observed_rows.sum()
    return {
        "n":                                   row_n.astype(int).tolist(),
        "observed_sum_sq_distance":            observed_rows.tolist(),
        "observed_mean_sq_distance_per_image": (observed_rows / np.maximum(row_n, 1)).tolist(),
        "observed_share_of_total":             (observed_rows / total if total else
                                                np.zeros_like(observed_rows)).tolist(),
        "chance_expected_sum_sq_distance":     expected_rows.tolist(),
        "qwk_from_row_sums":                   float(1 - total / expected_rows.sum()),
    }


def compute_run_metrics(labels, preds, probs) -> dict:
    labels, preds, probs = np.asarray(labels), np.asarray(preds), np.asarray(probs)
    classes    = list(range(NUM_CLASSES))
    labels_bin = label_binarize(labels, classes=classes)
    cm         = confusion_matrix(labels, preds, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=classes, zero_division=0)
    has_both = [0 < labels_bin[:, g].sum() < len(labels) for g in classes]
    auc_per_class = [float(roc_auc_score(labels_bin[:, g], probs[:, g])) if ok else float("nan")
                     for g, ok in enumerate(has_both)]
    distance = np.abs(preds - labels)
    return {
        "accuracy":        float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mcc":             float(matthews_corrcoef(labels, preds)),
        "kappa_unweighted":         float(cohen_kappa_score(labels, preds, labels=classes)),
        "kappa_quadratic_weighted": float(cohen_kappa_score(labels, preds, labels=classes,
                                                            weights="quadratic")),
        "auc_macro_ovr":   (float(roc_auc_score(labels_bin, probs, multi_class="ovr",
                                                average="macro"))
                            if all(has_both) else float("nan")),
        "mean_abs_grade_error": float(distance.mean()),
        "per_class": {
            "precision": precision.tolist(),
            "recall":    recall.tolist(),
            "f1":        f1.tolist(),
            "support":   support.tolist(),
            "auc_ovr":   auc_per_class,
        },
        "confusion_matrix":      cm.tolist(),
        "error_distance_counts": np.bincount(distance, minlength=NUM_CLASSES).tolist(),
        "row_squared_cost":      row_squared_cost(cm),
    }


def flatten_metrics(m: dict) -> dict:
    row = {k: m[k] for k in HEADLINE_METRICS}
    for stat in PER_CLASS_STATS:
        for grade, value in enumerate(m["per_class"][stat]):
            row[f"{stat}_grade{grade}"] = value
    for d, count in enumerate(m["error_distance_counts"]):
        row[f"off_by_{d}"] = count
    return row


# ── Vectorised metrics for the bootstrap ───────────────────────────────────────

def _metrics_from_cms(cms: np.ndarray) -> dict:
    """cms (B, K, K) float counts, rows = true grade. Same formulas as scikit-learn."""
    k = cms.shape[1]
    n      = cms.sum(axis=(1, 2))
    diag   = np.diagonal(cms, axis1=1, axis2=2)
    true_n = cms.sum(axis=2)
    pred_n = cms.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_n > 0, diag / pred_n, 0.0)
        recall    = np.where(true_n > 0, diag / true_n, 0.0)
        f1        = np.where(true_n + pred_n > 0, 2 * diag / (true_n + pred_n), 0.0)
        present   = (true_n + pred_n) > 0
        n_present = present.sum(axis=1)
        out = {
            "accuracy":        diag.sum(axis=1) / n,
            "precision_macro": (precision * present).sum(axis=1) / n_present,
            "recall_macro":    (recall * present).sum(axis=1) / n_present,
            "f1_macro":        (f1 * present).sum(axis=1) / n_present,
        }
        expected = true_n[:, :, None] * pred_n[:, None, :] / n[:, None, None]
        idx = np.arange(k)
        for name, w in (("kappa_unweighted", 1.0 - np.eye(k)),
                        ("kappa_quadratic_weighted", (idx[:, None] - idx[None, :]) ** 2.0)):
            out[name] = 1 - (w * cms).sum(axis=(1, 2)) / (w * expected).sum(axis=(1, 2))
        cov_ytyp = diag.sum(axis=1) * n - (pred_n * true_n).sum(axis=1)
        cov_ypyp = n ** 2 - (pred_n ** 2).sum(axis=1)
        cov_ytyt = n ** 2 - (true_n ** 2).sum(axis=1)
        denom = np.sqrt(cov_ytyt * cov_ypyp)
        out["mcc"] = np.where(denom > 0, cov_ytyp / np.where(denom > 0, denom, 1), 0.0)
    for g in range(k):
        out[f"precision_grade{g}"] = precision[:, g]
        out[f"recall_grade{g}"]    = recall[:, g]
        out[f"f1_grade{g}"]        = f1[:, g]
    return out


def _weighted_auc(scores, is_pos, weights) -> np.ndarray:
    """One-vs-rest AUC per row of weights (B, N); tied scores count one half, as in sklearn."""
    order  = np.argsort(scores, kind="mergesort")
    s, pos = scores[order], is_pos[order]
    w      = weights[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    w_pos  = np.add.reduceat(w * pos, starts, axis=1)
    w_neg  = np.add.reduceat(w * ~pos, starts, axis=1)
    below  = np.cumsum(w_neg, axis=1) - w_neg
    num    = (w_pos * (below + 0.5 * w_neg)).sum(axis=1)
    den    = w_pos.sum(axis=1) * w_neg.sum(axis=1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _resample_metrics(labels, preds, probs, counts) -> dict:
    """Metrics for each row of counts (B, N) = how often each test image is drawn."""
    k = NUM_CLASSES
    onehot = np.zeros((len(labels), k * k))
    onehot[np.arange(len(labels)), labels * k + preds] = 1.0
    out = _metrics_from_cms((counts @ onehot).reshape(-1, k, k))
    aucs = np.stack([_weighted_auc(probs[:, g], labels == g, counts) for g in range(k)], axis=1)
    for g in range(k):
        out[f"auc_ovr_grade{g}"] = aucs[:, g]
    out["auc_macro_ovr"] = aucs.mean(axis=1)
    out["mean_abs_grade_error"] = counts @ np.abs(preds - labels) / counts.sum(axis=1)
    return out


def check_vectorised_metrics(run: dict) -> None:
    """On the unresampled test set the vectorised formulas must equal the sklearn values."""
    ones = np.ones((1, len(run["labels"])))
    vec = _resample_metrics(run["labels"], run["preds"], run["probs"], ones)
    bad = {k: (float(v[0]), run["flat"][k]) for k, v in vec.items()
           if not np.isclose(v[0], run["flat"][k], rtol=0, atol=1e-9, equal_nan=True)}
    if bad:
        raise AssertionError(f"{run['name']}: vectorised metrics differ from scikit-learn: {bad}")


def bootstrap_counts(n: int, n_boot: int, seed: int = BOOTSTRAP_SEED):
    """Yield (chunk, n) arrays: times each of the n test images is drawn per resample."""
    for chunk_index, start in enumerate(range(0, n_boot, BOOTSTRAP_CHUNK)):
        size  = min(BOOTSTRAP_CHUNK, n_boot - start)
        draws = np.random.default_rng([seed, chunk_index]).integers(0, n, size=(size, n))
        flat  = (draws + np.arange(size)[:, None] * n).ravel()
        yield np.bincount(flat, minlength=size * n).reshape(size, n).astype(float)


def bootstrap_tables(runs: list, n_boot: int):
    rows = []
    for run in runs:
        chunks = [_resample_metrics(run["labels"], run["preds"], run["probs"], counts)
                  for counts in bootstrap_counts(len(run["labels"]), n_boot)]
        for metric in chunks[0]:
            values = np.concatenate([c[metric] for c in chunks])
            finite = values[np.isfinite(values)]
            low, high = np.percentile(finite, [2.5, 97.5]) if finite.size else (np.nan, np.nan)
            rows.append({"config": run["config"], "seed": run["seed"], "run": run["name"],
                         "metric": metric, "variation_source": TEST_RESAMPLING,
                         "point_estimate": run["flat"][metric],
                         "ci95_low": low, "ci95_high": high, "ci95_width": high - low,
                         "n_resamples": len(values), "n_undefined": len(values) - finite.size,
                         "bootstrap_seed": BOOTSTRAP_SEED})
    per_run = pd.DataFrame(rows)
    by_config = (per_run.groupby(["config", "metric"], sort=False)
                 .agg(n_seeds=("seed", "nunique"),
                      point_estimate_mean=("point_estimate", "mean"),
                      ci95_low_mean=("ci95_low", "mean"),
                      ci95_high_mean=("ci95_high", "mean"),
                      ci95_width_mean=("ci95_width", "mean"),
                      ci95_width_min=("ci95_width", "min"),
                      ci95_width_max=("ci95_width", "max"))
                 .reset_index())
    by_config.insert(2, "variation_source",
                     TEST_RESAMPLING + "; per-run intervals averaged over seeds")
    return per_run, by_config


# ── Loading ────────────────────────────────────────────────────────────────────

def _config_rank(config):
    return CONFIG_ORDER.index(config) if config in CONFIG_ORDER else len(CONFIG_ORDER)


def ordered_configs(names) -> list:
    return sorted(dict.fromkeys(names), key=lambda c: (_config_rank(c), c))


def load_runs(runs_root: str) -> list:
    runs = []
    for run_dir in glob.glob(os.path.join(runs_root, "*__seed*")):
        if not os.path.isfile(os.path.join(run_dir, "metrics.json")):
            continue
        name = os.path.basename(run_dir)
        config, seed = parse_run_name(name)
        run = {"name": name, "config": config, "seed": seed, "dir": run_dir}
        for key in ("probs", "logits", "labels", "indices", "preds"):
            run[key] = np.load(os.path.join(run_dir, f"{key}.npy"))
        val_path = os.path.join(run_dir, "val_predictions.npz")
        run["val"] = None
        if os.path.exists(val_path):
            with np.load(val_path) as z:
                run["val"] = {k: z[k] for k in z.files}
        for key, fname in (("metrics_json", "metrics.json"), ("config_json", "config.json")):
            path = os.path.join(run_dir, fname)
            run[key] = {}
            if os.path.exists(path):
                with open(path) as f:
                    run[key] = json.load(f)
        runs.append(run)
    runs.sort(key=lambda r: (_config_rank(r["config"]), r["config"], r["seed"]))
    return runs


def check_alignment(runs: list) -> None:
    ref = runs[0]
    for run in runs[1:]:
        if not (np.array_equal(run["indices"], ref["indices"])
                and np.array_equal(run["labels"], ref["labels"])):
            raise ValueError(f"{run['name']} does not share the test-set rows of {ref['name']}")


def consistency_warnings(runs: list) -> list:
    warnings = []
    smoke = [r["name"] for r in runs if r["config_json"].get("smoke_run")]
    if smoke:
        warnings.append(f"smoke runs included: {smoke}")
    for key in ("cudnn_deterministic", "dataloader_generator"):
        values = {str(r["config_json"].get("determinism", {}).get(key)) for r in runs}
        if len(values) > 1:
            warnings.append(f"runs differ in determinism setting '{key}': {sorted(values)}")
    gpus = {tuple(g.get("name") for g in r["config_json"].get("environment", {})
                  .get("hardware", {}).get("gpus", [])) for r in runs}
    if len(gpus) > 1:
        warnings.append(f"runs used different GPUs: {sorted(gpus)}")
    versions = {json.dumps(r["config_json"].get("environment", {}).get("versions", {}),
                           sort_keys=True) for r in runs}
    if len(versions) > 1:
        warnings.append("runs used different library versions (see each config.json environment)")
    return warnings


# ── Across seeds ───────────────────────────────────────────────────────────────

def aggregate(summary_all: pd.DataFrame):
    metric_cols = [c for c in summary_all.columns if c not in ID_COLUMNS]
    wide, long = [], []
    for config in ordered_configs(summary_all["config"]):
        group = summary_all[summary_all["config"] == config]
        row = {"config": config, "n_seeds": len(group),
               "seeds": " ".join(str(s) for s in group["seed"])}
        for metric in metric_cols:
            values = group[metric].astype(float)
            mean, sd = values.mean(), values.std(ddof=1)
            low, high = values.min(), values.max()
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                        f"{metric}_min": low, f"{metric}_max": high})
            long.append({"config": config, "metric": metric, "n_seeds": len(values),
                         "mean": mean, "sd": sd, "min": low, "max": high, "range": high - low,
                         "mean_pm_sd": f"{mean:.4f} ± {sd:.4f}"})
        wide.append(row)
    return pd.DataFrame(wide), pd.DataFrame(long)


def paired_comparison(summary_all: pd.DataFrame, a: str, b: str):
    metric_cols = [c for c in summary_all.columns if c not in ID_COLUMNS + TRAINING_COLUMNS]
    A = summary_all[summary_all["config"] == a].set_index("seed")
    B = summary_all[summary_all["config"] == b].set_index("seed")
    seeds = [s for s in A.index if s in B.index]
    if not seeds:
        return None, None
    long, summary = [], []
    for metric in metric_cols:
        av = A.loc[seeds, metric].astype(float).to_numpy()
        bv = B.loc[seeds, metric].astype(float).to_numpy()
        delta = bv - av
        for seed, x, y, d in zip(seeds, av, bv, delta):
            long.append({"metric": metric, "seed": seed, a: x, b: y, f"delta_{b}_minus_{a}": d})
        multi = len(seeds) > 1
        a_sd = av.std(ddof=1) if multi else np.nan
        b_sd = bv.std(ddof=1) if multi else np.nan
        overlap = not (av.max() < bv.min() or bv.max() < av.min())
        summary.append({
            "metric": metric, "n_paired_seeds": len(seeds),
            f"{a}_mean": av.mean(), f"{a}_sd": a_sd, f"{a}_min": av.min(), f"{a}_max": av.max(),
            f"{b}_mean": bv.mean(), f"{b}_sd": b_sd, f"{b}_min": bv.min(), f"{b}_max": bv.max(),
            "delta_mean": delta.mean(), "delta_sd": delta.std(ddof=1) if multi else np.nan,
            "delta_min": delta.min(), "delta_max": delta.max(),
            "n_delta_positive": int((delta > 0).sum()),
            "n_delta_negative": int((delta < 0).sum()),
            "n_delta_zero":     int((delta == 0).sum()),
            "all_deltas_same_sign": bool((delta > 0).all() or (delta < 0).all()),
            "seed_ranges_overlap":  overlap,
            "abs_delta_mean_gt_larger_seed_sd": bool(abs(delta.mean()) > max(a_sd, b_sd)) if multi else None,
            "exceeds_seed_variance": (not overlap) if multi else None,
        })
    return pd.DataFrame(long), pd.DataFrame(summary)


def qwk_ranking(summary_all: pd.DataFrame):
    pivot = summary_all.pivot(index="seed", columns="config", values="kappa_quadratic_weighted")
    configs = ordered_configs(pivot.columns)
    pivot = pivot[configs]
    complete = pivot.dropna()

    rows, orderings = [], {}
    for seed, values in complete.iterrows():
        ordered = values.sort_values(ascending=False, kind="mergesort")
        ranks = values.rank(ascending=False, method="min")
        parts, previous = [], None
        for i, (config, qwk) in enumerate(ordered.items()):
            next_qwk = ordered.iloc[i + 1] if i + 1 < len(ordered) else np.nan
            rows.append({"seed": seed, "rank": int(ranks[config]), "config": config,
                         "kappa_quadratic_weighted": qwk, "gap_to_next_lower": qwk - next_qwk})
            if previous is not None:
                parts.append("=" if qwk == previous else ">")
            parts.append(config)
            previous = qwk
        orderings[int(seed)] = " ".join(parts)

    rank_matrix = complete.rank(axis=1, ascending=False, method="min")
    pairwise = pd.DataFrame(np.nan, index=configs, columns=configs)
    for ci in configs:
        for cj in configs:
            if ci != cj and len(complete):
                pairwise.loc[ci, cj] = (complete[ci] > complete[cj]).mean()
    pairwise.index.name = "fraction_of_seeds_row_qwk_gt_column_qwk"

    distinct = sorted(set(orderings.values()))
    tops = {int(s): row.idxmax() for s, row in complete.iterrows()}
    stability = {
        "metric": "kappa_quadratic_weighted on the test set",
        "configs_ranked": configs,
        "configs_not_yet_available": [c for c in CONFIG_ORDER if c not in configs],
        "seeds_ranked": [int(s) for s in complete.index],
        "seeds_excluded_missing_configs": [int(s) for s in pivot.index if s not in complete.index],
        "ordering_per_seed": orderings,
        "identical_ordering_across_seeds": (len(distinct) == 1) if orderings else None,
        "n_distinct_orderings": len(distinct),
        "top_config_per_seed": tops,
        "top_config_identical_across_seeds": (len(set(tops.values())) == 1) if tops else None,
        "rank_range_per_config": {c: [int(rank_matrix[c].min()), int(rank_matrix[c].max())]
                                  for c in configs if len(rank_matrix)},
        "ties_present": any("=" in o for o in orderings.values()),
    }
    return pd.DataFrame(rows), rank_matrix, pairwise, stability


# ── Grade-4 decision threshold ─────────────────────────────────────────────────

def grade4_margin(logits):
    """margin = logit_4 − max other logit; fallback = argmax over the other grades."""
    others = [g for g in range(NUM_CLASSES) if g != TARGET_GRADE]
    other_logits = logits[:, others].astype(np.float64)
    margin = logits[:, TARGET_GRADE].astype(np.float64) - other_logits.max(axis=1)
    return margin, np.asarray(others)[other_logits.argmax(axis=1)]


def decide(margin, fallback, threshold, strict=False):
    call = margin > threshold if strict else margin >= threshold
    return np.where(call, TARGET_GRADE, fallback)


def operating_metrics(labels, preds, argmax_preds) -> dict:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    tp, positives = cm[TARGET_GRADE, TARGET_GRADE], cm[TARGET_GRADE].sum()
    called, negatives = cm[:, TARGET_GRADE].sum(), len(labels) - positives
    no_dr = labels == 0
    accuracy = float((preds == labels).mean())
    no_dr_spec = float((preds[no_dr] == 0).mean())
    return {
        "n_predicted_grade4":                 int(called),
        "recall_grade4":                      tp / positives,
        "precision_grade4":                   tp / called if called else np.nan,
        "specificity_grade4_one_vs_rest":     (negatives - (called - tp)) / negatives,
        "accuracy":                           accuracy,
        "accuracy_change_vs_argmax":          accuracy - float((argmax_preds == labels).mean()),
        "no_dr_specificity":                  no_dr_spec,
        "no_dr_specificity_change_vs_argmax": no_dr_spec - float((argmax_preds[no_dr] == 0).mean()),
        "no_dr_called_grade4_rate":           float((preds[no_dr] == TARGET_GRADE).mean()),
        "kappa_quadratic_weighted":
            float(_metrics_from_cms(cm[None].astype(float))["kappa_quadratic_weighted"][0]),
    }


def threshold_for_recall(margin, labels, target):
    """Largest t such that 'predict grade 4 when margin >= t' reaches the target recall."""
    positive = np.sort(margin[labels == TARGET_GRADE])[::-1]
    needed = int(np.ceil(target * len(positive) - 1e-9))
    if len(positive) == 0 or needed < 1:
        return None
    return float(positive[needed - 1])


def threshold_analysis(run: dict):
    labels, logits = run["labels"], run["logits"]
    margin, fallback = grade4_margin(logits)
    argmax_preds = decide(margin, fallback, 0.0, strict=True)
    if not np.array_equal(argmax_preds, run["preds"]):
        raise ValueError(f"{run['name']}: argmax of logits.npy does not match preds.npy")

    curve = [{"rule": "argmax (default)", "threshold_margin": 0.0, "lambda_equivalent": 1.0,
              **operating_metrics(labels, argmax_preds, argmax_preds)}]
    for t in np.r_[np.inf, np.unique(margin)[::-1]]:
        curve.append({"rule": "grade 4 if margin >= t", "threshold_margin": t,
                      "lambda_equivalent": float(np.exp(-t)),
                      **operating_metrics(labels, decide(margin, fallback, t), argmax_preds)})

    ids = {"config": run["config"], "seed": run["seed"], "run": run["name"]}
    points = [{**ids, "operating_point": "argmax (default)", "threshold_selected_on": "-",
               "recall_grade4_on_selection_split": np.nan, "threshold_margin": 0.0,
               "lambda_equivalent": 1.0, **curve[0]}]
    points[0].pop("rule")
    selections = {"test": (margin, labels)}
    if run["val"] is not None:
        selections["val"] = (grade4_margin(run["val"]["logits"])[0], run["val"]["labels"])
    for split, (sel_margin, sel_labels) in selections.items():
        for target in RECALL_TARGETS:
            t = threshold_for_recall(sel_margin, sel_labels, target)
            row = {**ids, "operating_point": f"recall_grade4 >= {target:.2f}",
                   "threshold_selected_on": split}
            if t is None:
                points.append(row)
                continue
            achieved = float((sel_margin[sel_labels == TARGET_GRADE] >= t).mean())
            points.append({**row, "recall_grade4_on_selection_split": achieved,
                           "threshold_margin": t, "lambda_equivalent": float(np.exp(-t)),
                           **operating_metrics(labels, decide(margin, fallback, t), argmax_preds)})
    return pd.DataFrame(curve), points


# ── Reports ────────────────────────────────────────────────────────────────────

def _pm(mean, sd, digits=4):
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if np.isfinite(sd) else f"{mean:.{digits}f} (1 seed)"


def findings_markdown(args, runs, warnings, summary_all, paired, stability) -> str:
    from repro_check import check_reproduction   # imported here: repro_check imports analyze

    configs = ordered_configs(summary_all["config"])
    seeds = sorted(summary_all["seed"].unique())
    lines = ["# Numbers from analyze.py", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
             f"`{args.runs_root}`: {len(runs)} completed run(s). No interpretation.", "",
             "| configuration | " + " | ".join(f"seed {s}" for s in seeds) + " |",
             "|---|" + "---|" * len(seeds)]
    present = set(zip(summary_all["config"], summary_all["seed"]))
    for c in configs:
        lines.append(f"| {c} | " + " | ".join("done" if (c, s) in present else "—" for s in seeds) + " |")
    if warnings:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]

    lines += ["", "## Definitions", "",
              "- SD: sample standard deviation across training seeds (ddof = 1). "
              "Range: observed minimum to maximum across seeds.",
              "- kappa_unweighted: Cohen's kappa, no weights. kappa_quadratic_weighted (QWK): "
              "Cohen's kappa with quadratic weights.",
              "- Exceeds seed variance: the observed seed ranges of the two configurations do "
              "not overlap. Also shown: whether every paired per-seed delta has the same sign.",
              f"- Bootstrap intervals (bootstrap_ci_*.csv) resample the test images for a fixed "
              f"trained model ({args.n_boot} resamples, seed {BOOTSTRAP_SEED}). They describe "
              "test-set sampling variation and are reported separately from seed variation.",
              "- Grade-4 thresholds: no_dr_specificity = share of true grade-0 images predicted "
              "as grade 0."]

    lines += ["", "## 1. Seed-42 reproduction of b0_baseline", ""]
    report = check_reproduction(args.runs_root)
    if not report["available"]:
        lines.append("b0_baseline seed 42 is not available.")
    else:
        lines += ["| metric | original | seed-42 rerun | exact |", "|---|---|---|---|"]
        for r in report["metrics"]:
            lines.append(f"| {r['metric']} | {r['expected']!r} | {r['observed']!r} | "
                         f"{'yes' if r['exact'] else 'no'} |")
        for key in ("confusion_matrix", "selected_epoch", "xai_sample_indices"):
            item = report[key]
            lines.append(f"| {key} | {item['expected']} | {item['observed']} | "
                         f"{'yes' if item['exact'] else 'no'} |")
        lines += ["", f"All exact: {'yes' if report['all_exact'] else 'no'}"]

    lines += ["", "## 2. Kappa per configuration (test set)", "",
              "| configuration | seeds | unweighted kappa, mean ± SD | range | "
              "QWK, mean ± SD | range |", "|---|---|---|---|---|---|"]
    for c in configs:
        g = summary_all[summary_all["config"] == c]
        ku, kq = g["kappa_unweighted"], g["kappa_quadratic_weighted"]
        lines.append(f"| {c} | {len(g)} | {_pm(ku.mean(), ku.std(ddof=1))} | "
                     f"{ku.min():.4f}–{ku.max():.4f} | {_pm(kq.mean(), kq.std(ddof=1))} | "
                     f"{kq.min():.4f}–{kq.max():.4f} |")

    section = 3
    shown = HEADLINE_METRICS + [f"recall_grade{g}" for g in range(NUM_CLASSES)]
    for (a, b), (_, summary) in paired.items():
        lines += ["", f"## {section}. {a} vs {b} (paired by seed; delta = {b} − {a})", ""]
        section += 1
        if summary is None:
            lines.append("No seeds available for both configurations.")
            continue
        lines += [f"Paired seeds: {summary['n_paired_seeds'].iloc[0]}. All metrics are in "
                  f"paired_{a}_vs_{b}_summary.csv.", "",
                  f"| metric | {a} mean ± SD | {b} mean ± SD | delta mean ± SD | deltas +/− | "
                  "ranges overlap | exceeds seed variance |", "|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["metric"].isin(shown)].iterrows():
            lines.append(
                f"| {r['metric']} | {_pm(r[f'{a}_mean'], r[f'{a}_sd'])} | "
                f"{_pm(r[f'{b}_mean'], r[f'{b}_sd'])} | {_pm(r['delta_mean'], r['delta_sd'])} | "
                f"{r['n_delta_positive']}/{r['n_delta_negative']} | "
                f"{'yes' if r['seed_ranges_overlap'] else 'no'} | "
                f"{'n/a' if r['exceeds_seed_variance'] is None else ('yes' if r['exceeds_seed_variance'] else 'no')} |")

    lines += ["", f"## {section}. Per-seed QWK ranking of all configurations", ""]
    if stability["configs_not_yet_available"]:
        lines.append(f"Not yet available: {', '.join(stability['configs_not_yet_available'])}.")
    lines += ["| seed | ordering by test QWK (highest first) |", "|---|---|"]
    for seed, ordering in stability["ordering_per_seed"].items():
        lines.append(f"| {seed} | {ordering} |")
    lines += ["",
              f"- Seeds ranked: {stability['seeds_ranked']}; excluded (missing a configuration): "
              f"{stability['seeds_excluded_missing_configs']}",
              f"- Identical ordering across seeds: {stability['identical_ordering_across_seeds']} "
              f"({stability['n_distinct_orderings']} distinct ordering(s))",
              f"- Top configuration per seed: {stability['top_config_per_seed']}; identical: "
              f"{stability['top_config_identical_across_seeds']}",
              f"- Rank range per configuration: {stability['rank_range_per_config']}",
              f"- Ties present: {stability['ties_present']}"]
    return "\n".join(lines) + "\n"


def plain_plots(out_dir, summary_all, curves) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed; plots skipped")
        return
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    try:
        configs = ordered_configs(summary_all["config"])
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, config in enumerate(configs):
            group = summary_all[summary_all["config"] == config]
            ax.scatter([i] * len(group), group["kappa_quadratic_weighted"])
            for _, r in group.iterrows():
                ax.annotate(str(r["seed"]), (i, r["kappa_quadratic_weighted"]),
                            xytext=(4, 0), textcoords="offset points", fontsize=7)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs)
        ax.set_ylabel("test QWK")
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "qwk_by_seed.png"), dpi=100)
        plt.close(fig)

        for config in THRESHOLD_CONFIGS:
            selected = {name: c for name, c in curves.items() if name.startswith(config + "__")}
            if not selected:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for name, curve in selected.items():
                sweep = curve[curve["rule"] != "argmax (default)"]
                ax.plot(sweep["recall_grade4"], sweep["accuracy"], label=name.split("__")[1])
            ax.set_xlabel("recall, grade 4")
            ax.set_ylabel("overall accuracy")
            ax.set_title(config)
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"grade4_tradeoff_{config}.png"), dpi=100)
            plt.close(fig)
    except Exception as exc:     # plots are optional; never lose the data files over them
        print(f"  [WARN] plotting failed: {exc}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Post-hoc analysis of saved runs (no training)")
    ap.add_argument("--runs-root", default=RUNS_ROOT)
    ap.add_argument("--out",       default=ANALYSIS_DIR)
    ap.add_argument("--n-boot",    type=int, default=BOOTSTRAP_RESAMPLES)
    ap.add_argument("--no-plots",  action="store_true")
    args = ap.parse_args(argv)

    runs = load_runs(args.runs_root)
    if not runs:
        print(f"No completed runs under {args.runs_root}")
        return 1
    check_alignment(runs)
    warnings = consistency_warnings(runs)
    for w in warnings:
        print(f"[WARN] {w}")
    os.makedirs(args.out, exist_ok=True)
    print(f"Analysing {len(runs)} run(s) from {args.runs_root} → {args.out}")

    summary_rows, per_class_rows, distance_rows, cost_rows = [], [], [], []
    for run in runs:
        m = compute_run_metrics(run["labels"], run["preds"], run["probs"])
        run["flat"] = flatten_metrics(m)
        check_vectorised_metrics(run)
        ids = {"config": run["config"], "seed": run["seed"], "run": run["name"]}
        training = run["metrics_json"].get("training", {})
        summary_rows.append({**ids, **run["flat"],
                             **{c: training.get(c) for c in TRAINING_COLUMNS}})

        cm = np.asarray(m["confusion_matrix"])
        run_out = os.path.join(args.out, "per_run", run["name"])
        os.makedirs(run_out, exist_ok=True)
        np.save(os.path.join(run_out, "confusion_matrix.npy"), cm)
        pd.DataFrame(cm, index=[f"true_{g}" for g in range(NUM_CLASSES)],
                     columns=[f"pred_{g}" for g in range(NUM_CLASSES)]
                     ).to_csv(os.path.join(run_out, "confusion_matrix.csv"))

        for g in range(NUM_CLASSES):
            per_class_rows.append({**ids, "grade": g, "grade_name": CLASS_NAMES[g],
                                   **{s: m["per_class"][s][g]
                                      for s in ("precision", "recall", "f1", "support", "auc_ovr")}})
        counts = m["error_distance_counts"]
        distance_rows.append({**ids, **{f"off_by_{d}": c for d, c in enumerate(counts)},
                              "n": sum(counts), "mean_abs_grade_error": m["mean_abs_grade_error"]})
        rc = m["row_squared_cost"]
        for g in range(NUM_CLASSES):
            cost_rows.append({**ids, "true_grade": g, "n": rc["n"][g],
                              "observed_sum_sq_distance": rc["observed_sum_sq_distance"][g],
                              "observed_mean_sq_distance_per_image":
                                  rc["observed_mean_sq_distance_per_image"][g],
                              "observed_share_of_total": rc["observed_share_of_total"][g],
                              "chance_expected_sum_sq_distance":
                                  rc["chance_expected_sum_sq_distance"][g],
                              "run_kappa_quadratic_weighted": m["kappa_quadratic_weighted"]})

    def save(df, name):
        df.to_csv(os.path.join(args.out, name), index=False)

    summary_all = pd.DataFrame(summary_rows)
    save(summary_all, "summary_all.csv")
    save(pd.DataFrame(per_class_rows), "per_class_all.csv")
    save(pd.DataFrame(distance_rows), "error_distance_all.csv")
    save(pd.DataFrame(cost_rows), "row_squared_cost_all.csv")

    wide, long = aggregate(summary_all)
    save(wide, "summary_aggregated.csv")
    save(long, "summary_aggregated_long.csv")

    print(f"  bootstrap: {args.n_boot} resamples × {len(runs)} run(s) ...")
    boot_runs, boot_configs = bootstrap_tables(runs, args.n_boot)
    save(boot_runs, "bootstrap_ci_per_run.csv")
    save(boot_configs, "bootstrap_ci_by_config.csv")

    paired = {}
    for a, b in PAIRS:
        long_df, summary_df = paired_comparison(summary_all, a, b)
        paired[(a, b)] = (long_df, summary_df)
        if summary_df is not None:
            save(long_df, f"paired_{a}_vs_{b}.csv")
            save(summary_df, f"paired_{a}_vs_{b}_summary.csv")

    ranking, rank_matrix, pairwise, stability = qwk_ranking(summary_all)
    save(ranking, "qwk_ranking_per_seed.csv")
    rank_matrix.to_csv(os.path.join(args.out, "qwk_rank_matrix.csv"))
    pairwise.to_csv(os.path.join(args.out, "qwk_pairwise_order.csv"))
    write_json(os.path.join(args.out, "qwk_ranking_stability.json"), stability)

    threshold_dir = os.path.join(args.out, "thresholds")
    os.makedirs(threshold_dir, exist_ok=True)
    curves, points = {}, []
    for run in runs:
        if run["config"] in THRESHOLD_CONFIGS:
            curve, run_points = threshold_analysis(run)
            curve.to_csv(os.path.join(threshold_dir, f"{run['name']}__grade4_curve.csv"), index=False)
            curves[run["name"]] = curve
            points += run_points
    if points:
        pd.DataFrame(points).to_csv(os.path.join(threshold_dir, "grade4_operating_points.csv"),
                                    index=False)

    with open(os.path.join(args.out, "findings_numbers.md"), "w", encoding="utf-8") as f:
        f.write(findings_markdown(args, runs, warnings, summary_all, paired, stability))
    if not args.no_plots:
        plain_plots(args.out, summary_all, curves)

    print("\nKappa per configuration (test set, mean ± SD over seeds)")
    for c in ordered_configs(summary_all["config"]):
        g = summary_all[summary_all["config"] == c]
        print(f"  {c:<12} n={len(g)}  unweighted {_pm(g['kappa_unweighted'].mean(), g['kappa_unweighted'].std(ddof=1))}"
              f"  QWK {_pm(g['kappa_quadratic_weighted'].mean(), g['kappa_quadratic_weighted'].std(ddof=1))}")
    print(f"QWK ordering identical across ranked seeds {stability['seeds_ranked']}: "
          f"{stability['identical_ordering_across_seeds']}")
    print(f"Outputs → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
