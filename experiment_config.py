"""
Constants shared by the multi-seed revision experiments
(run_suite.py, run_experiment.py, repro_check.py, analyze.py, benchmark.py).

Plain Python with no heavy imports, so analyze.py stays standalone.
"""

import os
import json

# ── Locations ──────────────────────────────────────────────────────────────────
# /kaggle/working on Kaggle. Set the DR_WORK_DIR environment variable to run elsewhere.
WORK_DIR         = os.environ.get("DR_WORK_DIR", "/kaggle/working")
RUNS_ROOT        = os.path.join(WORK_DIR, "runs")
SMOKE_ROOT       = os.path.join(WORK_DIR, "runs_smoke")
ANALYSIS_DIR     = os.path.join(WORK_DIR, "analysis")
ENVIRONMENT_JSON = os.path.join(WORK_DIR, "environment.json")
PROBE_JSON       = os.path.join(WORK_DIR, "deterministic_algorithms_probe.json")

# ── Split seed vs training seed ────────────────────────────────────────────────
# The 70/15/15 stratified split is generated with SPLIT_SEED in EVERY run, so the test
# set is identical everywhere. Only train_seed varies (weight init, shuffling,
# augmentation sampling); the list of training seeds is SEEDS in run_suite.py.
SPLIT_SEED  = 42
N_TEST      = 550
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ── Configurations, in run order ───────────────────────────────────────────────
# num_workers is part of the augmentation RNG path (every DataLoader worker has its own
# seed), so each configuration keeps the value its original run used: 2 in train.py,
# 4 in train_effnet.py.
# b0_clahe_matched (added for the revision) feeds b0_clahe's LAB-CLAHE images through
# b0_baseline's data pipeline (dataset.py transforms, 2 workers), so preprocessing is the
# only difference from b0_baseline.
CONFIG_ORDER = ["b0_baseline", "b0_clahe", "b0_clahe_matched", "resnet50", "vgg16", "inceptionv3"]
CONFIGS = {
    "b0_baseline": {"model_name": "efficientnet_b0", "preprocessing": "baseline",
                    "num_workers": 2, "original_script": "train.py"},
    "b0_clahe":    {"model_name": "efficientnet_b0", "preprocessing": "lab_clahe",
                    "num_workers": 4, "original_script": "train_effnet.py"},
    "b0_clahe_matched": {"model_name": "efficientnet_b0", "preprocessing": "lab_clahe_cache",
                         "num_workers": 2, "original_script": None},
    "resnet50":    {"model_name": "resnet50",        "preprocessing": "baseline",
                    "num_workers": 2, "original_script": "train.py"},
    "vgg16":       {"model_name": "vgg16",           "preprocessing": "baseline",
                    "num_workers": 2, "original_script": "train.py"},
    "inceptionv3": {"model_name": "inceptionv3",     "preprocessing": "baseline",
                    "num_workers": 2, "original_script": "train.py"},
}
PREPROCESSING = {
    "baseline":  "preprocess.py: black-border crop -> resize 224x224 (INTER_AREA) -> CLAHE "
                 "(clip 2.0, 8x8 tiles) on each BGR channel; cached PNGs read by dataset.py; "
                 "train augmentation from dataset.get_transforms",
    "lab_clahe": "preprocess_effnet.py: black-border crop -> CLAHE (clip 2.0, 8x8 tiles) on "
                 "the LAB L-channel -> resize 224x224 (INTER_AREA), applied on the fly to raw "
                 "images; train augmentation from preprocess_effnet.get_transforms",
    "lab_clahe_cache": "preprocess_effnet.preprocess_image (black-border crop -> CLAHE (clip 2.0, "
                       "8x8 tiles) on the LAB L-channel -> resize 224x224 (INTER_AREA)), written "
                       "once to PNG by `python preprocess_effnet.py --skip-visualize` and read by "
                       "dataset.py; train augmentation from dataset.get_transforms, as b0_baseline",
}

# The run whose result must match the original before anything else starts.
REPRO_RUN = ("b0_baseline", 42)

# ── XAI samples ────────────────────────────────────────────────────────────────
# Positional indices into the SPLIT_SEED test split (test_df.iloc[idx]), one per grade,
# used identically for every configuration and run. They are the indices in the original
# baseline notebook's xai_sample_indices.json; that notebook picked them among images its
# EfficientNet-B0 classified correctly, which also guarantees each index has its grade.
XAI_SAMPLE_INDICES = {0: 263, 1: 269, 2: 285, 3: 352, 4: 311}

# ── Run directory layout ───────────────────────────────────────────────────────
RUN_FILES = ["probs.npy", "logits.npy", "labels.npy", "indices.npy", "preds.npy",
             "val_predictions.npz", "best_model.pth", "epoch_log.csv",
             "config.json", "metrics.json"]
EPOCH_LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                     "val_kappa", "val_qwk", "learning_rate", "wall_clock_seconds",
                     "is_selected_epoch"]

# ── Original single-seed results (from the published runs) ────────────────────
# Confusion matrices: rows = true grade, columns = predicted grade.
ORIGINAL_RESULTS = {
    "b0_baseline": {
        "metrics": {
            "accuracy":        0.7509090909090909,
            "cohen_kappa":     0.633963877311104,
            "mcc":             0.6388268576698753,
            "precision_macro": 0.5919383474782948,
            "recall_macro":    0.6300522427481328,
            "f1_macro":        0.6001622560049165,
            "auc_macro_ovr":   0.9208481694618698,
        },
        "confusion_matrix": [[260, 11, 0, 0, 0], [7, 35, 14, 0, 0], [1, 30, 79, 19, 21],
                             [0, 3, 4, 13, 9], [0, 6, 8, 4, 26]],
        "selected_epoch": 4,
        "xai_sample_indices": {0: 263, 1: 269, 2: 285, 3: 352, 4: 311},
    },
    "b0_clahe": {
        "confusion_matrix": [[260, 9, 2, 0, 0], [6, 34, 16, 0, 0], [1, 29, 96, 12, 12],
                             [0, 3, 7, 13, 6], [0, 7, 13, 5, 19]],
        "selected_epoch": 4,
        "xai_sample_indices": {0: 260, 1: 271, 2: 251, 3: 352, 4: 262},
    },
    "resnet50": {
        "confusion_matrix": [[261, 8, 2, 0, 0], [9, 36, 11, 0, 0], [4, 23, 88, 21, 14],
                             [0, 2, 5, 15, 7], [0, 8, 6, 7, 23]],
        "selected_epoch": 20,
    },
    "vgg16": {
        "confusion_matrix": [[261, 9, 0, 0, 1], [3, 34, 17, 1, 1], [0, 23, 69, 49, 9],
                             [0, 1, 5, 17, 6], [0, 4, 4, 14, 22]],
        "selected_epoch": 6,
    },
    "inceptionv3": {
        "confusion_matrix": [[259, 10, 2, 0, 0], [9, 28, 17, 1, 1], [5, 20, 97, 11, 17],
                             [0, 1, 8, 11, 9], [0, 7, 5, 7, 25]],
        "selected_epoch": 3,
    },
}


def run_name(config_name: str, train_seed: int) -> str:
    return f"{config_name}__seed{train_seed}"


def parse_run_name(name: str):
    config_name, seed = name.rsplit("__seed", 1)
    return config_name, int(seed)


def write_json(path: str, obj) -> None:
    """Atomic JSON write; numpy scalars/arrays are converted via .tolist()."""
    def default(o):
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Not JSON serialisable: {type(o).__name__}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=default)
    os.replace(tmp, path)
