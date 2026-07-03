"""
Central dataset-path resolver.

Kaggle mounts the APTOS 2019 competition data at different locations depending
on how it was attached to the notebook, e.g.:
    /kaggle/input/aptos2019-blindness-detection/
    /kaggle/input/competitions/aptos2019-blindness-detection/
This module probes the known locations (and falls back to a recursive search)
so every script agrees on the path without hard-coding it. Import DATASET_CSV /
DATASET_ROOT from here instead of hard-coding "/kaggle/input/...".
"""

import os
import glob

# Known mount points, most-specific first.
_CANDIDATES = [
    "/kaggle/input/competitions/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
]


def find_dataset_root() -> str:
    """Return the directory that holds APTOS train.csv + train_images/."""
    for path in _CANDIDATES:
        if os.path.exists(os.path.join(path, "train.csv")):
            return path
    # Fall back to searching anywhere under /kaggle/input.
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    # Nothing found — return the conventional path so errors stay readable.
    return _CANDIDATES[-1]


DATASET_ROOT = find_dataset_root()
DATASET_CSV  = os.path.join(DATASET_ROOT, "train.csv")
