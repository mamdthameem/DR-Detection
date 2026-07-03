"""
EfficientNet-B0 preprocessing module (LAB-CLAHE variant) for APTOS 2019.

Difference from preprocess.py: CLAHE is applied to the LAB **L-channel only**
(then converted back to RGB), which preserves colour better than the original
per-BGR-channel CLAHE and typically gives cleaner fundus contrast.

preprocess_image(img) order:
    (1) grayscale-threshold crop to remove black borders
    (2) CLAHE on LAB L-channel (clipLimit=2.0, tileGrid=8x8) -> back to RGB
    (3) resize to 224x224

This module is self-contained and API-compatible with dataset.py
(get_dataloaders / get_splits / get_transforms / APTOSDataset), and reads RAW
images on the fly. Run as a script to (a) save a 3-sample before/after check
figure and (b) build a cached copy of the preprocessed images so the existing
evaluate.py / xai_*.py scripts can consume them via --img-dir.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # safe for headless / Kaggle script runs
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from paths import DATASET_ROOT, DATASET_CSV          # auto-resolved APTOS location
from dataset import get_splits, get_class_weights     # reuse identical splits + weights

# ── Paths & constants ──────────────────────────────────────────────────────────
RAW_IMG_DIR   = os.path.join(DATASET_ROOT, "train_images")   # raw fundus images
LAB_CACHE     = "/kaggle/working/data/aptos_lab"             # cache for eval + XAI
OUTPUTS_DIR   = "/kaggle/working/outputs"

IMG_SIZE      = 224
NUM_CLASSES   = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ── Deterministic image preprocessing ──────────────────────────────────────────

def crop_black_borders(img_bgr: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """Crop to the bounding box of the largest non-black contour (fundus disc)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cropped = img_bgr[y:y + h, x:x + w]
    # Guard against degenerate crops (< 10% of original area)
    if cropped.size < 0.1 * img_bgr.size:
        return img_bgr
    return cropped


def apply_clahe_lab(img_bgr: np.ndarray, clip_limit: float = 2.0,
                    tile_grid: tuple = (8, 8)) -> np.ndarray:
    """CLAHE on the LAB L-channel only, then convert back to BGR."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def preprocess_image(img_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Full deterministic pipeline. Input: BGR uint8 (cv2.imread output).
    Output: (size, size, 3) RGB uint8 — crop -> LAB-L CLAHE -> resize.
    """
    img = crop_black_borders(img_bgr)
    img = apply_clahe_lab(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_raw(img_dir: str, id_code: str) -> np.ndarray:
    """Read a raw APTOS image by id_code, trying common extensions."""
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(img_dir, id_code + ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
    raise FileNotFoundError(f"Raw image not found for '{id_code}' in {img_dir}")


# ── Dataset (on-the-fly preprocessing) ─────────────────────────────────────────

class APTOSDataset(Dataset):
    """Reads RAW images, applies preprocess_image in __getitem__, then transform."""

    def __init__(self, df: pd.DataFrame, img_dir: str = RAW_IMG_DIR, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row     = self.df.loc[idx]
        label   = int(row["diagnosis"])
        img_bgr = _read_raw(self.img_dir, row["id_code"])
        rgb     = preprocess_image(img_bgr)          # (224,224,3) RGB uint8
        img     = Image.fromarray(rgb)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ── Transform pipelines ────────────────────────────────────────────────────────
# preprocess_image already crops + CLAHEs + resizes to 224, so the transforms do
# only augmentation (train) + tensor/normalize. No Resize needed here.

def get_transforms(is_train: bool = True):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_train:
        return transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),   # mild
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


# ── DataLoader factory (API-compatible with dataset.py) ────────────────────────

def get_dataloaders(img_dir: str     = RAW_IMG_DIR,
                    csv_path: str     = DATASET_CSV,
                    batch_size: int   = 32,
                    num_workers: int  = 4,
                    seed: int         = 42):
    """Returns (train_loader, val_loader, test_loader, class_weights,
                (train_df, val_df, test_df)) — same shape as dataset.get_dataloaders."""
    train_df, val_df, test_df = get_splits(csv_path, seed=seed)   # identical splits

    train_ds = APTOSDataset(train_df, img_dir, get_transforms(True))
    val_ds   = APTOSDataset(val_df,   img_dir, get_transforms(False))
    test_ds  = APTOSDataset(test_df,  img_dir, get_transforms(False))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    class_weights = get_class_weights(train_df)
    return train_loader, val_loader, test_loader, class_weights, (train_df, val_df, test_df)


# ── Offline cache (so evaluate.py / xai_*.py can use --img-dir) ─────────────────

def build_cache(out_dir: str = LAB_CACHE, raw_dir: str = RAW_IMG_DIR,
                csv_path: str = DATASET_CSV) -> None:
    """Write preprocess_image(raw) for every image to out_dir as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    failed = []
    for id_code in tqdm(df["id_code"], desc="Building LAB cache"):
        try:
            rgb = preprocess_image(_read_raw(raw_dir, id_code))
            cv2.imwrite(os.path.join(out_dir, id_code + ".png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            failed.append((id_code, str(exc)))
    print(f"LAB cache complete: {len(df) - len(failed)}/{len(df)} → {out_dir}")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[:3]}")


# ── Visualization: 3 samples before vs after ───────────────────────────────────

def visualize_preprocessing(n: int = 3, save_path: str = None,
                            raw_dir: str = RAW_IMG_DIR, seed: int = 42) -> str:
    """Save an n-row before/after figure for sanity-checking CLAHE output."""
    if save_path is None:
        save_path = os.path.join(OUTPUTS_DIR, "effnet_preprocess_check.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.read_csv(DATASET_CSV).sample(n=n, random_state=seed).reset_index(drop=True)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, row in df.iterrows():
        raw     = _read_raw(raw_dir, row["id_code"])
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        proc    = preprocess_image(raw)
        axes[i, 0].imshow(raw_rgb)
        axes[i, 0].set_title(f"Before  (class {row['diagnosis']} – {CLASS_NAMES[int(row['diagnosis'])]})", fontsize=10)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(proc)
        axes[i, 1].set_title("After  (crop + LAB-L CLAHE + 224)", fontsize=10)
        axes[i, 1].axis("off")

    plt.suptitle("EfficientNet-B0 Preprocessing – Before vs After", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Preprocess check figure saved → {save_path}")
    return save_path


# ── Standalone ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LAB-CLAHE preprocessing for EfficientNet-B0")
    ap.add_argument("--raw-dir",        default=RAW_IMG_DIR)
    ap.add_argument("--out-dir",        default=LAB_CACHE)
    ap.add_argument("--n-vis",          type=int, default=3)
    ap.add_argument("--skip-cache",     action="store_true", help="Don't build the LAB cache")
    ap.add_argument("--skip-visualize", action="store_true", help="Don't save the before/after figure")
    args = ap.parse_args()

    np.random.seed(42)
    if not args.skip_visualize:
        visualize_preprocessing(args.n_vis, raw_dir=args.raw_dir)
    if not args.skip_cache:
        build_cache(args.out_dir, args.raw_dir, DATASET_CSV)


if __name__ == "__main__":
    main()
