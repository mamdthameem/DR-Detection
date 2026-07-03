"""
Task 1: Preprocessing pipeline for APTOS 2019 fundus images.
Pipeline: black-border crop → resize 224×224 → CLAHE per BGR channel → save PNG
Also outputs a class distribution bar chart.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import argparse

# ── Paths ──────────────────────────────────────────────────────────────────────
from paths import DATASET_ROOT as DATASET_PATH   # APTOS path auto-resolved (see paths.py)
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
OUTPUTS_DIR       = "/kaggle/working/outputs"
IMG_SIZE          = 224

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ── Image processing helpers ───────────────────────────────────────────────────

def crop_black_borders(img: np.ndarray, tolerance: int = 7) -> np.ndarray:
    """
    Remove the black circular border common in fundus photographs.
    Finds the bounding box of the largest non-black contour and crops to it.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    img_h, img_w = img.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)
    cropped = img[y1:y2, x1:x2]
    # Guard against degenerate crops (< 10% of original area)
    if cropped.size < 0.1 * img.size:
        return img
    return cropped


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE independently to each BGR channel."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    b, g, r = cv2.split(img)
    return cv2.merge([clahe.apply(b), clahe.apply(g), clahe.apply(r)])


def preprocess_image(img_path: str, size: int = IMG_SIZE) -> np.ndarray:
    """Full single-image preprocessing pipeline."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = crop_black_borders(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = apply_clahe(img)
    return img


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_class_distribution(df: pd.DataFrame, save_path: str) -> None:
    """Bar chart of per-class image counts."""
    counts = df["diagnosis"].value_counts().sort_index()
    colors = ["#2196F3", "#4CAF50", "#FFC107", "#FF5722", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(CLASS_NAMES, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("DR Grade", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("APTOS 2019 – Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved class distribution chart → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess APTOS 2019 fundus images")
    parser.add_argument("--dataset-path",  default=DATASET_PATH)
    parser.add_argument("--output-path",   default=PREPROCESSED_PATH)
    parser.add_argument("--outputs-dir",   default=OUTPUTS_DIR)
    parser.add_argument("--img-size",      type=int, default=IMG_SIZE)
    args = parser.parse_args()

    np.random.seed(42)
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)

    csv_path = os.path.join(args.dataset_path, "train.csv")
    img_dir  = os.path.join(args.dataset_path, "train_images")
    df = pd.read_csv(csv_path)
    print(f"Total images in CSV: {len(df)}")

    # Class distribution chart
    save_class_distribution(df, os.path.join(args.outputs_dir, "class_distribution.png"))

    # Preprocess each image
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_id = row["id_code"]
        # Try common extensions
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            failed.append(img_id)
            continue
        try:
            processed = preprocess_image(img_path, args.img_size)
            cv2.imwrite(os.path.join(args.output_path, img_id + ".png"), processed)
        except Exception as exc:
            print(f"\n  [WARN] Failed {img_id}: {exc}")
            failed.append(img_id)

    print(f"\nPreprocessing complete: {len(df) - len(failed)}/{len(df)} images saved.")
    if failed:
        print(f"  Failed IDs ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
