"""
Task 2: PyTorch Dataset for APTOS 2019.
Provides: APTOSDataset, stratified 70/15/15 splits, augmentation, class weights.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import argparse

# ── Paths & constants ──────────────────────────────────────────────────────────
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
from paths import DATASET_CSV   # APTOS path auto-resolved (see paths.py)
IMAGENET_MEAN     = [0.485, 0.456, 0.406]
IMAGENET_STD      = [0.229, 0.224, 0.225]
IMG_SIZE          = 224
NUM_CLASSES       = 5
TRAIN_RATIO       = 0.70
VAL_RATIO         = 0.15   # remaining 0.15 → test


# ── Dataset ────────────────────────────────────────────────────────────────────

class APTOSDataset(Dataset):
    """Loads preprocessed fundus images from disk with optional transforms."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.img_dir   = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.loc[idx]
        label = int(row["diagnosis"])
        path  = os.path.join(self.img_dir, row["id_code"] + ".png")
        img   = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


# ── Transforms ─────────────────────────────────────────────────────────────────

def get_transforms(is_train: bool = True, img_size: int = IMG_SIZE):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ── Splits ─────────────────────────────────────────────────────────────────────

def get_splits(csv_path: str = DATASET_CSV,
               train_ratio: float = TRAIN_RATIO,
               val_ratio: float = VAL_RATIO,
               seed: int = 42):
    """Stratified train / val / test split."""
    df = pd.read_csv(csv_path)
    test_size = round(1.0 - train_ratio - val_ratio, 10)

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["diagnosis"], random_state=seed)

    relative_val = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val,
        stratify=train_val_df["diagnosis"], random_state=seed)

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


# ── Class weights ──────────────────────────────────────────────────────────────

def get_class_weights(train_df: pd.DataFrame,
                      num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency weights, normalised so they sum to num_classes."""
    counts  = train_df["diagnosis"].value_counts().sort_index().values.astype(float)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ── DataLoader factory ─────────────────────────────────────────────────────────

def get_dataloaders(img_dir: str      = PREPROCESSED_PATH,
                    csv_path: str     = DATASET_CSV,
                    batch_size: int   = 32,
                    num_workers: int  = 2,
                    img_size: int     = IMG_SIZE,
                    seed: int         = 42):
    """
    Returns (train_loader, val_loader, test_loader, class_weights,
             (train_df, val_df, test_df)).
    """
    train_df, val_df, test_df = get_splits(csv_path, seed=seed)

    train_ds = APTOSDataset(train_df, img_dir, get_transforms(True,  img_size))
    val_ds   = APTOSDataset(val_df,   img_dir, get_transforms(False, img_size))
    test_ds  = APTOSDataset(test_df,  img_dir, get_transforms(False, img_size))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    class_weights = get_class_weights(train_df)
    return train_loader, val_loader, test_loader, class_weights, (train_df, val_df, test_df)


# ── Standalone smoke-test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify data loading")
    parser.add_argument("--img-dir",    default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path",   default=DATASET_CSV)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    train_loader, val_loader, test_loader, class_weights, (train_df, val_df, test_df) = \
        get_dataloaders(args.img_dir, args.csv_path, args.batch_size)

    print(f"Split sizes  →  train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")
    print(f"Class weights: {class_weights.numpy().round(4)}")

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape: {imgs.shape}  |  labels sample: {labels[:8].tolist()}")
