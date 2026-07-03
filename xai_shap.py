"""
Task 7: SHAP explanations for EfficientNet-B0.
Uses shap.GradientExplainer with 50 background training samples.
Shows the SHAP attribution map for each image's predicted class
(red = positive contribution, blue = negative).
Individual panel images saved to outputs/xai/panels/shap_class_{i}.png.

NOTE: SHAP with GradientExplainer is GPU-memory intensive.
      Reduce --n-background (default 50) if you encounter OOM errors.
"""

import os
import json
import numpy as np
import torch
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from model import get_model
from dataset import get_splits, get_transforms, APTOSDataset

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
from paths import DATASET_CSV   # APTOS path auto-resolved (see paths.py)

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_background_batch(train_df, img_dir: str, transform,
                         n: int = 50, seed: int = 42) -> torch.Tensor:
    """Sample n training images as SHAP background reference distribution."""
    np.random.seed(seed)
    subset = train_df.sample(n=n, random_state=seed).reset_index(drop=True)
    ds     = APTOSDataset(subset, img_dir, transform)
    return torch.stack([ds[i][0] for i in range(len(ds))])


def denormalize(tensor_img: np.ndarray) -> np.ndarray:
    """Undo ImageNet normalisation for display; returns float32 in [0,1]."""
    img = tensor_img.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return img.clip(0.0, 1.0)


def shap_to_heatmap(shap_values_chw: np.ndarray) -> np.ndarray:
    """
    Sum SHAP values across colour channels to obtain a single (H,W) map.
    Positive = region pushes toward predicted class; negative = pushes away.
    """
    return shap_values_chw.sum(axis=0)   # (C,H,W) → (H,W)


def save_panel(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SHAP explanations for EfficientNet-B0")
    parser.add_argument("--img-dir",      default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path",     default=DATASET_CSV)
    parser.add_argument("--n-background", type=int, default=50,
                        help="Number of background samples (reduce to 20-30 if OOM)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    panels_dir = os.path.join(OUTPUTS_DIR, "xai", "panels")
    os.makedirs(panels_dir, exist_ok=True)

    # Load model
    ckpt  = os.path.join(CHECKPOINTS_DIR, "efficientnet_b0_best.pth")
    model = get_model("efficientnet_b0", pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Load XAI sample indices and data splits
    with open(os.path.join(OUTPUTS_DIR, "xai_sample_indices.json")) as f:
        xai_samples = json.load(f)

    train_df, _, test_df = get_splits(args.csv_path)
    transform = get_transforms(is_train=False)

    # Background reference distribution
    print(f"  Loading {args.n_background} background samples ...")
    background = get_background_batch(
        train_df, args.img_dir, transform, args.n_background).to(device)

    # Test images to explain
    test_tensors = []
    for cls in range(NUM_CLASSES):
        idx    = xai_samples[str(cls)]
        img_id = test_df.iloc[idx]["id_code"]
        path   = os.path.join(args.img_dir, img_id + ".png")
        img    = Image.open(path).convert("RGB")
        test_tensors.append(transform(img).unsqueeze(0))
    test_batch = torch.cat(test_tensors, dim=0).to(device)  # (5, 3, 224, 224)

    # Predicted classes (for selecting which SHAP output to visualise)
    with torch.no_grad():
        predicted = model(test_batch).argmax(1).cpu().numpy()

    # SHAP GradientExplainer
    print("  Running GradientExplainer (may take several minutes) ...")
    explainer  = shap.GradientExplainer(model, background)
    shap_vals  = explainer.shap_values(test_batch)
    # shap_vals: list of NUM_CLASSES arrays, each shape (5, 3, 224, 224)

    # ── Build figure ──
    fig, axes = plt.subplots(NUM_CLASSES, 2, figsize=(8, 4 * NUM_CLASSES))

    for i in range(NUM_CLASSES):
        pred_cls = int(predicted[i])

        # Tensor image → display array
        orig_np  = test_batch[i].cpu().numpy()   # (3, 224, 224)
        orig_rgb = denormalize(orig_np)           # (224, 224, 3) float [0,1]

        # SHAP values for predicted class, summed over channels
        sv_chw   = shap_vals[pred_cls][i]         # (3, 224, 224)
        heatmap  = shap_to_heatmap(sv_chw)        # (224, 224)
        vmax     = max(abs(heatmap.min()), abs(heatmap.max())) + 1e-8

        # Original
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"Original ({CLASS_NAMES[i]})", fontsize=10)
        axes[i, 0].axis("off")

        # SHAP heatmap (diverging red-blue)
        im = axes[i, 1].imshow(heatmap, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f"SHAP (pred: {CLASS_NAMES[pred_cls]})", fontsize=10)
        axes[i, 1].axis("off")
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Save individual panels
        save_panel(orig_rgb, os.path.join(panels_dir, f"original_class_{i}_shap.png"))

        # Normalise heatmap to [0,1] for PNG storage (0.5 = zero, <0.5 = blue, >0.5 = red)
        hm_norm = ((heatmap / vmax) * 0.5 + 0.5).clip(0, 1)  # [0,1]
        cmap    = plt.get_cmap("bwr")
        hm_rgb  = cmap(hm_norm)[..., :3]                      # (H,W,3) float
        save_panel(hm_rgb, os.path.join(panels_dir, f"shap_class_{i}.png"))

    plt.suptitle("SHAP Explanations: EfficientNet-B0\n"
                 "(Red = positive contribution  |  Blue = negative contribution)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "xai", "shap_efficientnet.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
