"""
Task 8: LIME explanations for EfficientNet-B0.
Uses LimeImageExplainer (superpixel-based).
Shows: original | LIME superpixels | top positive features (green mask).
Individual panel images saved to outputs/xai/panels/lime_class_{i}.png.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import argparse

from model import get_model
from dataset import get_splits, get_transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
DATASET_CSV       = "/kaggle/input/aptos2019-blindness-detection/train.csv"

NUM_CLASSES  = 5
CLASS_NAMES  = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
LIME_SAMPLES = 500    # perturbation samples per image
LIME_FEATS   = 10     # top superpixels to highlight


# ── Prediction function for LIME ───────────────────────────────────────────────

def predict_batch(images: np.ndarray, model, transform, device) -> np.ndarray:
    """
    LIME passes perturbed images as uint8 (N, H, W, C) numpy arrays.
    Returns softmax probabilities (N, num_classes).
    """
    pil_imgs = [Image.fromarray(img.astype(np.uint8)) for img in images]
    tensors  = torch.stack([transform(img) for img in pil_imgs]).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensors), dim=1).cpu().numpy()
    return probs


# ── Helper ─────────────────────────────────────────────────────────────────────

def save_panel(img: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype in (np.float32, np.float64):
        img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LIME explanations for EfficientNet-B0")
    parser.add_argument("--img-dir",      default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path",     default=DATASET_CSV)
    parser.add_argument("--num-samples",  type=int, default=LIME_SAMPLES)
    parser.add_argument("--num-features", type=int, default=LIME_FEATS)
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

    # Load XAI sample indices
    with open(os.path.join(OUTPUTS_DIR, "xai_sample_indices.json")) as f:
        xai_samples = json.load(f)

    _, _, test_df = get_splits(args.csv_path)
    transform = get_transforms(is_train=False)

    explainer = lime_image.LimeImageExplainer()

    fig, axes = plt.subplots(NUM_CLASSES, 3, figsize=(12, 4 * NUM_CLASSES))
    for col, title in enumerate(["Original",
                                  "LIME Superpixels (all)",
                                  "Top Positive Features"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, cls in enumerate(range(NUM_CLASSES)):
        idx    = xai_samples[str(cls)]
        img_id = test_df.iloc[idx]["id_code"]
        path   = os.path.join(args.img_dir, img_id + ".png")

        # LIME needs uint8 RGB numpy array
        img_array = np.array(Image.open(path).convert("RGB"))

        print(f"  LIME class {cls} ({CLASS_NAMES[cls]}) – {args.num_samples} samples ...")
        explanation = explainer.explain_instance(
            img_array,
            classifier_fn=lambda imgs: predict_batch(imgs, model, transform, device),
            top_labels=NUM_CLASSES,
            hide_color=0,
            num_samples=args.num_samples,
            random_seed=42,
        )

        # Positive superpixels only (green mask)
        temp_pos, mask_pos = explanation.get_image_and_mask(
            cls,
            positive_only=True,
            num_features=args.num_features,
            hide_rest=False,
        )

        # All superpixels with boundaries
        _, mask_all = explanation.get_image_and_mask(
            cls,
            positive_only=False,
            num_features=args.num_features,
            hide_rest=False,
        )
        overlay_all = mark_boundaries(img_array, mask_all)   # float64
        overlay_pos = mark_boundaries(temp_pos,  mask_pos)   # float64

        # Show
        axes[row, 0].imshow(img_array)
        axes[row, 0].set_ylabel(CLASS_NAMES[cls], fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(overlay_all)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay_pos)
        axes[row, 2].axis("off")

        # Save individual panel for xai_compare.py
        panel_uint8 = (overlay_pos * 255).clip(0, 255).astype(np.uint8)
        save_panel(panel_uint8, os.path.join(panels_dir, f"lime_class_{cls}.png"))

    plt.suptitle("LIME Explanations: EfficientNet-B0\n"
                 "(Green = top positive superpixels per DR class)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "xai", "lime_efficientnet.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
