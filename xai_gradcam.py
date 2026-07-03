"""
Task 6: Grad-CAM visualisations.
Primary: EfficientNet-B0 (5-class grid: original | heatmap | overlay).
Comparison: all 4 models on the same 5 images (overlay only).
Individual panel images are saved to outputs/xai/panels/ for xai_compare.py.
"""

import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import argparse

from model import get_model, MODEL_REGISTRY
from dataset import get_splits, get_transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
from paths import DATASET_CSV   # APTOS path auto-resolved (see paths.py)

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

# ── GradCAM target-layer selectors ────────────────────────────────────────────
# Each lambda receives the APTOSModel instance and returns a list of layers.
# conv_head  = last 1×1 conv in EfficientNet-B0 (spatial maps before GAP)
# layer4[-1] = last residual block in ResNet-50
# features[-3] = last Conv2d in VGG-16 (before final ReLU+MaxPool)
# Mixed_7c   = last inception-C block in InceptionV3

GRADCAM_LAYERS = {
    "efficientnet_b0": lambda m: [m.backbone.conv_head],
    "resnet50":        lambda m: [m.backbone.layer4[-1]],
    "vgg16":           lambda m: [m.backbone.features[-3]],
    "inceptionv3":     lambda m: [m.backbone.Mixed_7c],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device) -> torch.nn.Module:
    ckpt  = os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth")
    model = get_model(model_name, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def load_img_array(img_id: str, img_dir: str) -> np.ndarray:
    """Return the preprocessed image as float32 RGB in [0, 1]."""
    path = os.path.join(img_dir, img_id + ".png")
    img  = cv2.imread(path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(img, (224, 224))
    return (img / 255.0).astype(np.float32)


def get_tensor(img_id: str, img_dir: str, transform) -> torch.Tensor:
    path = os.path.join(img_dir, img_id + ".png")
    img  = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def compute_cam(model, model_name: str, tensor: torch.Tensor,
                target_class: int, device) -> np.ndarray:
    """Return a (224, 224) float32 GradCAM heatmap in [0, 1]."""
    target_layers = GRADCAM_LAYERS[model_name](model)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam(input_tensor=tensor.to(device), targets=targets)
    return grayscale[0]   # (H, W)


def save_panel(img: np.ndarray, path: str) -> None:
    """Save a uint8 (H, W, 3) image as PNG panel."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ── EfficientNet-B0 5-class grid ──────────────────────────────────────────────

def make_effnet_grid(model, device, xai_samples: dict,
                     test_df, img_dir: str) -> None:
    transform   = get_transforms(is_train=False)
    panels_dir  = os.path.join(OUTPUTS_DIR, "xai", "panels")
    fig, axes   = plt.subplots(NUM_CLASSES, 3, figsize=(12, 4 * NUM_CLASSES))

    for col, title in enumerate(["Original", "Grad-CAM Heatmap", "Overlay"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, cls in enumerate(range(NUM_CLASSES)):
        idx    = xai_samples[str(cls)]
        img_id = test_df.iloc[idx]["id_code"]

        orig   = load_img_array(img_id, img_dir)                     # (H,W,3) float
        tensor = get_tensor(img_id, img_dir, transform)

        cam_map = compute_cam(model, "efficientnet_b0", tensor, cls, device)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        overlay = show_cam_on_image(orig, cam_map, use_rgb=True)     # uint8

        # Save individual panels for xai_compare.py
        save_panel(orig,    os.path.join(panels_dir, f"original_class_{cls}.png"))
        save_panel(overlay, os.path.join(panels_dir, f"gradcam_class_{cls}.png"))

        axes[row, 0].imshow(orig)
        axes[row, 0].set_ylabel(CLASS_NAMES[cls], fontsize=11)
        axes[row, 1].imshow(heatmap)
        axes[row, 2].imshow(overlay)
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Grad-CAM: EfficientNet-B0", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "xai", "gradcam_efficientnet.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── All-models comparison grid ────────────────────────────────────────────────

def make_all_models_grid(device, xai_samples: dict, test_df,
                         img_dir: str, model_names: list) -> None:
    transform  = get_transforms(is_train=False)
    models     = {name: load_model(name, device) for name in model_names}
    n_models   = len(model_names)

    fig, axes = plt.subplots(NUM_CLASSES, n_models,
                             figsize=(4 * n_models, 4 * NUM_CLASSES))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, name in enumerate(model_names):
        axes[0, col].set_title(name, fontsize=10, fontweight="bold")

    for row, cls in enumerate(range(NUM_CLASSES)):
        idx    = xai_samples[str(cls)]
        img_id = test_df.iloc[idx]["id_code"]
        orig   = load_img_array(img_id, img_dir)
        tensor = get_tensor(img_id, img_dir, transform)

        for col, name in enumerate(model_names):
            try:
                cam_map = compute_cam(models[name], name, tensor, cls, device)
                overlay = show_cam_on_image(orig, cam_map, use_rgb=True)
            except Exception as exc:
                print(f"    [WARN] GradCAM failed {name} class {cls}: {exc}")
                overlay = (orig * 255).astype(np.uint8)
            axes[row, col].imshow(overlay)
            if col == 0:
                axes[row, col].set_ylabel(CLASS_NAMES[cls], fontsize=10)
            axes[row, col].axis("off")

    plt.suptitle("Grad-CAM Comparison: All Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "xai", "gradcam_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM XAI visualisation")
    parser.add_argument("--img-dir",  default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path", default=DATASET_CSV)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.join(OUTPUTS_DIR, "xai", "panels"), exist_ok=True)

    idx_path = os.path.join(OUTPUTS_DIR, "xai_sample_indices.json")
    with open(idx_path) as f:
        xai_samples = json.load(f)

    _, _, test_df = get_splits(args.csv_path)

    # EfficientNet-B0 detailed grid
    print("GradCAM – EfficientNet-B0 ...")
    effnet = load_model("efficientnet_b0", device)
    make_effnet_grid(effnet, device, xai_samples, test_df, args.img_dir)

    # All models overlay comparison
    available = [
        name for name in MODEL_REGISTRY
        if os.path.exists(os.path.join(CHECKPOINTS_DIR, f"{name}_best.pth"))
    ]
    print(f"GradCAM – all models ({available}) ...")
    make_all_models_grid(device, xai_samples, test_df, args.img_dir, available)


if __name__ == "__main__":
    main()
