"""
Task 9: Paper-ready XAI comparison grid.
Loads individual panel images saved by xai_gradcam.py, xai_shap.py, xai_lime.py.
Builds a (NUM_CLASSES rows) × 4 cols [Original | Grad-CAM | SHAP | LIME] figure.
Output: outputs/xai/xai_comparison_grid.png  (200 dpi for publication quality)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import argparse

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = "/kaggle/working/outputs"
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

PANEL_NAMES = {
    "Original":  "original_class_{i}.png",   # saved by xai_gradcam.py
    "Grad-CAM":  "gradcam_class_{i}.png",
    "SHAP":      "shap_class_{i}.png",        # saved by xai_shap.py
    "LIME":      "lime_class_{i}.png",        # saved by xai_lime.py
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_panel(path: str) -> np.ndarray:
    """Load a panel image as uint8 RGB numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing panel: {path}\n"
            "Run xai_gradcam.py, xai_shap.py, xai_lime.py first."
        )
    return np.array(Image.open(path).convert("RGB"))


def load_all_panels(panels_dir: str) -> dict:
    """
    Returns dict: method_name → list of 5 uint8 arrays (one per DR class).
    """
    data = {}
    for method, template in PANEL_NAMES.items():
        panels = []
        for cls in range(NUM_CLASSES):
            fname = template.format(i=cls)
            panels.append(load_panel(os.path.join(panels_dir, fname)))
        data[method] = panels
    return data


# ── Build grid ─────────────────────────────────────────────────────────────────

def build_comparison_grid(panels: dict, save_path: str, dpi: int = 200) -> None:
    n_methods = len(PANEL_NAMES)
    fig = plt.figure(figsize=(4 * n_methods, 4 * NUM_CLASSES + 0.6))
    gs  = gridspec.GridSpec(
        NUM_CLASSES, n_methods,
        figure=fig, hspace=0.04, wspace=0.04,
        left=0.06, right=0.99, top=0.96, bottom=0.01,
    )

    methods = list(PANEL_NAMES.keys())

    for col, method in enumerate(methods):
        for row in range(NUM_CLASSES):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(panels[method][row])
            ax.axis("off")

            # Column headers on first row
            if row == 0:
                ax.set_title(method, fontsize=13, fontweight="bold", pad=6)

            # Row labels (DR class) on first column
            if col == 0:
                ax.set_ylabel(CLASS_NAMES[row], fontsize=11,
                              rotation=90, labelpad=8, va="center")

    fig.suptitle(
        "XAI Comparison – EfficientNet-B0: Diabetic Retinopathy Detection",
        fontsize=13, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"Paper-ready XAI comparison grid saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build XAI comparison grid for paper")
    parser.add_argument("--output-dir", default=OUTPUTS_DIR)
    parser.add_argument("--dpi",        type=int, default=200)
    args = parser.parse_args()

    panels_dir = os.path.join(args.output_dir, "xai", "panels")
    save_path  = os.path.join(args.output_dir, "xai", "xai_comparison_grid.png")

    print("Loading XAI panels ...")
    panels = load_all_panels(panels_dir)
    print(f"  Loaded {sum(len(v) for v in panels.values())} panel images.")

    build_comparison_grid(panels, save_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
