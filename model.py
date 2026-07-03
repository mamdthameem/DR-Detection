"""
Task 3: Model definitions.
EfficientNet-B0 (primary) + ResNet-50, VGG-16, InceptionV3 (baselines).
All loaded via timm, pretrained ImageNet, custom 5-class head.
"""

import os
import torch
import torch.nn as nn
import timm
import argparse

# ── Constants ──────────────────────────────────────────────────────────────────
OUTPUTS_DIR = "/kaggle/working/outputs"
NUM_CLASSES = 5
IMG_SIZE    = 224

# timm model identifiers; in_features resolved dynamically via backbone.num_features
MODEL_REGISTRY = {
    "efficientnet_b0": "efficientnet_b0",
    "resnet50":        "resnet50",
    "vgg16":           "vgg16",
    "inceptionv3":     "inception_v3",
}


# ── Classifier head (shared architecture for all 4 models) ────────────────────

def build_head(in_features: int, num_classes: int = NUM_CLASSES) -> nn.Sequential:
    """Dropout(0.3) → Linear → ReLU → Dropout(0.2) → Linear."""
    return nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )


# ── Model wrapper ──────────────────────────────────────────────────────────────

class APTOSModel(nn.Module):
    """
    timm backbone (num_classes=0, global_pool='avg') + custom classification head.
    The pooled feature dimensionality is measured from a dummy forward pass, since
    timm's `num_features` can disagree with the real output for some backbones
    (e.g. VGG-16 reports 512 but outputs 4096) — no hard-coded in_features per model.
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        timm_name = MODEL_REGISTRY[model_name]
        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # Measure the real pooled-feature dim from a dummy forward. timm's
        # `num_features` can disagree with the actual output (e.g. VGG-16 reports
        # 512 but outputs 4096), which would build a mismatched head.
        self.backbone.eval()
        with torch.no_grad():
            in_features = self.backbone(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)).shape[1]
        self.backbone.train()
        self.classifier = build_head(in_features, num_classes)
        self.model_name = model_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, in_features) after GAP
        return self.classifier(features)


def get_model(model_name: str, pretrained: bool = True,
              num_classes: int = NUM_CLASSES) -> APTOSModel:
    return APTOSModel(model_name, pretrained=pretrained, num_classes=num_classes)


# ── Model summary ──────────────────────────────────────────────────────────────

def _count_parameters(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _estimate_flops(model: nn.Module, img_size: int = IMG_SIZE, device: str = "cpu"):
    try:
        from thop import profile, clever_format
        dummy = torch.randn(1, 3, img_size, img_size).to(device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops_str, _ = clever_format([macs * 2], "%.3f")
        return flops_str
    except Exception:
        return "N/A (install thop)"


def save_model_summary(output_path: str = None, img_size: int = IMG_SIZE) -> None:
    if output_path is None:
        output_path = os.path.join(OUTPUTS_DIR, "model_summary.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = ["=" * 64, "  Model Summary – DR Classification", "=" * 64, ""]
    for model_name in MODEL_REGISTRY:
        model = get_model(model_name, pretrained=False)
        model.eval()
        total, trainable = _count_parameters(model)
        flops_str = _estimate_flops(model, img_size)
        in_feats  = model.classifier[1].in_features
        lines += [
            f"Model          : {model_name}",
            f"  Backbone     : {MODEL_REGISTRY[model_name]}",
            f"  In-features  : {in_feats}",
            f"  Total params : {total:,}",
            f"  Train params : {trainable:,}",
            f"  FLOPs (est.) : {flops_str}",
            "",
        ]

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(text)
    print(f"Model summary saved → {output_path}")


# ── Standalone ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print and save model summaries")
    parser.add_argument("--output-dir", default=OUTPUTS_DIR)
    args = parser.parse_args()

    torch.manual_seed(42)
    save_model_summary(os.path.join(args.output_dir, "model_summary.txt"))
