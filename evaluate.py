"""
Task 5: Evaluation of all 4 trained models on the held-out test set.
Metrics: Accuracy, Precision, Recall, F1 (macro), AUC (macro OvR),
         Cohen's Kappa, Matthews Correlation Coefficient.
Outputs: confusion matrices, ROC curves, results_comparison.csv,
         xai_sample_indices.json (one correctly-classified image per DR class).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize
import argparse

from model import get_model, MODEL_REGISTRY
from dataset import get_dataloaders

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
DATASET_CSV       = "/kaggle/input/aptos2019-blindness-detection/train.csv"

NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Returns (true_labels, predicted_labels, softmax_probs)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for imgs, labels in tqdm(loader, desc="  Inference", leave=False):
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = F.softmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray,
                    probs: np.ndarray) -> dict:
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    return {
        "accuracy":         accuracy_score(labels, preds),
        "precision_macro":  precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro":     recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro":         f1_score(labels, preds, average="macro", zero_division=0),
        "auc_macro_ovr":    roc_auc_score(labels_bin, probs,
                                          multi_class="ovr", average="macro"),
        "cohen_kappa":      cohen_kappa_score(labels, preds),
        "mcc":              matthews_corrcoef(labels, preds),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, model_name: str, save_path: str) -> None:
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    ax.set_title(f"{model_name} – Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(labels, probs, model_name: str, save_path: str) -> None:
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    colors     = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    fig, ax    = plt.subplots(figsize=(8, 6))
    for i, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc_score   = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{CLASS_NAMES[i]} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} – ROC Curves (One-vs-Rest)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── XAI sample selection ───────────────────────────────────────────────────────

def find_xai_samples(labels: np.ndarray, preds: np.ndarray) -> dict:
    """
    For each DR class, find a correctly-classified test-set index.
    Picks the middle index among correct samples for representativeness.
    """
    samples = {}
    for cls in range(NUM_CLASSES):
        correct = np.where((labels == cls) & (preds == cls))[0]
        if len(correct) > 0:
            samples[str(cls)] = int(correct[len(correct) // 2])
        else:
            # Fallback: any sample of this class
            fallback = np.where(labels == cls)[0]
            if len(fallback) > 0:
                samples[str(cls)] = int(fallback[0])
    return samples


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DR models")
    parser.add_argument("--models",      nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--img-dir",     default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path",    default=DATASET_CSV)
    parser.add_argument("--batch-size",  type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    plot_dir = os.path.join(OUTPUTS_DIR, "plots")
    xai_dir  = os.path.join(OUTPUTS_DIR, "xai")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(xai_dir,  exist_ok=True)

    _, _, test_loader, _, (_, _, test_df) = get_dataloaders(
        args.img_dir, args.csv_path, args.batch_size)

    all_results = []
    xai_saved   = False

    for model_name in args.models:
        ckpt = os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth")
        if not os.path.exists(ckpt):
            print(f"[SKIP] Checkpoint not found: {ckpt}")
            continue

        print(f"\nEvaluating {model_name} ...")
        model = get_model(model_name, pretrained=False).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        labels, preds, probs = run_inference(model, test_loader, device)
        metrics = compute_metrics(labels, preds, probs)
        metrics["model"] = model_name
        all_results.append(metrics)

        plot_confusion_matrix(labels, preds, model_name,
                              os.path.join(plot_dir, f"{model_name}_confusion_matrix.png"))
        plot_roc_curves(labels, probs, model_name,
                        os.path.join(plot_dir, f"{model_name}_roc_curve.png"))

        # XAI sample indices are derived from EfficientNet-B0 predictions
        if model_name == "efficientnet_b0" and not xai_saved:
            xai_samples = find_xai_samples(labels, preds)
            idx_path    = os.path.join(OUTPUTS_DIR, "xai_sample_indices.json")
            with open(idx_path, "w") as f:
                json.dump(xai_samples, f, indent=2)
            print(f"  XAI sample indices saved → {idx_path}")
            print(f"  Samples: {xai_samples}")
            xai_saved = True

        print(f"  Acc={metrics['accuracy']:.4f}  F1={metrics['f1_macro']:.4f}  "
              f"AUC={metrics['auc_macro_ovr']:.4f}  κ={metrics['cohen_kappa']:.4f}")

    if not all_results:
        print("No models evaluated (missing checkpoints).")
        return

    # ── Results CSV ──
    results_df = pd.DataFrame(all_results)
    # Reorder columns for readability
    col_order = ["model", "accuracy", "precision_macro", "recall_macro",
                 "f1_macro", "auc_macro_ovr", "cohen_kappa", "mcc"]
    results_df = results_df[col_order]
    csv_path   = os.path.join(OUTPUTS_DIR, "results_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    # ── Console table ──
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (test set)")
    print("=" * 80)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
