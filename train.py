"""
Task 4: Training loop for all 4 models.
Optimizer: Adam | Loss: CrossEntropyLoss (class-weighted) | Scheduler: CosineAnnealingLR
Early stopping on val loss (patience=7) | Max 30 epochs
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from model import get_model, MODEL_REGISTRY
from dataset import get_dataloaders

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
DATASET_CSV       = "/kaggle/input/aptos2019-blindness-detection/train.csv"

# ── Hyperparameters (no magic numbers elsewhere) ───────────────────────────────
TRAIN_CONFIG = {
    "lr":           1e-4,
    "weight_decay": 1e-4,
    "max_epochs":   30,
    "patience":     7,
    "t_max":        10,
    "batch_size":   32,
    "num_workers":  2,
}


# ── Early stopping ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7):
        self.patience    = patience
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True if model improved (checkpoint should be saved)."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter   = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


# ── Train / validate ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_curves(train_losses, val_losses, train_accs, val_accs,
                model_name: str, save_path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train", color="#1f77b4")
    ax1.plot(epochs, val_losses,   label="Val",   color="#ff7f0e")
    ax1.set_title(f"{model_name} – Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train", color="#1f77b4")
    ax2.plot(epochs, val_accs,   label="Val",   color="#ff7f0e")
    ax2.set_title(f"{model_name} – Accuracy"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Single-model training ──────────────────────────────────────────────────────

def train_one_model(model_name: str, train_loader, val_loader,
                    class_weights: torch.Tensor, device, config: dict) -> str:
    model     = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_max"])
    stopper   = EarlyStopping(patience=config["patience"])

    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth")
    log_dir   = os.path.join(OUTPUTS_DIR, "training_logs")
    plot_dir  = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    print(f"\n{'='*64}\nTraining: {model_name}\n{'='*64}")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improved = stopper.step(vl_loss)
        if improved:
            torch.save(model.state_dict(), ckpt_path)

        for k, v in zip(history, [epoch, tr_loss, vl_loss, tr_acc, vl_acc]):
            history[k].append(v)

        tag = " ← saved" if improved else ""
        print(f"  Ep {epoch:3d}/{config['max_epochs']}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.4f}/{vl_acc:.4f}  "
              f"({time.time()-t0:.1f}s){tag}")

        if stopper.should_stop:
            print(f"  Early stop at epoch {epoch}.")
            break

    # Save training log
    pd.DataFrame(history).to_csv(
        os.path.join(log_dir, f"{model_name}_log.csv"), index=False)

    # Save loss/accuracy curves
    save_curves(history["train_loss"], history["val_loss"],
                history["train_acc"],  history["val_acc"],
                model_name,
                os.path.join(plot_dir, f"{model_name}_loss_curve.png"))

    print(f"  Best val loss: {stopper.best_loss:.4f}  |  checkpoint → {ckpt_path}")
    return ckpt_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DR classification models")
    parser.add_argument("--models",      nargs="+", default=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--img-dir",     default=PREPROCESSED_PATH)
    parser.add_argument("--csv-path",    default=DATASET_CSV)
    parser.add_argument("--batch-size",  type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--max-epochs",  type=int, default=TRAIN_CONFIG["max_epochs"])
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR,     exist_ok=True)

    config = {**TRAIN_CONFIG, "batch_size": args.batch_size, "max_epochs": args.max_epochs}

    train_loader, val_loader, _, class_weights, _ = get_dataloaders(
        args.img_dir, args.csv_path, config["batch_size"], config["num_workers"])

    for model_name in args.models:
        train_one_model(model_name, train_loader, val_loader,
                        class_weights, device, config)

    print("\nAll models trained successfully.")


if __name__ == "__main__":
    main()
