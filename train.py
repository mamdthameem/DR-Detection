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
from sklearn.metrics import cohen_kappa_score
import argparse

from model import get_model, MODEL_REGISTRY, NUM_CLASSES
from dataset import get_dataloaders

# ── Paths ──────────────────────────────────────────────────────────────────────
CHECKPOINTS_DIR   = "/kaggle/working/checkpoints"
OUTPUTS_DIR       = "/kaggle/working/outputs"
PREPROCESSED_PATH = "/kaggle/working/data/aptos_preprocessed"
from paths import DATASET_CSV   # APTOS path auto-resolved (see paths.py)

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
    """Returns (loss, accuracy, true_labels, predicted_labels)."""
    model.eval()
    total_loss = correct = total = 0
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
    return (total_loss / total, correct / total,
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy())


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
                    class_weights: torch.Tensor, device, config: dict,
                    out_paths: dict = None) -> dict:
    """
    out_paths: None → the original locations under CHECKPOINTS_DIR / OUTPUTS_DIR.
    Otherwise {"ckpt": ..., "log": ..., "plot": optional} (run_experiment.py).
    The order of RNG-consuming steps is unchanged from the original loop; the extra
    logging (kappa, lr, timing) only reads values and draws no random numbers.
    """
    model     = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_max"])
    stopper   = EarlyStopping(patience=config["patience"])

    if out_paths is None:
        out_paths = {
            "ckpt": os.path.join(CHECKPOINTS_DIR, f"{model_name}_best.pth"),
            "log":  os.path.join(OUTPUTS_DIR, "training_logs", f"{model_name}_log.csv"),
            "plot": os.path.join(OUTPUTS_DIR, "plots", f"{model_name}_loss_curve.png"),
        }
    ckpt_path = out_paths["ckpt"]
    for path in out_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [], "val_kappa": [], "val_qwk": [],
               "learning_rate": [], "wall_clock_seconds": []}
    best_epoch = None

    print(f"\n{'='*64}\nTraining: {model_name}\n{'='*64}")

    for epoch in range(1, config["max_epochs"] + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]   # rate used by this epoch's updates
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_labels, vl_preds = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        improved = stopper.step(vl_loss)
        if improved:
            torch.save(model.state_dict(), ckpt_path)
            best_epoch = epoch

        labels = list(range(NUM_CLASSES))
        vl_kappa = cohen_kappa_score(vl_labels, vl_preds, labels=labels)
        vl_qwk   = cohen_kappa_score(vl_labels, vl_preds, labels=labels, weights="quadratic")
        seconds  = time.time() - t0
        for k, v in zip(history, [epoch, tr_loss, vl_loss, tr_acc, vl_acc,
                                  vl_kappa, vl_qwk, lr, seconds]):
            history[k].append(v)

        tag = " ← saved" if improved else ""
        print(f"  Ep {epoch:3d}/{config['max_epochs']}  "
              f"loss {tr_loss:.4f}/{vl_loss:.4f}  "
              f"acc {tr_acc:.4f}/{vl_acc:.4f}  qwk {vl_qwk:.4f}  lr {lr:.3g}  "
              f"({seconds:.1f}s){tag}")

        if stopper.should_stop:
            print(f"  Early stop at epoch {epoch}.")
            break

    # Save training log
    log_df = pd.DataFrame(history)
    log_df["is_selected_epoch"] = log_df["epoch"] == best_epoch
    log_df.to_csv(out_paths["log"], index=False)

    # Save loss/accuracy curves
    if out_paths.get("plot"):
        save_curves(history["train_loss"], history["val_loss"],
                    history["train_acc"],  history["val_acc"],
                    model_name, out_paths["plot"])

    print(f"  Best val loss: {stopper.best_loss:.4f} (epoch {best_epoch})  |  "
          f"checkpoint → {ckpt_path}")
    return {
        "ckpt_path":     ckpt_path,
        "history":       log_df,
        "best_epoch":    best_epoch,
        "best_val_loss": stopper.best_loss,
        "epochs_run":    len(log_df),
        "stopped_early": stopper.should_stop,
    }


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
