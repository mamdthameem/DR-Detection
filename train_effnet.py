"""
Standalone EfficientNet-B0 training with LAB-CLAHE preprocessing.

Trains ONLY EfficientNet-B0 using the on-the-fly preprocessing from
preprocess_effnet.py, reusing train.train_one_model (same loop / early stopping
/ curves). The checkpoint is saved to the usual path
(/kaggle/working/checkpoints/efficientnet_b0_best.pth), so you can evaluate and
explain it with the existing evaluate.py / xai_*.py scripts — just point them at
the LAB cache with --img-dir /kaggle/working/data/aptos_lab.
"""

import os
import argparse
import numpy as np
import torch

from train import train_one_model, TRAIN_CONFIG, CHECKPOINTS_DIR, OUTPUTS_DIR
from preprocess_effnet import get_dataloaders, RAW_IMG_DIR


def main():
    ap = argparse.ArgumentParser(description="Train EfficientNet-B0 (LAB-CLAHE preprocessing)")
    ap.add_argument("--img-dir",     default=RAW_IMG_DIR, help="Raw APTOS train_images dir")
    ap.add_argument("--batch-size",  type=int, default=TRAIN_CONFIG["batch_size"])
    ap.add_argument("--max-epochs",  type=int, default=TRAIN_CONFIG["max_epochs"])
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Bump up (4) since preprocessing runs on the fly")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR,     exist_ok=True)

    config = {**TRAIN_CONFIG, "batch_size": args.batch_size,
              "max_epochs": args.max_epochs, "num_workers": args.num_workers}

    train_loader, val_loader, _, class_weights, _ = get_dataloaders(
        img_dir=args.img_dir, batch_size=config["batch_size"],
        num_workers=config["num_workers"])

    train_one_model("efficientnet_b0", train_loader, val_loader,
                    class_weights, device, config)
    print("\nEfficientNet-B0 (LAB-CLAHE) training complete.")
    print("Next: evaluate.py / xai_*.py with --img-dir /kaggle/working/data/aptos_lab")


if __name__ == "__main__":
    main()
