"""
Learning-rate schedule and BatchNorm checks (no training).

    python verify_lr_and_bn.py      # → /kaggle/working/lr_schedule.csv, batchnorm_check.json

1. The realised learning rate per epoch of train.py's optimizer and scheduler, built from
   TRAIN_CONFIG exactly as train_one_model builds them and read from the optimizer the way
   epoch_log.csv records it: the value in effect during an epoch's updates, followed by one
   scheduler.step() per epoch.
2. Which of the four backbones contain BatchNorm layers: the timm models the pipeline trains,
   and their torchvision counterparts.
"""

import os
import math
import argparse

import pandas as pd
import torch
import torch.nn as nn

from train import TRAIN_CONFIG
from model import get_model, MODEL_REGISTRY
from experiment_config import WORK_DIR, write_json


def realised_lr_schedule(config: dict, epochs: int) -> pd.DataFrame:
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["t_max"])
    rows = []
    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        closed_form = 0.5 * config["lr"] * (1 + math.cos(math.pi * (epoch - 1) / config["t_max"]))
        rows.append({"epoch": epoch, "learning_rate": lr, "learning_rate_repr": repr(lr),
                     "closed_form_cosine": closed_form,
                     "abs_diff_from_closed_form": abs(lr - closed_form),
                     "exactly_zero": lr == 0.0, "exactly_base_lr": lr == config["lr"]})
        optimizer.step()   # no gradients, so nothing changes; keeps the step-order check quiet
        scheduler.step()
    return pd.DataFrame(rows)


def norm_summary(model: nn.Module) -> dict:
    bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    other = [m for m in model.modules()
             if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.modules.instancenorm._InstanceNorm))]
    return {
        "batchnorm_layers":        len(bn),
        "batchnorm_types":         sorted({type(m).__name__ for m in bn}),
        "batchnorm_affine_params": sum(p.numel() for m in bn for p in m.parameters(recurse=False)),
        "other_norm_layers":       len(other),
    }


def batchnorm_check() -> dict:
    result = {"pipeline_builds_models_with": "timm (model.py: timm.create_model)",
              "timm": {}, "torchvision": {}}
    for name, identifier in MODEL_REGISTRY.items():
        result["timm"][name] = {"identifier": identifier,
                                **norm_summary(get_model(name, pretrained=False))}
    try:
        import torchvision.models as tvm
        builders = {
            "efficientnet_b0": lambda: tvm.efficientnet_b0(weights=None),
            "resnet50":        lambda: tvm.resnet50(weights=None),
            "vgg16":           lambda: tvm.vgg16(weights=None),
            "inceptionv3":     lambda: tvm.inception_v3(weights=None, aux_logits=False,
                                                        init_weights=False),
        }
        for name, build in builders.items():
            result["torchvision"][name] = norm_summary(build())
    except Exception as exc:
        result["torchvision"] = {"error": f"{type(exc).__name__}: {exc}"}

    for library in ("timm", "torchvision"):
        models = result[library]
        if "vgg16" in models and "batchnorm_layers" in models["vgg16"]:
            result[f"{library}_vgg16_has_no_batchnorm"] = models["vgg16"]["batchnorm_layers"] == 0
            result[f"{library}_other_three_have_batchnorm"] = all(
                models[n]["batchnorm_layers"] > 0 for n in models if n != "vgg16")
    return result


def main():
    ap = argparse.ArgumentParser(description="Learning-rate schedule and BatchNorm checks")
    ap.add_argument("--out-dir", default=WORK_DIR)
    ap.add_argument("--epochs",  type=int, default=TRAIN_CONFIG["max_epochs"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    schedule = realised_lr_schedule(TRAIN_CONFIG, args.epochs)
    schedule_path = os.path.join(args.out_dir, "lr_schedule.csv")
    schedule.to_csv(schedule_path, index=False)
    print(f"Realised schedule: Adam(lr={TRAIN_CONFIG['lr']}), "
          f"CosineAnnealingLR(T_max={TRAIN_CONFIG['t_max']}, eta_min=0), torch {torch.__version__}")
    for _, r in schedule.iterrows():
        print(f"  epoch {int(r['epoch']):2d}  lr {r['learning_rate_repr']}")
    by_epoch = schedule.set_index("epoch")
    for epoch in (11, 21):
        if epoch in by_epoch.index:
            r = by_epoch.loc[epoch]
            print(f"epoch {epoch}: lr = {r['learning_rate_repr']}  exactly 0.0: {r['exactly_zero']}  "
                  f"exactly {TRAIN_CONFIG['lr']!r}: {r['exactly_base_lr']}  "
                  f"|lr - {TRAIN_CONFIG['lr']!r}| = {abs(r['learning_rate'] - TRAIN_CONFIG['lr']):.3g}")
    print(f"→ {schedule_path}")

    bn = batchnorm_check()
    bn_path = os.path.join(args.out_dir, "batchnorm_check.json")
    write_json(bn_path, bn)
    for library in ("timm", "torchvision"):
        for name, info in bn[library].items():
            if isinstance(info, dict):
                print(f"  {library:<11} {name:<16} BatchNorm layers: {info.get('batchnorm_layers')}")
    print(f"timm: vgg16 has no BatchNorm = {bn.get('timm_vgg16_has_no_batchnorm')}, "
          f"other three have BatchNorm = {bn.get('timm_other_three_have_batchnorm')}")
    print(f"→ {bn_path}")


if __name__ == "__main__":
    main()
