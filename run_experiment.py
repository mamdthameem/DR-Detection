"""
One run of the multi-seed suite → one self-contained artefact directory.

    python run_experiment.py --config b0_baseline --seed 42
    python run_experiment.py --config b0_baseline --seed 42 --max-epochs 2 \
                             --runs-root /kaggle/working/runs_smoke

Writes {runs_root}/{config}__seed{seed}/:
  probs.npy logits.npy labels.npy indices.npy preds.npy   test set, rows aligned
  val_predictions.npz   the same five arrays for the validation set
  best_model.pth        checkpoint selected by early stopping
  epoch_log.csv         per epoch: losses, accuracies, val kappa (unweighted + quadratic),
                        optimizer learning rate, seconds, is_selected_epoch
  config.json           config, seeds, hyperparameters, augmentation, determinism, environment
  metrics.json          written last; its presence marks the run as complete

indices.npy holds positional indices into the SPLIT_SEED test split (test_df.iloc[i]),
the same indexing as XAI_SAMPLE_INDICES.

Reproducibility — the random-number path is identical to the original train.py /
train_effnet.py runs, so train_seed 42 can reproduce them:
  * random, numpy, torch and torch.cuda are seeded from train_seed right before the data
    loaders are built. The originals seeded torch and numpy at that point; seeding random
    and torch.cuda again draws nothing and changes no stream.
  * DataLoader shuffling and worker seeds come from the global torch RNG, as in the
    originals. --seeded-generator passes a seeded torch.Generator instead; that changes
    the shuffle order and augmentation draws, so seed 42 will no longer match.
  * cudnn.benchmark = False and cudnn.deterministic = False are the PyTorch defaults the
    originals ran with. --cudnn-deterministic restricts cuDNN to deterministic kernels,
    which can change the numbers.
  * worker_init_fn seeds numpy and random inside each worker from the torch seed PyTorch
    already gave that worker. No transform here draws from numpy or random.
  * torch.use_deterministic_algorithms is never enabled
    (environment_info.py --probe records whether it would raise).
"""

import os
import sys
import time
import random
import hashlib
import argparse
from datetime import datetime, timezone

import numpy as np
import torch

import dataset
import preprocess_effnet
from paths import DATASET_CSV
from train import train_one_model, TRAIN_CONFIG
from model import get_model, MODEL_REGISTRY, IMG_SIZE
from evaluate import run_inference_full, compute_metrics, find_xai_samples
from analyze import compute_run_metrics, legacy_metrics
from environment_info import collect_environment, source_provenance
from experiment_config import (RUNS_ROOT, PROBE_JSON, SPLIT_SEED, N_TEST, CONFIGS,
                               PREPROCESSING, XAI_SAMPLE_INDICES, run_name, write_json)


def seed_worker(worker_id: int) -> None:
    """Seed numpy/random inside a DataLoader worker from the torch seed PyTorch gave it."""
    worker_seed = torch.utils.data.get_worker_info().seed % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seeds(train_seed: int, cudnn_deterministic: bool) -> None:
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = cudnn_deterministic


def get_transform_module(config: dict):
    return preprocess_effnet if config["preprocessing"] == "lab_clahe" else dataset


def images_for(config: dict, args) -> str:
    return {"baseline":        args.img_dir,
            "lab_clahe":       args.raw_dir,
            "lab_clahe_cache": args.lab_dir}[config["preprocessing"]]


def build_loaders(config: dict, args, generator):
    common = dict(csv_path=args.csv_path, batch_size=TRAIN_CONFIG["batch_size"],
                  num_workers=config["num_workers"], seed=SPLIT_SEED,
                  worker_init_fn=seed_worker, generator=generator)
    if config["preprocessing"] == "lab_clahe":
        return preprocess_effnet.get_dataloaders(img_dir=args.raw_dir, **common)
    return dataset.get_dataloaders(img_dir=images_for(config, args), **common)


def check_lab_cache(lab_dir: str, raw_dir: str, splits, n_compare: int = 20) -> dict:
    """Every image must be in the LAB-CLAHE cache, with the pixels of on-the-fly preprocessing."""
    from PIL import Image
    ids = [id_code for df in splits for id_code in df["id_code"]]
    missing = [i for i in ids if not os.path.exists(os.path.join(lab_dir, i + ".png"))]
    if missing:
        raise RuntimeError(f"{len(missing)} of {len(ids)} images missing from {lab_dir} "
                           f"(e.g. {missing[:3]}). Build the cache first: "
                           "python preprocess_effnet.py --skip-visualize")
    compared = list(splits[2]["id_code"][:n_compare])
    for id_code in compared:
        cached = np.array(Image.open(os.path.join(lab_dir, id_code + ".png")).convert("RGB"))
        fresh = preprocess_effnet.preprocess_image(preprocess_effnet._read_raw(raw_dir, id_code))
        if not np.array_equal(cached, fresh):
            raise RuntimeError(f"{lab_dir}/{id_code}.png differs from on-the-fly LAB-CLAHE "
                               "preprocessing of the raw image; rebuild the cache")
    print(f"LAB-CLAHE cache OK: {len(ids)} images present; {len(compared)} test images "
          "pixel-identical to on-the-fly preprocessing")
    return {"cache_dir": lab_dir, "images_present": len(ids),
            "test_images_pixel_identical_to_on_the_fly": len(compared)}


def _ids_sha256(df) -> str:
    return hashlib.sha256("\n".join(df["id_code"]).encode()).hexdigest()


def _read_probe():
    if not os.path.exists(PROBE_JSON):
        return "not run (python environment_info.py --probe)"
    import json
    with open(PROBE_JSON) as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One multi-seed suite run with full artefacts")
    ap.add_argument("--config",     required=True, choices=list(CONFIGS))
    ap.add_argument("--seed",       required=True, type=int, help="train_seed")
    ap.add_argument("--runs-root",  default=RUNS_ROOT)
    ap.add_argument("--max-epochs", type=int, default=TRAIN_CONFIG["max_epochs"],
                    help="Smoke runs only; real runs keep the default")
    ap.add_argument("--img-dir",    default=dataset.PREPROCESSED_PATH,
                    help="Baseline preprocessed PNGs (preprocess.py output)")
    ap.add_argument("--raw-dir",    default=preprocess_effnet.RAW_IMG_DIR,
                    help="Raw APTOS train_images (b0_clahe preprocesses on the fly)")
    ap.add_argument("--lab-dir",    default=preprocess_effnet.LAB_CACHE,
                    help="LAB-CLAHE PNG cache (b0_clahe_matched)")
    ap.add_argument("--csv-path",   default=DATASET_CSV)
    ap.add_argument("--cudnn-deterministic", action="store_true",
                    help="Deterministic cuDNN kernels (originals: off)")
    ap.add_argument("--seeded-generator",    action="store_true",
                    help="Seeded DataLoader generator (originals: global RNG)")
    args = ap.parse_args(argv)

    t_start = time.time()
    config  = CONFIGS[args.config]
    name    = run_name(args.config, args.seed)
    run_dir = os.path.join(args.runs_root, name)
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print(f"[skip] {name}: metrics.json already present")
        return 0
    os.makedirs(run_dir, exist_ok=True)

    # ── Everything below up to set_seeds() draws no random numbers ─────────────
    train_df, val_df, test_df = dataset.get_splits(args.csv_path, seed=SPLIT_SEED)
    if len(test_df) != N_TEST:
        raise RuntimeError(f"Test split has {len(test_df)} images, expected {N_TEST}. "
                           f"Wrong train.csv? ({args.csv_path})")

    xai_detail = {str(grade): {"index": idx,
                               "grade_at_index": int(test_df.iloc[idx]["diagnosis"]),
                               "id_code": test_df.iloc[idx]["id_code"]}
                  for grade, idx in XAI_SAMPLE_INDICES.items()}
    xai_valid = all(d["grade_at_index"] == int(g) for g, d in xai_detail.items())
    if not xai_valid:
        print(f"[WARN] XAI_SAMPLE_INDICES do not match their grades in this split: {xai_detail}")

    cache_check = None
    if config["preprocessing"] == "lab_clahe_cache":
        cache_check = check_lab_cache(args.lab_dir, args.raw_dir, (train_df, val_df, test_df))

    hparams = {**TRAIN_CONFIG, "max_epochs": args.max_epochs,
               "num_workers": config["num_workers"]}
    transforms_module = get_transform_module(config)
    image_dir = images_for(config, args)

    write_json(os.path.join(run_dir, "config.json"), {
        "run_name":        name,
        "config_name":     args.config,
        "model_name":      config["model_name"],
        "timm_identifier": MODEL_REGISTRY[config["model_name"]],
        "original_script": config["original_script"],
        "train_seed":      args.seed,
        "split_seed":      SPLIT_SEED,
        "smoke_run":       args.max_epochs != TRAIN_CONFIG["max_epochs"],
        "preprocessing": {
            "variant":     config["preprocessing"],
            "description": PREPROCESSING[config["preprocessing"]],
            "image_dir":   image_dir,
            **({"cache_check": cache_check} if cache_check else {}),
        },
        "hyperparameters": {
            **hparams,
            "input_size":     IMG_SIZE,
            "optimizer":      "Adam",
            "scheduler":      f"CosineAnnealingLR(T_max={hparams['t_max']}, eta_min=0), "
                              "step() once per epoch after validation",
            "early_stopping": f"validation loss, patience {hparams['patience']}; the "
                              "checkpoint with the lowest validation loss is kept",
            "loss":           "CrossEntropyLoss(weight=inverse-frequency class weights "
                              "normalised to sum to 5)",
            "class_weights":  dataset.get_class_weights(train_df).tolist(),
        },
        "augmentation": {
            "train": str(transforms_module.get_transforms(True)),
            "eval":  str(transforms_module.get_transforms(False)),
        },
        "split": {
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "train_ids_sha256": _ids_sha256(train_df),
            "val_ids_sha256":   _ids_sha256(val_df),
            "test_ids_sha256":  _ids_sha256(test_df),
            "test_id_codes":    test_df["id_code"].tolist(),
        },
        "determinism": {
            "seeded_from_train_seed":     ["random", "numpy", "torch", "torch.cuda"],
            "cudnn_benchmark":            False,
            "cudnn_deterministic":        args.cudnn_deterministic,
            "dataloader_worker_init_fn":  "seed_worker: numpy and random <- worker torch seed",
            "dataloader_generator":       ("torch.Generator seeded with train_seed"
                                           if args.seeded_generator else
                                           "None: global torch RNG, as in the original runs"),
            "torch_use_deterministic_algorithms": False,
            "use_deterministic_algorithms_probe": _read_probe(),
        },
        "xai_sample_indices": {
            "indices":         XAI_SAMPLE_INDICES,
            "grades_verified": xai_valid,
            "detail":          xai_detail,
        },
        "environment": collect_environment(),
        "source":      source_provenance(),
        "created_at":  datetime.now(timezone.utc).isoformat(),
    })

    # ── Seeded section: same order as the original train.py main() ─────────────
    set_seeds(args.seed, args.cudnn_deterministic)
    generator = None
    if args.seeded_generator:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run {name}  |  device {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader, val_loader, test_loader, class_weights, _ = build_loaders(config, args, generator)

    t_train = time.time()
    result = train_one_model(config["model_name"], train_loader, val_loader,
                             class_weights, device, hparams,
                             out_paths={"ckpt": os.path.join(run_dir, "best_model.pth"),
                                        "log":  os.path.join(run_dir, "epoch_log.csv")})
    train_seconds = time.time() - t_train

    # ── Evaluate the selected checkpoint as evaluate.py did: fresh model, weights from disk
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = get_model(config["model_name"], pretrained=False).to(device)
    model.load_state_dict(torch.load(result["ckpt_path"], map_location=device))

    labels, preds, probs, logits = run_inference_full(model, test_loader, device)
    if not np.array_equal(labels, test_df["diagnosis"].to_numpy()):
        raise RuntimeError("Test inference order does not match the test split")
    v_labels, v_preds, v_probs, v_logits = run_inference_full(model, val_loader, device)

    np.save(os.path.join(run_dir, "probs.npy"),   probs)
    np.save(os.path.join(run_dir, "logits.npy"),  logits)
    np.save(os.path.join(run_dir, "labels.npy"),  labels)
    np.save(os.path.join(run_dir, "indices.npy"), np.arange(len(test_df)))
    np.save(os.path.join(run_dir, "preds.npy"),   preds)
    np.savez(os.path.join(run_dir, "val_predictions.npz"),
             probs=v_probs, logits=v_logits, labels=v_labels, preds=v_preds,
             indices=np.arange(len(val_df)))

    legacy      = legacy_metrics(labels, preds, probs)
    legacy_eval = compute_metrics(labels, preds, probs)
    test_metrics = compute_run_metrics(labels, preds, probs)

    write_json(os.path.join(run_dir, "metrics.json"), {
        "run_name":    name,
        "config_name": args.config,
        "train_seed":  args.seed,
        "n_test":      len(labels),
        "test":        test_metrics,
        "val":         compute_run_metrics(v_labels, v_preds, v_probs),
        "legacy_metrics": legacy,
        "legacy_metrics_match_evaluate_py": all(legacy[k] == legacy_eval[k] for k in legacy_eval),
        "xai_samples_by_original_rule":     find_xai_samples(labels, preds),
        "training": {
            "selected_epoch": result["best_epoch"],
            "epochs_run":     result["epochs_run"],
            "max_epochs":     args.max_epochs,
            "stopped_early":  result["stopped_early"],
            "best_val_loss":  result["best_val_loss"],
            "train_seconds":  train_seconds,
        },
        "total_seconds": time.time() - t_start,
        "completed_at":  datetime.now(timezone.utc).isoformat(),
    })

    print(f"Done {name}: selected epoch {result['best_epoch']}/{result['epochs_run']}  "
          f"acc {test_metrics['accuracy']:.4f}  "
          f"kappa {test_metrics['kappa_unweighted']:.4f}  "
          f"QWK {test_metrics['kappa_quadratic_weighted']:.4f}  → {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
