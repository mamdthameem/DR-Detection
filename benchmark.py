"""
Efficiency measurements for the four architectures. No training and no checkpoints.

    pip install thop --no-deps          # once per session; --no-deps leaves torch untouched
    python benchmark.py                 # → /kaggle/working/efficiency.json
    python benchmark.py --cpu-images 100

Every model is a fresh APTOSModel (5-class head) with random weights; weights do not
change parameter counts, operation counts or latency.

Parameters  total and trainable, the BatchNorm share, and the candidate counts examined
            for the 4.62M EfficientNet-B0 figure (efficientnet_b0_parameter_count).
Operations  thop.profile at 1x3x224x224 reports MACs (multiply-accumulate operations);
            torch.utils.flop_counter gives an independent count. FLOPs are not measured
            separately: flops_derived_as_2x_macs only applies the FLOPs ≈ 2 × MACs convention.
Latency     batch size 1, float32, torch.no_grad, eval mode; one timed forward pass per test
            image, with the image loaded and already on the device before timing starts.
            Warmup passes are discarded. On GPU torch.cuda.synchronize() runs before and after
            every timed pass. Mean and 95th percentile over the timed passes.
Memory      GPU: torch.cuda.max_memory_allocated / max_memory_reserved over the timed loop
            (includes the weights). CPU: process peak resident set size over the timed loop
            (Linux VmHWM, reset via /proc/self/clear_refs just before the loop).
Each (architecture, device) latency and memory measurement runs in a separate process.
"""

import os
import sys
import json
import time
import argparse
import tempfile
import subprocess
from datetime import datetime, timezone
from importlib import metadata

import numpy as np
import torch
import torch.nn as nn

from model import get_model, MODEL_REGISTRY, IMG_SIZE
from dataset import PREPROCESSED_PATH
from paths import DATASET_CSV
from experiment_config import WORK_DIR, N_TEST, write_json

EFFICIENCY_JSON = os.path.join(WORK_DIR, "efficiency.json")
WARMUP_PASSES   = 20
PAPER_B0_PARAMS = 4.62   # millions, as quoted in the paper

CONVENTIONS = {
    "parameters": "sum of p.numel() over model.parameters(); trainable = requires_grad. "
                  "BatchNorm running statistics are buffers, not parameters.",
    "thop": "thop.profile, batch size 1, 3x224x224, eval mode. Counts MACs (multiply-"
            "accumulate operations) of modules that have a thop rule; modules without a rule "
            "contribute neither operations nor parameters (listed per model). "
            "flops_derived_as_2x_macs = 2 * macs, a convention, not a separate measurement.",
    "torch_flop_counter": "torch.utils.flop_counter.FlopCounterMode, forward pass, batch size 1, "
                          "3x224x224. Counts convolution and matrix-multiplication operators as "
                          "multiply-adds (a matmul as m*n*p, no factor 2, no bias additions); "
                          "normalisation, activations, pooling and element-wise ops are not counted.",
    "latency": "batch size 1, float32, torch.no_grad, eval mode; model and input already on the "
               "device; warmup passes discarded; torch.cuda.synchronize() before and after each "
               "timed GPU pass; time.perf_counter; milliseconds per image.",
    "gpu_memory": "torch.cuda.max_memory_allocated and max_memory_reserved over the timed loop, "
                  "after resetting the peak counters; includes model weights.",
    "cpu_memory": "peak resident set size of the measuring process over the timed loop "
                  "(VmHWM after writing 5 to /proc/self/clear_refs); rss_at_start_mb and "
                  "rss_after_model_mb give the baseline.",
    "input_resolution": "224x224 for every model, the resolution used in training "
                        "(including InceptionV3, whose native resolution is 299x299).",
}


# ── Parameters and operation counts ────────────────────────────────────────────

def _count(module: nn.Module):
    params = list(module.parameters())
    return (sum(p.numel() for p in params),
            sum(p.numel() for p in params if p.requires_grad))


def _batchnorm(module: nn.Module):
    layers = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    return len(layers), sum(p.numel() for m in layers for p in m.parameters(recurse=False))


def _thop_version():
    for dist in ("thop", "ultralytics-thop"):
        try:
            return f"{dist} {metadata.version(dist)}"
        except metadata.PackageNotFoundError:
            continue
    return None


def thop_operations(model_name: str) -> dict:
    try:
        import thop
        from thop.profile import register_hooks
    except ImportError as exc:
        return {"available": False,
                "error": f"{exc}. Install with: pip install thop --no-deps"}
    model = get_model(model_name, pretrained=False).eval()
    without_rule = {}
    for module in model.modules():
        own = sum(p.numel() for p in module.parameters(recurse=False))
        is_leaf = next(module.children(), None) is None
        if type(module) not in register_hooks and (is_leaf or own):
            key = f"{type(module).__module__}.{type(module).__name__}"
            entry = without_rule.setdefault(key, {"modules": 0, "own_parameters": 0})
            entry["modules"] += 1
            entry["own_parameters"] += own
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        with torch.no_grad():
            macs, params = thop.profile(model, inputs=(x,), verbose=False)
    except Exception as exc:
        return {"available": True, "package": _thop_version(),
                "error": f"{type(exc).__name__}: {exc}",
                "modules_without_thop_rule": without_rule}
    return {
        "available":                   True,
        "package":                     _thop_version(),
        "input_shape":                 [1, 3, IMG_SIZE, IMG_SIZE],
        "macs":                        float(macs),
        "gmacs":                       float(macs) / 1e9,
        "flops_derived_as_2x_macs":    2 * float(macs),
        "params_reported_by_thop":     int(params),
        "modules_without_thop_rule":   without_rule,
    }


def torch_flop_count(model_name: str) -> dict:
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError as exc:
        return {"available": False, "error": str(exc)}
    model = get_model(model_name, pretrained=False).eval()
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), counter:
            model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    by_operator = {str(op): int(n) for op, n in counter.get_flop_counts().get("Global", {}).items()}
    total = int(counter.get_total_flops())
    return {"available": True, "total": total, "giga": total / 1e9, "by_operator": by_operator}


def static_measurements(model_name: str) -> dict:
    model = get_model(model_name, pretrained=False).eval()
    total, trainable = _count(model)
    n_bn, bn_params = _batchnorm(model)
    return {
        "timm_identifier":                   MODEL_REGISTRY[model_name],
        "params_total":                      total,
        "params_trainable":                  trainable,
        "params_backbone":                   _count(model.backbone)[0],
        "params_head":                       _count(model.classifier)[0],
        "head_in_features":                  model.classifier[1].in_features,
        "batchnorm_layers":                  n_bn,
        "params_batchnorm_affine":           bn_params,
        "params_excluding_batchnorm_affine": total - bn_params,
        "state_dict_numel_including_buffers": sum(v.numel() for v in model.state_dict().values()),
        "thop":                              thop_operations(model_name),
        "torch_flop_counter":                torch_flop_count(model_name),
    }


def b0_parameter_candidates(s: dict) -> dict:
    """Where could 4.62M come from? Every count below is computed, none is assumed."""
    rows = []

    def add(label, value, how):
        millions = round(value / 1e6, 2)
        rows.append({"candidate": label, "params": int(value), "millions_2dp": millions,
                     "rounds_to_paper_figure": millions == PAPER_B0_PARAMS, "how": how})

    feat = s["head_in_features"]
    add("APTOSModel total (the trained model)", s["params_total"], "model.parameters()")
    add("APTOSModel trainable", s["params_trainable"], "requires_grad parameters")
    add("total minus BatchNorm weight and bias", s["params_excluding_batchnorm_affine"],
        "BatchNorm affine parameters removed")
    if "params_reported_by_thop" in s["thop"]:
        add("thop.profile() parameter count", s["thop"]["params_reported_by_thop"],
            "second value returned by thop.profile at 1x3x224x224")
    add("state_dict incl. BatchNorm running statistics",
        s["state_dict_numel_including_buffers"], "numel over model.state_dict()")
    add("backbone only (timm efficientnet_b0, num_classes=0)", s["params_backbone"],
        "model.backbone")
    add("head only (trainable part of a frozen-backbone variant)", s["params_head"],
        "model.classifier")
    add(f"backbone + single Linear({feat}->5)", s["params_backbone"] + feat * 5 + 5,
        "timm's own classifier with num_classes=5")

    feature_dims, errors = {}, {}
    try:
        import timm
        add("timm efficientnet_b0 with its 1000-class classifier",
            _count(timm.create_model("efficientnet_b0", pretrained=False))[0],
            "timm.create_model('efficientnet_b0')")
        backbone = timm.create_model("efficientnet_b0", pretrained=False,
                                     num_classes=0, global_pool="avg").eval()
        with torch.no_grad():
            feature_dims = {str(r): int(backbone(torch.zeros(1, 3, r, r)).shape[1])
                            for r in (224, 240, 299)}
    except Exception as exc:
        errors["timm"] = str(exc)
    try:
        import torchvision
        tv = torchvision.models.efficientnet_b0(weights=None)
        add("torchvision efficientnet_b0 (1000 classes)", _count(tv)[0],
            "torchvision.models.efficientnet_b0(weights=None)")
        add("torchvision efficientnet_b0 features + the pipeline head",
            _count(tv.features)[0] + s["params_head"], "tv.features + model.classifier")
    except Exception as exc:
        errors["torchvision"] = str(exc)

    return {
        "paper_figure_millions": PAPER_B0_PARAMS,
        "candidates": rows,
        "candidates_rounding_to_paper_figure": [r["candidate"] for r in rows
                                                if r["rounds_to_paper_figure"]],
        "pooled_feature_size_by_input_resolution": feature_dims,
        "errors": errors,
    }


# ── Latency and memory (worker process) ────────────────────────────────────────

def _proc_status_mb(field):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _rss_mb():
    value = _proc_status_mb("VmRSS")
    if value is None:
        try:
            import psutil
            value = psutil.Process().memory_info().rss / 2**20
        except ImportError:
            pass
    return value


def _peak_rss_mb():
    value = _proc_status_mb("VmHWM")
    if value is None:
        try:
            import psutil
            info = psutil.Process().memory_info()
            value = getattr(info, "peak_wset", None)
            value = value / 2**20 if value else None
        except ImportError:
            pass
    return value


def _reset_peak_rss() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


class TestImages:
    """Preprocessed test images in split order, or one fixed random tensor if unavailable."""

    def __init__(self, img_dir: str, csv_path: str):
        self.paths, self.source = [], None
        try:
            from dataset import get_splits, get_transforms
            _, _, test_df = get_splits(csv_path)
            paths = [os.path.join(img_dir, f"{i}.png") for i in test_df["id_code"]]
            if paths and all(os.path.exists(p) for p in paths):
                self.paths, self.transform = paths, get_transforms(is_train=False)
                self.source = f"{len(paths)} preprocessed test images from {img_dir}"
        except Exception as exc:
            self.source = f"synthetic input (test images unavailable: {type(exc).__name__}: {exc})"
        if not self.paths:
            self.source = self.source or f"synthetic input (no preprocessed test images in {img_dir})"
            self.synthetic = torch.randn(1, 3, IMG_SIZE, IMG_SIZE,
                                         generator=torch.Generator().manual_seed(0))

    def get(self, i: int) -> torch.Tensor:
        if not self.paths:
            return self.synthetic
        from PIL import Image
        img = Image.open(self.paths[i % len(self.paths)]).convert("RGB")
        return self.transform(img).unsqueeze(0)


def latency_worker(args) -> int:
    import gc
    from environment_info import _cpu_model

    device = torch.device(args.device)
    cuda = device.type == "cuda"
    rss_start = _rss_mb()
    model = get_model(args.model, pretrained=False).eval().to(device)
    gc.collect()
    rss_model = _rss_mb()
    images = TestImages(args.img_dir, args.csv_path)

    times = []
    with torch.no_grad():
        warm = images.get(0).to(device)
        for _ in range(args.warmup):
            model(warm)
        allocated_before = None
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
        peak_reset = _reset_peak_rss()
        for i in range(args.n_images):
            x = images.get(i).to(device)
            if cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model(x)
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

    ms = np.asarray(times) * 1e3
    result = {
        "device":        str(device),
        "device_name":   torch.cuda.get_device_name(device) if cuda else _cpu_model(),
        "n_timed_images": int(len(ms)),
        "warmup_passes_discarded": args.warmup,
        "input_source":  images.source,
        "torch_num_threads": torch.get_num_threads(),
        "latency_ms_per_image": {
            "mean":   float(ms.mean()),
            "p95":    float(np.percentile(ms, 95)),
            "median": float(np.median(ms)),
            "std":    float(ms.std(ddof=1)) if len(ms) > 1 else None,
            "min":    float(ms.min()),
            "max":    float(ms.max()),
        },
        "cpu_memory": {
            "rss_at_start_mb":               rss_start,
            "rss_after_model_mb":            rss_model,
            "peak_rss_during_timed_loop_mb": _peak_rss_mb(),
            "peak_counter_reset_before_loop": peak_reset,
        },
    }
    if cuda:
        result["gpu_memory"] = {
            "allocated_before_loop_mb":      allocated_before / 2**20,
            "peak_allocated_during_loop_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_during_loop_mb":  torch.cuda.max_memory_reserved(device) / 2**20,
        }
    write_json(args.worker_out, result)
    return 0


def run_worker(model_name: str, device: str, n_images: int, args) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "result.json")
        cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
               "--model", model_name, "--device", device, "--n-images", str(n_images),
               "--warmup", str(args.warmup), "--img-dir", args.img_dir,
               "--csv-path", args.csv_path, "--worker-out", out]
        rc = subprocess.run(cmd).returncode
        if rc != 0 or not os.path.exists(out):
            return {"error": f"measurement process exited with code {rc}"}
        with open(out) as f:
            return json.load(f)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Parameter, operation, latency and memory benchmark")
    ap.add_argument("--out",        default=EFFICIENCY_JSON)
    ap.add_argument("--models",     nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY))
    ap.add_argument("--gpu-images", type=int, default=N_TEST)
    ap.add_argument("--cpu-images", type=int, default=N_TEST)
    ap.add_argument("--warmup",     type=int, default=WARMUP_PASSES)
    ap.add_argument("--skip-gpu",   action="store_true")
    ap.add_argument("--skip-cpu",   action="store_true")
    ap.add_argument("--img-dir",    default=PREPROCESSED_PATH)
    ap.add_argument("--csv-path",   default=DATASET_CSV)
    ap.add_argument("--worker",     action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model",      help=argparse.SUPPRESS)
    ap.add_argument("--device",     help=argparse.SUPPRESS)
    ap.add_argument("--n-images",   type=int, help=argparse.SUPPRESS)
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        return latency_worker(args)
    sys.stdout.reconfigure(line_buffering=True)   # keep order with the workers' output

    results = {
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "conventions": CONVENTIONS,
        "settings": {"batch_size": 1, "dtype": "float32", "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
                     "warmup_passes": args.warmup, "gpu_images": args.gpu_images,
                     "cpu_images": args.cpu_images, "cudnn_benchmark": False},
        "models": {},
    }
    for name in args.models:
        s = static_measurements(name)
        results["models"][name] = {"static": s}
        thop_text = (f"{s['thop']['gmacs']:.3f} GMACs" if "gmacs" in s["thop"]
                     else f"thop: {s['thop'].get('error', 'unavailable')}")
        print(f"[{name}] params {s['params_total']:,} (trainable {s['params_trainable']:,}), "
              f"BatchNorm layers {s['batchnorm_layers']}, {thop_text}")
    if "efficientnet_b0" in results["models"]:
        investigation = b0_parameter_candidates(results["models"]["efficientnet_b0"]["static"])
        results["efficientnet_b0_parameter_count"] = investigation
        print(f"EfficientNet-B0 counts rounding to {PAPER_B0_PARAMS}M: "
              f"{investigation['candidates_rounding_to_paper_figure']}")
    write_json(args.out, results)

    gpu_available = torch.cuda.is_available()
    for name in args.models:
        for key, device, n_images, skip in (
                ("gpu", "cuda", args.gpu_images, args.skip_gpu or not gpu_available),
                ("cpu", "cpu",  args.cpu_images, args.skip_cpu)):
            if skip:
                reason = "no CUDA device" if key == "gpu" and not gpu_available else "skipped by flag"
                results["models"][name][key] = {"skipped": reason}
                continue
            print(f"[{name}] {key} latency and memory over {n_images} images ...")
            measured = run_worker(name, device, n_images, args)
            results["models"][name][key] = measured
            if "latency_ms_per_image" in measured:
                lat = measured["latency_ms_per_image"]
                print(f"  mean {lat['mean']:.2f} ms, p95 {lat['p95']:.2f} ms")
            write_json(args.out, results)

    from environment_info import collect_environment
    results["environment"] = collect_environment()
    write_json(args.out, results)
    print(f"Efficiency results → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
