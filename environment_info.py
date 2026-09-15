"""
Runtime environment capture for the reproducibility table.

    python environment_info.py            # write /kaggle/working/environment.json (once)
    python environment_info.py --probe    # also test torch.use_deterministic_algorithms(True)

Every value is read from the running system; nothing is hard-coded. If environment.json
already exists and the current environment differs (e.g. a later Kaggle session got a
different GPU), the file is left alone, a warning is printed and the current environment
is written next to it as environment_mismatch_<timestamp>.json.
"""

import os
import sys
import glob
import hashlib
import platform
import argparse
import subprocess
from datetime import datetime, timezone
from importlib import metadata

from experiment_config import ENVIRONMENT_JSON, PROBE_JSON, write_json

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Versions ───────────────────────────────────────────────────────────────────

def _module_version(module_name: str):
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _dist_version(*dist_names):
    for name in dist_names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def collect_versions() -> dict:
    import numpy
    import torch
    return {
        "python":        platform.python_version(),
        "python_build":  sys.version,
        "platform":      platform.platform(),
        "torch":         torch.__version__,
        "torchvision":   _module_version("torchvision"),
        "cuda_build":    torch.version.cuda,
        "cudnn":         torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "nvidia_driver": _run(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"]),
        "numpy":         numpy.__version__,
        "opencv":        _module_version("cv2"),
        "thop":          _dist_version("thop", "ultralytics-thop"),
        "timm":          _module_version("timm"),
        "scikit_learn":  _module_version("sklearn"),
        "pandas":        _module_version("pandas"),
        "pillow":        _module_version("PIL"),
        "scipy":         _module_version("scipy"),
    }


# ── Hardware ───────────────────────────────────────────────────────────────────

def _cpuinfo_lines():
    try:
        with open("/proc/cpuinfo") as f:
            return f.read().splitlines()
    except OSError:
        return []


def _cpu_model():
    for line in _cpuinfo_lines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        pass
    cores, physical_id = set(), None
    for line in _cpuinfo_lines():
        if line.startswith("physical id"):
            physical_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            cores.add((physical_id, line.split(":", 1)[1].strip()))
    return len(cores) or None


def _ram_total_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        return None


def collect_hardware() -> dict:
    import torch
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index":              i,
                "name":               props.name,
                "total_memory_bytes": props.total_memory,
                "total_memory_gib":   round(props.total_memory / 2**30, 2),
                "compute_capability": f"{props.major}.{props.minor}",
            })
    ram = _ram_total_bytes()
    return {
        "gpus":                   gpus,
        "cpu_model":              _cpu_model(),
        "cpu_logical_cores":      os.cpu_count(),
        "cpu_physical_cores":     _physical_cores(),
        "cpu_cores_usable_by_process": (len(os.sched_getaffinity(0))
                                        if hasattr(os, "sched_getaffinity") else None),
        "ram_total_bytes":        ram,
        "ram_total_gib":          round(ram / 2**30, 2) if ram else None,
    }


def collect_environment() -> dict:
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "versions":     collect_versions(),
        "hardware":     collect_hardware(),
    }


def _comparable(env: dict) -> dict:
    hw = env.get("hardware", {})
    return {"versions": env.get("versions"),
            "gpus": [g.get("name") for g in hw.get("gpus", [])],
            "cpu_model": hw.get("cpu_model")}


def write_environment_json(path: str = ENVIRONMENT_JSON) -> dict:
    env = collect_environment()
    if not os.path.exists(path):
        write_json(path, env)
        print(f"Environment written → {path}")
        return env
    import json
    with open(path) as f:
        saved = json.load(f)
    old, new = _comparable(saved), _comparable(env)
    if old == new:
        print(f"Environment unchanged (matches {path})")
        return env
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mismatch_path = os.path.join(os.path.dirname(path), f"environment_mismatch_{stamp}.json")
    write_json(mismatch_path, env)
    print("!" * 72)
    print(f"WARNING: this session's environment differs from {path}")
    for key in new:
        if old.get(key) != new.get(key):
            print(f"  {key}:\n    saved:   {old.get(key)}\n    current: {new.get(key)}")
    print(f"Current environment written → {mismatch_path}")
    print("!" * 72)
    return env


# ── Source provenance ──────────────────────────────────────────────────────────

def source_provenance(code_dir: str = CODE_DIR) -> dict:
    hashes = {}
    for path in sorted(glob.glob(os.path.join(code_dir, "*.py"))):
        with open(path, "rb") as f:
            hashes[os.path.basename(path)] = hashlib.sha256(f.read()).hexdigest()
    status = _run(["git", "-C", code_dir, "status", "--porcelain"])
    return {
        "git_commit":        _run(["git", "-C", code_dir, "rev-parse", "HEAD"]),
        "git_uncommitted_changes": None if status is None else bool(status),
        "sha256":            hashes,
    }


# ── torch.use_deterministic_algorithms probe ───────────────────────────────────

def probe_deterministic_algorithms(out_path: str = PROBE_JSON) -> dict:
    """
    One forward + backward pass per architecture with
    torch.use_deterministic_algorithms(True), in this process only, to record whether any
    op raises. The training runs never enable it (see run_experiment.py).
    """
    import torch
    import torch.nn as nn
    from model import get_model, MODEL_REGISTRY, NUM_CLASSES

    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name in MODEL_REGISTRY:
        model = None
        try:
            model = get_model(name, pretrained=False).to(device).train()
            x = torch.randn(4, 3, 224, 224, device=device)
            y = torch.arange(4, device=device) % NUM_CLASSES
            criterion = nn.CrossEntropyLoss(weight=torch.ones(NUM_CLASSES, device=device))
            criterion(model(x), y).backward()
            results[name] = {"raised": False}
        except Exception as exc:
            first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            results[name] = {"raised": True, "error": f"{type(exc).__name__}: {first_line}"}
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        print(f"  {name:16s} {'RAISED  ' + results[name]['error'] if results[name]['raised'] else 'ok'}")

    report = {
        "device":                   str(device),
        "gpu":                      torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch":                    torch.__version__,
        "CUBLAS_WORKSPACE_CONFIG":  os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "results":                  results,
        "any_op_raised":            any(r["raised"] for r in results.values()),
        "enabled_in_training_runs": False,
    }
    write_json(out_path, report)
    print(f"Deterministic-algorithms probe → {out_path}")
    return report


def main():
    ap = argparse.ArgumentParser(description="Record the runtime environment")
    ap.add_argument("--out",   default=ENVIRONMENT_JSON)
    ap.add_argument("--probe", action="store_true",
                    help="Also probe torch.use_deterministic_algorithms(True)")
    args = ap.parse_args()

    if args.probe:
        # Must be set before CUDA/cuBLAS initialises; affects this process only.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    write_environment_json(args.out)
    if args.probe:
        probe_deterministic_algorithms()


if __name__ == "__main__":
    main()
