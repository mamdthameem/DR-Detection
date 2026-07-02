"""
Task 10: Pipeline orchestrator.
Runs all tasks in order with clear progress headers.

Usage examples:
  python main.py                          # full pipeline
  python main.py --skip-preprocess        # skip preprocessing (images already exist)
  python main.py --skip-train             # skip training (use existing checkpoints)
  python main.py --eval-only              # evaluation + XAI only
  python main.py --xai-only              # XAI visualisation only
  python main.py --models efficientnet_b0 resnet50  # train/eval only these two
"""

import os
import sys
import subprocess
import argparse

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def section(title: str) -> None:
    banner = "=" * 70
    print(f"\n{banner}\n  {title}\n{banner}")


def run(script: str, extra_args: list = None) -> None:
    cmd    = [sys.executable, os.path.join(SCRIPTS_DIR, script)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Script {script} exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="DR Classification Research Pipeline")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing (preprocessed images already exist)")
    parser.add_argument("--skip-train",      action="store_true",
                        help="Skip training (checkpoints already exist)")
    parser.add_argument("--eval-only",       action="store_true",
                        help="Run evaluation + XAI only (implies --skip-preprocess --skip-train)")
    parser.add_argument("--xai-only",        action="store_true",
                        help="Run XAI scripts only (requires checkpoints + xai_sample_indices.json)")
    parser.add_argument("--models",          nargs="+",
                        default=["efficientnet_b0", "resnet50", "vgg16", "inceptionv3"],
                        help="Models to train and evaluate")
    args = parser.parse_args()

    if args.eval_only:
        args.skip_preprocess = True
        args.skip_train      = True

    print("\n" + "=" * 70)
    print("  Diabetic Retinopathy Detection – Research Pipeline")
    print("=" * 70)
    print(f"  Models : {args.models}")
    print(f"  Flags  : skip-preprocess={args.skip_preprocess}  "
          f"skip-train={args.skip_train}  xai-only={args.xai_only}")

    if not args.xai_only:

        # ── Task 1: Preprocessing ──────────────────────────────────────────────
        if not args.skip_preprocess:
            section("TASK 1 – Preprocessing (CLAHE + crop + resize)")
            run("preprocess.py")
        else:
            section("TASK 1 – Preprocessing  [SKIPPED]")

        # ── Task 3: Model summary (no training needed) ────────────────────────
        section("TASK 3 – Model Summary (parameter counts + FLOPs)")
        run("model.py")

        # ── Tasks 2+4: Data loading + Training ────────────────────────────────
        if not args.skip_train:
            section("TASK 4 – Training (all models)")
            run("train.py", ["--models"] + args.models)
        else:
            section("TASK 4 – Training  [SKIPPED]")

        # ── Task 5: Evaluation ────────────────────────────────────────────────
        section("TASK 5 – Evaluation (metrics + confusion matrix + ROC)")
        run("evaluate.py", ["--models"] + args.models)

    # ── Task 6: Grad-CAM ──────────────────────────────────────────────────────
    section("TASK 6 – XAI: Grad-CAM")
    run("xai_gradcam.py")

    # ── Task 7: SHAP ──────────────────────────────────────────────────────────
    section("TASK 7 – XAI: SHAP")
    run("xai_shap.py")

    # ── Task 8: LIME ──────────────────────────────────────────────────────────
    section("TASK 8 – XAI: LIME")
    run("xai_lime.py")

    # ── Task 9: XAI comparison grid ───────────────────────────────────────────
    section("TASK 9 – XAI Comparison Grid (paper figure)")
    run("xai_compare.py")

    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print("  All outputs → /kaggle/working/outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
