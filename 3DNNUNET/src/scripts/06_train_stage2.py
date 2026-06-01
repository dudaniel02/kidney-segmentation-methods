#!/usr/bin/env python3
"""
Step 6 — Plan, preprocess, and train Stage 2 (tumour refinement on ROIs).

This runs the full nnU-Net pipeline on the cropped ROI dataset.
Same ResEnc architecture and 5-fold ensemble as Stage 1.

Usage:
    python scripts/06_train_stage2.py [--config configs/config.yaml] [--fold 0]
"""

import argparse
import subprocess
import sys

from utils.config import load_config


def main(cfg: dict, single_fold: int = None):
    dataset_id = cfg["datasets"]["stage2_id"]
    planner = cfg["architecture"]["planner"]
    configuration = cfg["architecture"]["configuration"]
    plans = cfg["architecture"]["plans_name"]
    trainer = cfg["architecture"]["trainer"]
    folds = [single_fold] if single_fold is not None else cfg["training"]["folds"]

    # --- Plan + Preprocess ---
    print("="*60)
    print("  Stage 2: Planning and Preprocessing")
    print("="*60)
    cmd_prep = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-pl", planner,
        "--verify_dataset_integrity",
    ]
    result = subprocess.run(cmd_prep, check=False)
    if result.returncode != 0:
        print("ERROR: Stage 2 preprocessing failed.")
        sys.exit(1)

    # --- Train each fold ---
    results = {}
    for fold in folds:
        print(f"\n{'='*60}")
        print(f"  Stage 2 Training — Fold {fold}")
        print(f"{'='*60}\n")

        cmd_train = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "-p", plans,
            "-tr", trainer,
        ]
        if cfg["training"].get("continue_training", False):
            cmd_train.append("--c")

        ret = subprocess.run(cmd_train, check=False)
        results[fold] = "OK" if ret.returncode == 0 else "FAILED"

    print("\n" + "="*60)
    print("  Stage 2 Training Summary")
    print("="*60)
    for fold, status in results.items():
        print(f"  Fold {fold}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, single_fold=args.fold)
