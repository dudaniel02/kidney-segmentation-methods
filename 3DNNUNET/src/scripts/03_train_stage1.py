#!/usr/bin/env python3
"""
Step 3 — Train Stage 1 (full kidney + tumour) on all configured folds.

By default trains folds 0–4 for a 5-fold cross-validation ensemble.
Set training.folds in config.yaml to [0] for a quick single-fold test.

Usage:
    python scripts/03_train_stage1.py [--config configs/config.yaml] [--fold 0]
"""

import argparse
import subprocess
import sys

from utils.config import load_config


def train_fold(cfg: dict, fold: int):
    dataset_id = cfg["datasets"]["stage1_id"]
    configuration = cfg["architecture"]["configuration"]
    trainer = cfg["architecture"]["trainer"]
    plans = cfg["architecture"]["plans_name"]

    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-p", plans,
        "-tr", trainer,
    ]

    if cfg["training"].get("continue_training", False):
        cmd.append("--c")

    print(f"\n{'='*60}")
    print(f"  Training Stage 1 — Fold {fold}")
    print(f"  Plans: {plans}  |  Config: {configuration}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: Fold {fold} training returned non-zero exit code.")
        return False
    return True


def main(cfg: dict, single_fold: int = None):
    folds = [single_fold] if single_fold is not None else cfg["training"]["folds"]

    results = {}
    for fold in folds:
        ok = train_fold(cfg, fold)
        results[fold] = "OK" if ok else "FAILED"

    print("\n" + "="*60)
    print("  Stage 1 Training Summary")
    print("="*60)
    for fold, status in results.items():
        print(f"  Fold {fold}: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train a single fold (overrides config)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, single_fold=args.fold)
