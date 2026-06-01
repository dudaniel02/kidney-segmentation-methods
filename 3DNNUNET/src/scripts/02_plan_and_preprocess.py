#!/usr/bin/env python3
"""
Step 2 — Run nnU-Net experiment planning + preprocessing.

Uses the ResEnc M planner by default (configurable in config.yaml).
If you already preprocessed with the standard planner, ResEnc M reuses
the same preprocessed data — only the plans JSON differs.

Usage:
    python scripts/02_plan_and_preprocess.py [--config configs/config.yaml]
"""

import argparse
import subprocess
import sys

from utils.config import load_config


def main(cfg: dict):
    dataset_id = cfg["datasets"]["stage1_id"]
    planner = cfg["architecture"]["planner"]

    # nnU-Net v2 plan + preprocess
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-pl", planner,
        "--verify_dataset_integrity",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("ERROR: Planning/preprocessing failed.")
        sys.exit(1)

    print("\nPreprocessing complete.")
    print(f"  Planner used: {planner}")
    print(f"  Plans will be stored as: {cfg['architecture']['plans_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
