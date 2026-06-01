#!/usr/bin/env python3
"""
Step 4 — Run Stage 1 inference using the 5-fold ensemble.

nnU-Net's predict command with -f 0 1 2 3 4 automatically averages
the softmax outputs from all fold models before taking argmax.
This typically adds 1–3% Dice over a single fold.

Usage:
    python scripts/04_predict_stage1.py [--config configs/config.yaml]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from utils.config import load_config


def main(cfg: dict):
    dataset_id = cfg["datasets"]["stage1_id"]
    configuration = cfg["architecture"]["configuration"]
    plans = cfg["architecture"]["plans_name"]
    trainer = cfg["architecture"]["trainer"]
    folds = cfg["training"]["folds"]

    # Input: the raw imagesTr directory
    input_dir = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"] / "imagesTr"
    output_dir = Path(cfg["paths"]["predictions_stage1"])
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_args = [str(f) for f in folds]

    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-p", plans,
        "-tr", trainer,
        "-f", *fold_args,
        "--save_probabilities",      # needed for softmax-based ROI extraction
    ]

    print(f"Running Stage 1 ensemble prediction (folds: {folds})")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("ERROR: Stage 1 prediction failed.")
        sys.exit(1)

    print("\nStage 1 prediction complete.")
    print(f"  Masks + softmax .npz files saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
