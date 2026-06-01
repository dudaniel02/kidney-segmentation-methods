#!/usr/bin/env python3
"""
Step 1 — Convert KiTS19 into nnU-Net v2 directory layout.

Usage:
    python scripts/01_make_dataset.py [--config configs/config.yaml]
"""

import argparse
import json
import os
from pathlib import Path

from utils.config import load_config


def main(cfg: dict):
    src = Path(cfg["paths"]["kits19_root"])
    out = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"]

    for sub in ("imagesTr", "labelsTr"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cases = sorted(p for p in src.glob("case_*") if p.is_dir())
    valid, skipped = [], []

    for c in cases:
        img = c / "imaging.nii.gz"
        lab = c / "segmentation.nii.gz"
        if img.exists() and lab.exists():
            valid.append(c)
        else:
            skipped.append(c)

    print(f"Total cases:     {len(cases)}")
    print(f"  Valid:          {len(valid)}")
    print(f"  Skipped:        {len(skipped)}")

    for c in valid:
        dst_img = out / "imagesTr" / f"{c.name}_0000.nii.gz"
        dst_lab = out / "labelsTr" / f"{c.name}.nii.gz"
        if not dst_img.exists():
            os.symlink(c / "imaging.nii.gz", dst_img)
        if not dst_lab.exists():
            os.symlink(c / "segmentation.nii.gz", dst_lab)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "kidney": 1, "tumor": 2},
        "numTraining": len(valid),
        "file_ending": ".nii.gz",
        "dataset_name": "KiTS19",
        "description": "Kidney and Tumor segmentation (KiTS19)",
    }
    with open(out / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nDataset ready at: {out}")
    print(f"  Training cases: {len(valid)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
