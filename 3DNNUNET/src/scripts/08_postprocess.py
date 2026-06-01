#!/usr/bin/env python3
"""
Step 8 — Apply post-processing to final combined segmentations.

Includes:
  - Connected component analysis (remove small spurious regions)
  - Keep only the 2 largest kidney components
  - Remove tiny tumour blobs

Usage:
    python scripts/08_postprocess.py [--config configs/config.yaml]
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from utils.config import load_config
from postprocessing.components import postprocess_segmentation


def main(cfg: dict):
    final_dir = Path(cfg["paths"]["final_output"])
    pp_dir = final_dir.parent / "final_postprocessed"
    pp_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(final_dir.glob("*.nii.gz"))
    print(f"Post-processing {len(pred_files)} segmentations...")
    print(f"  Output: {pp_dir}\n")

    cfg_pp = cfg["postprocessing"]

    for pred_file in pred_files:
        nii = nib.load(str(pred_file))
        mask = nii.get_fdata().astype(np.int32)

        # Get voxel spacing from header
        spacing = nii.header.get_zooms()[:3]

        mask_pp = postprocess_segmentation(mask, spacing, cfg_pp)

        # Count changes
        changed_voxels = np.sum(mask != mask_pp)
        case_name = pred_file.stem.replace(".nii", "")

        out_nii = nib.Nifti1Image(mask_pp.astype(np.int16), nii.affine, nii.header)
        nib.save(out_nii, str(pp_dir / pred_file.name))

        if changed_voxels > 0:
            print(f"  {case_name}: {changed_voxels} voxels changed")
        else:
            print(f"  {case_name}: no changes")

    print(f"\nPost-processed segmentations saved to: {pp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
