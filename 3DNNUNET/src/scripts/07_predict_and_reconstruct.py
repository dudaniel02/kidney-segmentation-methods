#!/usr/bin/env python3
"""
Step 7 — Predict Stage 2 and reconstruct full-volume segmentations.
Uses SimpleITK for resampling to handle resolution mismatches.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

from utils.config import load_config


def run_stage2_inference(cfg):
    dataset_id = cfg["datasets"]["stage2_id"]
    configuration = cfg["architecture"]["configuration"]
    plans = cfg["architecture"]["plans_name"]
    trainer = cfg["architecture"]["trainer"]
    folds = cfg["training"]["folds"]

    input_dir = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage2_name"] / "imagesTr"
    output_dir = Path(cfg["paths"]["predictions_stage2"])
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
    ]

    print("Running Stage 2 ensemble prediction...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("ERROR: Stage 2 prediction failed.")
        sys.exit(1)
    print("Stage 2 prediction complete.\n")


def reconstruct_full_volume(cfg):
    stage1_dir = Path(cfg["paths"]["predictions_stage1"])
    stage2_dir = Path(cfg["paths"]["predictions_stage2"])
    bbox_dir = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage2_name"] / "bbox_metadata"
    final_dir = Path(cfg["paths"]["final_output"])
    final_dir.mkdir(parents=True, exist_ok=True)

    stage2_preds = sorted(stage2_dir.glob("*.nii.gz"))
    print(f"Reconstructing {len(stage2_preds)} cases...")

    success = 0
    errors = 0

    for s2_file in stage2_preds:
        case_name = s2_file.stem.replace(".nii", "")

        s1_path = stage1_dir / f"{case_name}.nii.gz"
        if not s1_path.exists():
            print(f"  {case_name}: skipped (no Stage 1 prediction)")
            continue

        bbox_path = bbox_dir / f"{case_name}.json"
        if not bbox_path.exists():
            print(f"  {case_name}: skipped (no bbox metadata)")
            continue

        with open(bbox_path) as f:
            bbox = json.load(f)

        try:
            # Load Stage 1 full-volume prediction
            s1_sitk = sitk.ReadImage(str(s1_path))
            s1_mask = sitk.GetArrayFromImage(s1_sitk).astype(np.int32)

            # Load Stage 2 ROI prediction
            s2_sitk = sitk.ReadImage(str(s2_file))
            s2_arr = sitk.GetArrayFromImage(s2_sitk).astype(np.int32)

            # Expected ROI shape from bbox
            expected_shape = (
                bbox["z_max"] - bbox["z_min"],
                bbox["y_max"] - bbox["y_min"],
                bbox["x_max"] - bbox["x_min"],
            )

            # If shapes don't match, resample Stage 2 prediction to expected size
            if s2_arr.shape != expected_shape:
                # Load the original ROI image to get its reference geometry
                roi_img_path = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage2_name"] / "imagesTr" / f"{case_name}_0000.nii.gz"
                if roi_img_path.exists():
                    ref_sitk = sitk.ReadImage(str(roi_img_path))
                else:
                    # Fallback: create reference from bbox and Stage 1 metadata
                    ref_sitk = sitk.Image([expected_shape[2], expected_shape[1], expected_shape[0]], sitk.sitkUInt8)
                    ref_sitk.SetSpacing(s1_sitk.GetSpacing())
                    ref_sitk.SetDirection(s1_sitk.GetDirection())

                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_sitk)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                s2_resampled = resampler.Execute(s2_sitk)
                s2_arr = sitk.GetArrayFromImage(s2_resampled).astype(np.int32)

            # If still mismatched, crop/pad to fit
            if s2_arr.shape != expected_shape:
                fitted = np.zeros(expected_shape, dtype=np.int32)
                min_z = min(s2_arr.shape[0], expected_shape[0])
                min_y = min(s2_arr.shape[1], expected_shape[1])
                min_x = min(s2_arr.shape[2], expected_shape[2])
                fitted[:min_z, :min_y, :min_x] = s2_arr[:min_z, :min_y, :min_x]
                s2_arr = fitted

            # Build final mask
            final = s1_mask.copy()

            # Clear old tumour in ROI region
            roi_slice = (
                slice(bbox["z_min"], bbox["z_max"]),
                slice(bbox["y_min"], bbox["y_max"]),
                slice(bbox["x_min"], bbox["x_max"]),
            )
            roi_region = final[roi_slice]
            roi_region[roi_region == 2] = 0

            # Insert refined tumour
            roi_region[s2_arr == 1] = 2
            final[roi_slice] = roi_region

            # Save
            out_sitk = sitk.GetImageFromArray(final.astype(np.int16))
            out_sitk.CopyInformation(s1_sitk)
            sitk.WriteImage(out_sitk, str(final_dir / f"{case_name}.nii.gz"))

            success += 1
            print(f"  {case_name}: OK")

        except Exception as e:
            errors += 1
            print(f"  {case_name}: ERROR — {e}")

    print(f"\nReconstruction complete: {success} OK, {errors} errors")
    print(f"Final segmentations saved to: {final_dir}")


def main(cfg):
    run_stage2_inference(cfg)
    reconstruct_full_volume(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
