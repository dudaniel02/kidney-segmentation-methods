#!/usr/bin/env python3
"""
Step 5 — Extract tumour-focused ROIs from Stage 1 predictions for Stage 2.
Uses SimpleITK for saving to ensure nnU-Net compatibility.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from utils.config import load_config


def compute_bbox(mask, padding):
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    shape = mask.shape
    return {
        "z_min": max(0, int(mins[0] - padding)),
        "z_max": min(shape[0], int(maxs[0] + padding + 1)),
        "y_min": max(0, int(mins[1] - padding)),
        "y_max": min(shape[1], int(maxs[1] + padding + 1)),
        "x_min": max(0, int(mins[2] - padding)),
        "x_max": min(shape[2], int(maxs[2] + padding + 1)),
        "original_shape": list(shape),
    }


def extract_roi_for_case(case_name, ct_path, pred_path, gt_path, softmax_path,
                         cfg, out_images, out_labels, bbox_dir):
    roi_cfg = cfg["roi"]
    padding = roi_cfg["padding_voxels"]
    use_softmax = roi_cfg.get("use_softmax", False)
    softmax_thresh = roi_cfg.get("softmax_threshold", 0.3)
    min_voxels = roi_cfg.get("min_tumour_voxels", 50)

    # Load prediction with SimpleITK
    pred_sitk = sitk.ReadImage(str(pred_path))
    pred = sitk.GetArrayFromImage(pred_sitk).astype(np.int32)

    if use_softmax and softmax_path.exists():
        probs = np.load(str(softmax_path))["probabilities"]
        kidney_prob = probs[1] + probs[2]
        roi_mask = (kidney_prob > softmax_thresh).astype(np.int32)
    else:
        roi_mask = (pred > 0).astype(np.int32)

    tumour_voxels = int(np.sum(pred == 2))
    if tumour_voxels < min_voxels:
        print(f"  {case_name}: skipped (only {tumour_voxels} tumour voxels)")
        return False

    bbox = compute_bbox(roi_mask, padding)
    if bbox is None:
        print(f"  {case_name}: skipped (no kidney/tumour found)")
        return False

    crop_shape = (
        bbox["z_max"] - bbox["z_min"],
        bbox["y_max"] - bbox["y_min"],
        bbox["x_max"] - bbox["x_min"],
    )
    if any(s < 2 for s in crop_shape):
        print(f"  {case_name}: skipped (degenerate crop {crop_shape})")
        return False

    # Load CT with SimpleITK
    ct_sitk = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(ct_sitk)
    ct_crop = ct[
        bbox["z_min"]:bbox["z_max"],
        bbox["y_min"]:bbox["y_max"],
        bbox["x_min"]:bbox["x_max"],
    ].copy()

    # Load GT, crop, convert to binary
    gt_sitk = sitk.ReadImage(str(gt_path))
    gt = sitk.GetArrayFromImage(gt_sitk).astype(np.int32)
    gt_crop = gt[
        bbox["z_min"]:bbox["z_max"],
        bbox["y_min"]:bbox["y_max"],
        bbox["x_min"]:bbox["x_max"],
    ].copy()
    gt_binary = (gt_crop == 2).astype(np.uint8)

    # Compute new origin for cropped volume
    original_origin = np.array(ct_sitk.GetOrigin())
    original_spacing = np.array(ct_sitk.GetSpacing())
    original_direction = np.array(ct_sitk.GetDirection()).reshape(3, 3)

    # SimpleITK uses (x, y, z) ordering for origin/spacing
    # but numpy array is (z, y, x), so offset is [z, y, x] in voxels
    voxel_offset_xyz = np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"]], dtype=float)
    new_origin = original_origin + original_direction @ (voxel_offset_xyz * original_spacing)

    # Save CT crop
    ct_out = sitk.GetImageFromArray(ct_crop.astype(np.float32))
    ct_out.SetSpacing(ct_sitk.GetSpacing())
    ct_out.SetDirection(ct_sitk.GetDirection())
    ct_out.SetOrigin(new_origin.tolist())
    sitk.WriteImage(ct_out, str(out_images / f"{case_name}_0000.nii.gz"))

    # Save label as uint8
    gt_out = sitk.GetImageFromArray(gt_binary)
    gt_out.SetSpacing(ct_sitk.GetSpacing())
    gt_out.SetDirection(ct_sitk.GetDirection())
    gt_out.SetOrigin(new_origin.tolist())
    gt_out = sitk.Cast(gt_out, sitk.sitkUInt8)
    sitk.WriteImage(gt_out, str(out_labels / f"{case_name}.nii.gz"))

    # Verify
    try:
        sitk.ReadImage(str(out_labels / f"{case_name}.nii.gz"))
    except Exception as e:
        print(f"  {case_name}: ERROR — corrupt ({e})")
        (out_images / f"{case_name}_0000.nii.gz").unlink(missing_ok=True)
        (out_labels / f"{case_name}.nii.gz").unlink(missing_ok=True)
        return False

    with open(bbox_dir / f"{case_name}.json", "w") as f:
        json.dump(bbox, f, indent=2)

    print(f"  {case_name}: OK  (tumour: {tumour_voxels}, shape: {crop_shape})")
    return True


def main(cfg):
    stage2_root = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage2_name"]
    out_images = stage2_root / "imagesTr"
    out_labels = stage2_root / "labelsTr"
    bbox_dir = stage2_root / "bbox_metadata"

    import shutil
    if stage2_root.exists():
        print(f"Removing old Stage 2 dataset at {stage2_root}...")
        shutil.rmtree(stage2_root)

    for d in (out_images, out_labels, bbox_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_images = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"] / "imagesTr"
    raw_labels = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"] / "labelsTr"
    pred_dir = Path(cfg["paths"]["predictions_stage1"])

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"ERROR: No predictions found in {pred_dir}")
        return

    print(f"Extracting ROIs from {len(pred_files)} Stage 1 predictions")
    print(f"  Padding: {cfg['roi']['padding_voxels']} voxels")
    print(f"  Softmax ROI: {cfg['roi'].get('use_softmax', False)}")
    print()

    valid_count = 0
    for pred_file in pred_files:
        case_name = pred_file.stem.replace(".nii", "")
        ct_path = raw_images / f"{case_name}_0000.nii.gz"
        gt_path = raw_labels / f"{case_name}.nii.gz"
        softmax_path = pred_dir / f"{case_name}.npz"

        if not ct_path.exists() or not gt_path.exists():
            print(f"  {case_name}: skipped (missing CT or GT)")
            continue

        ok = extract_roi_for_case(
            case_name, ct_path, pred_file, gt_path, softmax_path,
            cfg, out_images, out_labels, bbox_dir,
        )
        if ok:
            valid_count += 1

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": valid_count,
        "file_ending": ".nii.gz",
        "dataset_name": "KiTS19_TumorROI",
        "description": "Tumour ROI crops from Stage 1 for refinement",
    }
    with open(stage2_root / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nStage 2 dataset ready: {stage2_root}")
    print(f"  Valid ROI cases: {valid_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
