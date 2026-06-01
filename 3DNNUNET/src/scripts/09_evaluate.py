 #!/usr/bin/env python3
"""
Step 9 — Evaluate final segmentations against ground truth.

Metrics:
  - Dice Similarity Coefficient (per-class and mean)
  - 95th percentile Hausdorff Distance
  - Normalised Surface Dice (optional)

Produces a CSV summary and prints aggregate statistics.

Usage:
    python scripts/09_evaluate.py [--config configs/config.yaml] [--use-postprocessed]
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from utils.config import load_config


# ── Metrics ──────────────────────────────────────────────────


def dice_score(pred: np.ndarray, gt: np.ndarray, label: int) -> float:
    """Compute Dice coefficient for a single label."""
    p = (pred == label).astype(np.float64)
    g = (gt == label).astype(np.float64)
    intersection = np.sum(p * g)
    denom = np.sum(p) + np.sum(g)
    if denom == 0:
        return 1.0 if np.sum(g) == 0 else 0.0
    return 2.0 * intersection / denom


def hausdorff_95(pred: np.ndarray, gt: np.ndarray, label: int, spacing: tuple) -> float:
    """Compute 95th percentile Hausdorff distance in mm."""
    p = (pred == label)
    g = (gt == label)

    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float("inf")

    # Surface voxels via erosion
    p_surface = p ^ ndimage.binary_erosion(p)
    g_surface = g ^ ndimage.binary_erosion(g)

    # Distance transforms
    dt_p = ndimage.distance_transform_edt(~p_surface, sampling=spacing)
    dt_g = ndimage.distance_transform_edt(~g_surface, sampling=spacing)

    # Directed distances
    d_p2g = dt_g[p_surface]
    d_g2p = dt_p[g_surface]

    if len(d_p2g) == 0 or len(d_g2p) == 0:
        return float("inf")

    return max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))


# ── Main ─────────────────────────────────────────────────────


def main(cfg: dict, use_postprocessed: bool = False):
    if use_postprocessed:
        pred_dir = Path(cfg["paths"]["final_output"]).parent / "final_postprocessed"
    else:
        pred_dir = Path(cfg["paths"]["final_output"])

    gt_dir = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"] / "labelsTr"

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"ERROR: No predictions in {pred_dir}")
        return

    print(f"Evaluating {len(pred_files)} cases")
    print(f"  Predictions: {pred_dir}")
    print(f"  Ground truth: {gt_dir}")
    print(f"  Post-processed: {use_postprocessed}\n")

    results = []
    for pred_file in pred_files:
        case_name = pred_file.stem.replace(".nii", "")
        gt_path = gt_dir / f"{case_name}.nii.gz"

        if not gt_path.exists():
            continue

        pred_nii = nib.load(str(pred_file))
        gt_nii = nib.load(str(gt_path))
        pred = pred_nii.get_fdata().astype(np.int32)
        gt = gt_nii.get_fdata().astype(np.int32)
        spacing = pred_nii.header.get_zooms()[:3]

        kidney_dice = dice_score(pred, gt, label=1)
        tumour_dice = dice_score(pred, gt, label=2)

        row = {
            "case": case_name,
            "kidney_dice": round(kidney_dice, 4),
            "tumour_dice": round(tumour_dice, 4),
        }

        if "hausdorff95" in cfg["evaluation"]["metrics"]:
            kidney_hd95 = hausdorff_95(pred, gt, label=1, spacing=spacing)
            tumour_hd95 = hausdorff_95(pred, gt, label=2, spacing=spacing)
            row["kidney_hd95"] = round(kidney_hd95, 2)
            row["tumour_hd95"] = round(tumour_hd95, 2)

        results.append(row)

    # Print summary
    kidney_dices = [r["kidney_dice"] for r in results]
    tumour_dices = [r["tumour_dice"] for r in results]

    print(f"{'='*60}")
    print(f"  RESULTS ({len(results)} cases)")
    print(f"{'='*60}")
    print(f"  Kidney Dice:  mean={np.mean(kidney_dices):.4f}  "
          f"std={np.std(kidney_dices):.4f}  "
          f"median={np.median(kidney_dices):.4f}")
    print(f"  Tumour Dice:  mean={np.mean(tumour_dices):.4f}  "
          f"std={np.std(tumour_dices):.4f}  "
          f"median={np.median(tumour_dices):.4f}")

    if "hausdorff95" in cfg["evaluation"]["metrics"]:
        kidney_hd = [r["kidney_hd95"] for r in results if r["kidney_hd95"] != float("inf")]
        tumour_hd = [r["tumour_hd95"] for r in results if r["tumour_hd95"] != float("inf")]
        if kidney_hd:
            print(f"  Kidney HD95:  mean={np.mean(kidney_hd):.2f} mm")
        if tumour_hd:
            print(f"  Tumour HD95:  mean={np.mean(tumour_hd):.2f} mm")

    # Save CSV
    csv_path = pred_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Detailed results: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--use-postprocessed", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, use_postprocessed=args.use_postprocessed)
