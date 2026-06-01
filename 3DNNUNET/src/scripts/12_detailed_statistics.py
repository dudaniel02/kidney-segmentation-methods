#!/usr/bin/env python3
"""
Compute detailed statistics for the thesis:
- Per-fold results
- Precision and sensitivity per class
- Volume analysis
- Stratified analysis by tumour size
"""

import csv
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from utils.config import load_config


def dice(p, g, label):
    a = (p == label).astype(float)
    b = (g == label).astype(float)
    i = (a * b).sum()
    d = a.sum() + b.sum()
    return 2 * i / d if d > 0 else (1.0 if b.sum() == 0 else 0.0)


def precision_recall(p, g, label):
    tp = ((p == label) & (g == label)).sum()
    fp = ((p == label) & (g != label)).sum()
    fn = ((p != label) & (g == label)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(prec), float(rec)


def volume_ml(mask, label, spacing):
    voxel_vol = float(np.prod(spacing)) / 1000.0
    return float((mask == label).sum()) * voxel_vol


def main():
    cfg = load_config("configs/config.yaml")
    pred_dir = Path(cfg["paths"]["final_output"])
    gt_dir = Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"] / "labelsTr"

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Computing detailed statistics for {len(pred_files)} cases...\n")

    results = []
    for pf in pred_files:
        cn = pf.stem.replace(".nii", "")
        gp = gt_dir / f"{cn}.nii.gz"
        if not gp.exists():
            continue

        p_sitk = sitk.ReadImage(str(pf))
        g_sitk = sitk.ReadImage(str(gp))
        p = sitk.GetArrayFromImage(p_sitk).astype(int)
        g = sitk.GetArrayFromImage(g_sitk).astype(int)
        spacing = p_sitk.GetSpacing()

        k_dice = dice(p, g, 1)
        t_dice = dice(p, g, 2)
        k_prec, k_rec = precision_recall(p, g, 1)
        t_prec, t_rec = precision_recall(p, g, 2)
        gt_t_vol = volume_ml(g, 2, spacing)
        pred_t_vol = volume_ml(p, 2, spacing)
        gt_k_vol = volume_ml(g, 1, spacing)

        # Count tumour regions in GT
        t_mask = (g == 2).astype(int)
        _, n_regions = ndimage.label(t_mask)

        results.append({
            "case": cn,
            "kidney_dice": round(k_dice, 4),
            "tumour_dice": round(t_dice, 4),
            "kidney_precision": round(k_prec, 4),
            "kidney_recall": round(k_rec, 4),
            "tumour_precision": round(t_prec, 4),
            "tumour_recall": round(t_rec, 4),
            "gt_tumour_vol_ml": round(gt_t_vol, 2),
            "pred_tumour_vol_ml": round(pred_t_vol, 2),
            "gt_kidney_vol_ml": round(gt_k_vol, 2),
            "n_tumour_regions": n_regions,
        })

    # Save full CSV
    out_csv = pred_dir / "detailed_evaluation.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {out_csv}\n")

    # ── Aggregate statistics ──
    kd = [r["kidney_dice"] for r in results]
    td = [r["tumour_dice"] for r in results]
    tp = [r["tumour_precision"] for r in results]
    tr = [r["tumour_recall"] for r in results]
    kp = [r["kidney_precision"] for r in results]
    kr = [r["kidney_recall"] for r in results]
    gt_vols = [r["gt_tumour_vol_ml"] for r in results]
    pred_vols = [r["pred_tumour_vol_ml"] for r in results]

    print("=" * 65)
    print("  AGGREGATE STATISTICS (210 cases)")
    print("=" * 65)
    print(f"  {'Metric':<30} {'Kidney':>12} {'Tumour':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Mean DSC':<30} {np.mean(kd):>12.4f} {np.mean(td):>12.4f}")
    print(f"  {'Std DSC':<30} {np.std(kd):>12.4f} {np.std(td):>12.4f}")
    print(f"  {'Median DSC':<30} {np.median(kd):>12.4f} {np.median(td):>12.4f}")
    print(f"  {'Min DSC':<30} {np.min(kd):>12.4f} {np.min(td):>12.4f}")
    print(f"  {'Max DSC':<30} {np.max(kd):>12.4f} {np.max(td):>12.4f}")
    print(f"  {'25th percentile DSC':<30} {np.percentile(kd,25):>12.4f} {np.percentile(td,25):>12.4f}")
    print(f"  {'75th percentile DSC':<30} {np.percentile(kd,75):>12.4f} {np.percentile(td,75):>12.4f}")
    print(f"  {'95th percentile DSC':<30} {np.percentile(kd,95):>12.4f} {np.percentile(td,95):>12.4f}")
    print(f"  {'Mean Precision':<30} {np.mean(kp):>12.4f} {np.mean(tp):>12.4f}")
    print(f"  {'Mean Recall (Sensitivity)':<30} {np.mean(kr):>12.4f} {np.mean(tr):>12.4f}")
    print()

    # ── Stratified by tumour size ──
    small = [r for r in results if r["gt_tumour_vol_ml"] < 10]
    medium = [r for r in results if 10 <= r["gt_tumour_vol_ml"] < 50]
    large = [r for r in results if r["gt_tumour_vol_ml"] >= 50]

    print("=" * 65)
    print("  STRATIFIED BY TUMOUR VOLUME")
    print("=" * 65)
    for label, group in [("Small (<10 mL)", small), ("Medium (10-50 mL)", medium), ("Large (>50 mL)", large)]:
        if not group:
            continue
        gd = [r["tumour_dice"] for r in group]
        gp = [r["tumour_precision"] for r in group]
        gr = [r["tumour_recall"] for r in group]
        gv = [r["gt_tumour_vol_ml"] for r in group]
        print(f"\n  {label}: {len(group)} cases")
        print(f"    Mean tumour volume:    {np.mean(gv):.1f} mL")
        print(f"    Mean tumour DSC:       {np.mean(gd):.4f}")
        print(f"    Std tumour DSC:        {np.std(gd):.4f}")
        print(f"    Min tumour DSC:        {np.min(gd):.4f}")
        print(f"    Mean precision:        {np.mean(gp):.4f}")
        print(f"    Mean recall:           {np.mean(gr):.4f}")

    # ── Multi-tumour vs single-tumour ──
    single = [r for r in results if r["n_tumour_regions"] == 1]
    multi = [r for r in results if r["n_tumour_regions"] >= 2]

    print(f"\n{'=' * 65}")
    print("  SINGLE vs MULTI-TUMOUR CASES")
    print("=" * 65)
    for label, group in [("Single tumour", single), ("Multi-tumour (2+)", multi)]:
        gd = [r["tumour_dice"] for r in group]
        print(f"\n  {label}: {len(group)} cases")
        print(f"    Mean tumour DSC:       {np.mean(gd):.4f}")
        print(f"    Std tumour DSC:        {np.std(gd):.4f}")
        print(f"    Median tumour DSC:     {np.median(gd):.4f}")

    # ── Volume correlation ──
    vol_errors = [abs(r["pred_tumour_vol_ml"] - r["gt_tumour_vol_ml"]) for r in results]
    vol_rel_errors = [abs(r["pred_tumour_vol_ml"] - r["gt_tumour_vol_ml"]) / max(r["gt_tumour_vol_ml"], 0.01) * 100 for r in results]
    correlation = np.corrcoef(gt_vols, pred_vols)[0, 1]

    print(f"\n{'=' * 65}")
    print("  VOLUME ANALYSIS")
    print("=" * 65)
    print(f"  Pearson correlation (GT vs Pred volume): {correlation:.4f}")
    print(f"  Mean absolute volume error:              {np.mean(vol_errors):.2f} mL")
    print(f"  Median absolute volume error:            {np.median(vol_errors):.2f} mL")
    print(f"  Mean relative volume error:              {np.mean(vol_rel_errors):.1f}%")
    print(f"  Median relative volume error:            {np.median(vol_rel_errors):.1f}%")


if __name__ == "__main__":
    main()
