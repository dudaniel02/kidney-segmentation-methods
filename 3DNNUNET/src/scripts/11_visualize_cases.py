#!/usr/bin/env python3
"""
Case visualizations for 3D nnUNet kidney and tumour segmentation.

Saves both a high-resolution PNG and a PDF for each view of each case.
PDF keeps title/legend as vectors so it prints crisply at any size.
Each figure has 3 panels: Ground Truth / Prediction / Overlap Analysis.

Axis → plane mapping (empirically verified against KiTS19 volumes):
    axis 0 → Coronal   (anterior-posterior primary)
    axis 1 → Sagittal  (left-right primary)
    axis 2 → Axial     (superior-inferior primary)
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation, zoom
from utils.config import load_config

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        14,
    'axes.titlesize':   18,
    'axes.labelsize':   15,
    'legend.fontsize':  14,
    'figure.titlesize': 19,
    'pdf.fonttype':     42,   # embed as TrueType so PDF text is selectable
    'ps.fonttype':      42,
})

OUTPUT_DPI = 600   # PNG raster resolution

# Colour palette — distinct hues, readable on dark background and for
# common forms of colour-vision deficiency
KIDNEY_COLOR = np.array([0.15, 0.65, 0.85])   # cyan-blue
TUMOUR_COLOR = np.array([0.95, 0.25, 0.10])   # vivid red
TP_COLOR     = np.array([0.10, 0.85, 0.20])   # bright green
FP_COLOR     = np.array([1.00, 0.60, 0.00])   # orange
FN_COLOR     = np.array([0.80, 0.10, 0.90])   # magenta

# Anatomical direction labels per plane (radiological convention).
# Verify against your data; flip as needed if axes appear mirrored.
VIEW_ORIENT = {
    "Axial":    {"top": "A", "bottom": "P", "left": "R", "right": "L"},
    "Coronal":  {"top": "S", "bottom": "I", "left": "R", "right": "L"},
    "Sagittal": {"top": "S", "bottom": "I", "left": "A", "right": "P"},
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_case(cfg, case_name):
    gt_path   = (Path(cfg["paths"]["nnunet_raw"]) / cfg["datasets"]["stage1_name"]
                 / "labelsTr" / f"{case_name}.nii.gz")
    pred_path = Path(cfg["paths"]["final_output"]) / f"{case_name}.nii.gz"
    ct_path   = Path(cfg["paths"]["kits19_root"]) / case_name / "imaging.nii.gz"
    ct_sitk   = sitk.ReadImage(str(ct_path))
    ct        = sitk.GetArrayFromImage(ct_sitk).astype(float)
    gt        = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))).astype(int)
    pred      = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))).astype(int)
    spacing   = ct_sitk.GetSpacing()   # (x_sp, y_sp, z_sp) in mm
    return ct, gt, pred, spacing


def normalize_ct(ct_slice, window_center=50, window_width=400):
    low  = window_center - window_width / 2
    high = window_center + window_width / 2
    return np.clip((ct_slice - low) / (high - low), 0.0, 1.0)


def find_best_slice(gt, pred, axis, label=2):
    union    = ((gt == label) | (pred == label)).astype(int)
    sum_axes = tuple(i for i in range(3) if i != axis)
    return int(np.argmax(union.sum(axis=sum_axes)))


def get_slice(vol, axis, idx):
    if axis == 0:   return vol[idx, :, :]
    elif axis == 1: return vol[:, idx, :]
    else:           return vol[:, :, idx]


def get_pixel_aspect(spacing, axis):
    """Return (row_spacing_mm, col_spacing_mm) for the given view axis.

    Corrected mapping (verified empirically against KiTS19 volumes):
      axis 0 → Coronal:   rows = SI (z_sp), cols = LR (x_sp)
      axis 1 → Sagittal:  rows = SI (z_sp), cols = AP (y_sp)
      axis 2 → Axial:     rows = AP (y_sp), cols = LR (x_sp)
    """
    x_sp, y_sp, z_sp = spacing
    if axis == 0:   return z_sp, x_sp
    elif axis == 1: return z_sp, y_sp
    else:           return y_sp, x_sp


def crop_to_roi(images, gt_s, pred_s, margin=30):
    combined = ((gt_s > 0) | (pred_s > 0)).astype(int)
    coords   = np.argwhere(combined > 0)
    if len(coords) == 0:
        return images, gt_s, pred_s
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    h, w  = gt_s.shape
    r_min = max(0, r_min - margin)
    r_max = min(h, r_max + margin + 1)
    c_min = max(0, c_min - margin)
    c_max = min(w, c_max + margin + 1)
    return ([s[r_min:r_max, c_min:c_max] for s in images],
            gt_s[r_min:r_max, c_min:c_max],
            pred_s[r_min:r_max, c_min:c_max])


def resize_slice(img, row_sp, col_sp):
    min_sp       = min(row_sp, col_sp)
    row_scale    = row_sp / min_sp
    col_scale    = col_sp / min_sp
    order        = 1 if img.ndim == 3 else 0
    zoom_factors = (row_scale, col_scale, 1) if img.ndim == 3 else (row_scale, col_scale)
    return zoom(img, zoom_factors, order=order)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def _outline(mask, width=2):
    return binary_dilation(mask, iterations=width) & ~mask


def make_overlay(ct_norm, mask, alpha=0.45):
    """CT with filled colour overlay and bright contour for each label."""
    rgb = np.stack([ct_norm] * 3, axis=-1)
    for label, color in [(1, KIDNEY_COLOR), (2, TUMOUR_COLOR)]:
        m       = mask == label
        outline = _outline(m)
        for c in range(3):
            rgb[:, :, c] = np.where(m,
                                    rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                                    rgb[:, :, c])
            rgb[:, :, c] = np.where(outline, color[c], rgb[:, :, c])
    return np.clip(rgb, 0, 1)


def make_overlap(ct_norm, gt_s, pred_s, alpha=0.65):
    """CT with TP/FP/FN colour coding on tumour pixels."""
    rgb = np.stack([ct_norm] * 3, axis=-1)
    kidney = gt_s == 1
    for c in range(3):
        rgb[:, :, c] = np.where(kidney,
                                rgb[:, :, c] * 0.6 + KIDNEY_COLOR[c] * 0.4,
                                rgb[:, :, c])
    gt_t   = gt_s   == 2
    pred_t = pred_s == 2
    for m, color in [(gt_t & pred_t,   TP_COLOR),
                     (~gt_t & pred_t,  FP_COLOR),
                     (gt_t & ~pred_t,  FN_COLOR)]:
        outline = _outline(m)
        for c in range(3):
            rgb[:, :, c] = np.where(m,
                                    rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                                    rgb[:, :, c])
            rgb[:, :, c] = np.where(outline, color[c], rgb[:, :, c])
    return np.clip(rgb, 0, 1)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _add_orient_labels(ax, view_name):
    orient  = VIEW_ORIENT.get(view_name, {})
    stroke  = [pe.withStroke(linewidth=2.5, foreground='black')]
    kw = dict(color='white', fontsize=11, fontweight='bold',
               path_effects=stroke, transform=ax.transAxes, clip_on=True)
    if orient.get("top"):
        ax.text(0.50, 0.97, orient["top"],    ha='center', va='top',    **kw)
    if orient.get("bottom"):
        ax.text(0.50, 0.03, orient["bottom"], ha='center', va='bottom', **kw)
    if orient.get("left"):
        ax.text(0.02, 0.50, orient["left"],   ha='left',   va='center', **kw)
    if orient.get("right"):
        ax.text(0.98, 0.50, orient["right"],  ha='right',  va='center', **kw)


def _build_figure(gt_img, pred_img, overlap_img, case_name, desc, view_name, dice):
    h, w         = gt_img.shape[:2]
    panel_width  = 6.0
    panel_height = panel_width * (h / w)
    fig_width    = panel_width * 3 + 1.5
    fig_height   = panel_height + 2.8

    BG = '#111111'
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BG)

    panel_titles = ['Ground Truth', 'Model Prediction', 'Overlap Analysis']
    for ax, img, title in zip(axes, [gt_img, pred_img, overlap_img], panel_titles):
        ax.imshow(img, aspect='equal', interpolation='bicubic')
        ax.set_title(title, fontweight='bold', color='white', pad=8, fontsize=17)
        ax.axis('off')
        ax.set_facecolor(BG)
        _add_orient_labels(ax, view_name)

    legend_patches = [
        mpatches.Patch(facecolor=KIDNEY_COLOR, edgecolor='white', lw=0.8, label='Kidney'),
        mpatches.Patch(facecolor=TUMOUR_COLOR, edgecolor='white', lw=0.8, label='Tumour'),
        mpatches.Patch(facecolor=TP_COLOR,     edgecolor='white', lw=0.8, label='Correct (TP)'),
        mpatches.Patch(facecolor=FP_COLOR,     edgecolor='white', lw=0.8, label='Over-segmented (FP)'),
        mpatches.Patch(facecolor=FN_COLOR,     edgecolor='white', lw=0.8, label='Missed (FN)'),
    ]
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=5,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.01),
        frameon=True,
        facecolor='#222222',
        edgecolor='#555555',
        labelcolor='white',
    )
    fig.suptitle(
        f'{case_name}  —  {desc}  —  {view_name} View  |  Tumour Dice: {dice:.3f}',
        fontweight='bold',
        color='white',
        y=0.97,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13, top=0.91, wspace=0.04)
    return fig


# ---------------------------------------------------------------------------
# Main visualisation
# ---------------------------------------------------------------------------

def visualize_view(cfg, case_name, dice, desc, axis_idx, view_name, out_dir):
    ct, gt, pred, spacing = load_case(cfg, case_name)
    best_z = find_best_slice(gt, pred, axis_idx)

    ct_s   = normalize_ct(get_slice(ct,   axis_idx, best_z))
    gt_s   = get_slice(gt,   axis_idx, best_z)
    pred_s = get_slice(pred, axis_idx, best_z)

    [ct_s], gt_s, pred_s = crop_to_roi([ct_s], gt_s, pred_s, margin=30)

    row_sp, col_sp = get_pixel_aspect(spacing, axis_idx)
    ct_s   = resize_slice(ct_s,   row_sp, col_sp)
    gt_s   = resize_slice(gt_s,   row_sp, col_sp)
    pred_s = resize_slice(pred_s, row_sp, col_sp)

    gt_img      = make_overlay(ct_s, gt_s)
    pred_img    = make_overlay(ct_s, pred_s)
    overlap_img = make_overlap(ct_s, gt_s, pred_s)

    stem = f'{case_name}_{view_name.lower()}'
    fig  = _build_figure(gt_img, pred_img, overlap_img,
                         case_name, desc, view_name, dice)

    # PNG — high-DPI raster
    png_path = out_dir / f'{stem}.png'
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor())

    # PDF — vector text/legend, rasterized image content at OUTPUT_DPI
    pdf_path = out_dir / f'{stem}.pdf'
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor(),
                    dpi=OUTPUT_DPI)

    plt.close(fig)
    print(f"    Saved: {stem}.png + .pdf")


def main(cfg):
    out_dir = Path(cfg["paths"]["final_output"]).parent / "figures" / "case_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("case_00067",  0.9848, "Best Case"),
        ("case_00001",  0.8153, "Hardest Case"),
        ("case_00184",  0.9316, "Multi-Tumour (11 regions)"),
    ]
    # Axis → plane mapping verified empirically against KiTS19 volumes
    views = [(0, "Coronal"), (1, "Sagittal"), (2, "Axial")]

    print("=== Generating Case Visualizations ===")
    for case_name, dice, desc in cases:
        print(f"\n  {case_name} ({desc})")
        for axis_idx, view_name in views:
            visualize_view(cfg, case_name, dice, desc, axis_idx, view_name, out_dir)

    n_png = len(list(out_dir.glob('*.png')))
    n_pdf = len(list(out_dir.glob('*.pdf')))
    print(f"\nAll saved to: {out_dir}")
    print(f"Total: {n_png} PNG files, {n_pdf} PDF files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
