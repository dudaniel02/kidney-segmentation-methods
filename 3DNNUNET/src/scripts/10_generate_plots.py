#!/usr/bin/env python3
"""Generate all thesis figures as PDF vector graphics."""

import argparse
import csv
import re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.rcsetup  # noqa – ensure rcParams available
import numpy as np
from utils.config import load_config

# ---------------------------------------------------------------------------
# Global style — applied once so every figure inherits these defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':        14,
    'axes.titlesize':   17,
    'axes.labelsize':   15,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  13,
    'figure.titlesize': 18,
    'axes.linewidth':   1.2,
    'lines.linewidth':  2.0,
    'pdf.fonttype':     42,   # embed TrueType fonts in PDF (required by many journals)
    'ps.fonttype':      42,
})

KIDNEY_COLOR   = '#2ca02c'
TUMOUR_COLOR   = '#d62728'
MEAN_LINE_COLOR = '#1f77b4'

EXT = '.pdf'   # vector format — scales to any size without blur


# ---------------------------------------------------------------------------
# Log parsing helpers (unchanged)
# ---------------------------------------------------------------------------

def parse_last_log_only(log_dir):
    """Parse only the last (complete) training log in a fold directory."""
    log_files = sorted(log_dir.glob("training_log_*.txt"))
    if not log_files:
        return {'train_loss': [], 'val_loss': [], 'dice_scores': [], 'learning_rates': []}

    log_file = log_files[-1]
    train_losses, val_losses, dice_scores, lrs = [], [], [], []
    current_epoch, current_lr = None, None

    with open(log_file) as f:
        for line in f:
            line = line.strip()
            m = re.search(r'Epoch (\d+)\s*$', line)
            if m:
                current_epoch = int(m.group(1))
            m = re.search(r'Current learning rate: ([\d.e-]+)', line)
            if m:
                current_lr = float(m.group(1))
            m = re.search(r'train_loss ([-\d.]+)', line)
            if m and current_epoch is not None:
                train_losses.append((current_epoch, float(m.group(1))))
            m = re.search(r'val_loss ([-\d.]+)', line)
            if m and current_epoch is not None:
                val_losses.append((current_epoch, float(m.group(1))))
            m = re.search(r'Pseudo dice \[(.*?)\]', line)
            if m and current_epoch is not None:
                dice_vals = []
                for v in re.findall(r'[\d.]+', m.group(1)):
                    try:
                        fv = float(v)
                        if 0 <= fv <= 1:
                            dice_vals.append(fv)
                    except ValueError:
                        pass
                if dice_vals:
                    dice_scores.append((current_epoch, dice_vals))
                if current_lr is not None:
                    lrs.append((current_epoch, current_lr))

    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'dice_scores': dice_scores,
        'learning_rates': lrs,
    }


def align_folds(all_folds_data, key, max_epochs=1000):
    arrays = []
    for data in all_folds_data:
        if not data[key]:
            continue
        vals = np.full(max_epochs, np.nan)
        for epoch, v in data[key]:
            if epoch < max_epochs:
                vals[epoch] = v
        arrays.append(vals)
    if not arrays:
        return None, None, None
    stacked = np.array(arrays)
    with np.errstate(all='ignore'):
        mean = np.nanmean(stacked, axis=0)
        std  = np.nanstd(stacked, axis=0)
    epochs = np.arange(max_epochs)
    mask = ~np.isnan(mean)
    return epochs[mask], mean[mask], std[mask]


def align_folds_dice(all_folds_data, cls_idx, max_epochs=1000):
    arrays = []
    for data in all_folds_data:
        if not data['dice_scores']:
            continue
        vals = np.full(max_epochs, np.nan)
        for epoch, dices in data['dice_scores']:
            if epoch < max_epochs and cls_idx < len(dices):
                vals[epoch] = dices[cls_idx]
        arrays.append(vals)
    if not arrays:
        return None, None, None
    stacked = np.array(arrays)
    with np.errstate(all='ignore'):
        mean = np.nanmean(stacked, axis=0)
        std  = np.nanstd(stacked, axis=0)
    epochs = np.arange(max_epochs)
    mask = ~np.isnan(mean)
    return epochs[mask], mean[mask], std[mask]


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_loss_curves(all_folds, stage_name, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (key, title) in enumerate([('train_loss', 'Training Loss'), ('val_loss', 'Validation Loss')]):
        ax = axes[i]
        epochs, mean, std = align_folds(all_folds, key)
        if epochs is not None:
            ax.plot(epochs, mean, color=MEAN_LINE_COLOR, label='Mean (5 folds)')
            ax.fill_between(epochs, mean - std, mean + std,
                            color=MEAN_LINE_COLOR, alpha=0.2, label='±1 std')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{stage_name} — {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major')

    fig.suptitle(f'{stage_name} Loss Curves', y=1.01)
    plt.tight_layout()
    fname = f'{stage_name.lower().replace(" ", "_")}_loss_curves{EXT}'
    plt.savefig(out_dir / fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_dice_curves(all_folds, stage_name, out_dir, class_names):
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6))
    if n == 1:
        axes = [axes]
    colors = [KIDNEY_COLOR, TUMOUR_COLOR]

    for cls_idx, (ax, cls_name) in enumerate(zip(axes, class_names)):
        c = colors[cls_idx] if cls_idx < len(colors) else MEAN_LINE_COLOR
        epochs, mean, std = align_folds_dice(all_folds, cls_idx)
        if epochs is not None:
            ax.plot(epochs, mean, color=c, label='Mean (5 folds)')
            ax.fill_between(epochs, mean - std, mean + std,
                            color=c, alpha=0.2, label='±1 std')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Pseudo Dice')
        ax.set_title(f'{stage_name} — {cls_name} Dice per Epoch')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major')

    fig.suptitle(f'{stage_name} Dice Curves', y=1.01)
    plt.tight_layout()
    fname = f'{stage_name.lower().replace(" ", "_")}_dice_curves{EXT}'
    plt.savefig(out_dir / fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_lr(all_folds, stage_name, out_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    if all_folds[0]['learning_rates']:
        e, lr = zip(*all_folds[0]['learning_rates'])
        ax.plot(e, lr, color=MEAN_LINE_COLOR)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{stage_name} — Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    fname = f'{stage_name.lower().replace(" ", "_")}_lr_schedule{EXT}'
    plt.savefig(out_dir / fname, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fname}")


def plot_dice_boxplot(csv_path, out_dir):
    rows = list(csv.DictReader(open(csv_path)))
    kd = [float(r['kidney_dice']) for r in rows]
    td = [float(r['tumour_dice']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [kd, td],
        labels=['Kidney', 'Tumour'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        flierprops=dict(marker='o', markersize=6, alpha=0.6),
        widths=0.5,
    )
    bp['boxes'][0].set(facecolor=KIDNEY_COLOR, alpha=0.5)
    bp['boxes'][1].set(facecolor=TUMOUR_COLOR, alpha=0.5)
    ax.scatter(
        [1, 2], [np.mean(kd), np.mean(td)],
        marker='D', color='white', edgecolors='black', s=80, zorder=3,
        label=f'Mean  (Kidney = {np.mean(kd):.3f},  Tumour = {np.mean(td):.3f})',
    )
    ax.set_ylabel('Dice Score')
    ax.set_title('Per-Case Dice Score Distribution')
    ax.set_ylim(0.75, 1.01)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(out_dir / f'dice_boxplot{EXT}', bbox_inches='tight')
    plt.close()
    print(f"  Saved: dice_boxplot{EXT}")


def plot_hd95_boxplot(csv_path, out_dir):
    rows = list(csv.DictReader(open(csv_path)))
    if 'kidney_hd95' not in rows[0]:
        return
    kh = [float(r['kidney_hd95']) for r in rows if float(r['kidney_hd95']) < 1000]
    th = [float(r['tumour_hd95']) for r in rows if float(r['tumour_hd95']) < 1000]

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(
        [kh, th],
        labels=['Kidney', 'Tumour'],
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2.5),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        flierprops=dict(marker='o', markersize=6, alpha=0.6),
        widths=0.5,
    )
    bp['boxes'][0].set(facecolor=KIDNEY_COLOR, alpha=0.5)
    bp['boxes'][1].set(facecolor=TUMOUR_COLOR, alpha=0.5)
    ax.scatter(
        [1, 2], [np.mean(kh), np.mean(th)],
        marker='D', color='white', edgecolors='black', s=80, zorder=3,
        label=f'Mean  (Kidney = {np.mean(kh):.1f} mm,  Tumour = {np.mean(th):.1f} mm)',
    )
    ax.set_ylabel('HD95 (mm)')
    ax.set_title('Per-Case Hausdorff Distance (95th Percentile)')
    ax.set_ylim(0, max(np.percentile(kh, 99), np.percentile(th, 99)) * 1.3)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(out_dir / f'hd95_boxplot{EXT}', bbox_inches='tight')
    plt.close()
    print(f"  Saved: hd95_boxplot{EXT}")


def plot_tumour_histogram(csv_path, out_dir):
    rows = list(csv.DictReader(open(csv_path)))
    td = [float(r['tumour_dice']) for r in rows]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(td, bins=25, color=TUMOUR_COLOR, alpha=0.7, edgecolor='black', linewidth=0.6)
    ax.axvline(np.mean(td),   color='black', linestyle='--', linewidth=2.0,
               label=f'Mean = {np.mean(td):.3f}')
    ax.axvline(np.median(td), color='gray',  linestyle=':',  linewidth=2.0,
               label=f'Median = {np.median(td):.3f}')
    ax.set_xlabel('Tumour Dice Score')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Distribution of Tumour Dice Scores')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(out_dir / f'tumour_dice_histogram{EXT}', bbox_inches='tight')
    plt.close()
    print(f"  Saved: tumour_dice_histogram{EXT}")


def plot_comparison(out_dir):
    methods = ['Stage 1 Only\n(5-fold ensemble)', 'Stage 1 + Stage 2\n(Two-stage pipeline)']
    kidney  = [0.978, 0.979]
    tumour  = [0.932, 0.946]
    x = np.arange(len(methods))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w / 2, kidney, w, label='Kidney Dice',
                color=KIDNEY_COLOR, alpha=0.75, edgecolor='black', linewidth=1.2)
    b2 = ax.bar(x + w / 2, tumour, w, label='Tumour Dice',
                color=TUMOUR_COLOR, alpha=0.75, edgecolor='black', linewidth=1.2)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                f'{b.get_height():.3f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dice Score')
    ax.set_title('Effect of Stage 2 Tumour Refinement')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.88, 1.0)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major')
    plt.tight_layout()
    plt.savefig(out_dir / f'stage_comparison{EXT}', bbox_inches='tight')
    plt.close()
    print(f"  Saved: stage_comparison{EXT}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg):
    out_dir = Path(cfg["paths"]["final_output"]).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path(cfg["paths"]["nnunet_results"])
    plans   = cfg['architecture']['plans_name']
    trainer = cfg['architecture']['trainer']
    folds   = cfg["training"]["folds"]

    s1_base = results_base / cfg["datasets"]["stage1_name"] / f"{trainer}__{plans}__3d_fullres"
    s2_base = results_base / cfg["datasets"]["stage2_name"] / f"{trainer}__{plans}__3d_fullres"

    print("\n=== Stage 1 Training Curves ===")
    s1_data = []
    for fold in folds:
        fd = s1_base / f"fold_{fold}"
        if fd.exists():
            d = parse_last_log_only(fd)
            s1_data.append(d)
            print(f"  Fold {fold}: {len(d['train_loss'])} epochs")
        else:
            s1_data.append({'train_loss': [], 'val_loss': [], 'dice_scores': [], 'learning_rates': []})

    if any(d['train_loss'] for d in s1_data):
        plot_loss_curves(s1_data, "Stage 1", out_dir)
        plot_dice_curves(s1_data, "Stage 1", out_dir, ["Kidney", "Tumour"])
        plot_lr(s1_data, "Stage 1", out_dir)

    print("\n=== Stage 2 Training Curves ===")
    s2_data = []
    for fold in folds:
        fd = s2_base / f"fold_{fold}"
        if fd.exists():
            d = parse_last_log_only(fd)
            s2_data.append(d)
            print(f"  Fold {fold}: {len(d['train_loss'])} epochs")
        else:
            s2_data.append({'train_loss': [], 'val_loss': [], 'dice_scores': [], 'learning_rates': []})

    if any(d['train_loss'] for d in s2_data):
        plot_loss_curves(s2_data, "Stage 2", out_dir)
        plot_dice_curves(s2_data, "Stage 2", out_dir, ["Tumour"])
        plot_lr(s2_data, "Stage 2", out_dir)

    print("\n=== Evaluation Plots ===")
    raw_csv = Path(cfg["paths"]["final_output"]) / "evaluation_results.csv"
    if raw_csv.exists():
        plot_dice_boxplot(raw_csv, out_dir)
        plot_hd95_boxplot(raw_csv, out_dir)
        plot_tumour_histogram(raw_csv, out_dir)

    print("\n=== Comparison Plot ===")
    plot_comparison(out_dir)

    print(f"\nAll figures saved to: {out_dir}")
    print(f"Total PDFs: {len(list(out_dir.glob('*.pdf')))} plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
