# /home/coder/projects/MedNeXt/src/train_mednext_kits19.py
#
# MedNeXt for KiTS19 Kidney & Tumour Segmentation
#
# Paper: Roy et al. "MedNeXt: Transformer-driven Scaling of ConvNets for
#        Medical Image Segmentation", MICCAI 2023. arXiv:2303.09975
#
# Architecture:
#   - Fully ConvNeXt 3D Encoder-Decoder (4 enc + bottleneck + 4 dec layers)
#   - Each block: depthwise conv (large kernel) -> LayerNorm -> 1x1 expand
#                 -> GELU -> 1x1 compress  (inverted bottleneck, Transformer-style)
#   - Residual ConvNeXt up/downsampling blocks (preserve semantic richness)
#   - Deep supervision at each decoder scale (weights: 1.0, 0.5, 0.25, 0.125)
#   - UpKern: initialise large-kernel model from trained small-kernel model
#             (we use kernel_size=3 directly — no UpKern needed at this size)
#   - Model variant B, kernel 3x3x3: ~30M params, fits V100 32GB at bs=2
#
# Run:
#   cd /home/coder/projects/MedNeXt
#   source .venv/bin/activate
#   screen -S mednext
#   python -u src/train_mednext_kits19.py \
#     2>&1 | tee outputs/logs/train_$(date +%Y%m%d_%H%M%S).log

import os, re, json, time, random, math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class CFG:
    seed: int = 42

    # Paths
    data_dir: str    = "/home/coder/kits19/data"
    project_dir: str = ""
    outputs_dir: str = ""

    # Training
    epochs: int          = 100
    batch_size: int      = 2
    lr: float            = 1e-4
    weight_decay: float  = 1e-5
    num_workers: int     = 4
    amp: bool            = True
    grad_clip: float     = 1.0
    steps_per_epoch: int = 250
    val_steps: int       = 60
    print_every_pct: int = 10

    # Patch / crop (same for train and val — no train/val distribution gap)
    crop_xyz: Tuple = (128, 128, 128)
    fg_oversample_prob: float = 0.7

    # CT windowing
    win_center: float = 40.0
    win_width: float  = 400.0

    # MedNeXt variant
    # model_id: S (~5M), B (~30M), M (~60M), L (~120M)
    # kernel_size: 3 or 5 (5 requires UpKern init from 3-kernel model)
    model_id: str    = "B"
    kernel_size: int = 3
    deep_supervision: bool = True

    # Deep supervision loss weights (main -> coarsest decoder output)
    ds_weights: Tuple = (1.0, 0.5, 0.25, 0.125)

    # Output
    out_ch: int = 2   # kidney, tumour


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def window_ct(x, center, width):
    lo, hi = center - width/2, center + width/2
    return ((np.clip(x, lo, hi) - lo) / (hi - lo + 1e-8)).astype(np.float32)

def load_nifti(p):
    return nib.load(p).get_fdata().astype(np.float32)

def case_id(name):
    m = re.match(r"case_(\d{5})$", name)
    return int(m.group(1)) if m else None

def list_cases(root):
    out = []
    for name in sorted(os.listdir(root)):
        cid = case_id(name)
        if cid is None: continue
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "segmentation.nii.gz")):
            out.append(cid)
    return out


# ─────────────────────────────────────────────
# Per-worker LRU volume cache
# ─────────────────────────────────────────────
@lru_cache(maxsize=6)
def load_volume_cached(case_dir: str, win_center: float, win_width: float):
    ct  = load_nifti(os.path.join(case_dir, "imaging.nii.gz"))
    seg = load_nifti(os.path.join(case_dir, "segmentation.nii.gz")).astype(np.uint8)
    ct  = window_ct(ct, win_center, win_width)
    ct  = (ct - ct.mean()) / (ct.std() + 1e-6)
    mk  = np.stack([((seg==1)|(seg==2)).astype(np.float32),
                     (seg==2).astype(np.float32)], axis=0)   # (2, X, Y, Z)
    return ct, mk


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred, target, eps=1e-6):
    """pred/target: (B,1,X,Y,Z) binary float."""
    pred = pred.float(); target = target.float()
    inter = (pred * target).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    d = (2*inter + eps) / (denom + eps)
    return torch.where(denom == 0, torch.ones_like(d), d).squeeze(1).mean()

def dice_posonly(pred, target, eps=1e-6):
    """Only count samples where target has foreground."""
    ts   = target.float().sum(dim=(2,3,4)).squeeze(1)
    keep = ts > 0
    if keep.sum() == 0: return None
    p2 = pred[keep]; t2 = target[keep]
    inter = (p2.float() * t2.float()).sum(dim=(2,3,4))
    denom = p2.float().sum(dim=(2,3,4)) + t2.float().sum(dim=(2,3,4))
    return ((2*inter + eps) / (denom + eps)).squeeze(1).mean()


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def soft_dice_loss(logits, target, eps=1e-6):
    """logits/target: (B, C, X, Y, Z)"""
    p    = torch.sigmoid(logits)
    loss = 0.0
    C    = logits.shape[1]
    for c in range(C):
        inter = (p[:,c] * target[:,c]).sum(dim=(1,2,3))
        denom = p[:,c].sum(dim=(1,2,3)) + target[:,c].sum(dim=(1,2,3))
        loss += (1 - (2*inter + eps) / (denom + eps)).mean()
    return loss / C

def focal_loss(logits, target, gamma=1.5, pos_weight=None):
    """logits/target: (B, C, X, Y, Z)"""
    C  = logits.shape[1]
    pw = (pos_weight if pos_weight is not None
          else torch.ones(C, device=logits.device)).view(1, C, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float(),
                                              pos_weight=pw, reduction="none")
    pt  = torch.where(target > 0.5,
                      torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    return (((1 - pt) ** gamma) * bce).mean()

def seg_loss(logits, target, pos_weight=None):
    return soft_dice_loss(logits, target) + focal_loss(logits, target,
                                                        pos_weight=pos_weight)

def deep_supervision_loss(outputs, target, weights, pos_weight=None):
    """
    outputs: list of tensors at different resolutions (highest first)
    target:  full-resolution mask (B, C, X, Y, Z)
    weights: loss weight per scale
    """
    total = 0.0
    for out, w in zip(outputs, weights):
        if w == 0.0: continue
        # Downsample target to match this output's spatial size
        if out.shape[2:] != target.shape[2:]:
            tgt_ds = F.interpolate(target.float(), size=out.shape[2:],
                                   mode="nearest")
        else:
            tgt_ds = target.float()
        total += w * seg_loss(out, tgt_ds, pos_weight)
    return total


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class KiTS19Dataset(Dataset):
    def __init__(self, root, ids, cfg, mode):
        self.root = root
        self.ids  = ids
        self.cfg  = cfg
        self.mode = mode
        self.rng  = random.Random(cfg.seed + (0 if mode == "train" else 9999))

        # Build fg slice index from segmentation only (fast)
        print(f"  Indexing {len(ids)} {mode} cases...", flush=True)
        self.fg_cases = []   # cases that have any foreground
        for i, cid in enumerate(ids):
            d   = os.path.join(root, f"case_{cid:05d}")
            seg = load_nifti(os.path.join(d, "segmentation.nii.gz")).astype(np.uint8)
            if seg.max() > 0:
                self.fg_cases.append(cid)
            if (i+1) % 30 == 0 or (i+1) == len(ids):
                print(f"    [{i+1}/{len(ids)}]  fg_cases_so_far={len(self.fg_cases)}",
                      flush=True)
        print(f"  {mode}: {len(ids)} total, {len(self.fg_cases)} with foreground",
              flush=True)

    def __len__(self): return len(self.ids)

    def _random_crop(self, ct, mk):
        cx, cy, cz = self.cfg.crop_xyz
        X, Y, Z    = ct.shape
        cl = lambda a, lo, hi: max(lo, min(hi, a))

        # Foreground oversampling: pick a random fg voxel as crop centre
        if self.rng.random() < self.cfg.fg_oversample_prob and mk[0].sum() > 0:
            fg = np.argwhere(mk[0] > 0)
            ix, iy, iz = fg[self.rng.randrange(len(fg))]
        else:
            ix = self.rng.randrange(X)
            iy = self.rng.randrange(Y)
            iz = self.rng.randrange(Z)

        x0 = cl(ix - cx//2, 0, max(0, X-cx))
        y0 = cl(iy - cy//2, 0, max(0, Y-cy))
        z0 = cl(iz - cz//2, 0, max(0, Z-cz))

        ct_c = ct[x0:x0+cx, y0:y0+cy, z0:z0+cz]
        mk_c = mk[:, x0:x0+cx, y0:y0+cy, z0:z0+cz]

        # Pad if volume smaller than crop (rare in KiTS19 but safe)
        px = max(0, cx - ct_c.shape[0])
        py = max(0, cy - ct_c.shape[1])
        pz = max(0, cz - ct_c.shape[2])
        if px or py or pz:
            ct_c = np.pad(ct_c, ((0,px),(0,py),(0,pz)))
            mk_c = np.pad(mk_c, ((0,0),(0,px),(0,py),(0,pz)))
        return ct_c, mk_c

    def _augment(self, ct, mk):
        """Light augmentation: random flips along each axis."""
        for ax in range(3):
            if self.rng.random() < 0.5:
                ct  = np.flip(ct,  axis=ax).copy()
                mk  = np.flip(mk,  axis=ax+1).copy()  # mk has channel dim first
        return ct, mk

    def __getitem__(self, idx):
        # Oversample foreground cases
        if self.mode == "train" and self.rng.random() < self.cfg.fg_oversample_prob:
            cid = self.fg_cases[self.rng.randrange(len(self.fg_cases))]
        else:
            cid = self.ids[idx]

        d        = os.path.join(self.root, f"case_{cid:05d}")
        ct, mk   = load_volume_cached(d, self.cfg.win_center, self.cfg.win_width)
        ct_c, mk_c = self._random_crop(ct, mk)

        if self.mode == "train":
            ct_c, mk_c = self._augment(ct_c, mk_c)

        return (torch.from_numpy(ct_c[None]).float(),   # (1, X, Y, Z)
                torch.from_numpy(mk_c).float())          # (2, X, Y, Z)


# ─────────────────────────────────────────────
# Model import
# ─────────────────────────────────────────────
def build_model(cfg: CFG) -> nn.Module:
    """
    Import MedNeXt from the cloned repo and build the model.
    Tries two import paths to handle different repo layouts.
    """
    # Try all known API variants across different MedNeXt repo versions
    errors = []

    # Variant 1: README API — num_channels / num_classes
    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            num_channels     = 1,
            num_classes      = cfg.out_ch,
            model_id         = cfg.model_id,
            kernel_size      = cfg.kernel_size,
            deep_supervision = cfg.deep_supervision,
        )
        print(f"  Imported via create_mednext_v1 (num_channels API)", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"v1 num_channels: {e}")

    # Variant 2: in_channels / n_channels / n_classes
    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            in_channels      = 1,
            n_channels       = 32,
            n_classes        = cfg.out_ch,
            model_id         = cfg.model_id,
            kernel_size      = cfg.kernel_size,
            deep_supervision = cfg.deep_supervision,
        )
        print(f"  Imported via create_mednext_v1 (in_channels API)", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"v1 in_channels: {e}")

    # Variant 3: direct MedNeXt class (correct import path per GitHub issue #22)
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
        model = MedNeXt(
            in_channels      = 1,
            n_channels       = 32,
            n_classes        = cfg.out_ch,
            exp_r            = 4,
            kernel_size      = cfg.kernel_size,
            deep_supervision = cfg.deep_supervision,
            do_res           = True,
            do_res_up_down   = True,
            block_counts     = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
        print(f"  Imported via MedNextV1.MedNeXt direct", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"direct MedNeXt: {e}")

    # Show all errors so user can see exactly what failed
    for err in errors:
        print(f"  IMPORT ATTEMPT FAILED: {err}", flush=True)

    raise ImportError(
        "Could not import MedNeXt. Make sure you ran:\n"
        "  cd mednext_repo && pip install -e . && cd .."
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    cfg.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg.outputs_dir = os.path.join(cfg.project_dir, "outputs")
    ckpt_dir = os.path.join(cfg.outputs_dir, "checkpoints")
    for d in [ckpt_dir, os.path.join(cfg.outputs_dir, "logs")]:
        os.makedirs(d, exist_ok=True)

    if not os.path.isdir(cfg.data_dir):
        raise RuntimeError(f"Data not found: {cfg.data_dir}")

    ts             = time.strftime("%Y%m%d_%H%M%S")
    metrics_latest = os.path.join(cfg.outputs_dir, "metrics_latest.json")
    metrics_final  = os.path.join(cfg.outputs_dir, f"metrics_run_{ts}.json")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"device={device}  gpu={gpu_name}", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Building MedNeXt-{cfg.model_id} kernel={cfg.kernel_size}...", flush=True)
    model = build_model(cfg).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_total/1e6:.1f}M total, {n_train/1e6:.1f}M trainable",
          flush=True)

    # Output bias init to prevent sigmoid ~ 0.5 at step 0
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == cfg.out_ch:
            if m.bias is not None:
                nn.init.constant_(m.bias, -2.944)   # sigmoid(-2.944) ~ 0.05

    # ── Data ─────────────────────────────────────────────────────────────────
    labeled_ids = sorted(list_cases(cfg.data_dir))
    print(f"labeled_cases={len(labeled_ids)}", flush=True)

    train_ids = labeled_ids[:min(180, len(labeled_ids))]
    val_ids   = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]
    print(f"train={len(train_ids)}  val={len(val_ids)}", flush=True)

    train_ds = KiTS19Dataset(cfg.data_dir, train_ids, cfg, "train")
    val_ds   = KiTS19Dataset(cfg.data_dir, val_ids,   cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=max(0, cfg.num_workers//2),
                              pin_memory=True, drop_last=False,
                              persistent_workers=(max(0, cfg.num_workers//2) > 0))

    # ── Optimiser ────────────────────────────────────────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
    def lr_lambda(ep):
        warmup = 5
        if ep < warmup: return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, cfg.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    # Positive class weights: kidney ~5% of crop, tumour ~2%
    pw = torch.tensor([5.0, 15.0], device=device)

    best_mean_dice = -1.0
    history: List[Dict] = []
    run_start = time.time()

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
        "model": f"MedNeXt-{cfg.model_id}-k{cfg.kernel_size}",
        "params_M": round(n_total/1e6, 1),
        "cfg": {k: str(v) for k, v in cfg.__dict__.items()},
        "best_mean_dice": best_mean_dice,
        "history": history, "status": "running",
    }
    write_json(metrics_latest, summary)

    def finalize(status, error=None):
        summary.update({"status": status, "best_mean_dice": best_mean_dice,
                        "total_time_s": time.time() - run_start})
        if error: summary["error"] = error
        write_json(metrics_latest, summary)
        write_json(metrics_final, summary)
        print(f"\n=== RUN END ===  status={status}  "
              f"best_mean_dice={best_mean_dice:.4f}  "
              f"time={summary['total_time_s']:.0f}s\n"
              f"saved: {metrics_final}", flush=True)

    tpe = max(1, int(round(cfg.steps_per_epoch * cfg.print_every_pct / 100)))
    vpe = max(1, int(round(cfg.val_steps       * cfg.print_every_pct / 100)))

    def actx():
        return torch.amp.autocast("cuda", enabled=cfg.amp) if device == "cuda" \
               else torch.amp.autocast("cpu", enabled=False)

    try:
        for ep in range(1, cfg.epochs + 1):
            ep_start = time.time()
            cur_lr   = opt.param_groups[0]["lr"]
            print(f"\n=== Epoch {ep}/{cfg.epochs}  lr={cur_lr:.2e} ===", flush=True)

            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            r_loss = r_dk = r_dt = 0.0
            it = iter(train_loader)

            for step in range(cfg.steps_per_epoch):
                try:    ct, mk = next(it)
                except: it = iter(train_loader); ct, mk = next(it)

                ct = ct.to(device, non_blocking=True)
                mk = mk.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with actx():
                    out = model(ct)

                    # MedNeXt with deep_supervision=True returns a list of
                    # tensors [full_res, half_res, quarter_res, eighth_res]
                    if isinstance(out, (list, tuple)):
                        loss = deep_supervision_loss(out, mk, cfg.ds_weights, pw)
                        logits = out[0]   # full-resolution output for metrics
                    else:
                        loss   = seg_loss(out, mk, pw)
                        logits = out

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()

                with torch.no_grad():
                    pred  = (torch.sigmoid(logits) > 0.5).float()
                    r_dk += dice_score(pred[:,0:1], mk[:,0:1]).item()
                    r_dt += dice_score(pred[:,1:2], mk[:,1:2]).item()
                r_loss += float(loss.item())

                if (step+1) % tpe == 0 or (step+1) == cfg.steps_per_epoch:
                    n   = step + 1
                    pct = int(round(100 * n / cfg.steps_per_epoch))
                    print(f"Train {pct:3d}%  loss={r_loss/n:.4f}  "
                          f"kidney_dice={r_dk/n:.4f}  tumor_dice={r_dt/n:.4f}",
                          flush=True)

            sched.step()
            train_loss = r_loss / cfg.steps_per_epoch

            # ── Validation ────────────────────────────────────────────────────
            print("Running validation...", flush=True)
            model.eval()
            v_loss = k_sum = t_sum = 0.0
            count  = 0
            it     = iter(val_loader)

            with torch.no_grad():
                for step in range(cfg.val_steps):
                    try:    ct, mk = next(it)
                    except: it = iter(val_loader); ct, mk = next(it)

                    ct = ct.to(device, non_blocking=True)
                    mk = mk.to(device, non_blocking=True)

                    with actx():
                        out  = model(ct)
                        if isinstance(out, (list, tuple)):
                            loss   = deep_supervision_loss(out, mk, cfg.ds_weights, pw)
                            logits = out[0]
                        else:
                            loss   = seg_loss(out, mk, pw)
                            logits = out
                    v_loss += float(loss.item())

                    pred    = (torch.sigmoid(logits) > 0.5).float()
                    tgt     = mk.float()
                    kd      = dice_score(pred[:,0:1], tgt[:,0:1]).item()
                    td      = dice_score(pred[:,1:2], tgt[:,1:2]).item()
                    kd_pos  = dice_posonly(pred[:,0:1], tgt[:,0:1])
                    td_pos  = dice_posonly(pred[:,1:2], tgt[:,1:2])
                    kd_pv   = float(kd_pos.item()) if kd_pos is not None else float("nan")
                    td_pv   = float(td_pos.item()) if td_pos is not None else float("nan")
                    k_sum  += kd; t_sum += td; count += 1

                    # Debug first 3 batches
                    if step < 3:
                        pk = torch.sigmoid(logits[:,0:1])
                        pt = torch.sigmoid(logits[:,1:2])
                        print(
                            f"[val debug] step={step} "
                            f"tgt_k={int(tgt[:,0:1].sum())} tgt_t={int(tgt[:,1:2].sum())} "
                            f"pred_k={int(pred[:,0:1].sum())} pred_t={int(pred[:,1:2].sum())} "
                            f"p_k={pk.mean():.3f}/{pk.max():.3f} "
                            f"p_t={pt.mean():.3f}/{pt.max():.3f} "
                            f"kd_pos={kd_pv:.4f}  td_pos={td_pv:.4f}",
                            flush=True)

                    if (step+1) % vpe == 0 or (step+1) == cfg.val_steps:
                        pct = int(round(100*(step+1)/cfg.val_steps))
                        print(f"Val {pct:3d}%  loss={v_loss/(step+1):.4f}  "
                              f"kidney_dice={k_sum/count:.4f}  "
                              f"tumor_dice={t_sum/count:.4f}  "
                              f"mean_dice={(k_sum+t_sum)/(2*count):.4f}",
                              flush=True)

            val_loss    = v_loss  / max(1, cfg.val_steps)
            kidney_dice = k_sum   / max(1, count)
            tumor_dice  = t_sum   / max(1, count)
            mean_dice   = (kidney_dice + tumor_dice) / 2

            # Checkpointing
            last_path = os.path.join(ckpt_dir, "mednext_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "mednext_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "cfg": cfg.__dict__}, best_path)

            ep_time = time.time() - ep_start
            print(f"epoch {ep}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"kidney_dice={kidney_dice:.4f}  tumor_dice={tumor_dice:.4f}  "
                  f"mean_dice={mean_dice:.4f}  best={best_mean_dice:.4f}  "
                  f"time={ep_time:.0f}s", flush=True)

            history.append({
                "epoch": ep, "train_loss": train_loss, "val_loss": val_loss,
                "kidney_dice": kidney_dice, "tumor_dice": tumor_dice,
                "mean_dice": mean_dice, "best_mean_dice": best_mean_dice,
                "epoch_time_s": ep_time,
                "checkpoint_best": os.path.relpath(best_path, cfg.project_dir)
                                   if best_path else None,
            })
            summary.update({"best_mean_dice": best_mean_dice, "history": history,
                            "total_time_s": time.time() - run_start})
            write_json(metrics_latest, summary)

        finalize("completed")

    except KeyboardInterrupt:
        finalize("interrupted")
    except Exception as e:
        finalize("error", error=repr(e))
        raise


if __name__ == "__main__":
    main()