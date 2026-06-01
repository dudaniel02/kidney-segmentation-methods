# /home/coder/projects/TwoStageSeg/src/train_stage1.py
#
# Two-Stage Segmentation — Stage 1: Kidney Localisation
#
# Goal: Train MedNeXt-S on the full CT volume (downsampled) to produce
#       a coarse kidney mask. This mask is used at inference time to
#       compute a tight bounding box, which Stage 2 uses as its ROI crop.
#
# Design decisions:
#   - MedNeXt-S (small, ~5M params): speed over accuracy — we only need
#     a good bounding box, not a perfect segmentation
#   - Input: full volume downsampled to 96x96x96 (fits on V100 at bs=2)
#   - Output: 1 channel — kidney (label 1 or 2, i.e. kidney+tumor region)
#   - No tumor output here — that is entirely Stage 2's job
#   - Loss: soft dice + focal, no deep supervision (keep it simple)
#
# Run:
#   cd /home/coder/projects/TwoStageSeg
#   source .venv/bin/activate
#   nohup python -u src/train_stage1.py \
#     > outputs/logs/stage1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > outputs/stage1.pid
#   tail -f outputs/logs/stage1_*.log

import os, re, json, time, random, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
    epochs: int           = 80
    batch_size: int       = 2
    lr: float             = 1e-4
    weight_decay: float   = 1e-5
    num_workers: int      = 4
    amp: bool             = True
    grad_clip: float      = 1.0
    steps_per_epoch: int  = 200
    val_steps: int        = 50
    print_every_pct: int  = 10

    # Input volume size — full CT downsampled to this before feeding the model
    # 96^3 fits comfortably at bs=2 on V100 32GB with MedNeXt-S
    input_size: Tuple = (96, 96, 96)

    # CT windowing
    win_center: float = 40.0
    win_width: float  = 400.0

    # MedNeXt variant — S (small) is enough for localisation
    model_id: str    = "S"
    kernel_size: int = 3

    # Stage 1 output: kidney only (1 channel)
    # kidney = seg==1 OR seg==2 (the full kidney+tumor region)
    out_ch: int = 1

    # Bounding box padding added at inference time (in voxels at full res)
    # Stage 2 will use: bbox expanded by this amount on each side
    bbox_pad_vox: int = 20


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
    # Stage 1: kidney mask only (kidney+tumor region as one class)
    mk  = ((seg == 1) | (seg == 2)).astype(np.float32)[None]  # (1, X, Y, Z)
    return ct, mk


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred, target, eps=1e-6):
    """pred/target: (B, 1, X, Y, Z) binary float."""
    pred = pred.float(); target = target.float()
    inter = (pred * target).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    d = (2*inter + eps) / (denom + eps)
    return torch.where(denom == 0, torch.ones_like(d), d).squeeze(1).mean()

def bbox_iou(pred_mask, gt_mask):
    """
    Compute 3D bounding box IoU between predicted and GT kidney masks.
    This is the key Stage 1 metric — we care about bbox quality, not
    perfect voxel-wise segmentation.
    pred_mask/gt_mask: (X, Y, Z) binary numpy arrays
    """
    def get_bbox(m):
        fg = np.argwhere(m > 0)
        if len(fg) == 0: return None
        return fg.min(0), fg.max(0)   # (min_xyz, max_xyz)

    pb = get_bbox(pred_mask)
    gb = get_bbox(gt_mask)
    if pb is None or gb is None: return 0.0

    pmin, pmax = pb; gmin, gmax = gb
    # Intersection
    imin = np.maximum(pmin, gmin)
    imax = np.minimum(pmax, gmax)
    if np.any(imax < imin): return 0.0
    inter = np.prod(imax - imin + 1)
    # Union
    pvol = np.prod(pmax - pmin + 1)
    gvol = np.prod(gmax - gmin + 1)
    return float(inter / (pvol + gvol - inter + 1e-6))


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def soft_dice_loss(logits, target, eps=1e-6):
    p     = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1,2,3,4))
    denom = p.sum(dim=(1,2,3,4)) + target.sum(dim=(1,2,3,4))
    return (1 - (2*inter + eps) / (denom + eps)).mean()

def focal_loss(logits, target, gamma=1.5, pos_weight=None):
    pw  = (pos_weight if pos_weight is not None
           else torch.ones(1, device=logits.device)).view(1, 1, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float(),
                                              pos_weight=pw, reduction="none")
    pt  = torch.where(target > 0.5,
                      torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    return (((1 - pt) ** gamma) * bce).mean()

def seg_loss(logits, target, pos_weight=None):
    return soft_dice_loss(logits, target) + focal_loss(logits, target,
                                                        pos_weight=pos_weight)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Stage1Dataset(Dataset):
    """
    Loads full CT volumes, resizes to cfg.input_size, returns
    (ct, kidney_mask) pairs. No patch cropping — full volume only.
    The model sees the whole patient every step.
    """
    def __init__(self, root, ids, cfg, mode):
        self.root = root
        self.ids  = ids
        self.cfg  = cfg
        self.mode = mode
        self.rng  = random.Random(cfg.seed + (0 if mode == "train" else 9999))
        print(f"  Stage1Dataset {mode}: {len(ids)} cases", flush=True)

    def __len__(self): return len(self.ids)

    def _augment(self, ct, mk):
        """Random flips along each axis — cheap and effective for 3D."""
        for ax in range(3):
            if self.rng.random() < 0.5:
                ct = np.flip(ct, axis=ax).copy()
                mk = np.flip(mk, axis=ax+1).copy()
        return ct, mk

    def __getitem__(self, idx):
        cid    = self.ids[idx]
        d      = os.path.join(self.root, f"case_{cid:05d}")
        ct, mk = load_volume_cached(d, self.cfg.win_center, self.cfg.win_width)

        if self.mode == "train":
            ct, mk = self._augment(ct, mk)

        # Resize full volume to fixed input_size
        ct_t = torch.from_numpy(ct[None, None]).float()   # (1,1,X,Y,Z)
        mk_t = torch.from_numpy(mk[None]).float()          # (1,1,X,Y,Z)

        ct_r = F.interpolate(ct_t, size=self.cfg.input_size,
                              mode="trilinear", align_corners=False).squeeze(0)
        mk_r = F.interpolate(mk_t, size=self.cfg.input_size,
                              mode="nearest").squeeze(0)

        return ct_r, mk_r   # (1, 96, 96, 96), (1, 96, 96, 96)


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
def build_model(cfg: CFG) -> nn.Module:
    errors = []

    # Variant 1: num_channels / num_classes API
    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            num_channels     = 1,
            num_classes      = cfg.out_ch,
            model_id         = cfg.model_id,
            kernel_size      = cfg.kernel_size,
            deep_supervision = False,   # not needed for localisation
        )
        print(f"  MedNeXt-{cfg.model_id} via num_channels API", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"num_channels API: {e}")

    # Variant 2: in_channels / n_channels / n_classes API
    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            in_channels      = 1,
            n_channels       = 32,
            n_classes        = cfg.out_ch,
            model_id         = cfg.model_id,
            kernel_size      = cfg.kernel_size,
            deep_supervision = False,
        )
        print(f"  MedNeXt-{cfg.model_id} via in_channels API", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"in_channels API: {e}")

    # Variant 3: direct class import
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
        model = MedNeXt(
            in_channels      = 1,
            n_channels       = 32,
            n_classes        = cfg.out_ch,
            exp_r            = 2,        # S variant uses exp_r=2
            kernel_size      = cfg.kernel_size,
            deep_supervision = False,
            do_res           = True,
            do_res_up_down   = True,
            block_counts     = [2, 2, 2, 2, 2, 2, 2, 2, 2],
        )
        print(f"  MedNeXt-{cfg.model_id} via direct MedNeXt class", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"direct class: {e}")

    for err in errors:
        print(f"  IMPORT FAILED: {err}", flush=True)
    raise ImportError("Could not import MedNeXt — check mednext_repo is installed.")


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

    ts             = time.strftime("%Y%m%d_%H%M%S")
    metrics_latest = os.path.join(cfg.outputs_dir, "stage1_metrics_latest.json")
    metrics_final  = os.path.join(cfg.outputs_dir, f"stage1_metrics_{ts}.json")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"device={device}  gpu={gpu_name}", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Building Stage1 MedNeXt-{cfg.model_id} kernel={cfg.kernel_size}...",
          flush=True)
    model = build_model(cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_total/1e6:.1f}M", flush=True)

    # Bias init: sigmoid(-2.944) ~ 0.05 to avoid predicting everything at step 0
    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == cfg.out_ch:
            if m.bias is not None:
                nn.init.constant_(m.bias, -2.944)

    # ── Data ─────────────────────────────────────────────────────────────────
    labeled_ids = sorted(list_cases(cfg.data_dir))
    train_ids   = labeled_ids[:min(180, len(labeled_ids))]
    val_ids     = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]
    print(f"cases: train={len(train_ids)}  val={len(val_ids)}", flush=True)

    train_ds = Stage1Dataset(cfg.data_dir, train_ids, cfg, "train")
    val_ds   = Stage1Dataset(cfg.data_dir, val_ids,   cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
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

    # Kidney covers ~5-10% of the full downsampled volume
    pw = torch.tensor([8.0], device=device)

    best_kidney_dice = -1.0
    history: List[Dict] = []
    run_start = time.time()

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
        "stage": 1, "model": f"MedNeXt-{cfg.model_id}-k{cfg.kernel_size}",
        "params_M": round(n_total/1e6, 1),
        "cfg": {k: str(v) for k, v in cfg.__dict__.items()},
        "best_kidney_dice": best_kidney_dice,
        "history": history, "status": "running",
    }
    write_json(metrics_latest, summary)

    def finalize(status, error=None):
        summary.update({"status": status,
                        "best_kidney_dice": best_kidney_dice,
                        "total_time_s": time.time() - run_start})
        if error: summary["error"] = error
        write_json(metrics_latest, summary)
        write_json(metrics_final, summary)
        print(f"\n=== STAGE 1 END ===  status={status}  "
              f"best_kidney_dice={best_kidney_dice:.4f}  "
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
            print(f"\n=== Stage1 Epoch {ep}/{cfg.epochs}  lr={cur_lr:.2e} ===",
                  flush=True)

            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            r_loss = r_dk = 0.0
            it = iter(train_loader)

            for step in range(cfg.steps_per_epoch):
                try:    ct, mk = next(it)
                except: it = iter(train_loader); ct, mk = next(it)

                ct = ct.to(device, non_blocking=True)
                mk = mk.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with actx():
                    logits = model(ct)
                    # Handle deep_supervision=False but model still returns list
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    loss = seg_loss(logits, mk, pw)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()

                with torch.no_grad():
                    pred  = (torch.sigmoid(logits) > 0.5).float()
                    r_dk += dice_score(pred, mk).item()
                r_loss += float(loss.item())

                if (step+1) % tpe == 0 or (step+1) == cfg.steps_per_epoch:
                    n   = step + 1
                    pct = int(round(100 * n / cfg.steps_per_epoch))
                    print(f"Train {pct:3d}%  loss={r_loss/n:.4f}  "
                          f"kidney_dice={r_dk/n:.4f}", flush=True)

            sched.step()
            train_loss   = r_loss / cfg.steps_per_epoch
            train_kdice  = r_dk   / cfg.steps_per_epoch

            # ── Validation ────────────────────────────────────────────────────
            print("Running validation...", flush=True)
            model.eval()
            v_loss = k_sum = bbox_iou_sum = 0.0
            count  = 0
            it     = iter(val_loader)

            with torch.no_grad():
                for step in range(cfg.val_steps):
                    try:    ct, mk = next(it)
                    except: it = iter(val_loader); ct, mk = next(it)

                    ct = ct.to(device, non_blocking=True)
                    mk = mk.to(device, non_blocking=True)

                    with actx():
                        logits = model(ct)
                        if isinstance(logits, (list, tuple)):
                            logits = logits[0]
                        loss = seg_loss(logits, mk, pw)
                    v_loss += float(loss.item())

                    pred = (torch.sigmoid(logits) > 0.5).float()
                    kd   = dice_score(pred, mk).item()
                    k_sum += kd

                    # Compute bbox IoU — the metric that actually matters for
                    # Stage 1, since Stage 2 depends on bbox quality
                    pred_np = pred[0, 0].cpu().numpy()
                    gt_np   = mk[0, 0].cpu().numpy()
                    biou    = bbox_iou(pred_np, gt_np)
                    bbox_iou_sum += biou
                    count += 1

                    if step < 3:
                        pk = torch.sigmoid(logits)
                        print(
                            f"[val debug] step={step} "
                            f"tgt_k={int(mk.sum())} pred_k={int(pred.sum())} "
                            f"p_k={pk.mean():.3f}/{pk.max():.3f} "
                            f"kidney_dice={kd:.4f}  bbox_iou={biou:.4f}",
                            flush=True)

                    if (step+1) % vpe == 0 or (step+1) == cfg.val_steps:
                        pct = int(round(100*(step+1)/cfg.val_steps))
                        print(f"Val {pct:3d}%  loss={v_loss/(step+1):.4f}  "
                              f"kidney_dice={k_sum/count:.4f}  "
                              f"bbox_iou={bbox_iou_sum/count:.4f}",
                              flush=True)

            val_loss    = v_loss        / max(1, cfg.val_steps)
            kidney_dice = k_sum         / max(1, count)
            mean_biou   = bbox_iou_sum  / max(1, count)

            # Save checkpoints
            last_path = os.path.join(ckpt_dir, "stage1_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if kidney_dice > best_kidney_dice:
                best_kidney_dice = kidney_dice
                best_path = os.path.join(ckpt_dir, "stage1_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "cfg": cfg.__dict__}, best_path)
                print(f"  *** New best kidney_dice={best_kidney_dice:.4f} — "
                      f"saved stage1_best.pt ***", flush=True)

            ep_time = time.time() - ep_start
            print(f"epoch {ep}: train_loss={train_loss:.4f}  "
                  f"train_kidney={train_kdice:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_kidney={kidney_dice:.4f}  "
                  f"bbox_iou={mean_biou:.4f}  "
                  f"best={best_kidney_dice:.4f}  "
                  f"time={ep_time:.0f}s", flush=True)

            history.append({
                "epoch": ep,
                "train_loss": train_loss, "train_kidney_dice": train_kdice,
                "val_loss": val_loss, "val_kidney_dice": kidney_dice,
                "bbox_iou": mean_biou,
                "best_kidney_dice": best_kidney_dice,
                "epoch_time_s": ep_time,
                "checkpoint_best": os.path.relpath(best_path, cfg.project_dir)
                                   if best_path else None,
            })
            summary.update({"best_kidney_dice": best_kidney_dice,
                            "history": history,
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