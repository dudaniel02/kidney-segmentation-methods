# /home/coder/projects/TwoStageSeg/src/train_stage2.py
#
# Two-Stage Segmentation — Stage 2: Kidney + Tumour within ROI
#
# Goal: Train MedNeXt-B on kidney ROI crops extracted using Stage 1's
#       predicted bounding box. The tight ROI crop means:
#         - Tumour goes from ~1% of full volume -> ~15-25% of the ROI crop
#         - Model never sees background-only patches
#         - Tumour dice should climb significantly vs single-stage
#
# Pipeline at training time:
#   1. Load full CT + seg
#   2. Run Stage 1 model (frozen) -> kidney mask at 64^3
#   3. Upsample mask to original resolution -> compute bbox + padding
#   4. Crop CT and seg to bbox -> resize crop to 128^3
#   5. Train MedNeXt-B on (128^3 crop, kidney+tumour masks)
#
# Pipeline at inference time (handled by inference_twostage.py):
#   Same steps 1-4, then forward through Stage 2 model.
#
# Run:
#   cd /home/coder/projects/TwoStageSeg
#   source .venv/bin/activate
#   nohup python -u src/train_stage2.py \
#     > outputs/logs/stage2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > outputs/stage2.pid
#   tail -f outputs/logs/stage2_*.log

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

    # Stage 1 checkpoint — used to generate ROI crops during training
    stage1_ckpt: str = ""   # filled at runtime from outputs/checkpoints/stage1_best.pt

    # Training
    epochs: int           = 100
    batch_size: int       = 2
    lr: float             = 1e-4
    weight_decay: float   = 1e-5
    num_workers: int      = 4
    amp: bool             = True
    grad_clip: float      = 1.0
    steps_per_epoch: int  = 250
    val_steps: int        = 30   # ~2 full passes over 30 val cases at bs=2
    print_every_pct: int  = 10

    # Stage 1 input size (must match train_stage1.py)
    stage1_input_size: Tuple = (64, 64, 64)

    # Stage 2 crop size — ROI crop resized to this before MedNeXt-B
    crop_size: Tuple = (128, 128, 128)

    # Bounding box padding (voxels at original resolution)
    # Generous padding so we never clip the kidney edge
    bbox_pad: int = 20

    # Fallback: if Stage 1 predicts empty mask, use this fixed crop
    # centred on the volume (rare but must be handled)
    fallback_crop_frac: float = 0.6

    # CT windowing
    win_center: float = 40.0
    win_width: float  = 400.0

    # MedNeXt-B for Stage 2
    model_id: str    = "B"
    kernel_size: int = 3
    deep_supervision: bool = True
    ds_weights: Tuple = (1.0, 0.5, 0.25, 0.125)

    # Output: kidney + tumour (2 channels)
    out_ch: int = 2


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
    pred = pred.float(); target = target.float()
    inter = (pred * target).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    d = (2*inter + eps) / (denom + eps)
    return torch.where(denom == 0, torch.ones_like(d), d).squeeze(1).mean()

def dice_posonly(pred, target, eps=1e-6):
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
def soft_dice_loss(logits, target, eps=1e-6, smooth=0.05):
    p = torch.sigmoid(logits)
    C = logits.shape[1]
    loss = 0.0
    # Label smoothing: prevents overconfident predictions on training set
    target_s = target.float() * (1 - smooth) + smooth * 0.5
    for c in range(C):
        inter = (p[:,c] * target_s[:,c]).sum(dim=(1,2,3))
        denom = p[:,c].sum(dim=(1,2,3)) + target_s[:,c].sum(dim=(1,2,3))
        loss += (1 - (2*inter + eps) / (denom + eps)).mean()
    return loss / C

def focal_loss(logits, target, gamma=1.5, pos_weight=None):
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
    total = 0.0
    for out, w in zip(outputs, weights):
        if w == 0.0: continue
        tgt_ds = F.interpolate(target.float(), size=out.shape[2:],
                               mode="nearest") if out.shape[2:] != target.shape[2:] \
                 else target.float()
        total += w * seg_loss(out, tgt_ds, pos_weight)
    return total


# ─────────────────────────────────────────────
# Stage 1 model (frozen, CPU inference for bbox)
# ─────────────────────────────────────────────
def build_stage1_model(ckpt_path: str, input_size: Tuple, device: str) -> nn.Module:
    """Load the trained Stage 1 MedNeXt-S for bbox extraction."""
    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
        model = MedNeXt(
            in_channels=1, n_channels=32, n_classes=1,
            exp_r=2, kernel_size=3, deep_supervision=False,
            do_res=True, do_res_up_down=True,
            block_counts=[2,2,2,2,2,2,2,2,2],
        )
    except ImportError:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(num_channels=1, num_classes=1,
                                   model_id="S", kernel_size=3,
                                   deep_supervision=False)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in sd: sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Keep on CPU to save GPU memory for Stage 2 training
    return model.cpu()


# ─────────────────────────────────────────────
# ROI extraction
# ─────────────────────────────────────────────
def get_kidney_bbox(ct: np.ndarray, stage1_model: nn.Module,
                    stage1_input_size: Tuple, bbox_pad: int,
                    fallback_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Stage 1 on a full CT volume to get kidney bounding box.

    Returns:
        bbox_min: (3,) array of [x0, y0, z0] at original resolution
        bbox_max: (3,) array of [x1, y1, z1] at original resolution
    """
    X, Y, Z = ct.shape

    # Downsample CT to stage1 input size
    ct_t = torch.from_numpy(ct[None, None]).float()   # (1,1,X,Y,Z)
    ct_ds = F.interpolate(ct_t, size=stage1_input_size,
                           mode="trilinear", align_corners=False)  # (1,1,64,64,64)

    with torch.no_grad():
        logits = stage1_model(ct_ds)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        sig     = torch.sigmoid(logits)
        # Lower threshold — Stage1 runs on CPU without AMP so outputs
        # can be slightly lower confidence than during GPU training
        pred_ds = (sig > 0.3).float()   # (1,1,64,64,64)

    # Upsample prediction back to original resolution
    pred_full = F.interpolate(pred_ds, size=(X, Y, Z),
                               mode="nearest").squeeze().numpy()  # (X,Y,Z)

    fg = np.argwhere(pred_full > 0)

    if len(fg) == 0:
        # Fallback: use central fraction of volume
        cx = int(X * (1 - fallback_frac) / 2)
        cy = int(Y * (1 - fallback_frac) / 2)
        cz = int(Z * (1 - fallback_frac) / 2)
        bbox_min = np.array([cx, cy, cz])
        bbox_max = np.array([X-cx, Y-cy, Z-cz])
        print("  [WARN] Stage1 predicted empty mask — using fallback bbox", flush=True)
    else:
        bbox_min = np.maximum(fg.min(0) - bbox_pad, 0)
        bbox_max = np.minimum(fg.max(0) + bbox_pad, np.array([X, Y, Z]))

    return bbox_min, bbox_max


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class Stage2Dataset(Dataset):
    """
    For each case:
      1. Load full CT + masks
      2. Run Stage 1 (CPU, frozen) to get kidney bbox
      3. Crop CT + masks to bbox
      4. Resize crop to cfg.crop_size (128^3)
      5. Return (ct_crop, kidney_mask, tumour_mask)

    The bbox is computed once per case per epoch call and cached
    via the LRU volume cache. Stage 1 inference on CPU at 64^3
    takes ~0.1s per case — negligible vs data loading.
    """
    def __init__(self, root, ids, cfg, mode, stage1_model):
        self.root          = root
        self.ids           = ids
        self.cfg           = cfg
        self.mode          = mode
        self.stage1_model  = stage1_model
        self.rng           = random.Random(cfg.seed + (0 if mode=="train" else 9999))

        # Pre-compute bboxes for all cases at init time
        # This avoids running Stage 1 on every __getitem__ call
        print(f"  Pre-computing Stage1 bboxes for {len(ids)} {mode} cases...",
              flush=True)
        self.bboxes = {}
        for i, cid in enumerate(ids):
            d      = os.path.join(root, f"case_{cid:05d}")
            ct, _  = load_volume_cached(d, cfg.win_center, cfg.win_width)
            bmin, bmax = get_kidney_bbox(ct, stage1_model,
                                          cfg.stage1_input_size,
                                          cfg.bbox_pad,
                                          cfg.fallback_crop_frac)
            self.bboxes[cid] = (bmin, bmax)
            if (i+1) % 30 == 0 or (i+1) == len(ids):
                print(f"    [{i+1}/{len(ids)}] done", flush=True)

        print(f"  {mode}: bboxes ready", flush=True)

    def __len__(self): return len(self.ids)

    def _augment(self, ct, mk):
        # Flips
        for ax in range(3):
            if self.rng.random() < 0.5:
                ct = np.flip(ct, axis=ax).copy()
                mk = np.flip(mk, axis=ax+1).copy()
        # Gamma — simulates scanner variability in HU response
        if self.rng.random() < 0.4:
            gamma   = self.rng.uniform(0.7, 1.5)
            ct_min  = ct.min(); ct_max = ct.max()
            if ct_max > ct_min:
                ct_norm = (ct - ct_min) / (ct_max - ct_min + 1e-8)
                ct = (ct_norm ** gamma) * (ct_max - ct_min) + ct_min
        # Gaussian noise
        if self.rng.random() < 0.3:
            ct = (ct + np.random.normal(0, self.rng.uniform(0.01, 0.05),
                  ct.shape)).astype(np.float32)
        # Intensity shift + scale
        if self.rng.random() < 0.4:
            ct = (ct * self.rng.uniform(0.85, 1.15)
                     + self.rng.uniform(-0.15, 0.15)).astype(np.float32)
        return ct, mk

    def __getitem__(self, idx):
        cid       = self.ids[idx]
        d         = os.path.join(self.root, f"case_{cid:05d}")
        ct, mk    = load_volume_cached(d, self.cfg.win_center, self.cfg.win_width)
        bmin, bmax = self.bboxes[cid]
        # Bbox jitter during training: +-10 vox random shift so the model
        # never sees the exact same crop twice
        if self.mode == 'train':
            X, Y, Z = ct.shape
            jitter  = np.array([self.rng.randint(-10, 10) for _ in range(3)])
            bmin    = np.clip(bmin + jitter, 0, np.array([X,Y,Z]) - 1)
            bmax    = np.clip(bmax + jitter, bmin + 1, np.array([X,Y,Z]))

        # Crop to kidney ROI
        x0,y0,z0 = bmin; x1,y1,z1 = bmax
        ct_crop = ct[x0:x1, y0:y1, z0:z1]           # (X', Y', Z')
        mk_crop = mk[:, x0:x1, y0:y1, z0:z1]        # (2, X', Y', Z')

        if self.mode == "train":
            ct_crop, mk_crop = self._augment(ct_crop, mk_crop)

        # Resize to fixed crop_size
        ct_t = torch.from_numpy(ct_crop[None, None]).float()   # (1,1,X',Y',Z')
        mk_t = torch.from_numpy(mk_crop[None]).float()          # (1,2,X',Y',Z')

        ct_r = F.interpolate(ct_t, size=self.cfg.crop_size,
                              mode="trilinear", align_corners=False).squeeze(0)
        mk_r = F.interpolate(mk_t, size=self.cfg.crop_size,
                              mode="nearest").squeeze(0)

        return ct_r, mk_r   # (1,128,128,128), (2,128,128,128)


# ─────────────────────────────────────────────
# Stage 2 model (MedNeXt-B)
# ─────────────────────────────────────────────
def build_stage2_model(cfg: CFG) -> nn.Module:
    errors = []

    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            num_channels=1, num_classes=cfg.out_ch,
            model_id=cfg.model_id, kernel_size=cfg.kernel_size,
            deep_supervision=cfg.deep_supervision,
        )
        print(f"  MedNeXt-{cfg.model_id} via num_channels API", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"num_channels: {e}")

    try:
        from nnunet_mednext import create_mednext_v1
        model = create_mednext_v1(
            in_channels=1, n_channels=32, n_classes=cfg.out_ch,
            model_id=cfg.model_id, kernel_size=cfg.kernel_size,
            deep_supervision=cfg.deep_supervision,
        )
        print(f"  MedNeXt-{cfg.model_id} via in_channels API", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"in_channels: {e}")

    try:
        from nnunet_mednext.network_architecture.mednextv1.MedNextV1 import MedNeXt
        model = MedNeXt(
            in_channels=1, n_channels=32, n_classes=cfg.out_ch,
            exp_r=4, kernel_size=cfg.kernel_size,
            deep_supervision=cfg.deep_supervision,
            do_res=True, do_res_up_down=True,
            block_counts=[2,2,2,2,2,2,2,2,2],
        )
        print(f"  MedNeXt-{cfg.model_id} via direct class", flush=True)
        return model
    except (ImportError, TypeError) as e:
        errors.append(f"direct class: {e}")

    for err in errors: print(f"  FAILED: {err}", flush=True)
    raise ImportError("Could not import MedNeXt.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    cfg.project_dir  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg.outputs_dir  = os.path.join(cfg.project_dir, "outputs")
    cfg.stage1_ckpt  = os.path.join(cfg.outputs_dir, "checkpoints", "stage1_best.pt")
    ckpt_dir         = os.path.join(cfg.outputs_dir, "checkpoints")
    for d in [ckpt_dir, os.path.join(cfg.outputs_dir, "logs")]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(cfg.stage1_ckpt):
        raise RuntimeError(
            f"Stage 1 checkpoint not found: {cfg.stage1_ckpt}\n"
            f"Train Stage 1 first: python src/train_stage1.py")

    ts             = time.strftime("%Y%m%d_%H%M%S")
    metrics_latest = os.path.join(cfg.outputs_dir, "stage2_metrics_latest.json")
    metrics_final  = os.path.join(cfg.outputs_dir, f"stage2_metrics_{ts}.json")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"device={device}  gpu={gpu_name}", flush=True)

    # ── Stage 1 model (CPU, frozen) ──────────────────────────────────────────
    print(f"Loading Stage 1 from {cfg.stage1_ckpt}...", flush=True)
    stage1 = build_stage1_model(cfg.stage1_ckpt, cfg.stage1_input_size, device)
    print(f"  Stage 1 loaded and frozen on CPU", flush=True)

    # ── Stage 2 model ────────────────────────────────────────────────────────
    print(f"Building Stage 2 MedNeXt-{cfg.model_id} kernel={cfg.kernel_size}...",
          flush=True)
    model = build_stage2_model(cfg).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_total/1e6:.1f}M", flush=True)

    for m in model.modules():
        if isinstance(m, nn.Conv3d) and m.out_channels == cfg.out_ch:
            if m.bias is not None:
                nn.init.constant_(m.bias, -2.944)

    # ── Data ─────────────────────────────────────────────────────────────────
    labeled_ids = sorted(list_cases(cfg.data_dir))
    train_ids   = labeled_ids[:min(180, len(labeled_ids))]
    val_ids     = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]
    print(f"cases: train={len(train_ids)}  val={len(val_ids)}", flush=True)

    train_ds = Stage2Dataset(cfg.data_dir, train_ids, cfg, "train", stage1)
    val_ds   = Stage2Dataset(cfg.data_dir, val_ids,   cfg, "val",   stage1)

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

    # Kidney ~40-60% of ROI crop, tumour ~15-25% — much better than full volume
    # Lowered from [3, 10] — high train/val gap means model is overconfident
    # on tumor. Reducing focal pressure lets dice loss drive learning instead.
    pw = torch.tensor([3.0, 5.0], device=device)

    best_mean_dice = -1.0
    history: List[Dict] = []
    start_epoch = 1
    run_start = time.time()

    # ── Resume from checkpoint if available ──────────────────────────────────
    resume_ckpt = os.path.join(ckpt_dir, "stage2_last.pt")
    if os.path.exists(resume_ckpt):
        print(f"Resuming from {resume_ckpt}...", flush=True)
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Restore best dice from metrics file if available
        if os.path.exists(metrics_latest):
            try:
                prev = json.load(open(metrics_latest))
                best_mean_dice = prev.get("best_mean_dice", -1.0)
                history        = prev.get("history", [])
                print(f"  Restored best_mean_dice={best_mean_dice:.4f}  "
                      f"start_epoch={start_epoch}", flush=True)
            except Exception:
                pass
    else:
        print("No checkpoint found — training from scratch.", flush=True)

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
        "stage": 2, "model": f"MedNeXt-{cfg.model_id}-k{cfg.kernel_size}",
        "params_M": round(n_total/1e6, 1),
        "stage1_ckpt": cfg.stage1_ckpt,
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
        print(f"\n=== STAGE 2 END ===  status={status}  "
              f"best_mean_dice={best_mean_dice:.4f}  "
              f"time={summary['total_time_s']:.0f}s\n"
              f"saved: {metrics_final}", flush=True)

    tpe = max(1, int(round(cfg.steps_per_epoch * cfg.print_every_pct / 100)))
    vpe = max(1, int(round(cfg.val_steps       * cfg.print_every_pct / 100)))

    def actx():
        return torch.amp.autocast("cuda", enabled=cfg.amp) if device == "cuda" \
               else torch.amp.autocast("cpu", enabled=False)

    try:
        for ep in range(start_epoch, cfg.epochs + 1):
            ep_start = time.time()
            cur_lr   = opt.param_groups[0]["lr"]
            print(f"\n=== Stage2 Epoch {ep}/{cfg.epochs}  lr={cur_lr:.2e} ===",
                  flush=True)

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
                    if isinstance(out, (list, tuple)):
                        loss   = deep_supervision_loss(out, mk, cfg.ds_weights, pw)
                        logits = out[0]
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
                        out = model(ct)
                        if isinstance(out, (list, tuple)):
                            loss   = deep_supervision_loss(out, mk, cfg.ds_weights, pw)
                            logits = out[0]
                        else:
                            loss   = seg_loss(out, mk, pw)
                            logits = out
                    v_loss += float(loss.item())

                    sig    = torch.sigmoid(logits)
                    pred_k = (sig[:,0:1] > 0.5).float()
                    pred_t = (sig[:,1:2] > 0.3).float()
                    pred   = torch.cat([pred_k, pred_t], dim=1)
                    tgt    = mk.float()
                    kd     = dice_score(pred[:,0:1], tgt[:,0:1]).item()
                    td     = dice_score(pred[:,1:2], tgt[:,1:2]).item()
                    kd_pos = dice_posonly(pred[:,0:1], tgt[:,0:1])
                    td_pos = dice_posonly(pred[:,1:2], tgt[:,1:2])
                    kd_pv  = float(kd_pos.item()) if kd_pos is not None else float("nan")
                    td_pv  = float(td_pos.item()) if td_pos is not None else float("nan")
                    k_sum += kd; t_sum += td; count += 1

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

            val_loss    = v_loss / max(1, cfg.val_steps)
            kidney_dice = k_sum  / max(1, count)
            tumor_dice  = t_sum  / max(1, count)
            mean_dice   = (kidney_dice + tumor_dice) / 2

            last_path = os.path.join(ckpt_dir, "stage2_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "stage2_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "cfg": cfg.__dict__}, best_path)
                print(f"  *** New best mean_dice={best_mean_dice:.4f} — "
                      f"saved stage2_best.pt ***", flush=True)

            ep_time = time.time() - ep_start
            print(f"epoch {ep}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
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
    main()cd ~/projects/3DNNUNET
kill 1410353
rm -rf logs/*
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
python src/scripts/03_train_stage1.py --config configs/config.yaml --fold 0