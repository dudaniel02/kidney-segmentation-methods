# /home/coder/projects/VISTA3D/src/train_vista3d_kits19.py
#
# VISTA3D Fine-tuning on KiTS19 Kidney & Tumour Segmentation
#
# Paper: He et al. "VISTA3D: Versatile Imaging SegmenTation and Annotation
#        model for 3D Computed Tomography", CVPR 2025. arXiv:2406.05285
#
# Approach: Fine-tune the pretrained VISTA3D foundation model on KiTS19.
#   - Load pretrained SegResNet weights (trained on 11,454 CT volumes)
#   - Replace/add KiTS19-specific output heads for kidney + tumour
#   - Fine-tune with patch-based 3D training (128^3 crops, fg oversampling)
#   - Use VISTA3D's automatic branch only (class-prompt driven)
#
# Architecture (auto-branch only, from paper):
#   - Encoder: SegResNet (shared)
#   - Auto-decoder: SegResNet decoder + skip connections
#   - Auto-head: linear(256 -> 1) applied per class via class embedding lookup
#   - KiTS19 mapping: kidney=class_idx 2, tumour=class_idx 3 (left+right kidney)
#
# Run:
#   cd /home/coder/projects/VISTA3D
#   source .venv/bin/activate
#   screen -S vista3d
#   python -u src/train_vista3d_kits19.py \
#     2>&1 | tee outputs/logs/train_$(date +%Y%m%d_%H%M%S).log

import os, re, json, time, random, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# MONAI imports
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    RandFlipd, RandRotate90d, RandScaleIntensityd,
    RandShiftIntensityd, Compose,
)


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class CFG:
    seed: int = 42

    # Paths
    pretrained_ckpt: str = "/home/coder/projects/VISTA3D/models/model.pt"
    data_dir: str        = "/home/coder/kits19/data"
    project_dir: str     = ""
    outputs_dir: str     = ""

    # Training
    epochs: int          = 60
    batch_size: int      = 2
    lr: float            = 2e-4
    weight_decay: float  = 1e-5
    num_workers: int     = 4
    amp: bool            = True
    grad_clip: float     = 1.0
    steps_per_epoch: int = 250
    val_steps: int       = 50
    print_every_pct: int = 10

    # Patch / crop
    crop_xyz: Tuple      = (128, 128, 128)
    fg_oversample_prob: float = 0.7

    # CT windowing
    win_center: float    = 40.0
    win_width: float     = 400.0

    # Model
    # SegResNet init_filters — must match pretrained checkpoint
    init_filters: int    = 32
    # Whether to freeze encoder and only train decoder+head
    # Set True for first few epochs then unfreeze
    freeze_encoder: bool = False

    # Fine-tuning strategy:
    # 'full'   — fine-tune all weights
    # 'head'   — freeze encoder+decoder, train head only
    # 'decoder'— freeze encoder only, train decoder+head
    finetune_mode: str   = "decoder"

    # Output: 2 classes (kidney, tumour) — binary per channel
    out_ch: int          = 2

    # Sliding window inference
    sw_roi_size: Tuple   = (128, 128, 128)
    sw_overlap: float    = 0.5


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
    labeled = []
    for name in sorted(os.listdir(root)):
        cid = case_id(name)
        if cid is None: continue
        d = os.path.join(root, name)
        if not os.path.isdir(d): continue
        if os.path.exists(os.path.join(d, "segmentation.nii.gz")):
            labeled.append(cid)
    return labeled


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
    ts = target.float().sum(dim=(2,3,4)).squeeze(1)
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
    p = torch.sigmoid(logits)
    loss = 0.0
    for c in range(logits.shape[1]):
        inter = (p[:,c] * target[:,c]).sum(dim=(1,2,3))
        denom = p[:,c].sum(dim=(1,2,3)) + target[:,c].sum(dim=(1,2,3))
        loss += (1 - (2*inter + eps) / (denom + eps)).mean()
    return loss / logits.shape[1]

def focal_loss(logits, target, gamma=1.0, pos_weight=None):
    C = logits.shape[1]
    if pos_weight is None:
        pos_weight = torch.ones(C, device=logits.device)
    pw  = pos_weight.view(1, C, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float(),
                                              pos_weight=pw, reduction="none")
    pt  = torch.where(target > 0.5, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    return (((1 - pt) ** gamma) * bce).mean()

def seg_loss(logits, target, pos_weight=None):
    return soft_dice_loss(logits, target) + focal_loss(logits, target, pos_weight=pos_weight)


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
        print(f"  {mode}: {len(ids)} cases", flush=True)

    def __len__(self): return len(self.ids)

    def _load(self, cid):
        d   = os.path.join(self.root, f"case_{cid:05d}")
        ct  = load_nifti(os.path.join(d, "imaging.nii.gz"))
        seg = load_nifti(os.path.join(d, "segmentation.nii.gz")).astype(np.uint8)
        ct  = window_ct(ct, self.cfg.win_center, self.cfg.win_width)
        ct  = (ct - ct.mean()) / (ct.std() + 1e-6)
        mk  = np.stack([((seg==1)|(seg==2)).astype(np.float32),
                         (seg==2).astype(np.float32)], axis=0)
        return ct, mk  # ct: (X,Y,Z)  mk: (2,X,Y,Z)

    def _pad(self, ct, mk):
        cx, cy, cz = self.cfg.crop_xyz
        px = max(0, cx - ct.shape[0])
        py = max(0, cy - ct.shape[1])
        pz = max(0, cz - ct.shape[2])
        if px or py or pz:
            ct = np.pad(ct, ((0,px),(0,py),(0,pz)))
            mk = np.pad(mk, ((0,0),(0,px),(0,py),(0,pz)))
        return ct, mk

    def _crop(self, ct, mk, x0, y0, z0):
        cx, cy, cz = self.cfg.crop_xyz
        ct_c = ct[x0:x0+cx, y0:y0+cy, z0:z0+cz]
        mk_c = mk[:, x0:x0+cx, y0:y0+cy, z0:z0+cz]
        return self._pad(ct_c, mk_c)

    def _random_crop(self, ct, mk):
        cx, cy, cz = self.cfg.crop_xyz
        X, Y, Z = ct.shape
        cl = lambda a, lo, hi: max(lo, min(hi, a))
        if self.rng.random() < self.cfg.fg_oversample_prob and mk[0].sum() > 0:
            fg = np.argwhere(mk[0] > 0)
            ix, iy, iz = fg[self.rng.randrange(len(fg))]
        else:
            ix = self.rng.randrange(X)
            iy = self.rng.randrange(Y)
            iz = self.rng.randrange(Z)
        return self._crop(ct, mk,
                          cl(ix-cx//2, 0, max(0,X-cx)),
                          cl(iy-cy//2, 0, max(0,Y-cy)),
                          cl(iz-cz//2, 0, max(0,Z-cz)))

    def __getitem__(self, idx):
        cid      = self.ids[idx]
        ct, mk   = self._load(cid)
        ct_c, mk_c = self._random_crop(ct, mk)
        return (torch.from_numpy(ct_c[None]).float(),   # (1, X, Y, Z)
                torch.from_numpy(mk_c).float())          # (2, X, Y, Z)


# ─────────────────────────────────────────────
# VISTA3D-style model (auto-branch only)
# ─────────────────────────────────────────────
class VISTA3DAutoModel(nn.Module):
    """
    VISTA3D auto-branch for KiTS19 fine-tuning.

    Uses the pretrained SegResNet encoder+decoder from VISTA3D, with a
    lightweight KiTS19-specific segmentation head replacing the original
    127-class head.

    Architecture:
      - Shared encoder: SegResNet encoder (pretrained, optionally frozen)
      - Auto-decoder:   SegResNet decoder with skip connections (pretrained)
      - KiTS19 head:    Conv3d(init_filters, 2, 1) — kidney + tumour

    Fine-tuning strategy (cfg.finetune_mode):
      'decoder' — freeze encoder, train decoder + head  [default, ~10M params]
      'full'    — train everything  [~30M params]
      'head'    — freeze encoder+decoder, train head only  [~1K params]

    The pretrained weights give the model a strong prior on CT anatomy
    (trained on 11,454 scans covering 127 structures including kidney).
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg

        # SegResNet: same config as VISTA3D paper
        # blocks_down/up define the depth — must match checkpoint
        self.segresnet = SegResNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=cfg.out_ch,
            init_filters=cfg.init_filters,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            upsample_mode="deconv",
        )

        # Output bias init: sigmoid(-2.944) ~ 0.05
        # Walk the final conv layer regardless of MONAI version's naming
        for m in self.segresnet.modules():
            if isinstance(m, nn.Conv3d) and m.out_channels == cfg.out_ch:
                if m.bias is not None:
                    nn.init.constant_(m.bias, -2.944)
                break

    def load_pretrained(self, ckpt_path: str):
        """
        Load VISTA3D pretrained weights with partial matching.
        Skips layers that don't match (e.g. final head if out_channels differs).
        """
        print(f"  Loading pretrained weights from {ckpt_path}...", flush=True)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle various checkpoint formats
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

        # Strip common prefixes
        for prefix in ("module.", "model.", "net.", "segresnet."):
            if all(k.startswith(prefix) for k in list(sd.keys())[:5]):
                sd = {k[len(prefix):]: v for k, v in sd.items()}
                break

        own_sd   = self.segresnet.state_dict()
        matched  = {k: v for k, v in sd.items()
                    if k in own_sd and v.shape == own_sd[k].shape}
        skipped  = [k for k in sd if k not in matched]
        missing  = [k for k in own_sd if k not in matched]

        own_sd.update(matched)
        self.segresnet.load_state_dict(own_sd, strict=False)
        print(f"  Loaded {len(matched)} layers, "
              f"skipped {len(skipped)} (shape mismatch/missing), "
              f"randomly init {len(missing)} new layers", flush=True)
        return len(matched)

    def set_finetune_mode(self, mode: str):
        """Freeze/unfreeze layers based on fine-tuning strategy."""
        # First unfreeze everything
        for p in self.parameters():
            p.requires_grad_(True)

        if mode == "head":
            # Freeze all except final conv
            for name, p in self.named_parameters():
                if "conv_final" not in name:
                    p.requires_grad_(False)
        elif mode == "decoder":
            # Freeze encoder blocks only
            for name, p in self.named_parameters():
                if "down_layers" in name or "input_block" in name:
                    p.requires_grad_(False)
        # mode == "full": everything trainable (already done above)

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"  Finetune mode='{mode}': "
              f"trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M", flush=True)

    def forward(self, x):
        return self.segresnet(x)   # (B, 2, X, Y, Z)


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
    print("Building VISTA3D model...", flush=True)
    model = VISTA3DAutoModel(cfg).to(device)

    if os.path.exists(cfg.pretrained_ckpt):
        n_loaded = model.load_pretrained(cfg.pretrained_ckpt)
        if n_loaded == 0:
            print("  WARNING: no layers matched — training from scratch", flush=True)
    else:
        print(f"  WARNING: checkpoint not found at {cfg.pretrained_ckpt}",  flush=True)
        print(f"  Training from scratch. To use pretrained weights run:", flush=True)
        print(f"  wget -O models/model.pt "
              f"https://developer.download.nvidia.com/assets/Clara/monai/tutorials/"
              f"model_zoo/model_vista3d_v1.1.pt", flush=True)

    model.set_finetune_mode(cfg.finetune_mode)

    # ── Data ─────────────────────────────────────────────────────────────────
    labeled_ids = sorted(list_cases(cfg.data_dir))
    print(f"labeled_cases={len(labeled_ids)}", flush=True)

    train_ids = labeled_ids[:min(180, len(labeled_ids))]
    val_ids   = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]

    train_ds = KiTS19Dataset(cfg.data_dir, train_ids, cfg, "train")
    val_ds   = KiTS19Dataset(cfg.data_dir, val_ids,   cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=max(0, cfg.num_workers//2),
                              pin_memory=True, drop_last=False,
                              persistent_workers=(max(0, cfg.num_workers//2) > 0))

    # ── Optimiser + schedule ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt       = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(ep):
        warmup = 3
        if ep < warmup: return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, cfg.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    # Positive class weights — kidney ~5% of crop, tumour ~2%
    pw = torch.tensor([5.0, 10.0], device=device)

    best_mean_dice = -1.0
    history: List[Dict] = []
    run_start = time.time()

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
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

    # ── Unfreeze schedule: after epoch 10 unfreeze full model ────────────────
    UNFREEZE_EPOCH = 10

    try:
        for ep in range(1, cfg.epochs + 1):
            ep_start = time.time()

            # Progressive unfreezing
            if ep == UNFREEZE_EPOCH and cfg.finetune_mode != "full":
                print(f"\n  [Epoch {ep}] Unfreezing full model for end-to-end training",
                      flush=True)
                model.set_finetune_mode("full")
                trainable = [p for p in model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(trainable,
                                        lr=cfg.lr * 0.1,       # lower LR for encoder
                                        weight_decay=cfg.weight_decay)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cfg.epochs - UNFREEZE_EPOCH, eta_min=1e-6)

            cur_lr = opt.param_groups[0]["lr"]
            print(f"\n=== Epoch {ep}/{cfg.epochs}  lr={cur_lr:.2e}  "
                  f"mode={'full' if ep >= UNFREEZE_EPOCH else cfg.finetune_mode} ===",
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
                    logits = model(ct)                    # (B, 2, X, Y, Z)
                    loss   = seg_loss(logits, mk, pw)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                scaler.step(opt); scaler.update()

                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
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
                        logits = model(ct)
                        loss   = seg_loss(logits, mk, pw)
                    v_loss += float(loss.item())

                    pred     = (torch.sigmoid(logits) > 0.5).float()
                    tgt      = mk.float()
                    kd       = dice_score(pred[:,0:1], tgt[:,0:1]).item()
                    td       = dice_score(pred[:,1:2], tgt[:,1:2]).item()
                    kd_pos   = dice_posonly(pred[:,0:1], tgt[:,0:1])
                    td_pos   = dice_posonly(pred[:,1:2], tgt[:,1:2])
                    kd_pos_v = float(kd_pos.item()) if kd_pos is not None else float("nan")
                    td_pos_v = float(td_pos.item()) if td_pos is not None else float("nan")
                    k_sum += kd; t_sum += td; count += 1

                    # Debug first 3 steps
                    if step < 3:
                        pk = torch.sigmoid(logits[:,0:1])
                        pt = torch.sigmoid(logits[:,1:2])
                        print(
                            f"[val debug] step={step} "
                            f"tgt_k={int(tgt[:,0:1].sum())} tgt_t={int(tgt[:,1:2].sum())} "
                            f"pred_k={int(pred[:,0:1].sum())} pred_t={int(pred[:,1:2].sum())} "
                            f"p_k(mean/max)={pk.mean():.3f}/{pk.max():.3f} "
                            f"p_t(mean/max)={pt.mean():.3f}/{pt.max():.3f} "
                            f"kidney_posonly={kd_pos_v:.4f}  tumor_posonly={td_pos_v:.4f}",
                            flush=True)

                    if (step+1) % vpe == 0 or (step+1) == cfg.val_steps:
                        pct = int(round(100*(step+1)/cfg.val_steps))
                        print(f"Val {pct:3d}%  loss={v_loss/(step+1):.4f}  "
                              f"kidney_dice={k_sum/count:.4f}  "
                              f"tumor_dice={t_sum/count:.4f}  "
                              f"mean_dice={(k_sum+t_sum)/(2*count):.4f}  "
                              f"kidney_posonly~{kd_pos_v:.4f}  tumor_posonly~{td_pos_v:.4f}",
                              flush=True)

            val_loss    = v_loss / max(1, cfg.val_steps)
            kidney_dice = k_sum  / max(1, count)
            tumor_dice  = t_sum  / max(1, count)
            mean_dice   = (kidney_dice + tumor_dice) / 2

            last_path = os.path.join(ckpt_dir, "vista3d_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "vista3d_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "cfg": cfg.__dict__}, best_path)

            ep_time = time.time() - ep_start
            print(f"epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"kidney_dice={kidney_dice:.4f} tumor_dice={tumor_dice:.4f} "
                  f"mean_dice={mean_dice:.4f} best={best_mean_dice:.4f} "
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