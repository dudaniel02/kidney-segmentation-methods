# /home/coder/projects/UKAN/src/train_ukan_kits19_3d.py
#
# 3D U-KAN for KiTS19 Kidney & Tumour Segmentation
# Novel application: first use of U-KAN on 3D abdominal CT (KiTS19)
# Architecture: 3D UNet encoder-decoder with KAN channel-attention bottleneck
#
# Reference: Li et al. "U-KAN Makes Strong Backbone for Medical Image
#   Segmentation and Generation", AAAI 2025. arXiv:2406.02918
#
# Run:
#   cd /home/coder/projects/UKAN
#   source .venv/bin/activate
#   mkdir -p outputs/logs
#   nohup python -u src/train_ukan_kits19_3d.py \
#     > outputs/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

import os, re, json, time, random, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

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
    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    pin_memory: bool = True
    amp: bool = True
    grad_clip: float = 1.0
    steps_per_epoch: int = 250
    val_steps: int = 50
    print_every_pct: int = 10
    crop_xyz: Tuple[int, int, int] = (128, 128, 96)
    fg_oversample_prob: float = 0.7
    win_center: float = 40.0
    win_width: float = 400.0
    enc_channels: Tuple[int, ...] = (32, 64, 128, 256)
    kan_dims: Tuple[int, ...] = (256, 512, 256)
    grid_size: int = 5
    spline_order: int = 3
    out_ch: int = 2
    project_dir: str = ""
    data_dir: str = ""
    outputs_dir: str = ""


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
    labeled, unlabeled = [], []
    for name in sorted(os.listdir(root)):
        cid = case_id(name)
        if cid is None: continue
        d = os.path.join(root, name)
        if not os.path.isdir(d): continue
        has_seg = os.path.exists(os.path.join(d, "segmentation.nii.gz"))
        (labeled if has_seg else unlabeled).append(cid)
    return labeled, unlabeled


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def dice_score(pred, target, eps=1e-6):
    pred = pred.float(); target = target.float()
    inter = (pred * target).sum(dim=(2,3,4))
    denom = pred.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4))
    d = (2*inter + eps) / (denom + eps)
    return torch.where(denom == 0, torch.ones_like(d), d).squeeze(1)

def dice_posonly(pred, target, eps=1e-6):
    ts = target.float().sum(dim=(2,3,4)).squeeze(1)
    keep = ts > 0
    if keep.sum() == 0: return None
    return dice_score(pred[keep], target[keep], eps).mean()


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
    pw = pos_weight.view(1, C, 1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target.float(),
                                              pos_weight=pw, reduction="none")
    pt = torch.where(target > 0.5, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
    return (((1 - pt)**gamma) * bce).mean()

def seg_loss(logits, target, pos_weight=None):
    return soft_dice_loss(logits, target) + focal_loss(logits, target, pos_weight=pos_weight)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class KiTS19Dataset(Dataset):
    def __init__(self, root, ids, cfg, mode):
        self.dirs = [os.path.join(root, f"case_{i:05d}") for i in ids]
        self.cfg = cfg; self.mode = mode
        self.rng = random.Random(cfg.seed + (0 if mode == "train" else 9999))

    def __len__(self): return len(self.dirs)

    def _targets(self, seg):
        k = ((seg==1)|(seg==2)).astype(np.float32)
        t = (seg==2).astype(np.float32)
        return np.stack([k, t], 0)

    def _pad(self, ct, mk):
        cx, cy, cz = self.cfg.crop_xyz
        px = max(0, cx-ct.shape[0]); py = max(0, cy-ct.shape[1]); pz = max(0, cz-ct.shape[2])
        if px or py or pz:
            ct = np.pad(ct, ((0,px),(0,py),(0,pz)))
            mk = np.pad(mk, ((0,0),(0,px),(0,py),(0,pz)))
        return ct, mk

    def _crop(self, ct, mk, x0, y0, z0):
        cx, cy, cz = self.cfg.crop_xyz
        return self._pad(ct[x0:x0+cx, y0:y0+cy, z0:z0+cz],
                         mk[:, x0:x0+cx, y0:y0+cy, z0:z0+cz])

    def _random_crop(self, ct, mk):
        cx, cy, cz = self.cfg.crop_xyz
        X, Y, Z = ct.shape
        cl = lambda a, lo, hi: max(lo, min(hi, a))
        if self.rng.random() < self.cfg.fg_oversample_prob and mk[0].sum() > 0:
            fg = np.argwhere(mk[0] > 0)
            ix, iy, iz = fg[self.rng.randrange(len(fg))]
        else:
            ix, iy, iz = self.rng.randrange(X), self.rng.randrange(Y), self.rng.randrange(Z)
        return self._crop(ct, mk,
                          cl(ix-cx//2, 0, max(0,X-cx)),
                          cl(iy-cy//2, 0, max(0,Y-cy)),
                          cl(iz-cz//2, 0, max(0,Z-cz)))

    def __getitem__(self, idx):
        d   = self.dirs[idx]
        ct  = load_nifti(os.path.join(d, "imaging.nii.gz"))
        seg = load_nifti(os.path.join(d, "segmentation.nii.gz")).astype(np.uint8)
        ct  = window_ct(ct, self.cfg.win_center, self.cfg.win_width)
        ct  = (ct - ct.mean()) / (ct.std() + 1e-6)
        mk  = self._targets(seg)
        # Same foreground-oversampled crop for both train and val
        ct_c, mk_c = self._random_crop(ct, mk)
        return torch.from_numpy(ct_c[None]).float(), torch.from_numpy(mk_c).float()


# ─────────────────────────────────────────────
# KAN Layer  (B-Spline activations)
# Liu et al. "KAN: Kolmogorov-Arnold Networks", 2024
# ─────────────────────────────────────────────
class KANLinear(nn.Module):
    """
    KAN layer: phi(x) = w_base*SiLU(x) + w_spline*B_spline(x)
    B-spline activations are learned per edge (not per node).
    x: (N, in_features)  ->  (N, out_features)
    """
    def __init__(self, in_features, out_features,
                 grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_range=(-1, 1)):
        super().__init__()
        self.in_f = in_features; self.out_f = out_features
        self.grid_size = grid_size; self.spline_order = spline_order
        h    = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * scale_noise)
        self.base_weight   = nn.Parameter(torch.randn(out_features, in_features) * scale_base)
        self.scale_base    = scale_base
        self.scale_spline  = scale_spline
        self.norm          = nn.LayerNorm(out_features)

    def b_splines(self, x):
        x     = x.unsqueeze(-1)
        grid  = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left  = (x - grid[:-(k+1)]) / (grid[k:-1]   - grid[:-(k+1)] + 1e-8)
            right = (grid[(k+1):] - x)  / (grid[(k+1):] - grid[1:-k]    + 1e-8)
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases.contiguous()

    def forward(self, x):
        base_out   = F.linear(F.silu(x), self.base_weight)
        splines    = self.b_splines(x)
        N          = x.shape[0]
        spline_out = splines.view(N, -1) @ self.spline_weight.view(self.out_f, -1).T
        return self.norm(self.scale_base * base_out + self.scale_spline * spline_out)


# ─────────────────────────────────────────────
# KAN Bottleneck  (channel attention via KAN)
# ─────────────────────────────────────────────
class KANBlock(nn.Module):
    """
    KAN-based squeeze-excitation channel attention.

    Standard SE-Net uses a 2-layer ReLU MLP for channel gating.
    This replaces that MLP with a KAN — learnable B-spline activations
    adapt to per-patient CT intensity distributions.

    1. Global average pool  -> (B, C) channel descriptor
    2. KAN layers           -> learned non-linear channel importance
    3. Sigmoid gate         -> rescale feature map channels (residual)
    """
    def __init__(self, in_ch, kan_dims, grid_size=5, spline_order=3):
        super().__init__()
        mid = kan_dims[0]
        self.kan1 = KANLinear(in_ch, mid,   grid_size=grid_size, spline_order=spline_order)
        self.kan2 = KANLinear(mid,   in_ch, grid_size=grid_size, spline_order=spline_order)
        self.norm = nn.GroupNorm(8, in_ch)

    def forward(self, x):
        gap  = x.mean(dim=(2, 3, 4))                                  # (B, C)
        h    = self.kan1(gap)                                          # (B, mid)
        h    = self.kan2(h)                                            # (B, C)
        attn = torch.sigmoid(h).view(x.shape[0], x.shape[1], 1, 1, 1)
        return self.norm(x * attn + x)                                 # residual gate


# ─────────────────────────────────────────────
# 3D U-KAN
# ─────────────────────────────────────────────
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch), nn.SiLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.GroupNorm(8, out_ch), nn.SiLU(inplace=True),
    )


class UKAN3D(nn.Module):
    """
    3D U-KAN: UNet encoder-decoder with KAN channel-attention bottleneck.

    Encoder   : 4-level 3D conv blocks + MaxPool
    Bottleneck: conv block -> KANBlock (learnable B-spline channel gating)
    Decoder   : 4-level transpose conv + skip connections
    Heads     : main output + 2 deep supervision heads (aux3 at dec3, aux2 at dec2)

    Output bias initialized to -2.944 so sigmoid(bias)~0.05 at epoch 0.
    This prevents the "predict everything" collapse from fg oversampling.
    """
    def __init__(self, cfg: CFG):
        super().__init__()
        chs = cfg.enc_channels  # (32, 64, 128, 256)

        # Encoder
        self.enc1 = conv_block(1,      chs[0])
        self.enc2 = conv_block(chs[0], chs[1])
        self.enc3 = conv_block(chs[1], chs[2])
        self.enc4 = conv_block(chs[2], chs[3])
        self.pool = nn.MaxPool3d(2)

        # KAN Bottleneck
        self.bottleneck_conv = conv_block(chs[3], chs[3])
        self.kan_block = KANBlock(
            in_ch=chs[3], kan_dims=cfg.kan_dims,
            grid_size=cfg.grid_size, spline_order=cfg.spline_order,
        )

        # Decoder
        self.up4  = nn.ConvTranspose3d(chs[3], chs[3], 2, stride=2)
        self.dec4 = conv_block(chs[3]+chs[3], chs[3])
        self.up3  = nn.ConvTranspose3d(chs[3], chs[2], 2, stride=2)
        self.dec3 = conv_block(chs[2]+chs[2], chs[2])
        self.up2  = nn.ConvTranspose3d(chs[2], chs[1], 2, stride=2)
        self.dec2 = conv_block(chs[1]+chs[1], chs[1])
        self.up1  = nn.ConvTranspose3d(chs[1], chs[0], 2, stride=2)
        self.dec1 = conv_block(chs[0]+chs[0], chs[0])

        # Output heads
        self.head = nn.Conv3d(chs[0], cfg.out_ch, 1)
        self.aux3 = nn.Conv3d(chs[2], cfg.out_ch, 1)
        self.aux2 = nn.Conv3d(chs[1], cfg.out_ch, 1)

        # Bias init: start predicting ~5% positive everywhere (not 50%)
        for head in (self.head, self.aux3, self.aux2):
            nn.init.constant_(head.bias, -2.944)

    def forward(self, x, deep_sup=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # KAN Bottleneck
        b = self.bottleneck_conv(self.pool(e4))
        b = self.kan_block(b)

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.head(d1)
        if not deep_sup:
            return logits

        # Auxiliary heads upsampled to full resolution
        a3 = F.interpolate(self.aux3(d3), size=x.shape[2:],
                           mode="trilinear", align_corners=False)
        a2 = F.interpolate(self.aux2(d2), size=x.shape[2:],
                           mode="trilinear", align_corners=False)
        return logits, a3, a2


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    cfg = CFG()
    set_seed(cfg.seed)

    cfg.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg.outputs_dir = os.path.join(cfg.project_dir, "outputs")
    cfg.data_dir    = "/home/coder/kits19/data"

    if not os.path.isdir(cfg.data_dir):
        raise RuntimeError(f"Data not found: {cfg.data_dir}")

    ckpt_dir = os.path.join(cfg.outputs_dir, "checkpoints")
    for d in [ckpt_dir, os.path.join(cfg.outputs_dir, "logs")]:
        os.makedirs(d, exist_ok=True)

    ts             = time.strftime("%Y%m%d_%H%M%S")
    metrics_latest = os.path.join(cfg.outputs_dir, "metrics_latest.json")
    metrics_final  = os.path.join(cfg.outputs_dir, f"metrics_run_{ts}.json")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"device: {device}  gpu: {gpu_name}", flush=True)

    labeled_ids, _ = list_cases(cfg.data_dir)
    labeled_ids    = sorted(labeled_ids)
    print(f"labeled_cases={len(labeled_ids)}", flush=True)
    if not labeled_ids:
        raise RuntimeError("No labeled cases found.")

    train_ids = labeled_ids[:min(180, len(labeled_ids))]
    val_ids   = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]
    print(f"train={len(train_ids)}  val={len(val_ids)}", flush=True)

    train_ds = KiTS19Dataset(cfg.data_dir, train_ids, cfg, "train")
    val_ds   = KiTS19Dataset(cfg.data_dir, val_ids,   cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
                              drop_last=True, persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=max(0, cfg.num_workers//2),
                              pin_memory=cfg.pin_memory, drop_last=False,
                              persistent_workers=(max(0, cfg.num_workers//2) > 0))

    model  = UKAN3D(cfg).to(device)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model params: {n_par/1e6:.2f}M  "
          f"enc_ch={cfg.enc_channels}  kan_dims={cfg.kan_dims}", flush=True)

    # AdamW + 5-epoch linear warmup then cosine decay
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(ep):
        warmup = 5
        if ep < warmup: return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, cfg.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    # Positive class weights for focal loss
    # kidney ~5% of voxels in a crop -> weight 20
    # tumor  ~2% of voxels in a crop -> weight 50
    pw = torch.tensor([5.0, 10.0], device=device)  # mild boost, not overwhelming

    best_mean_dice = -1.0
    history: List[Dict] = []
    run_start = time.time()

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
        "n_params_M": round(n_par/1e6, 2),
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
        print(f"\n=== RUN END ===\nstatus: {status}\n"
              f"best_mean_dice: {best_mean_dice:.4f}\n"
              f"total_time_s: {summary['total_time_s']:.1f}\n"
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

            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            r_loss = r_dice_k = r_dice_t = 0.0
            it = iter(train_loader)

            for step in range(cfg.steps_per_epoch):
                try:    ct, mk = next(it)
                except: it = iter(train_loader); ct, mk = next(it)

                ct = ct.to(device, non_blocking=True)
                mk = mk.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with actx():
                    logits, a3, a2 = model(ct, deep_sup=True)
                    # Deep supervision: main (weight 1.0) + aux3 (0.4) + aux2 (0.2)
                    loss = (    seg_loss(logits, mk, pw)
                            + 0.4 * seg_loss(a3,    mk, pw)
                            + 0.2 * seg_loss(a2,    mk, pw))

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()

                with torch.no_grad():
                    pred = (torch.sigmoid(logits) > 0.5).float()
                    r_dice_k += dice_score(pred[:,0:1], mk[:,0:1]).mean().item()
                    r_dice_t += dice_score(pred[:,1:2], mk[:,1:2]).mean().item()
                r_loss += float(loss.item())

                if (step+1) % tpe == 0 or (step+1) == cfg.steps_per_epoch:
                    n   = step + 1
                    pct = int(round(100 * n / cfg.steps_per_epoch))
                    print(f"Train {pct:3d}%  loss={r_loss/n:.4f}  "
                          f"kidney_dice={r_dice_k/n:.4f}  tumor_dice={r_dice_t/n:.4f}",
                          flush=True)

            sched.step()
            train_loss = r_loss / cfg.steps_per_epoch

            # ── Validation ─────────────────────────────────────────────────────
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
                        logits = model(ct, deep_sup=False)
                        loss   = seg_loss(logits, mk, pw)
                    v_loss += float(loss.item())

                    pred     = (torch.sigmoid(logits) > 0.5).float()
                    tgt      = mk.float()
                    kd       = dice_score(pred[:,0:1], tgt[:,0:1]).mean().item()
                    td       = dice_score(pred[:,1:2], tgt[:,1:2]).mean().item()
                    kd_pos   = dice_posonly(pred[:,0:1], tgt[:,0:1])
                    td_pos   = dice_posonly(pred[:,1:2], tgt[:,1:2])
                    kd_pos_v = float(kd_pos.item()) if kd_pos is not None else float("nan")
                    td_pos_v = float(td_pos.item()) if td_pos is not None else float("nan")
                    k_sum += kd; t_sum += td; count += 1

                    if step < 3:
                        pk = torch.sigmoid(logits[:,0:1])
                        pt = torch.sigmoid(logits[:,1:2])
                        print(
                            f"[val debug] step={step} "
                            f"tgt_k={int(tgt[:,0:1].sum())} tgt_t={int(tgt[:,1:2].sum())} "
                            f"pred_k={int(pred[:,0:1].sum())} pred_t={int(pred[:,1:2].sum())} "
                            f"p_k(min/mean/max)={pk.min():.3f}/{pk.mean():.3f}/{pk.max():.3f} "
                            f"p_k_p90={torch.quantile(pk.float().flatten(),0.9):.3f} "
                            f"p_t(min/mean/max)={pt.min():.3f}/{pt.mean():.3f}/{pt.max():.3f} "
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

            last_path = os.path.join(ckpt_dir, "ukan_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "ukan_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict(),
                            "cfg": cfg.__dict__}, best_path)

            ep_time = time.time() - ep_start
            print(f"epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"kidney_dice={kidney_dice:.4f} tumor_dice={tumor_dice:.4f} "
                  f"mean_dice={mean_dice:.4f} best_mean_dice={best_mean_dice:.4f} "
                  f"epoch_time_s={ep_time:.1f}", flush=True)

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