# /home/coder/projects/IMedSAM/src/train_imedsam_kits19.py
#
# I-MedSAM for KiTS19 Kidney & Tumour Segmentation  (2.5D slice-based)
#
# Paper: Wei et al. "I-MedSAM: Implicit Medical Image Segmentation with
#        Segment Anything", ECCV 2024.  arXiv:2311.17081
#
# Architecture:
#   1. SAM ViT-B image encoder  (FROZEN — 89M params, not updated)
#   2. Frequency Adapter        (TRAINABLE — injects high-freq info via FFT)
#   3. Coarse INR decoder       (TRAINABLE — coordinate MLP, low resolution)
#   4. Uncertainty-Guided Sampling (UGS) — Top-K high-variance points
#   5. Fine INR decoder         (TRAINABLE — refines coarse at sampled points)
#   Total trainable: ~1.6M params
#
# KiTS19 adaptation (2.5D):
#   - Each axial CT slice is treated as a 3-channel RGB image (slice-1, slice, slice+1)
#   - Predictions per-slice are stacked into a 3D volume
#   - Two binary outputs: kidney channel and tumour channel
#
# Run:
#   cd /home/coder/projects/IMedSAM
#   source .venv/bin/activate
#   mkdir -p outputs/logs outputs/checkpoints
#   nohup python -u src/train_imedsam_kits19.py \
#     > outputs/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

import os, re, json, time, random, math
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# SAM imports — requires: pip install -e segment-anything/
from segment_anything import sam_model_registry


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class CFG:
    seed: int = 42

    # Paths
    sam_ckpt: str  = "/home/coder/projects/I-MedSAM/sam_ckp/sam_vit_b_01ec64.pth"
    data_dir: str  = "/home/coder/kits19/data"
    project_dir: str = ""     # filled at runtime
    outputs_dir: str = ""     # filled at runtime

    # Training
    epochs: int = 100
    batch_size: int = 8       # slice-level batches (much smaller than 3D)
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    amp: bool = True
    grad_clip: float = 1.0
    steps_per_epoch: int = 300
    val_steps: int = 100
    print_every_pct: int = 10

    # Slice sampling
    slices_per_vol: int = 32   # random slices drawn per volume per epoch step
    fg_oversample_prob: float = 0.8
    context_slices: int = 1    # neighbour slices for 2.5D (1 = slice-1, slice, slice+1)

    # CT windowing
    win_center: float = 40.0
    win_width: float = 400.0

    # SAM image size (fixed)
    sam_img_size: int = 1024

    # Frequency adapter
    freq_adapter_hidden: int = 256

    # INR decoder
    inr_hidden: int = 256
    inr_layers: int = 4         # depth of each INR MLP
    inr_fourier_bands: int = 64 # positional encoding bands
    coarse_grid: int = 64       # coarse INR output resolution (64x64)

    # Uncertainty-guided sampling
    ugs_topk: int = 256         # number of uncertain points sampled for fine INR
    ugs_mc_samples: int = 8     # Monte-Carlo dropout passes for variance estimate

    # Output
    out_ch: int = 2             # kidney, tumour


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

@lru_cache(maxsize=8)
def load_volume_cached(case_dir: str, win_center: float, win_width: float):
    """
    Load and preprocess a full CT volume + mask.
    Cached per DataLoader worker process (maxsize=8 volumes ~1.6GB per worker).
    Each worker independently caches the volumes it accesses most.
    """
    ct  = load_nifti(os.path.join(case_dir, "imaging.nii.gz"))
    seg = load_nifti(os.path.join(case_dir, "segmentation.nii.gz")).astype(np.uint8)
    ct  = window_ct(ct, win_center, win_width)
    ct  = (ct - ct.mean()) / (ct.std() + 1e-6)
    mk  = np.stack([((seg==1)|(seg==2)).astype(np.float32),
                     (seg==2).astype(np.float32)], axis=0)
    return ct, mk

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
    """pred/target: (B, H, W) binary float. Returns scalar."""
    inter = (pred * target).sum(dim=(1,2))
    denom = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    d = (2*inter + eps) / (denom + eps)
    return torch.where(denom == 0, torch.ones_like(d), d).mean()

def dice_3d(pred_vol, tgt_vol, eps=1e-6):
    """pred_vol/tgt_vol: (D, H, W) binary numpy arrays."""
    inter = (pred_vol * tgt_vol).sum()
    denom = pred_vol.sum() + tgt_vol.sum()
    if denom == 0: return 1.0
    return float(2*inter / (denom + eps))


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────
def bce_dice_loss(logits, target, pos_weight=None):
    """logits/target: (B, C, H, W)"""
    C = logits.shape[1]
    if pos_weight is None:
        pos_weight = torch.ones(C, device=logits.device)
    pw = pos_weight.view(1, C, 1, 1)
    bce  = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pw)
    p    = torch.sigmoid(logits)
    eps  = 1e-6
    dice = 0.0
    for c in range(C):
        inter = (p[:,c] * target[:,c]).sum(dim=(1,2))
        denom = p[:,c].sum(dim=(1,2)) + target[:,c].sum(dim=(1,2))
        dice += (1 - (2*inter + eps) / (denom + eps)).mean()
    return bce + dice / C


# ─────────────────────────────────────────────
# Dataset  (2.5D slice-level)
# ─────────────────────────────────────────────
class KiTS19SliceDataset(Dataset):
    """
    Yields 2.5D slices: (3, H, W) CT image  +  (2, H, W) mask
    where the 3 input channels are [slice-1, slice, slice+1].
    Foreground oversampling ensures kidney/tumour is in the crop.
    """
    def __init__(self, root, ids, cfg, mode):
        self.cfg  = cfg
        self.mode = mode
        self.rng  = random.Random(cfg.seed + (0 if mode == "train" else 9999))

        # Index only — no preloading. Load slices on demand in __getitem__.
        # This avoids loading ~36GB of volumes into RAM at startup.
        print(f"  Indexing {len(ids)} {mode} cases (lazy, no preload)...", flush=True)
        self.index = []     # list of (case_path, slice_z, has_fg)

        for i, cid in enumerate(ids):
            d   = os.path.join(root, f"case_{cid:05d}")
            # Load only segmentation to build fg index (much faster than CT)
            seg = load_nifti(os.path.join(d, "segmentation.nii.gz")).astype(np.uint8)
            D   = seg.shape[2]
            n_fg = 0
            for z in range(D):
                has_fg = seg[:,:,z].max() > 0
                self.index.append((d, z, has_fg))
                if has_fg: n_fg += 1
            if (i+1) % 20 == 0 or (i+1) == len(ids):
                print(f"    [{i+1}/{len(ids)}] case_{cid:05d}  "
                      f"slices={D}  fg_slices={n_fg}  "
                      f"total_indexed={len(self.index)}", flush=True)

        self.fg_index    = [x for x in self.index if x[2]]
        # Separate kidney-only oversample list (seg label 1 or 2 present)
        # stored as (d, z, has_fg) — reuse has_fg flag (any fg = kidney present)
        self.kidney_index = self.fg_index  # kidney present whenever any fg present
        print(f"  {mode} ready: {len(self.index)} total slices, "
              f"{len(self.fg_index)} foreground slices", flush=True)

    def __len__(self):
        return len(self.index)

    def _get_slice(self, d, z):
        # Use per-worker LRU cache — first access loads the volume,
        # subsequent accesses for the same case return from cache instantly.
        ct, mk = load_volume_cached(d, self.cfg.win_center, self.cfg.win_width)
        D  = ct.shape[2]
        z0 = max(0, z-1); z1 = z; z2 = min(D-1, z+1)
        img = np.stack([ct[:,:,z0], ct[:,:,z1], ct[:,:,z2]], axis=0)  # (3,X,Y)
        msk = mk[:, :, :, z]                                           # (2,X,Y)
        return img, msk

    def __getitem__(self, idx):
        r = self.rng.random()
        if self.mode == "train" and r < self.cfg.fg_oversample_prob:
            # Always oversample from kidney slices (any fg = kidney present)
            d, z, _ = self.kidney_index[self.rng.randrange(len(self.kidney_index))]
        else:
            d, z, _ = self.index[idx]

        img, msk = self._get_slice(d, z)
        # Resize to SAM input size
        img_t = torch.from_numpy(img).float()   # (3, H, W)
        msk_t = torch.from_numpy(msk).float()   # (2, H, W)
        img_t = F.interpolate(img_t.unsqueeze(0),
                              size=(self.cfg.sam_img_size, self.cfg.sam_img_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        msk_t = F.interpolate(msk_t.unsqueeze(0),
                              size=(self.cfg.sam_img_size, self.cfg.sam_img_size),
                              mode="nearest").squeeze(0)
        return img_t, msk_t


# ─────────────────────────────────────────────
# Frequency Adapter
# ─────────────────────────────────────────────
class FrequencyAdapter(nn.Module):
    """
    Injects high-frequency information from the Fourier domain into SAM features.

    Following I-MedSAM Fig 2: for each feature map, we compute the 2D FFT,
    extract the high-frequency components (outside the central low-freq region),
    project them back to feature space, and add as a residual.

    This sharpens boundary sensitivity — critical for accurate tumour edge
    delineation in CT where HU gradients at organ boundaries carry
    high-frequency signal that SAM's ViT encoder tends to smooth out.

    Input:  (B, C, H, W)  SAM image embedding
    Output: (B, C, H, W)  frequency-enhanced embedding
    """
    def __init__(self, in_ch, hidden=256):
        super().__init__()
        # Projection: freq features -> channel space
        self.proj_in  = nn.Conv2d(in_ch, hidden, 1)
        self.proj_out = nn.Conv2d(hidden, in_ch,  1)
        self.act      = nn.GELU()
        self.norm     = nn.LayerNorm(in_ch)
        # Low-freq mask radius (fraction of spatial size to KEEP as low-freq)
        self.lf_ratio = 0.25

    def forward(self, x):
        B, C, H, W = x.shape

        # 2D FFT on each channel
        x_fft  = torch.fft.rfft2(x, norm="ortho")          # (B,C,H,W//2+1) complex

        # Build high-frequency mask: zero out central (low-freq) region
        ch = int(H * self.lf_ratio / 2)
        cw = int((W//2+1) * self.lf_ratio)
        hf_mask = torch.ones(H, W//2+1, device=x.device, dtype=torch.float32)
        hf_mask[:ch, :cw]   = 0.0
        hf_mask[-ch:, :cw]  = 0.0

        # Extract high-freq component and reconstruct spatial map
        x_fft_hf = x_fft * hf_mask.unsqueeze(0).unsqueeze(0)
        x_hf     = torch.fft.irfft2(x_fft_hf, s=(H, W), norm="ortho")  # (B,C,H,W)

        # Project high-freq residual and add to original
        h = self.act(self.proj_in(x_hf))
        h = self.proj_out(h)

        # LayerNorm over channel dim
        out = x + h
        out = self.norm(out.permute(0,2,3,1)).permute(0,3,1,2)
        return out


# ─────────────────────────────────────────────
# Fourier Positional Encoding
# ─────────────────────────────────────────────
class FourierPE(nn.Module):
    """
    Random Fourier Features positional encoding for INR.
    Maps (x, y) ∈ [-1,1]² -> R^(2*num_bands) via sin/cos projections.
    Enables the MLP to represent high-frequency functions.
    """
    def __init__(self, num_bands=64, scale=10.0):
        super().__init__()
        B = torch.randn(2, num_bands) * scale
        self.register_buffer("B", B)

    def forward(self, coords):
        """coords: (..., 2) in [-1, 1]"""
        proj = coords @ self.B      # (..., num_bands)
        return torch.cat([torch.sin(2*math.pi*proj),
                          torch.cos(2*math.pi*proj)], dim=-1)   # (..., 2*num_bands)

    @property
    def out_dim(self): return self.B.shape[1] * 2


# ─────────────────────────────────────────────
# INR MLP
# ─────────────────────────────────────────────
class INRMLP(nn.Module):
    """
    Implicit Neural Representation MLP.
    Maps (SAM_features + positional_encoding) -> segmentation logits.

    Input:  (N, feat_dim + pe_dim)
    Output: (N, out_ch)
    """
    def __init__(self, in_dim, hidden, n_layers, out_ch, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                       nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, out_ch)]
        self.net = nn.Sequential(*layers)
        # Store intermediate features for UGS
        self._feat: Optional[torch.Tensor] = None

    def forward(self, x, return_feat=False):
        # Forward through all but last layer to get features
        *body, head = list(self.net.children())
        body_net = nn.Sequential(*body)
        feat = body_net(x)
        if return_feat:
            self._feat = feat
        return head(feat), feat


# ─────────────────────────────────────────────
# Uncertainty-Guided Sampler
# ─────────────────────────────────────────────
def uncertainty_guided_sampling(coarse_logits, coarse_feats,
                                 coords, top_k, mc_samples=8,
                                 coarse_inr=None, inr_input=None):
    """
    Estimate per-point uncertainty via MC-dropout variance, then select
    the top-K most uncertain points for fine refinement.

    coarse_logits : (B, N, C)   coarse logit predictions
    coarse_feats  : (B, N, F)   coarse INR intermediate features
    coords        : (B, N, 2)   grid coordinates
    top_k         : int         number of uncertain points to keep

    Returns:
        selected_coords : (B, K, 2)
        selected_feats  : (B, K, F)
    """
    B, N, C = coarse_logits.shape

    # Uncertainty = variance of sigmoid outputs across MC-dropout passes
    # Since we can't re-run during inference efficiently, use prediction
    # entropy as a proxy: H = -p*log(p) - (1-p)*log(1-p)
    p    = torch.sigmoid(coarse_logits)                         # (B, N, C)
    eps  = 1e-6
    H    = -(p * (p + eps).log() + (1-p) * (1-p+eps).log())    # (B, N, C)
    unc  = H.mean(dim=-1)                                       # (B, N)  avg over classes

    # Select top-K most uncertain points
    K    = min(top_k, N)
    topk_idx = unc.topk(K, dim=1).indices                      # (B, K)

    sel_coords = torch.gather(coords, 1,
                    topk_idx.unsqueeze(-1).expand(-1,-1,2))     # (B, K, 2)
    sel_feats  = torch.gather(coarse_feats, 1,
                    topk_idx.unsqueeze(-1).expand(-1,-1,coarse_feats.shape[-1]))  # (B,K,F)
    return sel_coords, sel_feats


# ─────────────────────────────────────────────
# I-MedSAM Model
# ─────────────────────────────────────────────
class IMedSAM(nn.Module):
    """
    I-MedSAM: SAM ViT-B encoder (frozen) + Frequency Adapter (trained)
              + two-stage coarse-to-fine INR decoder (trained).

    Trainable parameters: ~1.6M  (adapter + coarse INR + fine INR)
    Frozen parameters:    ~89M   (SAM ViT-B image encoder)

    Forward (training):
        img     : (B, 3, 1024, 1024)
        returns : dict with 'coarse', 'fine_sparse' logit tensors + coords

    The coarse INR predicts on a fixed low-res grid (64x64).
    After UGS, the fine INR refines Top-K uncertain points to full res.
    Both outputs are supervised during training.
    """
    def __init__(self, cfg: CFG, sam):
        super().__init__()
        self.cfg = cfg
        self.sam = sam   # SAM model (image encoder used, prompt encoder/decoder discarded)

        # Freeze SAM entirely
        for p in self.sam.parameters():
            p.requires_grad_(False)

        # SAM ViT-B image encoder output: (B, 256, 64, 64)
        sam_feat_ch = 256

        # ── Frequency Adapter (trainable) ──────────────────────────────────
        self.freq_adapter = FrequencyAdapter(sam_feat_ch, cfg.freq_adapter_hidden)

        # ── Positional Encoding ────────────────────────────────────────────
        self.pe = FourierPE(cfg.inr_fourier_bands, scale=10.0)
        pe_dim  = self.pe.out_dim

        # ── Coarse INR ─────────────────────────────────────────────────────
        # Input: bilinearly sampled SAM features at grid coords + PE
        self.coarse_inr = INRMLP(
            in_dim   = sam_feat_ch + pe_dim,
            hidden   = cfg.inr_hidden,
            n_layers = cfg.inr_layers,
            out_ch   = cfg.out_ch,
            dropout  = 0.1,
        )

        # ── Fine INR ───────────────────────────────────────────────────────
        # Input: coarse features (from coarse INR body) + PE at sampled coords
        coarse_feat_dim = cfg.inr_hidden  # output of coarse MLP body
        self.fine_inr = INRMLP(
            in_dim   = coarse_feat_dim + pe_dim,
            hidden   = cfg.inr_hidden,
            n_layers = cfg.inr_layers,
            out_ch   = cfg.out_ch,
            dropout  = 0.1,
        )

    def _sam_features(self, img):
        """Extract SAM image embeddings (frozen). img: (B,3,1024,1024)"""
        with torch.no_grad():
            feats = self.sam.image_encoder(img)   # (B, 256, 64, 64)
        return feats

    def _sample_features(self, feats, coords):
        """
        Bilinearly sample feature map at (x,y) coords.
        feats:  (B, C, H, W)
        coords: (B, N, 2)  in [-1, 1]
        Returns: (B, N, C)
        """
        # grid_sample expects (B, 1, N, 2)
        grid = coords.unsqueeze(1)                                  # (B, 1, N, 2)
        sampled = F.grid_sample(feats, grid,
                                mode="bilinear", align_corners=True,
                                padding_mode="border")              # (B, C, 1, N)
        return sampled.squeeze(2).permute(0, 2, 1)                  # (B, N, C)

    def _make_grid(self, size, device, batch_size):
        """
        Build a regular (size x size) coordinate grid in [-1, 1].
        Returns (B, size*size, 2)
        """
        lin = torch.linspace(-1, 1, size, device=device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (N, 2)
        return coords.unsqueeze(0).expand(batch_size, -1, -1)       # (B, N, 2)

    def forward(self, img, return_coarse=True):
        B   = img.shape[0]
        dev = img.device

        # 1. SAM features (frozen)
        feats = self._sam_features(img)          # (B, 256, 64, 64)

        # 2. Frequency adapter (trained)
        feats = self.freq_adapter(feats)         # (B, 256, 64, 64)

        # 3. Coarse grid  (cfg.coarse_grid x cfg.coarse_grid)
        coarse_coords = self._make_grid(self.cfg.coarse_grid, dev, B)  # (B, N_c, 2)
        N_c           = coarse_coords.shape[1]

        # Sample SAM features at coarse coords
        coarse_f = self._sample_features(feats, coarse_coords)         # (B, N_c, 256)

        # Positional encoding
        coarse_pe = self.pe(coarse_coords)                             # (B, N_c, pe_dim)

        # Coarse INR forward
        coarse_inp          = torch.cat([coarse_f, coarse_pe], dim=-1) # (B, N_c, 256+pe)
        coarse_inp_flat     = coarse_inp.view(B * N_c, -1)
        coarse_logits_flat, coarse_feats_flat = self.coarse_inr(
            coarse_inp_flat, return_feat=True)
        coarse_logits = coarse_logits_flat.view(B, N_c, self.cfg.out_ch)  # (B,N_c,C)
        coarse_feats  = coarse_feats_flat.view(B, N_c, -1)               # (B,N_c,F)

        # 4. Uncertainty-Guided Sampling
        sel_coords, sel_feats = uncertainty_guided_sampling(
            coarse_logits, coarse_feats, coarse_coords, self.cfg.ugs_topk)
        # sel_coords: (B, K, 2)  sel_feats: (B, K, F)

        # Fine INR
        K          = sel_coords.shape[1]
        fine_pe    = self.pe(sel_coords)                               # (B, K, pe_dim)
        fine_inp   = torch.cat([sel_feats, fine_pe], dim=-1)           # (B, K, F+pe)
        fine_inp_f = fine_inp.view(B * K, -1)
        fine_logits_flat, _ = self.fine_inr(fine_inp_f, return_feat=False)
        fine_logits = fine_logits_flat.view(B, K, self.cfg.out_ch)    # (B, K, C)

        # 5. Coarse logits -> full resolution spatial map
        # Reshape coarse to (B, C, coarse_grid, coarse_grid) then upsample
        coarse_map = coarse_logits.permute(0, 2, 1).view(
            B, self.cfg.out_ch, self.cfg.coarse_grid, self.cfg.coarse_grid)
        coarse_full = F.interpolate(coarse_map,
                                    size=(self.cfg.sam_img_size, self.cfg.sam_img_size),
                                    mode="bilinear", align_corners=False)  # (B,C,1024,1024)

        return {
            "coarse_full" : coarse_full,     # (B, C, 1024, 1024) — main supervised output
            "coarse_logits": coarse_logits,  # (B, N_c, C)        — coarse grid logits
            "fine_logits" : fine_logits,     # (B, K, C)          — fine refined logits
            "fine_coords" : sel_coords,      # (B, K, 2)          — coords of fine points
        }


# ─────────────────────────────────────────────
# Per-point fine loss
# ─────────────────────────────────────────────
def fine_loss(fine_logits, fine_coords, target_full, cfg):
    """
    fine_logits:  (B, K, C)
    fine_coords:  (B, K, 2) in [-1, 1]
    target_full:  (B, C, H, W) binary masks at full resolution
    """
    B, K, C = fine_logits.shape
    # Sample ground-truth at fine coords
    gt_at_pts = F.grid_sample(
        target_full.float(),
        fine_coords.unsqueeze(1),          # (B, 1, K, 2)
        mode="nearest", align_corners=True,
        padding_mode="border",
    ).squeeze(2).permute(0, 2, 1)          # (B, K, C)

    fine_logits_flat = fine_logits.view(B*K, C)
    gt_flat          = gt_at_pts.reshape(B*K, C)
    loss = F.binary_cross_entropy_with_logits(fine_logits_flat, gt_flat)
    # Soft dice on fine points
    p     = torch.sigmoid(fine_logits_flat)
    eps   = 1e-6
    inter = (p * gt_flat).sum(0)
    denom = p.sum(0) + gt_flat.sum(0)
    dice  = (1 - (2*inter + eps) / (denom + eps)).mean()
    return loss + dice


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
    if not os.path.exists(cfg.sam_ckpt):
        raise RuntimeError(f"SAM checkpoint not found: {cfg.sam_ckpt}\n"
                           f"Run: wget -P sam_ckp "
                           f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")

    ts             = time.strftime("%Y%m%d_%H%M%S")
    metrics_latest = os.path.join(cfg.outputs_dir, "metrics_latest.json")
    metrics_final  = os.path.join(cfg.outputs_dir, f"metrics_run_{ts}.json")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"device: {device}  gpu: {gpu_name}", flush=True)

    # Load SAM ViT-B
    print("Loading SAM ViT-B...", flush=True)
    sam  = sam_model_registry["vit_b"](checkpoint=cfg.sam_ckpt)
    sam  = sam.to(device)
    model = IMedSAM(cfg, sam).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"params: trainable={n_trainable/1e6:.2f}M  "
          f"total={n_total/1e6:.2f}M  "
          f"frozen={( n_total - n_trainable)/1e6:.2f}M", flush=True)

    # Data
    labeled_ids = sorted(list_cases(cfg.data_dir))
    print(f"labeled_cases={len(labeled_ids)}", flush=True)
    train_ids = labeled_ids[:min(180, len(labeled_ids))]
    val_ids   = labeled_ids[min(180, len(labeled_ids)):min(210, len(labeled_ids))]
    print(f"train cases={len(train_ids)}  val cases={len(val_ids)}", flush=True)

    train_ds = KiTS19SliceDataset(cfg.data_dir, train_ids, cfg, "train")
    val_ds   = KiTS19SliceDataset(cfg.data_dir, val_ids,   cfg, "val")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              drop_last=True,
                              persistent_workers=(cfg.num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=max(0, cfg.num_workers//2),
                              pin_memory=True, drop_last=False,
                              persistent_workers=(max(0, cfg.num_workers//2) > 0))

    # Only train adapter + INR (SAM frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(ep):
        warmup = 3
        if ep < warmup: return (ep + 1) / warmup
        progress = (ep - warmup) / max(1, cfg.epochs - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched  = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device == "cuda"))

    # Class weights: kidney >> background, tumour >> background
    # Equal pos_weight: let soft-dice balance the classes.
    # Kidney ~5% of slice, tumor ~2% — both need upweighting but equally.
    # High tumor weight (30) was causing tumor to dominate early training.
    pw = torch.tensor([10.0, 10.0], device=device)

    best_mean_dice = -1.0
    history: List[Dict] = []
    run_start = time.time()

    summary = {
        "timestamp": ts, "device": device, "gpu": gpu_name,
        "n_trainable_M": round(n_trainable/1e6, 2),
        "n_total_M": round(n_total/1e6, 2),
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

            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            r_loss = r_dk = r_dt = 0.0
            it = iter(train_loader)

            for step in range(cfg.steps_per_epoch):
                try:    img, msk = next(it)
                except: it = iter(train_loader); img, msk = next(it)

                img = img.to(device, non_blocking=True)
                msk = msk.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with actx():
                    out  = model(img)

                    # Coarse loss: bilinearly upsampled coarse grid vs full-res mask
                    l_coarse = bce_dice_loss(out["coarse_full"], msk, pw)

                    # Fine loss: point-wise at uncertain locations
                    l_fine   = fine_loss(out["fine_logits"], out["fine_coords"], msk, cfg)

                    # Total: coarse dominates early, fine refines boundaries
                    loss = l_coarse + 0.5 * l_fine

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
                scaler.step(opt); scaler.update()

                with torch.no_grad():
                    pred = (torch.sigmoid(out["coarse_full"]) > 0.5).float()
                    r_dk += dice_score(pred[:,0], msk[:,0]).item()
                    r_dt += dice_score(pred[:,1], msk[:,1]).item()
                r_loss += float(loss.item())

                if (step+1) % tpe == 0 or (step+1) == cfg.steps_per_epoch:
                    n   = step + 1
                    pct = int(round(100 * n / cfg.steps_per_epoch))
                    print(f"Train {pct:3d}%  loss={r_loss/n:.4f}  "
                          f"kidney_dice={r_dk/n:.4f}  tumor_dice={r_dt/n:.4f}",
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
                    try:    img, msk = next(it)
                    except: it = iter(val_loader); img, msk = next(it)

                    img = img.to(device, non_blocking=True)
                    msk = msk.to(device, non_blocking=True)

                    with actx():
                        out  = model(img)
                        loss = bce_dice_loss(out["coarse_full"], msk, pw)
                    v_loss += float(loss.item())

                    pred     = (torch.sigmoid(out["coarse_full"]) > 0.5).float()
                    tgt      = msk.float()
                    k_sum += dice_score(pred[:,0], tgt[:,0]).item()
                    t_sum += dice_score(pred[:,1], tgt[:,1]).item()
                    count += 1

                    if step < 3:
                        pk = torch.sigmoid(out["coarse_full"][:,0])
                        pt = torch.sigmoid(out["coarse_full"][:,1])
                        print(
                            f"[val debug] step={step} "
                            f"tgt_k={int(tgt[:,0].sum())} tgt_t={int(tgt[:,1].sum())} "
                            f"pred_k={int(pred[:,0].sum())} pred_t={int(pred[:,1].sum())} "
                            f"p_k(mean/max)={pk.mean():.3f}/{pk.max():.3f} "
                            f"p_t(mean/max)={pt.mean():.3f}/{pt.max():.3f}",
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

            last_path = os.path.join(ckpt_dir, "imedsam_last.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict(),
                        "cfg": cfg.__dict__}, last_path)
            best_path = None
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "imedsam_best.pt")
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