# /home/coder/projects/dinov2/src/train_dinov2_kits19_2d.py
# Run with logging that survives disconnects:
#   cd /home/coder/projects/dinov2
#   source .venv/bin/activate
#   mkdir -p outputs/logs
#   nohup python -u src/train_dinov2_kits19_2d.py > outputs/logs/train_$(date +%Y%m%d_%H%M%S).log 2>&1 &

import os
import json
import random
import time
import signal
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, Dinov2Model


# ----------------------------
# CT utils
# ----------------------------

def window_ct(x: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)

def ct_to_3ch(slice_hu: np.ndarray) -> np.ndarray:
    c1 = window_ct(slice_hu, center=40, width=400)
    c2 = window_ct(slice_hu, center=80, width=200)
    c3 = window_ct(slice_hu, center=0, width=1000)
    img = np.stack([c1, c2, c3], axis=-1)
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img

def pad_to_at_least(img: np.ndarray, min_h: int, min_w: int, pad_value: int = 0) -> np.ndarray:
    if img.ndim == 3:
        h, w, _ = img.shape
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=pad_value)
    else:
        h, w = img.shape
        pad_h = max(0, min_h - h)
        pad_w = max(0, min_w - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_value)

def random_crop(img: np.ndarray, mask: np.ndarray, crop_h: int, crop_w: int, rng: random.Random):
    h, w = img.shape[:2]
    y0 = 0 if h == crop_h else rng.randint(0, h - crop_h)
    x0 = 0 if w == crop_w else rng.randint(0, w - crop_w)
    return img[y0:y0 + crop_h, x0:x0 + crop_w], mask[y0:y0 + crop_h, x0:x0 + crop_w]

def center_crop(img: np.ndarray, mask: np.ndarray, crop_h: int, crop_w: int):
    h, w = img.shape[:2]
    y0 = max(0, (h - crop_h) // 2)
    x0 = max(0, (w - crop_w) // 2)
    return img[y0:y0 + crop_h, x0:x0 + crop_w], mask[y0:y0 + crop_h, x0:x0 + crop_w]

def load_case(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    img_p = os.path.join(case_dir, "imaging.nii.gz")
    seg_p = os.path.join(case_dir, "segmentation.nii.gz")
    vol = nib.load(img_p).get_fdata(dtype=np.float32)
    seg = nib.load(seg_p).get_fdata(dtype=np.float32).astype(np.uint8)
    vol = np.transpose(vol, (2, 0, 1))
    seg = np.transpose(seg, (2, 0, 1))
    return vol, seg


# ----------------------------
# Dataset
# ----------------------------

@dataclass
class SliceRef:
    case_dir: str
    z: int

class KiTS19Slices(Dataset):
    def __init__(
        self,
        data_root: str,
        case_ids: List[int],
        crop_hw: Tuple[int, int] = (448, 448),
        positive_fraction: float = 0.85,
        max_slices_per_case: int = 350,
        seed: int = 0,
        name: str = "train",
        mode: str = "train",
    ):
        super().__init__()
        self.rng = random.Random(seed)
        self.positive_fraction = float(positive_fraction)
        self.crop_h, self.crop_w = int(crop_hw[0]), int(crop_hw[1])
        self.mode = mode

        self.case_dirs = [os.path.join(data_root, f"case_{cid:05d}") for cid in case_ids]
        self.pos_pool: List[SliceRef] = []
        self.any_pool: List[SliceRef] = []

        n_cases = len(self.case_dirs)
        print_every = max(1, n_cases // 10)

        print(f"[{name}] Building slice index from {n_cases} cases (CPU-only)...", flush=True)
        t0 = time.time()
        for idx, cdir in enumerate(self.case_dirs, start=1):
            vol, seg = load_case(cdir)
            zdim = vol.shape[0]

            pos_z = np.where((seg > 0).reshape(zdim, -1).any(axis=1))[0].tolist()
            any_z = list(range(zdim))

            if len(any_z) > max_slices_per_case:
                step = max(1, len(any_z) // max_slices_per_case)
                any_z = any_z[::step]
            if len(pos_z) > max_slices_per_case:
                step = max(1, len(pos_z) // max_slices_per_case)
                pos_z = pos_z[::step]

            self.any_pool += [SliceRef(cdir, z) for z in any_z]
            self.pos_pool += [SliceRef(cdir, z) for z in pos_z]

            if idx % print_every == 0 or idx == n_cases:
                pct = int(round(100 * idx / n_cases))
                print(f"[{name}] Indexing: {pct}% ({idx}/{n_cases})", flush=True)

        if len(self.pos_pool) == 0:
            self.pos_pool = self.any_pool

        self._len = len(self.any_pool)
        print(f"[{name}] Done indexing in {(time.time() - t0):.1f}s. Dataset len={self._len}", flush=True)

    def __len__(self):
        return self._len

    def __getitem__(self, idx: int):
        pick_pos = self.rng.random() < self.positive_fraction
        ref = self.rng.choice(self.pos_pool if pick_pos else self.any_pool)

        vol, seg = load_case(ref.case_dir)
        hu = vol[ref.z]
        m = seg[ref.z]

        img = ct_to_3ch(hu)
        img = pad_to_at_least(img, self.crop_h, self.crop_w, pad_value=0)
        m = pad_to_at_least(m, self.crop_h, self.crop_w, pad_value=0)

        if self.mode == "train":
            img, m = random_crop(img, m, self.crop_h, self.crop_w, self.rng)
        else:
            img, m = center_crop(img, m, self.crop_h, self.crop_w)

        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(m).long()
        return x, y


# ----------------------------
# Model
# ----------------------------

class DinoV2Seg(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-base", num_classes: int = 3, freeze_backbone: bool = True):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = Dinov2Model.from_pretrained(model_name)
        self.patch = self.backbone.config.patch_size
        d = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Conv2d(d, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.processor.image_mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        _, _, H, W = x.shape
        ps = self.patch
        gh, gw = H // ps, W // ps

        out = self.backbone(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1:, :]
        B, N, D = patch_tokens.shape
        expected = gh * gw
        if N != expected:
            raise RuntimeError(f"token mismatch N={N} expected={expected} from H,W={H,W}, ps={ps}")

        feat = patch_tokens.transpose(1, 2).contiguous().view(B, D, gh, gw)
        logits_low = self.head(feat)
        logits = F.interpolate(logits_low, size=(H, W), mode="bilinear", align_corners=False)
        return logits


# ----------------------------
# Loss + Metrics
# ----------------------------

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(probs * onehot, dims)
    denom = torch.sum(probs + onehot, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()

@torch.no_grad()
def dice_per_class_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    pred = torch.argmax(logits, dim=1)

    def d(lbl: int):
        p = (pred == lbl).float()
        t = (target == lbl).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        return ((2.0 * inter + eps) / (denom + eps)).item()

    kidney = d(1)
    tumor = d(2)
    mean = 0.5 * (kidney + tumor)
    return kidney, tumor, mean


# ----------------------------
# Safe metrics writing
# ----------------------------

def write_json(path: str, obj: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)  # atomic replace


# ----------------------------
# Train
# ----------------------------

def main():
    run_start = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")

    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = os.path.join(proj, "data")
    out_dir = os.path.join(proj, "outputs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    metrics_latest = os.path.join(out_dir, "metrics_latest.json")
    metrics_final = os.path.join(out_dir, f"metrics_run_{ts}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print("device:", device, "gpu:", gpu_name, flush=True)

    train_ids = list(range(0, 180))
    val_ids = list(range(180, 210))
    crop_hw = (448, 448)

    train_ds = KiTS19Slices(data_root, train_ids, crop_hw=crop_hw, positive_fraction=0.85, seed=0, name="train", mode="train")
    val_ds = KiTS19Slices(data_root, val_ids, crop_hw=crop_hw, positive_fraction=0.0, seed=1, name="val", mode="val")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = DinoV2Seg("facebook/dinov2-base", num_classes=3, freeze_backbone=True).to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    epochs = 10
    steps_per_epoch = 300
    val_steps = 80

    train_print_every = max(1, steps_per_epoch // 10)
    val_print_every = max(1, val_steps // 5)

    history: List[Dict[str, Any]] = []
    best_mean_dice = -1.0

    # write an initial file so you can see it exists immediately
    summary: Dict[str, Any] = {
        "timestamp": ts,
        "project_dir": proj,
        "data_root": data_root,
        "device": device,
        "gpu": gpu_name,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "val_steps": val_steps,
        "crop_hw": crop_hw,
        "train_case_ids": [train_ids[0], train_ids[-1]],
        "val_case_ids": [val_ids[0], val_ids[-1]],
        "best_mean_dice": best_mean_dice,
        "total_time_s": None,
        "history": history,
        "status": "running",
    }
    write_json(metrics_latest, summary)

    def finalize(status: str):
        summary["status"] = status
        summary["best_mean_dice"] = best_mean_dice
        summary["total_time_s"] = time.time() - run_start
        write_json(metrics_latest, summary)
        write_json(metrics_final, summary)
        print("\n=== RUN END ===", flush=True)
        print("status:", status, flush=True)
        print("best_mean_dice:", f"{best_mean_dice:.4f}", flush=True)
        print("total_time_s:", f"{summary['total_time_s']:.1f}", flush=True)
        print("saved_latest:", metrics_latest, flush=True)
        print("saved_final:", metrics_final, flush=True)

    try:
        for ep in range(1, epochs + 1):
            ep_start = time.time()

            # ---- train ----
            model.train()
            run_loss = 0.0
            it = iter(train_loader)

            print(f"\n=== Epoch {ep}/{epochs} ===", flush=True)

            for step in range(steps_per_epoch):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    x, y = next(it)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(x)
                    ce = F.cross_entropy(logits, y)
                    dloss = soft_dice_loss(logits, y, num_classes=3)
                    loss = ce + dloss

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                run_loss += float(loss.item())

                if (step + 1) % train_print_every == 0 or (step + 1) == steps_per_epoch:
                    pct = int(round(100.0 * (step + 1) / steps_per_epoch))
                    avg = run_loss / (step + 1)
                    print(f"Train {pct}%  avg_loss={avg:.4f}", flush=True)

            train_loss = run_loss / max(1, steps_per_epoch)

            # ---- val ----
            print("Running validation...", flush=True)
            model.eval()
            vrun_loss = 0.0
            kidney_sum = 0.0
            tumor_sum = 0.0
            mean_sum = 0.0
            count = 0

            it = iter(val_loader)
            with torch.no_grad():
                for step in range(val_steps):
                    try:
                        x, y = next(it)
                    except StopIteration:
                        it = iter(val_loader)
                        x, y = next(it)

                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        logits = model(x)
                        ce = F.cross_entropy(logits, y)
                        dloss = soft_dice_loss(logits, y, num_classes=3)
                        loss = ce + dloss

                    vrun_loss += float(loss.item())

                    kd, td, md = dice_per_class_from_logits(logits, y)
                    kidney_sum += kd
                    tumor_sum += td
                    mean_sum += md
                    count += 1

                    if (step + 1) % val_print_every == 0 or (step + 1) == val_steps:
                        pct = int(round(100.0 * (step + 1) / val_steps))
                        avg_loss = vrun_loss / (step + 1)
                        print(
                            f"Val {pct}%  loss={avg_loss:.4f}  "
                            f"kidney_dice={kidney_sum/count:.4f}  tumor_dice={tumor_sum/count:.4f}  mean_dice={mean_sum/count:.4f}",
                            flush=True,
                        )

            val_loss = vrun_loss / max(1, val_steps)
            kidney_dice = kidney_sum / max(1, count)
            tumor_dice = tumor_sum / max(1, count)
            mean_dice = mean_sum / max(1, count)

            # checkpoints
            ckpt_path = os.path.join(ckpt_dir, f"dinov2_head_ep{ep}.pt")
            torch.save({"epoch": ep, "state_dict": model.state_dict()}, ckpt_path)

            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                best_path = os.path.join(ckpt_dir, "dinov2_head_best.pt")
                torch.save({"epoch": ep, "state_dict": model.state_dict()}, best_path)

            ep_time_s = time.time() - ep_start

            print(
                f"epoch {ep}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"kidney_dice={kidney_dice:.4f} tumor_dice={tumor_dice:.4f} mean_dice={mean_dice:.4f} "
                f"best_mean_dice={best_mean_dice:.4f} epoch_time_s={ep_time_s:.1f}",
                flush=True,
            )

            history.append({
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "kidney_dice": kidney_dice,
                "tumor_dice": tumor_dice,
                "mean_dice": mean_dice,
                "best_mean_dice_so_far": best_mean_dice,
                "epoch_time_s": ep_time_s,
                "checkpoint": os.path.relpath(ckpt_path, proj),
            })

            # write metrics after every epoch so you never lose them
            summary["best_mean_dice"] = best_mean_dice
            summary["history"] = history
            summary["total_time_s"] = time.time() - run_start
            summary["status"] = "running"
            write_json(metrics_latest, summary)

        finalize("completed")

    except KeyboardInterrupt:
        finalize("interrupted")

    except Exception as e:
        # save whatever we have and re-raise
        summary["error"] = repr(e)
        finalize("error")
        raise


if __name__ == "__main__":
    main()