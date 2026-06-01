import os
import time
import numpy as np
import nibabel as nib
import torch
from transformers import AutoImageProcessor, Dinov2Model


def window_ct(x: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)


def ct_to_3ch(slice_hu: np.ndarray) -> np.ndarray:
    # 3 windowings -> fake RGB
    c1 = window_ct(slice_hu, center=40, width=400)    # soft tissue
    c2 = window_ct(slice_hu, center=80, width=200)    # kidney-ish
    c3 = window_ct(slice_hu, center=0, width=1000)    # wide abdomen
    img = np.stack([c1, c2, c3], axis=-1)
    img = (img * 255.0).round().clip(0, 255).astype(np.uint8)
    return img


def pad_to_multiple(img: np.ndarray, multiple: int = 14) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Pad HxWxC to (ceil(H/m)*m, ceil(W/m)*m) using edge padding.
    Returns padded image and original (H,W).
    """
    h, w = img.shape[:2]
    nh = ((h + multiple - 1) // multiple) * multiple
    nw = ((w + multiple - 1) // multiple) * multiple
    pad_h = nh - h
    pad_w = nw - w
    img2 = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return img2, (h, w)


def load_kits19_slice(data_root: str, case_id: int = 0, slice_idx: int | None = None):
    case_dir = os.path.join(data_root, f"case_{case_id:05d}")
    img_p = os.path.join(case_dir, "imaging.nii.gz")
    seg_p = os.path.join(case_dir, "segmentation.nii.gz")

    img_nii = nib.load(img_p)
    vol = img_nii.get_fdata(dtype=np.float32)  # usually (H,W,Z)
    vol = np.transpose(vol, (2, 0, 1))         # -> (Z,H,W)

    seg = None
    if os.path.exists(seg_p):
        seg_nii = nib.load(seg_p)
        seg = seg_nii.get_fdata(dtype=np.float32)
        seg = np.transpose(seg, (2, 0, 1)).astype(np.uint8)

    zdim = vol.shape[0]
    if slice_idx is None:
        # pick a slice with any label if available, else middle slice
        if seg is not None:
            pos = np.where((seg > 0).reshape(zdim, -1).any(axis=1))[0]
            slice_idx = int(pos[len(pos) // 2]) if len(pos) > 0 else (zdim // 2)
        else:
            slice_idx = zdim // 2

    hu = vol[slice_idx]
    m = seg[slice_idx] if seg is not None else None
    return case_dir, slice_idx, hu, m


def main():
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "gpu:", torch.cuda.get_device_name(0) if device == "cuda" else "none")

    # Load one slice
    case_dir, z, hu, mask = load_kits19_slice(data_root, case_id=0, slice_idx=None)
    if mask is not None:
        print(f"example: case=00000 slice={z} unique_mask={np.unique(mask)}")
    else:
        print(f"example: case=00000 slice={z} (no mask file)")

    # Prep image
    img = ct_to_3ch(hu)              # uint8 HxWx3
    img_pad, orig_hw = pad_to_multiple(img, multiple=14)
    H, W = img_pad.shape[:2]
    print("input padded:", img_pad.shape, "orig_hw:", orig_hw)
    assert H % 14 == 0 and W % 14 == 0, (H, W)

    # Tensor in [0,1], shape (1,3,H,W)
    x = torch.from_numpy(img_pad).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)

    # Load DINOv2
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device).eval()

    # Normalize like processor
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std

    ps = model.config.patch_size  # typically 14
    gh = H // ps
    gw = W // ps

    with torch.no_grad():
        t0 = time.time()
        out = model(pixel_values=x)
        t1 = time.time()

    tokens = out.last_hidden_state                  # (B, 1+N, D)
    patch_tokens = tokens[:, 1:, :]                 # (B, N, D)
    B, N, D = patch_tokens.shape
    expected = gh * gw

    print("tokens:", tokens.shape, "patch_tokens:", patch_tokens.shape)
    print("grid:", (gh, gw), "embed_dim:", D, "expected_tokens:", expected)
    print(f"forward_time: {(t1 - t0) * 1000:.1f} ms")

    # This is the key fix: grid comes from (H,W)/patch_size, not sqrt(N)
    assert N == expected, f"Token mismatch: N={N} expected={expected} from H,W={H,W} ps={ps}"

    feat = patch_tokens.transpose(1, 2).contiguous().view(B, D, gh, gw)
    print("feat grid tensor:", feat.shape)  # (B, D, gh, gw)

    # Optional: show how you'd upsample to image size for segmentation heads later
    feat_up = torch.nn.functional.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
    print("feat upsampled:", feat_up.shape)  # (B, D, H, W)


if __name__ == "__main__":
    main()