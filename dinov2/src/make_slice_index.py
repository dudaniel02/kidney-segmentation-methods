import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "slice_index_kits19_train210.json"))

LABELED_MAX = 209  # 0..209 have segmentation in KiTS19
MIN_PIXELS = 20    # slice must have at least this many labeled pixels to count as kidney/tumor

def load_seg(seg_path: str) -> np.ndarray:
    seg = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.uint8)
    # seg is (H, W, Z) in nib, your earlier print showed that
    # we'll treat axis 2 as Z
    return seg

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    index = {
        "data_root": DATA_ROOT,
        "cases": [],
        "stats": {
            "num_cases": 0,
            "total_slices": 0,
            "kidney_slices": 0,
            "tumor_slices": 0,
        }
    }

    for cid in tqdm(range(LABELED_MAX + 1), desc="indexing cases"):
        case_dir = os.path.join(DATA_ROOT, f"case_{cid:05d}")
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        img_path = os.path.join(case_dir, "imaging.nii.gz")

        if not os.path.exists(img_path):
            print(f"missing imaging for case {cid}: {img_path}")
            continue
        if not os.path.exists(seg_path):
            print(f"missing segmentation for case {cid}: {seg_path}")
            continue

        seg = load_seg(seg_path)  # (H,W,Z)
        h, w, z = seg.shape

        kidney_slices = []
        tumor_slices = []
        any_pos_slices = []

        for zi in range(z):
            sl = seg[:, :, zi]
            k = int((sl == 1).sum())
            t = int((sl == 2).sum())
            if k >= MIN_PIXELS:
                kidney_slices.append(zi)
            if t >= MIN_PIXELS:
                tumor_slices.append(zi)
            if (k + t) >= MIN_PIXELS:
                any_pos_slices.append(zi)

        index["cases"].append({
            "case_id": cid,
            "case_dir": case_dir,
            "shape_hwc": [h, w, z],
            "kidney_slices": kidney_slices,
            "tumor_slices": tumor_slices,
            "any_pos_slices": any_pos_slices,
        })

        index["stats"]["num_cases"] += 1
        index["stats"]["total_slices"] += z
        index["stats"]["kidney_slices"] += len(kidney_slices)
        index["stats"]["tumor_slices"] += len(tumor_slices)

    with open(OUT_PATH, "w") as f:
        json.dump(index, f)

    print("Wrote:", OUT_PATH)
    print("Stats:", index["stats"])

if __name__ == "__main__":
    main()
