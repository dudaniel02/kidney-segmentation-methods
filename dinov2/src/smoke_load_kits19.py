import os
import numpy as np
import nibabel as nib

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_ROOT = os.path.abspath(DATA_ROOT)

case_id = 0
case_dir = os.path.join(DATA_ROOT, f"case_{case_id:05d}")
img_p = os.path.join(case_dir, "imaging.nii.gz")
seg_p = os.path.join(case_dir, "segmentation.nii.gz")

print("DATA_ROOT:", DATA_ROOT)
print("CASE_DIR:", case_dir)

img = nib.load(img_p)
vol = img.get_fdata(dtype=np.float32)
print("imaging shape (nib):", vol.shape, "dtype:", vol.dtype)

if os.path.exists(seg_p):
    seg = nib.load(seg_p).get_fdata(dtype=np.float32)
    seg = seg.astype(np.uint8)
    print("seg shape (nib):", seg.shape, "unique:", np.unique(seg)[:10])
else:
    print("segmentation missing (test case?)")
