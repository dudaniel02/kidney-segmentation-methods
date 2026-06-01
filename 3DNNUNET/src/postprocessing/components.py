"""
Post-processing — Connected component analysis and small-object removal.

Anatomical priors:
  - A patient has at most 2 kidneys
  - Remove tiny spurious kidney/tumour blobs
  - Tumour voxels outside kidney regions are suspicious
"""

import numpy as np
from scipy import ndimage


def remove_small_components(
    mask: np.ndarray,
    label: int,
    min_volume_voxels: int,
) -> np.ndarray:
    """Remove connected components of `label` smaller than threshold."""
    binary = (mask == label).astype(np.int32)
    labelled, n_components = ndimage.label(binary)

    if n_components == 0:
        return mask

    out = mask.copy()
    for comp_id in range(1, n_components + 1):
        comp_mask = labelled == comp_id
        if comp_mask.sum() < min_volume_voxels:
            out[comp_mask] = 0  # remove small component

    return out


def keep_n_largest(
    mask: np.ndarray,
    label: int,
    n: int,
) -> np.ndarray:
    """Keep only the N largest connected components of `label`."""
    binary = (mask == label).astype(np.int32)
    labelled, n_components = ndimage.label(binary)

    if n_components <= n:
        return mask

    # Sort components by size (descending)
    sizes = ndimage.sum(binary, labelled, range(1, n_components + 1))
    ranked = np.argsort(sizes)[::-1]  # largest first

    keep_ids = set(ranked[:n] + 1)  # +1 because component IDs start at 1

    out = mask.copy()
    for comp_id in range(1, n_components + 1):
        if comp_id not in keep_ids:
            out[labelled == comp_id] = 0

    return out


def voxel_volume_ml(spacing: tuple) -> float:
    """Compute single voxel volume in mL given spacing in mm."""
    return float(np.prod(spacing)) / 1000.0


def postprocess_segmentation(
    mask: np.ndarray,
    spacing: tuple,
    cfg_pp: dict,
) -> np.ndarray:
    """
    Apply all post-processing steps to a 3-class segmentation mask.
    
    Args:
        mask: integer array with 0=bg, 1=kidney, 2=tumour
        spacing: voxel spacing in mm (z, y, x)
        cfg_pp: postprocessing config dict from config.yaml
    """
    if not cfg_pp.get("remove_small_components", True):
        return mask

    voxel_ml = voxel_volume_ml(spacing)
    out = mask.copy()

    # Remove small kidney components
    min_kidney_vox = int(cfg_pp["min_kidney_volume_ml"] / voxel_ml)
    out = remove_small_components(out, label=1, min_volume_voxels=min_kidney_vox)

    # Remove small tumour components
    min_tumour_vox = int(cfg_pp["min_tumour_volume_ml"] / voxel_ml)
    out = remove_small_components(out, label=2, min_volume_voxels=min_tumour_vox)

    # Keep only N largest kidneys
    n_kidneys = cfg_pp.get("keep_n_largest_kidneys", 2)
    out = keep_n_largest(out, label=1, n=n_kidneys)

    return out
