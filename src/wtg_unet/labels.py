from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize


def make_structural_targets(
    mask: np.ndarray,
    radius_normalization: str = "max_per_image",
) -> tuple[np.ndarray, np.ndarray]:
    """Create binary centerline and normalized radius maps from a vessel mask."""
    binary = np.asarray(mask > 0.5, dtype=bool)
    centerline = skeletonize(binary).astype(np.float32)
    radius = ndimage.distance_transform_edt(binary).astype(np.float32)
    radius *= centerline

    if radius_normalization == "max_per_image":
        scale = float(radius.max())
        if scale > 0:
            radius /= scale
    elif radius_normalization == "none":
        pass
    else:
        raise ValueError(
            "radius_normalization must be 'max_per_image' or 'none', "
            f"got {radius_normalization!r}"
        )
    return centerline, radius.astype(np.float32)

