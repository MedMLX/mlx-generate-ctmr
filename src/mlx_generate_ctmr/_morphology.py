"""Label-preserving slice morphology with an explicit spatial axis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as _NDIMAGE

from mlx_generate_ctmr._arrays import IntegerScalar


def fill_label_envelope_holes[LabelScalar: IntegerScalar](
    labels: NDArray[LabelScalar],
    *,
    slice_axis: int,
    body_label: int,
) -> NDArray[LabelScalar]:
    """Return a copy of ``labels`` with enclosed background set to ``body_label``.

    ``labels`` is a 3D integer array. Each slice along ``slice_axis`` is treated
    independently: zero voxels that cannot reach the slice border through other
    zero voxels (in-plane, 4-connected) are set to ``body_label``; all other
    voxels keep their value.

    Raises:
        ValueError: If ``labels`` is not 3D.
    """
    if labels.ndim != 3:
        raise ValueError(f"label envelope requires a 3D array; got {labels.shape}")
    result = labels.copy()
    slices = np.moveaxis(result, slice_axis, -1)
    for index in range(slices.shape[-1]):
        plane = slices[..., index]
        support = plane != 0
        plane[_NDIMAGE.binary_fill_holes(support) & ~support] = body_label
    return result
