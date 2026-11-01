"""Independent MONAI geometry oracle for archive masks."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from _torch import from_numpy
from monai.data.meta_tensor import MetaTensor
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.spatial.array import Spacing

from mlx_generate_ctmr.mask_archive import (
    MaskGeometry,
    resample_mask_geometry,
)


@pytest.mark.parametrize(
    "shape,target_shape,spacing",
    [
        ((9, 11, 7), (8, 9, 6), (1.0, 1.0, 1.0)),
        ((8, 10, 6), (11, 13, 9), (1.0, 1.0, 1.0)),
        ((9, 11, 7), (8, 9, 6), (1.3, 1.3, 2.1)),
        ((9, 11, 7), (18, 22, 14), (0.7, 0.7, 0.8)),
    ],
)
def test_geometry_matches_monai(
    shape: tuple[int, int, int],
    target_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
) -> None:
    """Compare archive mask geometry transforms with MONAI."""
    labels = (np.arange(np.prod(shape)).reshape(shape) % 133).astype(np.int16)
    affine = np.diag((1.1, 1.1, 1.6, 1.0))
    image = MetaTensor(from_numpy(labels[None]), affine=torch.as_tensor(affine))
    reference = ResizeWithPadOrCrop(target_shape)(Spacing(spacing, mode="nearest")(image))
    actual, actual_affine = resample_mask_geometry(
        labels, affine, target=MaskGeometry(target_shape, spacing)
    )
    np.testing.assert_array_equal(actual, reference.numpy()[0])
    np.testing.assert_allclose(actual_affine, cast(MetaTensor, reference).affine.numpy(), atol=1e-7)
