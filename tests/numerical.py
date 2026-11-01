"""Output bounds derived from independent FP32 decoder reference uncertainty."""

from __future__ import annotations

import numpy as np

from mlx_generate_ctmr._arrays import NumericArray
from mlx_generate_ctmr.maisi_mlx_feasibility import VAE_ATOL, VAE_RTOL


def assert_mr_intensity_parity(actual: NumericArray, reference_decoded: NumericArray):
    """Require each int16 voxel to be attainable within the decoder error budget.

    Truncation is discontinuous at integer intensities. Propagate the unchanged
    FP32 decoder budget through the source's monotone lower clip and *1000 map;
    this permits a one-unit difference only near a corresponding boundary.
    """
    assert actual.dtype == np.int16 and actual.shape == reference_decoded.shape
    reference = reference_decoded.astype(np.float64)
    error = VAE_ATOL + VAE_RTOL * np.abs(reference)
    lower = np.maximum(reference - error, 0).astype(np.float32) * np.float32(1000)
    upper = np.maximum(reference + error, 0).astype(np.float32) * np.float32(1000)
    # These seeded fixtures do not reach the source's int16 overflow range.
    assert np.isfinite(upper).all() and np.max(upper) < np.iinfo(np.int16).max
    lower, upper = lower.astype(np.int32), upper.astype(np.int32)
    assert np.all(actual >= lower), "MR intensity is below the propagated decoder bound"
    assert np.all(actual <= upper), "MR intensity is above the propagated decoder bound"


def assert_ct_hu_parity(
    actual: NumericArray, reference_decoded: NumericArray, label: NumericArray
) -> None:
    """Propagate the unchanged decoder budget into clipped, masked HU units."""
    assert actual.dtype == reference_decoded.dtype == np.float32
    assert actual.shape == reference_decoded.shape == label.shape
    assert np.isfinite(actual).all() and np.isfinite(reference_decoded).all()
    assert np.all(label >= 0)
    reference = reference_decoded.astype(np.float64)
    error = VAE_ATOL + VAE_RTOL * np.abs(reference)
    lower = np.clip(reference - error, 0, 1).astype(np.float32)
    upper = np.clip(reference + error, 0, 1).astype(np.float32)
    lower = lower * np.float32(2000) - np.float32(1000)
    upper = upper * np.float32(2000) - np.float32(1000)
    lower = np.where(label == 0, np.float32(-1000), lower)
    upper = np.where(label == 0, np.float32(-1000), upper)
    assert np.all(actual >= lower), "CT HU is below the propagated decoder bound"
    assert np.all(actual <= upper), "CT HU is above the propagated decoder bound"


def assert_ct_hu_transform(
    actual: NumericArray, decoded: NumericArray, label: NumericArray
) -> None:
    """Require exact pinned-source HU arithmetic on the actual decoder input."""
    import torch
    from _torch import from_numpy

    assert actual.dtype == decoded.dtype == np.float32
    assert actual.shape == decoded.shape == label.shape
    assert np.all(label >= 0)
    hu = from_numpy(decoded).clamp(0, 1) * 2000 - 1000
    reference = torch.where(from_numpy(label) > 0, hu, -1000).numpy()
    np.testing.assert_array_equal(actual, reference, err_msg="CT HU source transform")
