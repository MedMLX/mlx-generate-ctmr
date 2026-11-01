"""Sliding-window VAE decode blending parity with MONAI."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from _torch import from_numpy
from medmlx_core.ops import as_fp32
from monai.inferers.inferer import SlidingWindowInferer

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.decode import (
    DEFAULT_OVERLAP,
    Z_SCALE,
    sliding_window_ncdhw,
)
from mlx_generate_ctmr.maisi_mlx_feasibility import VAE_ATOL, VAE_RTOL

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

_LATENT = 8
_ROI = (6, 6, 6)


def _nearest_x4(latent: Array, mx: Mlx) -> Array:
    """Produce a simple nearest-neighbor 4x decoded window."""
    image = mx.mean(latent, axis=1, keepdims=True)
    for axis in (2, 3, 4):
        image = mx.repeat(image, Z_SCALE, axis=axis)
    return image


def _ramp_latent(mx: Mlx) -> Array:
    """Build a deterministic native latent ramp."""
    axes = [np.arange(_LATENT, dtype=np.float32) * scale for scale in (1.0, 0.1, 0.01)]
    ramp = axes[0][:, None, None] + axes[1][None, :, None] + axes[2][None, None, :]
    return as_fp32(np.broadcast_to(ramp, (1, 4, _LATENT, _LATENT, _LATENT)).copy(), mx)


class _TorchDecoder(torch.nn.Module):
    def __init__(self, window_dependent: bool) -> None:
        """Mirror the native fixture decoder in Torch."""
        super().__init__()
        self.window_dependent = window_dependent

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """Average channels, optionally add the window mean, and upsample 4x."""
        image = window.mean(dim=1, keepdim=True)
        if self.window_dependent:
            image = image + window.mean()
        return torch.nn.functional.interpolate(image, scale_factor=Z_SCALE, mode="nearest-exact")


@pytest.mark.parametrize("window_dependent", [False, True])
def test_window_blending_matches_pinned_monai(mx: Mlx, window_dependent: bool) -> None:
    """Compare overlapping Gaussian blending with MONAI's SlidingWindowInferer."""
    latent = _ramp_latent(mx)

    def predict(window: Array) -> Array:
        """Decode one window like the Torch fixture decoder."""
        image = _nearest_x4(window, mx)
        return image + mx.mean(window) if window_dependent else image

    tiled = sliding_window_ncdhw(latent, predict, mx=mx, roi=_ROI, overlap=DEFAULT_OVERLAP)
    actual = as_numpy(tiled)
    inferer = SlidingWindowInferer(
        roi_size=list(_ROI),
        sw_batch_size=1,
        overlap=DEFAULT_OVERLAP,
        mode="gaussian",
        sigma_scale=0.125,
        progress=False,
    )
    with torch.inference_mode():
        expected = cast(
            torch.Tensor, inferer(from_numpy(as_numpy(latent)), _TorchDecoder(window_dependent))
        )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=VAE_RTOL, atol=VAE_ATOL)
    if not window_dependent:
        untiled = as_numpy(_nearest_x4(latent, mx))
        np.testing.assert_allclose(actual, untiled, rtol=VAE_RTOL, atol=VAE_ATOL)
