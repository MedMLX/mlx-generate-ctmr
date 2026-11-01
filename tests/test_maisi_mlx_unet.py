"""MAISI MLX rflow UNet parity against recorded goldens and pinned Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from _torch import from_numpy
from golden_fixtures import load_boundary, mlx_weights, torch_weights
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from mlx_generate_ctmr._arrays import NumericArray, as_numpy
from mlx_generate_ctmr.maisi_mlx_feasibility import UNET_ATOL, UNET_RTOL
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ArrayConstructor, Mlx


def _unet_inputs(arrays: dict[str, NumericArray], mx: Mlx) -> tuple[Array, Array, Array, Array]:
    """Return native x, timesteps, class labels and spacing from the diffusion golden."""
    labels = cast("ArrayConstructor", mx.array)(arrays["class_labels"], dtype=mx.int32)
    x, t, spacing = (as_fp32(arrays[key], mx) for key in ("x", "timesteps", "spacing"))
    return x, t, labels, spacing


def test_unet_denoise_matches_torch_golden(mx: Mlx) -> None:
    """Compare native UNet denoising with recorded Torch output."""
    manifest, arrays = load_boundary("diffusion")
    t_emb, y = denoise_unet_ncdhw(
        *_unet_inputs(arrays, mx),
        mlx_weights(arrays, mx),
        constructor=manifest["constructor"],
        mx=mx,
    )
    # The timestep sinusoid at t=1000 is not bit-identical across backends.
    np.testing.assert_allclose(as_numpy(t_emb), arrays["t_emb"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(as_numpy(y), arrays["y"], rtol=UNET_RTOL, atol=UNET_ATOL)


def test_unet_controlnet_residuals_match_pinned_torch(mx: Mlx) -> None:
    """Compare a UNet with a ControlNet mid residual with the source."""
    manifest, arrays = load_boundary("diffusion")
    constructor = manifest["constructor"]
    module = DiffusionModelUNetMaisi(**constructor).to("cpu").eval()
    module.load_state_dict(torch_weights(arrays), strict=True)
    mid = torch.full((1, constructor["num_channels"][-1], 2, 2, 2), 0.01, dtype=torch.float32)
    with torch.inference_mode():
        y_t = module(
            from_numpy(arrays["x"]),
            from_numpy(arrays["timesteps"]),
            class_labels=from_numpy(arrays["class_labels"]),
            spacing_tensor=from_numpy(arrays["spacing"]),
            mid_block_additional_residual=mid,
        )
    inputs, weights = _unet_inputs(arrays, mx), mlx_weights(arrays, mx)
    _t_emb, y = denoise_unet_ncdhw(
        *inputs,
        weights,
        constructor=constructor,
        mx=mx,
        mid_block_additional_residual=as_fp32(mid.numpy(), mx),
    )
    np.testing.assert_allclose(as_numpy(y), y_t.numpy(), rtol=UNET_RTOL, atol=UNET_ATOL)
