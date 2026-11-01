"""MAISI MLX ControlNet residual and conditioned-UNet parity."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from _torch import from_numpy
from golden_fixtures import load_boundary, mlx_weights, torch_weights
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from mlx_generate_ctmr import controlnet
from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.maisi_mlx_feasibility import (
    CONTROLNET_ATOL,
    CONTROLNET_RTOL,
    UNET_ATOL,
    UNET_RTOL,
)
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import ArrayConstructor, Mlx


def test_controlnet_residuals_match_torch_golden(mx: Mlx) -> None:
    """Compare ControlNet residuals with recorded Torch outputs."""
    manifest, arrays = load_boundary("controlnet")
    down, mid = controlnet.controlnet_residuals_ncdhw(
        as_fp32(arrays["x"], mx),
        as_fp32(arrays["timesteps"], mx),
        cast("ArrayConstructor", mx.array)(arrays["class_labels"], dtype=mx.int32),
        as_fp32(arrays["controlnet_cond"], mx),
        mlx_weights(arrays, mx),
        constructor=manifest["constructor"],
        mx=mx,
        conditioning_scale=1.0,
    )
    assert len(down) == 4
    for actual, key in ((mid, "mid"), *((d, f"down_{i}") for i, d in enumerate(down))):
        np.testing.assert_allclose(
            as_numpy(actual), arrays[key], rtol=CONTROLNET_RTOL, atol=CONTROLNET_ATOL
        )


@pytest.mark.parametrize("residual_scale", [1.0, 100.0])
def test_conditioned_unet_matches_pinned_torch(mx: Mlx, residual_scale: float) -> None:
    """Compare a UNet conditioned on recorded ControlNet residuals with the source."""
    unet_manifest, unet_arrays = load_boundary("diffusion")
    _cn_manifest, cn_arrays = load_boundary("controlnet")
    residuals = [cn_arrays[f"down_{index}"] * residual_scale for index in range(4)]
    mid = cn_arrays["mid"]
    module = DiffusionModelUNetMaisi(**unet_manifest["constructor"]).to("cpu").eval()
    module.load_state_dict(torch_weights(unet_arrays), strict=True)
    with torch.inference_mode():
        y_t = module(
            from_numpy(unet_arrays["x"]),
            from_numpy(unet_arrays["timesteps"]),
            class_labels=from_numpy(unet_arrays["class_labels"]),
            spacing_tensor=from_numpy(unet_arrays["spacing"]),
            down_block_additional_residuals=tuple(from_numpy(item) for item in residuals),
            mid_block_additional_residual=from_numpy(mid),
        )
    _t_emb, y = denoise_unet_ncdhw(
        as_fp32(unet_arrays["x"], mx),
        as_fp32(unet_arrays["timesteps"], mx),
        cast("ArrayConstructor", mx.array)(unet_arrays["class_labels"], dtype=mx.int32),
        as_fp32(unet_arrays["spacing"], mx),
        mlx_weights(unet_arrays, mx),
        constructor=unet_manifest["constructor"],
        mx=mx,
        down_block_additional_residuals=tuple(as_fp32(item, mx) for item in residuals),
        mid_block_additional_residual=as_fp32(mid, mx),
    )
    np.testing.assert_allclose(as_numpy(y), y_t.numpy(), rtol=UNET_RTOL, atol=UNET_ATOL)
