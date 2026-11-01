"""MAISI MLX shared blocks against pinned Torch and MONAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
import torch.nn.functional as torch_f
from _torch import from_numpy
from golden_fixtures import seed_torch
from medmlx_core.ops import as_fp32, upsample_trilinear_ncdhw
from monai.networks.nets.diffusion_model_unet import (
    DiffusionUNetResnetBlock,
    get_timestep_embedding,
)

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.embed import sinusoidal_timestep_embedding
from mlx_generate_ctmr.maisi_mlx_feasibility import RESIDUAL_ATOL, RESIDUAL_RTOL
from mlx_generate_ctmr.residual import residual_block_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx


def test_sinusoidal_timestep_embedding_matches_pinned_torch(mx: Mlx) -> None:
    """Compare timestep embeddings with the source and zero-pad odd dimensions."""
    steps = np.array([0.0, 64.5, 500.0, 1000.0], dtype=np.float32)
    expected = get_timestep_embedding(from_numpy(steps), 64).numpy()
    actual = sinusoidal_timestep_embedding(mx.array(steps), 64, mx=mx)
    # sin/cos of large timesteps is not bit-identical across backends.
    np.testing.assert_allclose(as_numpy(actual), expected, rtol=1e-4, atol=1e-4)
    odd = as_numpy(sinusoidal_timestep_embedding(mx.array(steps), 5, mx=mx))
    assert odd.shape == (4, 5)
    np.testing.assert_array_equal(odd[:, -1], 0.0)


def test_trilinear_upsample_matches_torch_align_corners_false(mx: Mlx) -> None:
    """Compare trilinear upsampling with Torch align-corners-false behavior."""
    values = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 1, 2, 2, 2)
    expected = torch_f.interpolate(
        from_numpy(values), scale_factor=2.0, mode="trilinear", align_corners=False
    ).numpy()
    actual = upsample_trilinear_ncdhw(as_fp32(values, mx), mx=mx)
    np.testing.assert_allclose(as_numpy(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("up", "down", "in_channels", "out_channels"), [(False, True, 8, 16), (True, False, 8, 8)]
)
def test_residual_skip_up_down_match_pinned_torch(
    mx: Mlx, up: bool, down: bool, in_channels: int, out_channels: int
) -> None:
    """Compare residual skip and resampling paths with the source."""
    module = DiffusionUNetResnetBlock(
        spatial_dims=3,
        in_channels=in_channels,
        temb_channels=8,
        out_channels=out_channels,
        up=up,
        down=down,
        norm_num_groups=8,
        norm_eps=1e-6,
    ).eval()
    seed_torch(22)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.02)
    x, emb = torch.randn(1, in_channels, 4, 4, 4), torch.randn(1, 8)
    with torch.inference_mode():
        expected = module(x, emb).numpy()
    y = residual_block_ncdhw(
        as_fp32(x.numpy(), mx),
        as_fp32(emb.numpy(), mx),
        {name: as_fp32(value.numpy(), mx) for name, value in module.state_dict().items()},
        num_groups=8,
        eps=1e-6,
        mx=mx,
        up=up,
        down=down,
    )
    np.testing.assert_allclose(as_numpy(y), expected, rtol=RESIDUAL_RTOL, atol=RESIDUAL_ATOL)
