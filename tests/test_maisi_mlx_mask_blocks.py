"""Paired-mask MAISI MLX blocks against pinned Torch and MONAI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from golden_fixtures import seed_torch
from medmlx_core.ops import as_fp32, split_conv_transpose3d_ncdhw
from monai.networks.nets.diffusion_model_unet import SpatialTransformer

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.attention import spatial_transformer_ncdhw
from mlx_generate_ctmr.maisi_mlx_feasibility import ATTENTION_ATOL, ATTENTION_RTOL

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx


@pytest.mark.parametrize("num_splits", [1, 4])
def test_conv_transpose3d_matches_torch(mx: Mlx, num_splits: int) -> None:
    """Compare native split 3D transposed convolution with Torch."""
    generator = torch.Generator().manual_seed(23)
    module = torch.nn.ConvTranspose3d(2, 3, 3, stride=2, padding=1, output_padding=1)
    assert module.bias is not None
    with torch.no_grad():
        module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * 0.05)
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator) * 0.05)
        x = torch.randn((1, 2, 8, 16, 12), generator=generator)
        expected = module(x).numpy()
    actual = split_conv_transpose3d_ncdhw(
        as_fp32(x.numpy(), mx),
        as_fp32(module.weight.detach().numpy(), mx),
        as_fp32(module.bias.detach().numpy(), mx),
        padding=1,
        output_padding=1,
        stride=2,
        num_splits=num_splits,
        dim_split=1,
        mx=mx,
    )
    np.testing.assert_allclose(as_numpy(actual), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("norm_eps", [1e-6, 1e-2])
def test_spatial_transformer_matches_pinned_torch(mx: Mlx, norm_eps: float) -> None:
    """Compare the single-context spatial transformer with MONAI."""
    module = SpatialTransformer(
        spatial_dims=3,
        in_channels=64,
        num_attention_heads=2,
        num_head_channels=32,
        num_layers=1,
        dropout=0.0,
        norm_num_groups=32,
        norm_eps=norm_eps,
        cross_attention_dim=10,
        upcast_attention=True,
        include_fc=True,
        use_combined_linear=False,
        use_flash_attention=False,
    ).eval()
    seed_torch(23)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.02)
    x, context = torch.randn(1, 64, 2, 3, 4), torch.randn(1, 1, 10)
    with torch.inference_mode():
        expected = module(x, context=context).numpy()
    actual = spatial_transformer_ncdhw(
        as_fp32(x.numpy(), mx),
        as_fp32(context.numpy(), mx),
        {name: as_fp32(value.numpy(), mx) for name, value in module.state_dict().items()},
        num_groups=32,
        eps=norm_eps,
        num_head_channels=32,
        cross_attention_dim=10,
        include_fc=True,
        mx=mx,
    )
    np.testing.assert_allclose(as_numpy(actual), expected, rtol=ATTENTION_RTOL, atol=ATTENTION_ATOL)
