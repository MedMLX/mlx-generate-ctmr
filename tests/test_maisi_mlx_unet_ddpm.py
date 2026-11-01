"""``ddpm-ct`` UNet/ControlNet parity: resblock_updown=False plus body-region conditioning.

Uses seeded tiny MONAI weights; no pretrained checkpoint is exercised, so this checks
the architecture and not released-weight output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from _torch import from_numpy
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.controlnet import controlnet_residuals_ncdhw
from mlx_generate_ctmr.maisi_mlx_feasibility import (
    CONTROLNET_ATOL,
    CONTROLNET_RTOL,
    UNET_ATOL,
    UNET_RTOL,
)
from mlx_generate_ctmr.unet import denoise_unet_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import (
        Array,
        ArrayConstructor,
        ControlnetConstructor,
        Mlx,
        RflowConstructor,
    )

_SHARED = {
    "spatial_dims": 3,
    "num_channels": [4, 8],
    "attention_levels": [False, True],
    "num_head_channels": [0, 2],
    "num_res_blocks": 1,
    "norm_num_groups": 2,
    "norm_eps": 1e-6,
    "use_flash_attention": False,
    "num_class_embeds": None,
    "resblock_updown": False,
    "include_fc": False,
    "with_conditioning": False,
    "use_combined_linear": False,
}
_UNET = cast(
    "RflowConstructor",
    {
        **_SHARED,
        "in_channels": 4,
        "out_channels": 4,
        "include_top_region_index_input": True,
        "include_bottom_region_index_input": True,
        "include_spacing_input": True,
    },
)
_CONTROLNET = cast(
    "ControlnetConstructor",
    {
        **_SHARED,
        "in_channels": 4,
        "conditioning_embedding_in_channels": 8,
        "conditioning_embedding_num_channels": [4, 4, 4],
    },
)


def _seeded_weights(module: torch.nn.Module, *, seed: int, mx: Mlx) -> dict[str, Array]:
    """Initialize reference weights from a fixed seed and return native copies."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.empty_like(parameter).normal_(0.0, 0.02, generator=generator))
    return {name: as_fp32(tensor.numpy(), mx) for name, tensor in module.state_dict().items()}


def test_ddpm_ct_unet_matches_pinned_torch(mx: Mlx) -> None:
    """Compare the region-conditioned UNet with the source."""
    module = DiffusionModelUNetMaisi(**_UNET).eval()
    weights = _seeded_weights(module, seed=4101, mx=mx)
    rng = np.random.default_rng(4102)
    x = rng.normal(size=(1, 4, 4, 4, 4)).astype(np.float32)
    timesteps = np.asarray([500.0], dtype=np.float32)
    top, bottom = (rng.normal(size=(1, 4)).astype(np.float32) for _ in range(2))
    spacing = rng.normal(size=(1, 3)).astype(np.float32)
    with torch.inference_mode():
        expected = module(
            from_numpy(x),
            from_numpy(timesteps),
            top_region_index_tensor=from_numpy(top),
            bottom_region_index_tensor=from_numpy(bottom),
            spacing_tensor=from_numpy(spacing),
        )
    array = cast("ArrayConstructor", mx.array)
    inputs = (array(x), array(timesteps), None, array(spacing), weights)
    _t_emb, actual = denoise_unet_ncdhw(
        *inputs,
        constructor=_UNET,
        mx=mx,
        top_region_index=array(top),
        bottom_region_index=array(bottom),
    )
    np.testing.assert_allclose(as_numpy(actual), expected.numpy(), rtol=UNET_RTOL, atol=UNET_ATOL)


def test_ddpm_ct_controlnet_matches_pinned_torch(mx: Mlx) -> None:
    """Compare CT DDPM ControlNet output with the source."""
    module = ControlNetMaisi(**_CONTROLNET).eval()
    weights = _seeded_weights(module, seed=4103, mx=mx)
    rng = np.random.default_rng(4104)
    x = rng.normal(size=(1, 4, 4, 4, 4)).astype(np.float32)
    timesteps = np.asarray([250.0], dtype=np.float32)
    cond = rng.normal(size=(1, 8, 16, 16, 16)).astype(np.float32)
    with torch.inference_mode():
        down_t, mid_t = module(
            from_numpy(x), from_numpy(timesteps), from_numpy(cond)
        )
    array = cast("ArrayConstructor", mx.array)
    down, mid = controlnet_residuals_ncdhw(
        array(x), array(timesteps), None, array(cond), weights, constructor=_CONTROLNET, mx=mx
    )
    for actual, expected in zip([*down, mid], [*down_t, mid_t], strict=True):
        np.testing.assert_allclose(
            as_numpy(actual), expected.numpy(), rtol=CONTROLNET_RTOL, atol=CONTROLNET_ATOL
        )
