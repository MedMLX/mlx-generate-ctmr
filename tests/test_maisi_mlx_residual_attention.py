"""MAISI MLX residual and attention parity against pinned Torch goldens."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from golden_fixtures import load_boundary, mlx_weights
from medmlx_core.ops import as_fp32

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.attention import spatial_attention_ncdhw
from mlx_generate_ctmr.maisi_mlx_feasibility import (
    ATTENTION_ATOL,
    ATTENTION_RTOL,
    RESIDUAL_ATOL,
    RESIDUAL_RTOL,
)
from mlx_generate_ctmr.residual import residual_block_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx


def test_residual_fp32_matches_torch_golden(mx: Mlx) -> None:
    """Compare FP32 residual outputs with the Torch recording."""
    manifest, arrays = load_boundary("residual")
    constructor = manifest["constructor"]
    y = residual_block_ncdhw(
        as_fp32(arrays["x"], mx),
        as_fp32(arrays["emb"], mx),
        mlx_weights(arrays, mx),
        num_groups=int(constructor["norm_num_groups"]),
        eps=float(constructor["norm_eps"]),
        mx=mx,
    )
    np.testing.assert_allclose(as_numpy(y), arrays["y"], rtol=RESIDUAL_RTOL, atol=RESIDUAL_ATOL)


def test_attention_fp32_matches_torch_golden(mx: Mlx) -> None:
    """Compare FP32 attention outputs with the Torch recording."""
    manifest, arrays = load_boundary("attention")
    constructor = manifest["constructor"]
    assert manifest["attention_scale"] == pytest.approx(32**-0.5)
    y = spatial_attention_ncdhw(
        as_fp32(arrays["x"], mx),
        mlx_weights(arrays, mx),
        num_groups=int(constructor["norm_num_groups"]),
        eps=float(constructor["norm_eps"]),
        num_head_channels=int(constructor["num_head_channels"]),
        include_fc=bool(constructor["include_fc"]),
        mx=mx,
    )
    np.testing.assert_allclose(as_numpy(y), arrays["y"], rtol=ATTENTION_RTOL, atol=ATTENTION_ATOL)
