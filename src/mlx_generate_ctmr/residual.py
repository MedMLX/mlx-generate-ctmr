"""MLX port of ``DiffusionUNetResnetBlock`` (NCDHW, optional skip/up/down)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx, ResidualArithmetic

from collections.abc import Mapping

from medmlx_core import ops
from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr import fast_fp32

_WEIGHT_KEYS = (
    "norm1.weight",
    "norm1.bias",
    "conv1.conv.weight",
    "conv1.conv.bias",
    "time_emb_proj.weight",
    "time_emb_proj.bias",
    "norm2.weight",
    "norm2.bias",
    "conv2.conv.weight",
    "conv2.conv.bias",
)
_SKIP_KEYS = ("skip_connection.conv.weight", "skip_connection.conv.bias")


def residual_block_ncdhw(
    x: Array,
    emb: Array,
    weights: Mapping[str, Array],
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
    up: bool = False,
    down: bool = False,
    arithmetic: ResidualArithmetic = ops,
) -> Array:
    """Timestep-conditioned residual block on an NCDHW tensor.

    Steps: GroupNorm, SiLU, optional 2x nearest upsample (``up``) or 2x average
    pool (``down``), 3x3x3 conv, add the projected embedding, GroupNorm, SiLU,
    conv, then add the skip path. ``x`` is ``(N, C_in, D, H, W)`` and ``emb`` is
    ``(N, temb_dim)``. The skip path is resampled the same way as the main path
    and uses a 1x1x1 conv when ``C_in`` differs from the output channels.
    ``arithmetic`` supplies the primitives: ``medmlx_core.ops`` by default,
    ``source_fp32`` for the CUDA-matched CT/MR UNet, or ``fast_fp32``, which runs
    the fused ``fast_fp32.residual_block_ncdhw`` after the same validation.

    Raises:
        ValueError: If ``x`` is not NCDHW, ``emb`` is not 2D, ``up`` and ``down``
            are both set, or weights are missing.
    """

    require_ncdhw(x, name="residual input")
    if int(emb.ndim) != 2:
        raise ValueError(f"residual emb must be [N, temb]; got ndim={emb.ndim}")
    if up and down:
        raise ValueError("residual block cannot upsample and downsample together")
    missing = [key for key in _WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError("residual weights missing: " + ", ".join(missing))

    in_channels = int(x.shape[1])
    out_channels = int(weights["conv1.conv.weight"].shape[0])
    if in_channels != out_channels:
        skip_missing = [key for key in _SKIP_KEYS if key not in weights]
        if skip_missing:
            raise ValueError("residual skip weights missing: " + ", ".join(skip_missing))
    if arithmetic is fast_fp32:
        return fast_fp32.residual_block_ncdhw(
            x, emb, weights, num_groups=num_groups, eps=eps, mx=mx, up=up, down=down
        )

    hidden = arithmetic.group_norm_ncdhw(
        x,
        weights["norm1.weight"],
        weights["norm1.bias"],
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.silu(hidden, mx)
    skip = x
    if up:
        skip = arithmetic.upsample_nearest_ncdhw(skip, mx=mx)
        hidden = arithmetic.upsample_nearest_ncdhw(hidden, mx=mx)
    elif down:
        skip = arithmetic.avg_pool3d_ncdhw(skip, mx=mx)
        hidden = arithmetic.avg_pool3d_ncdhw(hidden, mx=mx)
    hidden = arithmetic.conv3d_ncdhw(
        hidden,
        weights["conv1.conv.weight"],
        weights["conv1.conv.bias"],
        padding=1,
        mx=mx,
    )
    temb = arithmetic.linear(
        arithmetic.silu(emb, mx),
        weights["time_emb_proj.weight"],
        weights["time_emb_proj.bias"],
        mx,
    )
    hidden = hidden + mx.reshape(temb, (int(temb.shape[0]), int(temb.shape[1]), 1, 1, 1))
    hidden = arithmetic.group_norm_ncdhw(
        hidden,
        weights["norm2.weight"],
        weights["norm2.bias"],
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.silu(hidden, mx)
    hidden = arithmetic.conv3d_ncdhw(
        hidden,
        weights["conv2.conv.weight"],
        weights["conv2.conv.bias"],
        padding=1,
        mx=mx,
    )
    if in_channels != out_channels:
        skip = arithmetic.conv3d_ncdhw(
            skip,
            weights["skip_connection.conv.weight"],
            weights["skip_connection.conv.bias"],
            padding=0,
            mx=mx,
        )
    return skip + hidden
