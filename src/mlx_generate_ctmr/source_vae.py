"""MAISI VAE arithmetic preserving the pinned CUDA FP32 decoder operation order."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

import math

from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr.source_fp32 import (
    conv3d_ncdhw as conv3d_ncdhw,
)
from mlx_generate_ctmr.source_fp32 import (
    silu as silu,
)
from mlx_generate_ctmr.source_fp32 import (
    source_kernel,
)


def _reduce(groups: Array, *, variance: bool, mx: Mlx) -> Array:
    """Per-row mean (or variance when `variance`) of a (rows, columns) float32 array.

    Two Metal kernels (per-block partials, then a final combine) follow the
    reference's reduction partition so summation order, and hence rounding, matches.
    """
    rows, columns = (int(n) for n in groups.shape)
    vector = columns >= 128
    width = 2 if variance else 4
    work = min(512, columns // width if vector else columns)
    threads = 1 << (work.bit_length() - 1)
    per_thread = (columns + threads - 1) // threads
    # The reference Reduce.cuh targets 128 SMs with 1536 resident threads each.
    # Reproduce its partition; native Metal dispatch remains independent of it.
    blocks = (
        max(min(128 * (1536 // threads), (per_thread + 15) // 16), (per_thread + 255) // 256)
        if per_thread >= 256
        else 1
    )
    partial = source_kernel("vae_reduce_partial", mx)(
        inputs=[groups],
        template=[
            ("TGS", threads),
            ("CTAS", blocks),
            ("VEC", width),
            ("VECTOR", vector),
            ("VARIANCE", variance),
        ],
        output_shapes=[(rows * blocks, 3)],
        output_dtypes=[mx.float32],
        grid=(rows * blocks * threads, 1, 1),
        threadgroup=(threads, 1, 1),
    )[0]
    return source_kernel("vae_reduce_final", mx)(
        inputs=[partial],
        template=[("TGS", threads), ("CTAS", blocks), ("N", columns), ("VARIANCE", variance)],
        output_shapes=[(rows,)],
        output_dtypes=[mx.float32],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
    )[0]


def group_norm_ncdhw(
    array: Array,
    weight: Array,
    bias: Array,
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
) -> Array:
    """Apply source-ordered group normalization to a channels-first 5D array.

    Mean and population variance use separate reductions, followed by sqrt,
    division and channel affine. Channel count must be divisible by ``num_groups``;
    callers supply per-channel weights and biases in the expected shapes.
    """
    require_ncdhw(array, name="MAISI normalization input")
    if int(array.shape[1]) % num_groups:
        raise ValueError("group_norm channels must be divisible by num_groups")
    groups = mx.reshape(array, (int(array.shape[0]) * num_groups, -1))
    mean = _reduce(groups, variance=False, mx=mx)
    variance = _reduce(groups, variance=True, mx=mx)
    normalized = (groups - mean[:, None]) / mx.sqrt(variance[:, None] + eps)
    return mx.reshape(normalized, array.shape) * mx.reshape(weight, (1, -1, 1, 1, 1)) + mx.reshape(
        bias, (1, -1, 1, 1, 1)
    )


def upsample_trilinear_ncdhw(array: Array, *, mx: Mlx, scale: int = 2) -> Array:
    """Upsample channels-first 5D data by exactly two on each spatial axis.

    Uses the source's half-pixel trilinear coordinates and explicit nested FMA rounding;
    other scale factors are rejected.
    """
    require_ncdhw(array, name="MAISI trilinear input")
    if scale != 2:
        raise ValueError("NV-Generate VAE trilinear upsampling requires scale=2")
    shape = (*tuple(array.shape[:2]), *(int(n) * 2 for n in array.shape[2:]))
    return source_kernel("vae_trilinear", mx)(
        inputs=[array],
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        grid=(math.prod(shape), 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
