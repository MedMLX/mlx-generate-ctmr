"""Native Metal arithmetic that follows the pinned CUDA FP32 generation reference.

CUDA's reduction order and elementary-function approximations are observable
under repeated denoising and classifier-free guidance. These kernels follow
those operation orders without importing Torch or executing any CUDA computation.
Matrix-product dispatch depends on batch, layout and dimensions, and the
manifest (``source_fp32_manifest.json``) records only the grids that were tested,
so agreement with the CUDA reference is not claimed for other shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, MetalKernel, Mlx

import math
from functools import cache
from pathlib import Path

import numpy as np
from medmlx_core.layout import require_ncdhw
from medmlx_core.ops import (
    avg_pool3d_ncdhw as avg_pool3d_ncdhw,
)
from medmlx_core.ops import (
    upsample_nearest_ncdhw as upsample_nearest_ncdhw,
)

from mlx_generate_ctmr._source_fp32_kernels import KERNELS


@cache
def source_kernel(name: str, mx: Mlx) -> MetalKernel:
    """Create and cache a named Metal kernel from the packaged source definitions."""
    inputs, outputs, header, source = KERNELS[name]
    return mx.fast.metal_kernel(
        name=f"nv_generate_{name}",
        input_names=list(inputs),
        output_names=list(outputs),
        header=header,
        source=source,
    )


@cache
def _tables(mx: Mlx) -> tuple[Array, Array]:
    """Load the packaged constants used by source-ordered FP32 operations."""
    with np.load(Path(__file__).with_name("source_fp32_tables.npz"), allow_pickle=False) as data:
        rsqrt = np.cumsum(data["rsqrt"], dtype=np.uint32).view(np.float32)
        positive = np.cumsum(data["exp2_positive"], dtype=np.uint32)
        negative = np.cumsum(data["exp2_negative"], dtype=np.uint32)
    exp_table = mx.array(np.concatenate((positive, negative)))
    rsqrt_table = mx.array(rsqrt)
    mx.eval(exp_table, rsqrt_table)
    return exp_table, rsqrt_table


def _apply(
    name: str,
    inputs: list[Array],
    shape: tuple[int, ...],
    mx: Mlx,
    *,
    template: list[tuple[str, int | bool]] | None = None,
    threads: int = 256,
) -> Array:
    """Launch the named source kernel with one thread per output element.

    Returns a float32 array of `shape`.
    """
    return source_kernel(name, mx)(
        inputs=inputs,
        template=template or [],
        output_shapes=[shape],
        output_dtypes=[mx.float32],
        grid=(math.prod(shape), 1, 1),
        threadgroup=(threads, 1, 1),
    )[0]


def _bias(weight: Array, bias: Array | None, mx: Mlx) -> Array:
    """Return the supplied bias or zeros for the output channels."""
    return mx.zeros((int(weight.shape[0]),), dtype=mx.float32) if bias is None else bias


def conv3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    mx: Mlx,
    stride: int = 1,
) -> Array:
    """cuDNN FP32 channel/kernel order, including padding and the final bias add."""
    require_ncdhw(array, name="source conv3d input")
    kernel = int(weight.shape[2])
    if tuple(weight.shape[2:]) != (kernel, kernel, kernel):
        raise ValueError("NV-Generate source convolution requires a cubic kernel")
    shape = (
        int(array.shape[0]),
        int(weight.shape[0]),
        *((int(size) + 2 * padding - kernel) // stride + 1 for size in array.shape[2:]),
    )
    return _apply(
        "ops_conv",
        [array, weight, _bias(weight, bias, mx)],
        shape,
        mx,
        template=[("KS", kernel), ("P", padding), ("S", stride)],
    )


def group_norm_ncdhw(
    array: Array,
    weight: Array,
    bias: Array,
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
) -> Array:
    """CUDA Welford reduction, reciprocal square root, and affine rounding."""
    require_ncdhw(array, name="source group_norm input")
    if int(array.shape[1]) % num_groups:
        raise ValueError("group_norm channels must be divisible by num_groups")
    groups = mx.reshape(array, (int(array.shape[0]) * num_groups, -1))
    threads = 512 if int(groups.shape[1]) >= 512 else 32
    mean, reciprocal = source_kernel("norm_kernel", mx)(
        inputs=[groups, mx.array([eps], dtype=mx.float32), _tables(mx)[1]],
        template=[("TGS", threads)],
        output_shapes=[(int(groups.shape[0]),)] * 2,
        output_dtypes=[mx.float32] * 2,
        grid=(int(groups.shape[0]) * threads, 1, 1),
        threadgroup=(threads, 1, 1),
    )
    return _apply(
        "norm_affine",
        [array, mean, reciprocal, weight, bias],
        tuple(array.shape),
        mx,
        template=[("G", num_groups)],
    )


def _exp(array: Array, mx: Mlx, *, as_silu: bool = False) -> Array:
    """Apply the source-ordered FP32 exponential operation."""
    return _apply(
        "math_kernel",
        [array, _tables(mx)[0]],
        tuple(array.shape),
        mx,
        template=[("X_SIZE", int(array.size)), ("SILU", as_silu)],
    )


def silu(array: Array, mx: Mlx) -> Array:
    """Torch SiLU divides x by 1+exp(-x)."""
    return _exp(array, mx, as_silu=True)


def swish(array: Array, mx: Mlx) -> Array:
    """MONAI Swish multiplies x by the separately rounded sigmoid."""
    return _apply(
        "math_swish",
        [array, _tables(mx)[0]],
        tuple(array.shape),
        mx,
        template=[("N", int(array.size))],
    )


def linear(array: Array, weight: Array, bias: Array | None, mx: Mlx) -> Array:
    """Image embeddings and strided QKV products with source reduction order."""
    rows = int(array.size) // int(array.shape[-1])
    cin, cout = int(weight.shape[1]), int(weight.shape[0])
    shape = (*tuple(array.shape[:-1]), cout)
    inputs = [array, weight, _bias(weight, bias, mx)]
    if int(array.ndim) > 2:
        name = "ops_qkv"
        if int(array.shape[0]) == 1 and cin == 512:
            name = "ops_qkvs" if int(array.shape[-2]) == 512 else "ops_qkv4"
        return _apply(name, inputs, shape, mx, template=[("ROWS", rows)])
    if rows == 1 and (cout, cin) == (128, 512):
        return _apply("ops_dot", inputs, shape, mx, threads=128)
    if rows == 1 and (cout, cin) != (64, 512):
        parts = 4 if cin <= 64 else (16 if cout <= 128 else 32)
        return _apply(
            "ops_gemv",
            inputs,
            shape,
            mx,
            template=[("P", parts), ("SPAN", (cin + parts - 1) // parts)],
            threads=128,
        )
    return _apply("ops_linear", inputs, shape, mx, template=[("ROWS", rows)])


def attention_projection(array: Array, weight: Array, bias: Array, mx: Mlx) -> Array:
    """Contiguous attention output uses a different GEMM from the strided QKV."""
    rows = int(array.size) // int(array.shape[-1])
    name = "ops_projection"
    if rows == 2048 and int(array.shape[-1]) == 256:
        name = "ops_qkv"
    elif rows == 256 and int(array.shape[-1]) == 512:
        name = "ops_sliced"
    return _apply(
        name,
        [array, weight, bias],
        (*tuple(array.shape[:-1]), int(weight.shape[0])),
        mx,
        template=[("ROWS", rows)],
    )


def sinusoidal_timestep_embedding(
    timesteps: Array,
    embedding_dim: int,
    *,
    mx: Mlx,
    max_period: int = 10000,
) -> Array:
    """Source exp/sin/cos on the image scheduler's bounded 0..1000 domain."""
    if int(timesteps.ndim) != 1:
        raise ValueError("Timesteps should be a 1d-array")
    if embedding_dim < 2 or max_period < 1:
        raise ValueError("Source timestep embedding requires dim >= 2 and max_period >= 1")
    if not bool(mx.all(mx.isfinite(timesteps) & (timesteps >= 0) & (timesteps <= 1000)).item()):
        raise ValueError("NV-Generate source timesteps must be finite and in 0..1000")
    half = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32)
    frequencies = _exp(exponent / half, mx)
    phase = timesteps.astype(mx.float32)[:, None] * frequencies[None, :]
    cosine, sine = source_kernel("embed_kernel", mx)(
        inputs=[phase],
        template=[("N", int(phase.size))],
        output_shapes=[tuple(phase.shape)] * 2,
        output_dtypes=[mx.float32] * 2,
        grid=(int(phase.size), 1, 1),
        threadgroup=(256, 1, 1),
    )
    result = mx.concatenate([cosine, sine], axis=-1)
    return mx.pad(result, [(0, 0), (0, 1)]) if embedding_dim % 2 else result


def attention(query: Array, key: Array, value: Array, *, scale: float, mx: Mlx) -> Array:
    """Separate source QK, scaling, softmax and probability/value products."""
    scores = (query @ mx.swapaxes(key, -1, -2)) * scale
    columns = int(scores.shape[-1])
    rows = int(scores.size) // columns
    block = columns > 2048
    threads = 1024 if block else 128
    lanes = 1024 if block else 32
    name = "attn_block" if block else "attn_softmax"
    template: list[tuple[str, int | bool]] = [
        ("COLS", columns),
        ("ITS", (columns + lanes - 1) // lanes),
    ]
    if not block:
        template.append(("ROWS", rows))
    probabilities = source_kernel(name, mx)(
        inputs=[scores, _tables(mx)[0]],
        template=template,
        output_shapes=[tuple(scores.shape)],
        output_dtypes=[mx.float32],
        grid=(rows * lanes, 1, 1),
        threadgroup=(threads, 1, 1),
    )[0]
    return probabilities @ value
