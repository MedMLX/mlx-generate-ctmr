"""Timestep, class, and spacing embeddings for the from-mask UNet/ControlNet."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, MlpArithmetic, Mlx

import math

from medmlx_core import ops


def sinusoidal_timestep_embedding(
    timesteps: Array,
    embedding_dim: int,
    *,
    mx: Mlx,
    max_period: int = 10000,
) -> Array:
    """Sinusoidal timestep embedding, ``(N,)`` timesteps to ``(N, embedding_dim)``.

    Layout is ``[cos, sin]`` along the last axis (as in Ho et al.'s DDPM code),
    with one zero column appended when ``embedding_dim`` is odd. This is the
    plain-MLX version used by the paired-mask UNet and by the CT/MR UNet under
    ``fast_fp32``; ``source_fp32`` has the CUDA-matched version.
    """

    if int(timesteps.ndim) != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    freqs = mx.exp(exponent / half_dim)
    args = mx.reshape(timesteps.astype(mx.float32), (-1, 1)) * mx.reshape(freqs, (1, -1))
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if embedding_dim % 2 == 1:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


def embed_class(indices: Array, weight: Array, mx: Mlx) -> Array:
    """``nn.Embedding`` gather: ``weight[indices]`` with layout ``[num, dim]``."""

    if int(indices.ndim) != 1:
        raise ValueError(f"class indices must be rank-1; got ndim={indices.ndim}")
    return weight[indices]


def silu_mlp(
    array: Array,
    weight1: Array,
    bias1: Array,
    weight2: Array,
    bias2: Array,
    mx: Mlx,
    *,
    arithmetic: MlpArithmetic = ops,
) -> Array:
    """``Linear -> SiLU -> Linear`` used by ``time_embed`` and ``spacing_layer``."""

    hidden = arithmetic.silu(arithmetic.linear(array, weight1, bias1, mx), mx)
    return arithmetic.linear(hidden, weight2, bias2, mx)


def add_class_to_time(time_emb: Array, class_emb: Array) -> Array:
    """``emb = time_embed(t_emb) + class_embedding(class)``."""

    return time_emb + class_emb


def concat_spacing(emb: Array, spacing_emb: Array, mx: Mlx) -> Array:
    """UNet-only ``concat(emb, spacing_layer(spacing))`` on the last axis."""

    return mx.concatenate([emb, spacing_emb], axis=-1)


def concat_top_region_index(emb: Array, region_emb: Array, mx: Mlx) -> Array:
    """``ddpm-ct`` UNet ``concat(emb, top_region_index_layer(top_index))``."""

    return mx.concatenate([emb, region_emb], axis=-1)


def concat_bottom_region_index(emb: Array, region_emb: Array, mx: Mlx) -> Array:
    """``ddpm-ct`` UNet ``concat(emb, bottom_region_index_layer(bottom_index))``."""

    return mx.concatenate([emb, region_emb], axis=-1)
