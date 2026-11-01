"""Fast FP32 arithmetic for the image UNet, ControlNet and VAE decoder.

Implements the same arithmetic surface as ``source_fp32``/``source_vae`` with
native kernels ported from lesiondiffusion-mlx, and adds fused residual and
attention blocks that callers select with ``arithmetic is fast_fp32``:

* 3x3x3 convolutions run as implicit GEMMs on the M5 neural accelerators
  (``_nax_conv``) in its exact split-FP16 mode: FP32 accumulation with about
  ``2^-22`` relative product error, not TF32. A preceding group norm and SiLU
  are applied while packing the input, the residual block's time embedding is
  a per-batch bias, and the skip path is added to the FP32 sum before the single
  FP32 store. A nearest 2x upsampling is folded into the weights. Other GPUs
  and ``MLX_GENERATE_NAX_CONV=0`` use MLX's FP32 ``conv3d`` after a separate
  group norm.
* Group normalization uses a shifted-sum chunk reduction (``_group_norm``).
* Self-attention over 32-wide heads uses the flash kernel in ``_flash_attention``
  without materializing scores, so the attention scratch budget does not apply.
* Pointwise convolutions are FP32 matmuls on the NCDHW layout.

Every operation accumulates in FP32, but neither reduction order nor the CUDA
elementary-function approximations of the source path are reproduced, so
results agree with ``source_fp32`` to FP32 rounding rather than bitwise.
The split-FP16 operands need ``|value| < 65504``; inputs reaching a convolution
through a fused group norm are normalized, and a non-finite overflow in the
others surfaces through the samplers' existing finiteness checks. VAE
convolutions ignore ``num_splits``: splitting only bounds memory and does not
change the convolution result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

import math
import weakref
from collections.abc import Mapping

from medmlx_core import ops
from medmlx_core.group_norm import (
    group_norm_affine,
    native_group_norm,
    native_group_norm_eligible,
)
from medmlx_core.ops import (
    avg_pool3d_ncdhw as avg_pool3d_ncdhw,
)
from medmlx_core.ops import (
    linear as linear,
)
from medmlx_core.ops import (
    silu as silu,
)
from medmlx_core.ops import (
    upsample_nearest_ncdhw as upsample_nearest_ncdhw,
)

from mlx_generate_ctmr._flash_attention import flash_attention, flash_eligible
from mlx_generate_ctmr._nax_conv import nax_available, nax_conv3d, pack_upsample_weight, pack_weight
from mlx_generate_ctmr.embed import (
    sinusoidal_timestep_embedding as sinusoidal_timestep_embedding,
)
from mlx_generate_ctmr.source_vae import (
    upsample_trilinear_ncdhw as upsample_trilinear_ncdhw,
)

# Packed weights keyed by (id(weight), upsample). The weak reference both
# validates that the id still names the same array and evicts on collection.
_packed: dict[tuple[int, bool], tuple[weakref.ref[Array], Array]] = {}


def _pack(weight: Array, *, upsample: bool, mx: Mlx) -> Array:
    """Cache packed convolution weights for the selected upsampling mode."""
    key = (id(weight), upsample)
    hit = _packed.get(key)
    if hit is not None and hit[0]() is weight:
        return hit[1]
    packed = (pack_upsample_weight if upsample else pack_weight)(weight, mx)
    _packed[key] = (weakref.ref(weight, lambda _: _packed.pop(key, None)), packed)
    return packed


def _use_nax(array: Array, weight: Array, mx: Mlx) -> bool:
    """Choose the accelerator convolution path when its constraints are met."""
    return (
        array.dtype == mx.float32
        and tuple(int(n) for n in weight.shape[2:]) == (3, 3, 3)
        and nax_available(mx)
    )


def group_norm_ncdhw(
    array: Array,
    weight: Array,
    bias: Array,
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
    apply_silu: bool = False,
) -> Array:
    """Group norm with FP32 statistics and an optional fused SiLU."""
    if native_group_norm_eligible(array, num_groups):
        return native_group_norm(
            array, weight, bias, groups=num_groups, eps=eps, silu=apply_silu, mx=mx
        )
    hidden = ops.group_norm_ncdhw(array, weight, bias, num_groups=num_groups, eps=eps, mx=mx)
    return silu(hidden, mx) if apply_silu else hidden


def pointwise_conv_ncdhw(array: Array, weight: Array, bias: Array | None, *, mx: Mlx) -> Array:
    """1x1x1 convolution as an FP32 ``(O, C) @ (B, C, DHW)`` matmul without layout copies."""
    batch, channels = int(array.shape[0]), int(array.shape[1])
    spatial = tuple(int(n) for n in array.shape[2:])
    out = mx.reshape(weight, (int(weight.shape[0]), channels)) @ mx.reshape(
        array, (batch, channels, math.prod(spatial))
    )
    if bias is not None:
        out = out + mx.reshape(bias, (-1, 1))
    return mx.reshape(out, (batch, int(weight.shape[0]), *spatial))


def conv3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array | None,
    *,
    padding: int,
    mx: Mlx,
    stride: int = 1,
) -> Array:
    """NCDHW convolution with a Torch ``(O, C, K, K, K)`` weight; see the module notes."""
    kernel = int(weight.shape[2])
    if kernel == 1 and padding == 0 and stride == 1:
        return pointwise_conv_ncdhw(array, weight, bias, mx=mx)
    if padding == 1 and stride in (1, 2) and _use_nax(array, weight, mx):
        return nax_conv3d(
            array,
            _pack(weight, upsample=False, mx=mx),
            _bias(weight, bias, mx),
            stride=stride,
            mx=mx,
        )
    return ops.conv3d_ncdhw(array, weight, bias, padding=padding, stride=stride, mx=mx)


def _bias(weight: Array, bias: Array | None, mx: Mlx) -> Array:
    """Return the bias or a zero bias for the output channels."""
    return mx.zeros((int(weight.shape[0]),), dtype=mx.float32) if bias is None else bias


def norm_conv3d_ncdhw(
    array: Array,
    weight: Array,
    bias: Array,
    *,
    norm: tuple[Array, Array] | None,
    num_groups: int,
    eps: float,
    apply_silu: bool,
    mx: Mlx,
    upsample: bool = False,
    residual: Array | None = None,
) -> Array:
    """``conv(upsample?(silu?(norm?(x)))) + bias (+ residual)`` for padding-1 3x3x3 kernels.

    ``bias`` is ``(O,)`` or a per-batch ``(B, O)``. ``norm`` is the group-norm
    ``(weight, bias)``; ``upsample`` is nearest 2x. On the NAX path the
    normalization, SiLU and upsampling happen while packing/indexing the input
    and bias and residual are added to the FP32 accumulator.
    """
    if _use_nax(array, weight, mx) and (
        norm is None or native_group_norm_eligible(array, num_groups)
    ):
        affine = (
            None
            if norm is None
            else group_norm_affine(array, norm[0], norm[1], groups=num_groups, eps=eps, mx=mx)
        )
        return nax_conv3d(
            array,
            _pack(weight, upsample=upsample, mx=mx),
            bias,
            affine=affine,
            silu=apply_silu,
            upsample=upsample,
            residual=residual,
            mx=mx,
        )
    hidden = array
    if norm is not None:
        hidden = group_norm_ncdhw(
            hidden, norm[0], norm[1], num_groups=num_groups, eps=eps, apply_silu=apply_silu, mx=mx
        )
    elif apply_silu:
        hidden = silu(hidden, mx)
    if upsample:
        hidden = upsample_nearest_ncdhw(hidden, mx=mx)
    out = ops.conv3d_ncdhw(hidden, weight, None, padding=1, mx=mx)
    out = out + mx.reshape(bias, (*tuple(int(n) for n in bias.shape), 1, 1, 1))
    return out if residual is None else residual + out


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
) -> Array:
    """Fused counterpart of ``residual.residual_block_ncdhw`` (validated by the caller).

    The time embedding projection joins conv1's bias per batch item; the skip
    path, after its optional 1x1x1 projection, joins conv2's accumulator.
    Average pooling does not commute with SiLU, so ``down`` blocks normalize
    separately before pooling. The projection runs on the smaller grid: after
    pooling, as in the source, and before nearest upsampling, which yields the
    same values on an eighth of the voxels.
    """
    temb = linear(silu(emb, mx), weights["time_emb_proj.weight"], weights["time_emb_proj.bias"], mx)
    bias1 = mx.reshape(weights["conv1.conv.bias"], (1, -1)) + temb
    norm1 = (weights["norm1.weight"], weights["norm1.bias"])
    project = int(x.shape[1]) != int(weights["conv1.conv.weight"].shape[0])

    def project_skip(skip: Array) -> Array:
        """Project the residual skip when its channels require it."""
        if not project:
            return skip
        return pointwise_conv_ncdhw(
            skip,
            weights["skip_connection.conv.weight"],
            weights["skip_connection.conv.bias"],
            mx=mx,
        )

    if down:
        skip = project_skip(avg_pool3d_ncdhw(x, mx=mx))
        hidden = avg_pool3d_ncdhw(
            group_norm_ncdhw(x, *norm1, num_groups=num_groups, eps=eps, apply_silu=True, mx=mx),
            mx=mx,
        )
        hidden = norm_conv3d_ncdhw(
            hidden,
            weights["conv1.conv.weight"],
            bias1,
            norm=None,
            num_groups=num_groups,
            eps=eps,
            apply_silu=False,
            mx=mx,
        )
    else:
        skip = project_skip(x)
        if up:
            skip = upsample_nearest_ncdhw(skip, mx=mx)
        hidden = norm_conv3d_ncdhw(
            x,
            weights["conv1.conv.weight"],
            bias1,
            norm=norm1,
            num_groups=num_groups,
            eps=eps,
            apply_silu=True,
            upsample=up,
            mx=mx,
        )
    return norm_conv3d_ncdhw(
        hidden,
        weights["conv2.conv.weight"],
        weights["conv2.conv.bias"],
        norm=(weights["norm2.weight"], weights["norm2.bias"]),
        num_groups=num_groups,
        eps=eps,
        apply_silu=True,
        residual=skip,
        mx=mx,
    )


def vae_resblock_ncdhw(
    array: Array,
    conv1: tuple[Array, Array],
    conv2: tuple[Array, Array],
    norm1: tuple[Array, Array],
    norm2: tuple[Array, Array],
    shortcut: tuple[Array, Array] | None,
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
) -> Array:
    """Fused ``MaisiResBlock``: GN-SiLU-conv twice, with the skip in conv2's accumulator."""
    hidden = norm_conv3d_ncdhw(
        array, *conv1, norm=norm1, num_groups=num_groups, eps=eps, apply_silu=True, mx=mx
    )
    skip = array if shortcut is None else conv3d_ncdhw(array, *shortcut, padding=0, mx=mx)
    return norm_conv3d_ncdhw(
        hidden,
        *conv2,
        norm=norm2,
        num_groups=num_groups,
        eps=eps,
        apply_silu=True,
        residual=skip,
        mx=mx,
    )


def self_attention_tokens(
    tokens: Array,
    weights: Mapping[str, Array],
    *,
    heads: int,
    scale: float,
    include_fc: bool,
    mx: Mlx,
    prefix: str = "attn",
    qkv_bias: bool = True,
) -> Array:
    """Q/K/V projections, multi-head self-attention and optional ``out_proj`` on ``(B, L, C)``.

    ``prefix`` names the projection weights (``{prefix}.to_q.weight`` and so on);
    ``qkv_bias=False`` projects without Q/K/V biases.
    """
    query, key, value = (
        linear(
            tokens,
            weights[f"{prefix}.to_{name}.weight"],
            weights[f"{prefix}.to_{name}.bias"] if qkv_bias else None,
            mx,
        )
        for name in ("q", "k", "v")
    )
    if flash_eligible(query, key, heads, mx):
        merged = flash_attention(query, key, value, heads=heads, scale=scale, mx=mx)
    else:
        batch, length, channels = (int(n) for n in query.shape)

        def split(array: Array) -> Array:
            """Split projected tokens into attention heads."""
            return mx.transpose(
                mx.reshape(array, (batch, length, heads, channels // heads)), (0, 2, 1, 3)
            )

        attended = mx.fast.scaled_dot_product_attention(
            split(query), split(key), split(value), scale=scale
        )
        merged = mx.reshape(mx.transpose(attended, (0, 2, 1, 3)), (batch, length, channels))
    if include_fc:
        merged = linear(
            merged, weights[f"{prefix}.out_proj.weight"], weights[f"{prefix}.out_proj.bias"], mx
        )
    return merged


def attention(query: Array, key: Array, value: Array, *, scale: float, mx: Mlx) -> Array:
    """MLX fused attention on ``(B, H, L, D)`` heads (the ``AttentionArithmetic`` surface)."""
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)


def attention_projection(array: Array, weight: Array, bias: Array, mx: Mlx) -> Array:
    """Attention output projection; the same FP32 matmul as ``linear``."""
    return linear(array, weight, bias, mx)
