"""MLX ports of MAISI self-attention and mask-UNet spatial transformer blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import (
        Activations,
        Array,
        AttentionArithmetic,
        Mlx,
        UnetArithmetic,
    )

from collections.abc import Callable, Mapping

from medmlx_core import ops
from medmlx_core.layout import (
    require_ncdhw,
    tokens_from_ncdhw,
    tokens_to_ncdhw,
)
from medmlx_core.ops import linear

from mlx_generate_ctmr import fast_fp32
from mlx_generate_ctmr.arithmetic_mode import active_arithmetic_mode, unet_arithmetic
from mlx_generate_ctmr.attention_budget import query_chunk_size

# Attention here is dense: it builds a (tokens x tokens) score matrix per head.
# Inputs above 32**3 tokens are refused before allocating, and queries are
# evaluated 1024 at a time once the token count reaches the trigger so peak
# memory stays bounded.
_MAX_SELF_ATTENTION_TOKENS = 32 * 32 * 32
_QUERY_CHUNK_TRIGGER_TOKENS = 32 * 32 * 32
_QUERY_CHUNK_TOKENS = 1024
_SELF_WEIGHT_KEYS = (
    "norm.weight",
    "norm.bias",
    "attn.to_q.weight",
    "attn.to_q.bias",
    "attn.to_k.weight",
    "attn.to_k.bias",
    "attn.to_v.weight",
    "attn.to_v.bias",
)
# ``SABlock.out_proj`` is an ``nn.Identity`` (no parameters) when the source
# constructs it with ``include_fc=False`` (``ddpm-ct``); only require it when
# the forward pass will actually apply it.
_SELF_FC_WEIGHT_KEYS = (
    "attn.out_proj.weight",
    "attn.out_proj.bias",
)
_TRANSFORMER_WEIGHT_KEYS = (
    "norm.weight",
    "norm.bias",
    "proj_in.conv.weight",
    "proj_in.conv.bias",
    "transformer_blocks.0.norm1.weight",
    "transformer_blocks.0.norm1.bias",
    "transformer_blocks.0.norm2.weight",
    "transformer_blocks.0.norm2.bias",
    "transformer_blocks.0.norm3.weight",
    "transformer_blocks.0.norm3.bias",
    "transformer_blocks.0.attn1.to_q.weight",
    "transformer_blocks.0.attn1.to_k.weight",
    "transformer_blocks.0.attn1.to_v.weight",
    "transformer_blocks.0.attn1.out_proj.weight",
    "transformer_blocks.0.attn1.out_proj.bias",
    "transformer_blocks.0.attn2.to_q.weight",
    "transformer_blocks.0.attn2.to_k.weight",
    "transformer_blocks.0.attn2.to_v.weight",
    "transformer_blocks.0.attn2.out_proj.weight",
    "transformer_blocks.0.attn2.out_proj.bias",
    "transformer_blocks.0.ff.linear1.weight",
    "transformer_blocks.0.ff.linear1.bias",
    "transformer_blocks.0.ff.linear2.weight",
    "transformer_blocks.0.ff.linear2.bias",
    "proj_out.conv.weight",
    "proj_out.conv.bias",
)
_NO_BIAS_KEYS = (
    "transformer_blocks.0.attn1.to_q.bias",
    "transformer_blocks.0.attn1.to_k.bias",
    "transformer_blocks.0.attn1.to_v.bias",
    "transformer_blocks.0.attn2.to_q.bias",
    "transformer_blocks.0.attn2.to_k.bias",
    "transformer_blocks.0.attn2.to_v.bias",
)


def spatial_attention_ncdhw(
    x: Array,
    weights: Mapping[str, Array],
    *,
    num_groups: int,
    eps: float,
    num_head_channels: int,
    include_fc: bool,
    mx: Mlx,
    arithmetic: UnetArithmetic | None = None,
) -> Array:
    """Residual self-attention over all spatial positions of an NCDHW tensor.

    Steps: GroupNorm, flatten to tokens, separate Q/K/V projections, scaled
    dot-product attention, optional ``out_proj``, reshape back, add the input.
    ``x`` is ``(N, C, D, H, W)`` float32 and the result has the same shape.
    ``weights`` uses the block-relative MONAI names (``norm.*``, ``attn.to_q.*``,
    ``attn.to_k.*``, ``attn.to_v.*``, and ``attn.out_proj.*`` when ``include_fc``).
    ``num_head_channels`` is the channel count per head (``C`` must be divisible by it).
    GroupNorm and the linear/attention products use ``arithmetic``, by default
    the active ``arithmetic_mode``'s: the CUDA-matched Metal kernels in
    ``source_fp32``, or ``fast_fp32``'s native group norm and FP32 flash
    attention, which needs no query chunking.

    Raises:
        ValueError: If ``x`` is not NCDHW, has more than 32768 spatial tokens, is
            missing weights, or ``C`` is not divisible by ``num_head_channels``.
    """

    require_ncdhw(x, name="attention input")
    _require_token_budget(x)
    required = _SELF_WEIGHT_KEYS + (_SELF_FC_WEIGHT_KEYS if include_fc else ())
    missing = [key for key in required if key not in weights]
    if missing:
        raise ValueError("attention weights missing: " + ", ".join(missing))
    channels = int(x.shape[1])
    heads, scale = _heads_and_scale(channels, num_head_channels)
    spatial = _spatial_shape(x)
    if arithmetic is None:
        arithmetic = unet_arithmetic()

    hidden = arithmetic.group_norm_ncdhw(
        x,
        weights["norm.weight"],
        weights["norm.bias"],
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    tokens = tokens_from_ncdhw(hidden, mx)
    if arithmetic is fast_fp32:
        merged = fast_fp32.self_attention_tokens(
            tokens, weights, heads=heads, scale=scale, include_fc=include_fc, mx=mx
        )
        return x + tokens_to_ncdhw(merged, spatial, mx)
    merged = _attention_tokens(
        tokens,
        tokens,
        weights,
        prefix="attn",
        heads=heads,
        scale=scale,
        qkv_bias=True,
        include_fc=include_fc,
        allow_query_chunking=True,
        arithmetic=arithmetic,
        mx=mx,
    )
    return x + tokens_to_ncdhw(merged, spatial, mx)


def spatial_transformer_ncdhw(
    x: Array,
    context: Array | None,
    weights: Mapping[str, Array],
    *,
    num_groups: int,
    eps: float,
    num_head_channels: int,
    cross_attention_dim: int,
    include_fc: bool,
    mx: Mlx,
) -> Array:
    """Mask-UNet spatial transformer block: self-attention, cross-attention, GEGLU.

    ``x`` is ``(N, C, D, H, W)``; ``context`` is ``(B, T, cross_attention_dim)``
    or ``None``, in which case the cross-attention step attends to its own input.
    The result has the shape of ``x``. Q/K/V projections have no bias and a single
    transformer layer is expected (``transformer_blocks.0.*`` weights). Uses plain
    ``medmlx_core`` ops rather than the CUDA-matched kernels; under the default
    fast ``arithmetic_mode`` the group norm, pointwise projections and
    self-attention use ``fast_fp32`` instead, and cross-attention to a single
    context token reduces to its projected value.

    Raises:
        ValueError: If shapes or weights do not match, or ``x`` has more than 32768
            spatial tokens.
    """

    require_ncdhw(x, name="transformer input")
    _require_token_budget(x)
    missing = [key for key in _TRANSFORMER_WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError("transformer weights missing: " + ", ".join(missing))
    extra_bias = [key for key in _NO_BIAS_KEYS if key in weights]
    if extra_bias:
        raise ValueError("transformer expects qkv_bias=false; found " + ", ".join(extra_bias))
    if context is not None:
        if int(context.ndim) != 3:
            raise ValueError(f"context must be [B, T, C]; got ndim={context.ndim}")
        if int(context.shape[-1]) != cross_attention_dim:
            raise ValueError(
                "context last dim "
                f"{int(context.shape[-1])} != cross_attention_dim {cross_attention_dim}"
            )
    channels = int(x.shape[1])
    heads, scale = _heads_and_scale(channels, num_head_channels)

    fast = active_arithmetic_mode() == "fast"
    arithmetic = fast_fp32 if fast else ops
    hidden = arithmetic.group_norm_ncdhw(
        x,
        weights["norm.weight"],
        weights["norm.bias"],
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.conv3d_ncdhw(
        hidden,
        weights["proj_in.conv.weight"],
        weights["proj_in.conv.bias"],
        padding=0,
        mx=mx,
    )
    spatial = _spatial_shape(hidden)
    tokens = tokens_from_ncdhw(hidden, mx)
    self_input = _layer_norm(
        tokens,
        weights["transformer_blocks.0.norm1.weight"],
        weights["transformer_blocks.0.norm1.bias"],
        mx=mx,
    )
    if fast:
        attended = fast_fp32.self_attention_tokens(
            self_input,
            weights,
            heads=heads,
            scale=scale,
            include_fc=include_fc,
            mx=mx,
            prefix="transformer_blocks.0.attn1",
            qkv_bias=False,
        )
    else:
        attended = _attention_tokens(
            self_input,
            self_input,
            weights,
            prefix="transformer_blocks.0.attn1",
            heads=heads,
            scale=scale,
            qkv_bias=False,
            include_fc=include_fc,
            allow_query_chunking=True,
            mx=mx,
        )
    tokens = attended + tokens

    if fast and context is not None and int(context.shape[1]) == 1:
        # A single key gets softmax weight exactly 1, so every query receives the
        # projected value; skip the query/key projections and broadcast it.
        value = linear(context, weights["transformer_blocks.0.attn2.to_v.weight"], None, mx)
        tokens = (
            linear(
                value,
                weights["transformer_blocks.0.attn2.out_proj.weight"],
                weights["transformer_blocks.0.attn2.out_proj.bias"],
                mx,
            )
            + tokens
        )
    else:
        cross_input = _layer_norm(
            tokens,
            weights["transformer_blocks.0.norm2.weight"],
            weights["transformer_blocks.0.norm2.bias"],
            mx=mx,
        )
        tokens = (
            _attention_tokens(
                cross_input,
                context if context is not None else cross_input,
                weights,
                prefix="transformer_blocks.0.attn2",
                heads=heads,
                scale=scale,
                qkv_bias=False,
                include_fc=True,
                allow_query_chunking=False,
                mx=mx,
            )
            + tokens
        )

    ff_input = _layer_norm(
        tokens,
        weights["transformer_blocks.0.norm3.weight"],
        weights["transformer_blocks.0.norm3.bias"],
        mx=mx,
    )
    tokens = (
        geglu(
            ff_input,
            weights["transformer_blocks.0.ff.linear1.weight"],
            weights["transformer_blocks.0.ff.linear1.bias"],
            weights["transformer_blocks.0.ff.linear2.weight"],
            weights["transformer_blocks.0.ff.linear2.bias"],
            mx=mx,
        )
        + tokens
    )
    hidden = tokens_to_ncdhw(tokens, spatial, mx)
    hidden = arithmetic.conv3d_ncdhw(
        hidden,
        weights["proj_out.conv.weight"],
        weights["proj_out.conv.bias"],
        padding=0,
        mx=mx,
    )
    return x + hidden


def geglu(
    array: Array,
    weight1: Array,
    bias1: Array,
    weight2: Array,
    bias2: Array,
    *,
    mx: Mlx,
) -> Array:
    """GEGLU MLP used by the mask-UNet transformer feed-forward block."""

    from importlib import import_module

    activations = cast("Activations", import_module("mlx.nn"))
    projected = linear(array, weight1, bias1, mx)
    channels = int(projected.shape[-1])
    if channels % 2 != 0:
        raise ValueError(f"GEGLU projected width must be even; got {channels}")
    half = channels // 2
    value = projected[..., :half]
    gate = projected[..., half:]
    return linear(value * activations.gelu(gate), weight2, bias2, mx)


def _attention_tokens(
    query_tokens: Array,
    key_value_tokens: Array,
    weights: Mapping[str, Array],
    *,
    prefix: str,
    heads: int,
    scale: float,
    qkv_bias: bool,
    include_fc: bool,
    allow_query_chunking: bool,
    mx: Mlx,
    arithmetic: AttentionArithmetic | None = None,
) -> Array:
    """Project query, key, and value tokens and combine their attention outputs."""
    project = linear if arithmetic is None else arithmetic.linear
    output_project = linear if arithmetic is None else arithmetic.attention_projection
    attention_op = None if arithmetic is None else arithmetic.attention
    query = _heads(
        project(
            query_tokens,
            weights[f"{prefix}.to_q.weight"],
            _bias(weights, f"{prefix}.to_q", qkv_bias=qkv_bias),
            mx,
        ),
        heads,
        mx,
    )
    key = _heads(
        project(
            key_value_tokens,
            weights[f"{prefix}.to_k.weight"],
            _bias(weights, f"{prefix}.to_k", qkv_bias=qkv_bias),
            mx,
        ),
        heads,
        mx,
    )
    value = _heads(
        project(
            key_value_tokens,
            weights[f"{prefix}.to_v.weight"],
            _bias(weights, f"{prefix}.to_v", qkv_bias=qkv_bias),
            mx,
        ),
        heads,
        mx,
    )
    attended = _scaled_dot_product_attention(
        query,
        key,
        value,
        scale=scale,
        allow_query_chunking=allow_query_chunking,
        attention_op=attention_op,
        mx=mx,
    )
    merged = _merge_heads(attended, mx)
    if include_fc:
        merged = output_project(
            merged,
            weights[f"{prefix}.out_proj.weight"],
            weights[f"{prefix}.out_proj.bias"],
            mx,
        )
    return merged


def _bias(weights: Mapping[str, Array], prefix: str, *, qkv_bias: bool) -> Array | None:
    """Return the named projection bias when the block uses one."""
    key = f"{prefix}.bias"
    if not qkv_bias:
        return None
    if key not in weights:
        raise ValueError(f"attention weights missing: {key}")
    return weights[key]


def _layer_norm(array: Array, weight: Array, bias: Array, *, mx: Mlx) -> Array:
    # Pinned transformer LayerNorm uses Torch's default, independent of GroupNorm.
    """Apply the source transformer layer norm with epsilon 1e-5."""
    return mx.fast.layer_norm(array, eps=1e-5, weight=weight, bias=bias)


def _scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    scale: float,
    allow_query_chunking: bool,
    mx: Mlx,
    attention_op: Callable[..., Array] | None = None,
) -> Array:
    # An explicit request budget also covers cross-attention; the legacy flag
    # controls only the default self-attention token-threshold policy.
    """Apply scaled attention, chunking queries when the budget requires it."""
    chunk_size = query_chunk_size(
        int(query.shape[0]),
        int(query.shape[1]),
        int(query.shape[2]),
        int(key.shape[2]),
        int(query.itemsize),
    )
    if chunk_size is None and allow_query_chunking:
        chunk_size = (
            _QUERY_CHUNK_TOKENS
            if int(query.shape[2]) >= _QUERY_CHUNK_TRIGGER_TOKENS
            else int(query.shape[2])
        )
    if chunk_size is not None and chunk_size < int(query.shape[2]):
        return _query_chunked_scaled_dot_product_attention(
            query,
            key,
            value,
            scale=scale,
            mx=mx,
            attention_op=attention_op,
            chunk_size=chunk_size,
        )
    apply = _sdpa if attention_op is None else attention_op
    return apply(query, key, value, scale=scale, mx=mx)


def _query_chunked_scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    scale: float,
    mx: Mlx,
    attention_op: Callable[..., Array] | None = None,
    chunk_size: int = _QUERY_CHUNK_TOKENS,
) -> Array:
    # Materialize Q, K, and V once so each chunked SDPA call does real work
    # instead of extending one large lazy graph across all query slices.
    """Evaluate attention over bounded query chunks and concatenate the outputs."""
    apply = _sdpa if attention_op is None else attention_op
    mx.eval(query, key, value)
    outputs: list[Array] = []
    total_tokens = int(query.shape[2])
    for start in range(0, total_tokens, chunk_size):
        end = min(start + chunk_size, total_tokens)
        attended = apply(query[:, :, start:end, :], key, value, scale=scale, mx=mx)
        mx.eval(attended)
        outputs.append(attended)
    if len(outputs) == 1:
        return outputs[0]
    return mx.concatenate(outputs, axis=2)


def _sdpa(query: Array, key: Array, value: Array, *, scale: float, mx: Mlx) -> Array:
    """Dispatch scaled dot-product attention to the native runtime."""
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale)


def _heads(projected: Array, heads: int, mx: Mlx) -> Array:
    """Reshape projected tokens into batch, head, token, and head-channel axes."""
    batch, tokens, channels = (int(dim) for dim in projected.shape)
    dim_head = channels // heads
    return mx.transpose(mx.reshape(projected, (batch, tokens, heads, dim_head)), (0, 2, 1, 3))


def _merge_heads(heads: Array, mx: Mlx) -> Array:
    """Combine attention heads into the final token-channel axis."""
    batch, n_heads, tokens, dim_head = (int(dim) for dim in heads.shape)
    return mx.reshape(mx.transpose(heads, (0, 2, 1, 3)), (batch, tokens, n_heads * dim_head))


def _heads_and_scale(channels: int, num_head_channels: int) -> tuple[int, float]:
    """Validate head width and derive head count and attention scale."""
    if channels % num_head_channels != 0:
        raise ValueError("num_channels must be divisible by num_head_channels")
    heads = channels // num_head_channels
    return heads, float(num_head_channels) ** -0.5


def _require_token_budget(x: Array) -> None:
    """Reject a spatial grid exceeding the transformer token limit."""
    tokens = int(x.shape[2]) * int(x.shape[3]) * int(x.shape[4])
    if tokens > _MAX_SELF_ATTENTION_TOKENS:
        raise ValueError(f"self-attention supports at most 32768 spatial tokens; got {tokens}")


def _spatial_shape(x: Array) -> tuple[int, int, int, int, int]:
    """Return the five dimensions of an NCDHW array."""
    return (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3]), int(x.shape[4]))
