"""MLX port of the MAISI image VAE (``AutoencoderKlMaisi``): decode and encode.

Tensors are NCDHW float32 and weights keep their Torch layouts. The production
decoder upsamples each spatial axis by 4. No MLX import at module load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx, VaeArithmetic, VaeConstructor

from collections.abc import Mapping

from medmlx_core import ops
from medmlx_core.layout import require_ncdhw
from medmlx_core.ops import (
    group_norm_ncdhw,
    pad_spatial_trailing_ncdhw,
    split_conv3d_ncdhw,
    split_conv_transpose3d_ncdhw,
)

from mlx_generate_ctmr import fast_fp32
from mlx_generate_ctmr.unet import bool_sequence, int_sequence


def decode_vae_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
    arithmetic: VaeArithmetic = ops,
) -> tuple[Array, Array]:
    """Decode a latent: ``post_quant_conv`` then the decoder.

    ``latent`` is ``(N, latent_channels, D, H, W)`` and must already be divided by
    the autoencoder scale factor. Returns ``(post_quant, decoded)`` where ``decoded``
    has ``out_channels`` channels and spatial axes multiplied by
    ``2 ** (len(num_channels) - 1)`` (4 for the production constructor).
    ``arithmetic`` selects the primitives
    (``medmlx_core.ops`` by default, ``source_vae`` for the CUDA-matched path).
    """

    require_ncdhw(latent, name="vae latent")
    _refuse_unsupported(constructor)
    post_quant = _conv(
        latent,
        weights,
        "post_quant_conv",
        padding=0,
        mx=mx,
        num_splits=1,
        dim_split=1,
        arithmetic=arithmetic,
    )
    decoded = run_vae_decoder_ncdhw(
        post_quant, weights, constructor=constructor, mx=mx, arithmetic=arithmetic
    )
    return post_quant, decoded


def encode_vae_ncdhw(
    image: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
) -> tuple[Array, Array]:
    """Encode an image: encoder then ``quant_conv_{mu,log_sigma}``.

    ``image`` is ``(N, in_channels, D, H, W)``. Returns ``(z_mu, z_sigma)``, each
    ``(N, latent_channels, D/s, H/s, W/s)`` for axes divisible by
    ``s = 2 ** (len(num_channels) - 1)`` (4 for production), with
    ``z_sigma = exp(log_var / 2)``
    after clamping ``log_var`` to [-30, 20]. Partitioned encoders require at least
    three final latent voxels per split along ``dim_split``; smaller chunks cannot
    retain the pinned splitter's convolution overlap and raise ``ValueError``.
    """

    require_ncdhw(image, name="vae image")
    _refuse_unsupported(constructor)
    hidden = _run_encoder(image, weights, constructor=constructor, mx=mx)
    z_mu = _conv(hidden, weights, "quant_conv_mu", padding=0, mx=mx, num_splits=1, dim_split=1)
    z_log_var = _conv(
        hidden,
        weights,
        "quant_conv_log_sigma",
        padding=0,
        mx=mx,
        num_splits=1,
        dim_split=1,
    )
    z_log_var = mx.clip(z_log_var, -30.0, 20.0)
    z_sigma = mx.exp(z_log_var / 2.0)
    return z_mu, z_sigma


def vae_resblock_ncdhw(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    num_groups: int,
    eps: float,
    mx: Mlx,
    num_splits: int,
    dim_split: int,
    arithmetic: VaeArithmetic = ops,
) -> Array:
    """MaisiResBlock: GN -> SiLU -> conv -> GN -> SiLU -> conv -> +skip.

    ``fast_fp32`` runs its fused block on the whole volume, without splits.
    """

    if arithmetic is fast_fp32:
        shortcut = f"{prefix}.nin_shortcut"
        has_shortcut = f"{shortcut}.conv.conv.weight" in weights
        return fast_fp32.vae_resblock_ncdhw(
            array,
            _conv_weights(weights, f"{prefix}.conv1"),
            _conv_weights(weights, f"{prefix}.conv2"),
            (_need(weights, f"{prefix}.norm1.weight"), _need(weights, f"{prefix}.norm1.bias")),
            (_need(weights, f"{prefix}.norm2.weight"), _need(weights, f"{prefix}.norm2.bias")),
            _conv_weights(weights, shortcut) if has_shortcut else None,
            num_groups=num_groups,
            eps=eps,
            mx=mx,
        )
    hidden = arithmetic.group_norm_ncdhw(
        array,
        _need(weights, f"{prefix}.norm1.weight"),
        _need(weights, f"{prefix}.norm1.bias"),
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.silu(hidden, mx)
    hidden = _conv(
        hidden,
        weights,
        f"{prefix}.conv1",
        padding=1,
        mx=mx,
        num_splits=num_splits,
        dim_split=dim_split,
        arithmetic=arithmetic,
    )
    hidden = arithmetic.group_norm_ncdhw(
        hidden,
        _need(weights, f"{prefix}.norm2.weight"),
        _need(weights, f"{prefix}.norm2.bias"),
        num_groups=num_groups,
        eps=eps,
        mx=mx,
    )
    hidden = arithmetic.silu(hidden, mx)
    hidden = _conv(
        hidden,
        weights,
        f"{prefix}.conv2",
        padding=1,
        mx=mx,
        num_splits=num_splits,
        dim_split=dim_split,
        arithmetic=arithmetic,
    )
    skip = array
    shortcut = f"{prefix}.nin_shortcut.conv.conv.weight"
    if shortcut in weights:
        skip = _conv(
            array,
            weights,
            f"{prefix}.nin_shortcut",
            padding=0,
            mx=mx,
            num_splits=num_splits,
            dim_split=dim_split,
            arithmetic=arithmetic,
        )
    return skip + hidden


def vae_upsample_ncdhw(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    mx: Mlx,
    num_splits: int,
    dim_split: int,
    use_convtranspose: bool = False,
    arithmetic: VaeArithmetic = ops,
) -> Array:
    """MaisiUpsample: either ConvTranspose3d or trilinear x2 then k=3 conv."""

    if use_convtranspose:
        return split_conv_transpose3d_ncdhw(
            array,
            _need(weights, f"{prefix}.conv.conv.conv.weight"),
            _need(weights, f"{prefix}.conv.conv.conv.bias"),
            padding=1,
            output_padding=1,
            stride=2,
            num_splits=num_splits,
            dim_split=dim_split,
            mx=mx,
        )

    upsampled = arithmetic.upsample_trilinear_ncdhw(array, mx=mx)
    return split_conv3d_ncdhw(
        upsampled,
        _need(weights, f"{prefix}.conv.conv.conv.weight"),
        _need(weights, f"{prefix}.conv.conv.conv.bias"),
        padding=1,
        stride=1,
        num_splits=num_splits,
        dim_split=dim_split,
        mx=mx,
        convolution=arithmetic.conv3d_ncdhw,
    )


def vae_downsample_ncdhw(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    mx: Mlx,
    num_splits: int,
    dim_split: int,
) -> Array:
    """Downsample with trailing padding inside each aligned convolution chunk."""

    if (
        num_splits > 1
        and dim_split in (0, 1, 2)
        and (int(array.shape[dim_split + 2]) // num_splits) % 2
    ):
        raise ValueError("VAE downsampling requires an even input length per split")

    def padded_convolution(
        chunk: Array,
        weight: Array,
        bias: Array | None,
        *,
        padding: int,
        stride: int,
        mx: Mlx,
    ) -> Array:
        """Convolve the padded VAE input before downsampling."""
        return ops.conv3d_ncdhw(
            pad_spatial_trailing_ncdhw(chunk, mx),
            weight,
            bias,
            padding=padding,
            stride=stride,
            mx=mx,
        )

    # Split the divisible, unpadded domain. Interior padded ends lie in discarded
    # overlap; only the final chunk retains the global trailing padding.
    return split_conv3d_ncdhw(
        array,
        _need(weights, f"{prefix}.conv.conv.conv.weight"),
        _need(weights, f"{prefix}.conv.conv.conv.bias"),
        padding=0,
        stride=2,
        num_splits=num_splits,
        dim_split=dim_split,
        mx=mx,
        convolution=padded_convolution,
    )


def run_vae_decoder_ncdhw(
    array: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
    arithmetic: VaeArithmetic = ops,
) -> Array:
    """Decode a post-quantization NCDHW latent using the supplied VAE constructor.

    The decoder uses the caller's arithmetic implementation and the checkpoint's
    Torch-layout weights. A constructor requesting decoder attention is rejected.
    """
    channels = int_sequence(constructor["num_channels"], name="num_channels")
    res_blocks = int_sequence(constructor["num_res_blocks"], name="num_res_blocks")
    attn = bool_sequence(constructor["attention_levels"], name="attention_levels")
    groups = int(constructor["norm_num_groups"])
    eps = float(constructor["norm_eps"])
    # Splits bound memory without changing results; the fast path does not need them.
    splits = 1 if arithmetic is fast_fp32 else int(constructor["num_splits"])
    dim_split = int(constructor["dim_split"])
    use_convtranspose = bool(constructor.get("use_convtranspose"))
    hidden = _conv(
        array,
        weights,
        "decoder.blocks.0",
        padding=1,
        mx=mx,
        num_splits=splits,
        dim_split=dim_split,
        arithmetic=arithmetic,
    )
    index = 1
    reversed_channels = list(reversed(channels))
    for level, (_block_out, n_res, has_attn) in enumerate(
        zip(reversed_channels, reversed(res_blocks), reversed(attn), strict=True)
    ):
        if has_attn:
            raise ValueError("image VAE dest has no attention blocks")
        for _ in range(n_res):
            hidden = vae_resblock_ncdhw(
                hidden,
                weights,
                f"decoder.blocks.{index}",
                num_groups=groups,
                eps=eps,
                mx=mx,
                num_splits=splits,
                dim_split=dim_split,
                arithmetic=arithmetic,
            )
            index += 1
        if level != len(reversed_channels) - 1:
            hidden = vae_upsample_ncdhw(
                hidden,
                weights,
                f"decoder.blocks.{index}",
                mx=mx,
                num_splits=splits,
                dim_split=dim_split,
                use_convtranspose=use_convtranspose,
                arithmetic=arithmetic,
            )
            index += 1
    norm = (
        _need(weights, f"decoder.blocks.{index}.weight"),
        _need(weights, f"decoder.blocks.{index}.bias"),
    )
    if arithmetic is fast_fp32:
        return fast_fp32.norm_conv3d_ncdhw(
            hidden,
            *_conv_weights(weights, f"decoder.blocks.{index + 1}"),
            norm=norm,
            num_groups=groups,
            eps=eps,
            apply_silu=False,
            mx=mx,
        )
    hidden = arithmetic.group_norm_ncdhw(
        hidden,
        *norm,
        num_groups=groups,
        eps=eps,
        mx=mx,
    )
    index += 1
    return _conv(
        hidden,
        weights,
        f"decoder.blocks.{index}",
        padding=1,
        mx=mx,
        num_splits=splits,
        dim_split=dim_split,
        arithmetic=arithmetic,
    )


def _run_encoder(
    array: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor,
    mx: Mlx,
) -> Array:
    """Run the VAE encoder on an NCDHW volume and return its final convolution output.

    With `num_splits > 1` convolutions are split along `dim_split`, which needs
    at least three latent voxels per split. Attention
    levels are not supported.
    """
    channels = int_sequence(constructor["num_channels"], name="num_channels")
    res_blocks = int_sequence(constructor["num_res_blocks"], name="num_res_blocks")
    attn = bool_sequence(constructor["attention_levels"], name="attention_levels")
    groups = int(constructor["norm_num_groups"])
    eps = float(constructor["norm_eps"])
    splits = int(constructor["num_splits"])
    dim_split = int(constructor["dim_split"])
    if splits > 1 and dim_split in (0, 1, 2):
        reduction = 1 << (len(channels) - 1)
        latent_length = int(array.shape[dim_split + 2]) // reduction
        # The pinned splitter crops a three-voxel halo even on the final encoder grid.
        if latent_length // splits < 3:
            raise ValueError("VAE encoder splits require at least three latent voxels of overlap")
    hidden = _conv(
        array,
        weights,
        "encoder.blocks.0",
        padding=1,
        mx=mx,
        num_splits=splits,
        dim_split=dim_split,
    )
    index = 1
    for level, (_block_out, n_res, has_attn) in enumerate(
        zip(channels, res_blocks, attn, strict=True)
    ):
        if has_attn:
            raise ValueError("image VAE dest has no attention blocks")
        for _ in range(n_res):
            hidden = vae_resblock_ncdhw(
                hidden,
                weights,
                f"encoder.blocks.{index}",
                num_groups=groups,
                eps=eps,
                mx=mx,
                num_splits=splits,
                dim_split=dim_split,
            )
            index += 1
        if level != len(channels) - 1:
            hidden = vae_downsample_ncdhw(
                hidden,
                weights,
                f"encoder.blocks.{index}",
                mx=mx,
                num_splits=splits,
                dim_split=dim_split,
            )
            index += 1
    hidden = group_norm_ncdhw(
        hidden,
        _need(weights, f"encoder.blocks.{index}.weight"),
        _need(weights, f"encoder.blocks.{index}.bias"),
        num_groups=groups,
        eps=eps,
        mx=mx,
    )
    index += 1
    return _conv(
        hidden,
        weights,
        f"encoder.blocks.{index}",
        padding=1,
        mx=mx,
        num_splits=splits,
        dim_split=dim_split,
    )


def _refuse_unsupported(constructor: VaeConstructor) -> None:
    """Reject VAE options that are outside the supported source contract."""
    if bool(constructor.get("with_encoder_nonlocal_attn")) or bool(
        constructor.get("with_decoder_nonlocal_attn")
    ):
        raise ValueError("image VAE dest has no nonlocal attention")
    if bool(constructor.get("use_flash_attention")):
        raise ValueError("image VAE dest does not use flash attention")


def _conv(
    array: Array,
    weights: Mapping[str, Array],
    prefix: str,
    *,
    padding: int,
    mx: Mlx,
    num_splits: int,
    dim_split: int,
    stride: int = 1,
    arithmetic: VaeArithmetic = ops,
) -> Array:
    """Apply a named VAE convolution with its optional bias."""
    weight, bias = _conv_weights(weights, prefix)
    return split_conv3d_ncdhw(
        array,
        weight,
        bias,
        padding=padding,
        stride=stride,
        num_splits=num_splits,
        dim_split=dim_split,
        mx=mx,
        convolution=arithmetic.conv3d_ncdhw,
    )


def _conv_weights(weights: Mapping[str, Array], prefix: str) -> tuple[Array, Array]:
    """Weight and bias of a MAISI-wrapped (``conv.conv``) or plain (``conv``) convolution."""
    maisi_key = f"{prefix}.conv.conv.weight"
    plain_key = f"{prefix}.conv.weight"
    if maisi_key in weights:
        weight_key, bias_key = maisi_key, f"{prefix}.conv.conv.bias"
    elif plain_key in weights:
        weight_key, bias_key = plain_key, f"{prefix}.conv.bias"
    else:
        raise ValueError(f"VAE weights missing: {maisi_key} or {plain_key}")
    return _need(weights, weight_key), _need(weights, bias_key)


def _need(weights: Mapping[str, Array], key: str) -> Array:
    """Return a required weight or report the missing name."""
    if key not in weights:
        raise ValueError(f"VAE weights missing: {key}")
    return weights[key]
