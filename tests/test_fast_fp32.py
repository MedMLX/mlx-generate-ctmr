"""Fast FP32 NAX convolution variants and fused norm-convolution against float64."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

type Host = NDArray[np.float64]

# Split-FP16 products carry about 2^-22 relative error and FP32 accumulation
# adds reduction-order rounding; scale by the reference magnitude.
_RELATIVE = 2e-6


def _conv_reference(x: Host, w: Host, stride: int) -> Host:
    """Float64 padding-1 3x3x3 convolution of NCDHW ``x`` with Torch-layout ``w``."""
    batch, _, d, h, width = x.shape
    padded = np.pad(x, [(0, 0), (0, 0), (1, 1), (1, 1), (1, 1)])
    out_shape = [(n - 1) // stride + 1 for n in (d, h, width)]
    out = np.zeros((batch, w.shape[0], *out_shape))
    for kd in range(3):
        for kh in range(3):
            for kw in range(3):
                window = padded[
                    :,
                    :,
                    kd : kd + stride * out_shape[0] : stride,
                    kh : kh + stride * out_shape[1] : stride,
                    kw : kw + stride * out_shape[2] : stride,
                ]
                out += np.einsum("bcdhw,oc->bodhw", window, w[:, :, kd, kh, kw])
    return out


def _group_norm_silu(x: Host, weight: Host, bias: Host, groups: int) -> Host:
    """Compute host group normalization and SiLU for a numerical reference."""
    grouped = x.reshape(x.shape[0], groups, -1)
    mean = grouped.mean(-1, keepdims=True)
    var = grouped.var(-1, keepdims=True)
    y = ((grouped - mean) / np.sqrt(var + 1e-6)).reshape(x.shape)
    y = y * weight[None, :, None, None, None] + bias[None, :, None, None, None]
    return y / (1 + np.exp(-y))


def _close(actual: Array, expected: Host) -> None:
    """Compare native outputs with the high-precision reference."""
    host = np.asarray(actual, dtype=np.float64)
    assert host.shape == expected.shape
    assert np.isfinite(host).all()
    np.testing.assert_allclose(host, expected, rtol=0, atol=_RELATIVE * np.abs(expected).max())


@pytest.mark.parametrize(
    ("shape", "out_channels", "stride"),
    [
        ((2, 64, 8, 16, 16), 64, 1),  # large grid: 8-row tiles, no split
        ((2, 512, 4, 8, 8), 512, 1),  # deep 8x8x4 level: 4-row tiles, 9-way split
        ((2, 256, 8, 16, 16), 128, 1),  # 4-row tiles, 3-way split
        ((1, 4, 8, 8, 8), 4, 1),  # padded channels and 16-column output tile
        ((1, 32, 8, 12, 16), 64, 2),
    ],
)
def test_nax_conv_matches_float64(
    metal: Mlx, shape: tuple[int, ...], out_channels: int, stride: int
) -> None:
    """Compare accelerator convolution variants with float64 reference results."""
    from mlx_generate_ctmr import fast_fp32
    from mlx_generate_ctmr._nax_conv import nax_available

    if not nax_available(metal):
        pytest.skip("NAX convolution needs an M5-class GPU")
    rng = np.random.default_rng(3)
    x = rng.standard_normal(shape).astype(np.float32)
    w = (rng.standard_normal((out_channels, shape[1], 3, 3, 3)) / np.sqrt(27 * shape[1])).astype(
        np.float32
    )
    b = rng.standard_normal(out_channels).astype(np.float32)
    y = fast_fp32.conv3d_ncdhw(
        metal.array(x), metal.array(w), metal.array(b), padding=1, stride=stride, mx=metal
    )
    expected = _conv_reference(x.astype(np.float64), w.astype(np.float64), stride)
    _close(y, expected + b[None, :, None, None, None])


@pytest.mark.parametrize("upsample", [False, True])
def test_fused_norm_conv_matches_float64(metal: Mlx, upsample: bool) -> None:
    """Compare fused normalization and convolution with float64 reference results."""
    from mlx_generate_ctmr import fast_fp32

    rng = np.random.default_rng(5)
    batch, channels, out_channels, groups = 2, 64, 32, 8
    x = (3 * rng.standard_normal((batch, channels, 4, 8, 8)) + 1).astype(np.float32)
    w = (rng.standard_normal((out_channels, channels, 3, 3, 3)) / np.sqrt(27 * channels)).astype(
        np.float32
    )
    batch_bias = rng.standard_normal((batch, out_channels)).astype(np.float32)
    norm_w = (1 + 0.1 * rng.standard_normal(channels)).astype(np.float32)
    norm_b = (0.1 * rng.standard_normal(channels)).astype(np.float32)
    scale = 2 if upsample else 1
    residual = rng.standard_normal((batch, out_channels, 4 * scale, 8 * scale, 8 * scale))
    residual = residual.astype(np.float32)
    y = fast_fp32.norm_conv3d_ncdhw(
        metal.array(x),
        metal.array(w),
        metal.array(batch_bias),
        norm=(metal.array(norm_w), metal.array(norm_b)),
        num_groups=groups,
        eps=1e-6,
        apply_silu=True,
        upsample=upsample,
        residual=metal.array(residual),
        mx=metal,
    )
    prepared = _group_norm_silu(
        x.astype(np.float64), norm_w.astype(np.float64), norm_b.astype(np.float64), groups
    )
    if upsample:
        prepared = prepared.repeat(2, 2).repeat(2, 3).repeat(2, 4)
    expected = _conv_reference(prepared, w.astype(np.float64), 1)
    expected += batch_bias[:, :, None, None, None] + residual
    _close(y, expected)


def test_trilinear_conv_matches_materialized_upsampling(metal: Mlx) -> None:
    """The fused VAE upsampling convolution equals the materialized sequence bitwise."""
    from mlx_generate_ctmr import fast_fp32
    from mlx_generate_ctmr._nax_conv import nax_available
    from mlx_generate_ctmr.source_vae import upsample_trilinear_ncdhw

    if not nax_available(metal):
        pytest.skip("NAX convolution needs an M5-class GPU")
    rng = np.random.default_rng(7)
    # Odd, unequal axes exercise the clamped upper neighbours on every axis.
    x = metal.array(rng.standard_normal((2, 24, 3, 5, 6)).astype(np.float32))
    w = metal.array((rng.standard_normal((32, 24, 3, 3, 3)) / np.sqrt(27 * 24)).astype(np.float32))
    b = metal.array(rng.standard_normal(32).astype(np.float32))
    fused = fast_fp32.trilinear_conv3d_ncdhw(x, w, b, mx=metal)
    materialized = fast_fp32.conv3d_ncdhw(
        upsample_trilinear_ncdhw(x, mx=metal), w, b, padding=1, mx=metal
    )
    assert fused.shape == (2, 32, 6, 10, 12)
    np.testing.assert_array_equal(
        np.asarray(fused, dtype=np.float32), np.asarray(materialized, dtype=np.float32)
    )


@pytest.mark.parametrize(
    ("shape", "out_channels", "fused", "trilinear"),
    [
        ((2, 32, 9, 6, 8), 32, True, False),  # norm, SiLU, residual epilogue
        ((1, 16, 5, 3, 4), 16, False, True),  # trilinear packing on a 10-deep grid
        ((1, 256, 6, 4, 4), 256, False, False),  # small grid: tap split and reduction
    ],
)
def test_depth_slabs_match_whole_volume(
    metal: Mlx,
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    out_channels: int,
    fused: bool,
    trilinear: bool,
) -> None:
    """A convolution packed and run in depth slabs equals the whole-volume one bitwise."""
    from medmlx_core.group_norm import group_norm_affine

    from mlx_generate_ctmr import _nax_conv

    if not _nax_conv.nax_available(metal):
        pytest.skip("NAX convolution needs an M5-class GPU")
    rng = np.random.default_rng(11)
    x = metal.array((3 * rng.standard_normal(shape) + 1).astype(np.float32))
    channels = shape[1]
    w = metal.array(
        (rng.standard_normal((out_channels, channels, 3, 3, 3)) / np.sqrt(27 * channels)).astype(
            np.float32
        )
    )
    b = metal.array(rng.standard_normal(out_channels).astype(np.float32))
    packed = _nax_conv.pack_weight(w, metal)
    out_spatial = tuple((2 if trilinear else 1) * n for n in shape[2:])
    affine = None
    residual = None
    if fused:
        norm_w = metal.array((1 + 0.1 * rng.standard_normal(channels)).astype(np.float32))
        norm_b = metal.array((0.1 * rng.standard_normal(channels)).astype(np.float32))
        affine = group_norm_affine(x, norm_w, norm_b, groups=8, eps=1e-6, mx=metal)
        residual = metal.array(
            rng.standard_normal((shape[0], out_channels, *out_spatial)).astype(np.float32)
        )

    def run() -> NDArray[np.float32]:
        """Convolve with the current slab budget."""
        y = _nax_conv.nax_conv3d(
            x,
            packed,
            b,
            affine=affine,
            silu=fused,
            trilinear=trilinear,
            residual=residual,
            mx=metal,
        )
        return np.asarray(y, dtype=np.float32)

    whole = run()
    plane = (out_spatial[1] + 2) * (out_spatial[2] + 2) * (channels + -channels % 16) * 4
    # Two-deep slabs: interior slab edges on every boundary and a short last slab.
    monkeypatch.setattr(_nax_conv, "_SLAB_PACK_BYTES", 4 * plane)
    slabbed = run()
    np.testing.assert_array_equal(slabbed, whole)
