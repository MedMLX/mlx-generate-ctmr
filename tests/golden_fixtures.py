"""Pinned Torch boundary goldens recorded on darwin-arm64 with synthetic weights."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

import pytest
from _contracts import AttentionManifest, BoundaryManifest, ResidualConstructor, SchedulerManifest
from fixture_parameters import boundary_arrays

from mlx_generate_ctmr._arrays import HostArray, NumericArray
from mlx_generate_ctmr.maisi_mlx_common import source_pins

if TYPE_CHECKING:
    import torch

    from mlx_generate_ctmr._types import (
        Array,
        ControlnetConstructor,
        Mlx,
        RflowConstructor,
        VaeConstructor,
    )

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "darwin-arm64" / "maisi"
type Arrays = dict[str, NumericArray]


@overload
def load_boundary(
    boundary: Literal["residual"],
) -> tuple[BoundaryManifest[ResidualConstructor], Arrays]: ...
@overload
def load_boundary(boundary: Literal["attention"]) -> tuple[AttentionManifest, Arrays]: ...
@overload
def load_boundary(boundary: Literal["scheduler"]) -> tuple[SchedulerManifest, Arrays]: ...
@overload
def load_boundary(
    boundary: Literal["diffusion"],
) -> tuple[BoundaryManifest[RflowConstructor], Arrays]: ...
@overload
def load_boundary(
    boundary: Literal["controlnet"],
) -> tuple[BoundaryManifest[ControlnetConstructor], Arrays]: ...
@overload
def load_boundary(boundary: Literal["vae"]) -> tuple[BoundaryManifest[VaeConstructor], Arrays]: ...
def load_boundary(boundary: str) -> tuple[Mapping[str, object], Arrays]:
    """Load one boundary, requiring it to have been recorded against the pinned sources."""
    manifest = json.loads((FIXTURE_DIR / f"{boundary}.json").read_text(encoding="utf-8"))
    pins = cast(Mapping[str, Mapping[str, object]], source_pins())
    for group, values in pins.items():
        for key, value in values.items():
            assert manifest["source"][group][key] == value, f"{boundary}: stale {group}.{key}"
    return manifest, cast(Arrays, boundary_arrays(FIXTURE_DIR, boundary))


def seed_torch(seed: int) -> None:
    """Seed Torch's global generator; its stub leaves ``seed`` untyped."""
    pytest.importorskip("torch").manual_seed(seed)


def mlx_weights(arrays: Mapping[str, HostArray], mx: Mlx) -> dict[str, Array]:
    """Convert recorded ``weight.*`` arrays to native float32 arrays."""
    from medmlx_core.ops import as_fp32

    return {
        name.removeprefix("weight."): as_fp32(array, mx)
        for name, array in arrays.items()
        if name.startswith("weight.")
    }


def torch_weights(arrays: Mapping[str, HostArray]) -> dict[str, torch.Tensor]:
    """Return recorded ``weight.*`` arrays as a Torch state dict."""
    from _torch import from_numpy

    return {
        name.removeprefix("weight."): from_numpy(array)
        for name, array in arrays.items()
        if name.startswith("weight.")
    }
