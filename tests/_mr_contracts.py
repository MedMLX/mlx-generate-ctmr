"""Typed metadata records for MR fixture tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import RflowConstructor, VaeConstructor


class Constructors(TypedDict):
    unet: RflowConstructor
    vae: VaeConstructor


class Metadata(TypedDict):
    source_revision: str
    monai_revision: str
    monai_version: str
    precision: str
    modality: int
    guidance_scale: float
    seed: int
    constructor: Constructors
