"""Sliding-window paired-mask decode and canonical label restoration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, HostArray, Mlx, VaeConstructor

import numpy as np
from medmlx_core.layout import require_ncdhw
from medmlx_core.maisi import MAISI_LABELS

from mlx_generate_ctmr.arithmetic_mode import vae_arithmetic
from mlx_generate_ctmr.decode import (
    DEFAULT_MODE,
    DEFAULT_OVERLAP,
    DEFAULT_ROI_SIZE,
    DEFAULT_SIGMA_SCALE,
    sliding_window_decode_ncdhw,
)
from mlx_generate_ctmr.mask_vae import (
    MASK_LOGIT_CHANNELS,
    PRODUCTION_MASK_VAE_CONSTRUCTOR,
)

BODY_LABEL = 200
_VALID_CANONICAL_LABELS = frozenset(item.value for item in MAISI_LABELS)


def load_mask_label_id_map(path: Path) -> HostArray:
    """Load ``label_dict_124_to_132.json`` as a dense ``(125,)`` int32 lookup table.

    Entry ``i`` is the canonical MAISI label id for the mask VAE's class ``i``. The
    JSON maps names to ``[source_class, target_label]`` pairs; every class 0..124
    must be listed and class 124 must map to body label 200. If multiple entries
    use the same source class, the last entry in the JSON object wins.

    Raises:
        ValueError: If the file is malformed, incomplete, or maps to a non-MAISI id.
    """

    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("mask label remap JSON must be an object")
    raw_mapping = cast(Mapping[object, object], raw)
    table = np.arange(MASK_LOGIT_CHANNELS, dtype=np.int32)
    seen: set[int] = set()
    for name, pair in raw_mapping.items():
        if not isinstance(name, str):
            raise ValueError("mask label remap keys must be strings")
        if not isinstance(pair, list):
            raise ValueError(f"mask label remap entry {name!r} must be [orig, target]")
        values = cast(list[object], pair)
        if len(values) != 2:
            raise ValueError(f"mask label remap entry {name!r} must be [orig, target]")
        source, target = values
        if (
            isinstance(source, bool)
            or not isinstance(source, int)
            or isinstance(target, bool)
            or not isinstance(target, int)
        ):
            raise ValueError(f"mask label remap entry {name!r} must be [orig, target]")
        if source < 0 or source >= MASK_LOGIT_CHANNELS:
            raise ValueError(f"mask label remap source {source} is outside 0..124")
        if target not in _VALID_CANONICAL_LABELS:
            raise ValueError(f"mask label remap target {target} is not a canonical MAISI id")
        table[source] = target
        seen.add(source)
    missing = [index for index in range(MASK_LOGIT_CHANNELS) if index not in seen]
    if missing:
        raise ValueError(f"mask label remap is missing source class {missing[0]}")
    if int(table[124]) != BODY_LABEL:
        raise ValueError("mask label remap must map source class 124 to body label 200")
    return table


def sliding_window_decode_mask_logits_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    constructor: VaeConstructor = PRODUCTION_MASK_VAE_CONSTRUCTOR,
    mx: Mlx,
    roi: Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    mode: str = DEFAULT_MODE,
    scale_factor: float | None = None,
) -> Array:
    """Decode mask latents to 125-channel logits with the pinned sliding-window contract."""

    require_ncdhw(latent, name="mask decode latent")
    logits = sliding_window_decode_ncdhw(
        latent,
        weights,
        constructor=constructor,
        mx=mx,
        roi=roi,
        overlap=overlap,
        sigma_scale=sigma_scale,
        mode=mode,
        scale_factor=scale_factor,
        arithmetic=vae_arithmetic(),
    )
    if int(logits.shape[1]) != MASK_LOGIT_CHANNELS:
        raise ValueError(
            f"mask decode logits must have {MASK_LOGIT_CHANNELS} channels; "
            f"got {int(logits.shape[1])}"
        )
    mx.eval(logits)
    return logits


def decode_mask_logits_to_labels_ncdhw(
    logits: Array,
    *,
    label_id_map: Sequence[int] | HostArray,
    mx: Mlx,
) -> Array:
    """Turn mask logits ``(N, 125, D, H, W)`` into canonical label ids ``(N, 1, D, H, W)`` int32.

    Takes the arg-max of float32 softmax probabilities per voxel and maps it
    through ``label_id_map`` (``(125,)``, from ``load_mask_label_id_map``).
    The map is cast to int32 and checked for shape only; callers supplying their
    own map are responsible for its label values and meaning. Logit finiteness
    is not validated here.
    """

    require_ncdhw(logits, name="mask logits")
    if int(logits.shape[1]) != MASK_LOGIT_CHANNELS:
        raise ValueError(
            f"mask logits must have {MASK_LOGIT_CHANNELS} channels; got {int(logits.shape[1])}"
        )
    lookup = np.asarray(label_id_map, dtype=np.int32)
    if lookup.shape != (MASK_LOGIT_CHANNELS,):
        raise ValueError(f"mask label remap must have shape (125,); got {lookup.shape}")
    probs = mx.softmax(logits.astype(mx.float32), axis=1)
    indices = mx.argmax(probs, axis=1, keepdims=True)
    labels = mx.array(lookup, dtype=mx.int32)[indices]
    mx.eval(labels)
    return labels


def decode_mask_latent_to_labels_ncdhw(
    latent: Array,
    weights: Mapping[str, Array],
    *,
    label_id_map: Sequence[int] | HostArray,
    constructor: VaeConstructor = PRODUCTION_MASK_VAE_CONSTRUCTOR,
    mx: Mlx,
    roi: Sequence[int] = DEFAULT_ROI_SIZE,
    overlap: float = DEFAULT_OVERLAP,
    sigma_scale: float = DEFAULT_SIGMA_SCALE,
    mode: str = DEFAULT_MODE,
    scale_factor: float | None = None,
) -> tuple[Array, Array]:
    """Decode mask latents to logits and remapped canonical labels."""

    logits = sliding_window_decode_mask_logits_ncdhw(
        latent,
        weights,
        constructor=constructor,
        mx=mx,
        roi=roi,
        overlap=overlap,
        sigma_scale=sigma_scale,
        mode=mode,
        scale_factor=scale_factor,
    )
    labels = decode_mask_logits_to_labels_ncdhw(
        logits,
        label_id_map=label_id_map,
        mx=mx,
    )
    return logits, labels


__all__ = [
    "BODY_LABEL",
    "decode_mask_latent_to_labels_ncdhw",
    "decode_mask_logits_to_labels_ncdhw",
    "load_mask_label_id_map",
    "sliding_window_decode_mask_logits_ncdhw",
]
