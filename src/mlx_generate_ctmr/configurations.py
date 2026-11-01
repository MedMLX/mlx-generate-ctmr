"""Typed NV-Generate configurations and checkpoint links. No MLX import."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from importlib.resources import files
from types import MappingProxyType
from typing import cast

from medmlx_core.errors import InvalidInputError


class Configuration(StrEnum):
    """Supported NV-Generate configurations."""

    CT = "ct"
    MR = "mr"
    MR_BRAIN = "mr-brain"


class Network(StrEnum):
    """Networks a configuration's checkpoints can contain."""

    VAE = "vae"
    UNET = "unet"
    CONTROLNET = "controlnet"


@dataclass(frozen=True, slots=True)
class CheckpointPin:
    """One Hugging Face checkpoint pinned to a 40-character lowercase-hex revision."""

    network: Network
    repo: str
    revision: str
    filename: str
    url: str

    def __post_init__(self) -> None:
        """Validate a checkpoint network and its declared revision."""
        if not isinstance(cast(object, self.network), Network):
            raise InvalidInputError(f"unknown checkpoint network {self.network!r}")
        if len(self.revision) != 40:
            raise InvalidInputError("checkpoint revision has the wrong length")
        if any(c not in "0123456789abcdef" for c in self.revision):
            raise InvalidInputError("checkpoint revision must be lowercase hex")
        expected = f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}"
        if self.url != expected:
            raise InvalidInputError(f"checkpoint URL does not match its pin: {self.url}")

    def to_payload(self) -> dict[str, str]:
        """Serialize the checkpoint source and network metadata."""
        return {
            "repo": self.repo,
            "revision": self.revision,
            "filename": self.filename,
            "url": self.url,
        }


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """Defaults and limits for one configuration, loaded from ``data/configurations.json``.

    ``output_size`` is the default output shape in voxels (array axes D, H, W) and
    ``spacing_mm`` the voxel size in mm along those axes. ``default_modality`` and
    ``modalities`` are indices into the 128-row modality embedding;
    ``modality_names`` maps upstream names to indices. ``guidance_scale`` is the
    classifier-free guidance default (0 for CT) and ``num_inference_steps`` the
    rflow step count.
    """

    name: Configuration
    checkpoints: tuple[CheckpointPin, ...]
    output_size: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    default_modality: int
    modalities: frozenset[int]
    guidance_scale: float
    num_inference_steps: int
    modality_names: Mapping[str, int]

    def __post_init__(self) -> None:
        """Validate the networks and settings for a named configuration."""
        expected = (
            {"vae", "unet", "controlnet"} if self.name is Configuration.CT else {"vae", "unet"}
        )
        if {pin.network for pin in self.checkpoints} != expected:
            raise InvalidInputError(f"{self.name} checkpoint links do not match its networks")
        if len(self.checkpoints) != len(expected):
            raise InvalidInputError(f"duplicate checkpoint links for {self.name}")
        if self.default_modality not in self.modalities or self.num_inference_steps != 30:
            raise InvalidInputError(f"invalid defaults for {self.name}")
        if len(self.output_size) != 3 or any(v <= 0 or v % 32 for v in self.output_size):
            raise InvalidInputError(f"invalid output grid for {self.name}")
        if len(self.spacing_mm) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.spacing_mm
        ):
            raise InvalidInputError(f"invalid spacing for {self.name}")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise InvalidInputError(f"invalid guidance scale for {self.name}")
        object.__setattr__(self, "modality_names", MappingProxyType(dict(self.modality_names)))

    @property
    def networks(self) -> tuple[str, ...]:
        """Return network roles in checkpoint order."""
        return tuple(pin.network for pin in self.checkpoints)

    def checkpoint(self, network: str) -> CheckpointPin:
        """Return the checkpoint pin for a requested network."""
        for pin in self.checkpoints:
            if pin.network == network:
                return pin
        raise InvalidInputError(f"{self.name} has no {network} checkpoint")

    def modality_index(self, modality: str | int | None) -> int:
        """Resolve a modality name or index; ``None`` gives the default.

        Raises:
            InvalidInputError: If the modality is unknown or not supported by this
                configuration.
        """
        if modality is None:
            return self.default_modality
        value = self.modality_names.get(modality) if isinstance(modality, str) else modality
        if (
            isinstance(cast(object, value), bool)
            or not isinstance(cast(object, value), int)
            or value not in self.modalities
        ):
            raise InvalidInputError(
                f"modality {modality!r} is unsupported for {self.name}; "
                f"supported indices: {sorted(self.modalities)}"
            )
        return value


@cache
def _load_configurations() -> Mapping[Configuration, ModelConfiguration]:
    """Load and validate the packaged model configuration catalog."""
    payload = json.loads(
        files("mlx_generate_ctmr").joinpath("data", "configurations.json").read_text()
    )
    checkpoint_networks = {
        identity: Network(network)
        for raw in payload["configurations"].values()
        for network, identity in raw["checkpoints"].items()
    }
    pins = {
        identity: CheckpointPin(
            network=checkpoint_networks[identity],
            repo=pin["model"],
            revision=pin["revision"],
            filename=pin["filename"],
            url=pin["url"],
        )
        for identity, pin in payload["checkpoints"].items()
    }
    configurations = {
        Configuration(name): ModelConfiguration(
            name=Configuration(name),
            checkpoints=tuple(pins[identity] for identity in raw["checkpoints"].values()),
            output_size=tuple(raw["output_size"]),
            spacing_mm=tuple(raw["spacing_mm"]),
            default_modality=raw["modality"],
            modalities=frozenset(raw["modalities"]),
            guidance_scale=raw["guidance_scale"],
            num_inference_steps=raw["num_inference_steps"],
            modality_names=payload["modality_indices"],
        )
        for name, raw in payload["configurations"].items()
    }
    if set(configurations) != set(Configuration):
        raise InvalidInputError("NV-Generate configuration resource is incomplete")
    return MappingProxyType(configurations)


def get_configuration(configuration: str = "ct") -> ModelConfiguration:
    """Return the configuration named ``"ct"``, ``"mr"`` or ``"mr-brain"``.

    Raises:
        InvalidInputError: If ``configuration`` is not one of those names.
    """
    if not isinstance(cast(object, configuration), str):
        raise InvalidInputError("configuration must be 'ct', 'mr', or 'mr-brain'")
    try:
        name = Configuration(configuration)
    except ValueError as exc:
        raise InvalidInputError("configuration must be 'ct', 'mr', or 'mr-brain'") from exc
    return _load_configurations()[name]
