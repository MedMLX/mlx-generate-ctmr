"""Shared typed interfaces for weight contracts and recorded golden manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from mlx_generate_ctmr._records import SourcePins


class WeightContract(Protocol):
    @property
    def DESTINATION_DTYPE(self) -> str:
        """Return the expected converted tensor dtype name."""
        ...

    @property
    def NETWORKS(self) -> tuple[str, ...]:
        """Return the supported network names."""
        ...

    def classify_torch_layout(self, name: str, torch_shape: Sequence[int]) -> str:
        """Classify a source tensor by name and shape."""
        ...

    def validate_catalog(
        self, catalog: Mapping[str, object], *, require_pinned_inventory: bool = True
    ) -> None:
        """Validate a weight catalog and its optional pinned inventory."""
        ...


class ResidualConstructor(TypedDict):
    norm_num_groups: int
    norm_eps: float


class AttentionConstructor(TypedDict):
    norm_num_groups: int
    norm_eps: float
    num_head_channels: int
    include_fc: bool


class SchedulerConstructor(TypedDict):
    num_train_timesteps: int
    base_img_size_numel: int
    spatial_dim: int
    use_timestep_transform: bool
    use_discrete_timesteps: bool


class BoundaryManifest[Constructor](TypedDict):
    source: SourcePins
    constructor: Constructor


class SchedulerManifest(BoundaryManifest[SchedulerConstructor]):
    num_inference_steps: int
    input_img_size_numel: int


class AttentionManifest(BoundaryManifest[AttentionConstructor]):
    attention_scale: float
