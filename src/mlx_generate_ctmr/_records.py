"""Serialized catalog and runner record shapes."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, Required, TypedDict


class CodePin(TypedDict):
    path: str
    revision: str


class MonaiPin(CodePin):
    distribution_version: str


class WeightPin(TypedDict):
    repo: str
    revision: str
    checkpoints: NotRequired[dict[str, dict[str, str]]]


class SourcePins(TypedDict):
    nv_generate_ctmr: CodePin
    monai: MonaiPin
    weights: WeightPin


class TensorSpec(TypedDict, total=False):
    name: Required[str]
    torch_shape: Required[list[int]]
    layout: Required[str]
    used_at_from_mask_infer: bool
    used_at_mask_generation_infer: bool


class NetworkInventory(TypedDict):
    tensor_count: int
    parameter_count: int
    tensors: list[TensorSpec]


class Catalog(TypedDict):
    destination_dtype: str
    source: NotRequired[SourcePins]
    networks: dict[str, NetworkInventory]
    notes: NotRequired[list[str]]
    object_type: NotRequired[str]


class MaskFacts(TypedDict):
    """Canonical mask shape and spacing (mm, array-axis order), and sorted unique labels."""

    shape: list[int]
    spacing: list[float]
    labels: list[int]


class ImageFacts(TypedDict):
    """Validated image shape and decoded array dtype; neither field records provenance."""

    shape: list[int]
    dtype: str


class PairFacts(TypedDict):
    """Validated image/label pair shape, decoded dtypes, and sorted unique labels."""

    shape: list[int]
    image_dtype: str
    label_dtype: str
    labels: list[int]


type JsonValue = (
    bool | int | float | str | Sequence[JsonValue] | Mapping[str, JsonValue] | SourcePins | None
)
type JsonObject = dict[str, JsonValue]


class RunResult[OutputPath: str | Path](TypedDict):
    outputs: dict[str, OutputPath]
    details: JsonObject


class QcCheckPayload(TypedDict):
    check_id: str
    verdict: str
    reason: str
    details: dict[str, object]


class QcReceiptPayload(TypedDict):
    valid: bool
    requested_anatomy: list[str]
    present_labels: list[int]
    label_counts: dict[str, int]
    passed: list[str]
    failed: list[str]
    unresolved: list[str]
    checks: list[QcCheckPayload]
    limitations: list[str]
