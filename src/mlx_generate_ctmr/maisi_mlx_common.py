"""Shared MAISI source pins and tensor-inventory validation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final, cast

from medmlx_core.errors import InvalidInputError

from mlx_generate_ctmr._records import SourcePins
from mlx_generate_ctmr.configurations import Configuration, get_configuration

LAYOUT_CONV3D_W: Final = "maisi.layout.conv3d_w"
LAYOUT_LINEAR_W: Final = "maisi.layout.linear_w"
LAYOUT_EMBED_W: Final = "maisi.layout.embed_w"
LAYOUT_GN_AFFINE: Final = "maisi.layout.gn_affine"
LAYOUT_VECTOR: Final = "maisi.layout.vector"
IDENTITY_WEIGHT_LAYOUTS: Final[frozenset[str]] = frozenset(
    {LAYOUT_LINEAR_W, LAYOUT_EMBED_W, LAYOUT_GN_AFFINE, LAYOUT_VECTOR}
)
DESTINATION_DTYPE: Final = "float32"
PINNED_NV_GENERATE_REVISION: Final = "5cb04e82fed71f2fe64a2617ab695be1f5a37fea"
PINNED_MONAI_REVISION: Final = "eccefc57550b111ed781d82249dfe77872a0e918"
PINNED_MONAI_VERSION: Final = "1.6.0"
PINNED_WEIGHTS_REPO: Final = get_configuration().checkpoint("unet").repo
PINNED_WEIGHTS_REVISION: Final = get_configuration().checkpoint("unet").revision


def source_pins(configuration: str = "ct") -> SourcePins:
    """Return the upstream code and checkpoint pins recorded in conversion receipts.

    The ``path`` values are provenance labels stored in ``conversion.json`` and
    compared when MR weights are loaded; they need not exist on disk.
    """
    config = get_configuration(configuration)
    pin = config.checkpoint("unet")
    source: SourcePins = {
        "nv_generate_ctmr": {
            "path": "third_party/nv_generate_ctmr/NV-Generate-CTMR",
            "revision": PINNED_NV_GENERATE_REVISION,
        },
        "monai": {
            "path": "third_party/monai/MONAI",
            "revision": PINNED_MONAI_REVISION,
            "distribution_version": PINNED_MONAI_VERSION,
        },
        "weights": {
            "repo": pin.repo,
            "revision": pin.revision,
        },
    }
    if config.name is not Configuration.CT:
        source["weights"]["checkpoints"] = {
            item.network: item.to_payload() for item in config.checkpoints
        }
    return source


def parameter_count_of(shape: Sequence[int]) -> int:
    """Multiply dimensions after integer conversion; an empty shape has one element."""
    return math.prod(int(dim) for dim in shape)


def validate_catalog(
    catalog: Mapping[str, object],
    *,
    tensor_counts: Mapping[str, int],
    parameter_counts: Mapping[str, int],
    inference_flag: str,
    classify_layout: Callable[[str, Sequence[int]], str],
    error_prefix: str,
    require_pinned_inventory: bool,
) -> None:
    """Check catalog network names, tensor layouts, counts and inference flags.

    With ``require_pinned_inventory``, also compare network totals against the
    supplied pinned counts. This checks inventory metadata, not checkpoint bytes
    or scientific validity; mismatches raise ``InvalidInputError``.
    """
    errors: list[str] = []
    for key, expected in (
        ("destination_dtype", DESTINATION_DTYPE),
    ):
        if catalog.get(key) != expected:
            errors.append(key)
    networks = catalog.get("networks")
    network_map = cast(dict[str, object], networks) if isinstance(networks, dict) else None
    if network_map is None or set(network_map) != set(tensor_counts):
        errors.append("networks")
        raise InvalidInputError(error_prefix + ", ".join(errors))
    for name in tensor_counts:
        _validate_network(
            name,
            network_map[name],
            errors,
            inference_flag=inference_flag,
            classify_layout=classify_layout,
            expected_tensors=tensor_counts[name] if require_pinned_inventory else None,
            expected_parameters=parameter_counts[name] if require_pinned_inventory else None,
        )
    if errors:
        raise InvalidInputError(error_prefix + ", ".join(errors))


def _validate_network(
    name: str,
    spec: object,
    errors: list[str],
    *,
    inference_flag: str,
    classify_layout: Callable[[str, Sequence[int]], str],
    expected_tensors: int | None,
    expected_parameters: int | None,
) -> None:
    """Validate one network entry of a conversion manifest, appending problem paths to `errors`.

    Checks tensor names, torch shapes, layout tags (via `classify_layout`), the
    boolean `inference_flag`, tensor and parameter counts, and the inference-flag
    pattern: every tensor is used for inference except in `vae`, where the
    encoder-only tensors must be excluded.
    """
    if not isinstance(spec, dict):
        errors.append(name)
        return
    network = cast(dict[str, object], spec)
    tensors = network.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        errors.append(f"{name}.tensors")
        return
    names: list[str] = []
    params = 0
    infer_true = 0
    tensor_items = cast(list[object], tensors)
    for index, raw in enumerate(tensor_items):
        if not isinstance(raw, dict):
            errors.append(f"{name}.tensors[{index}]")
            continue
        tensor = cast(dict[str, object], raw)
        tensor_name = tensor.get("name")
        torch_shape = tensor.get("torch_shape")
        layout = tensor.get("layout")
        used = tensor.get(inference_flag)
        if not isinstance(tensor_name, str) or not tensor_name:
            errors.append(f"{name}.tensors[{index}].name")
            continue
        if not isinstance(torch_shape, list):
            errors.append(f"{name}.{tensor_name}.torch_shape")
            continue
        raw_shape = cast(list[object], torch_shape)
        if not all(isinstance(dim, int) for dim in raw_shape):
            errors.append(f"{name}.{tensor_name}.torch_shape")
            continue
        shape = cast(list[int], raw_shape)
        try:
            expected_layout = classify_layout(tensor_name, shape)
        except InvalidInputError:
            errors.append(f"{name}.{tensor_name}.layout")
            continue
        if layout != expected_layout:
            errors.append(f"{name}.{tensor_name}.layout")
        if not isinstance(used, bool):
            errors.append(f"{name}.{tensor_name}.{inference_flag}")
        names.append(tensor_name)
        params += parameter_count_of(shape)
        if used is True:
            infer_true += 1
    if len(names) != len(set(names)):
        errors.append(f"{name}.duplicate")
    if network.get("tensor_count") != len(tensor_items):
        errors.append(f"{name}.tensor_count")
    if network.get("parameter_count") != params:
        errors.append(f"{name}.parameter_count")
    if expected_tensors is not None and len(tensor_items) != expected_tensors:
        errors.append(f"{name}.inventory_tensor_count")
    if expected_parameters is not None and params != expected_parameters:
        errors.append(f"{name}.inventory_parameter_count")
    if name == "vae" and infer_true == len(tensor_items):
        errors.append("vae.encoder_convert_only")
    if name != "vae" and infer_true != len(tensor_items):
        errors.append(f"{name}.infer_flags")
