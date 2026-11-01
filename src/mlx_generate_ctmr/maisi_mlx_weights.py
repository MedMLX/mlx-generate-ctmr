"""MAISI from-mask MLX weight catalog contract."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Final, cast

from medmlx_core.errors import InvalidInputError

from mlx_generate_ctmr._records import Catalog
from mlx_generate_ctmr.configurations import Configuration, get_configuration
from mlx_generate_ctmr.maisi_mlx_common import DESTINATION_DTYPE as DESTINATION_DTYPE
from mlx_generate_ctmr.maisi_mlx_common import IDENTITY_WEIGHT_LAYOUTS as IDENTITY_WEIGHT_LAYOUTS
from mlx_generate_ctmr.maisi_mlx_common import LAYOUT_CONV3D_W as LAYOUT_CONV3D_W
from mlx_generate_ctmr.maisi_mlx_common import LAYOUT_EMBED_W as LAYOUT_EMBED_W
from mlx_generate_ctmr.maisi_mlx_common import LAYOUT_GN_AFFINE as LAYOUT_GN_AFFINE
from mlx_generate_ctmr.maisi_mlx_common import LAYOUT_LINEAR_W as LAYOUT_LINEAR_W
from mlx_generate_ctmr.maisi_mlx_common import LAYOUT_VECTOR as LAYOUT_VECTOR
from mlx_generate_ctmr.maisi_mlx_common import PINNED_MONAI_REVISION as PINNED_MONAI_REVISION
from mlx_generate_ctmr.maisi_mlx_common import PINNED_MONAI_VERSION as PINNED_MONAI_VERSION
from mlx_generate_ctmr.maisi_mlx_common import (
    PINNED_NV_GENERATE_REVISION as PINNED_NV_GENERATE_REVISION,
)
from mlx_generate_ctmr.maisi_mlx_common import PINNED_WEIGHTS_REPO as PINNED_WEIGHTS_REPO
from mlx_generate_ctmr.maisi_mlx_common import PINNED_WEIGHTS_REVISION as PINNED_WEIGHTS_REVISION
from mlx_generate_ctmr.maisi_mlx_common import parameter_count_of as parameter_count_of
from mlx_generate_ctmr.maisi_mlx_common import source_pins as source_pins
from mlx_generate_ctmr.maisi_mlx_common import validate_catalog as _validate_catalog

CATALOG_NAME: Final = "maisi_mlx_from_mask_weight_catalog.json"

NETWORKS: Final[tuple[str, ...]] = ("vae", "unet", "controlnet")
VAE_FILE: Final = "autoencoder_v1.pt"
UNET_FILE: Final = "diff_unet_3d_rflow-ct.pt"
CONTROLNET_FILE: Final = "controlnet_3d_rflow-ct.pt"

# Mask-AE constant from config_network_rflow.json. Image-from-mask scale_factor
# lives in diff_unet_3d_rflow-ct.pt. Never substitute this value.
MASK_AUTOENCODER_SCALE_FACTOR: Final = 1.0055984258651733

INVENTORY_TENSOR_COUNTS: Final[dict[str, int]] = {
    "vae": 130,
    "unet": 435,
    "controlnet": 231,
}
INVENTORY_PARAMETER_COUNTS: Final[dict[str, int]] = {
    "vae": 20_944_897,
    "unet": 180_500_868,
    "controlnet": 72_037_072,
}


_LAYOUTS: Final[frozenset[str]] = IDENTITY_WEIGHT_LAYOUTS | {LAYOUT_CONV3D_W}


def catalog_path() -> Traversable:
    """Return the packaged catalog resource path."""

    return files("mlx_generate_ctmr").joinpath("data", CATALOG_NAME)


def load_catalog(configuration: str = "ct") -> Catalog:
    """Load and validate the checked-in from-mask weight catalog."""

    payload: object = json.loads(catalog_path().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InvalidInputError("MAISI MLX weight catalog must be a JSON object")
    catalog = cast(Catalog, payload)
    validate_catalog(catalog)
    config = get_configuration(configuration)
    if config.name is not Configuration.CT:
        catalog["source"] = source_pins(configuration)
        catalog["networks"] = {name: catalog["networks"][name] for name in config.networks}
        validate_catalog(catalog, configuration=configuration)
    return catalog


def validate_catalog(
    catalog: Mapping[str, object],
    *,
    require_pinned_inventory: bool = True,
    configuration: str = "ct",
) -> None:
    """Check a weight catalog against the pinned tensor inventory of ``configuration``.

    With ``require_pinned_inventory=False`` only internal consistency is checked
    (used for small seeded test catalogs).

    Raises:
        InvalidInputError: If the catalog is inconsistent or does not match.
    """
    config = get_configuration(configuration)
    _validate_catalog(
        catalog,
        tensor_counts={name: INVENTORY_TENSOR_COUNTS[name] for name in config.networks},
        parameter_counts={name: INVENTORY_PARAMETER_COUNTS[name] for name in config.networks},
        inference_flag="used_at_from_mask_infer",
        classify_layout=classify_torch_layout,
        error_prefix="invalid MAISI MLX weight catalog: ",
        require_pinned_inventory=require_pinned_inventory,
    )


def classify_torch_layout(name: str, torch_shape: Sequence[int]) -> str:
    """Return the inventory layout for one Torch parameter."""

    rank = len(torch_shape)
    if rank == 5 and name.endswith(".weight"):
        return LAYOUT_CONV3D_W
    if name == "class_embedding.weight" and rank == 2:
        return LAYOUT_EMBED_W
    if rank == 2 and name.endswith(".weight"):
        return LAYOUT_LINEAR_W
    if rank == 1 and name.endswith(".weight"):
        return LAYOUT_GN_AFFINE
    if rank == 1 and name.endswith(".bias"):
        return LAYOUT_VECTOR
    raise InvalidInputError(
        f"cannot classify parameter {name!r} with torch shape {list(torch_shape)}"
    )


def mlx_shape_for(layout: str, torch_shape: Sequence[int]) -> tuple[int, ...]:
    """Return the destination shape after the named layout remap."""

    if layout not in _LAYOUTS:
        raise InvalidInputError(f"unknown MAISI MLX layout {layout!r}")
    if layout == LAYOUT_CONV3D_W:
        if len(torch_shape) != 5:
            raise InvalidInputError(f"{layout} requires rank-5 [O,I,K]; got {list(torch_shape)}")
        out_c, in_c, k_d, k_h, k_w = (int(dim) for dim in torch_shape)
        return (out_c, k_d, k_h, k_w, in_c)
    return tuple(int(dim) for dim in torch_shape)
