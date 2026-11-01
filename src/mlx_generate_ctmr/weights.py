"""Load native safetensors checkpoints from explicit caller-supplied paths."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

from mlx_generate_ctmr._arrays import HostArray
from mlx_generate_ctmr._records import Catalog

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import ArrayFactory

import numpy as np
from medmlx_core.errors import InvalidInputError
from safetensors import safe_open
from safetensors.numpy import load_file

from mlx_generate_ctmr.configurations import get_configuration
from mlx_generate_ctmr.maisi_mlx_mask_weights import (
    LAYOUT_CONV_TRANSPOSE3D_W as MASK_LAYOUT_CONV_TRANSPOSE3D_W,
)
from mlx_generate_ctmr.maisi_mlx_mask_weights import load_catalog as load_mask_catalog
from mlx_generate_ctmr.maisi_mlx_weights import (
    LAYOUT_CONV3D_W,
    MASK_AUTOENCODER_SCALE_FACTOR,
    load_catalog,
    mlx_shape_for,
    validate_catalog,
)


def network_file(directory: Path) -> Path:
    """Return the consolidated native weights file."""
    return directory / "model.safetensors"


def load_network[Tensor, Dtype](
    path: Path,
    *,
    network: str,
    mx: ArrayFactory[Tensor, Dtype],
    catalog: Mapping[str, object] | None = None,
    configuration: str = "ct",
) -> tuple[dict[str, Tensor], float | None]:
    """Load one native image network (``"vae"``, ``"unet"`` or ``"controlnet"``).

    Returns ``(weights, scale_factor)``. ``weights`` maps tensor names to float32
    arrays in Torch layouts (convolution weights are transposed back from the
    channels-last layout stored on disk). ``scale_factor`` is the checkpoint's
    latent scale for the UNet and ``None`` for the other networks. ``path`` is an
    ``.safetensors`` file. A file named ``model.safetensors`` uses
    ``<network>.``-prefixed keys; other files use unprefixed keys for one network.
    Every selected tensor is checked against the catalog for name, shape, dtype
    and finiteness. A conversion receipt is not required.

    Raises:
        InvalidInputError: If the file does not match the configured tensor catalog.
    """

    get_configuration(configuration)
    if catalog is None:
        require_configuration(path.parent, configuration=configuration)
    else:
        validate_catalog(catalog, require_pinned_inventory=False, configuration=configuration)

    return _load_with_catalog(
        path,
        network=network,
        mx=mx,
        catalog=load_catalog(configuration) if catalog is None else cast(Catalog, catalog),
    )


def require_configuration(directory: Path, *, configuration: str = "ct") -> None:
    """Validate the selected fixed architecture; conversion receipts are not required."""
    get_configuration(configuration)


def load_mask_network[Tensor, Dtype](
    path: Path,
    *,
    network: str,
    mx: ArrayFactory[Tensor, Dtype],
    catalog: Mapping[str, object] | None = None,
) -> tuple[dict[str, Tensor], float | None]:
    """Load one native paired-mask network (``"vae"`` or ``"unet"``).

    Same contract as ``load_network``, using the mask catalog; the mask
    VAE's transposed-convolution weights are likewise returned in Torch layout.
    """

    return _load_with_catalog(
        path,
        network=network,
        mx=mx,
        catalog=load_mask_catalog() if catalog is None else cast(Catalog, catalog),
    )


def _load_with_catalog[Tensor, Dtype](
    path: Path,
    *,
    network: str,
    mx: ArrayFactory[Tensor, Dtype],
    catalog: Catalog,
) -> tuple[dict[str, Tensor], float | None]:
    """Load native tensors. Conv weights are returned in Torch [O,I,K] layout."""

    networks = catalog["networks"]
    if network not in networks:
        raise InvalidInputError(f"unknown network {network!r}")
    specs = {str(item["name"]): item for item in networks[network]["tensors"]}
    arrays = _read_arrays(path, network=network)
    weights: dict[str, Tensor] = {}
    scale_factor: float | None = None
    for name, array in arrays.items():
        if name == "scale_factor":
            scale_factor = _require_scale(array)
            continue
        spec = specs.get(name)
        if spec is None:
            raise InvalidInputError(f"{path.name} has unexpected tensor {name}")
        layout = str(spec["layout"])
        if layout == MASK_LAYOUT_CONV_TRANSPOSE3D_W:
            shape = spec["torch_shape"]
            expected = (shape[1], *shape[2:], shape[0])
        else:
            expected = mlx_shape_for(layout, spec["torch_shape"])
        if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
            raise InvalidInputError(
                f"{path.name} tensor {name} must be finite float32 with shape {expected}; "
                f"got {array.dtype} {array.shape}"
            )
        if layout == LAYOUT_CONV3D_W:
            array = np.transpose(array, (0, 4, 1, 2, 3))
        elif layout == MASK_LAYOUT_CONV_TRANSPOSE3D_W:
            array = np.transpose(array, (4, 0, 1, 2, 3))
        weights[name] = mx.array(array, dtype=mx.float32)
    missing = [name for name in specs if name not in weights]
    if missing:
        raise InvalidInputError(f"{path.name} is missing {missing[0]}")
    if network == "unet":
        if scale_factor is None:
            raise InvalidInputError(
                f"{path.name} is missing scale_factor. "
                f"Refuse to invent {MASK_AUTOENCODER_SCALE_FACTOR}."
            )
    elif scale_factor is not None:
        raise InvalidInputError(f"{path.name} must not contain scale_factor")
    return weights, scale_factor


def _read_arrays(path: Path, *, network: str) -> dict[str, HostArray]:
    """Read native checkpoint arrays and admit the supported file layout."""
    if path.name == "model.safetensors":
        prefix = network + "."
        with safe_open(path, framework="np") as tensors:
            names = tensors.keys()
            return {
                name.removeprefix(prefix): tensors.get_tensor(name)
                for name in names
                if name.startswith(prefix)
            }
    if path.suffix == ".safetensors":
        return cast(dict[str, HostArray], load_file(path))
    raise InvalidInputError(f"{path.name} must be a safetensors file")


def _require_scale(array: HostArray) -> float:
    """Validate and extract a scalar latent scale factor."""
    if array.size != 1:
        raise InvalidInputError(f"scale_factor must be a scalar; got {list(array.shape)}")
    value = float(array.reshape(()))
    if not np.isfinite(value) or value == 0.0:
        raise InvalidInputError(
            "scale_factor must be a finite non-zero checkpoint value; "
            f"refuse to invent {MASK_AUTOENCODER_SCALE_FACTOR}"
        )
    return value
