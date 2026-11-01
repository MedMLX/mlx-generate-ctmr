"""Build small native-format weight fixtures from recorded source-layout arrays."""

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mlx_generate_ctmr._arrays import HostArray
from mlx_generate_ctmr.maisi_mlx_common import source_pins
from mlx_generate_ctmr.maisi_mlx_mask_weights import CONV_TRANSPOSE3D_WEIGHT_NAMES


def write_native(
    directory: Path,
    networks: Mapping[str, Mapping[str, HostArray]],
    *,
    configuration: str | None = None,
    mask: bool = False,
    scale_factor: float = 1.25,
) -> None:
    """Write recorded values in the same packed layout used by the released bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, HostArray] = {}
    for network, arrays in networks.items():
        for name, array in arrays.items():
            if array.ndim == 5 and name.endswith(".weight"):
                axes = (
                    (1, 2, 3, 4, 0)
                    if mask and name in CONV_TRANSPOSE3D_WEIGHT_NAMES
                    else (0, 2, 3, 4, 1)
                )
                array = np.transpose(array, axes)
            payload[f"{network}.{name}"] = np.ascontiguousarray(array, dtype=np.float32)
        if network == "unet":
            payload["unet.scale_factor"] = np.asarray(scale_factor, dtype=np.float32)
    save_file(payload, directory / "model.safetensors")
    if configuration is not None:
        (directory / "conversion.json").write_text(
            json.dumps({"configuration": configuration, "source": source_pins(configuration)})
        )
