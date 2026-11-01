"""Native image and paired-mask safetensors layout admission."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from _contracts import WeightContract
from _native_fixture import write_native

from mlx_generate_ctmr import maisi_mlx_mask_weights as mask_contract
from mlx_generate_ctmr import maisi_mlx_weights as image_contract
from mlx_generate_ctmr import weights
from mlx_generate_ctmr._arrays import HostArray, NumericArray, as_numpy
from mlx_generate_ctmr._records import Catalog, TensorSpec

if TYPE_CHECKING:
    from pathlib import Path

    from mlx_generate_ctmr._types import Array, ArrayFactory, Dtype, Mlx


def synthetic_catalog(
    contract: WeightContract,
) -> tuple[Catalog, dict[str, dict[str, NumericArray]]]:
    """Small inventories preserving encoder-only flags and source layouts."""
    catalog: Catalog = {
        "destination_dtype": contract.DESTINATION_DTYPE,
        "networks": {},
    }
    arrays: dict[str, dict[str, NumericArray]] = {}
    flag = (
        "used_at_mask_generation_infer" if contract is mask_contract else "used_at_from_mask_infer"
    )
    for network in contract.NETWORKS:
        shapes = {
            "conv_in.conv.weight": (2, 3, 1, 2, 3),
            "conv_in.conv.bias": (2,),
            "norm.weight": (2,),
            "time_embed.0.weight": (2, 3),
        }
        if network == "vae":
            shapes["encoder.bias"] = (2,)
            if contract is mask_contract:
                shapes["decoder.blocks.3.conv.conv.conv.weight"] = (3, 2, 1, 2, 3)
        tensors: list[TensorSpec] = []
        arrays[network] = {}
        for index, (name, shape) in enumerate(shapes.items()):
            arrays[network][name] = (
                np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + index / 10
            )
            layout = contract.classify_torch_layout(name, shape)
            tensors.append(
                cast(
                    TensorSpec,
                    {
                        "name": name,
                        "torch_shape": list(shape),
                        "layout": layout,
                        flag: not name.startswith("encoder."),
                    },
                )
            )
        catalog["networks"][network] = {
            "tensors": tensors,
            "tensor_count": len(tensors),
            "parameter_count": sum(array.size for array in arrays[network].values()),
        }
    contract.validate_catalog(catalog, require_pinned_inventory=False)
    return catalog, arrays



@pytest.mark.parametrize("contract", [image_contract, mask_contract], ids=["image", "mask"])
def test_native_weight_layouts(contract: WeightContract, tmp_path: Path, metal: Mlx) -> None:
    """Validate native checkpoint layouts and restored source tensor values."""
    catalog, raw = synthetic_catalog(contract)
    arrays = {
        network: {name: cast(HostArray, value) for name, value in state.items()}
        for network, state in raw.items()
    }
    write_native(tmp_path, arrays, mask=contract is mask_contract)
    loader = weights.load_network if contract is image_contract else weights.load_mask_network
    for network in raw:
        loaded, scale = loader(
            tmp_path / "model.safetensors",
            network=network,
            mx=cast("ArrayFactory[Array, Dtype]", metal),
            catalog=catalog,
        )
        assert scale == (1.25 if network == "unet" else None)
        assert loaded.keys() == raw[network].keys()
        for name, value in loaded.items():
            assert np.array_equal(as_numpy(value), raw[network][name].astype(np.float32)), name
