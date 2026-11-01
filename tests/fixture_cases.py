"""Native calls used only by the Metal numerical tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

from mlx_generate_ctmr._arrays import NumericArray, as_numpy

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import (
        Array,
        ArrayConstructor,
        ControlnetConstructor,
        MaskUnetConstructor,
        Mlx,
        RflowConstructor,
        VaeConstructor,
    )


class CtConstructors(TypedDict):
    unet: RflowConstructor
    controlnet: ControlnetConstructor
    vae: VaeConstructor


def run_case(
    kind: str,
    inputs: Mapping[str, NumericArray],
    weights: Mapping[str, NumericArray],
    config: dict[str, object],
    mx: Mlx,
) -> dict[str, NumericArray]:
    """Exercise native model and scheduler APIs on Metal."""
    array = cast("ArrayConstructor", mx.array)
    arrays = {key: array(value) for key, value in inputs.items()}
    state = {key: array(value) for key, value in weights.items()}
    if kind in {"rflow", "mask_ddpm"}:
        output = _schedules(kind, arrays, mx)
    elif kind == "source_fp32":
        output = _arithmetic(arrays, mx)
    elif kind == "from_mask":
        output = _from_mask(inputs, arrays, state, config, mx)
    elif kind in {"vae", "mask_unet", "unet", "controlnet", "mask_encode", "mask_decode"}:
        output = _networks(kind, inputs, arrays, state, config, mx)
    else:
        raise ValueError(kind)
    for value in output.values():
        if isinstance(value, mx.array):
            mx.eval(value)
    return {
        key: as_numpy(value)
        if isinstance(value, mx.array) else cast(NumericArray, value)
        for key, value in output.items()
    }


def _networks(
    kind: str,
    inputs: Mapping[str, NumericArray],
    arrays: dict[str, Array],
    state: dict[str, Array],
    config: dict[str, object],
    mx: Mlx,
) -> dict[str, Array]:
    """Exercise the requested native network against recorded inputs."""
    from mlx_generate_ctmr import controlnet, decode, mask_decode, mask_unet, mask_vae, unet, vae

    x, t = arrays["x"], arrays["timesteps"]
    output: dict[str, Array] = {}
    if kind == "vae":
        output["mu"], output["sigma"] = vae.encode_vae_ncdhw(
            arrays["image"], state, constructor=cast("VaeConstructor", config), mx=mx
        )
        output["post_quant"], output["decoded"] = vae.decode_vae_ncdhw(
            arrays["z"], state, constructor=cast("VaeConstructor", config), mx=mx
        )
        output["tiled"] = decode.sliding_window_decode_ncdhw(
            arrays["z"],
            state,
            constructor=cast("VaeConstructor", config),
            mx=mx,
            roi=(2, 2, 2),
            overlap=0.5,
            scale_factor=1.25,
        )
        output["hu"] = decode.to_hu_ncdhw(output["decoded"], mx.ones_like(output["decoded"]), mx)
    elif kind == "mask_unet":
        output["embedding"], output["eps"] = mask_unet.denoise_mask_unet_ncdhw(
            x, t, arrays["context"], state, constructor=cast("MaskUnetConstructor", config), mx=mx
        )
    elif kind == "unet":
        output["embedding"], output["velocity"] = unet.denoise_unet_ncdhw(
            x,
            t,
            arrays["class_labels"],
            arrays["spacing"],
            state,
            constructor=cast("RflowConstructor", config),
            mx=mx,
        )
    elif kind == "controlnet":
        down, output["mid"] = controlnet.controlnet_residuals_ncdhw(
            x,
            t,
            arrays["class_labels"],
            arrays["cond"],
            state,
            constructor=cast("ControlnetConstructor", config),
            mx=mx,
            conditioning_scale=0.75,
        )
        output.update({f"down_{i}": value for i, value in enumerate(down)})
    elif kind == "mask_encode":
        output["mu"], output["sigma"] = mask_vae.encode_mask_vae_ncdhw(
            arrays["packed"], state, constructor=cast("VaeConstructor", config), mx=mx
        )
    elif kind == "mask_decode":
        output["post_quant"], output["logits"] = mask_vae.decode_mask_vae_ncdhw(
            arrays["z"], state, constructor=cast("VaeConstructor", config), mx=mx
        )
        output["tiled_logits"], output["labels"] = mask_decode.decode_mask_latent_to_labels_ncdhw(
            arrays["z"],
            state,
            label_id_map=inputs["lookup"],
            constructor=cast("VaeConstructor", config),
            roi=(1, 2, 2),
            overlap=0.5,
            scale_factor=1.25,
            mx=mx,
        )
    return output


def _schedules(kind: str, arrays: dict[str, Array], mx: Mlx) -> dict[str, Array | NumericArray]:
    """Exercise the requested diffusion scheduler against recorded inputs."""
    from mlx_generate_ctmr import mask_ddpm as ddpm
    from mlx_generate_ctmr import rflow as scheduler

    x = arrays["x"]
    output: dict[str, Array | NumericArray] = {}
    if kind == "rflow":
        times, next_times = scheduler.set_timesteps_rflow(
            num_inference_steps=7, input_img_size_numel=64
        )
        output.update(timesteps=times, next_timesteps=next_times)
        output["step"], output["origin"], dt = scheduler.step_rflow(
            x, x * 0.125, times[1], next_times[1]
        )
        output["dt"] = np.array(dt)
        output["sample"] = scheduler.sample_rflow(
            x, lambda current, step: current * 0.125 + step * 0.0001, times, mx=mx
        )
    elif kind == "mask_ddpm":
        schedule = ddpm.build_mask_ddpm_schedule()
        output.update(
            betas=schedule.betas,
            alphas=schedule.alphas,
            alphas_cumprod=schedule.alphas_cumprod,
            timesteps=schedule.timesteps,
        )
        for step in (999, 501, 0):
            output[f"step_{step}"], output[f"origin_{step}"] = ddpm.ddpm_step(
                x * 0.125, step, x, schedule, mx=mx, noise=arrays["noise"]
            )
    return output


def _arithmetic(arrays: dict[str, Array], mx: Mlx) -> dict[str, Array]:
    """Exercise source-ordered native arithmetic against recorded inputs."""
    from mlx_generate_ctmr import source_fp32 as fp32
    from mlx_generate_ctmr import source_vae

    x, t = arrays["x"], arrays["timesteps"]
    output: dict[str, Array] = {}
    output["conv"] = fp32.conv3d_ncdhw(x, arrays["kernel_w"], arrays["kernel_b"], padding=1, mx=mx)
    for name, ops in (("norm", fp32), ("vae_norm", source_vae)):
        output[name] = ops.group_norm_ncdhw(
            x, arrays["norm_w"], arrays["norm_b"], num_groups=2, eps=1e-6, mx=mx
        )
    output["silu"] = fp32.silu(x, mx)
    output["swish"] = fp32.swish(x, mx)
    output["linear"] = fp32.linear(arrays["linear_x"], arrays["linear_w"], arrays["linear_b"], mx)
    output["gemv"] = fp32.linear(arrays["linear_x"][:1], arrays["linear_w"], arrays["linear_b"], mx)
    output["qkv"] = fp32.linear(arrays["tokens"], arrays["linear_w"], arrays["linear_b"], mx)
    output["projection"] = fp32.attention_projection(
        arrays["tokens"], arrays["linear_w"], arrays["linear_b"], mx
    )
    output["embedding"] = fp32.sinusoidal_timestep_embedding(t, 5, mx=mx)
    output["attention"] = fp32.attention(arrays["q"], arrays["k"], arrays["v"], scale=0.5, mx=mx)
    output["trilinear"] = source_vae.upsample_trilinear_ncdhw(x, mx=mx)
    return output


def _from_mask(
    inputs: Mapping[str, NumericArray],
    arrays: dict[str, Array],
    state: dict[str, Array],
    config: dict[str, object],
    mx: Mlx,
) -> dict[str, Array | NumericArray]:
    """Exercise the native CT from-mask path against recorded inputs."""
    from mlx_generate_ctmr import from_mask

    constructors = cast("CtConstructors", config)
    x = arrays["x"]
    output: dict[str, Array | NumericArray] = {}
    result = from_mask.sample_ct_from_mask_ncdhw(
        inputs["label"],
        (1, 1.2, 1.5),
        mx=mx,
        latent=x,
        unet_weights={
            key.removeprefix("unet."): value
            for key, value in state.items()
            if key.startswith("unet.")
        },
        controlnet_weights={
            key.removeprefix("controlnet."): value
            for key, value in state.items()
            if key.startswith("controlnet.")
        },
        vae_weights={
            key.removeprefix("vae."): value
            for key, value in state.items()
            if key.startswith("vae.")
        },
        unet_constructor=constructors["unet"],
        controlnet_constructor=constructors["controlnet"],
        vae_constructor=constructors["vae"],
        scale_factor=1.25,
        num_inference_steps=2,
        roi=(4, 4, 4),
    )
    output.update(latent=result.latent, decoded=result.decoded, hu=result.hu, affine=result.affine)
    return output
