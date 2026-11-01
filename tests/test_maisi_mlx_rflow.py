"""MAISI MLX rflow schedule, Euler step and sampling parity."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from _torch import RFlowStep, from_numpy
from golden_fixtures import load_boundary
from medmlx_core.ops import as_fp32
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.maisi_mlx_feasibility import UNET_ATOL, UNET_RTOL
from mlx_generate_ctmr.rflow import sample_rflow, set_timesteps_rflow, step_rflow

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx


def test_rflow_schedule_rounds_to_pinned_fp16_table() -> None:
    """Compare rectified-flow schedule rounding with the recorded FP16 table."""
    manifest, arrays = load_boundary("scheduler")
    constructor = manifest["constructor"]
    timesteps, next_timesteps = set_timesteps_rflow(
        num_inference_steps=int(manifest["num_inference_steps"]),
        num_train_timesteps=int(constructor["num_train_timesteps"]),
        input_img_size_numel=int(manifest["input_img_size_numel"]),
        base_img_size_numel=int(constructor["base_img_size_numel"]),
        spatial_dim=int(constructor["spatial_dim"]),
        use_timestep_transform=bool(constructor["use_timestep_transform"]),
        use_discrete_timesteps=bool(constructor["use_discrete_timesteps"]),
    )
    np.testing.assert_array_equal(timesteps, arrays["timesteps"].astype(np.float32))
    np.testing.assert_array_equal(next_timesteps, arrays["next_timesteps"].astype(np.float32))


def test_rflow_step_matches_torch_golden(mx: Mlx) -> None:
    """Compare a rectified-flow step with recorded Torch values."""
    manifest, arrays = load_boundary("scheduler")
    z_next, z_origin, dt = step_rflow(
        as_fp32(arrays["z"], mx),
        as_fp32(arrays["v"], mx),
        float(arrays["timesteps"][0]),
        float(arrays["next_timesteps"][0]),
        num_train_timesteps=int(manifest["constructor"]["num_train_timesteps"]),
    )
    assert dt == pytest.approx(float(arrays["dt"]), rel=0.0, abs=1e-7)
    for actual, key in ((z_next, "z_next"), (z_origin, "z_origin")):
        np.testing.assert_allclose(as_numpy(actual), arrays[key], rtol=UNET_RTOL, atol=UNET_ATOL)


def test_rflow_sample_matches_live_torch(mx: Mlx) -> None:
    """Compare a seeded rectified-flow sample with live Torch."""
    manifest, arrays = load_boundary("scheduler")
    timesteps, next_t = set_timesteps_rflow(
        num_inference_steps=int(manifest["num_inference_steps"]),
        input_img_size_numel=int(manifest["input_img_size_numel"]),
        base_img_size_numel=int(manifest["constructor"]["base_img_size_numel"]),
    )
    scheduler = RFlowScheduler(**manifest["constructor"])
    scheduler.set_timesteps(
        num_inference_steps=int(manifest["num_inference_steps"]),
        device="cpu",
        input_img_size_numel=int(manifest["input_img_size_numel"]),
    )
    z_t, v_t = from_numpy(arrays["z"].copy()), from_numpy(arrays["v"])
    # Feed matching rounded source times to isolate the update equation.
    with torch.inference_mode():
        for t, nt in zip(timesteps.tolist(), next_t.tolist(), strict=True):
            z_t, _ = cast(RFlowStep, scheduler.step)(v_t, float(t), z_t, float(nt))
    z, v = as_fp32(arrays["z"], mx), as_fp32(arrays["v"], mx)

    def constant_v(_hidden: Array, _t: float) -> Array:
        """Return the fixed native velocity used by the fixture."""
        return v

    final = sample_rflow(z, constant_v, timesteps, mx=mx)
    np.testing.assert_allclose(as_numpy(final), z_t.numpy(), rtol=UNET_RTOL, atol=UNET_ATOL)
