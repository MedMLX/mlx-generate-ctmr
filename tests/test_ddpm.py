"""Image (``ddpm-ct``) and paired-mask DDPM schedules, steps and loops against pinned MONAI."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from _torch import DDPMSchedule, from_numpy
from monai.networks.schedulers.ddpm import DDPMScheduler

from mlx_generate_ctmr import ddpm, mask_ddpm
from mlx_generate_ctmr._arrays import HostArray, as_numpy

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, ArrayConstructor, Mlx


def _monai(steps: int = 1000) -> DDPMScheduler:
    """Return the pinned source DDPM scheduler with ``steps`` inference steps."""
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        variance_type="fixed_small",
        clip_sample=False,
    )
    scheduler.set_timesteps(steps)
    return scheduler


def _fix_torch_noise(monkeypatch: pytest.MonkeyPatch, noises: Iterator[HostArray]) -> None:
    """Make the source scheduler draw the given noise arrays in order."""

    def fixed_noise(*args: object, **kwargs: object) -> torch.Tensor:
        """Return the next fixed reference noise sample."""
        return from_numpy(next(noises))

    monkeypatch.setattr(torch, "randn", fixed_noise)


def test_schedules_match_pinned_monai() -> None:
    """Compare both DDPM coefficient tables and the mask timestep order with MONAI."""
    reference = cast(DDPMSchedule, _monai())
    image, mask = ddpm.build_ddpm_schedule(), mask_ddpm.build_mask_ddpm_schedule()
    for name in ("betas", "alphas", "alphas_cumprod"):
        expected = getattr(reference, name).numpy()
        np.testing.assert_array_equal(getattr(image, name), expected, err_msg=name)
        np.testing.assert_allclose(getattr(mask, name), expected, rtol=1e-5, atol=1e-8)
    np.testing.assert_array_equal(mask.timesteps, np.arange(999, -1, -1, dtype=np.int64))


@pytest.mark.parametrize("timestep", [0, 1, 501, 999])
def test_ddpm_steps_match_monai_with_fixed_noise(
    mx: Mlx, timestep: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare one image and one mask DDPM step with MONAI using fixed noise."""
    sample, epsilon, noise = np.random.default_rng(991).normal(size=(3, 1, 4, 2, 2, 2))
    sample, epsilon, noise = (value.astype(np.float32) for value in (sample, epsilon, noise))
    _fix_torch_noise(monkeypatch, iter([noise] * 2))
    expected, expected_origin = _monai().step(from_numpy(epsilon), timestep, from_numpy(sample))
    inputs = (mx.array(epsilon), timestep, mx.array(sample))
    image, image_origin = ddpm.ddpm_step(
        *inputs, ddpm.build_ddpm_schedule(), mx=mx, noise=mx.array(noise)
    )
    masked, _ = mask_ddpm.ddpm_step(
        *inputs, mask_ddpm.build_mask_ddpm_schedule(), mx=mx, noise=mx.array(noise)
    )
    pairs = ((image, expected), (image_origin, expected_origin), (masked, expected))
    for actual, reference in pairs:
        np.testing.assert_allclose(as_numpy(actual), reference.numpy(), rtol=1e-5, atol=1e-6)


def test_ddpm_full_trajectory_matches_monai_with_fixed_noise(
    mx: Mlx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the complete image 999..0 loop against MONAI's scheduler step-by-step."""
    rng = np.random.default_rng(1234)
    sample = rng.normal(size=(1, 2, 2, 2, 2)).astype(np.float32)
    noises = [rng.normal(size=sample.shape).astype(np.float32) for _ in range(1000)]
    weight = rng.normal(size=sample.shape).astype(np.float32) * 0.01
    _fix_torch_noise(monkeypatch, iter(noises))
    scheduler = _monai()
    torch_sample = from_numpy(sample)
    for timestep in range(999, -1, -1):
        torch_sample, _ = scheduler.step(torch_sample * from_numpy(weight), timestep, torch_sample)
    array = cast("ArrayConstructor", mx.array)
    mlx_weight = array(weight)

    def predict_eps(current: Array, _timestep: int) -> Array:
        """Predict deterministic diffusion noise from the current latent."""
        return current * mlx_weight

    actual = ddpm.sample_ddpm(
        array(sample),
        predict_eps,
        ddpm.build_ddpm_schedule(),
        mx=mx,
        step_noises=[array(noise) for noise in noises],
    )
    np.testing.assert_allclose(as_numpy(actual), torch_sample.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("steps", [1, 7, 30, 999])
def test_sparse_mask_ddpm_loop_matches_monai(
    mx: Mlx, steps: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare a reduced-step mask DDPM trajectory with MONAI."""
    rng = np.random.default_rng(42)
    sample = rng.normal(size=(1, 4, 2, 2, 2)).astype(np.float32)
    noise = rng.normal(size=sample.shape).astype(np.float32)
    scheduler = _monai(steps)
    _fix_torch_noise(monkeypatch, iter([noise] * steps))
    reference = from_numpy(sample)
    for index in range(steps):
        timestep = int(scheduler.timesteps[index].item())
        reference, _ = scheduler.step(reference * 0.1, timestep, reference)
    actual = mask_ddpm.sample_ddpm(
        mx.array(sample),
        lambda z, _t: z * 0.1,
        mask_ddpm.build_mask_ddpm_schedule(num_inference_steps=steps),
        mx=mx,
        step_noises=[mx.array(noise)] * steps,
    )
    np.testing.assert_allclose(as_numpy(actual), reference.numpy(), rtol=2e-5, atol=2e-5)
