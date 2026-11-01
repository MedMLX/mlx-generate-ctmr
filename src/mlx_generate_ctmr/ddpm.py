"""Image DDPM schedule and reverse step for ``ddpm-ct``. No MLX import at load.

``ddpm-ct`` is the released NV-Generate-CT alternative to ``rflow-ct``: the same
autoencoder with a DDPM-scheduled, body-region-conditioned diffusion UNet and
ControlNet (``diff_unet_3d_ddpm-ct.pt`` / ``controlnet_3d_ddpm-ct.pt``), sampled
over the full 1,000-step reverse trajectory. This module is independent of
``mask_ddpm.py`` (the mask-generation DDPM destination): the two share the same
scaled-linear-beta hyperparameters by upstream coincidence, not by code reuse,
and are pinned and tested separately like the rest of this package's per-destination
modules (``rflow.py`` vs. mask generation, ``source_fp32.py`` vs. ``source_vae.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr._arrays import FloatArray

NUM_TRAIN_TIMESTEPS = 1000
NUM_INFERENCE_STEPS = 1000
BETA_START = 0.0015
BETA_END = 0.0195
PREDICTION_TYPE = "epsilon"
VARIANCE_TYPE = "fixed_small"


@dataclass(frozen=True, slots=True)
class DdpmSchedule:
    """Pinned fp32 ``ddpm-ct`` schedule: scaled-linear beta, full reverse trajectory."""

    betas: FloatArray
    alphas: FloatArray
    alphas_cumprod: FloatArray


def build_ddpm_schedule(
    *,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
) -> DdpmSchedule:
    """Rebuild the pinned ``ddpm-ct`` scaled-linear-beta DDPM schedule in fp32."""

    if int(num_train_timesteps) != NUM_TRAIN_TIMESTEPS:
        raise ValueError(f"ddpm-ct requires num_train_timesteps={NUM_TRAIN_TIMESTEPS}")
    if float(beta_start) != BETA_START:
        raise ValueError(f"ddpm-ct requires beta_start={BETA_START}")
    if float(beta_end) != BETA_END:
        raise ValueError(f"ddpm-ct requires beta_end={BETA_END}")

    # Torch linspace rounds its endpoints/step to fp32, then uses one FMA from
    # the nearer endpoint. A float64 intermediate retains that single rounding.
    start = np.float32(beta_start**0.5)
    end = np.float32(beta_end**0.5)
    step = np.float32((float(end) - float(start)) / (num_train_timesteps - 1))
    indices = np.arange(num_train_timesteps, dtype=np.float64)
    roots = np.where(
        indices < num_train_timesteps // 2,
        float(start) + indices * float(step),
        float(end) - (num_train_timesteps - 1 - indices) * float(step),
    ).astype(np.float32)
    betas = roots * roots
    alphas = np.float32(1.0) - betas
    # The source CPU scheduler accumulates cumprod in double and stores fp32.
    alphas_cumprod = np.cumprod(alphas, dtype=np.float64).astype(np.float32)
    return DdpmSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def ddpm_step(
    model_output: Array,
    timestep: int,
    sample: Array,
    schedule: DdpmSchedule,
    *,
    mx: Mlx,
    noise: Array | None = None,
) -> tuple[Array, Array]:
    """One epsilon-prediction, fixed-small-variance, no-clip DDPM reverse step.

    ``model_output`` is the predicted noise and ``sample`` the current latent, both
    ``(N, C, D, H, W)``. ``timestep`` is an integer in ``0..len(schedule.betas)-1``.
    Returns ``(prev, pred_original)``: the sample for the previous timestep and
    the implied clean sample. Noise is added only for ``timestep > 0``; pass
    ``noise`` (same shape as ``sample``) to make the step deterministic, otherwise
    it is drawn from MLX's global RNG.
    """

    _validate_schedule(schedule)
    require_ncdhw(sample, name="ddpm-ct sample")
    require_ncdhw(model_output, name="ddpm-ct model_output")
    if tuple(int(dim) for dim in model_output.shape) != tuple(int(dim) for dim in sample.shape):
        raise ValueError(
            "ddpm-ct model_output shape "
            f"{tuple(int(dim) for dim in model_output.shape)} != sample shape "
            f"{tuple(int(dim) for dim in sample.shape)}"
        )
    if int(timestep) < 0 or int(timestep) >= int(schedule.alphas_cumprod.shape[0]):
        raise ValueError(f"timestep {timestep} is outside the ddpm-ct schedule")

    alpha_prod_t = schedule.alphas_cumprod[int(timestep)]
    alpha_prod_t_prev = (
        schedule.alphas_cumprod[int(timestep) - 1] if int(timestep) > 0 else np.float32(1.0)
    )
    beta_prod_t = np.float32(1.0) - alpha_prod_t
    beta_prod_t_prev = np.float32(1.0) - alpha_prod_t_prev

    # These are fp32 CPU scalars in the source scheduler. Keep each rounding
    # before applying the resulting coefficients to the Metal arrays.
    pred_original = (
        sample - mx.array(np.sqrt(beta_prod_t), dtype=mx.float32) * model_output
    ) / mx.array(np.sqrt(alpha_prod_t), dtype=mx.float32)
    pred_original_coeff = np.sqrt(alpha_prod_t_prev) * schedule.betas[int(timestep)] / beta_prod_t
    current_sample_coeff = np.sqrt(schedule.alphas[int(timestep)]) * beta_prod_t_prev / beta_prod_t
    pred_prev = (
        mx.array(pred_original_coeff, dtype=mx.float32) * pred_original
        + mx.array(current_sample_coeff, dtype=mx.float32) * sample
    )
    if int(timestep) == 0:
        return pred_prev, pred_original

    variance = beta_prod_t_prev / beta_prod_t * schedule.betas[int(timestep)]
    variance = np.maximum(variance, np.float32(1.0e-20))
    step_noise = (
        noise.astype(mx.float32)
        if noise is not None
        else mx.random.normal(sample.shape).astype(mx.float32)
    )
    require_ncdhw(step_noise, name="ddpm-ct step noise")
    if tuple(int(dim) for dim in step_noise.shape) != tuple(int(dim) for dim in sample.shape):
        raise ValueError("ddpm-ct step noise shape must match the sample shape")
    return (
        pred_prev + mx.array(np.sqrt(variance), dtype=mx.float32) * step_noise,
        pred_original,
    )


def init_latent_ncdhw(
    shape: tuple[int, int, int, int, int],
    *,
    seed: int,
    mx: Mlx,
) -> Array:
    """Seeded fp32 ``mx.random.normal`` NCDHW image latent."""

    if len(shape) != 5:
        raise ValueError(f"ddpm-ct latent shape must be NCDHW; got {tuple(shape)}")
    mx.random.seed(int(seed))
    latent = mx.random.normal(tuple(int(dim) for dim in shape)).astype(mx.float32)
    require_ncdhw(latent, name="ddpm-ct latent noise")
    return latent


def sample_ddpm(
    latent: Array,
    predict_eps: Callable[[Array, int], Array],
    schedule: DdpmSchedule | None = None,
    *,
    mx: Mlx,
    step_noises: list[Array] | None = None,
) -> Array:
    """Run the full 999..0 ``ddpm-ct`` reverse loop, evaluating after every step.

    ``predict_eps(z, t)`` returns the model's noise estimate for latent ``z`` at
    integer timestep ``t``. ``step_noises``, if given, holds one pre-drawn noise
    array per step (1000 entries, in loop order) for reproducible runs. The final
    entry is unused because timestep 0 adds no noise.
    """

    require_ncdhw(latent, name="ddpm-ct latent")
    if schedule is None:
        schedule = build_ddpm_schedule()
    _validate_schedule(schedule)
    if step_noises is not None and len(step_noises) != NUM_INFERENCE_STEPS:
        raise ValueError(
            f"step_noises length {len(step_noises)} != num_inference_steps {NUM_INFERENCE_STEPS}"
        )
    hidden = latent.astype(mx.float32)
    for index, timestep in enumerate(range(NUM_TRAIN_TIMESTEPS - 1, -1, -1)):
        eps = predict_eps(hidden, int(timestep))
        noise = None if step_noises is None else step_noises[index]
        hidden, _pred_original = ddpm_step(eps, int(timestep), hidden, schedule, mx=mx, noise=noise)
        mx.eval(hidden)
    mx.eval(hidden)
    return hidden


def _validate_schedule(schedule: DdpmSchedule) -> None:
    """Require the schedule arrays to be the pinned (1000,) float schedule bit-for-bit."""
    expected_shape = (NUM_TRAIN_TIMESTEPS,)
    for name, values in (
        ("betas", schedule.betas),
        ("alphas", schedule.alphas),
        ("alphas_cumprod", schedule.alphas_cumprod),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"ddpm-ct {name} must have shape {expected_shape}")
    expected = build_ddpm_schedule()
    for name, values, reference in (
        ("betas", schedule.betas, expected.betas),
        ("alphas", schedule.alphas, expected.alphas),
        ("alphas_cumprod", schedule.alphas_cumprod, expected.alphas_cumprod),
    ):
        if not np.array_equal(values, reference):
            raise ValueError(f"ddpm-ct {name} do not match the pinned schedule")


__all__ = [
    "BETA_END",
    "BETA_START",
    "NUM_INFERENCE_STEPS",
    "NUM_TRAIN_TIMESTEPS",
    "PREDICTION_TYPE",
    "VARIANCE_TYPE",
    "DdpmSchedule",
    "build_ddpm_schedule",
    "ddpm_step",
    "init_latent_ncdhw",
    "sample_ddpm",
]
