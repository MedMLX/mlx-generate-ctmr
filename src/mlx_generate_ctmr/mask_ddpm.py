"""Paired-mask DDPM schedule, seeding, and sampling. No MLX import at module load."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, HostArray, MaskUnetConstructor, Mlx, VaeConstructor

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from medmlx_core.layout import require_ncdhw

from mlx_generate_ctmr.decode import DEFAULT_OVERLAP, DEFAULT_ROI_SIZE
from mlx_generate_ctmr.mask_decode import decode_mask_latent_to_labels_ncdhw
from mlx_generate_ctmr.mask_unet import (
    PRODUCTION_MASK_UNET_CONSTRUCTOR,
    anatomy_size_context,
    denoise_mask_unet_ncdhw,
)

# Latent (N, C, D, H, W); the VAE upsamples 4x per axis, so 64**3 decodes to a 256**3
# label volume (at 1.5 mm, a 384 mm cube).
PRODUCTION_MASK_LATENT_SHAPE: tuple[int, int, int, int, int] = (1, 4, 64, 64, 64)
PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS = 1000
PRODUCTION_MASK_NUM_INFERENCE_STEPS = 1000
PRODUCTION_MASK_BETA_START = 0.0015
PRODUCTION_MASK_BETA_END = 0.0195
PRODUCTION_MASK_PREDICTION_TYPE = "epsilon"
PRODUCTION_MASK_VARIANCE_TYPE = "fixed_small"


@dataclass(frozen=True, slots=True)
class MaskDdpmSchedule:
    """Pinned fp32 DDPM schedule arrays plus the selected inference timesteps.

    ``betas``, ``alphas`` and ``alphas_cumprod`` cover all 1000 training timesteps.
    ``timesteps`` holds the descending integer timesteps actually visited
    (``num_inference_steps`` of them).
    """

    timesteps: np.ndarray[tuple[int, ...], np.dtype[np.int64]]
    betas: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    alphas: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    alphas_cumprod: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    prediction_type: str
    variance_type: str
    clip_sample: bool


def build_mask_ddpm_schedule(
    *,
    num_inference_steps: int = PRODUCTION_MASK_NUM_INFERENCE_STEPS,
    num_train_timesteps: int = PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS,
    beta_start: float = PRODUCTION_MASK_BETA_START,
    beta_end: float = PRODUCTION_MASK_BETA_END,
    prediction_type: str = PRODUCTION_MASK_PREDICTION_TYPE,
    variance_type: str = PRODUCTION_MASK_VARIANCE_TYPE,
    clip_sample: bool = False,
) -> MaskDdpmSchedule:
    """Rebuild the pinned 1,000-step scaled-linear beta DDPM schedule in fp32.

    ``num_inference_steps`` (1..1000) selects that many evenly spaced timesteps from
    999 down to 0 (just 999 when it is 1). At each selected timestep the reverse
    formula uses the immediately preceding training index, not the next selected
    timestep. Every other argument must equal its pinned default.

    Raises:
        ValueError: If an argument differs from the pinned schedule.
    """

    if int(num_train_timesteps) != PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS:
        raise ValueError(
            f"paired-mask DDPM requires num_train_timesteps={PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS}"
        )
    if not 1 <= int(num_inference_steps) <= PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS:
        raise ValueError(
            "paired-mask DDPM requires num_inference_steps in "
            f"1..{PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS}"
        )
    if float(beta_start) != PRODUCTION_MASK_BETA_START:
        raise ValueError(f"paired-mask DDPM requires beta_start={PRODUCTION_MASK_BETA_START}")
    if float(beta_end) != PRODUCTION_MASK_BETA_END:
        raise ValueError(f"paired-mask DDPM requires beta_end={PRODUCTION_MASK_BETA_END}")
    if prediction_type != PRODUCTION_MASK_PREDICTION_TYPE:
        raise ValueError(
            f"paired-mask DDPM requires prediction_type={PRODUCTION_MASK_PREDICTION_TYPE!r}"
        )
    if variance_type != PRODUCTION_MASK_VARIANCE_TYPE:
        raise ValueError(
            f"paired-mask DDPM requires variance_type={PRODUCTION_MASK_VARIANCE_TYPE!r}"
        )
    if bool(clip_sample):
        raise ValueError("paired-mask DDPM requires clip_sample=false")

    # Torch linspace rounds its endpoints/step to fp32, then uses one FMA
    # from the nearer endpoint. A float64 intermediate retains that single rounding.
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
    if num_inference_steps == 1:
        timesteps = np.asarray([num_train_timesteps - 1], dtype=np.int64)
    else:
        step = np.float32(-(num_train_timesteps - 1) / (num_inference_steps - 1))
        indices = np.arange(num_inference_steps, dtype=np.float64)
        timesteps = (
            np.where(
                indices < num_inference_steps // 2,
                (num_train_timesteps - 1) + indices * float(step),
                -(num_inference_steps - 1 - indices) * float(step),
            )
            .astype(np.float32)
            .round()
            .astype(np.int64)
        )
    return MaskDdpmSchedule(
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        prediction_type=prediction_type,
        variance_type=variance_type,
        clip_sample=False,
    )


def seed_mask_generation(seed: int, *, mx: Mlx) -> None:
    """Seed Python, NumPy, and MLX so paired-mask host/device helpers stay aligned."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    mx.random.seed(int(seed))


def initialize_mask_noise_latents_ncdhw(
    shape: Sequence[int],
    *,
    seed: int,
    mx: Mlx,
) -> Array:
    """Seeded fp32 ``mx.random.normal`` NCDHW mask latent."""

    if len(shape) != 5:
        raise ValueError(f"mask latent shape must be NCDHW; got {tuple(shape)}")
    seed_mask_generation(seed, mx=mx)
    latent = mx.random.normal(tuple(int(dim) for dim in shape)).astype(mx.float32)
    require_ncdhw(latent, name="mask latent noise")
    return latent


def ddpm_step(
    model_output: Array,
    timestep: int,
    sample: Array,
    schedule: MaskDdpmSchedule,
    *,
    mx: Mlx,
    noise: Array | None = None,
) -> tuple[Array, Array]:
    """One epsilon-prediction, fixed-small-variance, no-clip DDPM reverse step.

    ``model_output`` (predicted noise) and ``sample`` are ``(N, C, D, H, W)``;
    ``timestep`` is an integer index into the schedule. Returns
    ``(prev_sample, pred_original)``. Noise is added only for ``timestep > 0``, from
    ``noise`` if given, otherwise from MLX's global RNG.
    """

    _validate_schedule(schedule, require_timesteps=False)
    require_ncdhw(sample, name="ddpm sample")
    require_ncdhw(model_output, name="ddpm model_output")
    if tuple(int(dim) for dim in model_output.shape) != tuple(int(dim) for dim in sample.shape):
        raise ValueError(
            "ddpm model_output shape "
            f"{tuple(int(dim) for dim in model_output.shape)} != sample shape "
            f"{tuple(int(dim) for dim in sample.shape)}"
        )
    if int(timestep) < 0 or int(timestep) >= int(schedule.alphas_cumprod.shape[0]):
        raise ValueError(f"timestep {timestep} is outside the DDPM schedule")

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
    require_ncdhw(step_noise, name="ddpm step noise")
    if tuple(int(dim) for dim in step_noise.shape) != tuple(int(dim) for dim in sample.shape):
        raise ValueError("ddpm step noise shape must match the sample shape")
    return (
        pred_prev + mx.array(np.sqrt(variance), dtype=mx.float32) * step_noise,
        pred_original,
    )


def sample_ddpm(
    latent: Array,
    predict_eps: Callable[[Array, int], Array],
    schedule: MaskDdpmSchedule,
    *,
    mx: Mlx,
    step_noises: Sequence[Array] | None = None,
) -> Array:
    """Run the schedule's selected DDPM steps, evaluating after each step.

    ``step_noises``, when supplied, has one array per selected timestep in loop
    order. The entry for timestep 0 is ignored because that step adds no noise.
    Reduced schedules still apply the one-training-step reverse formula at each
    selected index; they do not use the next selected index as the predecessor.
    """

    require_ncdhw(latent, name="ddpm latent")
    _validate_schedule(schedule, require_timesteps=True)
    if step_noises is not None and len(step_noises) != int(schedule.timesteps.shape[0]):
        raise ValueError(
            f"step_noises length {len(step_noises)} != timesteps {int(schedule.timesteps.shape[0])}"
        )
    hidden = latent.astype(mx.float32)
    for index, timestep in enumerate(schedule.timesteps.tolist()):
        eps = predict_eps(hidden, int(timestep))
        noise = None if step_noises is None else step_noises[index]
        hidden, _pred_original = ddpm_step(
            eps,
            int(timestep),
            hidden,
            schedule,
            mx=mx,
            noise=noise,
        )
        mx.eval(hidden)
    mx.eval(hidden)
    return hidden


def sample_mask_latent_ncdhw(
    *,
    anatomy_size: HostArray,
    unet_weights: Mapping[str, Array],
    constructor: MaskUnetConstructor = PRODUCTION_MASK_UNET_CONSTRUCTOR,
    mx: Mlx,
    num_inference_steps: int = PRODUCTION_MASK_NUM_INFERENCE_STEPS,
    input_noise: Array | None = None,
    latent_shape: Sequence[int] = PRODUCTION_MASK_LATENT_SHAPE,
    seed: int = 0,
    schedule: MaskDdpmSchedule | None = None,
    step_noises: Sequence[Array] | None = None,
) -> Array:
    """Sample one paired-mask latent with the conditioned UNet and the DDPM loop.

    ``anatomy_size`` is the 10-element size-condition vector (see
    ``mask_condition.MASK_ANATOMY_SIZE_SLOTS``). The start noise is ``input_noise``
    or a standard-normal draw of ``latent_shape`` seeded with ``seed``; ``step_noises``
    optionally fixes the per-step noise. An explicit ``schedule`` takes precedence
    over ``num_inference_steps``. Generating start noise seeds Python, NumPy and
    MLX global RNGs; with supplied start noise, they are seeded only if step noises
    must also be generated. Returns the final latent ``(N, 4, D, H, W)``.
    """

    if schedule is None:
        schedule = build_mask_ddpm_schedule(num_inference_steps=num_inference_steps)
    if input_noise is None:
        hidden = initialize_mask_noise_latents_ncdhw(latent_shape, seed=seed, mx=mx)
    else:
        hidden = input_noise.astype(mx.float32)
        require_ncdhw(hidden, name="mask input noise")
        if step_noises is None:
            seed_mask_generation(seed, mx=mx)
    # The pinned source casts the size condition to FP16 before mask inference. Keep
    # that rounding (compute stays FP32): on the seed-0 capture it cut labels with
    # Dice below 1 from 24 to 2.
    context = anatomy_size_context(anatomy_size, mx=mx).astype(mx.float16).astype(mx.float32)

    def _predict_eps(current: Array, timestep: int) -> Array:
        """Predict mask diffusion noise for one latent and timestep."""
        t = mx.array([float(timestep)], dtype=mx.float32)
        _t_emb, eps = denoise_mask_unet_ncdhw(
            current,
            t,
            context,
            unet_weights,
            constructor=constructor,
            mx=mx,
        )
        return eps

    return sample_ddpm(
        hidden,
        _predict_eps,
        schedule,
        mx=mx,
        step_noises=step_noises,
    )


def sample_mask_labels_ncdhw(
    *,
    anatomy_size: HostArray,
    unet_weights: Mapping[str, Array],
    vae_weights: Mapping[str, Array],
    label_id_map: Sequence[int] | HostArray,
    scale_factor: float,
    constructor: MaskUnetConstructor = PRODUCTION_MASK_UNET_CONSTRUCTOR,
    vae_constructor: VaeConstructor,
    mx: Mlx,
    num_inference_steps: int = PRODUCTION_MASK_NUM_INFERENCE_STEPS,
    input_noise: Array | None = None,
    latent_shape: Sequence[int] = PRODUCTION_MASK_LATENT_SHAPE,
    seed: int = 0,
    schedule: MaskDdpmSchedule | None = None,
    step_noises: Sequence[Array] | None = None,
    decode_roi: Sequence[int] = DEFAULT_ROI_SIZE,
    decode_overlap: float = DEFAULT_OVERLAP,
) -> tuple[Array, Array]:
    """Sample a paired-mask latent and decode it to canonical MAISI label ids.

    Returns ``(latent, labels)`` where ``labels`` is ``(N, 1, 4D, 4H, 4W)`` int32
    holding ids remapped through ``label_id_map`` (see
    ``mask_decode.load_mask_label_id_map``), before any host post-processing.
    ``scale_factor`` is the mask UNet checkpoint's latent scale; ``decode_roi`` and
    ``decode_overlap`` set the VAE decode window in latent voxels.
    """

    latent = sample_mask_latent_ncdhw(
        anatomy_size=anatomy_size,
        unet_weights=unet_weights,
        constructor=constructor,
        mx=mx,
        num_inference_steps=num_inference_steps,
        input_noise=input_noise,
        latent_shape=latent_shape,
        seed=seed,
        schedule=schedule,
        step_noises=step_noises,
    )
    logits, labels = decode_mask_latent_to_labels_ncdhw(
        latent,
        vae_weights,
        label_id_map=label_id_map,
        constructor=vae_constructor,
        mx=mx,
        scale_factor=scale_factor,
        roi=decode_roi,
        overlap=decode_overlap,
    )
    mx.eval(latent, logits, labels)
    return latent, labels


def _validate_schedule(
    schedule: MaskDdpmSchedule,
    *,
    require_timesteps: bool,
) -> None:
    """Require the pinned epsilon/fixed_small/no-clip schedule with exact beta arrays.

    When `require_timesteps` is set, the selected inference timesteps must also be
    the pinned schedule for the same number of steps.
    """
    if schedule.prediction_type != PRODUCTION_MASK_PREDICTION_TYPE:
        raise ValueError("paired-mask DDPM schedule must use epsilon prediction")
    if schedule.variance_type != PRODUCTION_MASK_VARIANCE_TYPE:
        raise ValueError("paired-mask DDPM schedule must use fixed_small variance")
    if schedule.clip_sample:
        raise ValueError("paired-mask DDPM schedule must not clip samples")
    expected_shape = (PRODUCTION_MASK_NUM_TRAIN_TIMESTEPS,)
    for name, values in (
        ("betas", schedule.betas),
        ("alphas", schedule.alphas),
        ("alphas_cumprod", schedule.alphas_cumprod),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"paired-mask DDPM {name} must have shape {expected_shape}")
    expected_schedule = build_mask_ddpm_schedule()
    for name, values, expected in (
        ("betas", schedule.betas, expected_schedule.betas),
        ("alphas", schedule.alphas, expected_schedule.alphas),
        ("alphas_cumprod", schedule.alphas_cumprod, expected_schedule.alphas_cumprod),
    ):
        if not np.array_equal(values, expected):
            raise ValueError(f"paired-mask DDPM {name} do not match the pinned schedule")
    if require_timesteps:
        if schedule.timesteps.ndim != 1:
            raise ValueError("paired-mask DDPM timesteps must be a one-dimensional schedule")
        expected = build_mask_ddpm_schedule(num_inference_steps=int(schedule.timesteps.size))
        if not np.array_equal(schedule.timesteps, expected.timesteps):
            raise ValueError(
                "paired-mask DDPM timesteps must match the pinned selected-step schedule"
            )
