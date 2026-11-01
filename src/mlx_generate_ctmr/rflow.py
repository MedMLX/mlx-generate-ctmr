"""Rectified-flow schedule and Euler step for from-mask CT. No MLX import."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Array, Mlx

import numpy as np

from mlx_generate_ctmr._arrays import FloatArray

_TRAIN_STEPS = 1000
_BASE_NUMEL = 32 * 32 * 32


def timestep_transform_rflow(
    t: float,
    input_img_size_numel: int,
    *,
    base_img_size_numel: int = _BASE_NUMEL,
    scale: float = 1.0,
    num_train_timesteps: int = _TRAIN_STEPS,
    spatial_dim: int = 3,
) -> float:
    """Shift a timestep so larger volumes spend more steps at high noise.

    With ``u = t / num_train_timesteps`` and
    ``r = (input_img_size_numel / base_img_size_numel) ** (1 / spatial_dim) * scale``,
    returns ``num_train_timesteps * r * u / (1 + (r - 1) * u)``.
    ``input_img_size_numel`` is the latent voxel count (D*H*W). Mirrors MONAI's
    ``timestep_transform``.
    """

    if spatial_dim < 1:
        raise ValueError(f"spatial_dim must be >= 1; got {spatial_dim}")
    if input_img_size_numel < 1 or base_img_size_numel < 1:
        raise ValueError("image size numel must be positive")
    unit = float(t) / float(num_train_timesteps)
    ratio = (float(input_img_size_numel) / float(base_img_size_numel)) ** (
        1.0 / float(spatial_dim)
    ) * float(scale)
    return float(num_train_timesteps) * ratio * unit / (1.0 + (ratio - 1.0) * unit)


def set_timesteps_rflow(
    *,
    num_inference_steps: int = 30,
    num_train_timesteps: int = _TRAIN_STEPS,
    input_img_size_numel: int,
    base_img_size_numel: int = _BASE_NUMEL,
    spatial_dim: int = 3,
    use_timestep_transform: bool = True,
    use_discrete_timesteps: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Build the rflow inference schedule as ``(timesteps, next_timesteps)``.

    Both are float32 arrays of length ``num_inference_steps``. ``timesteps`` falls
    from ``num_train_timesteps`` (pure noise) toward 0 (data) after the
    size-dependent transform, rounded through float16 because MONAI stores the
    schedule in FP16. ``next_timesteps`` is ``timesteps`` shifted by one with a
    final 0. ``input_img_size_numel`` is the latent voxel count (D*H*W).

    Raises:
        ValueError: If an option differs from the only supported rflow-ct setup or
            ``num_inference_steps`` is outside ``1..num_train_timesteps``.
    """

    if use_discrete_timesteps:
        raise ValueError("rflow-ct dest keeps fractional timesteps")
    if not use_timestep_transform:
        raise ValueError("rflow-ct dest requires use_timestep_transform=true")
    if spatial_dim != 3:
        raise ValueError("rflow-ct dest is 3D")
    if num_inference_steps < 1 or num_inference_steps > num_train_timesteps:
        raise ValueError(
            f"num_inference_steps {num_inference_steps} must be in 1..{num_train_timesteps}"
        )
    steps: list[float] = []
    for index in range(num_inference_steps):
        raw = (1.0 - index / num_inference_steps) * num_train_timesteps
        steps.append(
            timestep_transform_rflow(
                raw,
                input_img_size_numel,
                base_img_size_numel=base_img_size_numel,
                # MONAI's set_timesteps does not forward its ``scale`` argument to
                # the transform, so inference always uses 1.0.
                scale=1.0,
                num_train_timesteps=num_train_timesteps,
                spatial_dim=spatial_dim,
            )
        )
    timesteps = np.asarray(steps, dtype=np.float16).astype(np.float32)
    next_timesteps = np.concatenate([timesteps[1:], np.zeros((1,), dtype=np.float32)])
    return timesteps, next_timesteps


def step_rflow(
    z: Array,
    v: Array,
    timestep: float,
    next_timestep: float,
    *,
    num_train_timesteps: int = _TRAIN_STEPS,
) -> tuple[Array, Array, float]:
    """One Euler step: returns ``(z_next, z_origin, dt)``.

    ``v`` is the predicted velocity, pointing from noise (``t = T``) toward data
    (``t = 0``). ``z_next = z + v * dt`` with ``dt = (t - int(next_t)) / T``, and
    ``z_origin = z + v * t / T`` is the implied clean sample. The ``int()`` on
    ``next_timestep`` matches the pinned source, which truncates it before the
    subtraction.
    """

    dt = (float(timestep) - int(float(next_timestep))) / float(num_train_timesteps)
    z_next = z + v * dt
    z_origin = z + v * (float(timestep) / float(num_train_timesteps))
    return z_next, z_origin, dt


def init_latent_ncdhw(
    shape: Sequence[int],
    *,
    seed: int,
    mx: Mlx,
    noise_factor: float = 1.0,
) -> Array:
    """Return a seeded standard-normal float32 latent of NCDHW ``shape``.

    Seeds MLX's global RNG with ``seed`` (a side effect). ``noise_factor`` must be 1.0.
    """

    if len(shape) != 5:
        raise ValueError(f"latent shape must be NCDHW; got {tuple(shape)}")
    if noise_factor != 1.0:
        raise ValueError("rflow-ct dest noise_factor is 1.0")
    mx.random.seed(int(seed))
    return mx.random.normal(tuple(int(dim) for dim in shape)).astype(mx.float32)


def sample_rflow(
    z: Array,
    velocity_fn: Callable[[Array, float], Array],
    timesteps: FloatArray,
    *,
    mx: Mlx,
    num_train_timesteps: int = _TRAIN_STEPS,
    guidance_scale: float = 0.0,
) -> Array:
    """Integrate ``z`` along the schedule with Euler steps and evaluate the result.

    ``velocity_fn(z, t)`` returns the model velocity for latent ``z`` at timestep
    ``t``; ``timesteps`` comes from ``set_timesteps_rflow``. ``guidance_scale``
    must be 0 here: modality classifier-free guidance, where used, lives inside
    ``velocity_fn`` (see ``image.guided_velocity``).
    """

    if float(guidance_scale) != 0.0:
        raise ValueError("CFG>0 majority vote is not an MLX destination")
    if timesteps.ndim != 1 or timesteps.size < 1:
        raise ValueError("timesteps must be a 1-D schedule")
    next_timesteps = np.concatenate([timesteps[1:], np.zeros((1,), dtype=np.float32)])
    hidden = z
    for timestep, next_timestep in zip(timesteps.tolist(), next_timesteps.tolist(), strict=True):
        velocity = velocity_fn(hidden, float(timestep))
        hidden, _origin, _dt = step_rflow(
            hidden,
            velocity,
            float(timestep),
            float(next_timestep),
            num_train_timesteps=num_train_timesteps,
        )
    mx.eval(hidden)
    return hidden
