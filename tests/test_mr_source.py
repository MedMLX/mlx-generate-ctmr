"""Pinned MONAI scheduler and nonzero CFG oracles for native MRI synthesis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch
from _torch import RFlowStep, from_numpy
from golden_fixtures import seed_torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from mlx_generate_ctmr._arrays import as_numpy
from mlx_generate_ctmr.maisi_mlx_from_mask import PRODUCTION_UNET_CONSTRUCTOR
from mlx_generate_ctmr.mr import mr_timesteps, sample_mr_latent_ncdhw

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import Mlx


def _scheduler(steps: int, numel: int) -> RFlowScheduler:
    """Return the pinned MR rectified-flow scheduler for a latent size."""
    scheduler = RFlowScheduler(
        num_train_timesteps=1000, use_discrete_timesteps=False, use_timestep_transform=True
    )
    scheduler.set_timesteps(num_inference_steps=steps, input_img_size_numel=numel)
    return scheduler


@pytest.mark.parametrize(
    "shape",
    [
        (128, 128, 128),
        (256, 256, 128),
        (384, 384, 128),
        (512, 512, 128),
        (128, 256, 256),
        (256, 128, 256),
        (256, 256, 256),
    ],
)
def test_mr_schedule_matches_source_fp16_on_every_public_output_grid(
    shape: tuple[int, int, int],
) -> None:
    """Compare MR schedules with source FP16 values for public grids."""
    latent_shape = tuple(value // 4 for value in shape)
    # The source passes a tensor element count; keep that path.
    scheduler = _scheduler(30, cast(int, torch.prod(torch.tensor(latent_shape))))
    np.testing.assert_array_equal(
        mr_timesteps(latent_numel=math.prod(latent_shape)), scheduler.timesteps.float().numpy()
    )


def test_mr_cfg_trajectory_matches_pinned_monai_forward_and_scheduler(mx: Mlx) -> None:
    """Compare the guided MR trajectory with the MONAI source."""
    constructor = PRODUCTION_UNET_CONSTRUCTOR.copy()
    constructor.update(
        num_channels=[8, 16],
        num_res_blocks=1,
        attention_levels=[False, True],
        num_head_channels=[0, 4],
        norm_num_groups=4,
    )
    seed_torch(42)
    source = DiffusionModelUNetMaisi(**constructor).eval()
    # Source UNet output is zero-initialized. Fill it so CFG ordering is observable.
    with torch.no_grad():
        conv = cast("torch.nn.Conv3d", source.out[-1].conv)
        conv.weight.normal_(std=0.02)
        assert conv.bias is not None
        conv.bias.normal_(std=0.02)
    weights = {key: mx.array(value.numpy()) for key, value in source.state_dict().items()}
    noise = np.random.default_rng(3).normal(size=(1, 4, 8, 8, 8)).astype(np.float32)
    hidden = from_numpy(noise.copy())
    scheduler = _scheduler(2, 8**3)
    next_steps = torch.cat((scheduler.timesteps[1:], torch.zeros(1)))
    with torch.no_grad():
        for timestep, next_timestep in zip(scheduler.timesteps, next_steps, strict=True):
            conditional, unconditional = source(
                torch.cat((hidden, hidden)),
                timesteps=timestep.repeat(2),
                class_labels=torch.tensor([9, 0]),
                spacing_tensor=torch.tensor([[94.0, 94.0, 136.0]]).repeat(2, 1),
            ).chunk(2)
            velocity = unconditional + 2.0 * (conditional - unconditional)
            hidden, _ = cast(RFlowStep, scheduler.step)(velocity, timestep, hidden, next_timestep)
    actual = sample_mr_latent_ncdhw(
        mx.array(noise),
        weights,
        modality_code=9,
        spacing_mm=(0.94, 0.94, 1.36),
        guidance_scale=2.0,
        mx=mx,
        num_inference_steps=2,
        constructor=constructor,
    )
    np.testing.assert_allclose(as_numpy(actual), hidden.numpy(), rtol=1e-4, atol=1e-5)
