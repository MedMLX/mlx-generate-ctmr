"""CT-from-mask parity against pinned Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from _torch import RFlowStep, from_numpy
from golden_fixtures import load_boundary, mlx_weights, torch_weights
from medmlx_core.ops import as_fp32
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.schedulers.rectified_flow import RFlowScheduler

from mlx_generate_ctmr._arrays import HostArray, as_numpy
from mlx_generate_ctmr.condition import binarize_labels_ncdhw
from mlx_generate_ctmr.from_mask import IDENTITY_AFFINE, FromMaskArrays, sample_ct_from_mask_ncdhw
from mlx_generate_ctmr.maisi_mlx_from_mask import (
    E2E_DECODE_ATOL,
    E2E_DECODE_RTOL,
    E2E_HU_ATOL,
    E2E_HU_RTOL,
    E2E_LATENT_ATOL,
    E2E_LATENT_RTOL,
)
from mlx_generate_ctmr.rflow import set_timesteps_rflow

if TYPE_CHECKING:
    from mlx_generate_ctmr._types import ArrayConstructor, Mlx

_STEPS = 2
_SCALE = 1.0
_SPACING_MM = (1.5, 1.5, 2.0)


def _tiny_label() -> HostArray:
    """Create a small categorical mask for from-mask tests."""
    label = np.zeros((1, 1, 16, 16, 16), dtype=np.int16)
    label[:, :, 2:14, 2:14, 2:14] = 200
    label[:, :, 4:10, 4:10, 4:10] = 1
    label[:, :, 6:8, 6:8, 6:8] = 26
    return label


def _torch_binarize(label: HostArray) -> torch.Tensor:
    """Pack a label mask into source-format bit channels with Torch."""
    ids = from_numpy(label.astype(np.int64))
    mask = 2 ** torch.arange(8).to(ids.dtype)
    packed = ids.unsqueeze(-1).bitwise_and(mask).ne(0).byte().squeeze(1)
    return packed.permute(0, 4, 1, 2, 3).float()


def _sample(label: HostArray, mx: Mlx) -> FromMaskArrays:
    """Run the native from-mask graph on the recorded synthetic goldens."""
    cn_manifest, cn_arrays = load_boundary("controlnet")
    unet_manifest, unet_arrays = load_boundary("diffusion")
    vae_manifest, vae_arrays = load_boundary("vae")
    return sample_ct_from_mask_ncdhw(
        label,
        _SPACING_MM,
        unet_weights=mlx_weights(unet_arrays, mx),
        controlnet_weights=mlx_weights(cn_arrays, mx),
        vae_weights=mlx_weights(vae_arrays, mx),
        unet_constructor=unet_manifest["constructor"],
        controlnet_constructor=cn_manifest["constructor"],
        vae_constructor=vae_manifest["constructor"],
        mx=mx,
        scale_factor=_SCALE,
        latent=as_fp32(unet_arrays["x"], mx),
        num_inference_steps=_STEPS,
    )


def test_binarize_labels_matches_pinned_torch(mx: Mlx) -> None:
    """Compare native mask-bit packing with the source computation."""
    label = _tiny_label()
    packed = binarize_labels_ncdhw(cast("ArrayConstructor", mx.array)(label, dtype=mx.int32), mx=mx)
    np.testing.assert_array_equal(as_numpy(packed), _torch_binarize(label).numpy())


def test_tiny_from_mask_e2e_matches_pinned_torch(mx: Mlx) -> None:
    """Compare a small seeded CT from-mask pipeline with the source."""
    cn_manifest, cn_arrays = load_boundary("controlnet")
    unet_manifest, unet_arrays = load_boundary("diffusion")
    vae_manifest, vae_arrays = load_boundary("vae")
    controlnet = ControlNetMaisi(**cn_manifest["constructor"]).eval()
    controlnet.load_state_dict(torch_weights(cn_arrays), strict=True)
    unet = DiffusionModelUNetMaisi(**unet_manifest["constructor"]).eval()
    unet.load_state_dict(torch_weights(unet_arrays), strict=True)
    # The decoder fixture records no encoder weights.
    vae = AutoencoderKlMaisi(**vae_manifest["constructor"]).eval()
    vae.load_state_dict(torch_weights(vae_arrays), strict=False)

    label = _tiny_label()
    timesteps, next_t = set_timesteps_rflow(num_inference_steps=_STEPS, input_img_size_numel=64)
    cond_t = _torch_binarize(label)
    spacing_t = torch.tensor([list(_SPACING_MM)], dtype=torch.float32) * 100.0
    class_t = torch.tensor([1], dtype=torch.int64)
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=False,
        use_timestep_transform=True,
        spatial_dim=3,
        base_img_size_numel=32768,
    )
    scheduler.set_timesteps(num_inference_steps=_STEPS, device="cpu", input_img_size_numel=64)
    z_t = from_numpy(unet_arrays["x"].copy())
    with torch.inference_mode():
        for t, nt in zip(timesteps.tolist(), next_t.tolist(), strict=True):
            time = torch.tensor([t], dtype=torch.float32)
            down, mid = controlnet(z_t, time, cond_t, class_labels=class_t)
            velocity = unet(
                z_t,
                time,
                class_labels=class_t,
                spacing_tensor=spacing_t,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
            )
            z_t, _ = cast(RFlowStep, scheduler.step)(velocity, float(t), z_t, float(nt))
        decoded_t = vae.decode(z_t / _SCALE)
        hu_t = torch.clip(decoded_t, 0.0, 1.0) * 2000.0 - 1000.0
        hu_t = torch.where(from_numpy(label) == 0, torch.tensor(-1000.0), hu_t)

    result = _sample(label, mx)
    np.testing.assert_array_equal(result.affine, IDENTITY_AFFINE)
    for actual, expected, rtol, atol in (
        (result.latent, z_t, E2E_LATENT_RTOL, E2E_LATENT_ATOL),
        (result.decoded, decoded_t, E2E_DECODE_RTOL, E2E_DECODE_ATOL),
        (result.hu, hu_t, E2E_HU_RTOL, E2E_HU_ATOL),
    ):
        np.testing.assert_allclose(as_numpy(actual), expected.numpy(), rtol=rtol, atol=atol)
    np.testing.assert_array_equal(as_numpy(result.hu)[label == 0], -1000.0)
