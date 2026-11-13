"""Record MAISI model boundaries on HCC-TACE CT and annotation crops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi
from monai.networks.schedulers.rectified_flow import RFlowScheduler

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/darwin-arm64"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from fixture_parameters import load_fixture  # noqa: E402
from real_mask import real_liver_mask  # noqa: E402


def packed(label):
    """Pack MAISI integer labels into eight source-format binary channels."""
    ids = torch.from_numpy(label.astype(np.int64))
    bits = 2 ** torch.arange(8, dtype=ids.dtype)
    return ids.unsqueeze(-1).bitwise_and(bits).ne(0).squeeze(1).permute(0, 4, 1, 2, 3).float()


def construct(cls, settings, arrays, prefix=""):
    """Restore the saved seeded Torch graph without storing weights."""
    model = cls(**settings).eval()
    start = f"weight.{prefix}"
    state = {
        key.removeprefix(start): torch.from_numpy(value)
        for key, value in arrays.items()
        if key.startswith(start)
    }
    model.load_state_dict(state, strict=True)
    return model


def main() -> None:
    """Run the source encoder, conditioning and generation graphs with real inputs."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.14."):
        raise ValueError("MONAI 1.6.0 and Torch 2.14 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )[100:108, 210:218, 55:63]
    labels = real_liver_mask()[None, None]
    out = ROOT / "work/real-reference"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for name in ("vae", "mask_encode", "controlnet", "from_mask"):
        with load_fixture(FIXTURES / f"{name}.npz") as source:
            arrays = {key: source[key] for key in source.files}
        meta = json.loads(str(arrays["metadata"]))
        settings = meta["constructor"]
        with torch.inference_mode():
            if name == "vae":
                model = construct(AutoencoderKlMaisi, settings, arrays)
                image = np.clip(ct / np.float32(1000), -1, 1)[None, None]
                mu, sigma = model.encode(torch.from_numpy(image.copy()))
                arrays.update(
                    {"input.image": image, "output.mu": mu.numpy(), "output.sigma": sigma.numpy()}
                )
            elif name == "mask_encode":
                model = construct(AutoencoderKlMaisi, settings, arrays)
                image = packed(labels[:, :, :8, :8, :8])
                mu, sigma = model.encode(image)
                arrays.update(
                    {
                        "input.packed": image.numpy(),
                        "output.mu": mu.numpy(),
                        "output.sigma": sigma.numpy(),
                    }
                )
            elif name == "controlnet":
                model = construct(ControlNetMaisi, settings, arrays)
                image = packed(labels)
                down, mid = model(
                    torch.from_numpy(arrays["input.x"]),
                    torch.from_numpy(arrays["input.timesteps"]),
                    image,
                    conditioning_scale=0.75,
                    class_labels=torch.from_numpy(arrays["input.class_labels"]).long(),
                )
                full_down, full_mid = model(
                    torch.from_numpy(arrays["input.x"]),
                    torch.from_numpy(arrays["input.timesteps"]),
                    image,
                    conditioning_scale=1.0,
                    class_labels=torch.from_numpy(arrays["input.class_labels"]).long(),
                )
                boundary = {
                    "x": arrays["input.x"],
                    "timesteps": arrays["input.timesteps"],
                    "class_labels": arrays["input.class_labels"],
                    "controlnet_cond": image.numpy(),
                    "mid": full_mid.numpy(),
                }
                boundary.update(
                    {f"down_{index}": tensor.numpy() for index, tensor in enumerate(full_down)}
                )
                np.savez_compressed(out / "maisi_controlnet.npz", **boundary)
                arrays.update({"input.cond": image.numpy(), "output.mid": mid.numpy()})
                arrays.update(
                    {f"output.down_{index}": tensor.numpy() for index, tensor in enumerate(down)}
                )
            else:
                control = construct(ControlNetMaisi, settings["controlnet"], arrays, "controlnet.")
                unet = construct(DiffusionModelUNetMaisi, settings["unet"], arrays, "unet.")
                vae = construct(AutoencoderKlMaisi, settings["vae"], arrays, "vae.")
                scheduler = RFlowScheduler(
                    num_train_timesteps=1000,
                    use_discrete_timesteps=False,
                    use_timestep_transform=True,
                    spatial_dim=3,
                    base_img_size_numel=32768,
                )
                scheduler.set_timesteps(
                    num_inference_steps=2, device="cpu", input_img_size_numel=64
                )
                times = scheduler.timesteps.tolist()
                latent = torch.from_numpy(arrays["input.x"].copy())
                condition = packed(labels)
                cls = torch.from_numpy(arrays["input.class_labels"]).long()
                spacing = torch.from_numpy(arrays["input.spacing"])
                for index, timestep in enumerate(times):
                    next_t = times[index + 1] if index + 1 < len(times) else 0.0
                    t = torch.tensor([timestep], dtype=torch.float32)
                    down, mid = control(latent, t, condition, class_labels=cls)
                    velocity = unet(
                        latent,
                        t,
                        class_labels=cls,
                        spacing_tensor=spacing,
                        down_block_additional_residuals=down,
                        mid_block_additional_residual=mid,
                    )
                    latent, _ = scheduler.step(velocity, timestep, latent, next_t)
                decoded = vae.decode(latent / 1.25)
                hu = torch.clamp(decoded, 0, 1) * 2000 - 1000
                hu = torch.where(torch.from_numpy(labels) == 0, torch.tensor(-1000.0), hu)
                arrays.update(
                    {
                        "input.label": labels,
                        "output.latent": latent.numpy(),
                        "output.decoded": decoded.numpy(),
                        "output.hu": hu.numpy(),
                    }
                )
        meta["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT and annotations"
        arrays["metadata"] = np.asarray(json.dumps(meta))
        np.savez_compressed(
            out / f"{name}.npz",
            **{key: value for key, value in arrays.items() if not key.startswith("weight.")},
        )


if __name__ == "__main__":
    main()
