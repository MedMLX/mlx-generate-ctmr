"""Independent installed MONAI oracle for source archive inference geometry."""

from __future__ import annotations

import numpy as np
import pytest
from _torch import from_numpy
from monai.transforms.spatial.array import Rand3DElastic, RandAffine, RandZoom

from mlx_generate_ctmr.mask_augmentation import (
    TUMOR_AUGMENTATIONS,
    augment_archive_mask,
    augment_body,
    deform_tumor_geometry,
)


@pytest.mark.parametrize("seed", [0, 7, 1337])
def test_body_zoom_matches_source(seed: int) -> None:
    """Compare body zoom with source augmentation behavior."""
    data = (np.arange(21 * 23 * 25).reshape(21, 23, 25) % 133).astype(np.int16)
    transform = RandZoom(min_zoom=0.99, max_zoom=1.01, mode="nearest", align_corners=None, prob=1.0)
    transform.set_random_state(seed=seed)
    reference = transform(from_numpy(data[None])).numpy()[0].astype(np.int16)
    np.testing.assert_array_equal(augment_body(data, seed), reference)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("seed", [0, 7])
def test_tumordeform_tumor_geometryation_matches_monai(index: int, seed: int) -> None:
    """Compare tumor deformation geometry with MONAI."""
    policy = TUMOR_AUGMENTATIONS[index]
    shape = (21, 23, 25)
    mask = np.zeros(shape, dtype=bool)
    mask[6:16, 6:16, 6:16] = True
    translation = (
        policy.translation,
        policy.translation,
        policy.translation if policy.elastic else 0,
    )
    scale = (policy.scale, policy.scale, policy.scale if policy.elastic else 0)
    if policy.elastic:
        transform = Rand3DElastic(
            mode="nearest",
            prob=1.0,
            sigma_range=(5, 8),
            magnitude_range=(100, 200),
            translate_range=translation,
            rotate_range=policy.rotation,
            scale_range=scale,
            padding_mode="zeros",
        )
    else:
        transform = RandAffine(
            mode="nearest",
            prob=1.0,
            translate_range=translation,
            rotate_range=policy.rotation,
            scale_range=scale,
            padding_mode="zeros",
        )
    transform.set_random_state(seed=seed)
    reference = transform(from_numpy(mask[None]), spatial_size=shape).numpy()[0].astype(bool)
    result = deform_tumor_geometry(
        mask, policy, np.random.RandomState(seed), np.random.RandomState(seed)
    )
    np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("tumor", [0, 23, 24, 26, 27, 128])
def test_composed_augmentation_matches_pinned_source(tumor: int) -> None:
    """Compare composed mask augmentation with the pinned source recording."""
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "archive_augmentation_source.npz"
    with np.load(path, allow_pickle=False) as fixture:
        assert fixture["source_revision"].tolist() == ["5cb04e82fed71f2fe64a2617ab695be1f5a37fea"]
        actual = augment_archive_mask(fixture[f"input_{tumor}"], seed=4)
        np.testing.assert_array_equal(actual, fixture[f"output_{tumor}"])
