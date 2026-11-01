"""Host morphology matching the pinned MAISI paired-mask cleanup path."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as _NDIMAGE

from mlx_generate_ctmr._arrays import BoolArray, HostArray, IntArray

_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
_GENERAL_SINGLE_COMPONENT_LABELS = (
    1,
    4,
    10,
    12,
    3,
    28,
    29,
    30,
    31,
    32,
    5,
    14,
    13,
    6,
    7,
    8,
    9,
    10,
    *range(33, 60),
)
_TUMOR_HOST = {23: 28, 24: 4, 26: 1, 27: 62, 128: 200}


def postprocess_mask(labels: HostArray, *, target_tumor_label: int | None) -> IntArray:
    """Clean up a generated MAISI label volume, following the pinned source without Torch.

    ``labels`` is a 3D array of MAISI ids (axes ``(X, Y, Z)``). The body support is
    eroded, reduced to its largest 26-connected component and re-dilated; selected
    organ labels undergo largest-component filtering; tumor labels other than the target are
    replaced by their host organ, and the target tumor is closed and refined
    against its host organ.
    ``target_tumor_label`` is ``None`` or one of 23, 24, 26, 27, 128. Returns an
    int16 array of the same shape. Like the pinned source, enclosed background is
    left unfilled; the CT stage conditions on this map unchanged.

    Two source behaviours are deliberately not reproduced: the hepatic-union
    threshold, whose ``>1`` test cannot hold for mutually exclusive labels, and
    restoration outside retained support, which would resurrect rejected components.
    Original hepatic vessel voxels inside retained support are restored as liver
    label 1, and airway voxels inside body support are restored as airway label 132.

    Raises:
        ValueError: If ``target_tumor_label`` is unsupported, ``labels`` is not 3D, or a
            lung tumor has no labeled neighbouring tissue.
    """

    if target_tumor_label not in {None, *_TUMOR_HOST}:
        raise ValueError(f"unsupported generated tumor label: {target_tumor_label}")
    data = np.asarray(labels, dtype=np.int16).copy()
    if data.ndim != 3:
        raise ValueError(f"paired mask postprocess expects 3D labels; got {data.shape}")
    hepatic_vessel = data == 25
    airway = data == 132

    body = _erode(data > 0)
    body = _keep_largest_binary(body)
    body_support = _dilate(body)
    data *= body_support.astype(np.int16)

    for tumor, host in _TUMOR_HOST.items():
        if tumor == target_tumor_label:
            data[_close_label(data, tumor)] = tumor
            data[_close_label(data, tumor)] = tumor
        else:
            data[data == tumor] = host

    if target_tumor_label is not None and target_tumor_label not in {26, 128}:
        data = suppress_non_largest_components(data, (target_tumor_label,), default_value=200)
    target_tumor = (
        np.zeros(data.shape, dtype=bool)
        if target_tumor_label is None
        else data == target_tumor_label
    )
    organs = (
        _GENERAL_SINGLE_COMPONENT_LABELS
        if target_tumor_label != 128
        else _GENERAL_SINGLE_COMPONENT_LABELS[:18]
    )
    before_components = data.copy()
    data = suppress_non_largest_components(
        data,
        organs,
        default_value=200,
    )
    organ_remove_mask = before_components != data

    intestinal = np.isin(data, (12, 13, 19, 62))
    intestinal = _keep_largest_binary(intestinal)
    intestinal_remove = np.logical_and(np.isin(data, (19, 62)), ~intestinal)
    data[intestinal_remove] = 200

    for organ in organs:
        data[_close_label(data, organ)] = organ

    retained_hepatic_mass = body_support
    if target_tumor_label == 26 and target_tumor.any():
        target_tumor, retained_hepatic_mass = _refine_hepatic(
            data,
            target_tumor,
            organ_remove_mask,
            intestinal_remove,
        )

    if target_tumor_label in {23, 27} and target_tumor.any():
        _refine_other_tumor(data, target_tumor_label, organ_remove_mask)
    # Restoration must not resurrect components rejected by the cleanup above.
    data[hepatic_vessel & retained_hepatic_mass & body_support] = 1
    data[airway & body_support] = 132
    if target_tumor_label is not None:
        data[target_tumor] = target_tumor_label
    return data


def _refine_hepatic(
    data: IntArray,
    target_tumor: BoolArray,
    organ_remove_mask: BoolArray,
    intestinal_remove: BoolArray,
) -> tuple[BoolArray, BoolArray]:
    """Refine the liver, vessel, and hepatic-tumor support in place."""
    data[_fill_removed(data, 1, intestinal_remove)] = 1
    data[_fill_removed(data, 1, intestinal_remove)] = 1
    data[_fill_removed(data, 3, organ_remove_mask)] = 3
    data[_fill_removed(data, 3, organ_remove_mask)] = 3
    dilated_tumor = _dilate(_dilate(data == 26))
    data[_fill_removed(data, 26, dilated_tumor & organ_remove_mask)] = 26
    # These labels are mutually exclusive in one label map, so ``> 1`` is
    # unreachable. Use ``> 0`` to build the liver-vessel-tumor union LCC.
    hepatic_mass = (
        (data == 26).astype(np.int8) + (data == 25).astype(np.int8) + (data == 1).astype(np.int8)
    ) > 0
    retained_hepatic_mass = _keep_largest_binary(hepatic_mass)
    data[hepatic_mass & ~retained_hepatic_mass] = 200
    target_tumor &= retained_hepatic_mass
    data[_close_label(data, 1)] = 1
    data[_close_label(data, 1)] = 1
    data[_close_label(data, 1)] = 1
    return target_tumor, retained_hepatic_mass


def _refine_other_tumor(data: IntArray, target: int, removed: BoolArray) -> None:
    """Restore target tumor label 23 (lung) or 27 into voxels removed by cleanup near it, in place.

    For lung tumors, a dominant neighbouring label in 28-32 is also re-closed.
    Raises ValueError for a lung tumor without labeled neighbours.
    """
    dilated = _dilate(data == target)
    if target == 23:
        neighbors = data[dilated & (data != target) & (data != 0)]
        if neighbors.size == 0:
            raise ValueError("lung tumor has no labeled neighboring tissue")
        values, counts = np.unique(neighbors, return_counts=True)
        mode = int(values[np.argmax(counts)])
        if mode in (28, 29, 30, 31, 32):
            dilated = _dilate(dilated)
            data[_fill_removed(data, mode, dilated & (data != target))] = mode
    dilated = _dilate(dilated)
    data[_fill_removed(data, target, dilated & removed)] = target
    if target == 23:
        for organ in (28, 29, 30, 31, 32):
            for _ in range(3):
                data[_close_label(data, organ)] = organ


def suppress_non_largest_components[Scalar: np.generic](
    labels: NDArray[Scalar],
    target_labels: Sequence[int],
    *,
    default_value: int,
) -> NDArray[Scalar]:
    """Replace every non-largest 26-connected component of each target label."""

    result = labels.copy()
    for target in target_labels:
        components, count = _NDIMAGE.label(
            result == target,
            structure=_CONNECTIVITY_26,
        )
        if count <= 1:
            continue
        sizes = np.bincount(components.ravel())
        largest = int(np.argmax(sizes[1:]) + 1)
        result[(components != 0) & (components != largest)] = default_value
    return result


def _keep_largest_binary(mask: BoolArray) -> BoolArray:
    """Keep the largest 26-connected component of a binary mask."""
    components, count = _NDIMAGE.label(mask, structure=_CONNECTIVITY_26)
    if count == 0:
        return np.zeros(mask.shape, dtype=bool)
    sizes = np.bincount(components.ravel())
    largest = int(np.argmax(sizes[1:]) + 1)
    return components == largest


def _dilate(mask: BoolArray) -> BoolArray:
    """Dilate a binary mask with 26-connectivity."""
    return _NDIMAGE.binary_dilation(mask, structure=_CONNECTIVITY_26, border_value=0)


def _erode(mask: BoolArray) -> BoolArray:
    """Erode a binary mask with 26-connectivity."""
    return _NDIMAGE.binary_erosion(mask, structure=_CONNECTIVITY_26, border_value=0)


def _close_label(labels: HostArray, target: int) -> BoolArray:
    """Close a target label with two dilation-erosion passes."""
    mask = labels == target
    for _ in range(2):
        mask = _erode(_dilate(mask))
    return mask


def _fill_removed(labels: HostArray, target: int, removed: BoolArray) -> BoolArray:
    """Find removed voxels within three dilations of a target label."""
    mask = labels == target
    for _ in range(3):
        mask = _dilate(mask)
    return mask & removed


__all__ = ["postprocess_mask", "suppress_non_largest_components"]
