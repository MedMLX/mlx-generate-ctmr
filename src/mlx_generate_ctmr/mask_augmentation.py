"""Host geometry for the pinned archive-mask inference augmentation routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, convolve1d
from scipy.special import erf

from mlx_generate_ctmr._arrays import BoolArray, FloatArray, HostArray, IntArray


@dataclass(frozen=True, slots=True)
class TumorAugmentation:
    """Per-tumor-type parameters for the pinned archive augmentation.

    ``label`` is the tumor's MAISI id. ``host`` is the label written over the
    tumor's original voxels before the deformed tumor is painted back: the
    surrounding organ, or 200 (body) for bone lesions; lung tumors use their most
    common neighbouring lobe instead. ``translation`` (voxels), ``rotation``
    (radians per axis) and ``scale`` (fractional) are the maximum random affine
    amounts, as in MONAI's ``translate_range``, ``rotate_range`` and
    ``scale_range``. ``retention`` is the minimum fraction of the original tumor
    volume that must remain inside the allowed support (the host organ, or bone for
    bone lesions) after deformation. ``elastic`` adds a random elastic displacement
    field on top of the affine transform.
    """

    label: int
    host: int
    translation: float
    rotation: tuple[float, float, float]
    scale: float
    retention: float
    elastic: bool = True


# Order matters: the first tumor label present in the mask selects the policy.
# The parameters feed MONAI's RandAffine / Rand3DElastic in the pinned source;
# tests compare the port against MONAI and against recorded source output.
TUMOR_AUGMENTATIONS = (
    TumorAugmentation(128, 200, 5, (0, 0, 0.1), 0.15, 0.8, False),
    TumorAugmentation(26, 1, 10, (np.pi / 36, np.pi / 36, np.pi / 36), 0.2, 0.8),
    TumorAugmentation(23, 28, 20, (np.pi / 36, np.pi / 36, np.pi), 0.15, 0.85),
    TumorAugmentation(24, 4, 15, (np.pi / 36, np.pi / 36, np.pi / 36), 0.1, 0.8),
    TumorAugmentation(27, 62, 5, (np.pi / 36, np.pi / 36, np.pi / 36), 0.1, 0.8),
)


def _close(mask: BoolArray) -> BoolArray:
    """Apply the mask-cleanup closing operation to a binary support."""
    structure = np.ones((5, 5, 5), dtype=bool)
    return binary_erosion(
        binary_dilation(mask, structure=structure, border_value=1),
        structure=structure,
        border_value=1,
    )


def augment_archive_mask(labels: HostArray, *, seed: int, max_attempts: int = 1000) -> IntArray:
    """Augment tumor geometry within its organ, or perform source body zoom.

    ``labels`` is a 3D MAISI label volume; the result is a new int16 array of the
    same shape. If the mask holds a tumor label from ``TUMOR_AUGMENTATIONS`` the
    tumor is randomly deformed (seeded by ``seed``) until enough of it stays inside
    its host support; otherwise the whole volume gets a small random zoom. Bone
    and colon lesions relax their retention threshold after repeated attempts;
    the colon fallback uses the original tumor intersected with its support.
    The accepted result is morphologically closed, which can extend beyond the support.

    Unlike the upstream unbounded loops, exhaustion raises explicitly. No learned
    arithmetic, Torch imports or model inference are performed by this host action.

    Raises:
        ValueError: If ``labels`` is not 3D, ``max_attempts`` is below 1, a lung tumor
            has no neighbouring lobe, or no acceptable deformation is found within
            ``max_attempts`` tries.
    """

    if labels.ndim != 3 or max_attempts < 1:
        raise ValueError("archive augmentation requires a 3D mask and positive max_attempts")
    data = np.asarray(labels, dtype=np.int16).copy()
    policy = next((p for p in TUMOR_AUGMENTATIONS if np.any(data == p.label)), None)
    if policy is None:
        return augment_body(data, seed)
    tumor = data == policy.label
    host, support = _host_support(data, tumor, policy)
    data[tumor] = host
    rng = np.random.RandomState(seed)
    affine_rng = np.random.RandomState(seed)
    tumor_size = int(np.count_nonzero(tumor))
    for attempt in range(max_attempts):
        threshold = policy.retention
        if policy.label == 128 and attempt >= 40:
            threshold = 0.75
        if policy.label == 27 and attempt >= 20:
            threshold = 0.75
            moved = tumor & support
        else:
            moved = deform_tumor_geometry(tumor, policy, rng, affine_rng) & support
        if np.count_nonzero(moved) >= tumor_size * threshold:
            data[_close(moved)] = policy.label
            return data
    raise ValueError(
        f"archive tumor augmentation exhausted {max_attempts} attempts for label {policy.label}"
    )


def _host_support(
    data: IntArray, tumor: BoolArray, policy: TumorAugmentation
) -> tuple[int, BoolArray]:
    """Compute the host-label support used by the tumor augmentation policy."""
    if policy.label == 128:
        support = (
            ((data >= 33) & (data <= 56))
            | ((data >= 63) & (data <= 97))
            | np.isin(data, (127, 114))
            | tumor
        )
        return 200, support
    if policy.label == 23:
        neighbors = data[
            binary_dilation(tumor, structure=np.ones((3, 3, 3), dtype=bool), border_value=1)
            & ~tumor
            & np.isin(data, (28, 29, 30, 31, 32))
        ]
        if not neighbors.size:
            raise ValueError("archive lung tumor has no neighboring lung lobe")
        values, counts = np.unique(neighbors, return_counts=True)
        host = int(values[np.argmax(counts)])
        return host, _close(np.isin(data, (28, 29, 30, 31, 32)) | tumor)
    return policy.host, _close((data == policy.host) | tumor)


def augment_body(data: IntArray, seed: int) -> IntArray:
    """Zoom the whole label volume by a random factor in [0.99, 1.01] (nearest, same shape).

    Mirrors the source's body augmentation: the zoomed volume is center-cropped or
    edge-padded back to the original shape.
    """
    rng = np.random.RandomState(seed)
    rng.random_sample()  # RandomizableTransform probability draw, even when prob=1.
    factor = float(rng.uniform(0.99, 1.01))
    shape = np.asarray(data.shape)
    zoom_shape = np.floor(shape * factor).astype(int)
    if np.any(zoom_shape < 1):
        raise ValueError("body zoom would produce an empty axis")
    indices = tuple(
        np.floor(np.arange(int(n)) * float(old) / int(n)).astype(int)
        for old, n in zip(shape, zoom_shape, strict=True)
    )
    zoomed = data[np.ix_(*indices)]
    start = np.maximum(zoom_shape // 2 - shape // 2, 0)
    extent = np.minimum(shape, zoom_shape)
    slices = tuple(slice(int(s), int(s + n)) for s, n in zip(start, extent, strict=True))
    padding = np.maximum(shape - zoom_shape, 0)
    pads = tuple((int(v // 2), int(v - v // 2)) for v in padding)
    return np.asarray(np.pad(zoomed[slices], pads, mode="edge"), dtype=np.int16)


def _random_affine(policy: TumorAugmentation, rng: np.random.RandomState) -> FloatArray:
    """Sample a 4x4 float32 affine as rotation @ translation @ scale.

    The z translation and scale are randomized only for elastic policies. The
    parameters are drawn twice and the first draw discarded, matching how the MONAI
    transforms consume the generator.
    """

    def draw() -> tuple[list[float], list[float], list[float]]:
        """Draw one set of random rotation, translation, and scale parameters."""
        angles = [float(rng.uniform(-v, v)) for v in policy.rotation]
        ranges = (
            policy.translation,
            policy.translation,
            0 if not policy.elastic else policy.translation,
        )
        translation = [float(rng.uniform(-v, v)) for v in ranges]
        scales = [float(rng.uniform(-policy.scale, policy.scale) + 1) for _ in range(2)]
        scales.append(
            float(rng.uniform(-policy.scale, policy.scale) + 1)
            if policy.elastic
            else float(rng.uniform(0, 0) + 1)
        )
        return angles, translation, scales

    draw()  # RandAffine/Rand3DElastic randomize and grid call each draw affine parameters.
    angles, translation, scales = draw()
    matrix = np.eye(4, dtype=np.float32)
    for axis, angle in enumerate(angles):
        sine, cosine = np.sin(np.float32(angle)), np.cos(np.float32(angle))
        rotation = np.eye(4, dtype=np.float32)
        first, second = ((1, 2), (0, 2), (0, 1))[axis]
        sign = -1 if axis == 1 else 1
        rotation[first, first] = rotation[second, second] = cosine
        rotation[first, second], rotation[second, first] = -sign * sine, sign * sine
        matrix = matrix @ rotation
    shift = np.eye(4, dtype=np.float32)
    shift[:3, 3] = translation
    return np.asarray(
        matrix @ shift @ np.diag(np.asarray((*scales, 1), dtype=np.float32)), dtype=np.float32
    )


def _smooth_offset(offset: FloatArray, sigma: float) -> FloatArray:
    """Gaussian-smooth a channel-first (C, D, H, W) offset field over its spatial axes.

    The kernel is the erf-integrated Gaussian; borders are zero-padded.
    """
    sigma32 = np.float32(sigma)
    tail = int(max(float(sigma32) * 3, 0.5) + 0.5)
    x = np.arange(-tail, tail + 1, dtype=np.float32)
    factor = np.float32(0.70710678) / sigma32
    kernel = np.maximum(
        np.float32(0.5)
        * (erf(factor * (x + np.float32(0.5))) - erf(factor * (x - np.float32(0.5)))),
        0,
    ).astype(np.float32)
    result = offset
    for axis in (1, 2, 3):
        result = convolve1d(result, kernel, axis=axis, mode="constant", cval=0)
    return result


def deform_tumor_geometry(
    tumor: BoolArray,
    policy: TumorAugmentation,
    rng: np.random.RandomState,
    affine_rng: np.random.RandomState,
) -> BoolArray:
    """Return a randomly rotated, shifted, scaled (and optionally elastic) copy of ``tumor``.

    The transform is applied about the volume center with nearest-neighbour
    sampling; voxels mapped outside the volume are dropped. ``rng`` and
    ``affine_rng`` are separate generators because the source draws the affine
    parameters from its own random state.
    """
    shape = tumor.shape
    rng.random_sample()
    displacement = None
    if policy.elastic:
        offset = rng.uniform(-1, 1, (3, *shape)).astype(np.float32)
        magnitude, sigma = float(rng.uniform(100, 200)), float(rng.uniform(5, 8))
        displacement = _smooth_offset(offset, sigma) * np.float32(magnitude)
    matrix = _random_affine(policy, affine_rng)
    output = np.zeros(shape, dtype=bool)
    x = np.arange(shape[0], dtype=np.float32)[:, None, None] - np.float32((shape[0] - 1) / 2)
    y = np.arange(shape[1], dtype=np.float32)[None, :, None] - np.float32((shape[1] - 1) / 2)
    for start in range(0, shape[2], 16):
        stop = min(start + 16, shape[2])
        z = np.arange(start, stop, dtype=np.float32)[None, None, :] - np.float32((shape[2] - 1) / 2)
        coordinates = np.broadcast_arrays(x, y, z)
        if displacement is not None:
            coordinates = tuple(
                c + displacement[i, :, :, start:stop] for i, c in enumerate(coordinates)
            )
        transformed = tuple(
            sum(matrix[axis, i] * c for i, c in enumerate(coordinates)) + matrix[axis, 3]
            for axis in range(3)
        )
        indices = tuple(
            np.rint(c.astype(np.float64) + (shape[i] - 1) / 2).astype(np.int64)
            for i, c in enumerate(transformed)
        )
        valid = np.logical_and.reduce([((v >= 0) & (v < shape[i])) for i, v in enumerate(indices)])
        clipped = tuple(np.clip(v, 0, shape[i] - 1) for i, v in enumerate(indices))
        output[:, :, start:stop] = tumor[clipped] & valid
    return output
