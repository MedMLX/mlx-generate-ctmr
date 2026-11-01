"""Verified NiBabel NIfTI image/header surface used by this runner."""

from __future__ import annotations

from os import PathLike
from typing import Literal, Protocol, cast

import nibabel
import numpy as np
from nibabel.filebasedimages import ImageFileError as ImageFileError

from mlx_generate_ctmr._arrays import HostArray

type PathArg = str | PathLike[str]


class ArrayBuffer(Protocol):
    def __array__(
        self, dtype: np.dtype[np.generic] | None = None, copy: bool | None = None
    ) -> HostArray:
        """Convert an image data buffer to a host array."""
        ...


class NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel spacing from the header."""
        ...
    def get_xyzt_units(self) -> tuple[str, str]:
        """Return spatial and temporal unit names."""
        ...
    def set_xyzt_units(self, xyz: str, t: str) -> None:
        """Set spatial and temporal unit names."""
        ...
    def set_slope_inter(self, slope: float, inter: float) -> None:
        """Set stored-data scaling slope and intercept."""
        ...
    def set_data_dtype(self, dtype: type[np.generic] | np.dtype[np.generic]) -> None:
        """Set the stored image data dtype."""
        ...
    def copy(self) -> NiftiHeader:
        """Return a copy of the image header."""
        ...


class NiftiImage(Protocol):
    shape: tuple[int, ...]
    affine: np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    dataobj: ArrayBuffer | HostArray
    header: NiftiHeader

    def __init__(
        self, dataobj: HostArray, affine: HostArray, header: NiftiHeader | None = None
    ) -> None:
        """Construct a NIfTI image from data, affine, and header."""
        ...
    def get_data_dtype(self) -> np.dtype[np.generic]:
        """Return the stored image dtype."""
        ...
    def get_qform(
        self, *, coded: Literal[True]
    ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None, int]:
        """Return the qform affine and its code."""
        ...
    def get_sform(
        self, *, coded: Literal[True]
    ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None, int]:
        """Return the sform affine and its code."""
        ...
    def set_qform(self, affine: HostArray, code: int) -> None:
        """Set the qform affine and code."""
        ...
    def set_sform(self, affine: HostArray, code: int) -> None:
        """Set the sform affine and code."""
        ...


class Nibabel(Protocol):
    Nifti1Image: type[NiftiImage]

    def load(self, filename: PathArg) -> NiftiImage:
        """Load an image from a path."""
        ...
    def save(self, image: NiftiImage, filename: PathArg) -> None:
        """Write a NIfTI image to a path."""
        ...
    def as_closest_canonical(self, image: NiftiImage) -> NiftiImage:
        """Orient the image toward canonical axis order."""
        ...
    def aff2axcodes(self, affine: HostArray) -> tuple[str | None, ...]:
        """Derive orientation axis codes from an affine."""
        ...


# This module's inputs and outputs are NIfTI, so the wider loader's unrelated
# image formats are outside this runner's contract. The bound module is unchanged.
nib = cast(Nibabel, nibabel)
