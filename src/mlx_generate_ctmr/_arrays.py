"""Host array dtypes for checkpoint and NIfTI boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from mlx.core import array as Array

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type RealScalar = IntegerScalar | np.bool_ | np.float16 | np.float32 | np.float64
type RealArray = np.ndarray[tuple[int, ...], np.dtype[RealScalar]]

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]
type IntArray = np.ndarray[tuple[int, ...], np.dtype[np.int16]]
type IntegerScalar = (
    np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
)

type NumericScalar = (
    IntegerScalar | np.bool_ | np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128
)
type NumericArray = np.ndarray[tuple[int, ...], np.dtype[NumericScalar]]


# Native MLX arrays contain numeric/bool values and expose the NumPy array buffer.
as_numpy = cast("Callable[[Array], NumericArray]", np.asarray)
