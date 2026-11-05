from collections.abc import Mapping
from os import PathLike

import numpy as np

def load_file(
    filename: str | PathLike[str], *, backend: str = "mmap"
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]: ...
def save_file(
    tensor_dict: Mapping[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
    filename: str | PathLike[str],
    metadata: dict[str, str] | None = None,
) -> None: ...
