from __future__ import annotations

import math
from typing import Any

import numpy as np


def get_host_init(op_name: str, dtype: Any) -> np.ndarray | None:
    """Return the host initial value for CCCL scan/reduce interop."""
    dtype = np.dtype(dtype)
    if op_name == "sum":
        return np.asarray(0, dtype=dtype)
    if op_name == "min":
        if dtype.kind == "f":
            return np.asarray(math.inf, dtype=dtype)
        return np.asarray(np.iinfo(dtype).max, dtype=dtype)
    if op_name == "max":
        if dtype.kind == "f":
            return np.asarray(-math.inf, dtype=dtype)
        return np.asarray(np.iinfo(dtype).min, dtype=dtype)
    return None
