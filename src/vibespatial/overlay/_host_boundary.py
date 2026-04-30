"""Explicit overlay device-to-host boundaries.

Overlay still has a few topology and allocation stages that need small host
metadata. Route them through the CUDA runtime so profile gates can see the
boundary instead of hiding it behind raw CuPy scalar reads.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def overlay_device_to_host(value, *, reason: str, dtype=None) -> np.ndarray:
    """Copy a device value to host with runtime D2H accounting."""
    if cp is not None and hasattr(value, "__cuda_array_interface__"):
        from vibespatial.cuda._runtime import get_cuda_runtime

        host = get_cuda_runtime().copy_device_to_host(value, reason=reason)
    elif cp is not None and type(value).__module__.startswith("cupy"):
        from vibespatial.cuda._runtime import get_cuda_runtime

        host = get_cuda_runtime().copy_device_to_host(value, reason=reason)
    elif hasattr(value, "get"):
        host = value.get()
    else:
        host = value
    result = np.asarray(host)
    if dtype is not None:
        return result.astype(dtype, copy=False)
    return result


def overlay_int_scalar(value, *, reason: str) -> int:
    return int(overlay_device_to_host(value, reason=reason).reshape(-1)[0])


def overlay_bool_scalar(value, *, reason: str) -> bool:
    return bool(overlay_device_to_host(value, reason=reason).reshape(-1)[0])
