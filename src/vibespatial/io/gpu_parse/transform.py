"""GPU coordinate transform stage for fused ingest+reproject.

Provides in-place CRS transforms on device-resident coordinate arrays.

Primary path: delegates to vibeProj ``Transformer.transform_buffers()``
which supports arbitrary CRS pairs via GPU-accelerated PROJ pipelines.

Fallback path: hand-rolled NVRTC kernels for WGS84 (EPSG:4326) <->
Web Mercator (EPSG:3857) when vibeProj is not installed.

ADR-0033: Tier 1 NVRTC -- transcendental functions (log, tan, atan, exp)
           make this geometry-specific compute, not simple element-wise.
ADR-0002: fp64 throughout -- CRS transforms are CONSTRUCTIVE class;
           precision loss in coordinate transforms propagates to all
           downstream spatial operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

# ---------------------------------------------------------------------------
# vibeProj availability probe
# ---------------------------------------------------------------------------

try:
    from vibeproj import Transformer

    _HAS_VIBEPROJ = True
except ImportError:
    _HAS_VIBEPROJ = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NVRTC kernel source -- WGS84 <-> Web Mercator transforms (Tier 1)
# Retained as fallback when vibeProj is not installed.
# ---------------------------------------------------------------------------

# Semi-major axis * pi / 180 = 20037508.342789244 / 180
# Full semi-circumference = 20037508.342789244
_TRANSFORM_KERNEL_SOURCE = r"""
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* WGS84 (EPSG:4326) -> Web Mercator (EPSG:3857), in-place.
   x = longitude (degrees) -> meters
   y = latitude  (degrees) -> meters */
extern "C" __global__ void wgs84_to_mercator(
    double* __restrict__ x,
    double* __restrict__ y,
    const int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double lon = x[i];
        const double lat = y[i];

        /* x_merc = lon * 20037508.342789244 / 180.0 */
        x[i] = lon * (20037508.342789244 / 180.0);

        /* y_merc = log(tan((90 + lat) * PI / 360)) * 20037508.342789244 / PI
           Rewrite: (90+lat)*PI/360 = (PI/4) + (lat*PI/360)
                  = (PI/4) + (lat_rad/2)
           Use the identity: log(tan(PI/4 + x/2)) = atanh(sin(x))
           but the direct formula is fine for fp64 precision. */
        const double lat_rad = lat * (M_PI / 180.0);
        y[i] = log(tan(M_PI * 0.25 + lat_rad * 0.5)) * (20037508.342789244 / M_PI);
    }
}

/* Web Mercator (EPSG:3857) -> WGS84 (EPSG:4326), in-place.
   x = meters -> longitude (degrees)
   y = meters -> latitude  (degrees) */
extern "C" __global__ void mercator_to_wgs84(
    double* __restrict__ x,
    double* __restrict__ y,
    const int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        /* lon = x * 180 / 20037508.342789244 */
        x[i] = x[i] * (180.0 / 20037508.342789244);

        /* lat = atan(exp(y * PI / 20037508.342789244)) * 360 / PI - 90 */
        const double y_rad = y[i] * (M_PI / 20037508.342789244);
        y[i] = atan(exp(y_rad)) * (360.0 / M_PI) - 90.0;
    }
}
"""

_TRANSFORM_KERNEL_NAMES = ("wgs84_to_mercator", "mercator_to_wgs84")

# ---------------------------------------------------------------------------
# NVRTC warmup (ADR-0034 Level 2) -- only needed for fallback path
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("crs-transform", _TRANSFORM_KERNEL_SOURCE, _TRANSFORM_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helper (fallback path)
# ---------------------------------------------------------------------------

def _transform_kernels():
    from vibespatial.cuda._runtime import compile_kernel_group

    return compile_kernel_group(
        "crs-transform", _TRANSFORM_KERNEL_SOURCE, _TRANSFORM_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# CRS normalization
# ---------------------------------------------------------------------------

# Canonical EPSG code strings for the NVRTC fallback path.
_NVRTC_CRS_ALIASES: dict[str, str] = {
    "EPSG:4326": "EPSG:4326",
    "epsg:4326": "EPSG:4326",
    "WGS84": "EPSG:4326",
    "wgs84": "EPSG:4326",
    "OGC:CRS84": "EPSG:4326",
    "CRS84": "EPSG:4326",
    "EPSG:3857": "EPSG:3857",
    "epsg:3857": "EPSG:3857",
    "EPSG:900913": "EPSG:3857",
    "EPSG:3785": "EPSG:3857",
}


def _normalize_crs_nvrtc(crs: str) -> str:
    """Normalize a CRS string for the NVRTC fallback path.

    Only recognizes EPSG:4326 and EPSG:3857 (and common aliases).
    """
    # Handle pyproj CRS objects
    if hasattr(crs, "to_epsg"):
        code = crs.to_epsg()
        if code is not None:
            key = f"EPSG:{code}"
            if key in _NVRTC_CRS_ALIASES:
                return _NVRTC_CRS_ALIASES[key]
            return key
        # Fallback to string representation
        crs = str(crs)

    if isinstance(crs, str):
        canonical = _NVRTC_CRS_ALIASES.get(crs)
        if canonical is not None:
            return canonical

    raise ValueError(
        f"Unsupported CRS for NVRTC fallback: {crs!r}. "
        f"Install vibeProj for arbitrary CRS support, or use "
        f"EPSG:4326 and EPSG:3857."
    )


def _normalize_crs_for_equality(crs: str) -> str | None:
    """Best-effort CRS normalization for same-CRS short-circuit.

    Returns a canonical EPSG string if the CRS can be recognized, or
    the original string for pass-through to vibeProj.
    """
    if hasattr(crs, "to_epsg"):
        code = crs.to_epsg()
        if code is not None:
            return f"EPSG:{code}"
        return str(crs)

    if isinstance(crs, str):
        canonical = _NVRTC_CRS_ALIASES.get(crs)
        if canonical is not None:
            return canonical
        return crs

    return str(crs)


# ---------------------------------------------------------------------------
# NVRTC fallback implementation
# ---------------------------------------------------------------------------

def _transform_nvrtc(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    src_crs: str,
    dst_crs: str,
) -> None:
    """NVRTC fallback: WGS84 <-> Web Mercator only."""
    from vibespatial.cuda._runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
    )

    src = _normalize_crs_nvrtc(src_crs)
    dst = _normalize_crs_nvrtc(dst_crs)

    if src == dst:
        return

    n = len(d_x)
    if n == 0:
        return

    if src == "EPSG:4326" and dst == "EPSG:3857":
        kernel_name = "wgs84_to_mercator"
    elif src == "EPSG:3857" and dst == "EPSG:4326":
        kernel_name = "mercator_to_wgs84"
    else:
        raise ValueError(
            f"Unsupported CRS transform for NVRTC fallback: {src} -> {dst}. "
            f"Install vibeProj for arbitrary CRS support, or use "
            f"EPSG:4326 <-> EPSG:3857."
        )

    runtime = get_cuda_runtime()
    kernels = _transform_kernels()
    kernel = kernels[kernel_name]
    ptr = runtime.pointer

    grid, block = runtime.launch_config(kernel, n)
    params = (
        (ptr(d_x), ptr(d_y), np.int32(n)),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    runtime.launch(kernel, grid=grid, block=block, params=params)


# ---------------------------------------------------------------------------
# vibeProj implementation
# ---------------------------------------------------------------------------

def _transform_vibeproj(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    src_crs: str,
    dst_crs: str,
) -> None:
    """vibeProj path: supports arbitrary CRS pairs on device."""
    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    # vibeProj's transform_buffers writes to separate output arrays.
    # Allocate temporaries, then copy back for in-place semantics.
    out_x = cp.empty_like(d_x)
    out_y = cp.empty_like(d_y)
    t.transform_buffers(d_x, d_y, out_x=out_x, out_y=out_y)
    d_x[:] = out_x
    d_y[:] = out_y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_coordinates_inplace(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    src_crs: str,
    dst_crs: str,
) -> None:
    """Transform device coordinate arrays between CRS in-place.

    Parameters
    ----------
    d_x : cp.ndarray
        Float64 device array of x-coordinates (longitude or easting).
        Modified in-place.
    d_y : cp.ndarray
        Float64 device array of y-coordinates (latitude or northing).
        Modified in-place.
    src_crs : str
        Source CRS identifier (e.g. ``"EPSG:4326"``).
    dst_crs : str
        Destination CRS identifier (e.g. ``"EPSG:3857"``).

    Raises
    ------
    ValueError
        If the CRS pair is not supported (when vibeProj is not installed
        and the pair is not WGS84 <-> Web Mercator) or arrays are mismatched.

    Notes
    -----
    Primary path delegates to vibeProj which supports arbitrary CRS pairs.
    Fallback path uses hand-rolled NVRTC kernels for EPSG:4326 <-> EPSG:3857.
    fp64 precision is used throughout (CONSTRUCTIVE class per ADR-0002).
    """
    # Quick same-CRS short-circuit before any validation.
    src_norm = _normalize_crs_for_equality(src_crs)
    dst_norm = _normalize_crs_for_equality(dst_crs)
    if src_norm == dst_norm:
        return  # No-op: same CRS

    n = len(d_x)
    if n != len(d_y):
        raise ValueError(
            f"Coordinate arrays must have equal length: "
            f"len(d_x)={n}, len(d_y)={len(d_y)}"
        )
    if n == 0:
        return  # Nothing to transform

    # Primary path: vibeProj (arbitrary CRS pairs)
    if _HAS_VIBEPROJ:
        _transform_vibeproj(d_x, d_y, src_crs, dst_crs)
        return

    # Fallback: NVRTC kernels (WGS84 <-> Web Mercator only)
    logger.debug(
        "vibeProj not available; falling back to NVRTC kernels for %s -> %s",
        src_crs, dst_crs,
    )
    _transform_nvrtc(d_x, d_y, src_crs, dst_crs)
