"""GPU coordinate transform stage for fused ingest+reproject.

Provides in-place CRS transforms on device-resident coordinate arrays.
Currently supports WGS84 (EPSG:4326) <-> Web Mercator (EPSG:3857).

ADR-0033: Tier 1 NVRTC -- transcendental functions (log, tan, atan, exp)
           make this geometry-specific compute, not simple element-wise.
ADR-0002: fp64 throughout -- CRS transforms are CONSTRUCTIVE class;
           precision loss in coordinate transforms propagates to all
           downstream spatial operations.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)

# ---------------------------------------------------------------------------
# NVRTC kernel source -- WGS84 <-> Web Mercator transforms (Tier 1)
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
# NVRTC warmup (ADR-0034 Level 2)
# ---------------------------------------------------------------------------
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("crs-transform", _TRANSFORM_KERNEL_SOURCE, _TRANSFORM_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helper
# ---------------------------------------------------------------------------

def _transform_kernels():
    return compile_kernel_group(
        "crs-transform", _TRANSFORM_KERNEL_SOURCE, _TRANSFORM_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# CRS normalization
# ---------------------------------------------------------------------------

# Canonical EPSG code strings for supported CRS.
_CRS_ALIASES: dict[str, str] = {
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


def _normalize_crs(crs: str) -> str:
    """Normalize a CRS string to a canonical EPSG code.

    Supports string aliases and pyproj CRS objects (via .to_epsg()).
    """
    # Handle pyproj CRS objects
    if hasattr(crs, "to_epsg"):
        code = crs.to_epsg()
        if code is not None:
            key = f"EPSG:{code}"
            if key in _CRS_ALIASES:
                return _CRS_ALIASES[key]
            return key
        # Fallback to string representation
        crs = str(crs)

    if isinstance(crs, str):
        canonical = _CRS_ALIASES.get(crs)
        if canonical is not None:
            return canonical

    raise ValueError(
        f"Unsupported CRS: {crs!r}. "
        f"GPU coordinate transform currently supports EPSG:4326 and EPSG:3857."
    )


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
        If the CRS pair is not supported or arrays are mismatched.

    Notes
    -----
    This is a device-only operation: no host-device transfers occur.
    fp64 precision is used throughout (CONSTRUCTIVE class per ADR-0002).
    """
    src = _normalize_crs(src_crs)
    dst = _normalize_crs(dst_crs)

    if src == dst:
        return  # No-op: same CRS

    n = len(d_x)
    if n != len(d_y):
        raise ValueError(
            f"Coordinate arrays must have equal length: "
            f"len(d_x)={n}, len(d_y)={len(d_y)}"
        )
    if n == 0:
        return  # Nothing to transform

    # Determine which kernel to launch
    if src == "EPSG:4326" and dst == "EPSG:3857":
        kernel_name = "wgs84_to_mercator"
    elif src == "EPSG:3857" and dst == "EPSG:4326":
        kernel_name = "mercator_to_wgs84"
    else:
        raise ValueError(
            f"Unsupported CRS transform: {src} -> {dst}. "
            f"GPU coordinate transform currently supports "
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
