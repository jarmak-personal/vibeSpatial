"""GPU coordinate transform stage for fused ingest+reproject.

Provides in-place CRS transforms on device-resident coordinate arrays
via vibeProj ``Transformer.transform_buffers()``, which supports
arbitrary CRS pairs with GPU-accelerated PROJ pipelines.

ADR-0002: fp64 throughout -- CRS transforms are CONSTRUCTIVE class;
           precision loss in coordinate transforms propagates to all
           downstream spatial operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibeproj import Transformer

# ---------------------------------------------------------------------------
# CRS normalization for same-CRS short-circuit
# ---------------------------------------------------------------------------

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
    """Best-effort CRS normalization for same-CRS short-circuit."""
    if hasattr(crs, "to_epsg"):
        code = crs.to_epsg()
        if code is not None:
            return f"EPSG:{code}"
        return str(crs)

    if isinstance(crs, str):
        canonical = _CRS_ALIASES.get(crs)
        if canonical is not None:
            return canonical
        return crs

    return str(crs)


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

    Delegates to vibeProj ``Transformer.transform_buffers()`` which
    supports arbitrary CRS pairs via GPU-accelerated PROJ pipelines.

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
        If coordinate arrays have mismatched lengths.

    Notes
    -----
    fp64 precision is used throughout (CONSTRUCTIVE class per ADR-0002).
    """
    src_norm = _normalize_crs(src_crs)
    dst_norm = _normalize_crs(dst_crs)
    if src_norm == dst_norm:
        return

    n = len(d_x)
    if n != len(d_y):
        raise ValueError(
            f"Coordinate arrays must have equal length: "
            f"len(d_x)={n}, len(d_y)={len(d_y)}"
        )
    if n == 0:
        return

    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    out_x = cp.empty_like(d_x)
    out_y = cp.empty_like(d_y)
    t.transform_buffers(d_x, d_y, out_x=out_x, out_y=out_y)
    d_x[:] = out_x
    d_y[:] = out_y
