from __future__ import annotations

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)


def _require_cupy() -> None:
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is not installed; GPU polygon-intersection output builders are unavailable")


def build_device_backed_polygon_intersection_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity,
    ring_offsets,
    runtime_selection,
) -> OwnedGeometryArray:
    """Build a device-resident polygon OwnedGeometryArray from GPU outputs."""
    _require_cupy()
    runtime = get_cuda_runtime()
    d_validity = runtime.from_host(validity)
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=device_x,
                y=device_y,
                geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
                empty_mask=~cp.asarray(d_validity),
                ring_offsets=runtime.from_host(ring_offsets),
                bounds=None,
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    result.runtime_history.append(runtime_selection)
    return result


def build_empty_device_backed_polygon_intersection_output(
    row_count: int,
    runtime_selection,
) -> OwnedGeometryArray:
    """Build an all-empty device-resident polygon result."""
    import cupy as cp

    runtime = get_cuda_runtime()
    return build_device_backed_polygon_intersection_output(
        runtime.allocate((0,), cp.float64),
        runtime.allocate((0,), cp.float64),
        row_count=row_count,
        validity=cp.zeros(row_count, dtype=cp.bool_),
        ring_offsets=cp.zeros(row_count + 1, dtype=cp.int32),
        runtime_selection=runtime_selection,
    )
