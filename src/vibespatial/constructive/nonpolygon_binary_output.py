from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.cuda._runtime import DeviceArray, get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    build_device_resident_owned,
)
from vibespatial.runtime.residency import Residency


def _require_cupy() -> None:
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        raise RuntimeError("CuPy is not installed; GPU constructive output builders are unavailable")


def build_device_backed_point_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident Point OwnedGeometryArray."""
    _require_cupy()
    runtime = get_cuda_runtime()
    d_validity = runtime.from_host(validity)
    return build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=device_x,
                y=device_y,
                geometry_offsets=runtime.from_host(geometry_offsets),
                empty_mask=~cp.asarray(d_validity),
            ),
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def build_device_backed_linestring_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident LineString OwnedGeometryArray."""
    _require_cupy()
    runtime = get_cuda_runtime()
    d_validity = runtime.from_host(validity)
    return build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=device_x,
                y=device_y,
                geometry_offsets=runtime.from_host(geometry_offsets),
                empty_mask=~cp.asarray(d_validity),
            ),
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def build_device_backed_multipoint_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident MultiPoint OwnedGeometryArray."""
    _require_cupy()
    runtime = get_cuda_runtime()
    d_validity = runtime.from_host(validity)
    return build_device_resident_owned(
        device_families={
            GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTIPOINT,
                x=device_x,
                y=device_y,
                geometry_offsets=runtime.from_host(geometry_offsets),
                empty_mask=~cp.asarray(d_validity),
            ),
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.MULTIPOINT], dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def build_device_backed_multilinestring_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
    part_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident MultiLineString OwnedGeometryArray."""
    _require_cupy()
    runtime = get_cuda_runtime()
    d_validity = runtime.from_host(validity)
    return build_device_resident_owned(
        device_families={
            GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTILINESTRING,
                x=device_x,
                y=device_y,
                geometry_offsets=runtime.from_host(geometry_offsets),
                part_offsets=runtime.from_host(part_offsets),
                empty_mask=~cp.asarray(d_validity),
            ),
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.MULTILINESTRING], dtype=cp.int8),
        validity=d_validity,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def build_point_result_from_source(
    points: OwnedGeometryArray,
    new_validity: np.ndarray | None,
    *,
    d_new_validity: DeviceArray | None = None,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray sharing Point buffers with new validity."""
    new_device_state = None
    host_validity = new_validity
    host_tags = points.tags.copy()
    host_family_row_offsets = points.family_row_offsets.copy()
    if points.device_state is not None:
        runtime = get_cuda_runtime()
        if d_new_validity is not None:
            d_validity = d_new_validity
        elif new_validity is not None:
            d_validity = runtime.from_host(new_validity)
        else:
            raise ValueError(
                "Either new_validity or d_new_validity must be provided"
            )
        new_device_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
            families=dict(points.device_state.families),
        )
        host_validity = None
        host_tags = None
        host_family_row_offsets = None

    return OwnedGeometryArray(
        validity=host_validity,
        tags=host_tags,
        family_row_offsets=host_family_row_offsets,
        families=dict(points.families),
        residency=(
            Residency.DEVICE
            if new_device_state is not None
            else points.residency
        ),
        device_state=new_device_state,
        _row_count=points.row_count,
    )


def host_prefix_offsets(counts: np.ndarray) -> np.ndarray:
    """Build exclusive offsets from per-row counts on host."""
    offsets = np.empty(counts.size + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def empty_linestring_output(
    row_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return host metadata for an all-empty linestring output."""
    return np.zeros(row_count, dtype=bool), np.zeros(row_count + 1, dtype=np.int32)
