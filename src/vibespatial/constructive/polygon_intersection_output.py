from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime.residency import Residency


def build_device_backed_polygon_intersection_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    ring_offsets: np.ndarray,
    runtime_selection,
) -> OwnedGeometryArray:
    """Build a device-resident polygon OwnedGeometryArray from GPU outputs."""
    runtime = get_cuda_runtime()
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=None,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        runtime_history=[runtime_selection],
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None,
                )
            },
        ),
    )


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
        validity=np.zeros(row_count, dtype=bool),
        ring_offsets=np.zeros(row_count + 1, dtype=np.int32),
        runtime_selection=runtime_selection,
    )
