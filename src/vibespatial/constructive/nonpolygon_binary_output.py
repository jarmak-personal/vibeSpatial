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


def build_device_backed_point_output(
    device_x,
    device_y,
    *,
    row_count: int,
    validity: np.ndarray,
    geometry_offsets: np.ndarray,
) -> OwnedGeometryArray:
    """Build a device-resident Point OwnedGeometryArray."""
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=get_geometry_buffer_schema(GeometryFamily.POINT),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POINT: point_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POINT,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
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
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.LINESTRING], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    ls_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: ls_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.LINESTRING,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
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
    runtime = get_cuda_runtime()
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.MULTIPOINT], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    empty_mask = ~validity

    mp_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.MULTIPOINT,
        schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOINT),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.MULTIPOINT: mp_buffer},
        residency=Residency.DEVICE,
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.MULTIPOINT,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                )
            },
        ),
    )


def build_point_result_from_source(
    points: OwnedGeometryArray,
    new_validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray sharing Point buffers with new validity."""
    new_device_state = None
    if points.device_state is not None:
        runtime = get_cuda_runtime()
        new_device_state = OwnedGeometryDeviceState(
            validity=runtime.from_host(new_validity),
            tags=points.device_state.tags,
            family_row_offsets=points.device_state.family_row_offsets,
            families=dict(points.device_state.families),
        )

    return OwnedGeometryArray(
        validity=new_validity,
        tags=points.tags.copy(),
        family_row_offsets=points.family_row_offsets.copy(),
        families=dict(points.families),
        residency=points.residency,
        device_state=new_device_state,
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
