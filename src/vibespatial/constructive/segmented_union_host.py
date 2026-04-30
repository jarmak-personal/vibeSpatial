from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.geometry.owned import TAG_FAMILIES, OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None


def normalize_group_offsets(group_offsets: Any) -> np.ndarray:
    """Normalize group offsets to a host int64 ndarray."""
    return np.asarray(group_offsets, dtype=np.int64)


def group_has_only_polygon_families(
    geometries: OwnedGeometryArray,
    polygon_tags: set[int],
) -> bool:
    """Return True when all valid rows are polygon-family geometries."""
    if cp is not None and getattr(geometries, "device_state", None) is not None:
        from vibespatial.cuda._runtime import get_cuda_runtime

        state = geometries._ensure_device_state()
        polygon_families = {
            TAG_FAMILIES[tag]
            for tag in polygon_tags
            if tag in TAG_FAMILIES
        }
        if set(state.families).issubset(polygon_families):
            return True
        d_validity = cp.asarray(state.validity).astype(cp.bool_, copy=False)
        d_tags = cp.asarray(state.tags).astype(cp.int8, copy=False)
        d_polygon = cp.zeros(int(geometries.row_count), dtype=cp.bool_)
        for tag in polygon_tags:
            d_polygon |= d_tags == np.int8(tag)
        d_allowed = cp.asarray(cp.all((~d_validity) | d_polygon), dtype=cp.bool_).reshape(1)
        allowed = get_cuda_runtime().copy_device_to_host(
            d_allowed,
            reason="segmented union polygon-family admission scalar fence",
        )
        return bool(allowed[0])

    valid_tags = np.isin(geometries.tags[geometries.validity], list(polygon_tags))
    return bool(np.all(valid_tags))


def singleton_indices(index: int) -> np.ndarray:
    """Return a 1-row host index vector for OwnedGeometryArray.take()."""
    return np.array([index], dtype=np.intp)


def group_indices(start: int, end: int) -> np.ndarray:
    """Return the host index vector for a contiguous group slice."""
    return np.arange(start, end, dtype=np.intp)


def valid_row_indices(owned: OwnedGeometryArray):
    """Return indices of valid rows in an OwnedGeometryArray."""
    if cp is not None and getattr(owned, "device_state", None) is not None:
        return cp.flatnonzero(cp.asarray(owned._ensure_device_state().validity)).astype(
            cp.int64,
            copy=False,
        )
    return np.flatnonzero(owned.validity)


def concat_owned_arrays(
    arrays: list[OwnedGeometryArray],
) -> OwnedGeometryArray:
    """Concatenate OwnedGeometryArrays without forcing device results to host."""
    from vibespatial.geometry.owned import OwnedGeometryArray as _OGA

    if not arrays:
        return _OGA(
            validity=np.array([], dtype=bool),
            tags=np.array([], dtype=np.int8),
            family_row_offsets=np.array([], dtype=np.int32),
            families={},
        )

    return _OGA.concat(arrays)


def concat_family_buffers(
    family,
    buffers,
):
    """Concatenate FamilyGeometryBuffers for a single family."""
    from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
    from vibespatial.geometry.owned import FamilyGeometryBuffer as _FGB

    if len(buffers) == 1:
        return buffers[0]

    schema = get_geometry_buffer_schema(family)
    total_rows = sum(b.row_count for b in buffers)

    all_x = [b.x for b in buffers]
    all_y = [b.y for b in buffers]
    new_x = np.concatenate(all_x) if any(a.size for a in all_x) else np.empty(0, dtype=np.float64)
    new_y = np.concatenate(all_y) if any(a.size for a in all_y) else np.empty(0, dtype=np.float64)
    new_empty_mask = np.concatenate([b.empty_mask for b in buffers])

    if all(b.bounds is not None for b in buffers):
        new_bounds = np.concatenate([b.bounds for b in buffers])
    else:
        new_bounds = None

    coord_cursor = 0
    geom_offset_parts: list[np.ndarray] = []
    for b in buffers:
        shifted = b.geometry_offsets[:-1] + coord_cursor
        geom_offset_parts.append(shifted)
        coord_cursor += int(b.geometry_offsets[-1])
    geom_offset_parts.append(np.array([coord_cursor], dtype=np.int32))
    new_geometry_offsets = np.concatenate(geom_offset_parts)

    new_part_offsets = None
    new_ring_offsets = None

    if family is GeometryFamily.POLYGON:
        ring_cursor = 0
        ring_parts: list[np.ndarray] = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts)

    elif family is GeometryFamily.MULTILINESTRING:
        part_cursor = 0
        part_parts: list[np.ndarray] = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts)

    elif family is GeometryFamily.MULTIPOLYGON:
        part_cursor = 0
        part_parts_list: list[np.ndarray] = []
        for b in buffers:
            shifted = b.part_offsets[:-1] + part_cursor
            part_parts_list.append(shifted)
            part_cursor += int(b.part_offsets[-1])
        part_parts_list.append(np.array([part_cursor], dtype=np.int32))
        new_part_offsets = np.concatenate(part_parts_list)

        ring_cursor = 0
        ring_parts_list: list[np.ndarray] = []
        for b in buffers:
            shifted = b.ring_offsets[:-1] + ring_cursor
            ring_parts_list.append(shifted)
            ring_cursor += int(b.ring_offsets[-1])
        ring_parts_list.append(np.array([ring_cursor], dtype=np.int32))
        new_ring_offsets = np.concatenate(ring_parts_list)

    return _FGB(
        family=family,
        schema=schema,
        row_count=total_rows,
        x=new_x,
        y=new_y,
        geometry_offsets=new_geometry_offsets,
        empty_mask=new_empty_mask,
        part_offsets=new_part_offsets,
        ring_offsets=new_ring_offsets,
        bounds=new_bounds,
    )
