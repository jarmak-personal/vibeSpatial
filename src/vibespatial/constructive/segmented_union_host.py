from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.geometry.owned import OwnedGeometryArray


def normalize_group_offsets(group_offsets: Any) -> np.ndarray:
    """Normalize group offsets to a host int64 ndarray."""
    return np.asarray(group_offsets, dtype=np.int64)


def group_has_only_polygon_families(
    geometries: OwnedGeometryArray,
    polygon_tags: set[int],
) -> bool:
    """Return True when all valid rows are polygon-family geometries."""
    valid_tags = np.isin(geometries.tags[geometries.validity], list(polygon_tags))
    return bool(np.all(valid_tags))


def singleton_indices(index: int) -> np.ndarray:
    """Return a 1-row host index vector for OwnedGeometryArray.take()."""
    return np.array([index], dtype=np.intp)


def group_indices(start: int, end: int) -> np.ndarray:
    """Return the host index vector for a contiguous group slice."""
    return np.arange(start, end, dtype=np.intp)


def valid_row_indices(owned: OwnedGeometryArray) -> np.ndarray:
    """Return indices of valid rows in an OwnedGeometryArray."""
    return np.flatnonzero(owned.validity)


def concat_owned_arrays(
    arrays: list[OwnedGeometryArray],
) -> OwnedGeometryArray:
    """Concatenate OwnedGeometryArrays at the buffer level (zero-copy)."""
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        NULL_TAG,
        DiagnosticKind,
    )
    from vibespatial.geometry.owned import OwnedGeometryArray as _OGA

    TAG_FAMILIES_LOCAL = {v: k for k, v in FAMILY_TAGS.items()}

    if not arrays:
        return _OGA(
            validity=np.array([], dtype=bool),
            tags=np.array([], dtype=np.int8),
            family_row_offsets=np.array([], dtype=np.int32),
            families={},
        )

    if len(arrays) == 1:
        return arrays[0]

    for arr in arrays:
        arr._ensure_host_state()

    new_validity = np.concatenate([o.validity for o in arrays])
    new_tags = np.concatenate([o.tags for o in arrays])

    all_families = {}
    for owned in arrays:
        for family, buf in owned.families.items():
            all_families.setdefault(family, []).append(buf)

    new_families = {}
    for family, bufs in all_families.items():
        new_families[family] = concat_family_buffers(family, bufs)

    new_family_row_offsets = np.full(new_validity.size, -1, dtype=np.int32)
    family_cursor = {f: 0 for f in all_families}
    global_offset = 0
    for owned in arrays:
        n = owned.row_count
        for i in range(n):
            global_idx = global_offset + i
            tag = new_tags[global_idx]
            if tag == NULL_TAG:
                continue
            family = TAG_FAMILIES_LOCAL[int(tag)]
            new_family_row_offsets[global_idx] = (
                family_cursor[family] + owned.family_row_offsets[i]
            )
        for family in all_families:
            if family in owned.families:
                family_cursor[family] += owned.families[family].row_count
        global_offset += n

    result = _OGA(
        validity=new_validity,
        tags=new_tags,
        family_row_offsets=new_family_row_offsets,
        families=new_families,
    )
    result._record(
        DiagnosticKind.CREATED,
        f"segmented_union: buffer-level concat of {len(arrays)} arrays",
        visible=False,
    )
    return result


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
