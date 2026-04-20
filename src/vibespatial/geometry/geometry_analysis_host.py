from __future__ import annotations

import math

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime.config import BOUNDS_SPAN_EPSILON


def family_bounds_scalar(buffer: FamilyGeometryBuffer, row_index: int) -> tuple[float, float, float, float]:
    if bool(buffer.empty_mask[row_index]):
        return (math.nan, math.nan, math.nan, math.nan)

    if buffer.family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        start = int(buffer.geometry_offsets[row_index])
        end = int(buffer.geometry_offsets[row_index + 1])
        xs = buffer.x[start:end]
        ys = buffer.y[start:end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.POLYGON:
        ring_start = int(buffer.geometry_offsets[row_index])
        ring_end = int(buffer.geometry_offsets[row_index + 1])
        coord_start = int(buffer.ring_offsets[ring_start])
        coord_end = int(buffer.ring_offsets[ring_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.MULTILINESTRING:
        part_start = int(buffer.geometry_offsets[row_index])
        part_end = int(buffer.geometry_offsets[row_index + 1])
        coord_start = int(buffer.part_offsets[part_start])
        coord_end = int(buffer.part_offsets[part_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_start = int(buffer.geometry_offsets[row_index])
        polygon_end = int(buffer.geometry_offsets[row_index + 1])
        ring_start = int(buffer.part_offsets[polygon_start])
        ring_end = int(buffer.part_offsets[polygon_end])
        coord_start = int(buffer.ring_offsets[ring_start])
        coord_end = int(buffer.ring_offsets[ring_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    raise NotImplementedError(f"unsupported family: {buffer.family.value}")


def assemble_cached_bounds(geometry_array: OwnedGeometryArray) -> np.ndarray | None:
    if not geometry_array.families:
        return np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    if any(buffer.bounds is None for buffer in geometry_array.families.values()):
        return None
    if len(geometry_array.families) == 1:
        family, buffer = next(iter(geometry_array.families.items()))
        if (
            buffer.bounds is not None
            and buffer.bounds.shape == (geometry_array.row_count, 4)
            and bool(np.all(geometry_array.validity))
            and bool(np.all(geometry_array.tags == FAMILY_TAGS[family]))
            and np.array_equal(
                geometry_array.family_row_offsets,
                np.arange(geometry_array.row_count, dtype=np.int32),
            )
        ):
            return np.array(buffer.bounds, dtype=np.float64, copy=True)
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for family, buffer in geometry_array.families.items():
        row_indexes = np.flatnonzero(geometry_array.tags == FAMILY_TAGS[family])
        if row_indexes.size == 0:
            continue
        bounds[row_indexes] = buffer.bounds
    return bounds


def slice_bounds_vectorized(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    empty_mask: np.ndarray,
) -> np.ndarray:
    result = np.full((starts.size, 4), np.nan, dtype=np.float64)
    if starts.size == 0 or x.size == 0:
        return result
    non_empty = ~empty_mask
    if not np.any(non_empty):
        return result
    row_ids = np.repeat(np.flatnonzero(non_empty), (ends - starts)[non_empty])
    min_x = np.full(starts.size, math.inf, dtype=np.float64)
    min_y = np.full(starts.size, math.inf, dtype=np.float64)
    max_x = np.full(starts.size, -math.inf, dtype=np.float64)
    max_y = np.full(starts.size, -math.inf, dtype=np.float64)
    np.minimum.at(min_x, row_ids, x)
    np.minimum.at(min_y, row_ids, y)
    np.maximum.at(max_x, row_ids, x)
    np.maximum.at(max_y, row_ids, y)
    result[non_empty, 0] = min_x[non_empty]
    result[non_empty, 1] = min_y[non_empty]
    result[non_empty, 2] = max_x[non_empty]
    result[non_empty, 3] = max_y[non_empty]
    return result


def family_bounds_vectorized(buffer: FamilyGeometryBuffer) -> np.ndarray:
    if buffer.row_count == 0:
        return np.empty((0, 4), dtype=np.float64)
    if buffer.family is GeometryFamily.POINT:
        result = np.full((buffer.row_count, 4), np.nan, dtype=np.float64)
        non_empty = ~buffer.empty_mask
        if np.any(non_empty):
            coord_indexes = buffer.geometry_offsets[:-1][non_empty]
            result[non_empty, 0] = buffer.x[coord_indexes]
            result[non_empty, 1] = buffer.y[coord_indexes]
            result[non_empty, 2] = buffer.x[coord_indexes]
            result[non_empty, 3] = buffer.y[coord_indexes]
        return result
    if buffer.family in {GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        starts = buffer.geometry_offsets[:-1].astype(np.int64, copy=False)
        ends = buffer.geometry_offsets[1:].astype(np.int64, copy=False)
        return slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    if buffer.family is GeometryFamily.POLYGON:
        starts = buffer.ring_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        ends = buffer.ring_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        return slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    if buffer.family is GeometryFamily.MULTILINESTRING:
        starts = buffer.part_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        ends = buffer.part_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        return slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_starts = buffer.part_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        polygon_ends = buffer.part_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        starts = buffer.ring_offsets[polygon_starts].astype(np.int64, copy=False)
        ends = buffer.ring_offsets[polygon_ends].astype(np.int64, copy=False)
        return slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    raise NotImplementedError(f"unsupported family: {buffer.family.value}")


def compute_geometry_bounds_cpu_scalar(geometry_array: OwnedGeometryArray) -> np.ndarray:
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for row_index in range(geometry_array.row_count):
        if not bool(geometry_array.validity[row_index]):
            continue
        family = TAG_FAMILIES[int(geometry_array.tags[row_index])]
        family_buffer = geometry_array.families[family]
        family_row = int(geometry_array.family_row_offsets[row_index])
        bounds[row_index] = np.asarray(family_bounds_scalar(family_buffer, family_row), dtype=np.float64)
    return bounds


def compute_geometry_bounds_cpu_vectorized(geometry_array: OwnedGeometryArray) -> np.ndarray:
    cached = assemble_cached_bounds(geometry_array)
    if cached is not None:
        return cached
    family_bounds = {family: family_bounds_vectorized(buffer) for family, buffer in geometry_array.families.items()}
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for family, local_bounds in family_bounds.items():
        row_indexes = np.flatnonzero(geometry_array.tags == FAMILY_TAGS[family])
        if row_indexes.size == 0:
            continue
        bounds[row_indexes] = local_bounds[geometry_array.family_row_offsets[row_indexes]]
    geometry_array.cache_bounds(bounds)
    return bounds


def compute_total_bounds_from_bounds(bounds: np.ndarray) -> tuple[float, float, float, float]:
    if np.isnan(bounds).all():
        return (math.nan, math.nan, math.nan, math.nan)
    return (
        float(np.nanmin(bounds[:, 0])),
        float(np.nanmin(bounds[:, 1])),
        float(np.nanmax(bounds[:, 2])),
        float(np.nanmax(bounds[:, 3])),
    )


def compute_offset_spans_cpu(geometry_array: OwnedGeometryArray, *, level: str = "geometry") -> dict[GeometryFamily, np.ndarray]:
    result: dict[GeometryFamily, np.ndarray] = {}
    for family, buffer in geometry_array.families.items():
        if level == "geometry":
            offsets = buffer.geometry_offsets
        elif level == "part":
            offsets = buffer.part_offsets
        elif level == "ring":
            offsets = buffer.ring_offsets
        else:
            offsets = None
        if offsets is None:
            continue
        result[family] = np.diff(offsets)
    return result


def _spread_bits(value: int) -> int:
    value &= 0x00000000FFFFFFFF
    value = (value | (value << 16)) & 0x0000FFFF0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F0F0F0F0F
    value = (value | (value << 2)) & 0x3333333333333333
    value = (value | (value << 1)) & 0x5555555555555555
    return value


def _morton_code(x: int, y: int) -> int:
    return _spread_bits(x) | (_spread_bits(y) << 1)


def compute_morton_keys_cpu(bounds: np.ndarray, total: tuple[float, float, float, float], row_count: int, *, bits: int = 16) -> np.ndarray:
    minx, miny, maxx, maxy = total
    keys = np.full(row_count, np.iinfo(np.uint64).max, dtype=np.uint64)
    if any(math.isnan(value) for value in total):
        return keys
    span_x = max(maxx - minx, BOUNDS_SPAN_EPSILON)
    span_y = max(maxy - miny, BOUNDS_SPAN_EPSILON)
    scale = (1 << bits) - 1
    for row_index, row_bounds in enumerate(bounds):
        if np.isnan(row_bounds).any():
            continue
        center_x = (float(row_bounds[0]) + float(row_bounds[2])) * 0.5
        center_y = (float(row_bounds[1]) + float(row_bounds[3])) * 0.5
        norm_x = int(round(((center_x - minx) / span_x) * scale))
        norm_y = int(round(((center_y - miny) / span_y) * scale))
        keys[row_index] = np.uint64(_morton_code(norm_x, norm_y))
    return keys
