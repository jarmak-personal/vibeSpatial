from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    NULL_TAG,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime.config import SPATIAL_EPSILON

_EPSILON = SPATIAL_EPSILON
_POINT_TYPE_ID = 0
_LINESTRING_TYPE_ID = 1
_POLYGON_TYPE_ID = 3


def empty_index_array() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


def row_index_array(row_count: int) -> np.ndarray:
    return np.arange(row_count, dtype=np.int32)


def normalize_distances(distance, row_count: int) -> np.ndarray:
    if np.isscalar(distance):
        return np.full(row_count, float(distance), dtype=np.float64)
    values = np.asarray(distance, dtype=np.float64)
    if values.shape != (row_count,):
        raise ValueError(f"distance must be a scalar or length-{row_count} array")
    return values


def _point_ring(point, radius: float, quad_segs: int) -> Polygon:
    segments = max(int(quad_segs), 1) * 4
    angles = np.linspace(0.0, -2.0 * np.pi, num=segments, endpoint=False, dtype=np.float64)
    x = point.x + radius * np.cos(angles)
    y = point.y + radius * np.sin(angles)
    x[np.abs(x) <= _EPSILON] = 0.0
    y[np.abs(y) <= _EPSILON] = 0.0
    coords = np.column_stack((x, y))
    return Polygon(coords)


def point_buffer_owned_cpu(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometries = np.asarray(values, dtype=object)
    distances = normalize_distances(distance, len(geometries))
    row_count = len(geometries)
    result = np.empty(row_count, dtype=object)

    non_null_mask = np.array([g is not None for g in geometries], dtype=bool)
    type_ids = np.full(row_count, -1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    if np.any(non_null_mask):
        non_null_geoms = geometries[non_null_mask]
        type_ids[non_null_mask] = shapely.get_type_id(non_null_geoms)
        empty_mask[non_null_mask] = shapely.is_empty(non_null_geoms)

    point_mask = non_null_mask & ~empty_mask & (type_ids == _POINT_TYPE_ID) & (distances > 0.0)
    empty_rows_mask = non_null_mask & empty_mask
    fallback_mask = non_null_mask & ~empty_mask & ~point_mask

    result[~non_null_mask] = None

    if np.any(empty_rows_mask):
        empty_idx = np.flatnonzero(empty_rows_mask)
        result[empty_idx] = shapely.buffer(geometries[empty_idx], distances[empty_idx], quad_segs=quad_segs)

    point_rows = np.flatnonzero(point_mask)
    if point_rows.size > 0:
        point_geoms = geometries[point_rows]
        point_radii = distances[point_rows]
        px = shapely.get_x(point_geoms)
        py = shapely.get_y(point_geoms)
        n_arc = max(int(quad_segs), 1) * 4
        verts_per_ring = n_arc + 1
        angles = np.linspace(0.0, -2.0 * np.pi, num=n_arc, endpoint=False, dtype=np.float64)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        all_x = px[:, None] + point_radii[:, None] * cos_a[None, :]
        all_y = py[:, None] + point_radii[:, None] * sin_a[None, :]
        all_x[np.abs(all_x) <= _EPSILON] = 0.0
        all_y[np.abs(all_y) <= _EPSILON] = 0.0
        all_x = np.column_stack((all_x, all_x[:, 0]))
        all_y = np.column_stack((all_y, all_y[:, 0]))
        n_points = point_rows.size
        flat_coords = np.empty((n_points * verts_per_ring, 2), dtype=np.float64)
        flat_coords[:, 0] = all_x.ravel()
        flat_coords[:, 1] = all_y.ravel()
        ring_indices = np.repeat(np.arange(n_points, dtype=np.intp), verts_per_ring)
        polys = shapely.polygons(shapely.linearrings(flat_coords, indices=ring_indices))
        result[point_rows] = polys

    fallback_index = np.flatnonzero(fallback_mask).astype(np.int32)
    if fallback_index.size > 0:
        result[fallback_index] = shapely.buffer(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
        )

    fast_index = np.flatnonzero(point_mask | empty_rows_mask).astype(np.int32)
    return result, fast_index, fallback_index


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _batch_mitre_offset_uniform(
    flat_coords: np.ndarray,
    dists: np.ndarray,
    verts_per_line: int,
    *,
    mitre_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_lines = flat_coords.shape[0] // verts_per_line
    coords = flat_coords.reshape(n_lines, verts_per_line, 2)
    segments = coords[:, 1:, :] - coords[:, :-1, :]
    lengths = np.linalg.norm(segments, axis=2)
    has_degen = np.any(lengths <= _EPSILON, axis=1)
    safe_lengths = np.where(lengths > _EPSILON, lengths, 1.0)
    directions = segments / safe_lengths[:, :, None]
    signs = np.where(dists >= 0.0, 1.0, -1.0)
    magnitudes = np.abs(dists)
    normals = np.empty_like(directions)
    normals[:, :, 0] = -directions[:, :, 1] * signs[:, None]
    normals[:, :, 1] = directions[:, :, 0] * signs[:, None]

    out = np.empty_like(coords)
    out[:, 0, :] = coords[:, 0, :] + magnitudes[:, None] * normals[:, 0, :]
    out[:, -1, :] = coords[:, -1, :] + magnitudes[:, None] * normals[:, -1, :]

    ok_mask = ~has_degen
    if verts_per_line > 2:
        prev_dirs = directions[:, :-1, :]
        next_dirs = directions[:, 1:, :]
        inner_coords = coords[:, 1:-1, :]
        prev_norms = normals[:, :-1, :]
        next_norms = normals[:, 1:, :]
        prev_shifts = inner_coords + magnitudes[:, None, None] * prev_norms
        next_shifts = inner_coords + magnitudes[:, None, None] * next_norms

        denoms = prev_dirs[:, :, 0] * next_dirs[:, :, 1] - prev_dirs[:, :, 1] * next_dirs[:, :, 0]
        collinear = np.abs(denoms) <= _EPSILON
        safe_denoms = np.where(collinear, 1.0, denoms)

        deltas = next_shifts - prev_shifts
        t = (deltas[:, :, 0] * next_dirs[:, :, 1] - deltas[:, :, 1] * next_dirs[:, :, 0]) / safe_denoms
        intersections = prev_shifts + t[:, :, None] * prev_dirs

        miter_dists = np.linalg.norm(intersections - next_shifts, axis=2)
        safe_mags = np.where(magnitudes > _EPSILON, magnitudes, _EPSILON)
        miter_ratios = miter_dists / safe_mags[:, None]
        exceeded = ~collinear & (miter_ratios > mitre_limit)
        ok_mask &= ~np.any(exceeded, axis=1)

        out[:, 1:-1, :] = np.where(collinear[:, :, None], next_shifts, intersections)

    out_flat = out.reshape(-1, 2)
    line_indices = np.repeat(np.arange(n_lines, dtype=np.intp), verts_per_line)
    lines = shapely.linestrings(out_flat, indices=line_indices)
    return lines, ok_mask, out


def _empty_coords() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float64)


def _build_linestring_owned_from_chunks(
    coord_chunks: list[np.ndarray | None],
    validity: np.ndarray,
) -> OwnedGeometryArray:
    row_count = int(validity.size)
    line_rows = np.flatnonzero(validity)
    tags = np.full(row_count, NULL_TAG, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)
    if line_rows.size == 0:
        return OwnedGeometryArray(
            validity=validity.astype(bool, copy=True),
            tags=tags,
            family_row_offsets=family_row_offsets,
            families={},
        )

    tags[line_rows] = FAMILY_TAGS[GeometryFamily.LINESTRING]
    family_row_offsets[line_rows] = np.arange(line_rows.size, dtype=np.int32)
    coord_counts = np.empty(line_rows.size, dtype=np.int32)
    for output_index, row in enumerate(line_rows):
        chunk = coord_chunks[int(row)]
        coord_counts[output_index] = 0 if chunk is None else int(chunk.shape[0])
    geometry_offsets = np.empty(line_rows.size + 1, dtype=np.int32)
    geometry_offsets[0] = 0
    np.cumsum(coord_counts, out=geometry_offsets[1:])
    non_empty_chunks = [
        coord_chunks[int(row)]
        for row in line_rows
        if coord_chunks[int(row)] is not None and coord_chunks[int(row)].shape[0] > 0
    ]
    if non_empty_chunks:
        coords = np.concatenate(non_empty_chunks, axis=0).astype(np.float64, copy=False)
        x = np.ascontiguousarray(coords[:, 0])
        y = np.ascontiguousarray(coords[:, 1])
    else:
        x = np.empty(0, dtype=np.float64)
        y = np.empty(0, dtype=np.float64)
    buffer = FamilyGeometryBuffer(
        family=GeometryFamily.LINESTRING,
        schema=get_geometry_buffer_schema(GeometryFamily.LINESTRING),
        row_count=int(line_rows.size),
        x=x,
        y=y,
        geometry_offsets=geometry_offsets,
        empty_mask=coord_counts == 0,
    )
    return OwnedGeometryArray(
        validity=validity.astype(bool, copy=True),
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.LINESTRING: buffer},
    )


def _offset_from_coords_mitre(coords: np.ndarray, distance: float, *, mitre_limit: float) -> np.ndarray | None:
    if coords.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64)
    segments = coords[1:] - coords[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= _EPSILON):
        return None
    directions = segments / lengths[:, None]
    sign = 1.0 if distance >= 0.0 else -1.0
    normals = sign * np.column_stack((-directions[:, 1], directions[:, 0]))
    magnitude = abs(distance)
    n_verts = coords.shape[0]

    out = np.empty((n_verts, 2), dtype=np.float64)
    out[0] = coords[0] + magnitude * normals[0]
    out[-1] = coords[-1] + magnitude * normals[-1]

    if n_verts > 2:
        prev_dirs = directions[:-1]
        next_dirs = directions[1:]
        inner_coords = coords[1:-1]
        prev_norms = normals[:-1]
        next_norms = normals[1:]
        prev_shifts = inner_coords + magnitude * prev_norms
        next_shifts = inner_coords + magnitude * next_norms

        denoms = prev_dirs[:, 0] * next_dirs[:, 1] - prev_dirs[:, 1] * next_dirs[:, 0]
        collinear = np.abs(denoms) <= _EPSILON
        deltas = next_shifts - prev_shifts
        safe_denoms = np.where(collinear, 1.0, denoms)
        t = (deltas[:, 0] * next_dirs[:, 1] - deltas[:, 1] * next_dirs[:, 0]) / safe_denoms
        intersections = prev_shifts + t[:, None] * prev_dirs

        miter_dists = np.linalg.norm(intersections - next_shifts, axis=1)
        miter_ratios = miter_dists / max(magnitude, _EPSILON)
        if np.any(~collinear & (miter_ratios > mitre_limit)):
            return None

        out[1:-1] = np.where(collinear[:, None], next_shifts, intersections)

    return out


def _offset_single_linestring(
    line: LineString,
    distance: float,
    *,
    join_style: str,
    mitre_limit: float,
) -> LineString | None:
    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return LineString()
    segments = coords[1:] - coords[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= _EPSILON):
        return None
    directions = segments / lengths[:, None]
    sign = 1.0 if distance >= 0.0 else -1.0
    normals = sign * np.column_stack((-directions[:, 1], directions[:, 0]))
    magnitude = abs(distance)
    shifted_points: list[np.ndarray] = [coords[0] + magnitude * normals[0]]

    for vertex_index in range(1, coords.shape[0] - 1):
        prev_direction = directions[vertex_index - 1]
        next_direction = directions[vertex_index]
        prev_shift = coords[vertex_index] + magnitude * normals[vertex_index - 1]
        next_shift = coords[vertex_index] + magnitude * normals[vertex_index]
        denominator = _cross(prev_direction, next_direction)

        if abs(denominator) <= _EPSILON:
            shifted_points.append(next_shift)
            continue

        delta = next_shift - prev_shift
        t = _cross(delta, next_direction) / denominator
        intersection = prev_shift + (t * prev_direction)
        if join_style == "bevel":
            shifted_points.extend((prev_shift, next_shift))
            continue
        miter_ratio = np.linalg.norm(intersection - next_shift) / max(magnitude, _EPSILON)
        if miter_ratio > mitre_limit:
            return None
        shifted_points.append(intersection)

    shifted_points.append(coords[-1] + magnitude * normals[-1])
    return LineString(np.asarray(shifted_points, dtype=np.float64))


def offset_curve_owned_cpu(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, OwnedGeometryArray | None]:
    geometries = np.asarray(values, dtype=object)
    distances = normalize_distances(distance, len(geometries))
    row_count = len(geometries)
    result = np.empty(row_count, dtype=object)
    coord_chunks: list[np.ndarray | None] = [None] * row_count

    non_null_mask = np.array([g is not None for g in geometries], dtype=bool)
    type_ids = np.full(row_count, -1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    if np.any(non_null_mask):
        non_null_geoms = geometries[non_null_mask]
        type_ids[non_null_mask] = shapely.get_type_id(non_null_geoms)
        empty_mask[non_null_mask] = shapely.is_empty(non_null_geoms)

    result[~non_null_mask] = None

    empty_rows_mask = non_null_mask & empty_mask
    if np.any(empty_rows_mask):
        empty_idx = np.flatnonzero(empty_rows_mask)
        result[empty_idx] = geometries[empty_idx]
        for row_index in empty_idx:
            coord_chunks[int(row_index)] = _empty_coords()

    linestring_mask = non_null_mask & ~empty_mask & (type_ids == _LINESTRING_TYPE_ID) & (join_style != "round")
    fallback_mask = non_null_mask & ~empty_mask & ~linestring_mask

    linestring_rows = np.flatnonzero(linestring_mask)
    fast_list: list[int] = list(np.flatnonzero(empty_rows_mask))
    deferred_fallback: list[int] = list(np.flatnonzero(fallback_mask))

    if linestring_rows.size > 0 and join_style == "mitre":
        line_geoms = geometries[linestring_rows]
        line_dists = distances[linestring_rows]
        coord_counts = shapely.get_num_coordinates(line_geoms)
        all_coords = shapely.get_coordinates(line_geoms)
        unique_counts = np.unique(coord_counts)

        if unique_counts.size == 1 and unique_counts[0] >= 2:
            batch_lines, batch_ok, batch_coords = _batch_mitre_offset_uniform(
                all_coords, line_dists, int(unique_counts[0]), mitre_limit=mitre_limit,
            )
            ok_local = np.flatnonzero(batch_ok)
            fail_local = np.flatnonzero(~batch_ok)
            if ok_local.size > 0:
                ok_rows = linestring_rows[ok_local]
                result[ok_rows] = batch_lines[ok_local]
                fast_list.extend(ok_rows.tolist())
                for row_index, local_idx in zip(ok_rows, ok_local, strict=True):
                    coord_chunks[int(row_index)] = batch_coords[int(local_idx)]
            deferred_fallback.extend(linestring_rows[fail_local].tolist())
        else:
            out_coord_list: list[np.ndarray] = []
            out_indices_list: list[np.ndarray] = []
            succeeded: list[int] = []
            offset_start = 0
            out_idx = 0
            for local_idx in range(linestring_rows.size):
                row_index = linestring_rows[local_idx]
                n_coords = coord_counts[local_idx]
                coords = all_coords[offset_start:offset_start + n_coords]
                offset_start += n_coords
                dist = float(line_dists[local_idx])
                offset_coords = _offset_from_coords_mitre(coords, dist, mitre_limit=mitre_limit)
                if offset_coords is None:
                    deferred_fallback.append(row_index)
                else:
                    n_out = offset_coords.shape[0]
                    out_coord_list.append(offset_coords)
                    out_indices_list.append(np.full(n_out, out_idx, dtype=np.intp))
                    out_idx += 1
                    succeeded.append(local_idx)
                    fast_list.append(row_index)
                    coord_chunks[int(row_index)] = offset_coords
            if out_coord_list:
                flat_out = np.concatenate(out_coord_list, axis=0)
                flat_indices = np.concatenate(out_indices_list)
                lines = shapely.linestrings(flat_out, indices=flat_indices)
                result[linestring_rows[np.asarray(succeeded, dtype=np.intp)]] = lines
    else:
        for row_index in linestring_rows:
            offset = _offset_single_linestring(
                geometries[row_index],
                float(distances[row_index]),
                join_style=join_style,
                mitre_limit=mitre_limit,
            )
            if offset is None:
                deferred_fallback.append(row_index)
            else:
                result[row_index] = offset
                fast_list.append(row_index)
                if offset.is_empty:
                    coord_chunks[int(row_index)] = _empty_coords()
                else:
                    coord_chunks[int(row_index)] = np.asarray(offset.coords, dtype=np.float64)

    fallback_index = np.asarray(deferred_fallback, dtype=np.int32)
    if fallback_index.size > 0:
        fallback_result = shapely.offset_curve(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        result[fallback_index] = fallback_result
        fallback_counts = np.asarray(shapely.get_num_coordinates(fallback_result), dtype=np.int32)
        fallback_coords = shapely.get_coordinates(fallback_result)
        fallback_offsets = np.empty(fallback_counts.size + 1, dtype=np.int64)
        fallback_offsets[0] = 0
        np.cumsum(fallback_counts, out=fallback_offsets[1:])
        for local_idx, row_index in enumerate(fallback_index):
            start = int(fallback_offsets[local_idx])
            stop = int(fallback_offsets[local_idx + 1])
            if stop == start:
                coord_chunks[int(row_index)] = _empty_coords()
            else:
                coord_chunks[int(row_index)] = fallback_coords[start:stop]

    owned_result = None
    if not np.any(non_null_mask) or np.all(type_ids[non_null_mask] == _LINESTRING_TYPE_ID):
        owned_result = _build_linestring_owned_from_chunks(coord_chunks, non_null_mask)

    return (
        result,
        np.asarray(sorted(fast_list), dtype=np.int32),
        fallback_index,
        owned_result,
    )


def _non_null_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray([geometry is not None for geometry in values], dtype=bool)


def supports_point_buffer_surface(
    geometries: np.ndarray,
    *,
    cap_style,
    join_style,
    single_sided: bool,
) -> bool:
    if single_sided or cap_style != "round" or join_style != "round":
        return False
    non_null = _non_null_mask(geometries)
    if not np.any(non_null):
        return True
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == _POINT_TYPE_ID))


def supports_point_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    cap_style,
    join_style,
    single_sided: bool,
) -> bool:
    if not supports_point_buffer_surface(
        geometries,
        cap_style=cap_style,
        join_style=join_style,
        single_sided=single_sided,
    ):
        return False
    if quad_segs < 1 or len(geometries) == 0:
        return False
    non_null = _non_null_mask(geometries)
    return bool(np.all(non_null) and not np.any(shapely.is_empty(geometries)))


def supports_linestring_buffer_surface(
    geometries: np.ndarray,
    *,
    single_sided: bool,
) -> bool:
    if single_sided:
        return False
    non_null = _non_null_mask(geometries)
    if not np.any(non_null):
        return True
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == _LINESTRING_TYPE_ID))


def supports_linestring_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    single_sided: bool,
) -> bool:
    if not supports_linestring_buffer_surface(
        geometries,
        single_sided=single_sided,
    ):
        return False
    if quad_segs < 1 or len(geometries) == 0:
        return False
    non_null = _non_null_mask(geometries)
    return bool(np.all(non_null) and not np.any(shapely.is_empty(geometries)))


def supports_polygon_buffer_surface(
    geometries: np.ndarray,
    *,
    single_sided: bool,
) -> bool:
    if single_sided:
        return False
    non_null = _non_null_mask(geometries)
    if not np.any(non_null):
        return True
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == _POLYGON_TYPE_ID))


def supports_polygon_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    single_sided: bool,
) -> bool:
    if not supports_polygon_buffer_surface(
        geometries,
        single_sided=single_sided,
    ):
        return False
    if quad_segs < 1 or len(geometries) == 0:
        return False
    non_null = _non_null_mask(geometries)
    return bool(np.all(non_null) and not np.any(shapely.is_empty(geometries)))


def supports_offset_curve_surface(geometries: np.ndarray, *, join_style) -> bool:
    if join_style == "round":
        return False
    non_null = _non_null_mask(geometries)
    if not np.any(non_null):
        return True
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == _LINESTRING_TYPE_ID))


def benchmark_point_buffer_baseline(geometries: np.ndarray, distance: float, *, quad_segs: int) -> float:
    shapely.buffer(geometries, distance, quad_segs=quad_segs)
    started = perf_counter()
    shapely.buffer(geometries, distance, quad_segs=quad_segs)
    return perf_counter() - started


def benchmark_offset_curve_baseline(geometries: np.ndarray, distance: float, *, join_style: str) -> float:
    shapely.offset_curve(geometries, distance, join_style=join_style)
    started = perf_counter()
    shapely.offset_curve(geometries, distance, join_style=join_style)
    return perf_counter() - started
