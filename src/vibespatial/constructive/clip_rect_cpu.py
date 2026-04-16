"""CPU-only helpers for clip_by_rect host assembly and baselines."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.runtime.config import SPATIAL_EPSILON

EMPTY = GeometryCollection()
_POINT_EPSILON = SPATIAL_EPSILON


def normalize_values(values: Sequence[object | None] | np.ndarray | OwnedGeometryArray) -> tuple[np.ndarray, OwnedGeometryArray]:
    if isinstance(values, OwnedGeometryArray):
        shapely_values = np.asarray(values.to_shapely(), dtype=object)
        return shapely_values, values
    shapely_values = np.asarray(values, dtype=object)
    return shapely_values, from_shapely_geometries(shapely_values.tolist())


def point_clip_result_template(owned: OwnedGeometryArray) -> np.ndarray:
    result = np.empty(owned.row_count, dtype=object)
    result[:] = EMPTY
    result[~owned.validity] = None
    return result


def inside_left(point: tuple[float, float], xmin: float) -> bool:
    return point[0] >= xmin


def inside_right(point: tuple[float, float], xmax: float) -> bool:
    return point[0] <= xmax


def inside_bottom(point: tuple[float, float], ymin: float) -> bool:
    return point[1] >= ymin


def inside_top(point: tuple[float, float], ymax: float) -> bool:
    return point[1] <= ymax


def intersect_vertical(
    p0: tuple[float, float],
    p1: tuple[float, float],
    x: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    if abs(x1 - x0) <= _POINT_EPSILON:
        return float(x), float(y0)
    t = (x - x0) / (x1 - x0)
    return float(x), float(y0 + t * (y1 - y0))


def intersect_horizontal(
    p0: tuple[float, float],
    p1: tuple[float, float],
    y: float,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    if abs(y1 - y0) <= _POINT_EPSILON:
        return float(x0), float(y)
    t = (y - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0)), float(y)


def sutherland_hodgman_ring(
    coords: list[tuple[float, float]],
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    if len(coords) < 3:
        return []
    xmin, ymin, xmax, ymax = rect
    subject = coords[:-1] if coords[0] == coords[-1] else coords[:]
    if not subject:
        return []

    boundaries = (
        (lambda point: inside_left(point, xmin), lambda a, b: intersect_vertical(a, b, xmin)),
        (lambda point: inside_right(point, xmax), lambda a, b: intersect_vertical(a, b, xmax)),
        (lambda point: inside_bottom(point, ymin), lambda a, b: intersect_horizontal(a, b, ymin)),
        (lambda point: inside_top(point, ymax), lambda a, b: intersect_horizontal(a, b, ymax)),
    )

    output = subject
    for inside, intersect in boundaries:
        if not output:
            return []
        clipped: list[tuple[float, float]] = []
        previous = output[-1]
        previous_inside = inside(previous)
        for current in output:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    clipped.append(intersect(previous, current))
                clipped.append(current)
            elif previous_inside:
                clipped.append(intersect(previous, current))
            previous = current
            previous_inside = current_inside
        output = clipped

    if not output:
        return []
    deduped: list[tuple[float, float]] = []
    for point in output:
        if deduped and np.allclose(deduped[-1], point, atol=_POINT_EPSILON, rtol=0.0):
            continue
        deduped.append((float(point[0]), float(point[1])))
    if len(deduped) < 3:
        return []
    if not np.allclose(deduped[0], deduped[-1], atol=_POINT_EPSILON, rtol=0.0):
        deduped.append(deduped[0])
    unique = {(round(point[0], 12), round(point[1], 12)) for point in deduped[:-1]}
    if len(unique) < 3:
        return []
    return deduped


def liang_barsky_segment(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    rect: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    xmin, ymin, xmax, ymax = rect
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1 = 0.0
    u2 = 1.0
    for pk, qk in zip(p, q, strict=True):
        if abs(pk) <= _POINT_EPSILON:
            if qk < 0.0:
                return None
            continue
        t = qk / pk
        if pk < 0.0:
            u1 = max(u1, t)
        else:
            u2 = min(u2, t)
        if u1 > u2:
            return None
    cx0 = x0 + u1 * dx
    cy0 = y0 + u1 * dy
    cx1 = x0 + u2 * dx
    cy1 = y0 + u2 * dy
    if np.allclose((cx0, cy0), (cx1, cy1), atol=_POINT_EPSILON, rtol=0.0):
        return None
    return float(cx0), float(cy0), float(cx1), float(cy1)


def merge_clipped_segments(parts: list[list[tuple[float, float]]], segment: tuple[float, float, float, float]) -> None:
    start = (segment[0], segment[1])
    end = (segment[2], segment[3])
    if not parts:
        parts.append([start, end])
        return
    current = parts[-1]
    if np.allclose(current[-1], start, atol=_POINT_EPSILON, rtol=0.0):
        if not np.allclose(current[-1], end, atol=_POINT_EPSILON, rtol=0.0):
            current.append(end)
        return
    parts.append([start, end])


def build_linestring_result(parts: list[list[tuple[float, float]]]) -> object:
    normalized = [part for part in parts if len(part) >= 2]
    if not normalized:
        return EMPTY
    if len(normalized) == 1:
        return LineString(normalized[0])
    return MultiLineString(normalized)


def build_polygon_result(polygons: list[Polygon]) -> object:
    non_empty = [polygon for polygon in polygons if not polygon.is_empty]
    if not non_empty:
        return EMPTY
    if len(non_empty) == 1:
        return non_empty[0]
    return shapely.multipolygons(non_empty)


def clip_point_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    start = int(buffer.geometry_offsets[family_row])
    end = int(buffer.geometry_offsets[family_row + 1])
    if end <= start:
        return EMPTY
    xmin, ymin, xmax, ymax = rect
    if buffer.family.value == "point":
        x = float(buffer.x[start])
        y = float(buffer.y[start])
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return Point(x, y)
        return EMPTY

    points = [
        (float(buffer.x[index]), float(buffer.y[index]))
        for index in range(start, end)
        if xmin <= float(buffer.x[index]) <= xmax and ymin <= float(buffer.y[index]) <= ymax
    ]
    if not points:
        return EMPTY
    if len(points) == 1:
        return Point(points[0])
    return MultiPoint(points)


def clip_line_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    parts: list[list[tuple[float, float]]] = []
    if buffer.family.value == "linestring":
        spans = [(int(buffer.geometry_offsets[family_row]), int(buffer.geometry_offsets[family_row + 1]))]
    else:
        part_start = int(buffer.geometry_offsets[family_row])
        part_end = int(buffer.geometry_offsets[family_row + 1])
        spans = [
            (int(buffer.part_offsets[index]), int(buffer.part_offsets[index + 1]))
            for index in range(part_start, part_end)
        ]
    for start, end in spans:
        part_segments: list[list[tuple[float, float]]] = []
        for index in range(start, end - 1):
            segment = liang_barsky_segment(
                float(buffer.x[index]),
                float(buffer.y[index]),
                float(buffer.x[index + 1]),
                float(buffer.y[index + 1]),
                rect,
            )
            if segment is None:
                continue
            merge_clipped_segments(part_segments, segment)
        parts.extend(part_segments)
    return build_linestring_result(parts)


def polygon_ring_spans(buffer, family_row: int) -> list[list[tuple[float, float]]]:
    def ring_lookup(ring_index: int) -> tuple[int, int]:
        return int(buffer.ring_offsets[ring_index]), int(buffer.ring_offsets[ring_index + 1])

    if buffer.family.value == "polygon":
        polygon_indices = [(int(buffer.geometry_offsets[family_row]), int(buffer.geometry_offsets[family_row + 1]))]
    else:
        polygon_start = int(buffer.geometry_offsets[family_row])
        polygon_end = int(buffer.geometry_offsets[family_row + 1])
        polygon_indices = [
            (int(buffer.part_offsets[polygon_index]), int(buffer.part_offsets[polygon_index + 1]))
            for polygon_index in range(polygon_start, polygon_end)
        ]

    polygons: list[list[tuple[float, float]]] = []
    for ring_start, ring_end in polygon_indices:
        rings = []
        for ring_index in range(ring_start, ring_end):
            coord_start, coord_end = ring_lookup(ring_index)
            rings.append(
                [
                    (float(buffer.x[index]), float(buffer.y[index]))
                    for index in range(coord_start, coord_end)
                ]
            )
        polygons.append(rings)
    return polygons


def clip_polygon_family(buffer, family_row: int, rect: tuple[float, float, float, float]) -> object:
    polygons: list[Polygon] = []
    for rings in polygon_ring_spans(buffer, family_row):
        if not rings:
            continue
        exterior = sutherland_hodgman_ring(rings[0], rect)
        if not exterior:
            continue
        holes = []
        for ring in rings[1:]:
            clipped = sutherland_hodgman_ring(ring, rect)
            if clipped:
                holes.append(clipped)
        polygon = Polygon(exterior, holes=holes)
        if not polygon.is_empty:
            polygons.append(polygon)
    return build_polygon_result(polygons)


def reconstruct_polygon_result_from_rings(
    ring_polygon_map: list[int],
    ring_is_exterior: list[bool],
    out_x: np.ndarray,
    out_y: np.ndarray,
    out_ring_offsets: np.ndarray,
) -> object:
    if out_x.size == 0:
        return EMPTY

    polygons: list[Polygon] = []
    current_exterior = None
    current_holes: list[list[tuple[float, float]]] = []

    for ring_idx in range(len(ring_polygon_map)):
        start = int(out_ring_offsets[ring_idx])
        end = int(out_ring_offsets[ring_idx + 1])
        verts = end - start
        if verts < 4:
            if ring_is_exterior[ring_idx]:
                if current_exterior is not None:
                    poly = Polygon(current_exterior, holes=current_holes)
                    if not poly.is_empty:
                        polygons.append(poly)
                current_exterior = None
                current_holes = []
            continue

        coords = [(float(out_x[i]), float(out_y[i])) for i in range(start, end)]
        if ring_is_exterior[ring_idx]:
            if current_exterior is not None:
                poly = Polygon(current_exterior, holes=current_holes)
                if not poly.is_empty:
                    polygons.append(poly)
            current_exterior = coords
            current_holes = []
        else:
            current_holes.append(coords)

    if current_exterior is not None:
        poly = Polygon(current_exterior, holes=current_holes)
        if not poly.is_empty:
            polygons.append(poly)

    return build_polygon_result(polygons)


def supported_fast_row(
    family_name: str | None,
    local_index: int,
    owned: OwnedGeometryArray,
    shapely_value: object | None,
) -> bool:
    if family_name is None or local_index < 0 or shapely_value is None:
        return False
    if shapely_value.is_empty:
        return True
    if family_name in {"point", "multipoint", "linestring", "multilinestring"}:
        return True
    if family_name in {"polygon", "multipolygon"} and shapely.is_valid(shapely_value):
        return True
    return False


def materialize_candidates_vectorized(
    owned: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    from vibespatial.geometry.host_bridge import materialize_family_row
    from vibespatial.geometry.owned import TAG_FAMILIES

    if candidate_rows.size == 0:
        return np.empty(0, dtype=object)

    tags = owned.tags[candidate_rows]
    unique_tags = np.unique(tags)

    if len(unique_tags) == 1:
        tag = int(unique_tags[0])
        family = TAG_FAMILIES[tag]
        buffer = owned.families[family]
        local_rows = owned.family_row_offsets[candidate_rows].astype(np.int32)

        if family in (GeometryFamily.LINESTRING,):
            offsets = np.empty(len(local_rows) + 1, dtype=np.int32)
            offsets[0] = 0
            for idx, lr in enumerate(local_rows):
                s = int(buffer.geometry_offsets[lr])
                e = int(buffer.geometry_offsets[lr + 1])
                offsets[idx + 1] = offsets[idx] + (e - s)
            total_coords = int(offsets[-1])
            flat_xy = np.empty((total_coords, 2), dtype=np.float64)
            pos = 0
            for lr in local_rows:
                s = int(buffer.geometry_offsets[lr])
                e = int(buffer.geometry_offsets[lr + 1])
                n = e - s
                flat_xy[pos:pos + n, 0] = buffer.x[s:e]
                flat_xy[pos:pos + n, 1] = buffer.y[s:e]
                pos += n
            indices = np.repeat(np.arange(len(local_rows), dtype=np.int32), np.diff(offsets))
            return shapely.linestrings(flat_xy, indices=indices)

        if family in (GeometryFamily.POLYGON,):
            ring_coords: list[float] = []
            ring_coord_offsets = [0]
            ring_geom_indices: list[int] = []
            for geom_idx, lr in enumerate(local_rows):
                ring_start = int(buffer.geometry_offsets[lr])
                ring_end = int(buffer.geometry_offsets[lr + 1])
                for ring_idx in range(ring_start, ring_end):
                    cs = int(buffer.ring_offsets[ring_idx])
                    ce = int(buffer.ring_offsets[ring_idx + 1])
                    n = ce - cs
                    for ci in range(cs, ce):
                        ring_coords.append(float(buffer.x[ci]))
                        ring_coords.append(float(buffer.y[ci]))
                    ring_coord_offsets.append(ring_coord_offsets[-1] + n)
                    ring_geom_indices.append(geom_idx)
            flat_xy = np.asarray(ring_coords, dtype=np.float64).reshape(-1, 2)
            coord_offsets = np.asarray(ring_coord_offsets, dtype=np.int32)
            ring_indices_arr = np.repeat(
                np.arange(len(coord_offsets) - 1, dtype=np.int32),
                np.diff(coord_offsets),
            )
            rings = shapely.linearrings(flat_xy, indices=ring_indices_arr)
            geom_indices_arr = np.asarray(ring_geom_indices, dtype=np.int32)
            return shapely.polygons(rings, indices=geom_indices_arr)

        if family in (GeometryFamily.POINT,):
            xs = np.empty(len(local_rows), dtype=np.float64)
            ys = np.empty(len(local_rows), dtype=np.float64)
            for idx, lr in enumerate(local_rows):
                s = int(buffer.geometry_offsets[lr])
                xs[idx] = buffer.x[s]
                ys[idx] = buffer.y[s]
            return np.asarray(shapely.points(xs, ys), dtype=object)

    candidate_geoms = []
    for i in candidate_rows:
        family = TAG_FAMILIES[int(owned.tags[i])]
        buf = owned.families[family]
        local_row = int(owned.family_row_offsets[i])
        candidate_geoms.append(materialize_family_row(buf, local_row))
    return np.asarray(candidate_geoms, dtype=object)


def clip_by_rect_cpu(
    owned: OwnedGeometryArray,
    rect_intersects_bounds,
    rect: tuple[float, float, float, float],
    shapely_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = compute_geometry_bounds(owned)
    candidate_rows = np.flatnonzero(rect_intersects_bounds(bounds, rect)).astype(np.int32, copy=False)

    result = np.empty(owned.row_count, dtype=object)
    result[:] = EMPTY
    invalid_mask = ~owned.validity
    if invalid_mask.any():
        result[invalid_mask] = None

    if candidate_rows.size > 0:
        if shapely_values is not None:
            candidate_shapely = shapely_values[candidate_rows]
        else:
            if any(not buffer.host_materialized for buffer in owned.families.values()):
                owned._ensure_host_state()
            candidate_shapely = materialize_candidates_vectorized(owned, candidate_rows)
        clipped = shapely.clip_by_rect(candidate_shapely, *rect)
        result[candidate_rows] = np.asarray(clipped, dtype=object)

    return result, candidate_rows


def clip_by_rect_array(candidate_shapely: np.ndarray, rect: tuple[float, float, float, float]) -> np.ndarray:
    """Apply Shapely clip_by_rect to an object array and return object array output."""
    clipped = shapely.clip_by_rect(candidate_shapely, *rect)
    return np.asarray(clipped, dtype=object)


def clip_rect_gpu_available(geometries: np.ndarray, point_type_id: int) -> bool:
    non_null = np.asarray([geometry is not None for geometry in geometries], dtype=bool)
    if not np.any(non_null):
        return False
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == point_type_id))


def benchmark_clip_by_rect_baseline(
    shapely_values: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> float:
    started = perf_counter()
    shapely.clip_by_rect(shapely_values, xmin, ymin, xmax, ymax)
    return perf_counter() - started
