"""Pure-Python / NumPy / Shapely host-path helpers for the overlay pipeline.

These functions implement the CPU fallback path for overlay operations.
They are extracted from ``gpu.py`` to reduce file size and clarify the
boundary between GPU and host code.

None of these functions touch CuPy or CUDA directly.  When GPU-accelerated
batch PIP is needed (inside ``_build_polygon_output_from_faces``), the GPU
function is imported lazily to avoid circular imports.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import shapely  # hygiene:ok -- this module IS the host/Shapely fallback path

from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import RuntimeSelection
from vibespatial.runtime.residency import Residency

from .types import HalfEdgeGraph, OverlayFaceTable

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


# ---------------------------------------------------------------------------
# Small host helpers
# ---------------------------------------------------------------------------


def _signed_area_and_centroid(points: np.ndarray) -> tuple[float, float, float]:
    if points.shape[0] < 3:
        return 0.0, 0.0, 0.0
    closed = np.vstack((points, points[:1]))
    cross = (closed[:-1, 0] * closed[1:, 1]) - (closed[1:, 0] * closed[:-1, 1])
    twice_area = float(np.sum(cross))
    if twice_area == 0.0:
        return 0.0, float(points[:, 0].mean()), float(points[:, 1].mean())
    factor = 1.0 / (3.0 * twice_area)
    cx = float(np.sum((closed[:-1, 0] + closed[1:, 0]) * cross) * factor)
    cy = float(np.sum((closed[:-1, 1] + closed[1:, 1]) * cross) * factor)
    return twice_area * 0.5, cx, cy


def _face_sample_point(points: np.ndarray) -> tuple[float, float]:
    if points.shape[0] == 0:
        return 0.0, 0.0
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    epsilon = extent * 1e-6
    for index in range(points.shape[0]):
        start = points[index]
        end = points[(index + 1) % points.shape[0]]
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        midpoint_x = float((start[0] + end[0]) * 0.5)
        midpoint_y = float((start[1] + end[1]) * 0.5)
        return midpoint_x - (dy / length) * epsilon, midpoint_y + (dx / length) * epsilon
    return float(points[0, 0]), float(points[0, 1])


def _host_union_geometry(values):
    geometries = [geometry for geometry in values.to_shapely() if geometry is not None and not geometry.is_empty]
    if not geometries:
        return None
    return shapely.union_all(np.asarray(geometries, dtype=object), grid_size=0.0)


def _label_face_coverage(left, right, centroid_x: np.ndarray, centroid_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Host fallback: label face coverage via Shapely coverage tests.

    Uses vectorized shapely.covers() on the unioned geometry of each side.
    Only called when the GPU path is unavailable (no CuPy).
    """
    left_union = _host_union_geometry(left)
    right_union = _host_union_geometry(right)
    if centroid_x.size == 0:
        empty = np.asarray([], dtype=np.int8)
        return empty, empty
    points = shapely.points(centroid_x, centroid_y)
    left_covered = (
        np.asarray(shapely.covers(left_union, points), dtype=bool) if left_union is not None else np.zeros(points.size, dtype=bool)
    )
    right_covered = (
        np.asarray(shapely.covers(right_union, points), dtype=bool) if right_union is not None else np.zeros(points.size, dtype=bool)
    )
    return left_covered.astype(np.int8, copy=False), right_covered.astype(np.int8, copy=False)


def _ring_points_for_face(half_edge_graph: HalfEdgeGraph, faces: OverlayFaceTable, face_index: int) -> np.ndarray:
    start = int(faces.face_offsets[face_index])
    stop = int(faces.face_offsets[face_index + 1])
    edge_ids = faces.face_edge_ids[start:stop]
    if edge_ids.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    points = np.column_stack((half_edge_graph.src_x[edge_ids], half_edge_graph.src_y[edge_ids]))
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[:1]))
    return np.asarray(points, dtype=np.float64)


def _point_in_ring(x: float, y: float, ring: np.ndarray) -> bool:
    inside = False
    x0 = float(ring[-1, 0])
    y0 = float(ring[-1, 1])
    for x1, y1 in ring:
        x1 = float(x1)
        y1 = float(y1)
        crosses = (y1 > y) != (y0 > y)
        if crosses:
            denominator = y0 - y1
            if denominator != 0.0:
                x_intersection = ((x0 - x1) * (y - y1) / denominator) + x1
                if x < x_intersection:
                    inside = not inside
        x0 = x1
        y0 = y1
    return inside


def _closed_ring_coords(ring: np.ndarray) -> np.ndarray:
    coords = np.asarray(ring, dtype=np.float64)
    if coords.shape[0] == 0:
        return coords
    if np.allclose(coords[0], coords[-1]):
        return coords
    return np.vstack((coords, coords[:1]))


# ---------------------------------------------------------------------------
# Output buffer assembly helpers
# ---------------------------------------------------------------------------


def _append_polygon_buffer_row(
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    geometry_offsets: list[int],
    ring_offsets: list[int],
    bounds_payload: list[tuple[float, float, float, float]],
    rings: list[np.ndarray],
    coord_cursor: list[int],
) -> None:
    """Append a single-polygon row to GeoArrow buffer lists.

    Coordinates are collected as NumPy array chunks (not per-element floats)
    and concatenated once in ``_build_overlay_output_rows``.
    ``coord_cursor`` is a single-element list used as a mutable running
    coordinate count so that ``ring_offsets`` can be computed without
    flattening x_chunks on every call.
    """
    geometry_offsets.append(len(ring_offsets))
    exterior = _closed_ring_coords(rings[0])
    bounds_payload.append(
        (
            float(np.min(exterior[:, 0])),
            float(np.min(exterior[:, 1])),
            float(np.max(exterior[:, 0])),
            float(np.max(exterior[:, 1])),
        )
    )
    for ring in rings:
        coords = _closed_ring_coords(ring)
        ring_offsets.append(coord_cursor[0])
        x_chunks.append(np.ascontiguousarray(coords[:, 0], dtype=np.float64))
        y_chunks.append(np.ascontiguousarray(coords[:, 1], dtype=np.float64))
        coord_cursor[0] += coords.shape[0]


def _append_multipolygon_buffer_row(
    x_chunks: list[np.ndarray],
    y_chunks: list[np.ndarray],
    geometry_offsets: list[int],
    part_offsets: list[int],
    ring_offsets: list[int],
    bounds_payload: list[tuple[float, float, float, float]],
    polygons: list[list[np.ndarray]],
    coord_cursor: list[int],
) -> None:
    """Append a multi-polygon row to GeoArrow buffer lists.

    Same chunk-based coordinate collection as ``_append_polygon_buffer_row``.
    """
    geometry_offsets.append(len(part_offsets))
    min_x = np.inf
    min_y = np.inf
    max_x = -np.inf
    max_y = -np.inf
    for rings in polygons:
        part_offsets.append(len(ring_offsets))
        exterior = _closed_ring_coords(rings[0])
        min_x = min(min_x, float(np.min(exterior[:, 0])))
        min_y = min(min_y, float(np.min(exterior[:, 1])))
        max_x = max(max_x, float(np.max(exterior[:, 0])))
        max_y = max(max_y, float(np.max(exterior[:, 1])))
        for ring in rings:
            coords = _closed_ring_coords(ring)
            ring_offsets.append(coord_cursor[0])
            x_chunks.append(np.ascontiguousarray(coords[:, 0], dtype=np.float64))
            y_chunks.append(np.ascontiguousarray(coords[:, 1], dtype=np.float64))
            coord_cursor[0] += coords.shape[0]
    bounds_payload.append((min_x, min_y, max_x, max_y))


# ---------------------------------------------------------------------------
# Host output row assembly
# ---------------------------------------------------------------------------


def _build_overlay_output_rows(
    row_polygons: dict[int, list[list[np.ndarray]]],
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    # Lazy import to avoid circular dependency with gpu.py
    from vibespatial.overlay.gpu import _empty_polygon_output

    if not row_polygons:
        return _empty_polygon_output(runtime_selection)

    ordered_rows = sorted(row_polygons)
    validity = np.ones(len(ordered_rows), dtype=bool)
    tags = np.full(len(ordered_rows), -1, dtype=np.int8)
    family_row_offsets = np.full(len(ordered_rows), -1, dtype=np.int32)

    # Coordinate chunks: collect np.ndarray slices, concatenate once at the
    # end.  This replaces the previous per-element ``float(value)`` generator
    # that iterated every coordinate through Python (hitlist #13).
    polygon_x_chunks: list[np.ndarray] = []
    polygon_y_chunks: list[np.ndarray] = []
    polygon_geometry_offsets: list[int] = []
    polygon_ring_offsets: list[int] = []
    polygon_bounds: list[tuple[float, float, float, float]] = []
    polygon_coord_cursor: list[int] = [0]
    polygon_count = 0

    multipolygon_x_chunks: list[np.ndarray] = []
    multipolygon_y_chunks: list[np.ndarray] = []
    multipolygon_geometry_offsets: list[int] = []
    multipolygon_part_offsets: list[int] = []
    multipolygon_ring_offsets: list[int] = []
    multipolygon_bounds: list[tuple[float, float, float, float]] = []
    multipolygon_coord_cursor: list[int] = [0]
    multipolygon_count = 0

    for output_row, row_index in enumerate(ordered_rows):
        polygons = row_polygons[row_index]
        if len(polygons) == 1:
            tags[output_row] = FAMILY_TAGS[GeometryFamily.POLYGON]
            family_row_offsets[output_row] = polygon_count
            _append_polygon_buffer_row(
                polygon_x_chunks,
                polygon_y_chunks,
                polygon_geometry_offsets,
                polygon_ring_offsets,
                polygon_bounds,
                polygons[0],
                polygon_coord_cursor,
            )
            polygon_count += 1
            continue
        tags[output_row] = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
        family_row_offsets[output_row] = multipolygon_count
        _append_multipolygon_buffer_row(
            multipolygon_x_chunks,
            multipolygon_y_chunks,
            multipolygon_geometry_offsets,
            multipolygon_part_offsets,
            multipolygon_ring_offsets,
            multipolygon_bounds,
            polygons,
            multipolygon_coord_cursor,
        )
        multipolygon_count += 1

    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    if polygon_count:
        poly_x = np.concatenate(polygon_x_chunks) if polygon_x_chunks else np.empty(0, dtype=np.float64)
        poly_y = np.concatenate(polygon_y_chunks) if polygon_y_chunks else np.empty(0, dtype=np.float64)
        total_poly_coords = polygon_coord_cursor[0]
        families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=polygon_count,
            x=poly_x,
            y=poly_y,
            geometry_offsets=np.asarray([*polygon_geometry_offsets, len(polygon_ring_offsets)], dtype=np.int32),
            empty_mask=np.zeros(polygon_count, dtype=bool),
            ring_offsets=np.asarray([*polygon_ring_offsets, total_poly_coords], dtype=np.int32),
            bounds=np.asarray(polygon_bounds, dtype=np.float64),
        )
    if multipolygon_count:
        mpoly_x = np.concatenate(multipolygon_x_chunks) if multipolygon_x_chunks else np.empty(0, dtype=np.float64)
        mpoly_y = np.concatenate(multipolygon_y_chunks) if multipolygon_y_chunks else np.empty(0, dtype=np.float64)
        total_mpoly_coords = multipolygon_coord_cursor[0]
        families[GeometryFamily.MULTIPOLYGON] = FamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON),
            row_count=multipolygon_count,
            x=mpoly_x,
            y=mpoly_y,
            geometry_offsets=np.asarray(
                [*multipolygon_geometry_offsets, len(multipolygon_part_offsets)],
                dtype=np.int32,
            ),
            empty_mask=np.zeros(multipolygon_count, dtype=bool),
            part_offsets=np.asarray([*multipolygon_part_offsets, len(multipolygon_ring_offsets)], dtype=np.int32),
            ring_offsets=np.asarray([*multipolygon_ring_offsets, total_mpoly_coords], dtype=np.int32),
            bounds=np.asarray(multipolygon_bounds, dtype=np.float64),
        )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
        runtime_history=[runtime_selection],
    )


# ---------------------------------------------------------------------------
# Main host fallback: face -> polygon output assembly
# ---------------------------------------------------------------------------


def _build_polygon_output_from_faces(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    selected_face_indices: np.ndarray,
) -> OwnedGeometryArray:
    # Lazy imports to avoid circular dependency with gpu.py
    from vibespatial.overlay.gpu import (
        _BATCH_PIP_GPU_THRESHOLD,
        _batch_point_in_ring_gpu,
        _empty_polygon_output,
    )

    if selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)

    edge_face_ids = np.full(half_edge_graph.edge_count, -1, dtype=np.int32)
    for face_index in range(faces.face_count):
        start = int(faces.face_offsets[face_index])
        stop = int(faces.face_offsets[face_index + 1])
        edge_face_ids[faces.face_edge_ids[start:stop]] = face_index

    selected_face_mask = np.zeros(faces.face_count, dtype=bool)
    selected_face_mask[selected_face_indices] = True
    boundary_mask = np.zeros(half_edge_graph.edge_count, dtype=bool)
    for edge_id in range(half_edge_graph.edge_count):
        face_index = int(edge_face_ids[edge_id])
        if face_index < 0 or not selected_face_mask[face_index]:
            continue
        twin_face = int(edge_face_ids[edge_id ^ 1]) if (edge_id ^ 1) < edge_face_ids.size else -1
        if twin_face < 0 or not selected_face_mask[twin_face]:
            boundary_mask[edge_id] = True

    if not np.any(boundary_mask):
        return _empty_polygon_output(faces.runtime_selection)

    def _next_boundary_edge(edge_id: int) -> int:
        current = int(half_edge_graph.next_edge_ids[edge_id])
        guard = 0
        while True:
            twin_face = int(edge_face_ids[current ^ 1]) if (current ^ 1) < edge_face_ids.size else -1
            if twin_face < 0 or not selected_face_mask[twin_face]:
                return current
            current = int(half_edge_graph.next_edge_ids[current ^ 1])
            guard += 1
            if guard > half_edge_graph.edge_count:
                raise RuntimeError("overlay boundary walk did not converge")

    visited_boundary = np.zeros(half_edge_graph.edge_count, dtype=bool)
    cycle_rings: dict[int, np.ndarray] = {}
    cycle_samples: dict[int, tuple[float, float]] = {}
    cycle_areas: dict[int, float] = {}
    cycle_rows: dict[int, int] = {}
    cycle_selected_boundary: dict[int, bool] = {}
    cycle_id = 0
    for edge_id in np.flatnonzero(boundary_mask).tolist():
        edge_id = int(edge_id)
        if visited_boundary[edge_id]:
            continue
        cycle_edges: list[int] = []
        current = edge_id
        while not visited_boundary[current]:
            visited_boundary[current] = True
            cycle_edges.append(current)
            current = _next_boundary_edge(current)
        if current != edge_id or len(cycle_edges) < 3:
            continue
        row_ids = np.unique(half_edge_graph.row_indices[np.asarray(cycle_edges, dtype=np.int32)])
        if row_ids.size != 1:
            raise RuntimeError("overlay boundary cycle spans multiple source rows; row-wise output assembly would be ambiguous")
        points = np.column_stack(
            (
                half_edge_graph.src_x[np.asarray(cycle_edges, dtype=np.int32)],
                half_edge_graph.src_y[np.asarray(cycle_edges, dtype=np.int32)],
            )
        )
        ring = points if np.allclose(points[0], points[-1]) else np.vstack((points, points[:1]))
        cycle_rings[cycle_id] = ring
        cycle_samples[cycle_id] = _face_sample_point(ring[:-1])
        cycle_areas[cycle_id] = abs(float(_signed_area_and_centroid(ring[:-1])[0]))
        cycle_rows[cycle_id] = int(row_ids[0])
        cycle_selected_boundary[cycle_id] = True
        cycle_id += 1

    for face_index in np.flatnonzero(faces.bounded_mask != 0).tolist():
        face_index = int(face_index)
        if selected_face_mask[face_index]:
            continue
        ring = _ring_points_for_face(half_edge_graph, faces, face_index)
        if ring.shape[0] < 4:
            continue
        start = int(faces.face_offsets[face_index])
        stop = int(faces.face_offsets[face_index + 1])
        edge_ids = faces.face_edge_ids[start:stop]
        row_ids = np.unique(half_edge_graph.row_indices[edge_ids])
        if row_ids.size != 1:
            raise RuntimeError("overlay hole candidate spans multiple source rows; row-wise output assembly would be ambiguous")
        cycle_rings[cycle_id] = ring
        cycle_samples[cycle_id] = _face_sample_point(ring[:-1])
        cycle_areas[cycle_id] = abs(float(faces.signed_area[face_index]))
        cycle_rows[cycle_id] = int(row_ids[0])
        cycle_selected_boundary[cycle_id] = False
        cycle_id += 1

    selected_boundary_indices = [cycle_index for cycle_index in cycle_rings if cycle_selected_boundary[cycle_index]]

    # --- Phase 1: compute containment depth for selected boundary cycles ---
    # Collect all (cycle_index, container_index) pairs needing PIP tests.
    # Pre-filter on same-row and strictly-larger-area to minimise pair count.
    selected_containment_depth: dict[int, int] = {ci: 0 for ci in selected_boundary_indices}

    # Group by row to avoid O(N^2) row equality checks
    _depth_by_row: dict[int, list[int]] = defaultdict(list)
    for ci in selected_boundary_indices:
        _depth_by_row[cycle_rows[ci]].append(ci)

    depth_pairs: list[tuple[int, int]] = []
    for row_cycles in _depth_by_row.values():
        for cycle_index in row_cycles:
            ca = cycle_areas[cycle_index]
            for container_index in row_cycles:
                if container_index != cycle_index and cycle_areas[container_index] > ca:
                    depth_pairs.append((cycle_index, container_index))

    if len(depth_pairs) >= _BATCH_PIP_GPU_THRESHOLD and cp is not None:
        gpu_results = _batch_point_in_ring_gpu(depth_pairs, cycle_samples, cycle_rings)
        for i, (cycle_index, _) in enumerate(depth_pairs):
            if gpu_results[i]:
                selected_containment_depth[cycle_index] += 1
    else:
        for cycle_index, container_index in depth_pairs:
            sample_x, sample_y = cycle_samples[cycle_index]
            if _point_in_ring(sample_x, sample_y, cycle_rings[container_index]):
                selected_containment_depth[cycle_index] += 1

    exterior_indices = sorted(
        (
            cycle_index
            for cycle_index in selected_boundary_indices
            if selected_containment_depth[cycle_index] % 2 == 0
        ),
        key=lambda cycle_index: (cycle_rows[cycle_index], cycle_areas[cycle_index], cycle_index),
    )
    exterior_indices_set = set(exterior_indices)

    # --- Phase 2: assign non-exterior cycles to their containing exterior ---
    # Group exteriors by row for O(row_exteriors) lookup instead of O(all_exteriors).
    # Sort each row's exteriors by area ascending so the first PIP hit is the
    # smallest containing exterior (immediate parent).
    exteriors_by_row: dict[int, list[int]] = defaultdict(list)
    for ei in exterior_indices:
        exteriors_by_row[cycle_rows[ei]].append(ei)
    for row_list in exteriors_by_row.values():
        row_list.sort(key=lambda ei: cycle_areas[ei])

    hole_assign_pairs: list[tuple[int, int]] = []
    hole_assign_map: dict[int, list[int]] = defaultdict(list)
    for cycle_index in cycle_rings:
        if cycle_index in exterior_indices_set:
            continue
        row = cycle_rows[cycle_index]
        ca = cycle_areas[cycle_index]
        for exterior_index in exteriors_by_row.get(row, []):
            if cycle_areas[exterior_index] > ca:
                pair_idx = len(hole_assign_pairs)
                hole_assign_pairs.append((cycle_index, exterior_index))
                hole_assign_map[cycle_index].append(pair_idx)

    hole_map: dict[int, list[int]] = {cycle_index: [] for cycle_index in exterior_indices}
    candidate_holes: dict[int, int] = {}

    if len(hole_assign_pairs) >= _BATCH_PIP_GPU_THRESHOLD and cp is not None:
        gpu_results = _batch_point_in_ring_gpu(hole_assign_pairs, cycle_samples, cycle_rings)
        for cycle_index in cycle_rings:
            if cycle_index in exterior_indices_set:
                continue
            containing_exteriors = [
                hole_assign_pairs[pi][1]
                for pi in hole_assign_map.get(cycle_index, [])
                if gpu_results[pi]
            ]
            if not containing_exteriors:
                continue
            container = min(containing_exteriors, key=lambda ei: cycle_areas[ei])
            candidate_holes[cycle_index] = container
    else:
        for cycle_index in cycle_rings:
            if cycle_index in exterior_indices_set:
                continue
            row = cycle_rows[cycle_index]
            ca = cycle_areas[cycle_index]
            sample_x, sample_y = cycle_samples[cycle_index]
            # Exteriors sorted ascending by area -- first PIP hit is immediate parent.
            for exterior_index in exteriors_by_row.get(row, []):
                if cycle_areas[exterior_index] > ca and _point_in_ring(
                    sample_x, sample_y, cycle_rings[exterior_index],
                ):
                    candidate_holes[cycle_index] = exterior_index
                    break

    # --- Phase 3: compute local depth among sibling holes ---
    # For each candidate hole, count how many OTHER candidate holes sharing
    # the same container with strictly larger area contain its sample point.
    holes_by_container: dict[int, list[int]] = defaultdict(list)
    for ci, ctr in candidate_holes.items():
        holes_by_container[ctr].append(ci)

    local_depth_pairs: list[tuple[int, int]] = []
    local_depth_map: dict[int, list[int]] = defaultdict(list)
    for container, siblings in holes_by_container.items():
        for cycle_index in siblings:
            ca = cycle_areas[cycle_index]
            for other_index in siblings:
                if other_index != cycle_index and cycle_areas[other_index] > ca:
                    pair_idx = len(local_depth_pairs)
                    local_depth_pairs.append((cycle_index, other_index))
                    local_depth_map[cycle_index].append(pair_idx)

    if len(local_depth_pairs) >= _BATCH_PIP_GPU_THRESHOLD and cp is not None:
        gpu_results = _batch_point_in_ring_gpu(local_depth_pairs, cycle_samples, cycle_rings)
        for cycle_index, container in candidate_holes.items():
            local_depth = sum(
                1 for pi in local_depth_map.get(cycle_index, []) if gpu_results[pi]
            )
            if local_depth % 2 != 0:
                continue
            hole_map[container].append(cycle_index)
    else:
        for cycle_index, container in candidate_holes.items():
            sample_x, sample_y = cycle_samples[cycle_index]
            local_depth = sum(
                1
                for other_index in holes_by_container[container]
                if other_index != cycle_index
                and cycle_areas[other_index] > cycle_areas[cycle_index]
                and _point_in_ring(sample_x, sample_y, cycle_rings[other_index])
            )
            if local_depth % 2 != 0:
                continue
            hole_map[container].append(cycle_index)

    row_polygons: dict[int, list[list[np.ndarray]]] = {}
    for exterior_index in exterior_indices:
        rings = [cycle_rings[exterior_index], *(cycle_rings[hole_index] for hole_index in sorted(hole_map[exterior_index]))]
        row_polygons.setdefault(cycle_rows[exterior_index], []).append(rings)

    return _build_overlay_output_rows(row_polygons, faces.runtime_selection)
