from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection, plan_kernel_dispatch
from vibespatial.runtime.crossover import DispatchDecision
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.runtime.precision import KernelClass

from .point import point_buffer_owned_array

_EPSILON = 1e-12
_POINT_TYPE_ID = 0
_LINESTRING_TYPE_ID = 1
_POLYGON_TYPE_ID = 3


class StrokeOperation(StrEnum):
    BUFFER = "buffer"
    OFFSET_CURVE = "offset_curve"


class StrokePrimitive(StrEnum):
    EXPAND_DISTANCES = "expand_distances"
    EMIT_EDGE_FRAMES = "emit_edge_frames"
    CLASSIFY_VERTICES = "classify_vertices"
    EMIT_ARCS = "emit_arcs"
    PREFIX_SUM = "prefix_sum"
    SCATTER = "scatter"
    EMIT_GEOMETRY = "emit_geometry"


@dataclass(frozen=True)
class StrokeKernelStage:
    name: str
    primitive: StrokePrimitive
    purpose: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cccl_mapping: tuple[str, ...]
    disposition: IntermediateDisposition
    geometry_producing: bool = False


@dataclass(frozen=True)
class StrokeKernelPlan:
    operation: StrokeOperation
    stages: tuple[StrokeKernelStage, ...]
    fusion_steps: tuple[PipelineStep, ...]
    reason: str


@dataclass(frozen=True)
class BufferKernelResult:
    geometries: np.ndarray
    row_count: int
    fast_rows: np.ndarray
    fallback_rows: np.ndarray


@dataclass(frozen=True)
class OffsetCurveKernelResult:
    geometries: np.ndarray
    row_count: int
    fast_rows: np.ndarray
    fallback_rows: np.ndarray


@dataclass(frozen=True)
class StrokeBenchmark:
    dataset: str
    rows: int
    fast_rows: int
    fallback_rows: int
    owned_elapsed_seconds: float
    shapely_elapsed_seconds: float

    @property
    def speedup_vs_shapely(self) -> float:
        if self.owned_elapsed_seconds == 0.0:
            return float("inf")
        return self.shapely_elapsed_seconds / self.owned_elapsed_seconds


def plan_stroke_kernel(operation: StrokeOperation | str) -> StrokeKernelPlan:
    normalized = operation if isinstance(operation, StrokeOperation) else StrokeOperation(operation)
    stages = [
        StrokeKernelStage(
            name="expand_distances",
            primitive=StrokePrimitive.EXPAND_DISTANCES,
            purpose="Broadcast scalar or per-row distances into a dense device-side stroke vector.",
            inputs=("distance",),
            outputs=("stroke_distance",),
            cccl_mapping=("transform",),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        StrokeKernelStage(
            name="emit_edge_frames",
            primitive=StrokePrimitive.EMIT_EDGE_FRAMES,
            purpose="Compute unit tangents and normals for each segment before join and cap resolution.",
            inputs=("geometry_rows", "stroke_distance"),
            outputs=("segment_frames",),
            cccl_mapping=("transform", "gather"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
        StrokeKernelStage(
            name="classify_vertices",
            primitive=StrokePrimitive.CLASSIFY_VERTICES,
            purpose="Classify joins, caps, and ambiguous vertices so only hard cases spill to slower paths.",
            inputs=("segment_frames",),
            outputs=("vertex_classes",),
            cccl_mapping=("transform", "DeviceSelect"),
            disposition=IntermediateDisposition.EPHEMERAL,
        ),
    ]
    if normalized is StrokeOperation.BUFFER:
        stages.append(
            StrokeKernelStage(
                name="emit_arcs",
                primitive=StrokePrimitive.EMIT_ARCS,
                purpose="Emit quarter-circle arc samples for round joins and point buffers without per-vertex Python branching.",
                inputs=("vertex_classes", "stroke_distance"),
                outputs=("arc_samples",),
                cccl_mapping=("prefix_sum", "scatter"),
                disposition=IntermediateDisposition.EPHEMERAL,
            )
        )
    stages.extend(
        [
            StrokeKernelStage(
                name="prefix_counts",
                primitive=StrokePrimitive.PREFIX_SUM,
                purpose="Prefix-sum output vertex counts to allocate one contiguous result buffer.",
                inputs=("vertex_classes",),
                outputs=("output_offsets",),
                cccl_mapping=("prefix_sum",),
                disposition=IntermediateDisposition.EPHEMERAL,
            ),
            StrokeKernelStage(
                name="scatter_vertices",
                primitive=StrokePrimitive.SCATTER,
                purpose="Scatter emitted offset vertices or arc samples into deterministic output order.",
                inputs=("output_offsets",),
                outputs=("output_vertices",),
                cccl_mapping=("scatter",),
                disposition=IntermediateDisposition.EPHEMERAL,
            ),
            StrokeKernelStage(
                name="emit_geometry",
                primitive=StrokePrimitive.EMIT_GEOMETRY,
                purpose="Assemble final polygon or offset-curve geometry buffers.",
                inputs=("output_vertices",),
                outputs=("geometry_buffers",),
                cccl_mapping=("gather", "scatter"),
                disposition=IntermediateDisposition.PERSIST,
                geometry_producing=True,
            ),
        ]
    )
    fusion_steps = (
        PipelineStep(name="stroke_distance", kind=StepKind.DERIVED, output_name="stroke_distance"),
        PipelineStep(name="segment_frames", kind=StepKind.DERIVED, output_name="segment_frames"),
        PipelineStep(name="vertex_classes", kind=StepKind.FILTER, output_name="vertex_classes"),
        PipelineStep(name="output_offsets", kind=StepKind.ORDERING, output_name="output_offsets"),
        PipelineStep(
            name="geometry_buffers",
            kind=StepKind.GEOMETRY,
            output_name="geometry_buffers",
            reusable_output=True,
        ),
    )
    return StrokeKernelPlan(
        operation=normalized,
        stages=tuple(stages),
        fusion_steps=fusion_steps,
        reason=(
            "Stroke-style constructive kernels should expand distances once, derive edge frames in bulk, classify joins "
            "and caps, and then emit vertices through prefix-sum and scatter so future GPU implementations can lean on "
            "CCCL primitives instead of per-geometry Python control flow."
        ),
    )


def fusion_plan_for_stroke(operation: StrokeOperation | str):
    return plan_fusion(plan_stroke_kernel(operation).fusion_steps)


def _normalize_distances(distance, row_count: int) -> np.ndarray:
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


def point_buffer_owned(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 16,
) -> BufferKernelResult:
    geometries = np.asarray(values, dtype=object)
    distances = _normalize_distances(distance, len(geometries))
    row_count = len(geometries)
    result = np.empty(row_count, dtype=object)

    # Vectorized classification: identify nulls, empties, points, and fallback rows
    non_null_mask = np.array([g is not None for g in geometries], dtype=bool)
    type_ids = np.full(row_count, -1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    if np.any(non_null_mask):
        non_null_geoms = geometries[non_null_mask]
        type_ids[non_null_mask] = shapely.get_type_id(non_null_geoms)
        empty_mask[non_null_mask] = shapely.is_empty(non_null_geoms)

    # Points with positive radius -> vectorized buffer
    point_mask = non_null_mask & ~empty_mask & (type_ids == _POINT_TYPE_ID) & (distances > 0.0)
    # Empty geometries -> Shapely handles individually
    empty_rows_mask = non_null_mask & empty_mask
    # Fallback: non-null, non-empty, non-point-positive-radius
    fallback_mask = non_null_mask & ~empty_mask & ~point_mask

    # Handle null rows
    result[~non_null_mask] = None

    # Handle empty rows via Shapely batch
    if np.any(empty_rows_mask):
        empty_idx = np.flatnonzero(empty_rows_mask)
        result[empty_idx] = shapely.buffer(geometries[empty_idx], distances[empty_idx], quad_segs=quad_segs)

    # Vectorized point buffer: build all rings at once using numpy broadcasting,
    # then batch-construct polygons via shapely.polygons() to avoid per-row overhead
    point_rows = np.flatnonzero(point_mask)
    if point_rows.size > 0:
        point_geoms = geometries[point_rows]
        point_radii = distances[point_rows]
        px = shapely.get_x(point_geoms)
        py = shapely.get_y(point_geoms)
        n_arc = max(int(quad_segs), 1) * 4
        verts_per_ring = n_arc + 1  # closed ring
        angles = np.linspace(0.0, -2.0 * np.pi, num=n_arc, endpoint=False, dtype=np.float64)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        # (n_points, n_arc) via broadcasting
        all_x = px[:, None] + point_radii[:, None] * cos_a[None, :]
        all_y = py[:, None] + point_radii[:, None] * sin_a[None, :]
        all_x[np.abs(all_x) <= _EPSILON] = 0.0
        all_y[np.abs(all_y) <= _EPSILON] = 0.0
        # Close the ring: append first vertex
        all_x = np.column_stack((all_x, all_x[:, 0]))
        all_y = np.column_stack((all_y, all_y[:, 0]))
        # Flatten to interleaved coordinate array for shapely.polygons
        n_points = point_rows.size
        flat_coords = np.empty((n_points * verts_per_ring, 2), dtype=np.float64)
        flat_coords[:, 0] = all_x.ravel()
        flat_coords[:, 1] = all_y.ravel()
        # Per-coordinate ring index: each vertex maps to its ring
        ring_indices = np.repeat(np.arange(n_points, dtype=np.intp), verts_per_ring)
        polys = shapely.polygons(shapely.linearrings(flat_coords, indices=ring_indices))
        result[point_rows] = polys

    # Fallback rows via Shapely batch
    fallback_index = np.flatnonzero(fallback_mask).astype(np.int32)
    if fallback_index.size > 0:
        result[fallback_index] = shapely.buffer(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
        )

    fast_index = np.flatnonzero(point_mask | empty_rows_mask).astype(np.int32)

    return BufferKernelResult(
        geometries=result,
        row_count=len(result),
        fast_rows=fast_index,
        fallback_rows=fallback_index,
    )


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _batch_mitre_offset_uniform(
    flat_coords: np.ndarray,
    dists: np.ndarray,
    verts_per_line: int,
    *,
    mitre_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fully vectorized mitre offset for linestrings that all have the same vertex count.

    Returns (linestring_array, ok_mask) where ok_mask[i] indicates success.
    """
    n_lines = flat_coords.shape[0] // verts_per_line
    # Reshape to (n_lines, verts_per_line, 2)
    coords = flat_coords.reshape(n_lines, verts_per_line, 2)

    # segments: (n_lines, n_segs, 2)
    segments = coords[:, 1:, :] - coords[:, :-1, :]
    lengths = np.linalg.norm(segments, axis=2)  # (n_lines, n_segs)

    # Check for degenerate segments
    has_degen = np.any(lengths <= _EPSILON, axis=1)  # (n_lines,)

    # directions and normals: safe division (mask degenerate later)
    safe_lengths = np.where(lengths > _EPSILON, lengths, 1.0)
    directions = segments / safe_lengths[:, :, None]  # (n_lines, n_segs, 2)
    signs = np.where(dists >= 0.0, 1.0, -1.0)  # (n_lines,)
    magnitudes = np.abs(dists)  # (n_lines,)
    # normals: (-dy, dx) * sign
    normals = np.empty_like(directions)
    normals[:, :, 0] = -directions[:, :, 1] * signs[:, None]
    normals[:, :, 1] = directions[:, :, 0] * signs[:, None]

    # Output: (n_lines, verts_per_line, 2)
    out = np.empty_like(coords)
    # First vertex: coords[0] + magnitude * normals[0]
    out[:, 0, :] = coords[:, 0, :] + magnitudes[:, None] * normals[:, 0, :]
    # Last vertex: coords[-1] + magnitude * normals[-1]
    out[:, -1, :] = coords[:, -1, :] + magnitudes[:, None] * normals[:, -1, :]

    ok_mask = ~has_degen

    if verts_per_line > 2:
        # Inner vertices: vectorized mitre intersection across all rows
        prev_dirs = directions[:, :-1, :]   # (n_lines, n_inner, 2)
        next_dirs = directions[:, 1:, :]
        inner_coords = coords[:, 1:-1, :]
        prev_norms = normals[:, :-1, :]
        next_norms = normals[:, 1:, :]
        prev_shifts = inner_coords + magnitudes[:, None, None] * prev_norms
        next_shifts = inner_coords + magnitudes[:, None, None] * next_norms

        # Cross products: (n_lines, n_inner)
        denoms = prev_dirs[:, :, 0] * next_dirs[:, :, 1] - prev_dirs[:, :, 1] * next_dirs[:, :, 0]
        collinear = np.abs(denoms) <= _EPSILON
        safe_denoms = np.where(collinear, 1.0, denoms)

        deltas = next_shifts - prev_shifts
        t = (deltas[:, :, 0] * next_dirs[:, :, 1] - deltas[:, :, 1] * next_dirs[:, :, 0]) / safe_denoms
        intersections = prev_shifts + t[:, :, None] * prev_dirs

        # Mitre limit check
        miter_dists = np.linalg.norm(intersections - next_shifts, axis=2)
        safe_mags = np.where(magnitudes > _EPSILON, magnitudes, _EPSILON)
        miter_ratios = miter_dists / safe_mags[:, None]
        exceeded = ~collinear & (miter_ratios > mitre_limit)
        ok_mask &= ~np.any(exceeded, axis=1)

        out[:, 1:-1, :] = np.where(collinear[:, :, None], next_shifts, intersections)

    # Flatten back and batch-construct linestrings
    out_flat = out.reshape(-1, 2)
    line_indices = np.repeat(np.arange(n_lines, dtype=np.intp), verts_per_line)
    lines = shapely.linestrings(out_flat, indices=line_indices)

    return lines, ok_mask


def _offset_from_coords_mitre(coords: np.ndarray, distance: float, *, mitre_limit: float) -> np.ndarray | None:
    """Vectorized mitre offset on pre-extracted 2D coordinate array. Returns output coords or None on failure."""
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

    # For mitre joins, output has same number of vertices as input
    out = np.empty((n_verts, 2), dtype=np.float64)
    out[0] = coords[0] + magnitude * normals[0]
    out[-1] = coords[-1] + magnitude * normals[-1]

    if n_verts > 2:
        # Vectorized inner vertex computation
        prev_dirs = directions[:-1]   # (n_verts-2, 2)
        next_dirs = directions[1:]    # (n_verts-2, 2)
        inner_coords = coords[1:-1]   # (n_verts-2, 2)
        prev_norms = normals[:-1]
        next_norms = normals[1:]
        prev_shifts = inner_coords + magnitude * prev_norms
        next_shifts = inner_coords + magnitude * next_norms

        # Cross products for all inner vertices at once
        denoms = prev_dirs[:, 0] * next_dirs[:, 1] - prev_dirs[:, 1] * next_dirs[:, 0]
        collinear = np.abs(denoms) <= _EPSILON

        # For collinear vertices, use next_shift
        deltas = next_shifts - prev_shifts
        # Avoid division by zero for collinear cases
        safe_denoms = np.where(collinear, 1.0, denoms)
        t = (deltas[:, 0] * next_dirs[:, 1] - deltas[:, 1] * next_dirs[:, 0]) / safe_denoms
        intersections = prev_shifts + t[:, None] * prev_dirs

        # Check mitre limit
        miter_dists = np.linalg.norm(intersections - next_shifts, axis=1)
        miter_ratios = miter_dists / max(magnitude, _EPSILON)
        if np.any(~collinear & (miter_ratios > mitre_limit)):
            return None

        out[1:-1] = np.where(collinear[:, None], next_shifts, intersections)

    return out


def _offset_single_linestring(line: LineString, distance: float, *, join_style: str, mitre_limit: float) -> LineString | None:
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


def offset_curve_owned(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OffsetCurveKernelResult:
    geometries = np.asarray(values, dtype=object)
    distances = _normalize_distances(distance, len(geometries))
    row_count = len(geometries)
    result = np.empty(row_count, dtype=object)

    # Vectorized classification
    non_null_mask = np.array([g is not None for g in geometries], dtype=bool)
    type_ids = np.full(row_count, -1, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    if np.any(non_null_mask):
        non_null_geoms = geometries[non_null_mask]
        type_ids[non_null_mask] = shapely.get_type_id(non_null_geoms)
        empty_mask[non_null_mask] = shapely.is_empty(non_null_geoms)

    # Nulls
    result[~non_null_mask] = None

    # Empty geometries -> pass through
    empty_rows_mask = non_null_mask & empty_mask
    if np.any(empty_rows_mask):
        empty_idx = np.flatnonzero(empty_rows_mask)
        result[empty_idx] = geometries[empty_idx]

    # LineString + non-round join -> owned fast path
    linestring_mask = non_null_mask & ~empty_mask & (type_ids == _LINESTRING_TYPE_ID) & (join_style != "round")
    # Everything else -> Shapely fallback
    fallback_mask = non_null_mask & ~empty_mask & ~linestring_mask

    # Process owned linestring rows -- batch coordinate extraction, then vectorized offset
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
            # Fully vectorized path: all linestrings have the same vertex count
            batch_lines, batch_ok = _batch_mitre_offset_uniform(
                all_coords, line_dists, int(unique_counts[0]), mitre_limit=mitre_limit,
            )
            ok_mask = batch_ok
            fail_mask = ~ok_mask
            ok_local = np.flatnonzero(ok_mask)
            fail_local = np.flatnonzero(fail_mask)
            if ok_local.size > 0:
                result[linestring_rows[ok_local]] = batch_lines[ok_local]
                fast_list.extend(linestring_rows[ok_local].tolist())
            deferred_fallback.extend(linestring_rows[fail_local].tolist())
        else:
            # Mixed vertex counts: per-row offset with batch construction
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

    # Batch Shapely fallback
    fallback_index = np.asarray(deferred_fallback, dtype=np.int32)
    if fallback_index.size > 0:
        result[fallback_index] = shapely.offset_curve(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )

    return OffsetCurveKernelResult(
        geometries=result,
        row_count=row_count,
        fast_rows=np.asarray(sorted(fast_list), dtype=np.int32),
        fallback_rows=fallback_index,
    )


def _non_null_mask(values: np.ndarray) -> np.ndarray:
    return np.fromiter((geometry is not None for geometry in values), dtype=bool, count=len(values))


def _supports_point_buffer_surface(
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


def _supports_point_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    cap_style,
    join_style,
    single_sided: bool,
) -> bool:
    if not _supports_point_buffer_surface(
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


def _supports_linestring_buffer_surface(
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


def _supports_linestring_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    single_sided: bool,
) -> bool:
    if not _supports_linestring_buffer_surface(
        geometries,
        single_sided=single_sided,
    ):
        return False
    if quad_segs < 1 or len(geometries) == 0:
        return False
    non_null = _non_null_mask(geometries)
    return bool(np.all(non_null) and not np.any(shapely.is_empty(geometries)))


def _supports_polygon_buffer_surface(
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


def _supports_polygon_buffer_gpu_surface(
    geometries: np.ndarray,
    *,
    quad_segs: int,
    single_sided: bool,
) -> bool:
    if not _supports_polygon_buffer_surface(
        geometries,
        single_sided=single_sided,
    ):
        return False
    if quad_segs < 1 or len(geometries) == 0:
        return False
    non_null = _non_null_mask(geometries)
    if not bool(np.all(non_null) and not np.any(shapely.is_empty(geometries))):
        return False
    return True


def _supports_offset_curve_surface(geometries: np.ndarray, *, join_style) -> bool:
    if join_style == "round":
        return False
    non_null = _non_null_mask(geometries)
    if not np.any(non_null):
        return True
    type_ids = np.asarray(shapely.get_type_id(geometries[non_null]), dtype=np.int32)
    return bool(np.all(type_ids == _LINESTRING_TYPE_ID))


def evaluate_geopandas_buffer(
    values,
    distance,
    *,
    quad_segs: int,
    cap_style,
    join_style,
    mitre_limit: float,
    single_sided: bool,
    prebuilt_owned=None,
):
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("buffer"):
        geometries = np.asarray(values, dtype=object)
        detail = (
            f"cap_style={cap_style}, join_style={join_style}, mitre_limit={mitre_limit}, "
            f"single_sided={single_sided}, quad_segs={quad_segs}, rows={len(geometries)}"
        )
        # --- Point buffer surface ---
        if _supports_point_buffer_surface(
            geometries,
            cap_style=cap_style,
            join_style=join_style,
            single_sided=single_sided,
        ):
            gpu_available = has_gpu_runtime() and _supports_point_buffer_gpu_surface(
                geometries,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                single_sided=single_sided,
            )
            plan = plan_kernel_dispatch(
                kernel_name="point_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
                gpu_available=gpu_available,
            )
            dispatch_decision = plan.dispatch_decision
            if dispatch_decision is DispatchDecision.GPU:
                owned = prebuilt_owned if prebuilt_owned is not None else from_shapely_geometries(geometries.tolist())
                result = point_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return result, ExecutionMode.GPU

            result = point_buffer_owned(geometries, distance, quad_segs=quad_segs)
            if result.fallback_rows.size == 0:
                return np.asarray(result.geometries, dtype=object), ExecutionMode.CPU
            record_fallback_event(
                surface="geopandas.array.buffer",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="repo-owned buffer kernel still needs host fallback rows on this input; using explicit CPU fallback",
                detail=detail,
                pipeline="constructive/buffer",
            )
            return None, ExecutionMode.CPU

        # --- LineString buffer surface ---
        if _supports_linestring_buffer_gpu_surface(
            geometries,
            quad_segs=quad_segs,
            single_sided=single_sided,
        ):
            from .linestring import linestring_buffer_owned_array

            if plan_dispatch_selection(
                kernel_name="linestring_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
            ).selected is ExecutionMode.GPU:
                owned = prebuilt_owned if prebuilt_owned is not None else from_shapely_geometries(geometries.tolist())
                result = linestring_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    cap_style=cap_style,
                    join_style=join_style,
                    mitre_limit=mitre_limit,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return result, ExecutionMode.GPU

        # --- Polygon buffer surface ---
        if _supports_polygon_buffer_gpu_surface(
            geometries,
            quad_segs=quad_segs,
            single_sided=single_sided,
        ):
            from .polygon import polygon_buffer_owned_array

            if plan_dispatch_selection(
                kernel_name="polygon_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
            ).selected is ExecutionMode.GPU:
                owned = prebuilt_owned if prebuilt_owned is not None else from_shapely_geometries(geometries.tolist())
                result = polygon_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    join_style=join_style,
                    mitre_limit=mitre_limit,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return result, ExecutionMode.GPU

        return None, ExecutionMode.CPU


def evaluate_geopandas_offset_curve(
    values,
    distance,
    *,
    quad_segs: int,
    join_style,
    mitre_limit: float,
):
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace("offset_curve"):
        geometries = np.asarray(values, dtype=object)
        detail = f"join_style={join_style}, mitre_limit={mitre_limit}, quad_segs={quad_segs}, rows={len(geometries)}"
        if not _supports_offset_curve_surface(geometries, join_style=join_style):
            record_fallback_event(
                surface="geopandas.array.offset_curve",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="repo-owned offset-curve kernel cannot claim the GeoPandas surface for current rows/kwargs; using explicit CPU fallback",
                detail=detail,
                pipeline="constructive/offset_curve",
            )
            return None, ExecutionMode.CPU

        result = offset_curve_owned(
            geometries,
            distance,
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        if result.fallback_rows.size == 0:
            return np.asarray(result.geometries, dtype=object), ExecutionMode.CPU
        record_fallback_event(
            surface="geopandas.array.offset_curve",
            reason="repo-owned offset-curve kernel still needs host fallback rows on this input; using explicit CPU fallback",
            detail=detail,
            pipeline="constructive/offset_curve",
        )
        return None, ExecutionMode.CPU


def benchmark_point_buffer(values, *, distance: float, quad_segs: int = 16, dataset: str = "point-buffer") -> StrokeBenchmark:
    geometries = np.asarray(values, dtype=object)

    # Try GPU dispatch via point_buffer_owned_array (the real GPU kernel path)
    if has_gpu_runtime() and _supports_point_buffer_gpu_surface(
        geometries,
        quad_segs=quad_segs,
        cap_style="round",
        join_style="round",
        single_sided=False,
    ):
        from vibespatial.geometry.owned import from_shapely_geometries

        owned_array = from_shapely_geometries(list(geometries))
        # Warmup run to exclude JIT/compilation overhead
        point_buffer_owned_array(owned_array, distance, quad_segs=quad_segs, dispatch_mode=ExecutionMode.GPU)
        start = perf_counter()
        point_buffer_owned_array(owned_array, distance, quad_segs=quad_segs, dispatch_mode=ExecutionMode.GPU)
        owned_elapsed = perf_counter() - start
        fast_rows = len(geometries)
        fallback_rows = 0
    else:
        start = perf_counter()
        result_cpu = point_buffer_owned(geometries, distance, quad_segs=quad_segs)
        owned_elapsed = perf_counter() - start
        fast_rows = int(result_cpu.fast_rows.size)
        fallback_rows = int(result_cpu.fallback_rows.size)

    # Warmup Shapely too for fairness
    shapely.buffer(geometries, distance, quad_segs=quad_segs)
    start = perf_counter()
    shapely.buffer(geometries, distance, quad_segs=quad_segs)
    shapely_elapsed = perf_counter() - start

    return StrokeBenchmark(
        dataset=dataset,
        rows=len(geometries),
        fast_rows=fast_rows,
        fallback_rows=fallback_rows,
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )


def benchmark_offset_curve(values, *, distance: float, join_style: str = "mitre", dataset: str = "offset-curve") -> StrokeBenchmark:
    geometries = np.asarray(values, dtype=object)

    # Warmup both paths
    offset_curve_owned(geometries, distance, join_style=join_style)
    shapely.offset_curve(geometries, distance, join_style=join_style)

    start = perf_counter()
    owned = offset_curve_owned(geometries, distance, join_style=join_style)
    owned_elapsed = perf_counter() - start

    start = perf_counter()
    shapely.offset_curve(geometries, distance, join_style=join_style)
    shapely_elapsed = perf_counter() - start

    return StrokeBenchmark(
        dataset=dataset,
        rows=len(geometries),
        fast_rows=int(owned.fast_rows.size),
        fallback_rows=int(owned.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )
