from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from vibespatial.adaptive_runtime import plan_dispatch_selection, plan_kernel_dispatch
from vibespatial.crossover import DispatchDecision
from vibespatial.fallbacks import record_fallback_event
from vibespatial.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.owned_geometry import from_shapely_geometries
from vibespatial.point_constructive import point_buffer_owned_array
from vibespatial.precision import KernelClass
from vibespatial.runtime import ExecutionMode, has_gpu_runtime

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
    result = np.empty(len(geometries), dtype=object)
    fast_rows: list[int] = []
    fallback_rows: list[int] = []

    for row_index, geometry in enumerate(geometries):
        radius = float(distances[row_index])
        if geometry is None:
            result[row_index] = None
            continue
        if getattr(geometry, "is_empty", False):
            result[row_index] = shapely.buffer(geometry, radius, quad_segs=quad_segs)
            fast_rows.append(row_index)
            continue
        if geometry.geom_type == "Point" and radius > 0.0:
            result[row_index] = _point_ring(geometry, radius, quad_segs)
            fast_rows.append(row_index)
            continue
        fallback_rows.append(row_index)

    if fallback_rows:
        fallback_index = np.asarray(fallback_rows, dtype=np.int32)
        result[fallback_index] = shapely.buffer(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
        )
    else:
        fallback_index = np.asarray([], dtype=np.int32)

    return BufferKernelResult(
        geometries=result,
        row_count=len(result),
        fast_rows=np.asarray(fast_rows, dtype=np.int32),
        fallback_rows=fallback_index,
    )


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


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
    result = np.empty(len(geometries), dtype=object)
    fast_rows: list[int] = []
    fallback_rows: list[int] = []

    for row_index, geometry in enumerate(geometries):
        if geometry is None:
            result[row_index] = None
            continue
        if getattr(geometry, "is_empty", False):
            result[row_index] = geometry
            fast_rows.append(row_index)
            continue
        if geometry.geom_type != "LineString" or join_style == "round":
            fallback_rows.append(row_index)
            continue
        offset = _offset_single_linestring(
            geometry,
            float(distances[row_index]),
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
        if offset is None:
            fallback_rows.append(row_index)
            continue
        result[row_index] = offset
        fast_rows.append(row_index)

    if fallback_rows:
        fallback_index = np.asarray(fallback_rows, dtype=np.int32)
        result[fallback_index] = shapely.offset_curve(
            geometries[fallback_index],
            distances[fallback_index],
            quad_segs=quad_segs,
            join_style=join_style,
            mitre_limit=mitre_limit,
        )
    else:
        fallback_index = np.asarray([], dtype=np.int32)

    return OffsetCurveKernelResult(
        geometries=result,
        row_count=len(result),
        fast_rows=np.asarray(fast_rows, dtype=np.int32),
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
):
    from vibespatial.execution_trace import execution_trace

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
                owned = from_shapely_geometries(geometries.tolist())
                result = point_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return np.asarray(result.to_shapely(), dtype=object), ExecutionMode.GPU

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
            from vibespatial.linestring_constructive import linestring_buffer_owned_array

            if plan_dispatch_selection(
                kernel_name="linestring_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
            ).selected is ExecutionMode.GPU:
                owned = from_shapely_geometries(geometries.tolist())
                result = linestring_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    cap_style=cap_style,
                    join_style=join_style,
                    mitre_limit=mitre_limit,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return np.asarray(result.to_shapely(), dtype=object), ExecutionMode.GPU

        # --- Polygon buffer surface ---
        if _supports_polygon_buffer_gpu_surface(
            geometries,
            quad_segs=quad_segs,
            single_sided=single_sided,
        ):
            from vibespatial.polygon_constructive import polygon_buffer_owned_array

            if plan_dispatch_selection(
                kernel_name="polygon_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
            ).selected is ExecutionMode.GPU:
                owned = from_shapely_geometries(geometries.tolist())
                result = polygon_buffer_owned_array(
                    owned,
                    distance,
                    quad_segs=quad_segs,
                    join_style=join_style,
                    mitre_limit=mitre_limit,
                    dispatch_mode=ExecutionMode.GPU,
                )
                return np.asarray(result.to_shapely(), dtype=object), ExecutionMode.GPU

        return None, ExecutionMode.CPU


def evaluate_geopandas_offset_curve(
    values,
    distance,
    *,
    quad_segs: int,
    join_style,
    mitre_limit: float,
):
    from vibespatial.execution_trace import execution_trace

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
    start = perf_counter()
    owned = point_buffer_owned(geometries, distance, quad_segs=quad_segs)
    owned_elapsed = perf_counter() - start

    start = perf_counter()
    shapely.buffer(geometries, distance, quad_segs=quad_segs)
    shapely_elapsed = perf_counter() - start

    return StrokeBenchmark(
        dataset=dataset,
        rows=len(geometries),
        fast_rows=int(owned.fast_rows.size),
        fallback_rows=int(owned.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )


def benchmark_offset_curve(values, *, distance: float, join_style: str = "mitre", dataset: str = "offset-curve") -> StrokeBenchmark:
    geometries = np.asarray(values, dtype=object)
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
