from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.constructive.stroke_cpu import (
    benchmark_offset_curve_baseline,
    benchmark_point_buffer_baseline,
    offset_curve_owned_cpu,
    point_buffer_owned_cpu,
    supports_linestring_buffer_gpu_surface,
    supports_offset_curve_surface,
    supports_point_buffer_gpu_surface,
    supports_point_buffer_surface,
    supports_polygon_buffer_gpu_surface,
)
from vibespatial.geometry.owned import from_shapely_geometries

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.fusion import IntermediateDisposition, PipelineStep, StepKind, plan_fusion
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, combined_residency

from .point import point_buffer_owned_array


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


class BufferKernelResult:
    """Result of a buffer kernel invocation.

    When ``owned_result`` is set, ``geometries`` is materialized lazily on
    first access so that callers that stay on the device-resident path
    never pay for a D->H transfer.
    """

    __slots__ = (
        "_geometries",
        "fallback_rows",
        "fast_rows",
        "owned_result",
        "row_count",
    )

    def __init__(
        self,
        *,
        geometries: np.ndarray | None = None,
        row_count: int,
        fast_rows: np.ndarray,
        fallback_rows: np.ndarray,
        owned_result: OwnedGeometryArray | None = None,
    ) -> None:
        self._geometries = geometries
        self.row_count = row_count
        self.fast_rows = fast_rows
        self.fallback_rows = fallback_rows
        self.owned_result = owned_result

    @property
    def geometries(self) -> np.ndarray:
        if self._geometries is None:
            if self.owned_result is not None:
                self._geometries = np.asarray(
                    self.owned_result.to_shapely(), dtype=object
                )
            else:
                raise ValueError(
                    "BufferKernelResult has no geometries and no owned_result; "
                    "at least one must be provided"
                )
        return self._geometries


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


def point_buffer_owned(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 16,
) -> BufferKernelResult:
    result, fast_index, fallback_index = point_buffer_owned_cpu(
        values,
        distance,
        quad_segs=quad_segs,
    )
    return BufferKernelResult(
        geometries=result,
        row_count=len(result),
        fast_rows=fast_index,
        fallback_rows=fallback_index,
    )


def offset_curve_owned(
    values: Sequence[object | None] | np.ndarray,
    distance,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OffsetCurveKernelResult:
    result, fast_rows, fallback_rows = offset_curve_owned_cpu(
        values,
        distance,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
    return OffsetCurveKernelResult(
        geometries=result,
        row_count=len(result),
        fast_rows=fast_rows,
        fallback_rows=fallback_rows,
    )


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
        current_residency = (
            combined_residency(prebuilt_owned)
            if prebuilt_owned is not None
            else Residency.HOST
        )
        detail = (
            f"cap_style={cap_style}, join_style={join_style}, mitre_limit={mitre_limit}, "
            f"single_sided={single_sided}, quad_segs={quad_segs}, rows={len(geometries)}"
        )
        # --- Point buffer surface ---
        if supports_point_buffer_surface(
            geometries,
            cap_style=cap_style,
            join_style=join_style,
            single_sided=single_sided,
        ):
            gpu_available = has_gpu_runtime() and supports_point_buffer_gpu_surface(
                geometries,
                quad_segs=quad_segs,
                cap_style=cap_style,
                join_style=join_style,
                single_sided=single_sided,
            )
            selection = plan_dispatch_selection(
                kernel_name="point_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
                gpu_available=gpu_available,
                current_residency=current_residency,
            )
            if selection.selected is ExecutionMode.GPU:
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
        if supports_linestring_buffer_gpu_surface(
            geometries,
            quad_segs=quad_segs,
            single_sided=single_sided,
        ):
            from .linestring import (
                linestring_buffer_owned_array,
                supports_two_point_linestring_buffer_fast_path,
            )

            selection = plan_dispatch_selection(
                kernel_name="linestring_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
                current_residency=current_residency,
            )
            owned = prebuilt_owned if prebuilt_owned is not None else None
            force_two_point_gpu = False
            if selection.selected is not ExecutionMode.GPU and len(geometries) >= LINESTRING_TWO_POINT_BUFFER_GPU_THRESHOLD:
                if owned is None:
                    owned = from_shapely_geometries(geometries.tolist())
                force_two_point_gpu = supports_two_point_linestring_buffer_fast_path(
                    owned,
                    quad_segs=quad_segs,
                    cap_style=cap_style,
                    join_style=join_style,
                    single_sided=single_sided,
                )

            if selection.selected is ExecutionMode.GPU or force_two_point_gpu:
                if owned is None:
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
                return result, ExecutionMode.GPU

        # --- Polygon buffer surface ---
        if supports_polygon_buffer_gpu_surface(
            geometries,
            quad_segs=quad_segs,
            single_sided=single_sided,
        ):
            from .polygon import polygon_buffer_owned_array

            if plan_dispatch_selection(
                kernel_name="polygon_buffer",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=len(geometries),
                current_residency=current_residency,
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
        if not supports_offset_curve_surface(geometries, join_style=join_style):
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
    gpu_available = has_gpu_runtime() and supports_point_buffer_gpu_surface(
        geometries,
        quad_segs=quad_segs,
        cap_style="round",
        join_style="round",
        single_sided=False,
    )
    if plan_dispatch_selection(
        kernel_name="point_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=len(geometries),
        gpu_available=gpu_available,
    ).selected is ExecutionMode.GPU:
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

    shapely_elapsed = benchmark_point_buffer_baseline(
        geometries,
        distance,
        quad_segs=quad_segs,
    )

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

    offset_curve_owned(geometries, distance, join_style=join_style)

    start = perf_counter()
    owned = offset_curve_owned(geometries, distance, join_style=join_style)
    owned_elapsed = perf_counter() - start

    shapely_elapsed = benchmark_offset_curve_baseline(
        geometries,
        distance,
        join_style=join_style,
    )

    return StrokeBenchmark(
        dataset=dataset,
        rows=len(geometries),
        fast_rows=int(owned.fast_rows.size),
        fallback_rows=int(owned.fallback_rows.size),
        owned_elapsed_seconds=owned_elapsed,
        shapely_elapsed_seconds=shapely_elapsed,
    )
