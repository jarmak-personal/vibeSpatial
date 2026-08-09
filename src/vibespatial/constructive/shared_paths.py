"""Native shared-path classification and ordered collection assembly."""

from __future__ import annotations

import numpy as np

from vibespatial.constructive.shared_paths_cpu import shared_paths_cpu
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    tile_single_row,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    WorkloadShape,
    detect_workload_shape,
    estimate_pairwise_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.runtime.residency import combined_residency

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None


_LINEAL_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})


def _lineal_structure_only(owned: OwnedGeometryArray) -> bool:
    """Admit lineal/null structure without exporting row tags."""
    return all(family in _LINEAL_FAMILIES for family in owned.families)


def _release_classified_page(page) -> None:
    state = page.device_state
    if state is None:
        return
    from vibespatial.cuda._runtime import get_cuda_runtime

    runtime = get_cuda_runtime()
    for values in (
        state.left_rows,
        state.left_segments,
        state.left_lookup,
        state.right_rows,
        state.right_segments,
        state.right_lookup,
        state.kinds,
        state.point_x,
        state.point_y,
        state.overlap_x0,
        state.overlap_y0,
        state.overlap_x1,
        state.overlap_y1,
        state.ambiguous_rows,
    ):
        runtime.free(values)


def _build_candidate_overlap_lines(
    d_x0,
    d_y0,
    d_x1,
    d_y1,
) -> OwnedGeometryArray:
    capacity = int(d_x0.size)
    d_x = cp.empty(capacity * 2, dtype=cp.float64)
    d_y = cp.empty(capacity * 2, dtype=cp.float64)
    d_x[0::2] = d_x0
    d_y[0::2] = d_y0
    d_x[1::2] = d_x1
    d_y[1::2] = d_y1
    return build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_x,
                y=d_y,
                geometry_offsets=(cp.arange(capacity + 1, dtype=cp.int32) * cp.int32(2)),
                empty_mask=cp.zeros(capacity, dtype=cp.bool_),
                bounds=None,
            )
        },
        row_count=capacity,
        tags=cp.full(
            capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=cp.ones(capacity, dtype=cp.bool_),
        family_row_offsets=cp.arange(capacity, dtype=cp.int32),
        execution_mode="gpu",
    )


def _direction_atomic_lines(
    overlap_geometry: OwnedGeometryArray,
    d_rows,
    d_active,
    *,
    row_count: int,
):
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import LinePartCapacitySelection
    from vibespatial.constructive.grouped_mixed_union import (
        atomic_line_union_from_part_capacity_device,
    )

    selection = NativeDeviceSelection.from_mask(d_active)
    d_capacity_active = selection.active_capacity_mask()
    capacity_geometry = overlap_geometry._device_indexed_take(
        selection.partition_capacity_positions(),
        assume_unique_indices=True,
    )._apply_row_activity(
        d_capacity_active,
        assume_active_indices_unique=True,
    )
    d_capacity_rows = selection.gather_capacity(
        d_rows,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    return atomic_line_union_from_part_capacity_device(
        LinePartCapacitySelection(
            geometry=capacity_geometry,
            source_rows=d_capacity_rows,
            selection=selection.as_capacity_prefix(),
            coord_capacity=int(overlap_geometry.row_count) * 2,
        ),
        d_capacity_rows,
        output_row_count=row_count,
        preserve_source_orientation=True,
    )


def _atomic_lines_to_multilinestring_capacity(
    atomic_lines,
    *,
    d_validity,
    row_count: int,
) -> OwnedGeometryArray:
    """Represent every direction as a valid-empty or populated MultiLineString."""
    if atomic_lines is None:
        edge_capacity = 0
        d_edge_offsets = cp.zeros(row_count + 1, dtype=cp.int32)
        d_edge_x0 = cp.empty(0, dtype=cp.float64)
        d_edge_y0 = cp.empty(0, dtype=cp.float64)
        d_edge_x1 = cp.empty(0, dtype=cp.float64)
        d_edge_y1 = cp.empty(0, dtype=cp.float64)
    else:
        edge_capacity = int(atomic_lines.edge_x0.size)
        d_edge_offsets = cp.asarray(
            atomic_lines.edge_group_offsets,
            dtype=cp.int32,
        )
        d_edge_x0 = cp.asarray(atomic_lines.edge_x0, dtype=cp.float64)
        d_edge_y0 = cp.asarray(atomic_lines.edge_y0, dtype=cp.float64)
        d_edge_x1 = cp.asarray(atomic_lines.edge_x1, dtype=cp.float64)
        d_edge_y1 = cp.asarray(atomic_lines.edge_y1, dtype=cp.float64)

    d_x = cp.empty(edge_capacity * 2, dtype=cp.float64)
    d_y = cp.empty(edge_capacity * 2, dtype=cp.float64)
    d_x[0::2] = d_edge_x0
    d_y[0::2] = d_edge_y0
    d_x[1::2] = d_edge_x1
    d_y[1::2] = d_edge_y1
    d_part_offsets = cp.arange(edge_capacity + 1, dtype=cp.int32) * cp.int32(2)
    d_part_counts = d_edge_offsets[1:] - d_edge_offsets[:-1]
    return build_device_resident_owned(
        device_families={
            GeometryFamily.MULTILINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.MULTILINESTRING,
                x=d_x,
                y=d_y,
                geometry_offsets=d_edge_offsets,
                empty_mask=d_part_counts == 0,
                part_offsets=d_part_offsets,
                bounds=None,
            )
        },
        row_count=row_count,
        tags=cp.full(
            row_count,
            FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
            dtype=cp.int8,
        ),
        validity=cp.asarray(d_validity, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


@register_kernel_variant(
    "shared_paths",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "shared_paths"),
)
def _shared_paths_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    crs=None,
):
    """Classify overlap capacity and assemble ordered forward/backward parts."""
    if cp is None:
        raise RuntimeError("CuPy is required for native shared_paths")
    if left.row_count != right.row_count:
        raise ValueError("shared_paths GPU inputs must have equal row counts")

    from vibespatial.api._native_results import (
        _ordered_geometry_collection_from_owned_parts_at_capacity,
    )
    from vibespatial.spatial.segment_primitives import (
        PagedSegmentIntersectionResult,
        SegmentIntersectionKind,
        SegmentIntersectionResult,
        _extract_segments_gpu,
        classify_segment_intersections,
    )

    row_count = int(left.row_count)
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    d_validity = cp.asarray(left_state.validity, dtype=cp.bool_) & cp.asarray(
        right_state.validity,
        dtype=cp.bool_,
    )
    left_segments = _extract_segments_gpu(left)
    right_segments = _extract_segments_gpu(right)
    pages = []

    def _retain_page(page):
        if page.count > 0:
            pages.append(page)

    try:
        classified = classify_segment_intersections(
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
            precision=PrecisionMode.FP64,
            _cached_left_device_segments=left_segments,
            _cached_right_device_segments=right_segments,
            _require_same_row=True,
            _collect_ambiguous_rows=False,
            _compact_paged_non_disjoint=True,
            _classified_page_consumer=_retain_page,
        )
        if isinstance(classified, SegmentIntersectionResult):
            if classified.count > 0:
                pages.append(classified)
        elif not isinstance(classified, PagedSegmentIntersectionResult):
            raise RuntimeError("shared_paths received an unknown classifier result")
        if classified.runtime_selection.selected is not ExecutionMode.GPU:
            raise RuntimeError("shared_paths classifier did not remain on GPU")

        forward_atomic = None
        backward_atomic = None
        if pages:
            states = tuple(page.device_state for page in pages)
            if any(state is None for state in states):
                raise RuntimeError("shared_paths classification returned a host page")

            def _concat(field):
                values = tuple(cp.asarray(getattr(state, field)) for state in states)
                return values[0] if len(values) == 1 else cp.concatenate(values)

            d_rows = _concat("left_rows").astype(cp.int32, copy=False)
            d_left_lookup = _concat("left_lookup").astype(cp.int64, copy=False)
            d_right_lookup = _concat("right_lookup").astype(cp.int64, copy=False)
            d_kinds = _concat("kinds").astype(cp.int8, copy=False)
            d_overlap = d_kinds == cp.int8(SegmentIntersectionKind.OVERLAP)

            d_left_dx = (
                cp.asarray(left_segments.x1)[d_left_lookup]
                - cp.asarray(left_segments.x0)[d_left_lookup]
            )
            d_left_dy = (
                cp.asarray(left_segments.y1)[d_left_lookup]
                - cp.asarray(left_segments.y0)[d_left_lookup]
            )
            d_right_dx = (
                cp.asarray(right_segments.x1)[d_right_lookup]
                - cp.asarray(right_segments.x0)[d_right_lookup]
            )
            d_right_dy = (
                cp.asarray(right_segments.y1)[d_right_lookup]
                - cp.asarray(right_segments.y0)[d_right_lookup]
            )
            d_use_x = cp.abs(d_left_dx) >= cp.abs(d_left_dy)
            d_left_positive = cp.where(d_use_x, d_left_dx > 0, d_left_dy > 0)
            d_right_positive = cp.where(d_use_x, d_right_dx > 0, d_right_dy > 0)
            d_same_direction = d_left_positive == d_right_positive
            d_reverse_overlap = ~d_left_positive

            d_raw_x0 = _concat("overlap_x0").astype(cp.float64, copy=False)
            d_raw_y0 = _concat("overlap_y0").astype(cp.float64, copy=False)
            d_raw_x1 = _concat("overlap_x1").astype(cp.float64, copy=False)
            d_raw_y1 = _concat("overlap_y1").astype(cp.float64, copy=False)
            overlap_geometry = _build_candidate_overlap_lines(
                cp.where(d_reverse_overlap, d_raw_x1, d_raw_x0),
                cp.where(d_reverse_overlap, d_raw_y1, d_raw_y0),
                cp.where(d_reverse_overlap, d_raw_x0, d_raw_x1),
                cp.where(d_reverse_overlap, d_raw_y0, d_raw_y1),
            )
            forward_atomic = _direction_atomic_lines(
                overlap_geometry,
                d_rows,
                d_overlap & d_same_direction,
                row_count=row_count,
            )
            backward_atomic = _direction_atomic_lines(
                overlap_geometry,
                d_rows,
                d_overlap & ~d_same_direction,
                row_count=row_count,
            )

        output_rows = cp.arange(row_count, dtype=cp.int64)
        forward = _atomic_lines_to_multilinestring_capacity(
            forward_atomic,
            d_validity=d_validity,
            row_count=row_count,
        )
        backward = _atomic_lines_to_multilinestring_capacity(
            backward_atomic,
            d_validity=d_validity,
            row_count=row_count,
        )
        result = _ordered_geometry_collection_from_owned_parts_at_capacity(
            ((forward, output_rows), (backward, output_rows)),
            row_count=row_count,
            crs=crs,
        )
        if result is None:
            raise RuntimeError("shared_paths lost ordered native composition")
        return result
    finally:
        for page in pages:
            _release_classified_page(page)
        left_segments.free()
        right_segments.free()


def shared_paths_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
):
    """Return native ordered shared paths on GPU or explicit Shapely output on CPU."""
    row_count = int(left.row_count)
    workload = detect_workload_shape(row_count, right.row_count)
    if workload is WorkloadShape.BROADCAST_RIGHT:
        right = tile_single_row(right, row_count)
    if row_count == 0:
        return np.empty(0, dtype=object)
    if isinstance(precision, str):
        precision = PrecisionMode(precision)

    selection = plan_dispatch_selection(
        kernel_name="shared_paths",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=combined_residency(left, right),
        work_estimate=estimate_pairwise_work_from_owned(
            left,
            right,
            workload=workload,
            output_row_count=row_count,
            primary_unit_name="shared-segment-candidate",
        ),
    )
    precision_plan = selection.precision_plan

    if selection.selected is ExecutionMode.GPU:
        if _lineal_structure_only(left) and _lineal_structure_only(right):
            result = _shared_paths_gpu(left, right, crs=crs)
            record_dispatch_event(
                surface="shared_paths_owned",
                operation="shared_paths",
                implementation="shared_segment_topology_gpu",
                reason=(
                    "same-row segment overlap classification assembled ordered "
                    "forward/backward native MultiLineString capacity"
                ),
                detail=(
                    f"rows={row_count}, "
                    f"precision={precision_plan.compute_precision.value}, "
                    f"workload={workload.value}, "
                    "physical_shape=segment_candidate_atomic_line_capacity"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            return result
        record_fallback_event(
            surface="shared_paths_owned",
            reason="native shared_paths requires lineal or null geometry structure",
            detail=f"rows={row_count}",
            pipeline="shared_paths",
        )

    record_dispatch_event(
        surface="shared_paths_owned",
        operation="shared_paths",
        implementation="shapely_cpu",
        reason="GPU not available, not selected, or structurally inadmissible",
        detail=f"rows={row_count}",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    return shared_paths_cpu(left, right)
