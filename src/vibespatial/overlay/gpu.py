from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from vibespatial.cuda.cccl_precompile import request_warmup

request_warmup(
    [
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "radix_sort_i32_i32",
        "radix_sort_i64_i32",
        "radix_sort_u64_i32",
        "radix_sort_f64_i32",
        "unique_by_key_i32_i32",
        "unique_by_key_u64_i32",
        "segmented_reduce_sum_f64",
        "segmented_reduce_min_f64",
        "segmented_reduce_max_f64",
        "select_i32",
    ]
)
from vibespatial.cuda._runtime import (  # noqa: E402
    compile_kernel_group,
    get_cuda_runtime,
    maybe_trim_pool_memory,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.dispatch import record_dispatch_event  # noqa: E402
from vibespatial.runtime.fallbacks import record_fallback_event  # noqa: E402
from vibespatial.runtime.hotpath_trace import (  # noqa: E402
    attach_work_amplification,
    hotpath_stage,
    hotpath_timing_enabled,
)
from vibespatial.runtime.kernel_registry import register_kernel_variant  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import Residency  # noqa: E402
from vibespatial.spatial.segment_primitives import (  # noqa: E402
    DeviceBroadcastSegmentRelation,
    DeviceSegmentTable,
    SegmentIntersectionResult,
    _extract_segments_gpu,
)

from .types import (  # noqa: E402, F401  # Re-exported for backward compatibility
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    ComponentOverlayExecutionPlan,
    HalfEdgeGraph,
    HalfEdgeGraphDeviceState,
    MicrocellOverlayExecutionPlan,
    OverlayExecutionPlan,
    OverlayFaceDeviceState,
    OverlayFaceTable,
    PagedOverlayExecutionPlan,
    SplitEventDeviceState,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

# Peak live bytes per split event while endpoint/pair batches, packed keys,
# priority keys, sort order, and deduplicated payload overlap. This is a
# conservative execution-layout estimate, not the persistent table width.
_BYTES_PER_LIVE_SPLIT_EVENT = 192
_MIN_LIVE_SPLIT_EVENT_BUDGET = 64 * 1024
# The memory-derived one-fifth limit remains authoritative on smaller devices.
# This ceiling prevents one plan from reserving more than 6 GiB on large GPUs.
_MAX_LIVE_SPLIT_EVENT_BUDGET = 32 * 1024 * 1024


def _sync_hotpath() -> None:
    if hotpath_timing_enabled():
        get_cuda_runtime().synchronize()


@dataclass(frozen=True)
class _RowIsolatedTopologyPageShape:
    row_count: int
    rows_per_page: int
    live_event_budget: int
    worst_live_events_per_row: int
    complete_row_spans: tuple[tuple[int, int], ...] | None = None
    total_live_events: int | None = None

    @property
    def page_count(self) -> int:
        if self.complete_row_spans is not None:
            return len(self.complete_row_spans)
        return (self.row_count + self.rows_per_page - 1) // self.rows_per_page

    @property
    def single_row_oversized(self) -> bool:
        return self.worst_live_events_per_row > self.live_event_budget

    def row_spans(self) -> tuple[tuple[int, int], ...]:
        if self.complete_row_spans is not None:
            return self.complete_row_spans
        return tuple(
            (start, min(start + self.rows_per_page, self.row_count))
            for start in range(0, self.row_count, self.rows_per_page)
        )


def _compute_live_split_event_budget() -> int:
    """Derive a bounded live-event budget from the active query envelope."""
    try:
        runtime = get_cuda_runtime()
        remaining = getattr(runtime, "query_memory_remaining_bytes", None)
        if callable(remaining):
            free_bytes = int(remaining())
        else:
            stats = runtime.memory_pool_stats()
            if "free_bytes" in stats:
                free_bytes = int(stats["free_bytes"])
            else:
                free_bytes = int(cp.cuda.Device().mem_info[0])
    except Exception:
        return _MAX_LIVE_SPLIT_EVENT_BUDGET

    usable_bytes = max(free_bytes // 5, 0)
    event_budget = usable_bytes // _BYTES_PER_LIVE_SPLIT_EVENT
    return min(
        max(event_budget, _MIN_LIVE_SPLIT_EVENT_BUDGET),
        _MAX_LIVE_SPLIT_EVENT_BUDGET,
    )


def _row_isolated_topology_page_shape(
    *,
    row_count: int,
    max_left_segments_per_row: int,
    max_right_segments_per_row: int,
    include_same_side_splits: bool,
    live_event_budget: int | None = None,
) -> _RowIsolatedTopologyPageShape:
    """Plan complete-row topology pages from worst-case split-event work."""
    row_count = max(int(row_count), 0)
    left_span = max(int(max_left_segments_per_row), 0)
    right_span = max(int(max_right_segments_per_row), 0)
    budget = max(
        int(_compute_live_split_event_budget() if live_event_budget is None else live_event_budget),
        1,
    )

    endpoint_events = 2 * (left_span + right_span)
    pair_events = 4 * left_span * right_span
    if include_same_side_splits:
        pair_events += 2 * left_span * max(left_span - 1, 0)
        pair_events += 2 * right_span * max(right_span - 1, 0)
    worst_per_row = max(endpoint_events + pair_events, 1)
    rows_per_page = max(1, budget // worst_per_row)
    if row_count:
        rows_per_page = min(rows_per_page, row_count)
    return _RowIsolatedTopologyPageShape(
        row_count=row_count,
        rows_per_page=rows_per_page,
        live_event_budget=budget,
        worst_live_events_per_row=worst_per_row,
    )


def _compact_bounded_device_work_spans(
    row_live_events,
    *,
    live_event_budget: int,
    initial_span_limit: int = 256,
) -> tuple[tuple[int, int], ...]:
    """Pack exact device row weights into bounded complete-row spans.

    Only compact span totals cross to the host.  Overweight spans are refined
    on device until they fit or contain one intrinsically oversized row.
    """
    d_work = cp.asarray(row_live_events, dtype=cp.int64)
    row_count = int(d_work.size)
    if row_count == 0:
        return ()
    budget = max(int(live_event_budget), 1)
    d_prefix = cp.empty(row_count + 1, dtype=cp.int64)
    d_prefix[0] = 0
    d_prefix[1:] = cp.cumsum(d_work, dtype=cp.int64)

    initial_count = min(max(int(initial_span_limit), 1), row_count)
    width = (row_count + initial_count - 1) // initial_count
    pending = [
        (start, min(start + width, row_count))
        for start in range(0, row_count, width)
    ]
    leaves: list[tuple[int, int, int]] = []
    runtime = get_cuda_runtime()
    while pending:
        d_starts = cp.asarray([span[0] for span in pending], dtype=cp.int64)
        d_ends = cp.asarray([span[1] for span in pending], dtype=cp.int64)
        host_weights = runtime.copy_device_to_host(
            d_prefix[d_ends] - d_prefix[d_starts],
            reason="overlay compact topology page-weight planning packet",
        )
        refine: list[tuple[int, int]] = []
        for (start, end), weight in zip(pending, host_weights, strict=True):
            resolved_weight = int(weight)
            if resolved_weight <= budget or end - start <= 1:
                leaves.append((start, end, resolved_weight))
            else:
                midpoint = start + (end - start) // 2
                refine.extend(((start, midpoint), (midpoint, end)))
        pending = refine

    packed: list[tuple[int, int]] = []
    page_start = -1
    page_end = -1
    page_weight = 0
    for start, end, weight in sorted(leaves):
        if page_start < 0:
            page_start, page_end, page_weight = start, end, weight
            continue
        if page_end == start and page_weight <= budget and page_weight + weight <= budget:
            page_end = end
            page_weight += weight
            continue
        packed.append((page_start, page_end))
        page_start, page_end, page_weight = start, end, weight
    if page_start >= 0:
        packed.append((page_start, page_end))
    return tuple(packed)


def _row_isolated_device_topology_page_shape(
    left_segments: DeviceSegmentTable,
    right_segments: DeviceSegmentTable,
    *,
    row_count: int,
    right_geometry_source_rows,
    include_same_side_splits: bool,
    live_event_budget: int | None = None,
) -> _RowIsolatedTopologyPageShape:
    """Plan complete-row pages from exact device segment ownership."""
    budget = max(
        int(_compute_live_split_event_budget() if live_event_budget is None else live_event_budget),
        1,
    )
    d_left_rows = cp.asarray(left_segments.row_indices, dtype=cp.int32)
    d_right_rows = cp.asarray(right_segments.row_indices, dtype=cp.int32)
    if right_geometry_source_rows is not None:
        d_right_rows = cp.asarray(right_geometry_source_rows, dtype=cp.int32)[d_right_rows]
    d_left_counts = (
        cp.zeros(row_count, dtype=cp.int64)
        if int(d_left_rows.size) == 0
        else cp.bincount(d_left_rows, minlength=row_count).astype(cp.int64, copy=False)
    )
    d_right_counts = (
        cp.zeros(row_count, dtype=cp.int64)
        if int(d_right_rows.size) == 0
        else cp.bincount(d_right_rows, minlength=row_count).astype(cp.int64, copy=False)
    )
    d_events = 2 * (d_left_counts + d_right_counts)
    d_events += 4 * d_left_counts * d_right_counts
    if include_same_side_splits:
        d_events += 2 * d_left_counts * cp.maximum(d_left_counts - 1, 0)
        d_events += 2 * d_right_counts * cp.maximum(d_right_counts - 1, 0)
    spans = _compact_bounded_device_work_spans(
        d_events,
        live_event_budget=budget,
    )
    summary = get_cuda_runtime().copy_device_to_host(
        cp.stack(
            (
                cp.max(d_events),
                cp.sum(d_events, dtype=cp.int64),
            )
        ).astype(cp.int64, copy=False),
        reason="overlay compact topology work-summary planning packet",
    )
    max_rows_per_page = max((end - start for start, end in spans), default=0)
    return _RowIsolatedTopologyPageShape(
        row_count=row_count,
        rows_per_page=max(max_rows_per_page, 1),
        live_event_budget=budget,
        worst_live_events_per_row=int(summary[0]),
        complete_row_spans=spans,
        total_live_events=int(summary[1]),
    )


from vibespatial.overlay.gpu_kernels import (  # noqa: E402
    _BATCH_POINT_IN_RING_KERNEL_NAMES,
    _BATCH_POINT_IN_RING_KERNEL_SOURCE,
    _CONTAINMENT_BYPASS_KERNEL_NAMES,
    _CONTAINMENT_BYPASS_KERNEL_SOURCE,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES,
    _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE,
    _OVERLAY_FACE_WALK_KERNEL_NAMES,
    _OVERLAY_FACE_WALK_KERNEL_SOURCE,
    _OVERLAY_SPLIT_KERNEL_NAMES,
    _OVERLAY_SPLIT_KERNEL_SOURCE,
)

_OVERLAY_COORDINATE_SCALE = 1_000_000_000.0

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES),
        ("overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES),
        (
            "overlay-face-assembly",
            _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE,
            _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES,
        ),
        (
            "overlay-batch-pip",
            _BATCH_POINT_IN_RING_KERNEL_SOURCE,
            _BATCH_POINT_IN_RING_KERNEL_NAMES,
        ),
        (
            "overlay-containment-bypass",
            _CONTAINMENT_BYPASS_KERNEL_SOURCE,
            _CONTAINMENT_BYPASS_KERNEL_NAMES,
        ),
    ]
)


@register_kernel_variant(
    "overlay_split",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "split"),
)
def _overlay_split_kernels():
    return compile_kernel_group(
        "overlay-split", _OVERLAY_SPLIT_KERNEL_SOURCE, _OVERLAY_SPLIT_KERNEL_NAMES
    )


@register_kernel_variant(
    "overlay_face_walk",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "face-walk"),
)
def _overlay_face_walk_kernels():
    return compile_kernel_group(
        "overlay-face-walk", _OVERLAY_FACE_WALK_KERNEL_SOURCE, _OVERLAY_FACE_WALK_KERNEL_NAMES
    )


@register_kernel_variant(
    "overlay_face_assembly",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "face-assembly"),
)
def _overlay_face_assembly_kernels():
    return compile_kernel_group(
        "overlay-face-assembly",
        _OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE,
        _OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES,
    )


@register_kernel_variant(
    "overlay_batch_point_in_ring",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "batch-pip"),
)
def _batch_pip_kernels():
    return compile_kernel_group(
        "overlay-batch-pip", _BATCH_POINT_IN_RING_KERNEL_SOURCE, _BATCH_POINT_IN_RING_KERNEL_NAMES
    )


@register_kernel_variant(
    "overlay_containment_bypass",
    "gpu-nvrtc",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=False,
    preferred_residency=Residency.DEVICE,
    tags=("nvrtc", "overlay", "containment-bypass"),
)
def _containment_bypass_kernels():
    return compile_kernel_group(
        "overlay-containment-bypass",
        _CONTAINMENT_BYPASS_KERNEL_SOURCE,
        _CONTAINMENT_BYPASS_KERNEL_NAMES,
    )


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _filter_non_empty_owned_device(result_owned: OwnedGeometryArray) -> OwnedGeometryArray | None:
    """Filter null/empty overlay rows without materializing owned metadata on host.

    Physical shape: device rowset view over row-aligned owned geometry metadata.
    The filter consumes device validity, family tags, family row offsets, and
    per-family empty masks, then returns a device-resident indexed view.  This
    preserves row flow without paying variable-width coordinate compaction
    fences before a consumer actually needs contiguous output buffers.
    """
    if cp is None or result_owned.device_state is None:
        return None
    d_state = result_owned.device_state
    d_non_empty = cp.asarray(d_state.validity, dtype=cp.bool_).copy()
    for family, device_buffer in d_state.families.items():
        d_family_rows = (d_state.tags == np.int8(FAMILY_TAGS[family])) & d_state.validity
        if int(device_buffer.empty_mask.size) == 0:
            continue
        d_family_offsets = d_state.family_row_offsets[d_family_rows]
        d_non_empty[d_family_rows] &= ~cp.asarray(device_buffer.empty_mask)[d_family_offsets]
    keep_indices = cp.flatnonzero(d_non_empty).astype(cp.int64, copy=False)
    return OwnedGeometryArray._indexed_view(result_owned, keep_indices)


def _logical_polygonal_only(owned: OwnedGeometryArray) -> bool:
    """Return whether logical non-null rows are polygon/multipolygon rows."""
    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    state = getattr(owned, "device_state", None)
    if state is not None and state.trusted_polygonal_only is True:
        return True
    if set(owned.families) <= polygonal_families:
        if state is not None:
            state.trusted_polygonal_only = True
        return True

    polygon_tags = tuple(np.int8(FAMILY_TAGS[family]) for family in polygonal_families)
    host_tags = getattr(owned, "_tags", None)
    host_validity = getattr(owned, "_validity", None)
    if host_tags is not None and host_validity is not None:
        tags = np.asarray(host_tags, dtype=np.int8)
        validity = np.asarray(host_validity, dtype=bool)
        ok = bool(((~validity) | np.isin(tags, polygon_tags)).all())
        if ok and state is not None:
            state.trusted_polygonal_only = True
        return ok

    if state is not None and cp is not None:
        d_tags = cp.asarray(state.tags, dtype=cp.int8)
        d_validity = cp.asarray(state.validity, dtype=cp.bool_)
        d_polygonal = (d_tags == polygon_tags[0]) | (d_tags == polygon_tags[1])
        d_ok = cp.asarray(cp.all((~d_validity) | d_polygonal)).reshape(1)
        ok = bool(
            get_cuda_runtime().copy_device_to_host(
                d_ok,
                reason="overlay logical polygonal admission scalar fence",
            )[0]
        )
        if ok:
            state.trusted_polygonal_only = True
        return ok

    tags = np.asarray(owned.tags, dtype=np.int8)
    validity = np.asarray(owned.validity, dtype=bool)
    return bool(((~validity) | np.isin(tags, polygon_tags)).all())


# Pipeline stages extracted to separate modules.  Re-export for backward
# compatibility.
from vibespatial.overlay.assemble import (  # noqa: E402, F401
    _axis_aligned_box_bounds,
    _build_device_backed_fixed_polygon_output,
    _build_device_resident_polygon_output,
    _build_polygon_output_from_faces_gpu,
    _empty_polygon_output,
    _gather_coords_vectorised,
    _has_polygonal_families,
    _overlay_intersection_rectangles_gpu,
)
from vibespatial.overlay.faces import (  # noqa: E402, F401
    _assemble_faces_from_device_indices,
    build_gpu_overlay_faces,
)
from vibespatial.overlay.graph import (  # noqa: E402, F401
    _gpu_face_walk,
    _quantize_coordinate,
    build_gpu_half_edge_graph,
)
from vibespatial.overlay.split import (  # noqa: E402, F401
    _free_atomic_edge_excess,
    _free_split_event_device_state,
    _segment_metadata,
    _segment_metadata_gpu,
)


def build_gpu_split_events(
    left,
    right,
    *,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    _cached_left_segments: DeviceSegmentTable | None = None,
    _cached_right_segments: DeviceSegmentTable | None = None,
    right_segment_broadcast: DeviceBroadcastSegmentRelation | None = None,
    require_same_row: bool = False,
    use_same_row_fast_path: bool | None = None,
    same_row_single_group: bool = False,
    same_row_span_summary: tuple[int, int, int] | None = None,
    right_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    include_same_side_splits: bool = False,
) -> SplitEventTable:
    # Delegated to overlay/split.py — this re-export preserves import compatibility.
    from vibespatial.overlay.split import build_gpu_split_events as _impl

    return _impl(
        left,
        right,
        intersection_result=intersection_result,
        dispatch_mode=dispatch_mode,
        _cached_left_segments=_cached_left_segments,
        _cached_right_segments=_cached_right_segments,
        right_segment_broadcast=right_segment_broadcast,
        require_same_row=require_same_row,
        use_same_row_fast_path=use_same_row_fast_path,
        same_row_single_group=same_row_single_group,
        same_row_span_summary=same_row_span_summary,
        right_geometry_source_rows=right_geometry_source_rows,
        include_same_side_splits=include_same_side_splits,
    )


def build_gpu_atomic_edges(
    split_events: SplitEventTable,
    *,
    isolate_rows: bool = False,
) -> AtomicEdgeTable:
    # Delegated to overlay/split.py — this re-export preserves import compatibility.
    from vibespatial.overlay.split import build_gpu_atomic_edges as _impl

    return _impl(split_events, isolate_rows=isolate_rows)


def _polygonal_device_families(owned: OwnedGeometryArray) -> bool:
    state = getattr(owned, "device_state", None)
    families = set(state.families if state is not None else owned.families)
    return bool(families) and families <= {
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    }


def _pack_polygon_parts_by_component(
    polygon_parts,
    component_ids,
    *,
    component_count: int,
) -> OwnedGeometryArray:
    """Pack disjoint Polygon parts into aligned MultiPolygon component rows."""
    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.constructive.binary_constructive import (
        _assemble_sorted_polygon_part_capacity_gpu,
    )
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    part_count = int(polygon_parts.capacity)
    d_active = polygon_parts.selection.active_capacity_mask()
    d_component_ids = cp.asarray(component_ids, dtype=cp.int32)
    grouped_parts = NativeGroupedSelection(
        selection=polygon_parts.selection,
        group_codes=d_component_ids,
        group_count=component_count,
    )
    d_counts = grouped_parts.reduce_numeric(
        cp.ones(part_count, dtype=cp.int32),
        "count",
    ).values.astype(cp.int32, copy=False)
    d_sort_component_ids = cp.where(
        d_active,
        d_component_ids,
        cp.int32(component_count),
    ).astype(cp.uint64, copy=False)
    d_sort_keys = (d_sort_component_ids << cp.uint64(32)) | cp.arange(part_count, dtype=cp.uint64)
    ordered = sort_pairs(
        d_sort_keys,
        cp.arange(part_count, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    d_sorted_part_rows = ordered.values.astype(cp.int64, copy=False)
    sorted_parts = polygon_parts.geometry._device_indexed_take(
        d_sorted_part_rows,
    )._apply_row_activity(d_active[d_sorted_part_rows])
    result = _assemble_sorted_polygon_part_capacity_gpu(
        sorted_parts,
        polygon_parts.logical_count,
        d_counts,
        cp.arange(component_count, dtype=cp.int32),
        output_row_count=component_count,
        runtime_reason="polygon component capacity assembly",
        ring_capacity=polygon_parts.ring_capacity,
        coord_capacity=polygon_parts.coord_capacity,
    )
    if result is None:
        raise RuntimeError("component overlay requires exact Polygon part buffers")
    return result


def _single_row_interval_components(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> tuple[OwnedGeometryArray, OwnedGeometryArray, int] | None:
    """Create aligned rows for strictly x-separated polygon-part components."""
    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )
    from vibespatial.cuda.cccl_primitives import inclusive_max
    from vibespatial.kernels.core.geometry_analysis import (
        compute_geometry_bounds_device,
    )
    from vibespatial.overlay.graph import (
        _fp64_radix_keys,
        _stable_radix_lexicographic_order,
    )

    def _capacity_parts(owned):
        return _explode_polygonal_rows_to_polygon_capacity_gpu(owned)

    left_parts = _capacity_parts(left)
    right_parts = _capacity_parts(right)
    if left_parts is None or right_parts is None:
        return None
    left_part_count = int(left_parts.capacity)
    right_part_count = int(right_parts.capacity)
    part_count = left_part_count + right_part_count
    if part_count <= 1:
        return None

    bound_parts = [
        cp.asarray(
            compute_geometry_bounds_device(parts.geometry, preserve_indexed_view=True),
            dtype=cp.float64,
        ).reshape(parts.capacity, 4)
        for parts in (left_parts, right_parts)
    ]
    d_bounds = cp.concatenate(bound_parts)
    d_active = cp.concatenate(
        [parts.selection.active_capacity_mask() for parts in (left_parts, right_parts)]
    )
    d_bounds = cp.where(
        d_active[:, None],
        d_bounds,
        cp.asarray([cp.inf, cp.inf, -cp.inf, -cp.inf], dtype=cp.float64),
    )
    d_order = _stable_radix_lexicographic_order(
        _fp64_radix_keys(d_bounds[:, 0]),
    )
    d_sorted_xmin = d_bounds[d_order, 0]
    d_prefix_xmax = inclusive_max(d_bounds[d_order, 2])
    d_sorted_active = d_active[d_order]
    d_component_start = cp.zeros(part_count, dtype=cp.bool_)
    d_component_start[0] = d_sorted_active[0]
    d_component_start[1:] = d_sorted_active[1:] & (
        ~d_sorted_active[:-1] | (d_sorted_xmin[1:] > d_prefix_xmax[:-1])
    )
    d_component_count = cp.sum(d_component_start, dtype=cp.int32).reshape(1)
    component_count = int(
        get_cuda_runtime().copy_device_to_host(
            d_component_count,
            reason="overlay interval-component plan-count admission scalar fence",
        )[0]
    )
    if component_count <= 1:
        return None

    d_sorted_component_ids = cp.cumsum(d_component_start.astype(cp.int32), dtype=cp.int32) - 1
    d_component_ids = cp.full(part_count, component_count, dtype=cp.int32)
    d_component_ids[d_order] = cp.where(
        d_sorted_active,
        d_sorted_component_ids,
        cp.int32(component_count),
    )
    d_left_component_ids = d_component_ids[:left_part_count]
    d_right_component_ids = d_component_ids[left_part_count:]
    component_left = _pack_polygon_parts_by_component(
        left_parts,
        d_left_component_ids,
        component_count=component_count,
    )
    component_right = _pack_polygon_parts_by_component(
        right_parts,
        d_right_component_ids,
        component_count=component_count,
    )
    return component_left, component_right, component_count


def _try_single_row_component_overlay_plan(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode,
    same_row_span_summary: tuple[int, int, int] | None,
    include_same_side_splits: bool,
    grouped_right: bool,
) -> ComponentOverlayExecutionPlan | None:
    if (
        same_row_span_summary is None
        or left.row_count != 1
        or right.row_count < 1
        or not _polygonal_device_families(left)
        or not _polygonal_device_families(right)
    ):
        return None
    left_span, right_span, max_row_id = same_row_span_summary
    if int(max_row_id) != 0:
        return None
    page_shape = _row_isolated_topology_page_shape(
        row_count=1,
        max_left_segments_per_row=left_span,
        max_right_segments_per_row=right_span,
        include_same_side_splits=include_same_side_splits,
    )
    if not page_shape.single_row_oversized:
        return None
    components = _single_row_interval_components(left, right)
    if components is None:
        return None
    component_left, component_right, component_count = components
    record_dispatch_event(
        surface="vibespatial.overlay.gpu",
        operation="build_overlay_execution_plan",
        implementation="single_row_interval_component_topology_gpu",
        reason=(
            "one logical row exceeded the live topology target and its combined "
            "polygon parts had strictly separated x-interval components"
        ),
        detail=(
            f"components={component_count}; left_segments={left_span}; "
            f"right_segments={right_span}; event_budget={page_shape.live_event_budget}; "
            f"worst_events={page_shape.worst_live_events_per_row}"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return ComponentOverlayExecutionPlan(
        left=component_left,
        right=component_right,
        component_count=component_count,
        max_left_segments_per_component=int(left_span),
        max_right_segments_per_component=int(right_span),
        dispatch_mode=dispatch_mode,
        include_same_side_splits=include_same_side_splits or grouped_right,
    )


def _try_single_row_microcell_overlay_plan(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode,
    same_row_span_summary: tuple[int, int, int] | None,
    include_same_side_splits: bool,
) -> MicrocellOverlayExecutionPlan | None:
    if (
        same_row_span_summary is None
        or left.row_count != 1
        or right.row_count != 1
        or include_same_side_splits
        or not _polygonal_device_families(left)
        or not _polygonal_device_families(right)
    ):
        return None
    left_span, right_span, max_row_id = same_row_span_summary
    if int(max_row_id) != 0:
        return None
    page_shape = _row_isolated_topology_page_shape(
        row_count=1,
        max_left_segments_per_row=left_span,
        max_right_segments_per_row=right_span,
        include_same_side_splits=False,
    )
    if not page_shape.single_row_oversized:
        return None
    record_dispatch_event(
        surface="vibespatial.overlay.gpu",
        operation="build_overlay_execution_plan",
        implementation="single_row_connected_microcell_boundary_graph_gpu",
        reason=(
            "one connected logical row exceeded the live topology target and "
            "was lowered to bounded x-interval microcell pages"
        ),
        detail=(
            f"left_segments={left_span}; right_segments={right_span}; "
            f"event_budget={page_shape.live_event_budget}; "
            f"worst_events={page_shape.worst_live_events_per_row}"
        ),
        requested=dispatch_mode,
        selected=ExecutionMode.GPU,
    )
    return MicrocellOverlayExecutionPlan(
        left=left,
        right=right,
        max_left_segments=int(left_span),
        max_right_segments=int(right_span),
        dispatch_mode=dispatch_mode,
    )


def _pack_disjoint_component_result(result: OwnedGeometryArray) -> OwnedGeometryArray:
    """Pack strictly interval-disjoint component rows into one polygonal row."""
    from vibespatial.constructive.binary_constructive import (
        _pack_disjoint_multipart_intersection_parts_gpu,
    )
    from vibespatial.geometry.owned import build_empty_polygon_rows_device

    packed = _pack_disjoint_multipart_intersection_parts_gpu(
        result,
        cp.zeros(result.row_count, dtype=cp.int64),
        output_row_count=1,
        assume_disjoint=True,
    )
    return packed if packed is not None else build_empty_polygon_rows_device(1)


def _pack_disjoint_component_remnants(
    result: OwnedGeometryArray,
) -> tuple[OwnedGeometryArray, ...] | None:
    """Regroup component-row remnant carriers into one logical output row."""
    component_parts = getattr(
        result,
        "_polygon_intersection_lower_dimensional_parts",
        None,
    )
    if component_parts is None:
        return None
    if not component_parts:
        return ()

    from vibespatial.constructive.binary_constructive import (
        _explode_lineal_rows_to_line_capacity_gpu,
        _explode_point_rows_to_point_capacity_gpu,
    )
    from vibespatial.constructive.grouped_mixed_union import (
        pack_line_part_capacity_device,
        pack_point_part_capacity_device,
    )

    packed_parts: list[OwnedGeometryArray] = []
    for component_part in component_parts:
        line_parts = _explode_lineal_rows_to_line_capacity_gpu(component_part)
        if line_parts is not None:
            packed_line = pack_line_part_capacity_device(
                line_parts,
                cp.zeros(line_parts.capacity, dtype=cp.int64),
                output_row_count=1,
            )
            if packed_line is None:
                return None
            packed_parts.append(packed_line)
            continue
        point_parts = _explode_point_rows_to_point_capacity_gpu(component_part)
        if point_parts is not None:
            packed_point = pack_point_part_capacity_device(
                point_parts,
                cp.zeros(point_parts.capacity, dtype=cp.int64),
                output_row_count=1,
            )
            if packed_point is None:
                return None
            packed_parts.append(packed_point)
    return tuple(packed_parts)


def _intersection_lower_dimensional_remnants(
    plan: OverlayExecutionPlan,
    selected_faces,
    *,
    row_count: int,
):
    """Assemble exact non-area intersection components from live topology.

    A polygon intersection has a lower-dimensional remnant exactly when the
    two source boundaries meet at a graph edge or node that is not incident to
    a selected intersection face. Atomic-edge deduplication retains a two-bit
    source-membership mask, so coincident spans remain identifiable after the
    graph has collapsed duplicate geometry.

    Physical shape: half-edge and node capacity reduced directly to row-aligned
    line and point carriers. No source boundary reconstruction is required.
    """
    graph_state = plan.half_edge_graph.device_state
    face_state = plan.faces.device_state
    d_source_membership = getattr(graph_state, "source_membership", None)
    d_src_node_ids = graph_state.src_node_ids
    d_rows = graph_state.row_indices
    if d_source_membership is None or d_src_node_ids is None or d_rows is None:
        return None

    edge_count = int(plan.half_edge_graph.edge_count)
    if edge_count == 0:
        return cp.zeros(row_count, dtype=cp.bool_), ()
    if edge_count % 2 != 0:
        raise RuntimeError("overlay half-edge graph lost forward/reverse pairing")

    d_membership = cp.asarray(d_source_membership, dtype=cp.uint8)
    d_nodes = cp.asarray(d_src_node_ids, dtype=cp.int32)
    d_edge_rows = cp.asarray(d_rows, dtype=cp.int32)
    if any(int(values.size) != edge_count for values in (d_membership, d_nodes, d_edge_rows)):
        return None

    d_selected_face = cp.asarray(
        selected_faces.source_mask(),
        dtype=cp.bool_,
    )
    d_face_offsets = cp.asarray(face_state.face_offsets, dtype=cp.int32)
    d_face_edges = cp.asarray(face_state.face_edge_ids, dtype=cp.int32)
    d_selected_edges = cp.zeros(edge_count, dtype=cp.bool_)
    if int(d_face_edges.size) > 0:
        d_face_positions = cp.arange(int(d_face_edges.size), dtype=cp.int32)
        d_face_ids = cp.searchsorted(
            d_face_offsets[1:],
            d_face_positions,
            side="right",
        )
        d_selected_edges[d_face_edges] = d_selected_face[d_face_ids]

    # Both orientations belong to one geometric edge. An edge is covered by
    # area when either orientation is incident to a selected face.
    d_twins = cp.arange(edge_count, dtype=cp.int32) ^ cp.int32(1)
    d_area_incident_edges = d_selected_edges | d_selected_edges[d_twins]
    d_line_remnant_edges = (
        (cp.arange(edge_count, dtype=cp.int32) % cp.int32(2) == 0)
        & (d_membership == cp.uint8(3))
        & ~d_area_incident_edges
    )

    # Node capacity is bounded by edge capacity. Reducing in that capacity
    # avoids a node-count scalar fence while preserving exact source bits.
    d_node_membership = cp.zeros(edge_count, dtype=cp.uint32)
    cp.bitwise_or.at(
        d_node_membership,
        d_nodes,
        d_membership.astype(cp.uint32, copy=False),
    )
    d_selected_nodes = cp.zeros(edge_count, dtype=cp.uint32)
    cp.maximum.at(
        d_selected_nodes,
        d_nodes,
        d_selected_edges.astype(cp.uint32, copy=False),
    )
    d_line_incident_edges = d_line_remnant_edges | d_line_remnant_edges[d_twins]
    d_line_nodes = cp.zeros(edge_count, dtype=cp.uint32)
    cp.maximum.at(
        d_line_nodes,
        d_nodes,
        d_line_incident_edges.astype(cp.uint32, copy=False),
    )
    d_point_nodes = (
        (d_node_membership == cp.uint32(3))
        & (d_selected_nodes == cp.uint32(0))
        & (d_line_nodes == cp.uint32(0))
    )
    d_edge_ids = cp.arange(edge_count, dtype=cp.int32)
    d_node_representatives = cp.full(edge_count, edge_count, dtype=cp.int32)
    cp.minimum.at(d_node_representatives, d_nodes, d_edge_ids)
    d_point_remnant_edges = (
        d_point_nodes[d_nodes]
        & (d_node_representatives[d_nodes] == d_edge_ids)
    )

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.binary_constructive import (
        LinePartCapacitySelection,
        PointPartCapacitySelection,
    )
    from vibespatial.constructive.grouped_mixed_union import (
        pack_line_part_capacity_device,
        pack_point_part_capacity_device,
    )
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
        device_valid_nonempty_mask,
    )

    d_src_x = cp.asarray(graph_state.src_x, dtype=cp.float64)
    d_src_y = cp.asarray(graph_state.src_y, dtype=cp.float64)
    d_dst_x = d_src_x[d_twins]
    d_dst_y = d_src_y[d_twins]
    d_forward_edges = cp.arange(0, edge_count, 2, dtype=cp.int32)
    line_capacity = int(d_forward_edges.size)
    d_line_active = d_line_remnant_edges[d_forward_edges]
    d_line_x = cp.empty(line_capacity * 2, dtype=cp.float64)
    d_line_y = cp.empty(line_capacity * 2, dtype=cp.float64)
    d_line_x[0::2] = d_src_x[d_forward_edges]
    d_line_y[0::2] = d_src_y[d_forward_edges]
    d_line_x[1::2] = d_dst_x[d_forward_edges]
    d_line_y[1::2] = d_dst_y[d_forward_edges]
    line_geometry = build_device_resident_owned(
        device_families={
            GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.LINESTRING,
                x=d_line_x,
                y=d_line_y,
                geometry_offsets=(
                    cp.arange(line_capacity + 1, dtype=cp.int32) * cp.int32(2)
                ),
                empty_mask=cp.zeros(line_capacity, dtype=cp.bool_),
                bounds=None,
            )
        },
        row_count=line_capacity,
        tags=cp.full(
            line_capacity,
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            dtype=cp.int8,
        ),
        validity=cp.ones(line_capacity, dtype=cp.bool_),
        family_row_offsets=cp.arange(line_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    line_selection = NativeDeviceSelection.from_mask(d_line_active)
    d_line_capacity_active = line_selection.active_capacity_mask()
    line_parts = LinePartCapacitySelection(
        geometry=line_geometry._device_indexed_take(
            line_selection.partition_capacity_positions(),
            assume_unique_indices=True,
        )._apply_row_activity(
            d_line_capacity_active,
            assume_active_indices_unique=True,
        ),
        source_rows=line_selection.gather_capacity(
            d_edge_rows[d_forward_edges],
            fill_value=0,
        ).astype(cp.int32, copy=False),
        selection=line_selection.as_capacity_prefix(),
        coord_capacity=line_capacity * 2,
    )
    line_owned = pack_line_part_capacity_device(
        line_parts,
        line_parts.source_rows,
        output_row_count=row_count,
    )
    if line_owned is None:
        return None

    point_capacity = edge_count
    point_geometry = build_device_resident_owned(
        device_families={
            GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POINT,
                x=d_src_x,
                y=d_src_y,
                geometry_offsets=cp.arange(point_capacity + 1, dtype=cp.int32),
                empty_mask=cp.zeros(point_capacity, dtype=cp.bool_),
                bounds=None,
            )
        },
        row_count=point_capacity,
        tags=cp.full(
            point_capacity,
            FAMILY_TAGS[GeometryFamily.POINT],
            dtype=cp.int8,
        ),
        validity=cp.ones(point_capacity, dtype=cp.bool_),
        family_row_offsets=cp.arange(point_capacity, dtype=cp.int32),
        execution_mode="gpu",
    )
    point_selection = NativeDeviceSelection.from_mask(d_point_remnant_edges)
    d_point_capacity_active = point_selection.active_capacity_mask()
    point_parts = PointPartCapacitySelection(
        geometry=point_geometry._device_indexed_take(
            point_selection.partition_capacity_positions(),
            assume_unique_indices=True,
        )._apply_row_activity(
            d_point_capacity_active,
            assume_active_indices_unique=True,
        ),
        source_rows=point_selection.gather_capacity(
            d_edge_rows,
            fill_value=0,
        ).astype(cp.int32, copy=False),
        selection=point_selection.as_capacity_prefix(),
    )
    point_owned = pack_point_part_capacity_device(
        point_parts,
        point_parts.source_rows,
        output_row_count=row_count,
    )
    if point_owned is None:
        return None

    d_line_keep = cp.asarray(device_valid_nonempty_mask(line_owned), dtype=cp.bool_)
    d_point_keep = cp.asarray(device_valid_nonempty_mask(point_owned), dtype=cp.bool_)
    return d_line_keep | d_point_keep, (line_owned, point_owned)


def _intersection_lower_dimensional_remnant_mask(
    plan: OverlayExecutionPlan,
    selected_faces,
    *,
    row_count: int,
):
    """Return the row mask from the topology-native remnant carrier."""
    result = _intersection_lower_dimensional_remnants(
        plan,
        selected_faces,
        row_count=row_count,
    )
    return None if result is None else result[0]


def _build_overlay_execution_plan(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _right_segment_broadcast: DeviceBroadcastSegmentRelation | None = None,
    _row_isolated: bool = False,
    _use_same_row_fast_path: bool | None = None,
    _same_row_single_group: bool = False,
    _same_row_span_summary: tuple[int, int, int] | None = None,
    _include_same_side_splits: bool = False,
    _left_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    _right_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    _right_segment_source_rows: cp.ndarray | np.ndarray | None = None,
    _allow_row_paging: bool = True,
    _allow_component_decomposition: bool = True,
) -> (
    OverlayExecutionPlan
    | PagedOverlayExecutionPlan
    | ComponentOverlayExecutionPlan
    | MicrocellOverlayExecutionPlan
):
    planned_left_segments = None
    planned_right_segments = None
    if (
        _allow_component_decomposition
        and _row_isolated
        and _cached_right_segments is None
        and _left_geometry_source_rows is None
    ):
        grouped_right = (
            _right_geometry_source_rows is not None or _right_segment_source_rows is not None
        )
        component_plan = _try_single_row_component_overlay_plan(
            left,
            right,
            dispatch_mode=dispatch_mode,
            same_row_span_summary=_same_row_span_summary,
            include_same_side_splits=_include_same_side_splits,
            grouped_right=grouped_right,
        )
        if component_plan is not None:
            return component_plan
        if not grouped_right:
            microcell_plan = _try_single_row_microcell_overlay_plan(
                left,
                right,
                dispatch_mode=dispatch_mode,
                same_row_span_summary=_same_row_span_summary,
                include_same_side_splits=_include_same_side_splits,
            )
            if microcell_plan is not None:
                return microcell_plan

    if (
        _allow_row_paging
        and _right_segment_broadcast is not None
        and _row_isolated
        and left.row_count > 1
    ):
        left_span = getattr(
            left,
            "_active_family_row_segment_capacity_bound",
            None,
        )
        if left_span is not None:
            right_span = int(_right_segment_broadcast.physical_count)
            page_shape = _row_isolated_topology_page_shape(
                row_count=left.row_count,
                max_left_segments_per_row=int(left_span),
                max_right_segments_per_row=right_span,
                include_same_side_splits=_include_same_side_splits,
            )
            if page_shape.page_count > 1:
                record_dispatch_event(
                    surface="vibespatial.overlay.gpu",
                    operation="build_overlay_execution_plan",
                    implementation="broadcast_right_complete_row_topology_pages_gpu",
                    reason=(
                        "broadcast-right virtual segment topology exceeded the "
                        "live-event target and was partitioned at row boundaries"
                    ),
                    detail=(
                        f"rows={left.row_count}; pages={page_shape.page_count}; "
                        f"rows_per_page={page_shape.rows_per_page}; "
                        f"left_span={left_span}; right_span={right_span}; "
                        f"event_budget={page_shape.live_event_budget}"
                    ),
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
                )
                return PagedOverlayExecutionPlan(
                    left=left,
                    right=right,
                    row_count=left.row_count,
                    rows_per_page=page_shape.rows_per_page,
                    max_left_segments_per_row=int(left_span),
                    max_right_segments_per_row=right_span,
                    dispatch_mode=dispatch_mode,
                    use_same_row_fast_path=False,
                    include_same_side_splits=_include_same_side_splits,
                    right_segment_broadcast=_right_segment_broadcast,
                    allow_component_decomposition=False,
                )

    if (
        _allow_row_paging
        and _row_isolated
        and _cached_right_segments is None
        and _left_geometry_source_rows is None
        and _same_row_span_summary is not None
        and left.row_count > 1
    ):
        left_span, right_span, max_row_id = _same_row_span_summary
        right_source_rows = (
            _right_segment_source_rows
            if _right_segment_source_rows is not None
            else _right_geometry_source_rows
        )
        right_rows_admit_paging = right_source_rows is not None or right.row_count == left.row_count
        capacity_page_shape = _row_isolated_topology_page_shape(
            row_count=left.row_count,
            max_left_segments_per_row=left_span,
            max_right_segments_per_row=right_span,
            include_same_side_splits=_include_same_side_splits,
        )
        if capacity_page_shape.page_count <= 1:
            page_shape = capacity_page_shape
        elif right_rows_admit_paging:
            planned_left_segments = _extract_segments_gpu(left)
            try:
                planned_right_segments = _extract_segments_gpu(right)
                page_shape = _row_isolated_device_topology_page_shape(
                    planned_left_segments,
                    planned_right_segments,
                    row_count=left.row_count,
                    right_geometry_source_rows=right_source_rows,
                    include_same_side_splits=_include_same_side_splits,
                )
            except Exception:
                planned_left_segments.free()
                if planned_right_segments is not None:
                    planned_right_segments.free()
                raise
        else:
            page_shape = capacity_page_shape
        if (
            page_shape.page_count > 1
            and int(max_row_id) < left.row_count
            and right_rows_admit_paging
        ):
            record_dispatch_event(
                surface="vibespatial.overlay.gpu",
                operation="build_overlay_execution_plan",
                implementation="row_isolated_complete_row_topology_pages_gpu",
                reason=(
                    "row-isolated topology exceeded the live split-event layout "
                    "target and was partitioned at complete logical-row boundaries"
                ),
                detail=(
                    f"rows={left.row_count}; pages={page_shape.page_count}; "
                    f"max_rows_per_page={page_shape.rows_per_page}; "
                    f"worst_events_per_row={page_shape.worst_live_events_per_row}; "
                    f"total_events={page_shape.total_live_events}; "
                    f"event_budget={page_shape.live_event_budget}; "
                    f"single_row_oversized={page_shape.single_row_oversized}"
                ),
                requested=dispatch_mode,
                selected=ExecutionMode.GPU,
            )
            if planned_left_segments is not None:
                planned_left_segments.free()
                planned_left_segments = None
            if planned_right_segments is not None:
                planned_right_segments.free()
                planned_right_segments = None
            return PagedOverlayExecutionPlan(
                left=left,
                right=right,
                row_count=left.row_count,
                rows_per_page=page_shape.rows_per_page,
                max_left_segments_per_row=int(left_span),
                max_right_segments_per_row=int(right_span),
                dispatch_mode=dispatch_mode,
                use_same_row_fast_path=_use_same_row_fast_path,
                include_same_side_splits=_include_same_side_splits,
                right_geometry_source_rows=_right_geometry_source_rows,
                right_segment_source_rows=_right_segment_source_rows,
                allow_component_decomposition=_allow_component_decomposition,
                complete_row_spans=page_shape.row_spans(),
            )

    _sync_hotpath()
    try:
        with hotpath_stage(
            "overlay.plan.split_events",
            category="refine",
        ) as amplification_metadata:
            split_events = build_gpu_split_events(
                left,
                right,
                dispatch_mode=dispatch_mode,
                _cached_left_segments=planned_left_segments,
                _cached_right_segments=(
                    planned_right_segments
                    if planned_right_segments is not None
                    else _cached_right_segments
                ),
                right_segment_broadcast=_right_segment_broadcast,
                require_same_row=_row_isolated,
                use_same_row_fast_path=_use_same_row_fast_path,
                same_row_single_group=_same_row_single_group,
                same_row_span_summary=_same_row_span_summary,
                right_geometry_source_rows=_right_segment_source_rows,
                include_same_side_splits=_include_same_side_splits,
            )
            if amplification_metadata is not None:
                attach_work_amplification(
                    amplification_metadata,
                    operation="overlay.plan.split_events",
                    metric_family="constructive",
                    sums={
                        "source_segments": (
                            int(split_events.left_segment_count)
                            + int(split_events.right_segment_count)
                        ),
                        "split_events": int(split_events.count),
                    },
                    maxima={
                        "left_segments": int(split_events.left_segment_count),
                        "right_segments": int(split_events.right_segment_count),
                    },
                    unavailable=(
                        "emitted_fragments",
                        "fragment_coordinates",
                        "retained_output_parts",
                        "output_coordinates",
                        "constructive_bytes",
                        "peak_live_bytes",
                    ),
                )
        _sync_hotpath()
    except Exception as exc:
        raise RuntimeError(
            f"overlay plan build_gpu_split_events failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if planned_left_segments is not None:
            planned_left_segments.free()
        if planned_right_segments is not None:
            planned_right_segments.free()
    try:
        with hotpath_stage(
            "overlay.plan.atomic_edges",
            category="refine",
        ) as amplification_metadata:
            atomic_edges = build_gpu_atomic_edges(split_events, isolate_rows=_row_isolated)
            if amplification_metadata is not None:
                attach_work_amplification(
                    amplification_metadata,
                    operation="overlay.plan.atomic_edges",
                    metric_family="constructive",
                    sums={
                        "split_events": int(split_events.count),
                        "atomic_edges": int(atomic_edges.count),
                    },
                    maxima={"atomic_edges_per_plan": int(atomic_edges.count)},
                    unavailable=(
                        "emitted_fragments",
                        "fragment_coordinates",
                        "retained_output_parts",
                        "output_coordinates",
                        "constructive_bytes",
                        "peak_live_bytes",
                    ),
                )
        _sync_hotpath()
    except Exception as exc:
        _free_split_event_device_state(split_events)
        maybe_trim_pool_memory()
        raise RuntimeError(
            f"overlay plan build_gpu_atomic_edges failed: {type(exc).__name__}: {exc}"
        ) from exc
    # split_events are fully consumed by build_gpu_atomic_edges.
    with hotpath_stage("overlay.plan.release_split_events", category="setup"):
        _free_split_event_device_state(split_events)
        _sync_hotpath()
    maybe_trim_pool_memory()

    try:
        with hotpath_stage(
            "overlay.plan.half_edge_graph",
            category="refine",
        ) as amplification_metadata:
            half_edge_graph = build_gpu_half_edge_graph(
                atomic_edges,
                isolate_rows=_row_isolated,
            )
            if amplification_metadata is not None:
                attach_work_amplification(
                    amplification_metadata,
                    operation="overlay.plan.half_edge_graph",
                    metric_family="constructive",
                    sums={
                        "atomic_edges": int(atomic_edges.count),
                        "half_edges": int(half_edge_graph.edge_count),
                        "graph_nodes": int(half_edge_graph._node_count),
                    },
                    maxima={"half_edges_per_plan": int(half_edge_graph.edge_count)},
                    unavailable=(
                        "emitted_fragments",
                        "retained_output_parts",
                        "output_coordinates",
                        "constructive_bytes",
                        "peak_live_bytes",
                    ),
                )
            _sync_hotpath()
    except Exception as exc:
        _free_atomic_edge_excess(atomic_edges)
        maybe_trim_pool_memory()
        raise RuntimeError(
            f"overlay plan build_gpu_half_edge_graph failed: {type(exc).__name__}: {exc}"
        ) from exc
    # half_edge_graph retains the atomic-edge arrays it still needs.
    with hotpath_stage("overlay.plan.release_atomic_excess", category="setup"):
        _free_atomic_edge_excess(atomic_edges)
        _sync_hotpath()
    maybe_trim_pool_memory()

    try:
        with hotpath_stage(
            "overlay.plan.faces",
            category="refine",
        ) as amplification_metadata:
            faces = build_gpu_overlay_faces(
                left,
                right,
                half_edge_graph=half_edge_graph,
                row_isolated=_row_isolated,
                left_geometry_source_rows=_left_geometry_source_rows,
                right_geometry_source_rows=_right_geometry_source_rows,
                right_geometry_broadcast=_right_segment_broadcast is not None,
            )
            if amplification_metadata is not None:
                attach_work_amplification(
                    amplification_metadata,
                    operation="overlay.plan.faces",
                    metric_family="constructive",
                    sums={
                        "half_edges": int(half_edge_graph.edge_count),
                        "faces": int(faces.face_count),
                    },
                    maxima={"faces_per_plan": int(faces.face_count)},
                    unavailable=(
                        "retained_output_parts",
                        "output_coordinates",
                        "constructive_bytes",
                        "peak_live_bytes",
                    ),
                )
            _sync_hotpath()
    except Exception as exc:
        raise RuntimeError(
            f"overlay plan build_gpu_overlay_faces failed: {type(exc).__name__}: {exc}"
        ) from exc
    return OverlayExecutionPlan(
        split_events=None,
        atomic_edges=None,
        half_edge_graph=half_edge_graph,
        faces=faces,
        row_isolated=_row_isolated,
    )


def _materialize_overlay_execution_plan(
    plan: (
        OverlayExecutionPlan
        | PagedOverlayExecutionPlan
        | ComponentOverlayExecutionPlan
        | MicrocellOverlayExecutionPlan
    ),
    *,
    operation: str,
    requested: ExecutionMode,
    preserve_row_count: int | None = None,
    valid_empty_rows=None,
) -> tuple[OwnedGeometryArray, ExecutionMode]:
    if isinstance(plan, MicrocellOverlayExecutionPlan):
        if preserve_row_count is not None and preserve_row_count != 1:
            raise ValueError("microcell overlay plan represents exactly one logical output row")
        from vibespatial.overlay.contraction import overlay_contraction_owned

        result = overlay_contraction_owned(
            plan.left,
            plan.right,
            operation=operation,
            dispatch_mode=plan.dispatch_mode,
        )
        return result, ExecutionMode.GPU

    if isinstance(plan, ComponentOverlayExecutionPlan):
        if preserve_row_count is not None and preserve_row_count != 1:
            raise ValueError("component overlay plan represents exactly one logical output row")
        component_plan = _build_overlay_execution_plan(
            plan.left,
            plan.right,
            dispatch_mode=plan.dispatch_mode,
            _row_isolated=True,
            _same_row_span_summary=(
                plan.max_left_segments_per_component,
                plan.max_right_segments_per_component,
                plan.component_count - 1,
            ),
            _include_same_side_splits=plan.include_same_side_splits,
            _allow_component_decomposition=False,
        )
        component_result, selected = _materialize_overlay_execution_plan(
            component_plan,
            operation=operation,
            requested=requested,
            preserve_row_count=plan.component_count,
            valid_empty_rows=valid_empty_rows,
        )
        packed = _pack_disjoint_component_result(component_result)
        component_remnants = getattr(
            component_result,
            "_polygon_intersection_lower_dimensional_remnant",
            None,
        )
        if operation == "intersection" and component_remnants is not None:
            packed._polygon_intersection_lower_dimensional_remnant = cp.any(
                cp.asarray(component_remnants, dtype=cp.bool_)
            ).reshape(1)
            packed_remnants = _pack_disjoint_component_remnants(component_result)
            if packed_remnants is not None:
                packed._polygon_intersection_lower_dimensional_parts = packed_remnants
        return packed, selected

    if isinstance(plan, PagedOverlayExecutionPlan):
        if preserve_row_count is not None and preserve_row_count != plan.row_count:
            raise ValueError(
                "paged row-isolated overlay preserve_row_count must match the "
                f"planned row count ({plan.row_count}), got {preserve_row_count}"
            )
        page_results: list[OwnedGeometryArray] = []
        selected = ExecutionMode.GPU
        right_source_rows = (
            plan.right_segment_source_rows
            if plan.right_segment_source_rows is not None
            else plan.right_geometry_source_rows
        )
        for page_index in range(plan.page_count):
            row_start, row_end = plan.row_span(page_index)
            page_row_count = row_end - row_start
            d_left_rows = cp.arange(row_start, row_end, dtype=cp.int64)
            page_left = plan.left._device_indexed_take(
                d_left_rows,
                assume_unique_indices=True,
            )
            if plan.right_segment_broadcast is None and page_left.is_indexed_view:
                page_left = page_left.physicalize_device_rows(
                    allow_capacity_allocation=True,
                )
            if plan.right_segment_broadcast is not None:
                d_right_rows = cp.arange(plan.right.row_count, dtype=cp.int64)
                page_right = plan.right
                page_broadcast = DeviceBroadcastSegmentRelation(
                    physical_segments=(
                        plan.right_segment_broadcast.physical_segments
                    ),
                    logical_row_count=page_row_count,
                )
            else:
                page_broadcast = None
                if right_source_rows is None:
                    d_right_rows = cp.arange(row_start, row_end, dtype=cp.int64)
                else:
                    d_right_source_rows = cp.asarray(right_source_rows, dtype=cp.int32)
                    d_right_rows = cp.flatnonzero(
                        (d_right_source_rows >= np.int32(row_start))
                        & (d_right_source_rows < np.int32(row_end))
                    ).astype(cp.int64, copy=False)
                page_right = plan.right._device_indexed_take(
                    d_right_rows,
                    assume_unique_indices=True,
                )
                if page_right.is_indexed_view:
                    page_right = page_right.physicalize_device_rows(
                        allow_capacity_allocation=True,
                    )

            def _page_source_rows(
                source_rows,
                *,
                page_rows=d_right_rows,
                source_row_start=row_start,
            ):
                if source_rows is None:
                    return None
                return (
                    cp.asarray(source_rows, dtype=cp.int32)[page_rows] - np.int32(source_row_start)
                ).astype(cp.int32, copy=False)

            page = _build_overlay_execution_plan(
                page_left,
                page_right,
                dispatch_mode=plan.dispatch_mode,
                _cached_right_segments=(
                    None
                    if page_broadcast is None
                    else page_broadcast.physical_segments
                ),
                _right_segment_broadcast=page_broadcast,
                _row_isolated=True,
                _use_same_row_fast_path=plan.use_same_row_fast_path,
                _same_row_single_group=page_row_count == 1,
                _same_row_span_summary=(
                    plan.max_left_segments_per_row,
                    plan.max_right_segments_per_row,
                    page_row_count - 1,
                ),
                _include_same_side_splits=plan.include_same_side_splits,
                _left_geometry_source_rows=None,
                _right_geometry_source_rows=_page_source_rows(plan.right_geometry_source_rows),
                _right_segment_source_rows=_page_source_rows(plan.right_segment_source_rows),
                _allow_row_paging=False,
                _allow_component_decomposition=plan.allow_component_decomposition,
            )
            if not isinstance(
                page,
                (
                    OverlayExecutionPlan,
                    ComponentOverlayExecutionPlan,
                    MicrocellOverlayExecutionPlan,
                ),
            ):
                raise RuntimeError("row-isolated topology page nested paging")
            page_result, page_selected = _materialize_overlay_execution_plan(
                page,
                operation=operation,
                requested=requested,
                preserve_row_count=(page_row_count if preserve_row_count is not None else None),
                valid_empty_rows=(
                    None
                    if valid_empty_rows is None
                    else cp.asarray(valid_empty_rows, dtype=cp.bool_)[row_start:row_end]
                ),
            )
            page_result = _physicalize_paged_overlay_output(page_result)
            with hotpath_stage(
                "overlay.plan.page_retirement",
                category="setup",
            ) as amplification_metadata:
                # A page result is compact and self-contained after
                # physicalization. Complete its stream before planning the
                # next page so graph and CCCL scratch ownership is bounded by
                # one physical topology page rather than Python launch depth.
                get_cuda_runtime().synchronize_stream()
                if amplification_metadata is not None:
                    attach_work_amplification(
                        amplification_metadata,
                        operation="overlay.plan.page_retirement",
                        metric_family="group_compression",
                        sums={
                            "input_rows": int(page_row_count),
                            "output_groups": int(page_result.row_count),
                            "topology_pages": 1,
                        },
                        maxima={
                            "rows_per_page": int(page_row_count),
                            "planned_page_count": int(plan.page_count),
                        },
                        unavailable=(
                            "max_group_size",
                            "input_segments",
                            "input_coordinates",
                            "pre_reduction_fragments",
                            "output_parts",
                            "output_coordinates",
                        ),
                    )
            page_results.append(page_result)
            if page_selected is ExecutionMode.CPU:
                selected = ExecutionMode.CPU
            del page, page_left, page_right
            maybe_trim_pool_memory()
        result = OwnedGeometryArray.concat(page_results)
        if preserve_row_count is not None and result.row_count != preserve_row_count:
            raise RuntimeError(
                "paged row-isolated overlay assembly changed row cardinality: "
                f"expected {preserve_row_count}, got {result.row_count}"
            )
        if operation == "intersection":
            page_remnants = [
                getattr(page_result, "_polygon_intersection_lower_dimensional_remnant", None)
                for page_result in page_results
            ]
            if all(mask is not None for mask in page_remnants):
                result._polygon_intersection_lower_dimensional_remnant = cp.concatenate(
                    [cp.asarray(mask, dtype=cp.bool_) for mask in page_remnants]
                )
            page_parts = [
                getattr(
                    page_result,
                    "_polygon_intersection_lower_dimensional_parts",
                    None,
                )
                for page_result in page_results
            ]
            if all(parts is not None for parts in page_parts):
                part_counts = {len(parts) for parts in page_parts}
                if len(part_counts) == 1:
                    result._polygon_intersection_lower_dimensional_parts = tuple(
                        OwnedGeometryArray.concat(
                            [parts[part_index] for parts in page_parts]
                        )
                        for part_index in range(next(iter(part_counts)))
                    )
        return result, selected

    from vibespatial.overlay.faces import _select_overlay_face_selection_gpu

    selected_faces = _select_overlay_face_selection_gpu(
        plan.faces,
        operation=operation,
    )
    try:
        result = _build_polygon_output_from_faces_gpu(
            plan.half_edge_graph,
            plan.faces,
            selected_faces,
            preserve_row_count=preserve_row_count,
            d_valid_empty_rows=valid_empty_rows,
        )
        if result is None:
            raise RuntimeError("admitted GPU overlay face assembly returned no device result")
        if operation == "intersection" and plan.row_isolated:
            remnant_result = _intersection_lower_dimensional_remnants(
                plan,
                selected_faces,
                row_count=(result.row_count if preserve_row_count is None else preserve_row_count),
            )
            if remnant_result is not None:
                remnant_mask, remnant_parts = remnant_result
                result._polygon_intersection_lower_dimensional_remnant = remnant_mask
                result._polygon_intersection_lower_dimensional_parts = remnant_parts
        return result, ExecutionMode.GPU
    finally:
        del selected_faces
        maybe_trim_pool_memory()


def _physicalize_paged_overlay_output(
    result: OwnedGeometryArray,
) -> OwnedGeometryArray:
    """Compact one completed topology page before retaining later pages."""
    from vibespatial.geometry.owned import (
        build_null_owned_array,
        device_physicalize_owned_row_selections_exact,
    )
    from vibespatial.runtime.residency import Residency

    remnant_mask = getattr(
        result,
        "_polygon_intersection_lower_dimensional_remnant",
        None,
    )
    remnant_parts = getattr(
        result,
        "_polygon_intersection_lower_dimensional_parts",
        None,
    )
    arrays = [result, *(remnant_parts or ())]
    physicalized = device_physicalize_owned_row_selections_exact(
        [
            (owned, cp.ones(owned.row_count, dtype=cp.bool_))
            for owned in arrays
        ],
        reason="paged overlay output exact-allocation packet",
    )
    resolved = [
        (
            build_null_owned_array(source.row_count, residency=Residency.DEVICE)
            if physical is None
            else physical
        )
        for source, physical in zip(arrays, physicalized, strict=True)
    ]
    primary = resolved[0]
    if primary.row_count != result.row_count:
        raise RuntimeError("paged overlay output physicalization changed row count")
    if remnant_mask is not None:
        primary._polygon_intersection_lower_dimensional_remnant = remnant_mask
    if remnant_parts is not None:
        primary._polygon_intersection_lower_dimensional_parts = tuple(resolved[1:])
    return primary


def _expand_group_pair_positions(group_starts, group_ends, *, total_count: int | None = None):
    """Expand grouped pair boundaries into flat sorted-pair positions.

    Physical shape: device grouped-span expansion.  When the caller already
    knows the selected pair cardinality from a native relation/candidate
    buffer and must pass it as ``total_count``. Device grouped spans are not an
    allocation boundary.
    """
    if cp is not None and hasattr(group_starts, "__cuda_array_interface__"):
        d_group_starts = cp.asarray(group_starts, dtype=cp.int64)
        d_group_ends = cp.asarray(group_ends, dtype=cp.int64)
        if int(d_group_starts.size) == 0:
            return cp.empty(0, dtype=cp.int64)
        if total_count is None:
            raise ValueError("device grouped pair-position expansion requires relation cardinality")
        d_counts = (d_group_ends - d_group_starts).astype(cp.int64, copy=False)
        total = int(total_count)
        if total == 0:
            return cp.empty(0, dtype=cp.int64)
        d_offsets = cp.cumsum(d_counts, dtype=cp.int64) - d_counts
        d_positions = cp.arange(total, dtype=cp.int64)
        d_group_ids = cp.searchsorted(d_offsets + d_counts, d_positions, side="right").astype(
            cp.int64, copy=False
        )
        return (
            d_positions
            - d_offsets[d_group_ids].astype(cp.int64, copy=False)
            + d_group_starts[d_group_ids].astype(cp.int64, copy=False)
        )

    h_group_starts = np.asarray(group_starts, dtype=np.int64)
    h_group_ends = np.asarray(group_ends, dtype=np.int64)
    if h_group_starts.size == 0:
        return np.empty(0, dtype=np.int64)
    h_counts = (h_group_ends - h_group_starts).astype(np.int64, copy=False)
    total = int(h_counts.sum(dtype=np.int64))
    if total == 0:
        return np.empty(0, dtype=np.int64)
    h_offsets = np.cumsum(h_counts, dtype=np.int64) - h_counts
    h_positions = np.arange(total, dtype=np.int64)
    h_group_ids = np.searchsorted(h_offsets + h_counts, h_positions, side="right")
    return h_positions - h_offsets[h_group_ids] + h_group_starts[h_group_ids]


def overlay_intersection_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left,
        right,
        operation="intersection",
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_union_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left,
        right,
        operation="union",
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left,
        right,
        operation="difference",
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_symmetric_difference_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left,
        right,
        operation="symmetric_difference",
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def overlay_identity_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> OwnedGeometryArray:
    return _overlay_owned(
        left,
        right,
        operation="identity",
        dispatch_mode=dispatch_mode,
        _cached_right_segments=_cached_right_segments,
    )


def _overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    operation: str,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    _cached_right_segments: DeviceSegmentTable | None = None,
    _right_segment_broadcast: DeviceBroadcastSegmentRelation | None = None,
    _row_isolated: bool = False,
    _left_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    _right_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    _right_segment_source_rows: cp.ndarray | np.ndarray | None = None,
    _include_same_side_splits: bool = False,
) -> OwnedGeometryArray:
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    polygon_only = _logical_polygonal_only(left) and _logical_polygonal_only(right)
    if not polygon_only:
        raise NotImplementedError(
            "GPU overlay owned operations currently support polygon/multipolygon inputs"
        )

    if requested is ExecutionMode.CPU:
        # CPU-only mode: explicit CPU request, Shapely operations
        left_values = np.asarray(left.to_shapely(), dtype=object)
        right_values = np.asarray(right.to_shapely(), dtype=object)
        if operation == "intersection":
            values = shapely.intersection(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "union":
            values = shapely.union(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "difference":
            values = shapely.difference(left_values, right_values).tolist()  # CPU-only mode
        elif operation == "symmetric_difference":
            values = shapely.symmetric_difference(
                left_values, right_values
            ).tolist()  # CPU-only mode
        elif operation == "identity":
            values = [
                geometry
                for geometry in left_values.tolist()
                if geometry is not None and not geometry.is_empty
            ]
        else:
            raise ValueError(f"unsupported overlay operation: {operation}")
        result = from_shapely_geometries(
            values,
            residency=Residency.HOST,
        )
        result.runtime_history.append(
            RuntimeSelection(
                requested=requested,
                selected=ExecutionMode.CPU,
                reason=f"CPU requested for overlay {operation}",
            )
        )
        return result

    if requested is ExecutionMode.GPU and cp is None:
        raise RuntimeError("GPU execution was requested, but CuPy is not available")

    selected = ExecutionMode.GPU if cp is not None else ExecutionMode.CPU
    if requested is ExecutionMode.GPU and selected is not ExecutionMode.GPU:
        raise RuntimeError("GPU execution was requested, but no CUDA runtime is available")
    if requested is ExecutionMode.AUTO and selected is ExecutionMode.CPU:
        # Phase 24: AUTO mode, no GPU available — CPU fallback is expected.
        record_fallback_event(
            surface=f"geopandas.overlay.{operation}",
            reason="AUTO mode: no GPU runtime available",
            detail=f"operation={operation}, left_rows={left.row_count}, right_rows={right.row_count}",
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            pipeline="_overlay_owned",
            d2h_transfer=False,
        )
        return _overlay_owned(left, right, operation=operation, dispatch_mode=ExecutionMode.CPU)

    if operation == "intersection" and not _row_isolated:
        rectangle_fast_path = _overlay_intersection_rectangles_gpu(left, right, requested=requested)
        if rectangle_fast_path is not None:
            return rectangle_fast_path

    # Phase 20: The 10K row CPU threshold (_GPU_OVERLAY_MAX_ROWS) has been
    # removed.  Phases 7-15 eliminated the serial bottlenecks that made GPU
    # overlay slower than Shapely at high row counts.  For AUTO mode the GPU
    # path is now selected whenever a CUDA runtime is available; input
    # residency is already on-device when the caller used the zero-copy
    # pipeline, so no additional transfer heuristic is needed here — the
    # adaptive runtime handles crossover decisions upstream via
    # plan_dispatch_selection().
    plan = _build_overlay_execution_plan(
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        _cached_right_segments=_cached_right_segments,
        _right_segment_broadcast=_right_segment_broadcast,
        _row_isolated=_row_isolated,
        _use_same_row_fast_path=(True if operation == "intersection" and _row_isolated else None),
        _left_geometry_source_rows=_left_geometry_source_rows,
        _right_geometry_source_rows=_right_geometry_source_rows,
        _right_segment_source_rows=_right_segment_source_rows,
        _include_same_side_splits=_include_same_side_splits,
    )
    result, face_assembly_mode = _materialize_overlay_execution_plan(
        plan,
        operation=operation,
        requested=requested,
        preserve_row_count=left.row_count if _row_isolated else None,
    )
    del plan
    maybe_trim_pool_memory()
    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=face_assembly_mode,
            reason=f"GPU overlay {operation}: face assembly on {face_assembly_mode.value}",
        )
    )
    return result


# The geometry-only surface uses the same relation/grouped physical model as
# public overlay while this module continues to own pairwise topology plans.
from vibespatial.overlay.spatial_overlay import spatial_overlay_owned  # noqa: E402, F401
