"""Split event generation and atomic edge construction.

Extracted from overlay/gpu.py (Phase 30 modularisation).

Public API
----------
- ``build_gpu_split_events`` — create split events from segment intersections
- ``build_gpu_atomic_edges`` — build atomic edges from split events

Internal helpers
----------------
- ``_segment_metadata`` — host-side segment metadata extraction
- ``_segment_metadata_gpu`` — GPU-side segment metadata extraction
- ``_free_split_event_device_state`` — release split event GPU buffers
- ``_free_atomic_edge_excess`` — release unneeded atomic edge GPU buffers
"""
from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import (
    exclusive_sum,
    sort_pairs,
    unique_sorted_pairs,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.spatial.segment_primitives import (
    DeviceSegmentTable,
    SegmentIntersectionDeviceState,
    SegmentIntersectionResult,
    SegmentTable,
    _extract_segments_gpu,
    classify_segment_intersections,
)

from .types import (
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    SplitEventDeviceState,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _free_split_event_device_state(split_events: SplitEventTable) -> None:
    """Release SplitEventTable device arrays that are no longer needed.

    After build_gpu_atomic_edges has consumed the split events, the large
    float64 buffers (x, y, t, packed_keys) and int32 metadata arrays on
    device are dead.  Freeing them promptly reduces peak GPU memory by
    ~40-60% of the split event footprint.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = split_events.device_state
    if ds is None:
        return
    for arr in (ds.source_segment_ids, ds.packed_keys, ds.t, ds.x, ds.y,
                ds.source_side, ds.row_indices, ds.part_indices, ds.ring_indices):
        runtime.free(arr)


def _free_atomic_edge_excess(atomic_edges: AtomicEdgeTable) -> None:
    """Release AtomicEdgeTable device arrays NOT shared with HalfEdgeGraph.

    The HalfEdgeGraph holds references to src_x, src_y and per-edge metadata
    (source_segment_ids, source_side, row_indices, part_indices, ring_indices,
    direction) from the AtomicEdgeDeviceState.  Only dst_x and dst_y are
    exclusively consumed during half-edge graph construction and are safe to
    free.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = atomic_edges.device_state
    if ds is None:
        return
    runtime.free(ds.dst_x)
    runtime.free(ds.dst_y)


def _segment_metadata(
    source_segment_ids: np.ndarray,
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_count = left_segments.count
    source_side = np.where(source_segment_ids < left_count, 1, 2).astype(np.int8, copy=False)
    row_indices = np.empty(source_segment_ids.size, dtype=np.int32)
    part_indices = np.empty(source_segment_ids.size, dtype=np.int32)
    ring_indices = np.empty(source_segment_ids.size, dtype=np.int32)

    left_mask = source_side == 1
    if np.any(left_mask):
        left_ids = source_segment_ids[left_mask]
        row_indices[left_mask] = left_segments.row_indices[left_ids]
        part_indices[left_mask] = left_segments.part_indices[left_ids]
        ring_indices[left_mask] = left_segments.ring_indices[left_ids]

    right_mask = ~left_mask
    if np.any(right_mask):
        right_ids = source_segment_ids[right_mask] - left_count
        row_indices[right_mask] = right_segments.row_indices[right_ids]
        part_indices[right_mask] = right_segments.part_indices[right_ids]
        ring_indices[right_mask] = right_segments.ring_indices[right_ids]

    return source_side, row_indices, part_indices, ring_indices


def _segment_metadata_gpu(
    d_source_segment_ids,
    *,
    left_count: int,
    left_segments: SegmentTable | DeviceSegmentTable,
    right_segments: SegmentTable | DeviceSegmentTable,
):
    """Derive source_side / row / part / ring indices entirely on GPU.

    When *left_segments* and *right_segments* are ``DeviceSegmentTable``
    instances (GPU-resident), the lookup tables are used directly on
    device with zero host-device transfers.  When they are CPU-resident
    ``SegmentTable`` instances, the metadata arrays are uploaded once.
    """
    d_ids = cp.asarray(d_source_segment_ids)

    # source_side: 1 for left, 2 for right
    d_source_side = cp.where(d_ids < left_count, cp.int8(1), cp.int8(2))

    # Build combined lookup tables (left then right) so a single
    # gather with the raw source_segment_id works directly.
    # Use device arrays directly when available (DeviceSegmentTable),
    # upload from host only for legacy SegmentTable.
    def _to_device(arr):
        """Wrap a host or device array as a CuPy array."""
        return cp.asarray(arr)

    d_all_row = cp.concatenate((
        _to_device(left_segments.row_indices),
        _to_device(right_segments.row_indices),
    ))

    left_has_parts = (
        left_segments.part_indices is not None
        if isinstance(left_segments, DeviceSegmentTable)
        else hasattr(left_segments, "part_indices")
    )
    right_has_parts = (
        right_segments.part_indices is not None
        if isinstance(right_segments, DeviceSegmentTable)
        else hasattr(right_segments, "part_indices")
    )

    if left_has_parts and right_has_parts:
        d_all_part = cp.concatenate((
            _to_device(left_segments.part_indices),
            _to_device(right_segments.part_indices),
        ))
        d_all_ring = cp.concatenate((
            _to_device(left_segments.ring_indices),
            _to_device(right_segments.ring_indices),
        ))
    else:
        # Fallback: zero-fill part/ring indices when not available
        total = left_count + right_segments.count
        d_all_part = cp.zeros(total, dtype=cp.int32)
        d_all_ring = cp.zeros(total, dtype=cp.int32)

    # Right-side IDs are offset by left_count in the combined table,
    # which matches the segment numbering convention already.
    d_row_indices = d_all_row[d_ids]
    d_part_indices = d_all_part[d_ids]
    d_ring_indices = d_all_ring[d_ids]

    return d_source_side, d_row_indices, d_part_indices, d_ring_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_gpu_split_events(
    left,
    right,
    *,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    _cached_right_segments: DeviceSegmentTable | None = None,
) -> SplitEventTable:
    _require_gpu_arrays()
    runtime = get_cuda_runtime()

    # Lazy import to avoid circular dependency — kernel compile functions
    # live in gpu.py which imports from this module.
    from vibespatial.overlay.gpu import _overlay_split_kernels

    # GPU-native segment extraction -- no CPU loop, no host round-trip.
    # lyy.15: reuse pre-extracted right-side segments when provided
    # (N-vs-1 overlay caches the corridor segments once).
    left_segments = _extract_segments_gpu(left)
    _owns_right_segments = _cached_right_segments is None
    right_segments = (
        _cached_right_segments
        if _cached_right_segments is not None
        else _extract_segments_gpu(right)
    )

    result = intersection_result or classify_segment_intersections(
        left,
        right,
        dispatch_mode=dispatch_mode,
        _cached_right_device_segments=_cached_right_segments,
    )
    if result.runtime_selection.selected is not ExecutionMode.GPU:
        raise RuntimeError("build_gpu_split_events requires a GPU segment-intersection result")
    owns_intersection_state = False
    if result.device_state is None:
        device_state = SegmentIntersectionDeviceState(
            left_rows=runtime.from_host(result.left_rows),
            left_segments=runtime.from_host(result.left_segments),
            left_lookup=runtime.from_host(result.left_lookup),
            right_rows=runtime.from_host(result.right_rows),
            right_segments=runtime.from_host(result.right_segments),
            right_lookup=runtime.from_host(result.right_lookup),
            kinds=runtime.from_host(result.kinds.astype(np.int8, copy=False)),
            point_x=runtime.from_host(result.point_x.astype(np.float64, copy=False)),
            point_y=runtime.from_host(result.point_y.astype(np.float64, copy=False)),
            overlap_x0=runtime.from_host(result.overlap_x0.astype(np.float64, copy=False)),
            overlap_y0=runtime.from_host(result.overlap_y0.astype(np.float64, copy=False)),
            overlap_x1=runtime.from_host(result.overlap_x1.astype(np.float64, copy=False)),
            overlap_y1=runtime.from_host(result.overlap_y1.astype(np.float64, copy=False)),
            ambiguous_rows=runtime.allocate((0,), np.int32),
        )
        owns_intersection_state = True
    else:
        device_state = result.device_state

    left_count = int(left_segments.count)
    right_count = int(right_segments.count)
    segment_total = left_count + right_count
    base_event_count = segment_total * 2
    kernels = _overlay_split_kernels()

    # Segment coordinate arrays are already device-resident from
    # _extract_segments_gpu -- use them directly, no from_host.
    left_x0 = left_segments.x0
    left_y0 = left_segments.y0
    left_x1 = left_segments.x1
    left_y1 = left_segments.y1
    right_x0 = right_segments.x0
    right_y0 = right_segments.y0
    right_x1 = right_segments.x1
    right_y1 = right_segments.y1

    endpoint_source_ids = runtime.allocate((base_event_count,), np.int32)
    endpoint_t = runtime.allocate((base_event_count,), np.float64)
    endpoint_x = runtime.allocate((base_event_count,), np.float64)
    endpoint_y = runtime.allocate((base_event_count,), np.float64)
    endpoint_keys = runtime.allocate((base_event_count,), np.uint64)

    try:
        ptr = runtime.pointer
        endpoint_params = (
            (
                ptr(left_x0),
                ptr(left_y0),
                ptr(left_x1),
                ptr(left_y1),
                ptr(right_x0),
                ptr(right_y0),
                ptr(right_x1),
                ptr(right_y1),
                left_count,
                right_count,
                ptr(endpoint_source_ids),
                ptr(endpoint_t),
                ptr(endpoint_x),
                ptr(endpoint_y),
                ptr(endpoint_keys),
                base_event_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        endpoint_grid, endpoint_block = runtime.launch_config(kernels["emit_endpoint_split_events"], base_event_count)
        runtime.launch(
            kernels["emit_endpoint_split_events"],
            grid=endpoint_grid,
            block=endpoint_block,
            params=endpoint_params,
        )

        pair_counts = runtime.allocate((result.count,), np.int32)
        count_params = (
            (ptr(device_state.kinds), ptr(pair_counts), result.count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        count_grid, count_block = runtime.launch_config(kernels["count_pair_split_events"], result.count)
        runtime.launch(
            kernels["count_pair_split_events"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        pair_offsets = exclusive_sum(pair_counts)
        pair_event_count = (
            int(cp.asnumpy(pair_offsets[-1] + pair_counts[-1])) if int(result.count) else 0  # hygiene:ok — allocation-fence: need pair_event_count to size 5 output buffers
        )
        extra_source_ids = runtime.allocate((pair_event_count,), np.int32)
        extra_t = runtime.allocate((pair_event_count,), np.float64)
        extra_x = runtime.allocate((pair_event_count,), np.float64)
        extra_y = runtime.allocate((pair_event_count,), np.float64)
        extra_keys = runtime.allocate((pair_event_count,), np.uint64)

        try:
            scatter_params = (
                (
                    ptr(device_state.left_lookup),
                    ptr(device_state.right_lookup),
                    ptr(device_state.kinds),
                    ptr(device_state.point_x),
                    ptr(device_state.point_y),
                    ptr(device_state.overlap_x0),
                    ptr(device_state.overlap_y0),
                    ptr(device_state.overlap_x1),
                    ptr(device_state.overlap_y1),
                    ptr(left_x0),
                    ptr(left_y0),
                    ptr(left_x1),
                    ptr(left_y1),
                    ptr(right_x0),
                    ptr(right_y0),
                    ptr(right_x1),
                    ptr(right_y1),
                    ptr(pair_offsets),
                    left_count,
                    ptr(extra_source_ids),
                    ptr(extra_t),
                    ptr(extra_x),
                    ptr(extra_y),
                    ptr(extra_keys),
                    result.count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            )
            scatter_grid, scatter_block = runtime.launch_config(kernels["scatter_pair_split_events"], result.count)
            runtime.launch(
                kernels["scatter_pair_split_events"],
                grid=scatter_grid,
                block=scatter_block,
                params=scatter_params,
            )

            all_source_ids = cp.concatenate((endpoint_source_ids, extra_source_ids))
            all_t = cp.concatenate((endpoint_t, extra_t))
            all_x = cp.concatenate((endpoint_x, extra_x))
            all_y = cp.concatenate((endpoint_y, extra_y))
            all_keys = cp.concatenate((endpoint_keys, extra_keys))

            event_indices = cp.arange(int(all_keys.size), dtype=cp.int32)
            sorted_pairs = sort_pairs(all_keys, event_indices, synchronize=False)
            unique_pairs = unique_sorted_pairs(sorted_pairs.keys, sorted_pairs.values)

            unique_indices = unique_pairs.values
            unique_source_ids = all_source_ids[unique_indices]
            unique_t = all_t[unique_indices]
            unique_x = all_x[unique_indices]
            unique_y = all_y[unique_indices]
            unique_keys = unique_pairs.keys

            # Derive source_side / row / part / ring indices on GPU.
            d_source_side, d_row_indices, d_part_indices, d_ring_indices = (
                _segment_metadata_gpu(
                    unique_source_ids,
                    left_count=left_count,
                    left_segments=left_segments,
                    right_segments=right_segments,
                )
            )
            event_count = int(unique_source_ids.size)

            return SplitEventTable(
                left_segment_count=left_count,
                right_segment_count=right_count,
                runtime_selection=result.runtime_selection,
                device_state=SplitEventDeviceState(
                    source_segment_ids=unique_source_ids,
                    packed_keys=unique_keys,
                    t=unique_t,
                    x=unique_x,
                    y=unique_y,
                    source_side=d_source_side,
                    row_indices=d_row_indices,
                    part_indices=d_part_indices,
                    ring_indices=d_ring_indices,
                ),
                _count=event_count,
            )
        finally:
            runtime.free(pair_counts)
            runtime.free(pair_offsets)
            runtime.free(extra_source_ids)
            runtime.free(extra_t)
            runtime.free(extra_x)
            runtime.free(extra_y)
            runtime.free(extra_keys)
    finally:
        # Free DeviceSegmentTable arrays (x0/y0/x1/y1 are aliases of
        # left_x0 etc., plus row/segment/part/ring metadata).
        # lyy.15: skip freeing right_segments when they are cached
        # (caller owns the lifetime of the cached segments).
        _segs_to_free = [left_segments]
        if _owns_right_segments:
            _segs_to_free.append(right_segments)
        for _dst in _segs_to_free:
            runtime.free(_dst.x0)
            runtime.free(_dst.y0)
            runtime.free(_dst.x1)
            runtime.free(_dst.y1)
            runtime.free(_dst.row_indices)
            runtime.free(_dst.segment_indices)
            if _dst.part_indices is not None:
                runtime.free(_dst.part_indices)
            if _dst.ring_indices is not None:
                runtime.free(_dst.ring_indices)
        runtime.free(endpoint_source_ids)
        runtime.free(endpoint_t)
        runtime.free(endpoint_x)
        runtime.free(endpoint_y)
        runtime.free(endpoint_keys)
        if owns_intersection_state:
            runtime.free(device_state.left_rows)
            runtime.free(device_state.left_segments)
            runtime.free(device_state.left_lookup)
            runtime.free(device_state.right_rows)
            runtime.free(device_state.right_segments)
            runtime.free(device_state.right_lookup)
            runtime.free(device_state.kinds)
            runtime.free(device_state.point_x)
            runtime.free(device_state.point_y)
            runtime.free(device_state.overlap_x0)
            runtime.free(device_state.overlap_y0)
            runtime.free(device_state.overlap_x1)
            runtime.free(device_state.overlap_y1)
            runtime.free(device_state.ambiguous_rows)


def build_gpu_atomic_edges(split_events: SplitEventTable) -> AtomicEdgeTable:
    _require_gpu_arrays()
    runtime = get_cuda_runtime()

    # Lazy import to avoid circular dependency — kernel compile functions
    # live in gpu.py which imports from this module.
    from vibespatial.overlay.gpu import _overlay_split_kernels

    kernels = _overlay_split_kernels()
    device = split_events.device_state
    if split_events.count < 2:
        empty_device_i32 = runtime.allocate((0,), np.int32)
        empty_device_i8 = runtime.allocate((0,), np.int8)
        empty_device_f64 = runtime.allocate((0,), np.float64)
        return AtomicEdgeTable(
            left_segment_count=split_events.left_segment_count,
            right_segment_count=split_events.right_segment_count,
            runtime_selection=split_events.runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=empty_device_i32,
                direction=empty_device_i8,
                src_x=empty_device_f64,
                src_y=empty_device_f64,
                dst_x=empty_device_f64,
                dst_y=empty_device_f64,
                row_indices=empty_device_i32,
                part_indices=empty_device_i32,
                ring_indices=empty_device_i32,
                source_side=empty_device_i8,
            ),
            _count=0,
        )

    adjacency_mask = (
        device.source_segment_ids[:-1] == device.source_segment_ids[1:]
    ).astype(cp.uint8, copy=False)
    adjacency_counts = adjacency_mask.astype(cp.int32, copy=False)
    adjacency_offsets = exclusive_sum(adjacency_counts)
    pair_count = int(cp.asnumpy(adjacency_offsets[-1] + adjacency_counts[-1])) if int(adjacency_counts.size) else 0  # zcopy:ok(allocation-fence: need pair_count to size 6 atomic-edge output buffers) hygiene:ok

    out_source_ids = runtime.allocate((pair_count * 2,), np.int32)
    out_direction = runtime.allocate((pair_count * 2,), np.int8)
    out_src_x = runtime.allocate((pair_count * 2,), np.float64)
    out_src_y = runtime.allocate((pair_count * 2,), np.float64)
    out_dst_x = runtime.allocate((pair_count * 2,), np.float64)
    out_dst_y = runtime.allocate((pair_count * 2,), np.float64)
    try:
        ptr = runtime.pointer
        row_count = max(0, split_events.count - 1)
        params = (
            (
                ptr(device.source_segment_ids),
                ptr(device.x),
                ptr(device.y),
                ptr(adjacency_mask),
                ptr(adjacency_offsets),
                ptr(out_source_ids),
                ptr(out_direction),
                ptr(out_src_x),
                ptr(out_src_y),
                ptr(out_dst_x),
                ptr(out_dst_y),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernels["emit_atomic_edges"], row_count)
        runtime.launch(
            kernels["emit_atomic_edges"],
            grid=grid,
            block=block,
            params=params,
        )
        runtime.synchronize()

        # Derive source_side and row / part / ring indices on GPU via
        # searchsorted against split_events device metadata, avoiding
        # host round-trip.
        d_out_ids = cp.asarray(out_source_ids)  # zcopy:ok(already device-resident — cp.asarray is a no-op)
        left_count = split_events.left_segment_count
        d_source_side = cp.where(d_out_ids < left_count, cp.int8(1), cp.int8(2))

        se_device = split_events.device_state
        d_se_source_ids = cp.asarray(se_device.source_segment_ids)
        clip_idx = cp.clip(
            cp.searchsorted(d_se_source_ids, d_out_ids),
            0, max(split_events.count - 1, 0),
        )
        d_se_row = cp.asarray(se_device.row_indices)
        d_se_part = cp.asarray(se_device.part_indices)
        d_se_ring = cp.asarray(se_device.ring_indices)
        d_row_indices = d_se_row[clip_idx]
        d_part_indices = d_se_part[clip_idx]
        d_ring_indices = d_se_ring[clip_idx]

        # Row/part/ring stay on device; downstream build_gpu_half_edge_graph
        # reads device_state directly.  Host copies are lazily materialized
        # via AtomicEdgeTable.row_indices / part_indices / ring_indices
        # properties on first access.
        return AtomicEdgeTable(
            left_segment_count=split_events.left_segment_count,
            right_segment_count=split_events.right_segment_count,
            runtime_selection=split_events.runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=out_source_ids,
                direction=out_direction,
                src_x=out_src_x,
                src_y=out_src_y,
                dst_x=out_dst_x,
                dst_y=out_dst_y,
                row_indices=d_row_indices,
                part_indices=d_part_indices,
                ring_indices=d_ring_indices,
                source_side=d_source_side,
            ),
            _count=pair_count * 2,
        )
    finally:
        runtime.free(adjacency_mask)
        runtime.free(adjacency_offsets)
