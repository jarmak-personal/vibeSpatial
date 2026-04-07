"""Half-edge graph construction and GPU face walk.

Extracted from ``overlay/gpu.py`` — Stage 4 of the overlay module split.

Public API
----------
- ``build_gpu_half_edge_graph`` — constructs half-edge graph from atomic edges
- ``_gpu_face_walk`` — walks faces in the half-edge graph, computes shoelace
  contributions + face sample points
- ``_empty_half_edge_graph`` — creates an empty graph structure
- ``_quantize_coordinate`` — coordinate quantization helper
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
)
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled

from .types import (
    AtomicEdgeTable,
    HalfEdgeGraph,
    HalfEdgeGraphDeviceState,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

_OVERLAY_COORDINATE_SCALE = 1_000_000_000.0


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _quantize_coordinate(values):
    return cp.rint(values * _OVERLAY_COORDINATE_SCALE).astype(cp.int64, copy=False)


def _sync_hotpath(runtime) -> None:
    if hotpath_trace_enabled():
        runtime.synchronize()


def _largest_power_of_two_block_size(block_size: int) -> int:
    """Round a positive block size down to the nearest power of two."""
    capped = max(1, int(block_size))
    return 1 << (capped.bit_length() - 1)


def _empty_half_edge_graph(
    atomic_edges: AtomicEdgeTable,
) -> HalfEdgeGraph:
    runtime = get_cuda_runtime()
    empty_i32 = np.asarray([], dtype=np.int32)
    empty_i8 = np.asarray([], dtype=np.int8)
    empty_f64 = np.asarray([], dtype=np.float64)
    empty_device_i32 = runtime.allocate((0,), np.int32)
    empty_device_i8 = runtime.allocate((0,), np.int8)
    empty_device_f64 = runtime.allocate((0,), np.float64)
    return HalfEdgeGraph(
        left_segment_count=atomic_edges.left_segment_count,
        right_segment_count=atomic_edges.right_segment_count,
        runtime_selection=atomic_edges.runtime_selection,
        _edge_count=0,
        _source_segment_ids=empty_i32,
        _source_side=empty_i8,
        _row_indices=empty_i32,
        _part_indices=empty_i32,
        _ring_indices=empty_i32,
        _direction=empty_i8,
        _src_x=empty_f64,
        _src_y=empty_f64,
        _dst_x=empty_f64,
        _dst_y=empty_f64,
        _node_x=empty_f64,
        _node_y=empty_f64,
        _src_node_ids=empty_i32,
        _dst_node_ids=empty_i32,
        _angle=empty_f64,
        _sorted_edge_ids=empty_i32,
        _edge_positions=empty_i32,
        _next_edge_ids=empty_i32,
        device_state=HalfEdgeGraphDeviceState(
            node_x=empty_device_f64,
            node_y=empty_device_f64,
            src_node_ids=empty_device_i32,
            dst_node_ids=empty_device_i32,
            angle=empty_device_f64,
            sorted_edge_ids=empty_device_i32,
            edge_positions=empty_device_i32,
            next_edge_ids=empty_device_i32,
            src_x=empty_device_f64,
            src_y=empty_device_f64,
            source_segment_ids=empty_device_i32,
            source_side=empty_device_i8,
            row_indices=empty_device_i32,
            part_indices=empty_device_i32,
            ring_indices=empty_device_i32,
            direction=empty_device_i8,
        ),
    )


def build_gpu_half_edge_graph(
    atomic_edges: AtomicEdgeTable,
    *,
    isolate_rows: bool = False,
) -> HalfEdgeGraph:
    _require_gpu_arrays()
    _ = get_cuda_runtime()
    if atomic_edges.count == 0:
        return _empty_half_edge_graph(atomic_edges)

    device = atomic_edges.device_state
    edge_count = int(atomic_edges.count)
    all_x = cp.concatenate((device.src_x, device.dst_x))
    all_y = cp.concatenate((device.src_y, device.dst_y))
    all_rows = None
    if isolate_rows and device.row_indices is not None:
        d_rows = cp.asarray(device.row_indices)
        all_rows = cp.concatenate((d_rows, d_rows))

    # Shared split events already emit the exact same fp64 coordinate payload
    # for both sides of a proper intersection. Keep node grouping on those
    # exact coordinates instead of collapsing nearby-but-distinct vertices via
    # quantization, which can fuse separate nodes in dense polygon/circle
    # overlays and corrupt the successor graph.
    if all_rows is not None:
        point_order = cp.lexsort(cp.stack((all_y, all_x, all_rows)))
    else:
        point_order = cp.lexsort(cp.stack((all_y, all_x)))
    sorted_x = all_x[point_order]
    sorted_y = all_y[point_order]
    sorted_rows = all_rows[point_order] if all_rows is not None else None
    point_start_mask = cp.empty((int(all_x.size),), dtype=cp.bool_)
    point_start_mask[0] = True
    if int(all_x.size) > 1:
        point_start_mask[1:] = (sorted_x[1:] != sorted_x[:-1]) | (sorted_y[1:] != sorted_y[:-1])
        if sorted_rows is not None:
            point_start_mask[1:] |= sorted_rows[1:] != sorted_rows[:-1]
    del sorted_x, sorted_y, sorted_rows, all_rows
    point_node_ids_sorted = cp.cumsum(point_start_mask.astype(cp.int32), dtype=cp.int32) - 1
    point_node_ids = cp.empty((int(all_x.size),), dtype=cp.int32)
    point_node_ids[point_order] = point_node_ids_sorted
    del point_node_ids_sorted  # Phase 25 memory

    src_node_ids = point_node_ids[:edge_count]
    dst_node_ids = point_node_ids[edge_count:]
    node_x = all_x[point_order][point_start_mask]
    node_y = all_y[point_order][point_start_mask]
    del all_x, all_y, point_order, point_start_mask  # Phase 25 memory

    angle = cp.arctan2(device.dst_y - device.src_y, device.dst_x - device.src_x)
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    # Use the full fp64 angle for radial ordering at each node. Quantizing
    # the turn angle can collapse near-collinear but distinct rays onto the
    # same key, which breaks the half-edge successor relation on dense
    # polygon/circle intersections even when the split-event payload is exact.
    sorted_edge_ids = cp.lexsort(cp.stack((edge_ids, angle, src_node_ids)))
    sorted_src_nodes = src_node_ids[sorted_edge_ids]

    span_start_mask = cp.empty((edge_count,), dtype=cp.bool_)
    span_start_mask[0] = True
    if edge_count > 1:
        span_start_mask[1:] = sorted_src_nodes[1:] != sorted_src_nodes[:-1]
    del sorted_src_nodes  # Phase 25 memory
    span_group_ids = cp.cumsum(span_start_mask.astype(cp.int32), dtype=cp.int32) - 1
    span_starts = cp.flatnonzero(span_start_mask).astype(cp.int32, copy=False)
    span_ends = cp.concatenate((span_starts[1:], cp.asarray([edge_count], dtype=cp.int32)))
    del span_start_mask  # Phase 25 memory
    edge_positions = cp.empty((edge_count,), dtype=cp.int32)
    edge_positions[sorted_edge_ids] = cp.arange(edge_count, dtype=cp.int32)

    twin_edge_ids = edge_ids ^ 1
    del edge_ids  # Phase 25 memory
    twin_positions = edge_positions[twin_edge_ids]
    del twin_edge_ids  # Phase 25 memory
    twin_group_ids = span_group_ids[twin_positions]
    del span_group_ids  # Phase 25 memory
    twin_group_starts = span_starts[twin_group_ids]
    twin_group_ends = span_ends[twin_group_ids]
    del twin_group_ids, span_starts, span_ends  # Phase 25 memory
    previous_positions = twin_positions - 1
    del twin_positions  # Phase 25 memory
    previous_positions = cp.where(
        previous_positions < twin_group_starts,
        twin_group_ends - 1,
        previous_positions,
    )
    del twin_group_starts, twin_group_ends  # Phase 25 memory
    next_edge_ids = sorted_edge_ids[previous_positions]
    del previous_positions  # Phase 25 memory

    # Carry per-edge metadata from AtomicEdgeTable device state directly
    # -- no D->H transfer.  GPU consumers read device_state.row_indices etc.
    ae_ds = atomic_edges.device_state
    return HalfEdgeGraph(
        left_segment_count=atomic_edges.left_segment_count,
        right_segment_count=atomic_edges.right_segment_count,
        runtime_selection=atomic_edges.runtime_selection,
        _edge_count=edge_count,
        device_state=HalfEdgeGraphDeviceState(
            node_x=node_x,
            node_y=node_y,
            src_node_ids=src_node_ids,
            dst_node_ids=dst_node_ids,
            angle=angle,
            sorted_edge_ids=sorted_edge_ids,
            edge_positions=edge_positions,
            next_edge_ids=next_edge_ids,
            src_x=device.src_x,
            src_y=device.src_y,
            source_segment_ids=ae_ds.source_segment_ids,
            source_side=ae_ds.source_side,
            row_indices=ae_ds.row_indices,
            part_indices=ae_ds.part_indices,
            ring_indices=ae_ds.ring_indices,
            direction=ae_ds.direction,
        ),
    )


def _gpu_face_walk(half_edge_graph: HalfEdgeGraph) -> tuple[
    cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray,
    cp.ndarray, cp.ndarray, cp.ndarray, int,
]:
    """GPU face walk via pointer jumping + shoelace aggregation.

    Returns (face_offsets, face_edge_ids, bounded_mask, signed_area,
             centroid_x, centroid_y, label_x, label_y, face_count)
    as CuPy device arrays (except face_count which is int).
    """
    from vibespatial.overlay.gpu import _overlay_face_walk_kernels

    runtime = get_cuda_runtime()
    device = half_edge_graph.device_state
    edge_count = half_edge_graph.edge_count
    kernels = _overlay_face_walk_kernels()
    ptr = runtime.pointer

    # --- Step 1: Pointer jumping to find cycles (Tier 2 CuPy) ---
    with hotpath_stage("overlay.graph.face_id_pointer_jump", category="refine"):
        face_id = cp.arange(edge_count, dtype=cp.int32)
        jump = cp.asarray(device.next_edge_ids).copy()
        max_iterations = max(1, int(np.ceil(np.log2(edge_count))))

        for _ in range(max_iterations):
            face_id = cp.minimum(face_id, face_id[jump])
            jump = jump[jump]
        _sync_hotpath(runtime)

    # Phase 25 memory: jump is dead after pointer jumping.
    del jump

    # --- Step 3: Group edges by face_id via sort, then aggregate ---
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    with hotpath_stage("overlay.graph.group_faces_sort", category="sort"):
        sort_result = sort_pairs(face_id, edge_ids, synchronize=False)
        _sync_hotpath(runtime)
        sorted_face_ids = sort_result.keys
        sorted_edge_ids = sort_result.values

    # Find unique face_ids and segment boundaries
    with hotpath_stage("overlay.graph.face_span_boundaries", category="sort"):
        face_start_mask = cp.empty(edge_count, dtype=cp.bool_)
        face_start_mask[0] = True
        if edge_count > 1:
            face_start_mask[1:] = sorted_face_ids[1:] != sorted_face_ids[:-1]
        del sorted_face_ids  # Phase 25 memory
        starts = cp.flatnonzero(face_start_mask).astype(cp.int32, copy=False)
        ends = cp.concatenate((starts[1:], cp.asarray([edge_count], dtype=cp.int32)))
        del face_start_mask  # Phase 25 memory
        _sync_hotpath(runtime)

    # Per-face edge counts
    face_lengths = ends - starts

    # Filter faces with < 3 edges
    valid_face_indices = cp.flatnonzero(face_lengths >= 3).astype(cp.int32, copy=False)
    face_count = int(valid_face_indices.size)

    if face_count == 0:
        empty_i32 = cp.asarray([0], dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, cp.empty(0, dtype=cp.int32), empty_i8, empty_f64, empty_f64, empty_f64, empty_f64, empty_f64, 0

    valid_starts = starts[valid_face_indices]
    valid_ends = ends[valid_face_indices]
    valid_lengths = face_lengths[valid_face_indices]
    del starts, ends, face_lengths, valid_face_indices  # Phase 25 memory

    # Aggregate area/centroid directly from sorted face spans in one
    # cooperative kernel instead of five segmented reductions over the same
    # edge ordering.
    signed_area = cp.empty(face_count, dtype=cp.float64)
    centroid_x = cp.empty(face_count, dtype=cp.float64)
    centroid_y = cp.empty(face_count, dtype=cp.float64)
    metrics_params = (
        (
            ptr(device.src_x),
            ptr(device.src_y),
            ptr(device.next_edge_ids),
            ptr(sorted_edge_ids),
            ptr(valid_starts),
            ptr(valid_ends),
            ptr(signed_area),
            ptr(centroid_x),
            ptr(centroid_y),
            face_count,
            edge_count,
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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    metrics_block_size = _largest_power_of_two_block_size(
        min(
            runtime.optimal_block_size(kernels["compute_face_metrics"]),
            256,
        )
    )
    metrics_grid = (max(face_count, 1), 1, 1)
    metrics_block = (metrics_block_size, 1, 1)
    with hotpath_stage("overlay.graph.face_metrics_kernel", category="refine"):
        runtime.launch(
            kernels["compute_face_metrics"],
            grid=metrics_grid,
            block=metrics_block,
            params=metrics_params,
        )
        _sync_hotpath(runtime)

    # Build compact face_offsets and face_edge_ids in cycle traversal order.
    # Rank edges within each cycle by pointer-jumping on the predecessor map:
    # this avoids the prior O(cycle_length) per-edge root walk.
    with hotpath_stage("overlay.graph.cycle_rank_pointer_jump", category="sort"):
        next_edge_ids_i32 = cp.asarray(device.next_edge_ids, dtype=cp.int32)
        predecessor = cp.empty(edge_count, dtype=cp.int32)
        predecessor[next_edge_ids_i32] = edge_ids
        root_mask = face_id == edge_ids
        predecessor[root_mask] = edge_ids[root_mask]
        d_rank = cp.ones(edge_count, dtype=cp.int32)
        d_rank[root_mask] = 0
        max_rank_iterations = max(1, int(np.ceil(np.log2(max(1, edge_count)))))
        for _ in range(max_rank_iterations):
            grandparent = predecessor[predecessor]
            active = predecessor != grandparent
            d_rank = cp.where(active, d_rank + d_rank[predecessor], d_rank)
            predecessor = cp.where(active, grandparent, predecessor)
        del next_edge_ids_i32, predecessor, root_mask, grandparent, active
        _sync_hotpath(runtime)

    # Pack (face_id, rank) into a single sort key so sort_pairs gives us
    # edges grouped by face and ordered within each cycle.
    with hotpath_stage("overlay.graph.cycle_sort", category="sort"):
        packed_key = face_id.astype(cp.int64) * int(edge_count) + d_rank.astype(cp.int64)
        cycle_sort = sort_pairs(packed_key, edge_ids, synchronize=False)
        _sync_hotpath(runtime)
        cycle_sorted_edge_ids = cycle_sort.values

    # Build face_offsets from valid_lengths on device (CCCL exclusive_sum).
    # exclusive_sum produces face_count elements; append the total to form
    # a proper CSR offset array with face_count+1 entries so that
    # face_offsets[face_count] gives the total edge count.
    with hotpath_stage("overlay.graph.face_offsets", category="sort"):
        _prefix = exclusive_sum(valid_lengths.astype(cp.int32, copy=False))
        _total = valid_lengths.sum().reshape(1).astype(cp.int32)
        face_offsets = cp.concatenate((_prefix, _total))
        _sync_hotpath(runtime)

    # Extract cycle-ordered edges for valid faces only.  The cycle-sorted
    # array has the same segment boundaries (valid_starts, valid_ends) as
    # the face_id-sorted array since the packed key preserves face_id order.
    # Gather all valid-face edges into a contiguous output using vectorised
    # CuPy index arithmetic — no per-face host loop.
    total_edges = int(_total.item())

    # Build a flat gather index: for each slot in the output, compute the
    # source position in cycle_sorted_edge_ids.
    # slot_face = which valid face does this slot belong to?
    with hotpath_stage("overlay.graph.gather_face_edges", category="emit"):
        slot_ids = cp.arange(total_edges, dtype=cp.int32)
        slot_face = cp.searchsorted(face_offsets[1:], slot_ids, side='right').astype(cp.int32)
        # local offset within that face's segment
        slot_local = slot_ids - face_offsets[slot_face]
        # source position in the cycle-sorted full array
        src_pos = valid_starts[slot_face] + slot_local
        face_edge_ids = cycle_sorted_edge_ids[src_pos]
        # Phase 25 memory: face walk intermediates consumed.
        del slot_ids, slot_face, slot_local, src_pos, cycle_sorted_edge_ids
        del sorted_edge_ids, valid_starts, valid_ends, valid_lengths
        del packed_key, d_rank
        _sync_hotpath(runtime)

    # --- Step 4: Face sample points via kernel ---
    label_x = cp.empty(face_count, dtype=cp.float64)
    label_y = cp.empty(face_count, dtype=cp.float64)
    bounded_mask = cp.empty(face_count, dtype=cp.int8)
    sample_params = (
        (ptr(device.src_x), ptr(device.src_y), ptr(device.next_edge_ids),
         ptr(face_offsets), ptr(face_edge_ids), ptr(signed_area),
         ptr(label_x), ptr(label_y), ptr(bounded_mask), face_count,
         edge_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
         KERNEL_PARAM_I32),
    )
    sample_grid, sample_block = runtime.launch_config(kernels["compute_face_sample_points"], face_count)
    with hotpath_stage("overlay.graph.sample_points", category="refine"):
        runtime.launch(kernels["compute_face_sample_points"], grid=sample_grid, block=sample_block, params=sample_params)
        _sync_hotpath(runtime)

    return face_offsets, face_edge_ids, bounded_mask, signed_area, centroid_x, centroid_y, label_x, label_y, face_count
