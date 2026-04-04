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
    segmented_reduce_sum,
    sort_pairs,
)

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
    face_id = cp.arange(edge_count, dtype=cp.int32)
    jump = cp.asarray(device.next_edge_ids).copy()
    max_iterations = max(1, int(np.ceil(np.log2(edge_count))))

    for _ in range(max_iterations):
        face_id = cp.minimum(face_id, face_id[jump])
        jump = jump[jump]

    # --- Step 2: Per-edge shoelace contributions ---
    d_cross = cp.empty(edge_count, dtype=cp.float64)
    d_cx_contrib = cp.empty(edge_count, dtype=cp.float64)
    d_cy_contrib = cp.empty(edge_count, dtype=cp.float64)
    shoelace_params = (
        (ptr(device.src_x), ptr(device.src_y), ptr(device.next_edge_ids),
         ptr(d_cross), ptr(d_cx_contrib), ptr(d_cy_contrib), edge_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernels["compute_shoelace_contributions"], edge_count)
    runtime.launch(kernels["compute_shoelace_contributions"], grid=grid, block=block, params=shoelace_params)

    # Phase 25 memory: jump is dead after pointer jumping.
    del jump

    # --- Step 3: Group edges by face_id via sort, then aggregate ---
    edge_ids = cp.arange(edge_count, dtype=cp.int32)
    sort_result = sort_pairs(face_id, edge_ids, synchronize=False)
    sorted_face_ids = sort_result.keys
    sorted_edge_ids = sort_result.values

    # Find unique face_ids and segment boundaries
    face_start_mask = cp.empty(edge_count, dtype=cp.bool_)
    face_start_mask[0] = True
    if edge_count > 1:
        face_start_mask[1:] = sorted_face_ids[1:] != sorted_face_ids[:-1]
    del sorted_face_ids  # Phase 25 memory
    starts = cp.flatnonzero(face_start_mask).astype(cp.int32, copy=False)
    ends = cp.concatenate((starts[1:], cp.asarray([edge_count], dtype=cp.int32)))
    del face_start_mask  # Phase 25 memory

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

    # Segmented reduce for cross, cx_contrib, cy_contrib
    # Reorder contributions to match sorted edge order
    sorted_cross = d_cross[sorted_edge_ids]
    sorted_cx = d_cx_contrib[sorted_edge_ids]
    sorted_cy = d_cy_contrib[sorted_edge_ids]
    # Phase 25 memory: raw contribution arrays consumed after gather.
    del d_cross, d_cx_contrib, d_cy_contrib

    cross_sums = segmented_reduce_sum(sorted_cross, valid_starts, valid_ends, num_segments=face_count).values
    cx_sums = segmented_reduce_sum(sorted_cx, valid_starts, valid_ends, num_segments=face_count).values
    cy_sums = segmented_reduce_sum(sorted_cy, valid_starts, valid_ends, num_segments=face_count).values
    del sorted_cross, sorted_cx, sorted_cy  # Phase 25 memory

    # signed_area = twice_area * 0.5
    signed_area = cross_sums * 0.5
    # centroid: factor = 1 / (3 * twice_area), cx = sum_cx * factor
    twice_area = cross_sums
    safe_twice_area = cp.where(twice_area == 0.0, 1.0, twice_area)
    factor = 1.0 / (3.0 * safe_twice_area)
    centroid_x = cx_sums * factor
    centroid_y = cy_sums * factor
    # For zero-area faces, use mean of coordinates.  Compute the fallback
    # unconditionally on device (cheap segmented reduce + cp.where) to avoid
    # the implicit D2H sync that cp.any(zero_area_mask) would trigger.
    zero_area_mask = twice_area == 0.0
    sorted_src_x = cp.asarray(device.src_x)[sorted_edge_ids]
    sorted_src_y = cp.asarray(device.src_y)[sorted_edge_ids]
    mean_x = segmented_reduce_sum(sorted_src_x, valid_starts, valid_ends, num_segments=face_count).values
    mean_y = segmented_reduce_sum(sorted_src_y, valid_starts, valid_ends, num_segments=face_count).values
    safe_lengths = cp.where(valid_lengths == 0, 1, valid_lengths).astype(cp.float64)
    centroid_x = cp.where(zero_area_mask, mean_x / safe_lengths, centroid_x)
    centroid_y = cp.where(zero_area_mask, mean_y / safe_lengths, centroid_y)
    del zero_area_mask, sorted_src_x, sorted_src_y, mean_x, mean_y, safe_lengths  # Phase 25 memory
    del cross_sums, cx_sums, cy_sums, twice_area, safe_twice_area, factor  # Phase 25 memory

    # Build compact face_offsets and face_edge_ids in cycle traversal order.
    # GPU list ranking: each edge gets its position within its face cycle,
    # then sort by (face_id, rank) to produce contiguous cycle-ordered layout.
    # This replaces the prior serial host walk of next_edge_ids.
    d_rank = cp.empty(edge_count, dtype=cp.int32)
    max_walk = edge_count  # upper bound on cycle length
    rank_params = (
        (ptr(face_id), ptr(device.next_edge_ids),
         ptr(d_rank), edge_count, max_walk),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    rank_grid, rank_block = runtime.launch_config(kernels["list_rank_within_cycle"], edge_count)
    runtime.launch(
        kernels["list_rank_within_cycle"],
        grid=rank_grid, block=rank_block, params=rank_params,
    )

    # Pack (face_id, rank) into a single sort key so sort_pairs gives us
    # edges grouped by face and ordered within each cycle.
    packed_key = face_id.astype(cp.int64) * int(edge_count) + d_rank.astype(cp.int64)
    cycle_sort = sort_pairs(packed_key, edge_ids, synchronize=False)
    cycle_sorted_edge_ids = cycle_sort.values

    # Build face_offsets from valid_lengths on device (CCCL exclusive_sum).
    # exclusive_sum produces face_count elements; append the total to form
    # a proper CSR offset array with face_count+1 entries so that
    # face_offsets[face_count] gives the total edge count.
    _prefix = exclusive_sum(valid_lengths.astype(cp.int32, copy=False))
    _total = valid_lengths.sum().reshape(1).astype(cp.int32)
    face_offsets = cp.concatenate((_prefix, _total))

    # Extract cycle-ordered edges for valid faces only.  The cycle-sorted
    # array has the same segment boundaries (valid_starts, valid_ends) as
    # the face_id-sorted array since the packed key preserves face_id order.
    # Gather all valid-face edges into a contiguous output using vectorised
    # CuPy index arithmetic — no per-face host loop.
    total_edges = int(_total.item())

    # Build a flat gather index: for each slot in the output, compute the
    # source position in cycle_sorted_edge_ids.
    # slot_face = which valid face does this slot belong to?
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
    runtime.launch(kernels["compute_face_sample_points"], grid=sample_grid, block=sample_block, params=sample_params)

    return face_offsets, face_edge_ids, bounded_mask, signed_area, centroid_x, centroid_y, label_x, label_y, face_count
