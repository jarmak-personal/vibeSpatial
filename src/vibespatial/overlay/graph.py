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
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import (
    PairSortStrategy,
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
_FP64_SIGN_BIT = np.uint64(0x8000000000000000)


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _quantize_coordinate(values):
    return cp.rint(values * _OVERLAY_COORDINATE_SCALE).astype(cp.int64, copy=False)


def _fp64_radix_keys(values):
    """Map fp64 values to unsigned keys with the same ascending order."""
    bits = cp.asarray(values, dtype=cp.float64).view(cp.uint64)
    negative_fill = cp.uint64(0) - (bits >> cp.uint64(63))
    return bits ^ (negative_fill | _FP64_SIGN_BIT)


def _stable_radix_lexicographic_order(*least_to_most_keys):
    """Stable least-to-most radix passes without an N-by-K key stack."""
    if not least_to_most_keys:
        return cp.empty(0, dtype=cp.int32)
    item_count = int(least_to_most_keys[0].size)
    order = cp.arange(item_count, dtype=cp.int32)
    for key in least_to_most_keys:
        pass_keys = cp.asarray(key)[order]
        sorted_pass = sort_pairs(
            pass_keys,
            order,
            strategy=PairSortStrategy.RADIX,
            synchronize=False,
        )
        next_order = sorted_pass.values
        del pass_keys, sorted_pass, order
        order = next_order
    return order


def _stable_radix_order_pass(order, key):
    """Apply one stable radix pass to an existing device permutation."""
    normalized_key = cp.asarray(key)
    key_dtype = np.dtype(normalized_key.dtype)
    if key_dtype not in {
        np.dtype(np.int32),
        np.dtype(np.int64),
        np.dtype(np.uint64),
        np.dtype(np.float64),
    }:
        if key_dtype.kind == "u":
            normalized_key = normalized_key.astype(cp.uint64, copy=False)
        elif key_dtype.kind in {"b", "i"}:
            target_dtype = cp.int32 if key_dtype.itemsize <= 4 else cp.int64
            normalized_key = normalized_key.astype(target_dtype, copy=False)
        elif key_dtype.kind == "f":
            normalized_key = normalized_key.astype(cp.float64, copy=False)
        else:
            raise TypeError(f"unsupported device radix key dtype: {key_dtype}")
    pass_keys = normalized_key[order]
    sorted_pass = sort_pairs(
        pass_keys,
        order,
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    next_order = sorted_pass.values
    del pass_keys, sorted_pass
    return next_order


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
            source_membership=runtime.allocate((0,), np.uint8),
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
    runtime = get_cuda_runtime()
    if atomic_edges.count == 0:
        return _empty_half_edge_graph(atomic_edges)

    device = atomic_edges.device_state
    edge_count = int(atomic_edges.count)
    source_x = cp.asarray(device.src_x)
    source_y = cp.asarray(device.src_y)
    source_rows = None
    if isolate_rows and device.row_indices is not None:
        source_rows = cp.asarray(device.row_indices)

    # Atomic edges are emitted as adjacent forward/reverse twins and the
    # successor construction below already relies on edge_id ^ 1. Every target
    # endpoint is therefore represented by its twin's source endpoint. Build
    # the node relation from source endpoints once instead of concatenating and
    # sorting a duplicated 2*edge_count point relation.
    point_order = cp.arange(edge_count, dtype=cp.int32)
    source_y_keys = _fp64_radix_keys(source_y)
    point_order = _stable_radix_order_pass(point_order, source_y_keys)
    del source_y_keys
    source_x_keys = _fp64_radix_keys(source_x)
    point_order = _stable_radix_order_pass(point_order, source_x_keys)
    del source_x_keys
    if source_rows is not None:
        point_order = _stable_radix_order_pass(point_order, source_rows)

    # Mark the end of each exact endpoint group in sorted position space. An
    # exclusive sum of those markers is the node id for the current position:
    # [0, 1, 0] -> [0, 0, 1]. This avoids materializing sorted x/y/row columns.
    from vibespatial.overlay.gpu import _overlay_face_walk_kernels

    kernels = _overlay_face_walk_kernels()
    ptr = runtime.pointer
    point_group_ends = cp.empty((edge_count,), dtype=cp.int32)
    endpoint_grid, endpoint_block = runtime.launch_config(
        kernels["mark_endpoint_group_ends"],
        edge_count,
    )
    runtime.launch(
        kernels["mark_endpoint_group_ends"],
        grid=endpoint_grid,
        block=endpoint_block,
        params=(
            (
                ptr(source_x),
                ptr(source_y),
                ptr(source_rows if source_rows is not None else point_order),
                ptr(point_order),
                ptr(point_group_ends),
                np.int32(source_rows is not None),
                edge_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    point_node_ids_sorted = exclusive_sum(point_group_ends)
    del point_group_ends, source_rows
    src_node_ids = cp.empty((edge_count,), dtype=cp.int32)
    src_node_ids[point_order] = point_node_ids_sorted
    del point_node_ids_sorted, point_order, source_x, source_y

    angle = cp.arctan2(device.dst_y - device.src_y, device.dst_x - device.src_x)
    # Use the full fp64 angle for radial ordering at each node. Quantizing
    # the turn angle can collapse near-collinear but distinct rays onto the
    # same key, which breaks the half-edge successor relation on dense
    # polygon/circle intersections even when the split-event payload is exact.
    angle_keys = _fp64_radix_keys(angle)
    del angle
    sorted_edge_ids = cp.arange(edge_count, dtype=cp.int32)
    sorted_edge_ids = _stable_radix_order_pass(sorted_edge_ids, angle_keys)
    del angle_keys
    sorted_edge_ids = _stable_radix_order_pass(sorted_edge_ids, src_node_ids)

    # For each outgoing edge in radial order, write its clockwise predecessor
    # as the successor of its incoming twin. Group starts locate the wraparound
    # predecessor with an upper-bound search; all other positions are O(1).
    # This replaces edge-position, group-id, span-start/end, and twin-position
    # arrays with one exact edge-parallel pass.
    next_edge_ids = cp.empty((edge_count,), dtype=cp.int32)
    successor_grid, successor_block = runtime.launch_config(
        kernels["build_radial_successors"],
        edge_count,
    )
    runtime.launch(
        kernels["build_radial_successors"],
        grid=successor_grid,
        block=successor_block,
        params=(
            (
                ptr(src_node_ids),
                ptr(sorted_edge_ids),
                ptr(next_edge_ids),
                edge_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    del sorted_edge_ids
    # Carry per-edge metadata from AtomicEdgeTable device state directly
    # -- no D->H transfer.  GPU consumers read device_state.row_indices etc.
    ae_ds = atomic_edges.device_state
    return HalfEdgeGraph(
        left_segment_count=atomic_edges.left_segment_count,
        right_segment_count=atomic_edges.right_segment_count,
        runtime_selection=atomic_edges.runtime_selection,
        _edge_count=edge_count,
        # Exact node count is debug/host metadata and is reconstructed lazily
        # only when the public diagnostic property is requested.
        _node_count=0,
        isolate_rows=isolate_rows,
        device_state=HalfEdgeGraphDeviceState(
            node_x=None,
            node_y=None,
            src_node_ids=src_node_ids,
            dst_node_ids=None,
            angle=None,
            sorted_edge_ids=None,
            edge_positions=None,
            next_edge_ids=next_edge_ids,
            src_x=device.src_x,
            src_y=device.src_y,
            source_segment_ids=ae_ds.source_segment_ids,
            source_side=ae_ds.source_side,
            source_membership=ae_ds.source_membership,
            row_indices=ae_ds.row_indices,
            part_indices=ae_ds.part_indices,
            ring_indices=ae_ds.ring_indices,
            direction=ae_ds.direction,
        ),
    )


def _gpu_face_walk(
    half_edge_graph: HalfEdgeGraph,
    *,
    area_epsilon: float = 0.0,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    int,
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
        return (
            empty_i32,
            cp.empty(0, dtype=cp.int32),
            empty_i8,
            empty_f64,
            empty_f64,
            empty_f64,
            empty_f64,
            empty_f64,
            0,
        )

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

    # Every downstream GPU consumer uses face spans as unordered membership and
    # follows next_edge_ids whenever traversal order matters. Keep the first
    # face-id sort as the canonical membership carrier; a second list-ranking
    # array and global (face, rank) sort only duplicate the same topology.
    del face_id, edge_ids

    # Build face_offsets from valid_lengths on device (CCCL exclusive_sum).
    # exclusive_sum produces face_count elements; append the total to form
    # a proper CSR offset array with face_count+1 entries so that
    # face_offsets[face_count] gives the total edge count.
    with hotpath_stage("overlay.graph.face_offsets", category="sort"):
        _prefix = exclusive_sum(valid_lengths.astype(cp.int32, copy=False))
        _total = valid_lengths.sum().reshape(1).astype(cp.int32)
        face_offsets = cp.concatenate((_prefix, _total))
        _sync_hotpath(runtime)

    # Extract grouped edges for valid faces only. Face-local order is not part
    # of the device contract; traversal uses next_edge_ids.
    # Allocate by host-known edge capacity and let device CSR offsets delimit
    # the live prefix; this avoids a D2H scalar allocation fence.
    with hotpath_stage("overlay.graph.gather_face_edges", category="emit"):
        slot_ids = cp.arange(edge_count, dtype=cp.int32)
        live_slot = slot_ids < _total.reshape(1)[0]
        safe_slot_ids = cp.where(live_slot, slot_ids, 0)
        slot_face = cp.searchsorted(
            face_offsets[1:],
            safe_slot_ids,
            side="right",
        ).astype(cp.int32)
        if face_count:
            slot_face = cp.minimum(slot_face, np.int32(face_count - 1))
        slot_face = cp.where(live_slot, slot_face, 0)
        slot_local = safe_slot_ids - face_offsets[slot_face]
        # source position in the cycle-sorted full array
        src_pos = valid_starts[slot_face] + slot_local
        face_edge_ids = sorted_edge_ids[src_pos]
        # Phase 25 memory: face walk intermediates consumed.
        del slot_ids, live_slot, safe_slot_ids, slot_face, slot_local, src_pos
        del sorted_edge_ids, valid_starts, valid_ends, valid_lengths
        _sync_hotpath(runtime)

    # --- Step 4: Face sample points via kernel ---
    label_x = cp.empty(face_count, dtype=cp.float64)
    label_y = cp.empty(face_count, dtype=cp.float64)
    bounded_mask = cp.empty(face_count, dtype=cp.int8)
    sample_params = (
        (
            ptr(device.src_x),
            ptr(device.src_y),
            ptr(device.next_edge_ids),
            ptr(face_offsets),
            ptr(face_edge_ids),
            ptr(signed_area),
            ptr(centroid_x),
            ptr(centroid_y),
            ptr(label_x),
            ptr(label_y),
            ptr(bounded_mask),
            area_epsilon,
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
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    sample_grid, sample_block = runtime.launch_config(
        kernels["compute_face_sample_points"], face_count
    )
    with hotpath_stage("overlay.graph.sample_points", category="refine"):
        runtime.launch(
            kernels["compute_face_sample_points"],
            grid=sample_grid,
            block=sample_block,
            params=sample_params,
        )
        _sync_hotpath(runtime)

    return (
        face_offsets,
        face_edge_ids,
        bounded_mask,
        signed_area,
        centroid_x,
        centroid_y,
        label_x,
        label_y,
        face_count,
    )
