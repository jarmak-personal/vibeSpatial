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

from dataclasses import replace
from functools import partial

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_completion_retainer,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_timing_enabled
from vibespatial.spatial.segment_primitives import (
    DeviceBroadcastSegmentRelation,
    DeviceSegmentTable,
    PagedSegmentIntersectionResult,
    SegmentIntersectionDeviceState,
    SegmentIntersectionResult,
    SegmentTable,
    _extract_segments_gpu,
    classify_segment_intersections,
)

from .graph import _fp64_radix_keys, _stable_radix_order_pass
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

_MIN_SPLIT_EVENT_CANDIDATE_PAGE_PAIRS = 64 * 1024
_PLANARITY_NODE_ULP_WINDOW = np.uint64(8)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_gpu_arrays() -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for overlay split GPU primitives")


def _free_split_event_device_state(split_events: SplitEventTable) -> None:
    """Release SplitEventTable device arrays that are no longer needed.

    After build_gpu_atomic_edges has consumed the split events, the large
    float64 buffers (x, y, t) and int32 metadata arrays on
    device are dead.  Freeing them promptly reduces peak GPU memory by
    ~40-60% of the split event footprint.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = split_events.device_state
    if ds is None:
        return
    for arr in (
        ds.source_segment_ids,
        ds.t,
        ds.x,
        ds.y,
        ds.source_side,
        ds.row_indices,
        ds.part_indices,
        ds.ring_indices,
        ds.geometry_indices,
        ds.planarity_risk,
    ):
        if arr is not None:
            runtime.free(arr)


def _free_atomic_edge_excess(atomic_edges: AtomicEdgeTable) -> None:
    """Release AtomicEdgeTable device arrays NOT shared with HalfEdgeGraph.

    The HalfEdgeGraph holds references to src_x, src_y and per-edge metadata
    (source_segment_ids, source_side, source_membership, row_indices,
    part_indices, ring_indices, direction) from the AtomicEdgeDeviceState.
    Only dst_x and dst_y are exclusively consumed during half-edge graph
    construction and are safe to free.

    Phase 25: overlay pipeline memory optimization.
    """
    runtime = get_cuda_runtime()
    ds = atomic_edges.device_state
    if ds is None:
        return
    runtime.free(ds.dst_x)
    runtime.free(ds.dst_y)


def _sync_hotpath(runtime) -> None:
    if hotpath_timing_enabled():
        cp.cuda.get_current_stream().synchronize()


def _paired_event_planarity_risk(
    source_segment_ids,
    row_indices,
    x_values,
    y_values,
    event_priority,
):
    """Mark fp64-inconsistent reconstructions of one shared exact node.

    The mask controls exact re-noding only; it never merges coordinates.
    Exact-equal pair events already share one graph node. Distinct pair events
    are reconsidered only when their ordered fp64 payloads differ inside the
    predicate-refinement uncertainty window.
    """
    d_priority = cp.asarray(event_priority, dtype=cp.int8)
    event_count = int(d_priority.size)
    d_risk_u32 = cp.zeros(event_count, dtype=cp.uint32)
    d_pair_indices = compact_indices((d_priority > 0).astype(cp.uint8)).values
    pair_count = int(d_pair_indices.size)
    if pair_count <= 1:
        return d_risk_u32.astype(cp.bool_, copy=False)

    d_sources = cp.asarray(source_segment_ids, dtype=cp.int32)[d_pair_indices]
    d_rows = cp.asarray(row_indices, dtype=cp.int32)[d_pair_indices]
    d_x_keys = _fp64_radix_keys(
        cp.asarray(x_values, dtype=cp.float64)[d_pair_indices]
    )
    d_y_keys = _fp64_radix_keys(
        cp.asarray(y_values, dtype=cp.float64)[d_pair_indices]
    )
    d_order = cp.arange(pair_count, dtype=cp.int32)
    for key in (d_sources, d_y_keys, d_x_keys, d_rows):
        d_order = _stable_radix_order_pass(d_order, key)
    d_sorted_indices = d_pair_indices[d_order]
    d_sorted_sources = d_sources[d_order]
    d_sorted_rows = d_rows[d_order]
    d_sorted_x = d_x_keys[d_order]
    d_sorted_y = d_y_keys[d_order]
    d_near = (
        (d_sorted_rows[:-1] == d_sorted_rows[1:])
        & (d_sorted_sources[:-1] != d_sorted_sources[1:])
        & (
            (cp.maximum(d_sorted_x[1:], d_sorted_x[:-1])
             - cp.minimum(d_sorted_x[1:], d_sorted_x[:-1]))
            <= _PLANARITY_NODE_ULP_WINDOW
        )
        & (
            (cp.maximum(d_sorted_y[1:], d_sorted_y[:-1])
             - cp.minimum(d_sorted_y[1:], d_sorted_y[:-1]))
            <= _PLANARITY_NODE_ULP_WINDOW
        )
        & (
            (d_sorted_x[:-1] != d_sorted_x[1:])
            | (d_sorted_y[:-1] != d_sorted_y[1:])
        )
    )
    d_near_positions = compact_indices(d_near.astype(cp.uint8)).values
    d_risky_events = cp.concatenate(
        (
            d_sorted_indices[d_near_positions],
            d_sorted_indices[d_near_positions + 1],
        )
    )
    cp.bitwise_or.at(
        d_risk_u32,
        d_risky_events,
        cp.ones(d_risky_events.size, dtype=cp.uint32),
    )
    return d_risk_u32.astype(cp.bool_, copy=False)


def _canonicalize_source_endpoint_coordinates(
    source_segment_ids,
    t_values,
    x_values,
    y_values,
    event_priority,
    source_side,
    row_indices,
    part_indices,
    ring_indices,
    geometry_indices,
    *,
    segment_count: int,
    left_segments: DeviceSegmentTable,
    right_segments: DeviceSegmentTable,
    broadcast_right: bool = False,
):
    """Propagate paired-event endpoint coordinates across source-ring vertices.

    Proper and overlap split events sometimes canonicalize a source endpoint to
    its opposite-side coordinate.  The adjacent source segment's raw endpoint
    event can still carry the original fp64 payload, which fractures the
    half-edge graph.  Canonicalize only endpoint groups that include a paired
    event so dense endpoint-only geometry remains exact.
    """
    event_count = int(source_segment_ids.size)
    if event_count == 0:
        return x_values, y_values, event_priority, None

    d_t = cp.asarray(t_values, dtype=cp.float64)
    endpoint_mask = ((d_t == 0.0) | (d_t == 1.0)).astype(
        cp.uint8,
        copy=False,
    )
    endpoint_indices = compact_indices(endpoint_mask).values
    endpoint_count = int(endpoint_indices.size)
    if endpoint_count <= 1:
        return x_values, y_values, event_priority, None

    d_source_ids = cp.asarray(source_segment_ids, dtype=cp.int32)
    endpoint_source_ids = d_source_ids[endpoint_indices]
    endpoint_t = d_t[endpoint_indices]
    endpoint_x = cp.asarray(x_values, dtype=cp.float64)[endpoint_indices]
    endpoint_y = cp.asarray(y_values, dtype=cp.float64)[endpoint_indices]
    endpoint_priority = cp.asarray(event_priority, dtype=cp.int32)[endpoint_indices]
    source_endpoint_positions = endpoint_indices[endpoint_t == 0.0]
    source_ids = d_source_ids[source_endpoint_positions]
    source_side_values = cp.asarray(source_side, dtype=cp.int8)[source_endpoint_positions]
    source_rows = cp.asarray(row_indices, dtype=cp.int32)[source_endpoint_positions]
    source_parts = cp.asarray(part_indices, dtype=cp.int32)[source_endpoint_positions]
    source_rings = cp.asarray(ring_indices, dtype=cp.int32)[source_endpoint_positions]
    source_geometries = cp.asarray(geometry_indices, dtype=cp.int32)[source_endpoint_positions]
    source_id_start = cp.empty(source_ids.size, dtype=cp.bool_)
    source_id_start[0] = True
    if int(source_ids.size) > 1:
        source_id_start[1:] = source_ids[1:] != source_ids[:-1]
    source_id_positions = compact_indices(source_id_start.astype(cp.uint8, copy=False)).values
    source_ids = source_ids[source_id_positions]
    source_side_values = source_side_values[source_id_positions]
    source_rows = source_rows[source_id_positions]
    source_parts = source_parts[source_id_positions]
    source_rings = source_rings[source_id_positions]
    source_geometries = source_geometries[source_id_positions]
    source_count = int(source_ids.size)

    ring_start = cp.empty(source_count, dtype=cp.bool_)
    ring_start[0] = True
    if source_count > 1:
        ring_start[1:] = (
            (source_side_values[1:] != source_side_values[:-1])
            | (source_geometries[1:] != source_geometries[:-1])
            | (source_rows[1:] != source_rows[:-1])
            | (source_parts[1:] != source_parts[:-1])
            | (source_rings[1:] != source_rings[:-1])
        )
    ring_end = cp.empty(source_count, dtype=cp.bool_)
    ring_end[-1] = True
    if source_count > 1:
        ring_end[:-1] = ring_start[1:]
    ring_group_ids = cp.cumsum(ring_start.astype(cp.int32), dtype=cp.int32) - 1
    group_start_ids = source_ids[compact_indices(ring_start.astype(cp.uint8)).values]
    group_end_ids = source_ids[compact_indices(ring_end.astype(cp.uint8)).values]
    ring_start_ids = group_start_ids[ring_group_ids]
    ring_end_ids = group_end_ids[ring_group_ids]

    left_count = int(left_segments.count)
    right_count = int(right_segments.count)

    def _gather_source_coordinates(source_ids, left_values, right_values):
        if left_count == 0:
            right_ids = source_ids
            if broadcast_right:
                right_ids = right_ids % np.int32(right_count)
            return cp.asarray(right_values)[right_ids]
        if right_count == 0:
            return cp.asarray(left_values)[source_ids]
        left_ids = cp.minimum(source_ids, np.int32(left_count - 1))
        right_ids = cp.maximum(source_ids - np.int32(left_count), np.int32(0))
        if broadcast_right:
            right_ids %= np.int32(right_count)
        return cp.where(
            source_ids < np.int32(left_count),
            cp.asarray(left_values)[left_ids],
            cp.asarray(right_values)[right_ids],
        )

    group_closed = (
        _gather_source_coordinates(group_start_ids, left_segments.x0, right_segments.x0)
        == _gather_source_coordinates(group_end_ids, left_segments.x1, right_segments.x1)
    ) & (
        _gather_source_coordinates(group_start_ids, left_segments.y0, right_segments.y0)
        == _gather_source_coordinates(group_end_ids, left_segments.y1, right_segments.y1)
    )
    source_group_closed = cp.empty(segment_count, dtype=cp.bool_)
    source_group_closed[source_ids] = group_closed[ring_group_ids]
    source_ring_groups = cp.empty(segment_count, dtype=cp.int32)
    source_ring_groups[source_ids] = ring_group_ids
    source_ring_starts = cp.empty(segment_count, dtype=cp.int32)
    source_ring_ends = cp.empty(segment_count, dtype=cp.int32)
    source_ring_starts[source_ids] = ring_start_ids
    source_ring_ends[source_ids] = ring_end_ids
    endpoint_vertex_ids = cp.where(
        endpoint_t == 0.0,
        endpoint_source_ids,
        cp.where(
            endpoint_source_ids == source_ring_ends[endpoint_source_ids],
            cp.where(
                source_group_closed[endpoint_source_ids],
                source_ring_starts[endpoint_source_ids],
                np.int32(segment_count) + source_ring_groups[endpoint_source_ids],
            ),
            endpoint_source_ids + np.int32(1),
        ),
    ).astype(cp.int32, copy=False)

    order = cp.arange(endpoint_count, dtype=cp.int32)
    for key in (
        endpoint_indices,
        endpoint_priority,
        endpoint_vertex_ids,
    ):
        order = _stable_radix_order_pass(order, key)
    sorted_indices = endpoint_indices[order]
    sorted_x = endpoint_x[order]
    sorted_y = endpoint_y[order]
    sorted_priority = endpoint_priority[order]
    sorted_vertex_ids = endpoint_vertex_ids[order]

    group_start_mask = cp.empty(endpoint_count, dtype=cp.bool_)
    group_start_mask[0] = True
    if endpoint_count > 1:
        group_start_mask[1:] = sorted_vertex_ids[1:] != sorted_vertex_ids[:-1]
    group_end_mask = cp.empty(endpoint_count, dtype=cp.bool_)
    group_end_mask[-1] = True
    if endpoint_count > 1:
        group_end_mask[:-1] = group_start_mask[1:]
    group_end_positions = compact_indices(group_end_mask.astype(cp.uint8, copy=False)).values
    group_ids = cp.cumsum(group_start_mask.astype(cp.int32), dtype=cp.int32) - 1
    representative_positions = group_end_positions[group_ids]
    canonical_x = sorted_x[representative_positions]
    canonical_y = sorted_y[representative_positions]
    representative_priority = sorted_priority[representative_positions]

    should_update = representative_priority > 0
    out_x = cp.asarray(x_values, dtype=cp.float64).copy()
    out_y = cp.asarray(y_values, dtype=cp.float64).copy()
    out_x[sorted_indices] = cp.where(should_update, canonical_x, sorted_x)
    out_y[sorted_indices] = cp.where(should_update, canonical_y, sorted_y)
    out_priority = cp.asarray(event_priority, dtype=cp.int8).copy()
    out_priority[sorted_indices] = cp.where(
        should_update,
        representative_priority,
        sorted_priority,
    )
    out_planarity_risk = _paired_event_planarity_risk(
        source_segment_ids,
        row_indices,
        out_x,
        out_y,
        out_priority,
    )
    return (
        out_x,
        out_y,
        out_priority,
        out_planarity_risk,
    )


def _deduplicate_atomic_edge_geometry(
    source_ids,
    direction,
    src_x,
    src_y,
    dst_x,
    dst_y,
    *,
    occurrence_membership=None,
    occurrence_left_delta,
    occurrence_right_delta,
    occurrence_planarity_risk=None,
    row_indices=None,
    left_segment_count: int,
    preserve_source_orientation: bool = False,
):
    """Collapse duplicate geometric segments before half-edge graph build.

    Collinear overlap emits the same atomic segment once per source polygon.
    Some sources traverse that shared span in opposite directions, so a
    forward-only oriented dedup still leaves duplicate half-edges after the
    reverse pair is regenerated. The half-edge graph needs one geometric
    segment, not duplicate copies, otherwise coincident spans produce
    malformed self-touching face cycles.

    This helper canonicalizes segment endpoints before grouping, keeps the
    first representative for each unique quantized geometry, then regenerates
    exactly one forward/reverse pair. Topology consumers use the canonical
    orientation. Directional constructive consumers may retain the selected
    source representative's traversal while sharing the same undirected
    geometric deduplication.
    """
    d_direction = cp.asarray(direction)
    all_src_x = cp.asarray(src_x)
    all_src_y = cp.asarray(src_y)
    all_dst_x = cp.asarray(dst_x)
    all_dst_y = cp.asarray(dst_y)
    forward_indices = cp.flatnonzero(
        (d_direction == 0)
        & ((all_src_x != all_dst_x) | (all_src_y != all_dst_y))
    ).astype(cp.int32, copy=False)
    if int(forward_indices.size) == 0:
        return (
            source_ids,
            direction,
            src_x,
            src_y,
            dst_x,
            dst_y,
            cp.empty(0, dtype=cp.uint8),
            cp.empty(0, dtype=cp.int32),
            cp.empty(0, dtype=cp.int32),
            cp.empty(0, dtype=cp.bool_),
        )

    d_src_x = all_src_x[forward_indices]
    d_src_y = all_src_y[forward_indices]
    d_dst_x = all_dst_x[forward_indices]
    d_dst_y = all_dst_y[forward_indices]
    swap_mask = (d_src_x > d_dst_x) | ((d_src_x == d_dst_x) & (d_src_y > d_dst_y))
    canon_src_x = cp.where(swap_mask, d_dst_x, d_src_x)
    canon_src_y = cp.where(swap_mask, d_dst_y, d_src_y)
    canon_dst_x = cp.where(swap_mask, d_src_x, d_dst_x)
    canon_dst_y = cp.where(swap_mask, d_src_y, d_dst_y)

    sort_keys = [
        forward_indices,
        _fp64_radix_keys(canon_dst_y),
        _fp64_radix_keys(canon_dst_x),
        _fp64_radix_keys(canon_src_y),
        _fp64_radix_keys(canon_src_x),
    ]
    if row_indices is not None:
        d_rows = cp.asarray(row_indices)[forward_indices]
        sort_keys.append(d_rows)
    sort_order = cp.arange(int(forward_indices.size), dtype=cp.int32)
    for key in sort_keys:
        sort_order = _stable_radix_order_pass(sort_order, key)
    sorted_forward = forward_indices[sort_order]
    sorted_src_x = canon_src_x[sort_order]
    sorted_src_y = canon_src_y[sort_order]
    sorted_dst_x = canon_dst_x[sort_order]
    sorted_dst_y = canon_dst_y[sort_order]
    sorted_orientation_multiplier = cp.where(
        swap_mask[sort_order], cp.int32(-1), cp.int32(1)
    )
    if row_indices is not None:
        sorted_rows = d_rows[sort_order]

    unique_mask = cp.empty(int(sorted_forward.size), dtype=cp.bool_)
    unique_mask[0] = True
    if int(sorted_forward.size) > 1:
        unique_mask[1:] = (
            (sorted_src_x[1:] != sorted_src_x[:-1])
            | (sorted_src_y[1:] != sorted_src_y[:-1])
            | (sorted_dst_x[1:] != sorted_dst_x[:-1])
            | (sorted_dst_y[1:] != sorted_dst_y[:-1])
        )
        if row_indices is not None:
            unique_mask[1:] |= sorted_rows[1:] != sorted_rows[:-1]
    representatives = sorted_forward[unique_mask]
    unique_count = int(representatives.size)

    if occurrence_membership is None:
        sorted_membership = cp.where(
            cp.asarray(source_ids)[sorted_forward] < cp.int32(left_segment_count),
            cp.uint8(1),
            cp.uint8(2),
        )
    else:
        sorted_membership = cp.asarray(
            occurrence_membership,
            dtype=cp.uint8,
        )[sorted_forward]
    group_ids = cp.cumsum(unique_mask, dtype=cp.int32) - cp.int32(1)
    representative_membership_u32 = cp.zeros(unique_count, dtype=cp.uint32)
    cp.bitwise_or.at(
        representative_membership_u32,
        group_ids,
        sorted_membership.astype(cp.uint32, copy=False),
    )
    representative_membership = representative_membership_u32.astype(
        cp.uint8,
        copy=False,
    )
    representative_left_delta = cp.zeros(unique_count, dtype=cp.int32)
    representative_right_delta = cp.zeros(unique_count, dtype=cp.int32)
    representative_planarity_risk_u32 = cp.zeros(unique_count, dtype=cp.uint32)
    cp.add.at(
        representative_left_delta,
        group_ids,
        cp.asarray(occurrence_left_delta, dtype=cp.int32)[sorted_forward]
        * sorted_orientation_multiplier,
    )
    if occurrence_planarity_risk is not None:
        cp.bitwise_or.at(
            representative_planarity_risk_u32,
            group_ids,
            cp.asarray(occurrence_planarity_risk, dtype=cp.uint32)[sorted_forward],
        )
    representative_planarity_risk = representative_planarity_risk_u32.astype(
        cp.bool_,
        copy=False,
    )
    cp.add.at(
        representative_right_delta,
        group_ids,
        cp.asarray(occurrence_right_delta, dtype=cp.int32)[sorted_forward]
        * sorted_orientation_multiplier,
    )

    rep_source_ids = cp.asarray(source_ids)[representatives]
    rep_direction_raw = cp.asarray(direction, dtype=cp.int8)[representatives]
    rep_src_x_raw = cp.asarray(src_x)[representatives]
    rep_src_y_raw = cp.asarray(src_y)[representatives]
    rep_dst_x_raw = cp.asarray(dst_x)[representatives]
    rep_dst_y_raw = cp.asarray(dst_y)[representatives]
    rep_swap_mask = (rep_src_x_raw > rep_dst_x_raw) | (
        (rep_src_x_raw == rep_dst_x_raw) & (rep_src_y_raw > rep_dst_y_raw)
    )
    if preserve_source_orientation:
        rep_src_x = rep_src_x_raw
        rep_src_y = rep_src_y_raw
        rep_dst_x = rep_dst_x_raw
        rep_dst_y = rep_dst_y_raw
        rep_direction = rep_direction_raw
        orientation_multiplier = cp.where(rep_swap_mask, cp.int32(-1), cp.int32(1))
    else:
        rep_src_x = cp.where(rep_swap_mask, rep_dst_x_raw, rep_src_x_raw)
        rep_src_y = cp.where(rep_swap_mask, rep_dst_y_raw, rep_src_y_raw)
        rep_dst_x = cp.where(rep_swap_mask, rep_src_x_raw, rep_dst_x_raw)
        rep_dst_y = cp.where(rep_swap_mask, rep_src_y_raw, rep_dst_y_raw)
        rep_direction = cp.bitwise_xor(
            rep_direction_raw,
            rep_swap_mask.astype(cp.int8, copy=False),
        )
        orientation_multiplier = cp.ones(unique_count, dtype=cp.int32)

    out_size = unique_count * 2
    dedup_source_ids = cp.empty(out_size, dtype=cp.int32)
    dedup_direction = cp.empty(out_size, dtype=cp.int8)
    dedup_src_x = cp.empty(out_size, dtype=cp.float64)
    dedup_src_y = cp.empty(out_size, dtype=cp.float64)
    dedup_dst_x = cp.empty(out_size, dtype=cp.float64)
    dedup_dst_y = cp.empty(out_size, dtype=cp.float64)
    dedup_source_membership = cp.empty(out_size, dtype=cp.uint8)
    dedup_left_delta = cp.empty(out_size, dtype=cp.int32)
    dedup_right_delta = cp.empty(out_size, dtype=cp.int32)
    dedup_planarity_risk = cp.empty(out_size, dtype=cp.bool_)

    dedup_source_ids[0::2] = rep_source_ids
    dedup_source_ids[1::2] = rep_source_ids
    dedup_direction[0::2] = rep_direction
    dedup_direction[1::2] = cp.int8(1) - rep_direction
    dedup_src_x[0::2] = rep_src_x
    dedup_src_x[1::2] = rep_dst_x
    dedup_src_y[0::2] = rep_src_y
    dedup_src_y[1::2] = rep_dst_y
    dedup_dst_x[0::2] = rep_dst_x
    dedup_dst_x[1::2] = rep_src_x
    dedup_dst_y[0::2] = rep_dst_y
    dedup_dst_y[1::2] = rep_src_y
    dedup_source_membership[0::2] = representative_membership
    dedup_source_membership[1::2] = representative_membership
    dedup_left_delta[0::2] = representative_left_delta * orientation_multiplier
    dedup_left_delta[1::2] = -dedup_left_delta[0::2]
    dedup_right_delta[0::2] = representative_right_delta * orientation_multiplier
    dedup_right_delta[1::2] = -dedup_right_delta[0::2]
    dedup_planarity_risk[0::2] = representative_planarity_risk
    dedup_planarity_risk[1::2] = representative_planarity_risk
    return (
        dedup_source_ids,
        dedup_direction,
        dedup_src_x,
        dedup_src_y,
        dedup_dst_x,
        dedup_dst_y,
        dedup_source_membership,
        dedup_left_delta,
        dedup_right_delta,
        dedup_planarity_risk,
    )


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
    right_geometry_segments: SegmentTable | DeviceSegmentTable | None = None,
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

    d_all_row = cp.concatenate(
        (
            _to_device(left_segments.row_indices),
            _to_device(right_segments.row_indices),
        )
    )
    d_all_geometry = cp.concatenate(
        (
            _to_device(left_segments.row_indices),
            _to_device(
                (
                    right_segments if right_geometry_segments is None else right_geometry_segments
                ).row_indices
            ),
        )
    )

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
        d_all_part = cp.concatenate(
            (
                _to_device(left_segments.part_indices),
                _to_device(right_segments.part_indices),
            )
        )
        d_all_ring = cp.concatenate(
            (
                _to_device(left_segments.ring_indices),
                _to_device(right_segments.ring_indices),
            )
        )
    else:
        # Fallback: zero-fill part/ring indices when not available
        total = left_count + right_segments.count
        d_all_part = cp.zeros(total, dtype=cp.int32)
        d_all_ring = cp.zeros(total, dtype=cp.int32)

    # Right-side IDs are offset by left_count in the combined table,
    # which matches the segment numbering convention already.
    d_row_indices = d_all_row[d_ids]
    d_geometry_indices = d_all_geometry[d_ids]
    d_part_indices = d_all_part[d_ids]
    d_ring_indices = d_all_ring[d_ids]

    return (
        d_source_side,
        d_row_indices,
        d_part_indices,
        d_ring_indices,
        d_geometry_indices,
    )


def _broadcast_segment_metadata_gpu(
    d_source_segment_ids,
    *,
    left_count: int,
    left_segments: DeviceSegmentTable,
    right_segments: DeviceSegmentTable,
):
    """Derive metadata for virtual broadcast-right segment instances."""
    d_ids = cp.asarray(d_source_segment_ids, dtype=cp.int32)
    d_left_mask = d_ids < np.int32(left_count)
    right_physical_count = int(right_segments.count)
    d_source_side = cp.where(d_left_mask, cp.int8(1), cp.int8(2))
    d_row_indices = cp.empty(int(d_ids.size), dtype=cp.int32)
    d_geometry_indices = cp.empty(int(d_ids.size), dtype=cp.int32)
    d_part_indices = cp.zeros(int(d_ids.size), dtype=cp.int32)
    d_ring_indices = cp.zeros(int(d_ids.size), dtype=cp.int32)

    if left_count:
        d_left_ids = d_ids[d_left_mask]
        d_left_rows = cp.asarray(left_segments.row_indices, dtype=cp.int32)[d_left_ids]
        d_row_indices[d_left_mask] = d_left_rows
        d_geometry_indices[d_left_mask] = d_left_rows
        if left_segments.part_indices is not None:
            d_part_indices[d_left_mask] = cp.asarray(
                left_segments.part_indices,
                dtype=cp.int32,
            )[d_left_ids]
            d_ring_indices[d_left_mask] = cp.asarray(
                left_segments.ring_indices,
                dtype=cp.int32,
            )[d_left_ids]

    d_right_mask = ~d_left_mask
    if right_physical_count:
        d_right_virtual = d_ids[d_right_mask] - np.int32(left_count)
        d_right_ids = d_right_virtual % np.int32(right_physical_count)
        d_row_indices[d_right_mask] = d_right_virtual // np.int32(
            right_physical_count
        )
        d_geometry_indices[d_right_mask] = cp.asarray(
            right_segments.row_indices,
            dtype=cp.int32,
        )[d_right_ids]
        if right_segments.part_indices is not None:
            d_part_indices[d_right_mask] = cp.asarray(
                right_segments.part_indices,
                dtype=cp.int32,
            )[d_right_ids]
            d_ring_indices[d_right_mask] = cp.asarray(
                right_segments.ring_indices,
                dtype=cp.int32,
            )[d_right_ids]

    return (
        d_source_side,
        d_row_indices,
        d_part_indices,
        d_ring_indices,
        d_geometry_indices,
    )


def _emit_pair_split_event_batch(
    result: SegmentIntersectionResult,
    *,
    left_segments: DeviceSegmentTable,
    right_segments: DeviceSegmentTable,
    kernels,
    broadcast_right: bool = False,
):
    """Emit one compact classified page as a persistent split-event run."""
    if result.count == 0 or result.device_state is None:
        return None
    runtime = get_cuda_runtime()
    state = result.device_state
    left_count = int(left_segments.count)
    pair_counts = runtime.allocate((result.count,), np.int32)
    pair_offsets = None
    raw_source_ids = None
    raw_t = None
    raw_x = None
    raw_y = None
    raw_priority = None
    try:
        ptr = runtime.pointer
        count_params = (
            (ptr(state.kinds), ptr(pair_counts), result.count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        count_grid, count_block = runtime.launch_config(
            kernels["count_pair_split_events"],
            result.count,
        )
        runtime.launch(
            kernels["count_pair_split_events"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )
        pair_offsets = exclusive_sum(pair_counts)
        capacity = result.count * 4
        raw_source_ids = runtime.allocate((capacity,), np.int32)
        raw_t = runtime.allocate((capacity,), np.float64)
        raw_x = runtime.allocate((capacity,), np.float64)
        raw_y = runtime.allocate((capacity,), np.float64)
        raw_priority = runtime.allocate((capacity,), np.int8)
        scatter_params = (
            (
                ptr(state.left_lookup),
                ptr(state.right_lookup),
                ptr(state.left_rows),
                ptr(state.kinds),
                ptr(state.point_x),
                ptr(state.point_y),
                ptr(state.overlap_x0),
                ptr(state.overlap_y0),
                ptr(state.overlap_x1),
                ptr(state.overlap_y1),
                ptr(left_segments.x0),
                ptr(left_segments.y0),
                ptr(left_segments.x1),
                ptr(left_segments.y1),
                ptr(right_segments.x0),
                ptr(right_segments.y0),
                ptr(right_segments.x1),
                ptr(right_segments.y1),
                ptr(pair_offsets),
                left_count,
                int(right_segments.count),
                int(broadcast_right),
                ptr(raw_source_ids),
                ptr(raw_t),
                ptr(raw_x),
                ptr(raw_y),
                ptr(raw_priority),
                result.count,
            ),
            (
                # Classified pair columns.
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
                # Left and right source segment coordinates.
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                # Pair offsets and left source count.
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                # Exact output event columns.
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(
            kernels["scatter_pair_split_events"],
            result.count,
        )
        runtime.launch(
            kernels["scatter_pair_split_events"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        live_total = pair_offsets[-1].astype(cp.int32, copy=False) + pair_counts[-1]
        live_indices = compact_indices(
            (cp.arange(capacity, dtype=cp.int32) < live_total).astype(
                cp.uint8,
                copy=False,
            )
        ).values
        if int(live_indices.size) == 0:
            return None
        return (
            cp.asarray(raw_source_ids)[live_indices],
            cp.asarray(raw_t)[live_indices],
            cp.asarray(raw_x)[live_indices],
            cp.asarray(raw_y)[live_indices],
            cp.asarray(raw_priority)[live_indices],
        )
    finally:
        runtime.free(pair_counts)
        if pair_offsets is not None:
            runtime.free(pair_offsets)
        for values in (
            raw_source_ids,
            raw_t,
            raw_x,
            raw_y,
            raw_priority,
        ):
            if values is not None:
                runtime.free(values)


def _sort_deduplicate_split_event_run(run):
    """Sort exact composite event keys and keep the final duplicate payload."""
    source_ids, event_t, event_x, event_y, priority = run
    event_count = int(source_ids.size)
    if event_count < 2:
        return run
    order = cp.arange(event_count, dtype=cp.int32)
    order = _stable_radix_order_pass(order, event_t)
    order = _stable_radix_order_pass(order, source_ids)
    sorted_source_ids = source_ids[order]
    sorted_t = event_t[order]
    keep = cp.empty(event_count, dtype=cp.bool_)
    keep[:-1] = (sorted_source_ids[:-1] != sorted_source_ids[1:]) | (sorted_t[:-1] != sorted_t[1:])
    keep[-1] = True
    positions = cp.flatnonzero(keep).astype(cp.int64, copy=False)
    selected = order[positions]
    return (
        sorted_source_ids[positions],
        sorted_t[positions],
        event_x[selected],
        event_y[selected],
        priority[selected],
    )


def _merge_sorted_split_event_runs(left_run, right_run):
    """Merge unique exact ``(source id, fp64 t)`` device runs.

    Equal keys retain the highest-priority coordinate payload. Cross-operand
    arrangement nodes therefore survive later same-side and raw-endpoint runs
    independently of page count or balanced merge shape.
    """
    if left_run is None:
        return right_run
    if right_run is None:
        return left_run
    left_count = int(left_run[0].size)
    right_count = int(right_run[0].size)
    if left_count == 0:
        return right_run
    if right_count == 0:
        return left_run

    total = left_count + right_count
    left_positions = cp.empty(left_count, dtype=cp.int64)
    right_positions = cp.empty(right_count, dtype=cp.int64)
    from vibespatial.overlay.gpu import _overlay_split_kernels

    runtime = get_cuda_runtime()
    kernels = _overlay_split_kernels()
    grid, block = runtime.launch_config(
        kernels["rank_exact_split_event_merge"],
        total,
    )
    ptr = runtime.pointer
    runtime.launch(
        kernels["rank_exact_split_event_merge"],
        grid=grid,
        block=block,
        params=(
            (
                ptr(left_run[0]),
                ptr(left_run[1]),
                left_count,
                ptr(right_run[0]),
                ptr(right_run[1]),
                right_count,
                ptr(left_positions),
                ptr(right_positions),
                total,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    merged = tuple(cp.empty(total, dtype=values.dtype) for values in left_run)
    for output, left_values, right_values in zip(
        merged,
        left_run,
        right_run,
        strict=True,
    ):
        output[left_positions] = left_values
        output[right_positions] = right_values

    keep = cp.empty(total, dtype=cp.bool_)
    keep[:-1] = (merged[0][:-1] != merged[0][1:]) | (merged[1][:-1] != merged[1][1:])
    keep[-1] = True
    key_positions = cp.flatnonzero(keep).astype(cp.int64, copy=False)
    previous_positions = cp.maximum(key_positions - 1, 0)
    has_equal_previous = (
        (key_positions > 0)
        & (merged[0][previous_positions] == merged[0][key_positions])
        & (merged[1][previous_positions] == merged[1][key_positions])
    )
    payload_positions = cp.where(
        has_equal_previous
        & (merged[4][previous_positions] > merged[4][key_positions]),
        previous_positions,
        key_positions,
    )
    return (
        merged[0][key_positions],
        merged[1][key_positions],
        merged[2][payload_positions],
        merged[3][payload_positions],
        merged[4][payload_positions],
    )


def _merge_split_event_runs_balanced(runs):
    """Merge sorted event pages with logarithmic rather than linear depth."""
    live = [run for run in runs if run is not None]
    while len(live) > 1:
        next_level = []
        for index in range(0, len(live), 2):
            if index + 1 == len(live):
                next_level.append(live[index])
            else:
                next_level.append(
                    _merge_sorted_split_event_runs(
                        live[index],
                        live[index + 1],
                    )
                )
        live = next_level
    return None if not live else live[0]


class _SplitEventRunAccumulator:
    """Online exact merge tree with one retained run per binary level."""

    def __init__(self) -> None:
        self._levels = []

    @staticmethod
    def _merge(left_run, right_run):
        merged = _merge_sorted_split_event_runs(left_run, right_run)
        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            (left_run, right_run),
            lambda _owners: None,
        )
        return merged

    def append(self, run) -> None:
        if run is None:
            return
        carry = run
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            older = self._levels[level]
            self._levels[level] = None
            carry = self._merge(older, carry)
            level += 1
        if level == len(self._levels):
            self._levels.append(carry)
        else:
            self._levels[level] = carry

    def finish(self):
        merged = None
        for run in reversed(self._levels):
            if run is None:
                continue
            merged = run if merged is None else self._merge(merged, run)
        self._levels.clear()
        return merged

    def retained_runs(self) -> tuple:
        return tuple(run for run in self._levels if run is not None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_gpu_split_events(
    left,
    right,
    *,
    intersection_result: SegmentIntersectionResult | PagedSegmentIntersectionResult | None = None,
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
    _require_gpu_arrays()
    runtime = get_cuda_runtime()
    normalized_dispatch_mode = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    classification_dispatch_mode = (
        ExecutionMode.CPU if normalized_dispatch_mode is ExecutionMode.CPU else ExecutionMode.GPU
    )

    # Lazy import to avoid circular dependency — kernel compile functions
    # live in gpu.py which imports from this module.
    from vibespatial.overlay.gpu import (
        _compute_live_split_event_budget,
        _overlay_split_kernels,
    )

    # One classified candidate can emit four exact split events. Bound the
    # producer page by the downstream classify/emit/sort phase, not by the
    # narrower raw-pair carrier alone.
    candidate_page_budget = max(
        _MIN_SPLIT_EVENT_CANDIDATE_PAGE_PAIRS,
        _compute_live_split_event_budget() // 4,
    )

    # GPU-native segment extraction -- no CPU loop, no host round-trip.
    # lyy.15: reuse pre-extracted right-side segments when provided
    # (N-vs-1 overlay caches the corridor segments once).
    _owns_left_segments = _cached_left_segments is None
    with hotpath_stage("overlay.split.extract_left_segments", category="setup"):
        left_segments = (
            _extract_segments_gpu(left)
            if _cached_left_segments is None
            else _cached_left_segments
        )
    if right_segment_broadcast is not None:
        if not require_same_row:
            raise ValueError("broadcast-right segment topology requires row isolation")
        if right_geometry_source_rows is not None:
            raise ValueError(
                "broadcast-right segment topology cannot also remap grouped right rows"
            )
        if int(right_segment_broadcast.logical_row_count) != int(left.row_count):
            raise ValueError(
                "broadcast-right logical rows must match left row_count "
                f"({left.row_count}), got {right_segment_broadcast.logical_row_count}"
            )
        if _cached_right_segments is not None and (
            _cached_right_segments is not right_segment_broadcast.physical_segments
        ):
            raise ValueError("cached right segments disagree with broadcast relation")
    _owns_right_segments = (
        _cached_right_segments is None and right_segment_broadcast is None
    )
    with hotpath_stage("overlay.split.extract_right_segments", category="setup"):
        right_segments = (
            right_segment_broadcast.physical_segments
            if right_segment_broadcast is not None
            else (
                _cached_right_segments
                if _cached_right_segments is not None
                else _extract_segments_gpu(right)
            )
        )
    effective_right_segments = right_segments
    remapped_right_row_indices = None
    if right_geometry_source_rows is not None:
        d_right_geometry_source_rows = cp.asarray(
            right_geometry_source_rows,
            dtype=cp.int32,
        )
        if int(d_right_geometry_source_rows.size) != int(right.row_count):
            raise ValueError(
                "right_geometry_source_rows must match the right row_count "
                f"({right.row_count}), got {int(d_right_geometry_source_rows.size)}"
            )
        remapped_right_row_indices = d_right_geometry_source_rows[
            cp.asarray(right_segments.row_indices, dtype=cp.int32)
        ].astype(cp.int32, copy=False)
        effective_right_segments = DeviceSegmentTable(
            row_indices=remapped_right_row_indices,
            segment_indices=right_segments.segment_indices,
            x0=right_segments.x0,
            y0=right_segments.y0,
            x1=right_segments.x1,
            y1=right_segments.y1,
            count=right_segments.count,
            part_indices=right_segments.part_indices,
            ring_indices=right_segments.ring_indices,
        )
    original_right_segment_rows = cp.asarray(right_segments.row_indices, dtype=cp.int32)
    if use_same_row_fast_path is None:
        use_same_row_fast_path = not require_same_row
    kernels = _overlay_split_kernels()
    left_count = int(left_segments.count)
    right_physical_count = int(effective_right_segments.count)
    right_count = (
        int(right_segment_broadcast.logical_count)
        if right_segment_broadcast is not None
        else right_physical_count
    )
    segment_total = left_count + right_count
    pair_event_accumulator = _SplitEventRunAccumulator()
    right_right_event_accumulator = _SplitEventRunAccumulator()

    def _release_classified_page(page: SegmentIntersectionResult) -> None:
        state = page.device_state
        if state is None:
            return
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

    def _append_cross_pair_events(page: SegmentIntersectionResult) -> None:
        batch = _emit_pair_split_event_batch(
            page,
            left_segments=left_segments,
            right_segments=effective_right_segments,
            kernels=kernels,
            broadcast_right=right_segment_broadcast is not None,
        )
        if batch is not None:
            # Cross-operand nodes define the shared arrangement vertex. A
            # same-side endpoint can carry the source coordinate a few ULPs
            # away, so the shared arrangement coordinate must win the merge.
            batch = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                (batch[4] + cp.int8(4)).astype(cp.int8, copy=False),
            )
            pair_event_accumulator.append(_sort_deduplicate_split_event_run(batch))

    def _consume_classified_page(page: SegmentIntersectionResult) -> None:
        try:
            _append_cross_pair_events(page)
        finally:
            _release_classified_page(page)

    def _same_side_split_event_batch(
        *,
        side_owned,
        side_segments: DeviceSegmentTable,
        source_offset: int,
        stage_name: str,
    ):
        """Emit native split events for same-side row-isolated topology."""
        side_count = int(side_segments.count)
        if not require_same_row or side_count < 2:
            return None
        side_event_accumulator = _SplitEventRunAccumulator()

        def _consume_side_page(page: SegmentIntersectionResult) -> None:
            try:
                batch = _emit_pair_split_event_batch(
                    page,
                    left_segments=side_segments,
                    right_segments=side_segments,
                    kernels=kernels,
                )
                if batch is None:
                    return
                source_ids = (cp.asarray(batch[0], dtype=cp.int32) % np.int32(side_count)).astype(
                    cp.int32, copy=False
                ) + np.int32(source_offset)
                normalized = (
                    source_ids,
                    batch[1],
                    batch[2],
                    batch[3],
                    batch[4],
                )
                side_event_accumulator.append(
                    _sort_deduplicate_split_event_run(normalized)
                )
            finally:
                _release_classified_page(page)

        source_order = cp.arange(side_count, dtype=cp.int32)
        try:
            with hotpath_stage(
                f"overlay.split.classify_{stage_name}_self_intersections", category="refine"
            ):
                side_result = classify_segment_intersections(
                    side_owned,
                    side_owned,
                    dispatch_mode=classification_dispatch_mode,
                    _cached_left_device_segments=side_segments,
                    _cached_right_device_segments=side_segments,
                    _require_same_row=True,
                    _use_same_row_fast_path=(
                        bool(same_row_single_group) or same_row_span_summary is not None
                    ),
                    _same_row_single_group=bool(same_row_single_group),
                    _same_row_span_summary=(
                        (
                            side_count,
                            side_count,
                            int(same_row_span_summary[2]),
                        )
                        if same_row_span_summary is not None
                        else None
                    ),
                    _collect_ambiguous_rows=False,
                    _strict_upper_source_rows=(source_order, source_order),
                    _compact_paged_non_disjoint=False,
                    _candidate_page_budget=candidate_page_budget,
                    _classified_page_consumer=_consume_side_page,
                )
            if side_result.runtime_selection.selected is not ExecutionMode.GPU:
                raise RuntimeError(
                    f"{stage_name} same-side split-event classification requires GPU execution"
                )
            if isinstance(side_result, PagedSegmentIntersectionResult):
                for page in side_result.pages:
                    _consume_side_page(page)
            else:
                _consume_side_page(side_result)
                side_result = None
            return side_event_accumulator.finish()
        finally:
            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                side_event_accumulator.retained_runs(),
                lambda _owners: None,
            )

    def _consume_right_right_page(page: SegmentIntersectionResult) -> None:
        """Filter and emit one grouped right-right classification page."""
        state = page.device_state
        filtered_page = None
        try:
            if state is None or page.count == 0:
                return
            with hotpath_stage("overlay.split.filter_right_right_pairs", category="filter"):
                left_lookup = cp.asarray(state.left_lookup, dtype=cp.int32)
                right_lookup = cp.asarray(state.right_lookup, dtype=cp.int32)
                keep = compact_indices(
                    (
                        (
                            cp.asarray(state.left_rows, dtype=cp.int32)
                            == cp.asarray(state.right_rows, dtype=cp.int32)
                        )
                        & (
                            original_right_segment_rows[left_lookup]
                            < original_right_segment_rows[right_lookup]
                        )
                    ).astype(cp.uint8, copy=False)
                ).values
                _sync_hotpath(runtime)
            if int(keep.size) == 0:
                return

            filtered_state = SegmentIntersectionDeviceState(
                left_rows=cp.asarray(state.left_rows)[keep],
                left_segments=cp.asarray(state.left_segments)[keep],
                left_lookup=left_lookup[keep],
                right_rows=cp.asarray(state.right_rows)[keep],
                right_segments=cp.asarray(state.right_segments)[keep],
                right_lookup=right_lookup[keep],
                kinds=cp.asarray(state.kinds)[keep],
                point_x=cp.asarray(state.point_x)[keep],
                point_y=cp.asarray(state.point_y)[keep],
                overlap_x0=cp.asarray(state.overlap_x0)[keep],
                overlap_y0=cp.asarray(state.overlap_y0)[keep],
                overlap_x1=cp.asarray(state.overlap_x1)[keep],
                overlap_y1=cp.asarray(state.overlap_y1)[keep],
                ambiguous_rows=cp.empty(0, dtype=cp.int32),
            )
            filtered_page = SegmentIntersectionResult(
                candidate_pairs=page.candidate_pairs,
                runtime_selection=page.runtime_selection,
                precision_plan=page.precision_plan,
                robustness_plan=page.robustness_plan,
                device_state=filtered_state,
                _count=int(keep.size),
            )
            batch = _emit_pair_split_event_batch(
                filtered_page,
                left_segments=effective_right_segments,
                right_segments=effective_right_segments,
                kernels=kernels,
            )
            if batch is None:
                return
            source_ids = (cp.asarray(batch[0], dtype=cp.int32) % np.int32(right_count)).astype(
                cp.int32, copy=False
            ) + np.int32(left_count)
            normalized = (
                source_ids,
                batch[1],
                batch[2],
                batch[3],
                batch[4],
            )
            right_right_event_accumulator.append(
                _sort_deduplicate_split_event_run(normalized)
            )
        finally:
            if filtered_page is not None:
                _release_classified_page(filtered_page)
            _release_classified_page(page)

    try:
        with hotpath_stage("overlay.split.classify_intersections", category="refine"):
            result = intersection_result or classify_segment_intersections(
                left,
                right,
                dispatch_mode=classification_dispatch_mode,
                _cached_left_device_segments=left_segments,
                _cached_right_device_segments=effective_right_segments,
                _require_same_row=(
                    require_same_row and right_segment_broadcast is None
                ),
                _use_same_row_fast_path=(
                    False if right_segment_broadcast is not None else use_same_row_fast_path
                ),
                _same_row_single_group=same_row_single_group,
                _same_row_span_summary=same_row_span_summary,
                _collect_ambiguous_rows=False,
                # Overlay applies a constructive-only ULP near-touch pass to
                # exact-disjoint candidates before event emission.
                _compact_paged_non_disjoint=False,
                _candidate_page_budget=candidate_page_budget,
                _classified_page_consumer=_consume_classified_page,
            )
    except Exception as exc:
        raise RuntimeError(
            f"overlay split left-right classification failed: {type(exc).__name__}: {exc}"
        ) from exc
    if result.runtime_selection.selected is not ExecutionMode.GPU:
        raise RuntimeError("build_gpu_split_events requires a GPU segment-intersection result")
    streamed_intersection_pages = isinstance(result, PagedSegmentIntersectionResult)
    if streamed_intersection_pages:
        for page in result.pages:
            _consume_classified_page(page)
        empty = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        result = SegmentIntersectionResult(
            candidate_pairs=result.candidate_pairs,
            runtime_selection=result.runtime_selection,
            precision_plan=result.precision_plan,
            robustness_plan=result.robustness_plan,
            device_state=SegmentIntersectionDeviceState(
                left_rows=empty,
                left_segments=empty,
                left_lookup=empty,
                right_rows=empty,
                right_segments=empty,
                right_lookup=empty,
                kinds=empty.astype(cp.int8, copy=False),
                point_x=empty_f64,
                point_y=empty_f64,
                overlap_x0=empty_f64,
                overlap_y0=empty_f64,
                overlap_x1=empty_f64,
                overlap_y1=empty_f64,
                ambiguous_rows=empty,
            ),
            _count=0,
        )
    owns_intersection_state = False
    event_result = result
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
        event_result = SegmentIntersectionResult(
            candidate_pairs=result.candidate_pairs,
            runtime_selection=result.runtime_selection,
            precision_plan=result.precision_plan,
            robustness_plan=result.robustness_plan,
            device_state=device_state,
            _count=result.count,
        )
    else:
        device_state = result.device_state
        owns_intersection_state = intersection_result is None and not streamed_intersection_pages

    base_event_count = segment_total * 2

    # Segment coordinate arrays are already device-resident from
    # _extract_segments_gpu -- use them directly, no from_host.
    left_x0 = left_segments.x0
    left_y0 = left_segments.y0
    left_x1 = left_segments.x1
    left_y1 = left_segments.y1
    right_x0 = effective_right_segments.x0
    right_y0 = effective_right_segments.y0
    right_x1 = effective_right_segments.x1
    right_y1 = effective_right_segments.y1

    endpoint_source_ids = runtime.allocate((base_event_count,), np.int32)
    endpoint_t = runtime.allocate((base_event_count,), np.float64)
    endpoint_x = runtime.allocate((base_event_count,), np.float64)
    endpoint_y = runtime.allocate((base_event_count,), np.float64)

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
                right_physical_count,
                right_count,
                ptr(endpoint_source_ids),
                ptr(endpoint_t),
                ptr(endpoint_x),
                ptr(endpoint_y),
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
                # Physical left/right counts plus logical right count.
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                # Exact event columns.
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        endpoint_grid, endpoint_block = runtime.launch_config(
            kernels["emit_endpoint_split_events"], base_event_count
        )
        with hotpath_stage("overlay.split.emit_endpoint_events", category="emit"):
            runtime.launch(
                kernels["emit_endpoint_split_events"],
                grid=endpoint_grid,
                block=endpoint_block,
                params=endpoint_params,
            )

        if event_result.count:
            with hotpath_stage("overlay.split.emit_pair_event_run", category="emit"):
                _append_cross_pair_events(event_result)

        try:
            if remapped_right_row_indices is not None:
                try:
                    with hotpath_stage(
                        "overlay.split.classify_right_right_intersections", category="refine"
                    ):
                        right_right = classify_segment_intersections(
                            right,
                            right,
                            dispatch_mode=classification_dispatch_mode,
                            _cached_left_device_segments=effective_right_segments,
                            _cached_right_device_segments=effective_right_segments,
                            _require_same_row=True,
                            _use_same_row_fast_path=(
                                bool(same_row_single_group) or same_row_span_summary is not None
                            ),
                            _same_row_single_group=bool(same_row_single_group),
                            _same_row_span_summary=(
                                (
                                    int(same_row_span_summary[1]),
                                    int(same_row_span_summary[1]),
                                    int(same_row_span_summary[2]),
                                )
                                if same_row_span_summary is not None
                                else None
                            ),
                            _collect_ambiguous_rows=False,
                            _strict_upper_source_rows=(
                                original_right_segment_rows,
                                original_right_segment_rows,
                            ),
                            _compact_paged_non_disjoint=False,
                            _candidate_page_budget=candidate_page_budget,
                            _classified_page_consumer=_consume_right_right_page,
                        )
                    if right_right.runtime_selection.selected is not ExecutionMode.GPU:
                        raise RuntimeError(
                            "grouped right-right split-event classification requires GPU execution"
                        )
                    if isinstance(right_right, PagedSegmentIntersectionResult):
                        for page in right_right.pages:
                            _consume_right_right_page(page)
                    else:
                        _consume_right_right_page(right_right)
                        right_right = None
                except Exception as exc:
                    raise RuntimeError(
                        f"overlay split grouped right-right event pipeline failed: {type(exc).__name__}: {exc}"
                    ) from exc

            same_side_event_batches = []
            if include_same_side_splits and require_same_row and remapped_right_row_indices is None:
                left_self_events = _same_side_split_event_batch(
                    side_owned=left,
                    side_segments=left_segments,
                    source_offset=0,
                    stage_name="left",
                )
                if left_self_events is not None:
                    same_side_event_batches.append(left_self_events)
                right_self_events = _same_side_split_event_batch(
                    side_owned=right,
                    side_segments=effective_right_segments,
                    source_offset=left_count,
                    stage_name="right",
                )
                if right_self_events is not None:
                    same_side_event_batches.append(right_self_events)

            try:
                with hotpath_stage("overlay.split.external_merge_events", category="sort"):
                    endpoint_event_run = (
                        cp.asarray(endpoint_source_ids),
                        cp.asarray(endpoint_t),
                        cp.asarray(endpoint_x),
                        cp.asarray(endpoint_y),
                        cp.zeros(base_event_count, dtype=cp.int8),
                    )
                    event_run = _merge_split_event_runs_balanced(
                        [
                            endpoint_event_run,
                            pair_event_accumulator.finish(),
                            right_right_event_accumulator.finish(),
                            *same_side_event_batches,
                        ]
                    )
                    same_side_event_batches.clear()
                    (
                        unique_source_ids,
                        unique_t,
                        unique_x,
                        unique_y,
                        unique_priority,
                    ) = event_run
                    _sync_hotpath(runtime)

                # Derive source_side / row / part / ring indices on GPU.
                with hotpath_stage("overlay.split.segment_metadata", category="emit"):
                    metadata_builder = (
                        _broadcast_segment_metadata_gpu
                        if right_segment_broadcast is not None
                        else _segment_metadata_gpu
                    )
                    (
                        d_source_side,
                        d_row_indices,
                        d_part_indices,
                        d_ring_indices,
                        d_geometry_indices,
                    ) = metadata_builder(
                        unique_source_ids,
                        left_count=left_count,
                        left_segments=left_segments,
                        right_segments=effective_right_segments,
                        **(
                            {}
                            if right_segment_broadcast is not None
                            else {"right_geometry_segments": right_segments}
                        ),
                    )

                with hotpath_stage("overlay.split.canonicalize_source_endpoints", category="sort"):
                    (
                        unique_x,
                        unique_y,
                        unique_priority,
                        d_planarity_risk,
                    ) = _canonicalize_source_endpoint_coordinates(
                        unique_source_ids,
                        unique_t,
                        unique_x,
                        unique_y,
                        unique_priority,
                        d_source_side,
                        d_row_indices,
                        d_part_indices,
                        d_ring_indices,
                        d_geometry_indices,
                        segment_count=segment_total,
                        left_segments=left_segments,
                        right_segments=effective_right_segments,
                        broadcast_right=right_segment_broadcast is not None,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"overlay split event assembly failed: {type(exc).__name__}: {exc}"
                ) from exc
            event_count = int(unique_source_ids.size)
            # Two endpoints per source segment are the irreducible event
            # carrier. Only an interior paired event can create second-order
            # support requiring atomic re-noding; endpoint-only contacts are
            # canonicalized back to their exact source coordinates above.
            requires_renoding = event_count > base_event_count

            return SplitEventTable(
                left_segment_count=left_count,
                right_segment_count=right_count,
                runtime_selection=result.runtime_selection,
                requires_renoding=requires_renoding,
                device_state=SplitEventDeviceState(
                    source_segment_ids=unique_source_ids,
                    t=unique_t,
                    x=unique_x,
                    y=unique_y,
                    source_side=d_source_side,
                    row_indices=d_row_indices,
                    part_indices=d_part_indices,
                    ring_indices=d_ring_indices,
                    geometry_indices=d_geometry_indices,
                    planarity_risk=d_planarity_risk,
                ),
                _count=event_count,
            )
        finally:
            get_cuda_completion_retainer().defer(
                cp.cuda.get_current_stream(),
                (
                    pair_event_accumulator.retained_runs(),
                    right_right_event_accumulator.retained_runs(),
                    tuple(locals().get("same_side_event_batches", ())),
                ),
                lambda _owners: None,
            )
    finally:
        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            (
                endpoint_source_ids,
                endpoint_t,
                endpoint_x,
                endpoint_y,
            ),
            lambda _owners: None,
        )
        # Free DeviceSegmentTable arrays (x0/y0/x1/y1 are aliases of
        # left_x0 etc., plus row/segment/part/ring metadata).
        # lyy.15: skip freeing right_segments when they are cached
        # (caller owns the lifetime of the cached segments).
        _segs_to_free = [left_segments] if _owns_left_segments else []
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
        if remapped_right_row_indices is not None:
            runtime.free(remapped_right_row_indices)
        # The final split-event run may be the endpoint run itself when no
        # pair or same-side events were emitted. Let CuPy retain or release
        # these allocations by reference so ownership transfers correctly to
        # the returned SplitEventTable instead of leaving it with freed views.
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


def _source_ring_transition_state(split_events, kernels):
    """Return dense source metadata, tangents, and exact semantic ring signs."""
    runtime = get_cuda_runtime()
    state = split_events.device_state
    segment_total = split_events.left_segment_count + split_events.right_segment_count
    d_ids = cp.asarray(state.source_segment_ids, dtype=cp.int32)
    d_t = cp.asarray(state.t, dtype=cp.float64)
    d_x = cp.asarray(state.x, dtype=cp.float64)
    d_y = cp.asarray(state.y, dtype=cp.float64)
    source_rows = cp.empty(segment_total, dtype=cp.int32)
    source_parts = cp.empty(segment_total, dtype=cp.int32)
    source_rings = cp.empty(segment_total, dtype=cp.int32)
    source_geometries = cp.empty(segment_total, dtype=cp.int32)
    source_rows[d_ids] = cp.asarray(state.row_indices, dtype=cp.int32)
    source_parts[d_ids] = cp.asarray(state.part_indices, dtype=cp.int32)
    source_rings[d_ids] = cp.asarray(state.ring_indices, dtype=cp.int32)
    if state.geometry_indices is None:
        source_geometries[:] = source_rows
    else:
        source_geometries[d_ids] = cp.asarray(
            state.geometry_indices,
            dtype=cp.int32,
        )
    x0 = cp.empty(segment_total, dtype=cp.float64)
    y0 = cp.empty(segment_total, dtype=cp.float64)
    x1 = cp.empty(segment_total, dtype=cp.float64)
    y1 = cp.empty(segment_total, dtype=cp.float64)
    start = d_t == 0.0
    end = d_t == 1.0
    x0[d_ids[start]] = d_x[start]
    y0[d_ids[start]] = d_y[start]
    x1[d_ids[end]] = d_x[end]
    y1[d_ids[end]] = d_y[end]

    source_side = cp.where(
        cp.arange(segment_total, dtype=cp.int32) < split_events.left_segment_count,
        cp.int8(1),
        cp.int8(2),
    )
    ring_start_mask = cp.empty(segment_total, dtype=cp.bool_)
    ring_start_mask[0] = True
    if segment_total > 1:
        ring_start_mask[1:] = (
            (source_side[1:] != source_side[:-1])
            | (source_geometries[1:] != source_geometries[:-1])
            | (source_rows[1:] != source_rows[:-1])
            | (source_parts[1:] != source_parts[:-1])
            | (source_rings[1:] != source_rings[:-1])
        )
    ring_starts = cp.flatnonzero(ring_start_mask).astype(cp.int32, copy=False)
    ring_ends = cp.concatenate(
        (ring_starts[1:], cp.asarray([segment_total], dtype=cp.int32))
    )
    ring_count = int(ring_starts.size)
    transition = cp.empty(segment_total, dtype=cp.int32)
    grid, block = runtime.launch_config(
        kernels["derive_source_ring_transition_signs"], ring_count
    )
    ptr = runtime.pointer
    runtime.launch(
        kernels["derive_source_ring_transition_signs"],
        grid=grid,
        block=block,
        params=((
            ptr(x0), ptr(y0), ptr(x1), ptr(y1), ptr(ring_starts),
            ptr(ring_ends), ptr(source_rings), ptr(transition), ring_count,
        ), (KERNEL_PARAM_PTR,) * 8 + (KERNEL_PARAM_I32,)),
    )
    source_tangent_x = x1 - x0
    source_tangent_y = y1 - y0
    # ``runtime.launch`` receives raw pointers. Retain ephemeral ring-index and
    # endpoint owners until this stream reaches the launch; otherwise CuPy's
    # pool can recycle them while the exact orientation kernel is in flight.
    get_cuda_completion_retainer().defer(
        cp.cuda.get_current_stream(),
        (
            x0,
            y0,
            x1,
            y1,
            ring_starts,
            ring_ends,
        ),
        lambda _owners: None,
    )
    return (
        source_rows,
        source_parts,
        source_rings,
        source_tangent_x,
        source_tangent_y,
        transition,
    )


def _build_gpu_atomic_edges_once(
    split_events: SplitEventTable,
    *,
    isolate_rows: bool = False,
    preserve_source_orientation: bool = False,
    _parent_atomic_state: AtomicEdgeDeviceState | None = None,
) -> AtomicEdgeTable:
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
                source_membership=runtime.allocate((0,), np.uint8),
                tangent_x=empty_device_f64,
                tangent_y=empty_device_f64,
                left_coverage_delta=empty_device_i32,
                right_coverage_delta=empty_device_i32,
            ),
            _count=0,
        )

    adjacency_mask = (device.source_segment_ids[:-1] == device.source_segment_ids[1:]).astype(
        cp.uint8, copy=False
    )
    adjacency_counts = adjacency_mask.astype(cp.int32, copy=False)
    adjacency_offsets = exclusive_sum(adjacency_counts)
    segment_total = split_events.left_segment_count + split_events.right_segment_count
    pair_count = int(split_events.count) - int(segment_total)
    if pair_count < 0:
        raise RuntimeError("split event table has fewer events than source segments")

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
        d_adjacent_positions = cp.flatnonzero(adjacency_mask)
        raw_edge_rows = None
        if isolate_rows:
            d_adj_rows = cp.asarray(device.row_indices)[d_adjacent_positions]
            raw_edge_rows = cp.empty(pair_count * 2, dtype=cp.int32)
            raw_edge_rows[0::2] = d_adj_rows
            raw_edge_rows[1::2] = d_adj_rows
        (
            source_rows,
            source_parts,
            source_rings,
            source_tangent_x,
            source_tangent_y,
            source_transition,
        ) = _source_ring_transition_state(split_events, kernels)
        raw_ids = cp.asarray(out_source_ids, dtype=cp.int32)
        raw_sign = cp.where(
            cp.asarray(out_direction, dtype=cp.int8) == 0,
            cp.int32(1),
            cp.int32(-1),
        )
        parent_forward_edges = None
        if _parent_atomic_state is None:
            raw_membership = cp.where(
                raw_ids < split_events.left_segment_count,
                cp.uint8(1),
                cp.uint8(2),
            )
            raw_transition = source_transition[raw_ids] * raw_sign
            raw_left_delta = cp.where(
                raw_ids < split_events.left_segment_count,
                raw_transition,
                cp.int32(0),
            )
            raw_right_delta = cp.where(
                raw_ids < split_events.left_segment_count,
                cp.int32(0),
                raw_transition,
            )
        else:
            parent_forward_edges = cp.arange(
                0,
                int(_parent_atomic_state.source_segment_ids.size),
                2,
                dtype=cp.int32,
            )
            if int(parent_forward_edges.size) != segment_total:
                raise RuntimeError(
                    "parent atomic provenance must contain one forward occurrence "
                    "per re-noded source segment"
                )
            raw_parent_edges = parent_forward_edges[raw_ids]
            raw_membership = cp.asarray(
                _parent_atomic_state.source_membership,
                dtype=cp.uint8,
            )[raw_parent_edges]
            raw_left_delta = (
                cp.asarray(
                    _parent_atomic_state.left_coverage_delta,
                    dtype=cp.int32,
                )[raw_parent_edges]
                * raw_sign
            )
            raw_right_delta = (
                cp.asarray(
                    _parent_atomic_state.right_coverage_delta,
                    dtype=cp.int32,
                )[raw_parent_edges]
                * raw_sign
            )
        (
            dedup_source_ids,
            dedup_direction,
            dedup_src_x,
            dedup_src_y,
            dedup_dst_x,
            dedup_dst_y,
            dedup_source_membership,
            dedup_left_delta,
            dedup_right_delta,
            dedup_planarity_risk,
        ) = _deduplicate_atomic_edge_geometry(
            out_source_ids,
            out_direction,
            out_src_x,
            out_src_y,
            out_dst_x,
            out_dst_y,
            occurrence_membership=raw_membership,
            occurrence_left_delta=raw_left_delta,
            occurrence_right_delta=raw_right_delta,
            occurrence_planarity_risk=(
                None
                if device.planarity_risk is None
                else cp.repeat(
                    (
                        cp.asarray(device.planarity_risk, dtype=cp.bool_)[
                            d_adjacent_positions
                        ]
                        | cp.asarray(device.planarity_risk, dtype=cp.bool_)[
                            d_adjacent_positions + 1
                        ]
                    ),
                    2,
                )
            ),
            row_indices=raw_edge_rows,
            left_segment_count=split_events.left_segment_count,
            preserve_source_orientation=preserve_source_orientation,
        )
        runtime.free(out_source_ids)
        runtime.free(out_direction)
        runtime.free(out_src_x)
        runtime.free(out_src_y)
        runtime.free(out_dst_x)
        runtime.free(out_dst_y)

        # Derive source_side and row / part / ring indices on GPU.
        #
        # split_events.source_segment_ids is ordered by packed event key, not
        # by source segment id, so direct searchsorted on that array is
        # invalid once multiple rows/segments are interleaved. Source ids are
        # already dense over the left+right segment table, so build a direct
        # device lookup once and gather metadata without a second global sort.
        d_out_ids = cp.asarray(
            dedup_source_ids
        )  # zcopy:ok(already device-resident — cp.asarray is a no-op)
        d_row_indices = source_rows[d_out_ids]
        d_part_indices = source_parts[d_out_ids]
        d_ring_indices = source_rings[d_out_ids]

        d_forward_tangent = cp.asarray(dedup_direction, dtype=cp.int8) == 0
        if _parent_atomic_state is None:
            output_source_ids = dedup_source_ids
            d_source_side = cp.where(
                d_out_ids < split_events.left_segment_count,
                cp.int8(1),
                cp.int8(2),
            )
            selected_tangent_x = source_tangent_x[d_out_ids]
            selected_tangent_y = source_tangent_y[d_out_ids]
        else:
            selected_parent_edges = parent_forward_edges[d_out_ids]
            output_source_ids = cp.asarray(
                _parent_atomic_state.source_segment_ids,
                dtype=cp.int32,
            )[selected_parent_edges]
            d_source_side = cp.asarray(
                _parent_atomic_state.source_side,
                dtype=cp.int8,
            )[selected_parent_edges]
            selected_tangent_x = cp.asarray(
                _parent_atomic_state.tangent_x,
                dtype=cp.float64,
            )[selected_parent_edges]
            selected_tangent_y = cp.asarray(
                _parent_atomic_state.tangent_y,
                dtype=cp.float64,
            )[selected_parent_edges]
        d_tangent_x = cp.where(
            d_forward_tangent,
            selected_tangent_x,
            -selected_tangent_x,
        )
        d_tangent_y = cp.where(
            d_forward_tangent,
            selected_tangent_y,
            -selected_tangent_y,
        )

        # Row/part/ring stay on device; downstream build_gpu_half_edge_graph
        # reads device_state directly.  Host copies are lazily materialized
        # via AtomicEdgeTable.row_indices / part_indices / ring_indices
        # properties on first access.
        return AtomicEdgeTable(
            left_segment_count=split_events.left_segment_count,
            right_segment_count=split_events.right_segment_count,
            runtime_selection=split_events.runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=output_source_ids,
                direction=dedup_direction,
                src_x=dedup_src_x,
                src_y=dedup_src_y,
                dst_x=dedup_dst_x,
                dst_y=dedup_dst_y,
                row_indices=d_row_indices,
                part_indices=d_part_indices,
                ring_indices=d_ring_indices,
                source_side=d_source_side,
                source_membership=dedup_source_membership,
                tangent_x=d_tangent_x,
                tangent_y=d_tangent_y,
                left_coverage_delta=dedup_left_delta,
                right_coverage_delta=dedup_right_delta,
                planarity_risk=dedup_planarity_risk,
            ),
            _count=int(dedup_source_ids.size),
        )
    finally:
        runtime.free(adjacency_mask)
        runtime.free(adjacency_offsets)


def renode_gpu_atomic_edges(
    atomic_edges: AtomicEdgeTable,
    *,
    isolate_rows: bool,
    preserve_source_orientation: bool = False,
) -> AtomicEdgeTable:
    """Split second-order crossings until the atomic relation is planar.

    Refined intersection coordinates can create a new crossing a few ULPs
    from a source endpoint. A planar overlay graph therefore needs a fixed-
    point noding stage after the first source-segment split. Each pass is
    shaped by forward atomic segments and returns immediately when the merged
    event carrier contains only the two existing endpoints per segment.
    """
    _require_gpu_arrays()
    if atomic_edges.count < 4:
        return atomic_edges

    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import (
        FAMILY_TAGS,
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )
    from vibespatial.overlay.gpu import _overlay_split_kernels

    runtime = get_cuda_runtime()
    kernels = _overlay_split_kernels()
    current = atomic_edges
    refine_all = False

    while True:
        state = current.device_state
        if state is None:
            return current
        segment_count = current.count // 2
        if segment_count < 2:
            return current

        d_src_x = cp.asarray(state.src_x, dtype=cp.float64)
        d_src_y = cp.asarray(state.src_y, dtype=cp.float64)
        d_forward = cp.arange(0, current.count, 2, dtype=cp.int32)
        d_rows = cp.asarray(state.row_indices, dtype=cp.int32)[d_forward]
        d_parts = cp.asarray(state.part_indices, dtype=cp.int32)[d_forward]
        d_rings = cp.asarray(state.ring_indices, dtype=cp.int32)[d_forward]
        d_segment_ids = cp.arange(segment_count, dtype=cp.int32)
        # DeviceSegmentTable is a pointer-based kernel carrier; strided even/
        # odd half-edge views would be interpreted as dense arrays by CUDA.
        # Materialize each forward occurrence as one contiguous source segment.
        d_segment_x0 = cp.ascontiguousarray(d_src_x[0::2])
        d_segment_y0 = cp.ascontiguousarray(d_src_y[0::2])
        d_segment_x1 = cp.ascontiguousarray(d_src_x[1::2])
        d_segment_y1 = cp.ascontiguousarray(d_src_y[1::2])
        segments = DeviceSegmentTable(
            row_indices=d_rows,
            segment_indices=d_segment_ids,
            x0=d_segment_x0,
            y0=d_segment_y0,
            x1=d_segment_x1,
            y1=d_segment_y1,
            count=segment_count,
            part_indices=d_parts,
            ring_indices=d_rings,
        )
        if refine_all or state.planarity_risk is None:
            d_refine_segment_ids = d_segment_ids
        else:
            d_refine_segment_ids = cp.flatnonzero(
                cp.asarray(state.planarity_risk, dtype=cp.bool_)[d_forward]
            ).astype(cp.int32, copy=False)
        refine_segment_count = d_refine_segment_ids.shape[0]
        if refine_segment_count == 0:
            return current
        refine_segments = DeviceSegmentTable(
            row_indices=d_rows[d_refine_segment_ids],
            segment_indices=d_refine_segment_ids,
            x0=d_segment_x0[d_refine_segment_ids],
            y0=d_segment_y0[d_refine_segment_ids],
            x1=d_segment_x1[d_refine_segment_ids],
            y1=d_segment_y1[d_refine_segment_ids],
            count=refine_segment_count,
            part_indices=d_parts[d_refine_segment_ids],
            ring_indices=d_rings[d_refine_segment_ids],
        )
        line_owned = build_device_resident_owned(
            device_families={
                GeometryFamily.LINESTRING: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.LINESTRING,
                    x=d_src_x,
                    y=d_src_y,
                    geometry_offsets=(
                        cp.arange(segment_count + 1, dtype=cp.int32) * cp.int32(2)
                    ),
                    empty_mask=cp.zeros(segment_count, dtype=cp.bool_),
                    bounds=None,
                )
            },
            row_count=segment_count,
            tags=cp.full(
                segment_count,
                FAMILY_TAGS[GeometryFamily.LINESTRING],
                dtype=cp.int8,
            ),
            validity=cp.ones(segment_count, dtype=cp.bool_),
            family_row_offsets=d_segment_ids,
            execution_mode="gpu",
        )

        pair_runs = []

        def _release_page(page: SegmentIntersectionResult) -> None:
            page_state = page.device_state
            if page_state is None:
                return
            for values in (
                page_state.left_rows,
                page_state.left_segments,
                page_state.left_lookup,
                page_state.right_rows,
                page_state.right_segments,
                page_state.right_lookup,
                page_state.kinds,
                page_state.point_x,
                page_state.point_y,
                page_state.overlap_x0,
                page_state.overlap_y0,
                page_state.overlap_x1,
                page_state.overlap_y1,
                page_state.ambiguous_rows,
            ):
                runtime.free(values)

        def _consume_page(
            page: SegmentIntersectionResult,
            *,
            _segments=segments,
            _refine_segments=refine_segments,
            _refine_segment_ids=d_refine_segment_ids,
            _pair_runs=pair_runs,
            _segment_count=segment_count,
        ) -> None:
            try:
                batch = _emit_pair_split_event_batch(
                    page,
                    left_segments=_refine_segments,
                    right_segments=_segments,
                    kernels=kernels,
                )
                if batch is None:
                    return
                _pair_runs.append(
                    _sort_deduplicate_split_event_run(
                        (
                            cp.where(
                                cp.asarray(batch[0], dtype=cp.int32)
                                < np.int32(_refine_segments.count),
                                _refine_segment_ids[
                                    cp.minimum(
                                        cp.asarray(batch[0], dtype=cp.int32),
                                        np.int32(_refine_segments.count - 1),
                                    )
                                ],
                                cp.asarray(batch[0], dtype=cp.int32)
                                - np.int32(_refine_segments.count),
                            ),
                            batch[1],
                            batch[2],
                            batch[3],
                            batch[4],
                        )
                    )
                )
            finally:
                _release_page(page)

        classified = classify_segment_intersections(
            line_owned,
            line_owned,
            dispatch_mode=ExecutionMode.GPU,
            _cached_left_device_segments=refine_segments,
            _cached_right_device_segments=segments,
            _require_same_row=isolate_rows,
            # Re-noding is an atomic-segment relation refinement, not a
            # public grouped-row matrix. The counted sweep preserves row
            # isolation without inheriting caller-specific warp shape.
            _use_same_row_fast_path=False,
            _same_row_single_group=False,
            _collect_ambiguous_rows=False,
            _compact_paged_non_disjoint=False,
            _classified_page_consumer=_consume_page,
        )
        if isinstance(classified, PagedSegmentIntersectionResult):
            for page in classified.pages:
                _consume_page(page)
        else:
            _consume_page(classified)

        endpoint_ids = cp.repeat(d_segment_ids, 2)
        endpoint_run = (
            endpoint_ids,
            cp.tile(cp.asarray([0.0, 1.0], dtype=cp.float64), segment_count),
            d_src_x,
            d_src_y,
            cp.full(segment_count * 2, cp.int8(127), dtype=cp.int8),
        )
        merged = _merge_split_event_runs_balanced([endpoint_run, *pair_runs])
        if merged is None or int(merged[0].size) == segment_count * 2:
            return current

        d_event_source = cp.asarray(merged[0], dtype=cp.int32)
        d_event_forward_edges = d_forward[d_event_source]
        split_events = SplitEventTable(
            left_segment_count=segment_count,
            right_segment_count=0,
            runtime_selection=current.runtime_selection,
            device_state=SplitEventDeviceState(
                source_segment_ids=d_event_source,
                t=merged[1],
                x=merged[2],
                y=merged[3],
                source_side=cp.asarray(state.source_side, dtype=cp.int8)[
                    d_event_forward_edges
                ],
                row_indices=cp.asarray(state.row_indices, dtype=cp.int32)[
                    d_event_forward_edges
                ],
                part_indices=cp.asarray(state.part_indices, dtype=cp.int32)[
                    d_event_forward_edges
                ],
                ring_indices=cp.asarray(state.ring_indices, dtype=cp.int32)[
                    d_event_forward_edges
                ],
            ),
            _count=d_event_source.size,
        )
        rebuilt = _build_gpu_atomic_edges_once(
            split_events,
            isolate_rows=isolate_rows,
            preserve_source_orientation=preserve_source_orientation,
            _parent_atomic_state=state,
        )
        rebuilt = replace(
            rebuilt,
            left_segment_count=current.left_segment_count,
            right_segment_count=current.right_segment_count,
        )
        reached_fixed_point = rebuilt.count == current.count
        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            split_events.device_state,
            lambda _state: None,
        )
        # Several atomic columns may be views into shared pool allocations.
        # Keep their Python owners alive through the last consumer instead of
        # explicitly freeing an underlying allocation through one view.
        get_cuda_completion_retainer().defer(
            cp.cuda.get_current_stream(),
            (
                state,
                segments,
                refine_segments,
                d_refine_segment_ids,
                line_owned,
            ),
            lambda _owners: None,
        )
        current = rebuilt
        if reached_fixed_point:
            return current
        refine_all = True


def build_gpu_atomic_edges(
    split_events: SplitEventTable,
    *,
    isolate_rows: bool = False,
    preserve_source_orientation: bool = False,
) -> AtomicEdgeTable:
    """Build a planar exact atomic-edge carrier from source split events.

    Pairwise refined intersection coordinates can introduce second-order
    crossings after the first source-segment split. Atomic topology therefore
    includes fixed-point re-noding as part of its contract; consumers never
    receive a merely once-split graph and do not need operation-specific
    planarity repair.
    """
    initial = _build_gpu_atomic_edges_once(
        split_events,
        isolate_rows=isolate_rows,
        preserve_source_orientation=preserve_source_orientation,
    )
    if not split_events.requires_renoding:
        return initial
    return renode_gpu_atomic_edges(
        initial,
        isolate_rows=isolate_rows,
        preserve_source_orientation=preserve_source_orientation,
    )


def noded_boundary_segments_from_split_events_gpu(
    split_events: SplitEventTable,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Return one directed boundary atom for every adjacent split-event pair.

    Unlike :func:`build_gpu_atomic_edges`, this carrier intentionally preserves
    source multiplicity. Coverage union needs that multiplicity so coincident
    atoms can be removed by group-local odd parity after every partial overlap
    has been noded. The returned arrays own their gathered storage and remain
    valid after the split-event state is released.
    """
    _require_gpu_arrays()
    device = split_events.device_state
    if device is None or split_events.count < 2:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        return empty_f64, empty_f64, empty_f64, empty_f64, empty_i32

    d_source_ids = cp.asarray(device.source_segment_ids, dtype=cp.int32)
    d_adjacency = d_source_ids[:-1] == d_source_ids[1:]
    d_rows = cp.asarray(device.row_indices, dtype=cp.int32)
    d_x = cp.asarray(device.x, dtype=cp.float64)
    d_y = cp.asarray(device.y, dtype=cp.float64)
    return (
        d_x[:-1][d_adjacency],
        d_y[:-1][d_adjacency],
        d_x[1:][d_adjacency],
        d_y[1:][d_adjacency],
        d_rows[:-1][d_adjacency],
    )


def renode_grouped_boundary_segments_gpu(
    start_x,
    start_y,
    end_x,
    end_y,
    row_indices,
    *,
    placeholder_owned,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Reach a planar fixed point for grouped boundary atoms on the device.

    Intersection coordinates can induce a crossing between newly split atoms
    even when every original source pair was classified exactly.  Reclassify
    the segment relation itself in bounded pages and preserve one output atom
    per source occurrence; collective coverage depends on that multiplicity.
    """
    _require_gpu_arrays()
    from vibespatial.overlay.gpu import _overlay_split_kernels

    runtime = get_cuda_runtime()
    kernels = _overlay_split_kernels()
    current_start_x = cp.asarray(start_x, dtype=cp.float64)
    current_start_y = cp.asarray(start_y, dtype=cp.float64)
    current_end_x = cp.asarray(end_x, dtype=cp.float64)
    current_end_y = cp.asarray(end_y, dtype=cp.float64)
    current_rows = cp.asarray(row_indices, dtype=cp.int32)

    def _release_page(page: SegmentIntersectionResult) -> None:
        state = page.device_state
        if state is None:
            return
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

    def _consume_page(
        page: SegmentIntersectionResult,
        *,
        start_x,
        start_y,
        end_x,
        end_y,
        segments: DeviceSegmentTable,
        segment_count: int,
        pair_runs: list,
    ) -> None:
        try:
            batch = _emit_pair_split_event_batch(
                page,
                left_segments=segments,
                right_segments=segments,
                kernels=kernels,
            )
            if batch is None:
                return
            pair_runs.append(
                _sort_deduplicate_split_event_run(
                    (
                        (cp.asarray(batch[0], dtype=cp.int32) % segment_count).astype(
                            cp.int32,
                            copy=False,
                        ),
                        batch[1],
                        batch[2],
                        batch[3],
                        batch[4],
                    )
                )
            )
        finally:
            _release_page(page)

    while True:
        segment_count = current_start_x.size
        if segment_count < 2:
            return (
                current_start_x,
                current_start_y,
                current_end_x,
                current_end_y,
                current_rows,
            )

        d_segment_ids = cp.arange(segment_count, dtype=cp.int32)
        d_zero_metadata = cp.zeros(segment_count, dtype=cp.int32)
        segments = DeviceSegmentTable(
            row_indices=current_rows,
            segment_indices=d_segment_ids,
            x0=current_start_x,
            y0=current_start_y,
            x1=current_end_x,
            y1=current_end_y,
            count=segment_count,
            part_indices=d_zero_metadata,
            ring_indices=d_zero_metadata,
        )
        pair_runs = []
        consume_page = partial(
            _consume_page,
            start_x=current_start_x,
            start_y=current_start_y,
            end_x=current_end_x,
            end_y=current_end_y,
            segments=segments,
            segment_count=segment_count,
            pair_runs=pair_runs,
        )

        with hotpath_stage("overlay.split.renode_grouped_boundary", category="refine"):
            classified = classify_segment_intersections(
                placeholder_owned,
                placeholder_owned,
                dispatch_mode=ExecutionMode.GPU,
                _cached_left_device_segments=segments,
                _cached_right_device_segments=segments,
                _require_same_row=True,
                _use_same_row_fast_path=False,
                _collect_ambiguous_rows=False,
                _strict_upper_source_rows=(d_segment_ids, d_segment_ids),
                _compact_paged_non_disjoint=False,
                _classified_page_consumer=consume_page,
            )
            if isinstance(classified, PagedSegmentIntersectionResult):
                for page in classified.pages:
                    consume_page(page)
            else:
                consume_page(classified)

        endpoint_source_ids = cp.repeat(d_segment_ids, 2)
        endpoint_t = cp.tile(cp.asarray([0.0, 1.0], dtype=cp.float64), segment_count)
        endpoint_x = cp.empty(segment_count * 2, dtype=cp.float64)
        endpoint_y = cp.empty(segment_count * 2, dtype=cp.float64)
        endpoint_x[0::2] = current_start_x
        endpoint_x[1::2] = current_end_x
        endpoint_y[0::2] = current_start_y
        endpoint_y[1::2] = current_end_y
        endpoint_run = (
            endpoint_source_ids,
            endpoint_t,
            endpoint_x,
            endpoint_y,
            cp.full(segment_count * 2, cp.int8(127), dtype=cp.int8),
        )
        merged = _merge_split_event_runs_balanced([endpoint_run, *pair_runs])
        if merged is None:
            return (
                current_start_x,
                current_start_y,
                current_end_x,
                current_end_y,
                current_rows,
            )
        if int(merged[0].size) == segment_count * 2:
            return (
                current_start_x,
                current_start_y,
                current_end_x,
                current_end_y,
                current_rows,
            )

        merged_source_ids = cp.asarray(merged[0], dtype=cp.int32)
        adjacent = merged_source_ids[:-1] == merged_source_ids[1:]
        next_start_x = cp.asarray(merged[2], dtype=cp.float64)[:-1][adjacent]
        next_start_y = cp.asarray(merged[3], dtype=cp.float64)[:-1][adjacent]
        next_end_x = cp.asarray(merged[2], dtype=cp.float64)[1:][adjacent]
        next_end_y = cp.asarray(merged[3], dtype=cp.float64)[1:][adjacent]
        next_source_ids = merged_source_ids[:-1][adjacent]
        live = (next_start_x != next_end_x) | (next_start_y != next_end_y)
        current_start_x = next_start_x[live]
        current_start_y = next_start_y[live]
        current_end_x = next_end_x[live]
        current_end_y = next_end_y[live]
        current_rows = current_rows[next_source_ids[live]]
        if current_start_x.size == segment_count:
            return (
                current_start_x,
                current_start_y,
                current_end_x,
                current_end_y,
                current_rows,
            )
