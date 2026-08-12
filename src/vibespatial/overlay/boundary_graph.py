"""Exact device boundary relations and polygon reconstruction.

This module is the neutral sink for constructive paths that already own an
exact polygon boundary. It reduces duplicate atoms, builds the canonical
half-edge graph, classifies disconnected contour nesting, and emits owned
polygon buffers without routing through grouped union or host ring assembly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    get_cuda_runtime,
)
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import RuntimeSelection

from .types import (
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    OverlayFaceDeviceState,
    OverlayFaceTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def build_atomic_edges_from_boundary_segments_gpu(
    start_x: DeviceArray,
    start_y: DeviceArray,
    end_x: DeviceArray,
    end_y: DeviceArray,
    *,
    row_indices: DeviceArray | None = None,
    runtime_selection: RuntimeSelection,
) -> AtomicEdgeTable | None:
    """Build adjacent forward/reverse half-edges from exact boundary atoms."""
    if cp is None:
        raise RuntimeError("CuPy is required for device boundary reconstruction")
    boundary_count = int(start_x.size)
    if boundary_count == 0:
        return None

    d_segment_ids = cp.arange(boundary_count, dtype=cp.int32)
    total_atomic = boundary_count * 2
    d_src_x = cp.empty(total_atomic, dtype=cp.float64)
    d_src_y = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_x = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_y = cp.empty(total_atomic, dtype=cp.float64)
    d_source_ids = cp.empty(total_atomic, dtype=cp.int32)
    d_direction = cp.empty(total_atomic, dtype=cp.int8)
    if row_indices is None:
        d_row_indices = cp.zeros(total_atomic, dtype=cp.int32)
    else:
        d_boundary_rows = cp.asarray(row_indices, dtype=cp.int32)
        if int(d_boundary_rows.size) != boundary_count:
            raise ValueError("boundary row index count must match segment count")
        d_row_indices = cp.empty(total_atomic, dtype=cp.int32)
        d_row_indices[0::2] = d_boundary_rows
        d_row_indices[1::2] = d_boundary_rows
    d_part_indices = cp.zeros(total_atomic, dtype=cp.int32)
    d_ring_indices = cp.zeros(total_atomic, dtype=cp.int32)
    d_source_side = cp.ones(total_atomic, dtype=cp.int8)

    d_src_x[0::2] = start_x
    d_src_x[1::2] = end_x
    d_src_y[0::2] = start_y
    d_src_y[1::2] = end_y
    d_dst_x[0::2] = end_x
    d_dst_x[1::2] = start_x
    d_dst_y[0::2] = end_y
    d_dst_y[1::2] = start_y
    d_source_ids[0::2] = d_segment_ids
    d_source_ids[1::2] = d_segment_ids
    d_direction[0::2] = 0
    d_direction[1::2] = 1
    d_tangent_x = d_dst_x - d_src_x
    d_tangent_y = d_dst_y - d_src_y
    d_zero_delta = cp.zeros(total_atomic, dtype=cp.int32)

    return AtomicEdgeTable(
        left_segment_count=boundary_count,
        right_segment_count=0,
        runtime_selection=runtime_selection,
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=d_source_ids,
            direction=d_direction,
            src_x=d_src_x,
            src_y=d_src_y,
            dst_x=d_dst_x,
            dst_y=d_dst_y,
            row_indices=d_row_indices,
            part_indices=d_part_indices,
            ring_indices=d_ring_indices,
            source_side=d_source_side,
            tangent_x=d_tangent_x,
            tangent_y=d_tangent_y,
            left_coverage_delta=d_zero_delta,
            right_coverage_delta=d_zero_delta.copy(),
        ),
        _count=total_atomic,
    )


def undirected_boundary_segment_orders_gpu(
    start_x: DeviceArray,
    start_y: DeviceArray,
    end_x: DeviceArray,
    end_y: DeviceArray,
    row_indices: DeviceArray | None,
    active_mask: DeviceArray | None = None,
    keep_one_per_run: bool = False,
):
    """Return source positions for exact undirected boundary segment runs."""
    if cp is None:
        raise RuntimeError("CuPy is required for device boundary reconstruction")
    from .graph import _fp64_radix_keys, _stable_radix_order_pass

    start_x = cp.asarray(start_x, dtype=cp.float64)
    start_y = cp.asarray(start_y, dtype=cp.float64)
    end_x = cp.asarray(end_x, dtype=cp.float64)
    end_y = cp.asarray(end_y, dtype=cp.float64)
    row_indices = None if row_indices is None else cp.asarray(row_indices, dtype=cp.int32)
    segment_count = int(start_x.size)
    if segment_count == 0:
        return cp.empty(0, dtype=cp.int32)

    active = (
        cp.ones(segment_count, dtype=cp.bool_)
        if active_mask is None
        else cp.asarray(active_mask, dtype=cp.bool_)
    )
    if int(active.size) != segment_count:
        raise ValueError("boundary activity mask must match segment capacity")
    safe_start_x = cp.where(active, start_x, cp.float64(0.0))
    safe_start_y = cp.where(active, start_y, cp.float64(0.0))
    safe_end_x = cp.where(active, end_x, cp.float64(0.0))
    safe_end_y = cp.where(active, end_y, cp.float64(0.0))
    swap = (safe_start_x > safe_end_x) | (
        (safe_start_x == safe_end_x) & (safe_start_y > safe_end_y)
    )
    order = cp.arange(segment_count, dtype=cp.int32)
    for swapped_values, regular_values in (
        (safe_start_y, safe_end_y),
        (safe_start_x, safe_end_x),
        (safe_end_y, safe_start_y),
        (safe_end_x, safe_start_x),
    ):
        values = cp.where(swap, swapped_values, regular_values)
        keys = _fp64_radix_keys(values)
        del values
        order = _stable_radix_order_pass(order, keys)
        del keys
    if row_indices is not None:
        safe_rows = cp.where(active, row_indices, cp.int32(0))
        order = _stable_radix_order_pass(order, safe_rows)
    order = _stable_radix_order_pass(order, (~active).astype(cp.int8))

    run_starts_mask = cp.empty(segment_count, dtype=cp.bool_)
    run_starts_mask[0] = True
    if segment_count > 1:
        run_starts_mask[1:] = False
        sorted_active = active[order]
        run_starts_mask[1:] |= sorted_active[1:] != sorted_active[:-1]
        del sorted_active
        if row_indices is not None:
            sorted_rows = safe_rows[order]
            run_starts_mask[1:] = sorted_rows[1:] != sorted_rows[:-1]
            del sorted_rows
        for swapped_values, regular_values in (
            (safe_end_x, safe_start_x),
            (safe_end_y, safe_start_y),
            (safe_start_x, safe_end_x),
            (safe_start_y, safe_end_y),
        ):
            values = cp.where(swap, swapped_values, regular_values)
            sorted_values = values[order]
            del values
            run_starts_mask[1:] |= sorted_values[1:] != sorted_values[:-1]
            del sorted_values
    del swap

    run_starts = cp.flatnonzero(run_starts_mask).astype(cp.int32, copy=False)
    run_ends = cp.concatenate((run_starts[1:], cp.asarray([segment_count], dtype=cp.int32)))
    run_lengths = run_ends - run_starts
    selected_runs = (
        cp.ones(run_lengths.size, dtype=cp.bool_)
        if keep_one_per_run
        else (run_lengths & np.int32(1)) != 0
    )
    boundary_order = order[run_starts[selected_runs]]
    return boundary_order[active[boundary_order]]


def _vertical_microcell_boundary_segments_gpu(
    row_indices,
    x_left,
    x_right,
    y_lower_left,
    y_lower_right,
    y_upper_left,
    y_upper_right,
) -> tuple[Any, Any, Any, Any, Any]:
    """Atomize vertical cell sides by exact ``(row, x, y)`` event scans."""
    from .graph import _fp64_radix_keys, _stable_radix_order_pass

    cell_count = int(row_indices.size)
    if cell_count == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        return empty_i32, empty_f64, empty_f64, empty_f64, empty_f64

    d_rows = cp.asarray(row_indices, dtype=cp.int32)
    side_rows = cp.concatenate((d_rows, d_rows))
    side_x = cp.concatenate((x_left, x_right)).astype(cp.float64, copy=False)
    side_low = cp.minimum(
        cp.concatenate((y_lower_left, y_lower_right)),
        cp.concatenate((y_upper_left, y_upper_right)),
    ).astype(cp.float64, copy=False)
    side_high = cp.maximum(
        cp.concatenate((y_lower_left, y_lower_right)),
        cp.concatenate((y_upper_left, y_upper_right)),
    ).astype(cp.float64, copy=False)
    side_sign = cp.concatenate(
        (
            -cp.ones(cell_count, dtype=cp.int32),
            cp.ones(cell_count, dtype=cp.int32),
        )
    )

    event_rows = cp.concatenate((side_rows, side_rows))
    event_x = cp.concatenate((side_x, side_x))
    event_y = cp.concatenate((side_low, side_high))
    event_delta = cp.concatenate((side_sign, -side_sign))
    event_count = int(event_y.size)

    order = cp.arange(event_count, dtype=cp.int32)
    y_keys = _fp64_radix_keys(event_y)
    order = _stable_radix_order_pass(order, y_keys)
    del y_keys
    x_keys = _fp64_radix_keys(event_x)
    order = _stable_radix_order_pass(order, x_keys)
    del x_keys
    order = _stable_radix_order_pass(order, event_rows)

    sorted_rows = event_rows[order]
    sorted_x = event_x[order]
    sorted_y = event_y[order]
    sorted_delta = event_delta[order]
    event_run_start = cp.empty(event_count, dtype=cp.bool_)
    event_run_start[0] = True
    if event_count > 1:
        event_run_start[1:] = (
            (sorted_rows[1:] != sorted_rows[:-1])
            | (sorted_x[1:] != sorted_x[:-1])
            | (sorted_y[1:] != sorted_y[:-1])
        )
    run_starts = cp.flatnonzero(event_run_start).astype(cp.int32, copy=False)
    run_rows = sorted_rows[run_starts]
    run_x = sorted_x[run_starts]
    run_y = sorted_y[run_starts]
    run_delta = cp.add.reduceat(sorted_delta, run_starts).astype(
        cp.int32,
        copy=False,
    )
    run_count = int(run_starts.size)
    if run_count < 2:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        return empty_i32, empty_f64, empty_f64, empty_f64, empty_f64

    interface_start = cp.empty(run_count, dtype=cp.bool_)
    interface_start[0] = True
    interface_start[1:] = (run_rows[1:] != run_rows[:-1]) | (run_x[1:] != run_x[:-1])
    interface_starts = cp.flatnonzero(interface_start).astype(cp.int32, copy=False)
    interface_ids = cp.cumsum(interface_start.astype(cp.int32), dtype=cp.int32) - 1
    cumulative_delta = cp.cumsum(run_delta, dtype=cp.int32)
    interface_base = cumulative_delta[interface_starts] - run_delta[interface_starts]
    winding = cumulative_delta - interface_base[interface_ids]
    live = ~interface_start[1:] & (run_y[1:] > run_y[:-1]) & (winding[:-1] != 0)
    live_runs = cp.flatnonzero(live).astype(cp.int32, copy=False)
    return (
        run_rows[live_runs],
        run_x[live_runs],
        run_y[live_runs],
        run_x[live_runs],
        run_y[live_runs + 1],
    )


def microcell_boundary_segments_gpu(
    row_indices,
    x_left,
    x_right,
    y_lower_left,
    y_lower_right,
    y_upper_left,
    y_upper_right,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return exact surviving boundary atoms for selected microcell bands."""
    if cp is None:
        raise RuntimeError("CuPy is required for device boundary reconstruction")
    d_rows = cp.asarray(row_indices, dtype=cp.int32)
    cell_count = int(d_rows.size)
    if cell_count == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        return empty_i32, empty_f64, empty_f64, empty_f64, empty_f64

    x_left = cp.asarray(x_left, dtype=cp.float64)
    x_right = cp.asarray(x_right, dtype=cp.float64)
    y_lower_left = cp.asarray(y_lower_left, dtype=cp.float64)
    y_lower_right = cp.asarray(y_lower_right, dtype=cp.float64)
    y_upper_left = cp.asarray(y_upper_left, dtype=cp.float64)
    y_upper_right = cp.asarray(y_upper_right, dtype=cp.float64)

    vertical = _vertical_microcell_boundary_segments_gpu(
        d_rows,
        x_left,
        x_right,
        y_lower_left,
        y_lower_right,
        y_upper_left,
        y_upper_right,
    )
    vertical_rows, vertical_sx, vertical_sy, vertical_tx, vertical_ty = vertical
    edge_rows = cp.concatenate((d_rows, d_rows, vertical_rows))
    start_x = cp.concatenate((x_left, x_left, vertical_sx))
    start_y = cp.concatenate((y_lower_left, y_upper_left, vertical_sy))
    end_x = cp.concatenate((x_right, x_right, vertical_tx))
    end_y = cp.concatenate((y_lower_right, y_upper_right, vertical_ty))

    keep = undirected_boundary_segment_orders_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        edge_rows,
    )
    return (
        edge_rows[keep],
        start_x[keep],
        start_y[keep],
        end_x[keep],
        end_y[keep],
    )


def build_polygon_output_from_boundary_segments_gpu(
    start_x,
    start_y,
    end_x,
    end_y,
    *,
    row_indices,
    row_count: int,
    runtime_selection: RuntimeSelection,
    d_valid_empty_rows=None,
) -> OwnedGeometryArray:
    """Assemble exact boundary atoms through the canonical overlay graph."""
    if cp is None:
        raise RuntimeError("CuPy is required for device boundary reconstruction")
    from .assemble import _build_polygon_output_from_faces_gpu, _empty_polygon_output
    from .faces import _build_indexed_component_containment_device_state
    from .gpu import _overlay_face_walk_kernels
    from .graph import (
        _gpu_face_walk,
        build_endpoint_incidence_device_state,
        build_gpu_half_edge_graph,
    )

    start_x = cp.asarray(start_x, dtype=cp.float64)
    start_y = cp.asarray(start_y, dtype=cp.float64)
    end_x = cp.asarray(end_x, dtype=cp.float64)
    end_y = cp.asarray(end_y, dtype=cp.float64)
    row_indices = cp.asarray(row_indices, dtype=cp.int32)
    atomic_edges = build_atomic_edges_from_boundary_segments_gpu(
        start_x,
        start_y,
        end_x,
        end_y,
        row_indices=row_indices,
        runtime_selection=runtime_selection,
    )
    if atomic_edges is None:
        if d_valid_empty_rows is not None:
            from vibespatial.geometry.owned import build_empty_polygon_rows_device

            return build_empty_polygon_rows_device(
                row_count,
                validity=cp.asarray(d_valid_empty_rows, dtype=cp.bool_),
            )
        return _empty_polygon_output(runtime_selection, row_count=row_count)
    runtime = get_cuda_runtime()
    kernels = _overlay_face_walk_kernels()
    ptr = runtime.pointer
    incidence = build_endpoint_incidence_device_state(
        atomic_edges,
        isolate_rows=True,
    )
    segment_count = int(start_x.size)
    node_capacity = incidence.edge_count
    d_active_segments = cp.ones(segment_count, dtype=cp.int32)
    d_node_degree = cp.empty(node_capacity, dtype=cp.int32)
    d_node_queued = cp.zeros(node_capacity, dtype=cp.int32)
    d_leaf_queue = cp.empty(node_capacity, dtype=cp.int32)
    d_queue_ready = cp.zeros(node_capacity, dtype=cp.int32)
    d_queue_head = cp.zeros(1, dtype=cp.int32)
    d_queue_tail = cp.zeros(1, dtype=cp.int32)
    d_pending = cp.zeros(1, dtype=cp.int32)
    init_grid, init_block = runtime.launch_config(
        kernels["initialize_degree_two_core_frontier"],
        node_capacity,
    )
    runtime.launch(
        kernels["initialize_degree_two_core_frontier"],
        grid=init_grid,
        block=init_block,
        params=(
            (
                ptr(incidence.node_offsets),
                ptr(incidence.node_count),
                ptr(d_node_degree),
                ptr(d_node_queued),
                ptr(d_leaf_queue),
                ptr(d_queue_tail),
                ptr(d_queue_ready),
                ptr(d_pending),
                node_capacity,
            ),
            (KERNEL_PARAM_PTR,) * 8 + (KERNEL_PARAM_I32,),
        ),
    )
    peel_grid, peel_block = runtime.launch_config(
        kernels["peel_degree_two_core_frontier"],
        node_capacity,
    )
    runtime.launch(
        kernels["peel_degree_two_core_frontier"],
        grid=peel_grid,
        block=peel_block,
        params=(
            (
                ptr(incidence.incidence_edge_ids),
                ptr(incidence.node_offsets),
                ptr(incidence.src_node_ids),
                ptr(incidence.node_count),
                ptr(d_active_segments),
                ptr(d_node_degree),
                ptr(d_node_queued),
                ptr(d_leaf_queue),
                ptr(d_queue_head),
                ptr(d_queue_tail),
                ptr(d_queue_ready),
                ptr(d_pending),
                node_capacity,
                segment_count,
            ),
            (KERNEL_PARAM_PTR,) * 12 + (KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    # This is the single admitted physicalization boundary: after the O(E)
    # device peel, compact surviving undirected segments once. No convergence
    # state or peel round ever crosses to the host.
    keep_order = cp.flatnonzero(d_active_segments != 0).astype(cp.int32, copy=False)
    keep_count = int(keep_order.size)
    if keep_count == 0:
        if d_valid_empty_rows is not None:
            from vibespatial.geometry.owned import build_empty_polygon_rows_device

            return build_empty_polygon_rows_device(
                row_count,
                validity=cp.asarray(d_valid_empty_rows, dtype=cp.bool_),
            )
        return _empty_polygon_output(runtime_selection, row_count=row_count)
    atomic_edges = build_atomic_edges_from_boundary_segments_gpu(
        start_x[keep_order],
        start_y[keep_order],
        end_x[keep_order],
        end_y[keep_order],
        row_indices=row_indices[keep_order],
        runtime_selection=runtime_selection,
    )
    graph = build_gpu_half_edge_graph(atomic_edges, isolate_rows=True)
    face_walk = _gpu_face_walk(graph)
    (
        d_face_offsets,
        d_face_edge_ids,
        d_edge_face_ids,
        d_bounded_mask,
        d_signed_area,
        d_centroid_x,
        d_centroid_y,
        face_count,
    ) = face_walk
    if face_count == 0:
        return _empty_polygon_output(runtime_selection, row_count=row_count)

    face_ids = cp.arange(face_count, dtype=cp.int32)
    first_edges = d_face_edge_ids[d_face_offsets[:-1]]
    twin_faces = d_edge_face_ids[first_edges ^ cp.int32(1)]
    d_component = cp.where(
        face_walk.cycle_orientation < 0,
        face_ids,
        twin_faces,
    ).astype(
        cp.int32, copy=False
    )
    nesting = _build_indexed_component_containment_device_state(
        graph,
        d_face_offsets,
        d_face_edge_ids,
        face_walk.cycle_orientation,
        d_component,
        face_count,
        isolate_rows=True,
    )
    d_depth = nesting.component_depth
    d_bounded = d_bounded_mask != 0
    d_even_depth = (d_depth[d_component] & np.int32(1)) == 0
    selected_faces = cp.flatnonzero(d_bounded == d_even_depth).astype(
        cp.int32,
        copy=False,
    )
    empty_i8 = cp.zeros(face_count, dtype=cp.int8)
    faces = OverlayFaceTable(
        runtime_selection=runtime_selection,
        _face_count=face_count,
        device_state=OverlayFaceDeviceState(
            face_offsets=d_face_offsets,
            face_edge_ids=d_face_edge_ids,
            edge_face_ids=d_edge_face_ids,
            bounded_mask=d_bounded_mask,
            signed_area=d_signed_area,
            centroid_x=d_centroid_x,
            centroid_y=d_centroid_y,
            left_covered=empty_i8,
            right_covered=empty_i8.copy(),
        ),
    )
    result = _build_polygon_output_from_faces_gpu(
        graph,
        faces,
        selected_faces,
        preserve_row_count=row_count,
        d_valid_empty_rows=d_valid_empty_rows,
        component_nesting=nesting,
    )
    if result is None:
        raise RuntimeError("device boundary graph assembly did not return an owned result")
    return result
