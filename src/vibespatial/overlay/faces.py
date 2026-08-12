"""Exact device-resident face labeling and overlay face construction."""

from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_completion_retainer,
    get_cuda_runtime,
)
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.spatial.segment_primitives import SegmentIntersectionResult

from ._host_boundary import overlay_device_to_host
from .types import (
    AtomicEdgeTable,
    HalfEdgeGraph,
    IndexedComponentContainmentDeviceState,
    OverlayFaceDeviceState,
    OverlayFaceTable,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def _build_indexed_component_containment_device_state(
    half_edge_graph: HalfEdgeGraph,
    face_offsets,
    face_edge_ids,
    cycle_orientation,
    face_component,
    face_count: int,
    *,
    isolate_rows: bool,
    left_winding=None,
    right_winding=None,
) -> IndexedComponentContainmentDeviceState:
    """Build fixed-capacity exact containment metadata on device."""
    from vibespatial.overlay.gpu import _overlay_face_walk_kernels
    from vibespatial.overlay.graph import _stable_radix_order_pass

    runtime = get_cuda_runtime()
    kernels = _overlay_face_walk_kernels()
    device = half_edge_graph.device_state
    ptr = runtime.pointer

    face_bounds = cp.empty(face_count * 4, dtype=cp.float64)
    bounds_grid, bounds_block = runtime.launch_config(
        kernels["compute_face_bounds"],
        face_count,
    )
    runtime.launch(
        kernels["compute_face_bounds"],
        grid=bounds_grid,
        block=bounds_block,
        params=(
            (
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(device.src_x),
                ptr(device.src_y),
                ptr(face_bounds),
                face_count,
            ),
            (KERNEL_PARAM_PTR,) * 5 + (KERNEL_PARAM_I32,),
        ),
    )

    bounds_matrix = face_bounds.reshape(face_count, 4)
    face_ids = cp.arange(face_count, dtype=cp.int32)
    root_faces = cp.where(
        cycle_orientation < 0,
        face_ids,
        cp.int32(-1),
    ).astype(cp.int32, copy=False)

    # Keep candidate storage at face capacity.  Stable least-to-most passes
    # sort positive cycles by min-X and place invalid sentinels at the end.
    candidate_active = cycle_orientation > 0
    candidate_order = _stable_radix_order_pass(face_ids, bounds_matrix[:, 0])
    candidate_order = _stable_radix_order_pass(
        candidate_order,
        (~candidate_active).astype(cp.int32),
    )
    candidate_faces = cp.where(
        candidate_active[candidate_order],
        candidate_order,
        cp.int32(-1),
    ).astype(cp.int32, copy=False)

    leaf_count = 1 << max(0, (face_count - 1).bit_length())
    interval_max_x = cp.full(leaf_count * 2, -cp.inf, dtype=cp.float64)
    safe_candidates = cp.maximum(candidate_faces, cp.int32(0))
    interval_max_x[leaf_count : leaf_count + face_count] = cp.where(
        candidate_faces >= 0,
        bounds_matrix[safe_candidates, 2],
        -cp.inf,
    )
    level_start = leaf_count >> 1
    while level_start:
        interval_max_x[level_start : level_start * 2] = cp.maximum(
            interval_max_x[level_start * 2 : level_start * 4 : 2],
            interval_max_x[level_start * 2 + 1 : level_start * 4 : 2],
        )
        level_start >>= 1

    left_baseline = cp.zeros(face_count, dtype=cp.int32)
    right_baseline = cp.zeros(face_count, dtype=cp.int32)
    component_depth = cp.zeros(face_count, dtype=cp.int32)
    component_parent = cp.full(face_count, -1, dtype=cp.int32)
    reduce_winding = left_winding is not None and right_winding is not None
    if (left_winding is None) != (right_winding is None):
        raise ValueError("component winding baselines require both winding arrays")

    # This shape intentionally uses one 256-lane block per face-capacity root
    # lane.  The kernel skips inactive roots and maps its lanes to depth-eight
    # tree subtrees, preserving block saturation even for one deeply nested root.
    root_grid = (face_count, 1, 1)
    root_block = (256, 1, 1)
    runtime.launch(
        kernels["reduce_indexed_component_containment"],
        grid=root_grid,
        block=root_block,
        params=(
            (
                ptr(root_faces),
                ptr(candidate_faces),
                ptr(interval_max_x),
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(face_bounds),
                ptr(device.src_x),
                ptr(device.src_y),
                ptr(device.row_indices),
                ptr(face_component),
                ptr(left_winding),
                ptr(right_winding),
                ptr(left_baseline),
                ptr(right_baseline),
                ptr(component_depth),
                face_count,
                leaf_count,
                np.int32(isolate_rows),
                np.int32(reduce_winding),
            ),
            (KERNEL_PARAM_PTR,) * 15 + (KERNEL_PARAM_I32,) * 4,
        ),
    )
    runtime.launch(
        kernels["select_indexed_component_containment_parent"],
        grid=root_grid,
        block=root_block,
        params=(
            (
                ptr(root_faces),
                ptr(candidate_faces),
                ptr(interval_max_x),
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(face_bounds),
                ptr(device.src_x),
                ptr(device.src_y),
                ptr(device.row_indices),
                ptr(face_component),
                ptr(component_depth),
                ptr(component_parent),
                face_count,
                leaf_count,
                np.int32(isolate_rows),
            ),
            (KERNEL_PARAM_PTR,) * 12 + (KERNEL_PARAM_I32,) * 3,
        ),
    )

    return IndexedComponentContainmentDeviceState(
        face_component=face_component,
        face_bounds=face_bounds,
        root_faces=root_faces,
        candidate_faces=candidate_faces,
        interval_max_x=interval_max_x,
        left_baseline=left_baseline,
        right_baseline=right_baseline,
        component_depth=component_depth,
        component_parent=component_parent,
        face_capacity=face_count,
        leaf_count=leaf_count,
        transient_owners=(
            face_ids,
            candidate_active,
            candidate_order,
            safe_candidates,
        ),
    )


def _gpu_propagate_face_coverage(
    half_edge_graph: HalfEdgeGraph,
    face_offsets,
    face_edge_ids,
    edge_face_ids,
    cycle_orientation,
    face_count: int,
    *,
    isolate_rows: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Propagate exact operand winding over every dual-graph component."""
    from vibespatial.overlay.gpu import _overlay_face_walk_kernels

    runtime = get_cuda_runtime()
    kernels = _overlay_face_walk_kernels()
    device = half_edge_graph.device_state
    if device.left_coverage_delta is None or device.right_coverage_delta is None:
        raise RuntimeError("overlay dual propagation requires signed edge deltas")
    edge_count = half_edge_graph.edge_count
    if (
        int(device.left_coverage_delta.size) != edge_count
        or int(device.right_coverage_delta.size) != edge_count
        or int(edge_face_ids.size) != edge_count
        or int(face_offsets.size) != face_count + 1
    ):
        raise RuntimeError("overlay dual propagation received inconsistent topology capacity")

    ptr = runtime.pointer
    sentinel = np.iinfo(np.int32).min
    left_winding = cp.full(face_count, sentinel, dtype=cp.int32)
    right_winding = cp.full(face_count, sentinel, dtype=cp.int32)
    face_component = cp.full(face_count, -1, dtype=cp.int32)
    queue = cp.empty(edge_count, dtype=cp.int32)
    queue_head = cp.zeros(1, dtype=cp.int32)
    queue_tail = cp.zeros(1, dtype=cp.int32)
    queue_ready = cp.zeros(edge_count, dtype=cp.int32)
    pending = cp.zeros(1, dtype=cp.int32)
    propagation_grid, propagation_block = runtime.launch_config(
        kernels["initialize_dual_face_queue"],
        face_count,
    )
    runtime.launch(
        kernels["initialize_dual_face_queue"],
        grid=propagation_grid,
        block=propagation_block,
        params=(
            (
                ptr(cycle_orientation),
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(queue),
                ptr(queue_tail),
                ptr(queue_ready),
                ptr(pending),
                ptr(left_winding),
                ptr(right_winding),
                ptr(face_component),
                face_count,
            ),
            (KERNEL_PARAM_PTR,) * 10 + (KERNEL_PARAM_I32,),
        ),
    )
    queue_grid, queue_block = runtime.launch_config(
        kernels["propagate_dual_face_queue"],
        face_count,
    )
    runtime.launch(
        kernels["propagate_dual_face_queue"],
        grid=queue_grid,
        block=queue_block,
        params=(
            (
                ptr(face_offsets),
                ptr(face_edge_ids),
                ptr(edge_face_ids),
                ptr(device.left_coverage_delta),
                ptr(device.right_coverage_delta),
                ptr(queue),
                ptr(queue_head),
                ptr(queue_tail),
                ptr(queue_ready),
                ptr(pending),
                ptr(left_winding),
                ptr(right_winding),
                ptr(face_component),
                face_count,
                edge_count,
            ),
            (KERNEL_PARAM_PTR,) * 13 + (KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    containment = _build_indexed_component_containment_device_state(
        half_edge_graph,
        face_offsets,
        face_edge_ids,
        cycle_orientation,
        face_component,
        face_count,
        isolate_rows=isolate_rows,
        left_winding=left_winding,
        right_winding=right_winding,
    )
    left_baseline = containment.left_baseline
    right_baseline = containment.right_baseline

    left_covered = cp.empty(face_count, dtype=cp.int8)
    right_covered = cp.empty(face_count, dtype=cp.int8)
    runtime.launch(
        kernels["finalize_face_coverage"],
        grid=propagation_grid,
        block=propagation_block,
        params=(
            (
                ptr(face_component),
                ptr(left_baseline),
                ptr(right_baseline),
                ptr(left_winding),
                ptr(right_winding),
                ptr(left_covered),
                ptr(right_covered),
                face_count,
            ),
            (KERNEL_PARAM_PTR,) * 7 + (KERNEL_PARAM_I32,),
        ),
    )
    get_cuda_completion_retainer().defer(
        cp.cuda.get_current_stream(),
        (
            queue,
            queue_head,
            queue_tail,
            queue_ready,
            pending,
            left_winding,
            right_winding,
            face_component,
            containment,
            left_baseline,
            right_baseline,
        ),
        lambda _owners: None,
    )
    return left_covered, right_covered


def _overlay_face_selection_mask_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
) -> cp.ndarray:
    """Return the face-capacity operation mask without compacting topology."""
    device = faces.device_state
    bounded = cp.asarray(device.bounded_mask)
    left = cp.asarray(device.left_covered)
    right = cp.asarray(device.right_covered)

    if operation == "intersection":
        mask = (left != 0) & (right != 0)
    elif operation == "union":
        mask = (left != 0) | (right != 0)
    elif operation == "difference":
        mask = (left != 0) & (right == 0)
    elif operation == "right_difference":
        mask = (left == 0) & (right != 0)
    elif operation == "symmetric_difference":
        mask = left != right
    elif operation == "identity":
        mask = left != 0
    elif operation == "polygonize":
        mask = bounded != 0
    else:
        raise ValueError(f"unsupported overlay operation: {operation}")
    return mask


def _select_overlay_face_selection_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
):
    """Keep selected overlay faces in a capacity-backed device carrier."""
    from vibespatial.api._native_rowset import NativeDeviceSelection

    return NativeDeviceSelection.from_mask(
        _overlay_face_selection_mask_gpu(faces, operation=operation),
    )


def _select_overlay_face_indices_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
) -> cp.ndarray:
    """Compact selected face IDs for debug and explicit CPU assembly only."""
    return cp.flatnonzero(
        _overlay_face_selection_mask_gpu(faces, operation=operation)
    ).astype(cp.int32)


def _selected_face_indices_to_host(d_selected_face_indices: cp.ndarray) -> np.ndarray:
    """Export selected faces only at the admitted CPU face-assembly boundary."""
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    item_count = int(
        getattr(d_selected_face_indices, "size", len(d_selected_face_indices))
    )
    itemsize = int(
        getattr(getattr(d_selected_face_indices, "dtype", None), "itemsize", 0)
    )
    record_materialization_event(
        surface="vibespatial.overlay.faces._assemble_faces_from_device_indices",
        boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
        operation="selected_face_indices_to_host",
        reason="device selected overlay face indices were materialized for CPU face assembly",
        detail=f"faces={item_count}, bytes={item_count * itemsize}",
        d2h_transfer=True,
        strict_disallowed=False,
    )
    return overlay_device_to_host(
        d_selected_face_indices,
        reason=(
            "vibespatial.overlay.faces._assemble_faces_from_device_indices"
            "::selected_face_indices_to_host"
        ),
        dtype=np.int64,
    )


def _assemble_faces_from_device_indices(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    d_selected_face_indices: cp.ndarray,
) -> OwnedGeometryArray:
    """Assemble selected overlay faces through the admitted device path."""
    from vibespatial.overlay.assemble import (
        _build_polygon_output_from_faces_gpu,
        _empty_polygon_output,
    )

    if d_selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)
    result = _build_polygon_output_from_faces_gpu(
        half_edge_graph,
        faces,
        d_selected_face_indices,
    )
    if result is None:
        raise RuntimeError("admitted GPU face assembly returned no device result")
    return result


def build_gpu_overlay_faces(
    left,
    right,
    *,
    half_edge_graph: HalfEdgeGraph | None = None,
    atomic_edges: AtomicEdgeTable | None = None,
    split_events: SplitEventTable | None = None,
    intersection_result: SegmentIntersectionResult | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    row_isolated: bool = False,
    left_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    right_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    right_geometry_broadcast: bool = False,
) -> OverlayFaceTable:
    """Build exact face topology and winding labels without host fallback."""
    from vibespatial.overlay.gpu import build_gpu_atomic_edges, build_gpu_split_events
    from vibespatial.overlay.graph import _gpu_face_walk, build_gpu_half_edge_graph

    if cp is None:
        raise RuntimeError("CuPy is required for GPU overlay face construction")
    _ = (
        left_geometry_source_rows,
        right_geometry_source_rows,
        right_geometry_broadcast,
    )
    runtime = get_cuda_runtime()
    if half_edge_graph is None:
        if atomic_edges is None:
            if split_events is None:
                split_events = build_gpu_split_events(
                    left,
                    right,
                    intersection_result=intersection_result,
                    dispatch_mode=dispatch_mode,
                )
            atomic_edges = build_gpu_atomic_edges(split_events)
        half_edge_graph = build_gpu_half_edge_graph(atomic_edges)

    if half_edge_graph.device_state is None:
        raise RuntimeError("GPU overlay face construction requires device topology")
    if half_edge_graph.edge_count == 0:
        empty_i32 = runtime.allocate((0,), np.int32)
        empty_i8 = runtime.allocate((0,), np.int8)
        empty_f64 = runtime.allocate((0,), np.float64)
        return OverlayFaceTable(
            runtime_selection=half_edge_graph.runtime_selection,
            _face_count=0,
            device_state=OverlayFaceDeviceState(
                face_offsets=runtime.allocate((1,), np.int32),
                face_edge_ids=empty_i32,
                edge_face_ids=empty_i32,
                bounded_mask=empty_i8,
                signed_area=empty_f64,
                centroid_x=empty_f64,
                centroid_y=empty_f64,
                left_covered=empty_i8,
                right_covered=empty_i8,
                cycle_orientation=empty_i8,
            ),
        )

    face_walk_result = _gpu_face_walk(half_edge_graph)
    (
        face_offsets,
        face_edge_ids,
        edge_face_ids,
        bounded_mask,
        signed_area,
        centroid_x,
        centroid_y,
        face_count,
    ) = face_walk_result
    cycle_orientation = face_walk_result.cycle_orientation

    if face_count:
        left_covered, right_covered = _gpu_propagate_face_coverage(
            half_edge_graph,
            face_offsets,
            face_edge_ids,
            edge_face_ids,
            cycle_orientation,
            face_count,
            isolate_rows=row_isolated,
        )
    else:
        left_covered = cp.empty(0, dtype=cp.int8)
        right_covered = cp.empty(0, dtype=cp.int8)

    return OverlayFaceTable(
        runtime_selection=half_edge_graph.runtime_selection,
        _face_count=face_count,
        device_state=OverlayFaceDeviceState(
            face_offsets=face_offsets,
            face_edge_ids=face_edge_ids,
            edge_face_ids=edge_face_ids,
            bounded_mask=bounded_mask,
            signed_area=signed_area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            left_covered=left_covered,
            right_covered=right_covered,
            cycle_orientation=cycle_orientation,
        ),
    )
