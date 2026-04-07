"""Face labeling, selection, and overlay face construction.

Extracted from ``overlay/gpu.py`` — Stage 5 of the overlay module split.

Public API
----------
- ``build_gpu_overlay_faces`` — main face construction pipeline (calls
  graph + face walk + labeling)
- ``_gpu_label_face_coverage`` — GPU face coverage labeling
- ``_select_overlay_face_indices_gpu`` — select face indices by overlay
  operation type
- ``_assemble_faces_from_device_indices`` — assemble face data from
  selected indices
"""
from __future__ import annotations

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.spatial.segment_primitives import SegmentIntersectionResult

from .types import (
    AtomicEdgeTable,
    HalfEdgeGraph,
    OverlayFaceDeviceState,
    OverlayFaceTable,
    SplitEventTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None


def _sync_hotpath(runtime) -> None:
    if hotpath_trace_enabled():
        runtime.synchronize()


def _gpu_label_face_coverage(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    label_x: cp.ndarray,
    label_y: cp.ndarray,
    face_count: int,
    *,
    face_source_rows: cp.ndarray | None = None,
    left_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
    right_geometry_source_rows: cp.ndarray | np.ndarray | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU face labeling: test face sample points against input geometries.

    Returns (left_covered, right_covered) as CuPy int8 arrays.
    """
    from vibespatial.overlay.gpu import _overlay_face_label_kernels

    runtime = get_cuda_runtime()
    kernels = _overlay_face_label_kernels()
    ptr = runtime.pointer

    left_covered = cp.zeros(face_count, dtype=cp.int8)
    right_covered = cp.zeros(face_count, dtype=cp.int8)

    if face_count == 0:
        return left_covered, right_covered

    grid, block = runtime.launch_config(
        kernels["label_face_coverage_polygon"], face_count,
    )

    for side_name, side_input, out_covered, geometry_source_rows in [
        ("left", left, left_covered, left_geometry_source_rows),
        ("right", right, right_covered, right_geometry_source_rows),
    ]:
        with hotpath_stage(f"overlay.faces.coverage.{side_name}.prepare", category="setup"):
            device_state = side_input._ensure_device_state()

            has_poly = GeometryFamily.POLYGON in device_state.families
            has_mpoly = GeometryFamily.MULTIPOLYGON in device_state.families

            poly_count = 0
            mp_count = 0
            poly_buf = None
            mp_buf = None

            if has_poly:
                poly_buf = device_state.families[GeometryFamily.POLYGON]
                poly_count = side_input.families[GeometryFamily.POLYGON].row_count
            if has_mpoly:
                mp_buf = device_state.families[GeometryFamily.MULTIPOLYGON]
                mp_count = side_input.families[GeometryFamily.MULTIPOLYGON].row_count

            launch_poly = has_poly and poly_count > 0
            launch_mpoly = (has_mpoly and mp_count > 0
                            and mp_buf is not None and mp_buf.part_offsets is not None)

            d_poly_source_rows = None
            d_mp_source_rows = None
            if face_source_rows is not None:
                logical_rows = None
                if geometry_source_rows is not None:
                    logical_rows = cp.asarray(geometry_source_rows, dtype=cp.int32)
                    logical_row_count = logical_rows.shape[0]
                    if logical_row_count != side_input.row_count:
                        raise ValueError(
                            f"{side_name}_geometry_source_rows must match row_count "
                            f"({side_input.row_count}), got {logical_row_count}"
                        )
                d_tags = cp.asarray(device_state.tags)
                d_validity = cp.asarray(device_state.validity)
                d_family_rows = cp.asarray(device_state.family_row_offsets)
                if launch_poly:
                    d_poly_source_rows = cp.full(poly_count, -1, dtype=cp.int32)
                    d_poly_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
                    d_poly_slots = d_family_rows[d_poly_mask].astype(cp.int32, copy=False)
                    if logical_rows is None:
                        d_poly_rows = cp.flatnonzero(d_poly_mask).astype(cp.int32, copy=False)
                    else:
                        d_poly_rows = logical_rows[d_poly_mask].astype(cp.int32, copy=False)
                    d_poly_source_rows[d_poly_slots] = d_poly_rows
                if launch_mpoly:
                    d_mp_source_rows = cp.full(mp_count, -1, dtype=cp.int32)
                    d_mp_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
                    d_mp_slots = d_family_rows[d_mp_mask].astype(cp.int32, copy=False)
                    if logical_rows is None:
                        d_mp_rows = cp.flatnonzero(d_mp_mask).astype(cp.int32, copy=False)
                    else:
                        d_mp_rows = logical_rows[d_mp_mask].astype(cp.int32, copy=False)
                    d_mp_source_rows[d_mp_slots] = d_mp_rows
            _sync_hotpath(runtime)

        face_rows_ptr = 0 if face_source_rows is None else ptr(face_source_rows)
        poly_rows_ptr = 0 if d_poly_source_rows is None else ptr(d_poly_source_rows)
        mp_rows_ptr = 0 if d_mp_source_rows is None else ptr(d_mp_source_rows)

        if launch_poly and launch_mpoly:
            # Both families present — launch on separate CUDA streams so
            # the kernels can overlap.  They write to non-overlapping (or
            # idempotent) positions in out_covered.
            s_poly = runtime.create_stream()
            s_mpoly = runtime.create_stream()
            try:
                poly_params = (
                    (ptr(label_x), ptr(label_y), face_rows_ptr,
                     ptr(poly_buf.x), ptr(poly_buf.y),
                     ptr(poly_buf.geometry_offsets), ptr(poly_buf.ring_offsets),
                     poly_rows_ptr, poly_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                with hotpath_stage(
                    f"overlay.faces.coverage.{side_name}.mixed_family_overlap",
                    category="refine",
                ):
                    runtime.launch(kernels["label_face_coverage_polygon"],
                                   grid=grid, block=block, params=poly_params,
                                   stream=s_poly)
                    mp_params = (
                        (ptr(label_x), ptr(label_y), face_rows_ptr,
                         ptr(mp_buf.x), ptr(mp_buf.y),
                         ptr(mp_buf.geometry_offsets), ptr(mp_buf.part_offsets),
                         ptr(mp_buf.ring_offsets), mp_rows_ptr,
                         mp_count, ptr(out_covered), face_count),
                        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR,
                         KERNEL_PARAM_PTR,
                         KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                    )
                    runtime.launch(kernels["label_face_coverage_multipolygon"],
                                   grid=grid, block=block, params=mp_params,
                                   stream=s_mpoly)
                    s_poly.synchronize()
                    s_mpoly.synchronize()
            finally:
                runtime.destroy_stream(s_poly)
                runtime.destroy_stream(s_mpoly)
        else:
            # Single family — launch on the default (null) stream.
            if launch_poly:
                params = (
                    (ptr(label_x), ptr(label_y), face_rows_ptr,
                     ptr(poly_buf.x), ptr(poly_buf.y),
                     ptr(poly_buf.geometry_offsets), ptr(poly_buf.ring_offsets),
                     poly_rows_ptr, poly_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                with hotpath_stage(f"overlay.faces.coverage.{side_name}.polygon", category="refine"):
                    runtime.launch(kernels["label_face_coverage_polygon"],
                                   grid=grid, block=block, params=params)
                    _sync_hotpath(runtime)
            if launch_mpoly:
                params = (
                    (ptr(label_x), ptr(label_y), face_rows_ptr,
                     ptr(mp_buf.x), ptr(mp_buf.y),
                     ptr(mp_buf.geometry_offsets), ptr(mp_buf.part_offsets),
                     ptr(mp_buf.ring_offsets), mp_rows_ptr,
                     mp_count, ptr(out_covered), face_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR,
                     KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
                )
                with hotpath_stage(f"overlay.faces.coverage.{side_name}.multipolygon", category="refine"):
                    runtime.launch(kernels["label_face_coverage_multipolygon"],
                                   grid=grid, block=block, params=params)
                    _sync_hotpath(runtime)
    return left_covered, right_covered


def _select_overlay_face_indices_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
) -> cp.ndarray:
    """Device-resident face selection -- no _ensure_host() D->H triggers.

    Reads bounded_mask, left_covered, and right_covered directly from the
    OverlayFaceTable's device_state, applies the overlay operation mask
    entirely on device, and returns the selected face indices as a
    device-resident CuPy int32 array.
    """
    ds = faces.device_state
    d_bounded = cp.asarray(ds.bounded_mask)
    d_left = cp.asarray(ds.left_covered)
    d_right = cp.asarray(ds.right_covered)

    if operation == "intersection":
        d_mask = (d_bounded != 0) & (d_left != 0) & (d_right != 0)
    elif operation == "union":
        d_mask = (d_bounded != 0) & ((d_left != 0) | (d_right != 0))
    elif operation == "difference":
        d_mask = (d_bounded != 0) & (d_left != 0) & (d_right == 0)
    elif operation == "symmetric_difference":
        d_mask = (d_bounded != 0) & (d_left != d_right)
    elif operation == "identity":
        d_mask = (d_bounded != 0) & (d_left != 0)
    else:
        raise ValueError(f"unsupported overlay operation: {operation}")

    return cp.flatnonzero(d_mask).astype(cp.int32)


def _assemble_faces_from_device_indices(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    d_selected_face_indices: cp.ndarray,
) -> OwnedGeometryArray:
    """Try CPU face assembly first, fall back to GPU.

    Accepts device-resident (CuPy) face indices from
    ``_select_overlay_face_indices_gpu`` and handles the D->H conversion
    only for the CPU assembly path.  GPU assembly receives the CuPy array
    directly (zero-copy).

    The CPU-first ordering is intentional: CPU face boundary walking is
    faster for most cases.  GPU assembly handles the "spans multiple
    source rows" edge case (ADR-0016 Stage 8).
    """
    from vibespatial.overlay.assemble import (
        _build_polygon_output_from_faces_gpu,
        _empty_polygon_output,
    )
    from vibespatial.overlay.host_fallback import _build_polygon_output_from_faces

    if d_selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)
    try:
        selected_face_indices = cp.asnumpy(d_selected_face_indices)  # hygiene:ok(CPU assembly path requires host indices)
        return _build_polygon_output_from_faces(half_edge_graph, faces, selected_face_indices)
    except RuntimeError:
        result = _build_polygon_output_from_faces_gpu(
            half_edge_graph, faces, d_selected_face_indices,
        )
        if result is None:
            raise
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
) -> OverlayFaceTable:
    from vibespatial.overlay.gpu import (
        build_gpu_atomic_edges,
        build_gpu_split_events,
    )
    from vibespatial.overlay.graph import _gpu_face_walk, build_gpu_half_edge_graph
    from vibespatial.overlay.host_fallback import (
        _face_sample_point,
        _label_face_coverage,
        _signed_area_and_centroid,
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

    edge_count = half_edge_graph.edge_count
    if edge_count == 0:
        empty_device_i32 = runtime.allocate((1,), np.int32)
        empty_device_i32_flat = runtime.allocate((0,), np.int32)
        empty_device_i8 = runtime.allocate((0,), np.int8)
        empty_device_f64 = runtime.allocate((0,), np.float64)
        # Device-primary empty face table -- host arrays are None and
        # will be lazily materialized via _ensure_host if accessed.
        return OverlayFaceTable(
            runtime_selection=half_edge_graph.runtime_selection,
            _face_count=0,
            device_state=OverlayFaceDeviceState(
                face_offsets=empty_device_i32,
                face_edge_ids=empty_device_i32_flat,
                bounded_mask=empty_device_i8,
                signed_area=empty_device_f64,
                centroid_x=empty_device_f64,
                centroid_y=empty_device_f64,
                left_covered=empty_device_i8,
                right_covered=empty_device_i8,
            ),
        )

    # GPU face walk path: pointer jumping + shoelace aggregation
    if cp is not None and half_edge_graph.device_state is not None:
        (d_face_offsets, d_face_edge_ids, d_bounded_mask, d_signed_area,
         d_centroid_x, d_centroid_y, d_label_x, d_label_y, face_count) = _gpu_face_walk(half_edge_graph)

        if face_count > 0:
            d_face_source_rows = None
            if row_isolated and half_edge_graph.device_state.row_indices is not None:
                d_face_source_rows = cp.asarray(half_edge_graph.device_state.row_indices)[
                    d_face_edge_ids[d_face_offsets[:-1]]
                ].astype(cp.int32, copy=False)
            # GPU face labeling: test sample points against input geometries
            d_left_covered, d_right_covered = _gpu_label_face_coverage(
                left,
                right,
                d_label_x,
                d_label_y,
                face_count,
                face_source_rows=d_face_source_rows,
                left_geometry_source_rows=left_geometry_source_rows,
                right_geometry_source_rows=right_geometry_source_rows,
            )
            # Mask out unbounded faces (keep on device -- ADR-0005)
            with hotpath_stage("overlay.faces.mask_unbounded", category="filter"):
                d_left_covered = cp.where(d_bounded_mask != 0, d_left_covered, 0).astype(cp.int8)
                d_right_covered = cp.where(d_bounded_mask != 0, d_right_covered, 0).astype(cp.int8)
                _sync_hotpath(runtime)
        else:
            # _gpu_face_walk already returned device arrays for the zero case
            d_left_covered = cp.empty(0, dtype=cp.int8)
            d_right_covered = cp.empty(0, dtype=cp.int8)

        # Device-primary: host arrays are None, lazily materialized on demand
        return OverlayFaceTable(
            runtime_selection=half_edge_graph.runtime_selection,
            _face_count=face_count,
            device_state=OverlayFaceDeviceState(
                face_offsets=d_face_offsets,
                face_edge_ids=d_face_edge_ids,
                bounded_mask=d_bounded_mask,
                signed_area=d_signed_area,
                centroid_x=d_centroid_x,
                centroid_y=d_centroid_y,
                left_covered=d_left_covered,
                right_covered=d_right_covered,
            ),
        )

    # CPU fallback path
    visited = np.zeros(edge_count, dtype=bool)
    face_edge_groups: list[np.ndarray] = []
    signed_area_values: list[float] = []
    centroid_x_values: list[float] = []
    centroid_y_values: list[float] = []
    label_x_values: list[float] = []
    label_y_values: list[float] = []
    bounded_mask_values: list[int] = []

    for edge_id in range(edge_count):
        if visited[edge_id]:
            continue
        cycle_edges: list[int] = []
        current = edge_id
        while not visited[current]:
            visited[current] = True
            cycle_edges.append(current)
            current = int(half_edge_graph.next_edge_ids[current])
        if current != edge_id or len(cycle_edges) < 3:
            continue
        points = np.column_stack(
            (
                half_edge_graph.src_x[np.asarray(cycle_edges, dtype=np.int32)],
                half_edge_graph.src_y[np.asarray(cycle_edges, dtype=np.int32)],
            )
        )
        signed_area, centroid_x, centroid_y = _signed_area_and_centroid(points)
        face_edge_groups.append(np.asarray(cycle_edges, dtype=np.int32))
        signed_area_values.append(signed_area)
        centroid_x_values.append(centroid_x)
        centroid_y_values.append(centroid_y)
        sample_x, sample_y = _face_sample_point(points)
        label_x_values.append(sample_x)
        label_y_values.append(sample_y)
        bounded_mask_values.append(1 if signed_area > SPATIAL_EPSILON else 0)

    # Track whether coverage was computed on device (avoids D->H->D roundtrip).
    _gpu_coverage = False
    if not face_edge_groups:
        face_offsets = np.asarray([0], dtype=np.int32)
        face_edge_ids = np.asarray([], dtype=np.int32)
        bounded_mask = np.asarray([], dtype=np.int8)
        signed_area = np.asarray([], dtype=np.float64)
        centroid_x = np.asarray([], dtype=np.float64)
        centroid_y = np.asarray([], dtype=np.float64)
        left_covered = np.asarray([], dtype=np.int8)
        right_covered = np.asarray([], dtype=np.int8)
    else:
        face_lengths = np.asarray([group.size for group in face_edge_groups], dtype=np.int32)
        face_offsets = np.empty((face_lengths.size + 1,), dtype=np.int32)
        face_offsets[0] = 0
        face_offsets[1:] = np.cumsum(face_lengths, dtype=np.int32)
        face_edge_ids = np.concatenate(face_edge_groups).astype(np.int32, copy=False)
        bounded_mask = np.asarray(bounded_mask_values, dtype=np.int8)
        signed_area = np.asarray(signed_area_values, dtype=np.float64)
        centroid_x = np.asarray(centroid_x_values, dtype=np.float64)
        centroid_y = np.asarray(centroid_y_values, dtype=np.float64)
        label_x = np.asarray(label_x_values, dtype=np.float64)
        label_y = np.asarray(label_y_values, dtype=np.float64)
        # Prefer GPU labeling even when the face walk was done on CPU.
        # This avoids the Shapely roundtrip (to_shapely -> union_all -> covers).
        cpu_face_count_for_label = label_x.size
        if (
            cp is not None
            and cpu_face_count_for_label > 0
            and (_has_polygonal_families(left) or _has_polygonal_families(right))
        ):
            d_label_x = runtime.from_host(label_x)
            d_label_y = runtime.from_host(label_y)
            d_lc, d_rc = _gpu_label_face_coverage(
                left,
                right,
                d_label_x,
                d_label_y,
                cpu_face_count_for_label,
                left_geometry_source_rows=left_geometry_source_rows,
                right_geometry_source_rows=right_geometry_source_rows,
            )
            runtime.synchronize()
            # Apply bounded_mask on device to avoid D->H->D roundtrip.
            d_bounded_mask = runtime.from_host(bounded_mask)
            d_mask = d_bounded_mask != 0
            d_lc = cp.where(d_mask, d_lc, cp.int8(0)).astype(cp.int8, copy=False)
            d_rc = cp.where(d_mask, d_rc, cp.int8(0)).astype(cp.int8, copy=False)
            # Host copies deferred -- lazily materialised by property accessor.
            left_covered = None
            right_covered = None
            _gpu_coverage = True
        else:
            left_covered, right_covered = _label_face_coverage(left, right, label_x, label_y)
            left_covered = np.where(bounded_mask != 0, left_covered, 0).astype(np.int8, copy=False)
            right_covered = np.where(bounded_mask != 0, right_covered, 0).astype(np.int8, copy=False)

    cpu_face_count = max(0, int(face_offsets.size) - 1)
    # Build device state; reuse arrays already on device when GPU coverage ran.
    d_face_offsets = runtime.from_host(face_offsets)
    d_face_edge_ids = runtime.from_host(face_edge_ids)
    if not _gpu_coverage:
        d_bounded_mask = runtime.from_host(bounded_mask)
    d_signed_area = runtime.from_host(signed_area)
    d_centroid_x = runtime.from_host(centroid_x)
    d_centroid_y = runtime.from_host(centroid_y)
    d_left_covered = d_lc if _gpu_coverage else runtime.from_host(left_covered)
    d_right_covered = d_rc if _gpu_coverage else runtime.from_host(right_covered)
    return OverlayFaceTable(
        runtime_selection=half_edge_graph.runtime_selection,
        _face_count=cpu_face_count,
        _face_offsets=face_offsets,
        _face_edge_ids=face_edge_ids,
        _bounded_mask=bounded_mask,
        _signed_area=signed_area,
        _centroid_x=centroid_x,
        _centroid_y=centroid_y,
        _left_covered=left_covered,
        _right_covered=right_covered,
        device_state=OverlayFaceDeviceState(
            face_offsets=d_face_offsets,
            face_edge_ids=d_face_edge_ids,
            bounded_mask=d_bounded_mask,
            signed_area=d_signed_area,
            centroid_x=d_centroid_x,
            centroid_y=d_centroid_y,
            left_covered=d_left_covered,
            right_covered=d_right_covered,
        ),
    )


def _has_polygonal_families(geom: OwnedGeometryArray) -> bool:
    """Return True if the geometry array has POLYGON or MULTIPOLYGON families."""
    return (
        GeometryFamily.POLYGON in geom.families
        or GeometryFamily.MULTIPOLYGON in geom.families
    )
