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
from vibespatial.cuda.cccl_primitives import sort_pairs
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.spatial.segment_primitives import SegmentIntersectionResult

from ._host_boundary import overlay_device_to_host
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


_COOPERATIVE_FACE_COORDINATES_PER_POLYGON = 1024


def _collapsed_triangle_face_mask_gpu(
    half_edge_graph: HalfEdgeGraph,
    d_face_offsets,
    d_face_edge_ids,
    d_bounded_mask,
):
    """Identify arrangement triangles thinner than the fp64 construction envelope."""
    from .graph import _fp64_radix_keys

    device = half_edge_graph.device_state
    face_count = max(int(d_face_offsets.size) - 1, 0)
    if device is None or face_count == 0:
        return cp.empty(0, dtype=cp.bool_)
    d_lengths = d_face_offsets[1:] - d_face_offsets[:-1]
    d_starts = d_face_offsets[:-1]
    last_edge = max(int(d_face_edge_ids.size) - 1, 0)
    d_e0 = d_face_edge_ids[cp.minimum(d_starts, last_edge)]
    d_e1 = d_face_edge_ids[cp.minimum(d_starts + 1, last_edge)]
    d_e2 = d_face_edge_ids[cp.minimum(d_starts + 2, last_edge)]
    d_x = cp.asarray(device.src_x, dtype=cp.float64)
    d_y = cp.asarray(device.src_y, dtype=cp.float64)
    d_x0, d_x1, d_x2 = d_x[d_e0], d_x[d_e1], d_x[d_e2]
    d_y0, d_y1, d_y2 = d_y[d_e0], d_y[d_e1], d_y[d_e2]

    def _within_envelope(left, right):
        left_keys = _fp64_radix_keys(left)
        right_keys = _fp64_radix_keys(right)
        return cp.maximum(left_keys, right_keys) - cp.minimum(left_keys, right_keys) <= cp.uint64(4)

    d_near_pair = (
        (_within_envelope(d_x0, d_x1) & _within_envelope(d_y0, d_y1))
        | (_within_envelope(d_x1, d_x2) & _within_envelope(d_y1, d_y2))
        | (_within_envelope(d_x2, d_x0) & _within_envelope(d_y2, d_y0))
    )
    return (d_bounded_mask != 0) & (d_lengths == 3) & d_near_pair


def _logical_coverage_kernel_name(
    family: str,
    *,
    logical_row_count: int,
    coordinate_count: int,
    physical_polygon_count: int,
) -> str:
    prefix = f"label_face_coverage_{family}_logical_rows"
    if coordinate_count > (
        _COOPERATIVE_FACE_COORDINATES_PER_POLYGON * max(physical_polygon_count, 1)
    ):
        return f"{prefix}_block"
    if logical_row_count > 8:
        return f"{prefix}_warp"
    return prefix


def _physical_coverage_kernel_name(
    family: str,
    *,
    coordinate_count: int,
    physical_polygon_count: int,
    physical_geometry_count: int,
    has_bounds: bool,
) -> str:
    prefix = f"label_face_coverage_{family}"
    if coordinate_count > (
        _COOPERATIVE_FACE_COORDINATES_PER_POLYGON * max(physical_polygon_count, 1)
    ):
        return f"{prefix}_block"
    if family == "polygon" and has_bounds and physical_geometry_count > 8:
        return f"{prefix}_warp"
    return prefix


def _coverage_launch_config(runtime, kernel, kernel_name: str, face_count: int):
    if kernel_name.endswith("_block"):
        return (max(face_count, 1), 1, 1), (256, 1, 1)
    work_items = face_count * 32 if kernel_name.endswith("_warp") else face_count
    return runtime.launch_config(kernel, work_items)


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
    right_geometry_broadcast: bool = False,
    refine_label_x: cp.ndarray | None = None,
    refine_label_y: cp.ndarray | None = None,
    refine_face_mask: cp.ndarray | None = None,
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

    refine_inputs = (refine_label_x, refine_label_y, refine_face_mask)
    if any(value is not None for value in refine_inputs) and not all(
        value is not None for value in refine_inputs
    ):
        raise ValueError("face coverage refinement requires coordinates and a mask")
    if refine_face_mask is not None:
        refine_label_x = cp.asarray(refine_label_x, dtype=cp.float64)
        refine_label_y = cp.asarray(refine_label_y, dtype=cp.float64)
        refine_face_mask = cp.asarray(refine_face_mask, dtype=cp.bool_)
        if any(
            int(values.size) != face_count
            for values in (refine_label_x, refine_label_y, refine_face_mask)
        ):
            raise ValueError("face coverage refinement arrays must match face_count")

    if face_count == 0:
        return left_covered, right_covered

    grid, block = runtime.launch_config(
        kernels["label_face_coverage_polygon"],
        face_count,
    )

    for side_name, side_input, out_covered, geometry_source_rows, geometry_broadcast in [
        ("left", left, left_covered, left_geometry_source_rows, False),
        (
            "right",
            right,
            right_covered,
            right_geometry_source_rows,
            right_geometry_broadcast,
        ),
    ]:
        side_face_source_rows = None if geometry_broadcast else face_source_rows
        with hotpath_stage(f"overlay.faces.coverage.{side_name}.prepare", category="setup"):
            device_state = side_input._ensure_device_state(
                preserve_indexed_view=True,
            )

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
            launch_mpoly = (
                has_mpoly
                and mp_count > 0
                and mp_buf is not None
                and mp_buf.part_offsets is not None
            )
            poly_coordinate_count = int(poly_buf.x.size) if launch_poly else 0
            mp_coordinate_count = int(mp_buf.x.size) if launch_mpoly else 0
            mp_polygon_count = (
                max(int(mp_buf.part_offsets.size) - 1, 0) if launch_mpoly else 0
            )
            use_poly_same_row = (
                side_face_source_rows is not None
                and geometry_source_rows is None
                and launch_poly
                and not getattr(side_input, "is_indexed_view", False)
                and poly_count == side_input.row_count
            )
            use_mpoly_same_row = (
                side_face_source_rows is not None
                and geometry_source_rows is None
                and launch_mpoly
                and not getattr(side_input, "is_indexed_view", False)
                and mp_count == side_input.row_count
            )
            use_poly_logical_rows = (
                side_face_source_rows is not None
                and launch_poly
                and getattr(side_input, "is_indexed_view", False)
            )
            use_mpoly_logical_rows = (
                side_face_source_rows is not None
                and launch_mpoly
                and getattr(side_input, "is_indexed_view", False)
            )
            need_poly_source_rows = (
                launch_poly and not use_poly_same_row and not use_poly_logical_rows
            )
            need_mpoly_source_rows = (
                launch_mpoly and not use_mpoly_same_row and not use_mpoly_logical_rows
            )

            d_poly_source_rows = None
            d_mp_source_rows = None
            d_poly_logical_family_rows = None
            d_mpoly_logical_family_rows = None
            d_logical_source_rows = None
            d_polygon_bounds = None
            d_polygon_ring_bounds = None
            d_mpoly_polygon_bounds = None
            if side_face_source_rows is not None and (
                need_poly_source_rows
                or need_mpoly_source_rows
                or use_poly_logical_rows
                or use_mpoly_logical_rows
            ):
                if geometry_source_rows is not None:
                    d_logical_source_rows = cp.asarray(
                        geometry_source_rows,
                        dtype=cp.int32,
                    )
                    logical_row_count = d_logical_source_rows.shape[0]
                    if logical_row_count != side_input.row_count:
                        raise ValueError(
                            f"{side_name}_geometry_source_rows must match row_count "
                            f"({side_input.row_count}), got {logical_row_count}"
                        )
                else:
                    d_logical_source_rows = cp.arange(
                        side_input.row_count,
                        dtype=cp.int32,
                    )
                d_tags = cp.asarray(device_state.tags)
                d_validity = cp.asarray(device_state.validity)
                d_family_rows = cp.asarray(device_state.family_row_offsets)
                if use_poly_logical_rows:
                    d_poly_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
                    d_poly_logical_family_rows = cp.where(
                        d_poly_mask,
                        d_family_rows,
                        cp.int32(-1),
                    ).astype(cp.int32, copy=False)
                if use_mpoly_logical_rows:
                    d_mpoly_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
                    d_mpoly_logical_family_rows = cp.where(
                        d_mpoly_mask,
                        d_family_rows,
                        cp.int32(-1),
                    ).astype(cp.int32, copy=False)
                if need_poly_source_rows:
                    d_poly_source_rows = cp.full(poly_count, -1, dtype=cp.int32)
                    d_poly_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.POLYGON])
                    d_poly_slots = d_family_rows[d_poly_mask].astype(cp.int32, copy=False)
                    d_poly_rows = d_logical_source_rows[d_poly_mask].astype(
                        cp.int32,
                        copy=False,
                    )
                    d_poly_source_rows[d_poly_slots] = d_poly_rows
                if need_mpoly_source_rows:
                    d_mp_source_rows = cp.full(mp_count, -1, dtype=cp.int32)
                    d_mp_mask = d_validity & (d_tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
                    d_mp_slots = d_family_rows[d_mp_mask].astype(cp.int32, copy=False)
                    d_mp_rows = d_logical_source_rows[d_mp_mask].astype(
                        cp.int32,
                        copy=False,
                    )
                    d_mp_source_rows[d_mp_slots] = d_mp_rows
                if use_poly_logical_rows or use_mpoly_logical_rows:
                    logical_order = cp.arange(
                        side_input.row_count,
                        dtype=cp.int32,
                    )
                    logical_sort = sort_pairs(
                        d_logical_source_rows,
                        logical_order,
                        synchronize=False,
                    )
                    d_logical_source_rows = logical_sort.keys.astype(
                        cp.int32,
                        copy=False,
                    )
                    d_logical_order = logical_sort.values.astype(
                        cp.int32,
                        copy=False,
                    )
                    if d_poly_logical_family_rows is not None:
                        d_poly_logical_family_rows = d_poly_logical_family_rows[
                            d_logical_order
                        ]
                    if d_mpoly_logical_family_rows is not None:
                        d_mpoly_logical_family_rows = d_mpoly_logical_family_rows[
                            d_logical_order
                        ]
            if launch_poly and not use_poly_same_row and poly_count > 1:
                d_polygon_bounds = cp.empty(poly_count * 4, dtype=cp.float64)
                bounds_grid, bounds_block = runtime.launch_config(
                    kernels["compute_polygon_bounds"],
                    poly_count,
                )
                bounds_params = (
                    (
                        ptr(poly_buf.x),
                        ptr(poly_buf.y),
                        ptr(poly_buf.geometry_offsets),
                        ptr(poly_buf.ring_offsets),
                        poly_count,
                        ptr(d_polygon_bounds),
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                        KERNEL_PARAM_PTR,
                    ),
                )
                with hotpath_stage(
                    f"overlay.faces.coverage.{side_name}.polygon_bounds",
                    category="setup",
                ):
                    runtime.launch(
                        kernels["compute_polygon_bounds"],
                        grid=bounds_grid,
                        block=bounds_block,
                        params=bounds_params,
                    )
                    _sync_hotpath(runtime)
            use_physical_polygon_block = (
                launch_poly
                and not use_poly_same_row
                and not use_poly_logical_rows
                and poly_coordinate_count
                > (_COOPERATIVE_FACE_COORDINATES_PER_POLYGON * max(poly_count, 1))
            )
            if use_physical_polygon_block:
                ring_count = max(int(poly_buf.ring_offsets.size) - 1, 0)
                d_polygon_ring_bounds = cp.empty(ring_count * 4, dtype=cp.float64)
                bounds_grid, bounds_block = runtime.launch_config(
                    kernels["compute_polygon_ring_bounds"],
                    ring_count,
                )
                with hotpath_stage(
                    f"overlay.faces.coverage.{side_name}.polygon_ring_bounds",
                    category="setup",
                ):
                    runtime.launch(
                        kernels["compute_polygon_ring_bounds"],
                        grid=bounds_grid,
                        block=bounds_block,
                        params=(
                            (
                                ptr(poly_buf.x),
                                ptr(poly_buf.y),
                                ptr(poly_buf.ring_offsets),
                                ring_count,
                                ptr(d_polygon_ring_bounds),
                            ),
                            (
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_PTR,
                            ),
                        ),
                    )
                    _sync_hotpath(runtime)
            if use_mpoly_same_row:
                polygon_count = int(mp_buf.part_offsets.size) - 1
                if polygon_count > 0:
                    d_mpoly_polygon_bounds = cp.empty(polygon_count * 4, dtype=cp.float64)
                    bounds_grid, bounds_block = runtime.launch_config(
                        kernels["compute_multipolygon_polygon_bounds"],
                        polygon_count,
                    )
                    bounds_params = (
                        (
                            ptr(mp_buf.x),
                            ptr(mp_buf.y),
                            ptr(mp_buf.part_offsets),
                            ptr(mp_buf.ring_offsets),
                            polygon_count,
                            ptr(d_mpoly_polygon_bounds),
                        ),
                        (
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                        ),
                    )
                    with hotpath_stage(
                        f"overlay.faces.coverage.{side_name}.multipolygon_bounds",
                        category="setup",
                    ):
                        runtime.launch(
                            kernels["compute_multipolygon_polygon_bounds"],
                            grid=bounds_grid,
                            block=bounds_block,
                            params=bounds_params,
                        )
                        _sync_hotpath(runtime)
            _sync_hotpath(runtime)

        face_rows_ptr = (
            0 if side_face_source_rows is None else ptr(side_face_source_rows)
        )
        poly_rows_ptr = 0 if d_poly_source_rows is None else ptr(d_poly_source_rows)
        mp_rows_ptr = 0 if d_mp_source_rows is None else ptr(d_mp_source_rows)
        poly_logical_rows_ptr = (
            0 if d_poly_logical_family_rows is None else ptr(d_poly_logical_family_rows)
        )
        mpoly_logical_rows_ptr = (
            0 if d_mpoly_logical_family_rows is None else ptr(d_mpoly_logical_family_rows)
        )
        logical_source_rows_ptr = 0 if d_logical_source_rows is None else ptr(d_logical_source_rows)
        polygon_bounds_ptr = 0 if d_polygon_bounds is None else ptr(d_polygon_bounds)
        polygon_ring_bounds_ptr = (
            0 if d_polygon_ring_bounds is None else ptr(d_polygon_ring_bounds)
        )
        coverage_launches = []

        if launch_poly and launch_mpoly:
            # Both families present — launch on separate CUDA streams so
            # the kernels can overlap.  They write to non-overlapping (or
            # idempotent) positions in out_covered.
            s_poly = runtime.create_stream()
            s_mpoly = runtime.create_stream()
            try:
                if use_poly_logical_rows:
                    poly_params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(poly_buf.x),
                            ptr(poly_buf.y),
                            ptr(poly_buf.geometry_offsets),
                            ptr(poly_buf.ring_offsets),
                            polygon_bounds_ptr,
                            poly_logical_rows_ptr,
                            logical_source_rows_ptr,
                            side_input.row_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    poly_kernel_name = _logical_coverage_kernel_name(
                        "polygon",
                        logical_row_count=side_input.row_count,
                        coordinate_count=poly_coordinate_count,
                        physical_polygon_count=poly_count,
                    )
                else:
                    poly_params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(poly_buf.x),
                            ptr(poly_buf.y),
                            ptr(poly_buf.geometry_offsets),
                            ptr(poly_buf.ring_offsets),
                            polygon_bounds_ptr,
                            poly_rows_ptr,
                            poly_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    poly_kernel_name = _physical_coverage_kernel_name(
                        "polygon",
                        coordinate_count=poly_coordinate_count,
                        physical_polygon_count=poly_count,
                        physical_geometry_count=poly_count,
                        has_bounds=d_polygon_bounds is not None,
                    )
                    if poly_kernel_name.endswith("_block"):
                        poly_params = (
                            poly_params[0][:8]
                            + (polygon_ring_bounds_ptr,)
                            + poly_params[0][8:],
                            poly_params[1][:8]
                            + (KERNEL_PARAM_PTR,)
                            + poly_params[1][8:],
                        )
                poly_grid, poly_block = _coverage_launch_config(
                    runtime,
                    kernels[poly_kernel_name],
                    poly_kernel_name,
                    face_count,
                )
                with hotpath_stage(
                    f"overlay.faces.coverage.{side_name}.mixed_family_overlap",
                    category="refine",
                ):
                    runtime.launch(
                        kernels[poly_kernel_name],
                        grid=poly_grid,
                        block=poly_block,
                        params=poly_params,
                        stream=s_poly,
                    )
                    coverage_launches.append(
                        (poly_kernel_name, poly_grid, poly_block, poly_params)
                    )
                    if use_mpoly_logical_rows:
                        mp_params = (
                            (
                                ptr(label_x),
                                ptr(label_y),
                                face_rows_ptr,
                                ptr(mp_buf.x),
                                ptr(mp_buf.y),
                                ptr(mp_buf.geometry_offsets),
                                ptr(mp_buf.part_offsets),
                                ptr(mp_buf.ring_offsets),
                                mpoly_logical_rows_ptr,
                                logical_source_rows_ptr,
                                side_input.row_count,
                                ptr(out_covered),
                                face_count,
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
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_I32,
                            ),
                        )
                        mp_kernel_name = _logical_coverage_kernel_name(
                            "multipolygon",
                            logical_row_count=side_input.row_count,
                            coordinate_count=mp_coordinate_count,
                            physical_polygon_count=mp_polygon_count,
                        )
                    else:
                        mp_params = (
                            (
                                ptr(label_x),
                                ptr(label_y),
                                face_rows_ptr,
                                ptr(mp_buf.x),
                                ptr(mp_buf.y),
                                ptr(mp_buf.geometry_offsets),
                                ptr(mp_buf.part_offsets),
                                ptr(mp_buf.ring_offsets),
                                mp_rows_ptr,
                                mp_count,
                                ptr(out_covered),
                                face_count,
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
                                KERNEL_PARAM_PTR,
                                KERNEL_PARAM_I32,
                            ),
                        )
                        mp_kernel_name = _physical_coverage_kernel_name(
                            "multipolygon",
                            coordinate_count=mp_coordinate_count,
                            physical_polygon_count=mp_polygon_count,
                            physical_geometry_count=mp_count,
                            has_bounds=False,
                        )
                    mp_grid, mp_block = _coverage_launch_config(
                        runtime,
                        kernels[mp_kernel_name],
                        mp_kernel_name,
                        face_count,
                    )
                    runtime.launch(
                        kernels[mp_kernel_name],
                        grid=mp_grid,
                        block=mp_block,
                        params=mp_params,
                        stream=s_mpoly,
                    )
                    coverage_launches.append((mp_kernel_name, mp_grid, mp_block, mp_params))
                    s_poly.synchronize()
                    s_mpoly.synchronize()
            finally:
                runtime.destroy_stream(s_poly)
                runtime.destroy_stream(s_mpoly)
        else:
            # Single family — launch on the default (null) stream.
            if launch_poly:
                if use_poly_same_row:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(poly_buf.x),
                            ptr(poly_buf.y),
                            ptr(poly_buf.geometry_offsets),
                            ptr(poly_buf.ring_offsets),
                            poly_count,
                            ptr(out_covered),
                            face_count,
                        ),
                        (
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    stage_name = f"overlay.faces.coverage.{side_name}.polygon_same_row"
                    kernel_name = "label_face_coverage_polygon_same_row"
                elif use_poly_logical_rows:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(poly_buf.x),
                            ptr(poly_buf.y),
                            ptr(poly_buf.geometry_offsets),
                            ptr(poly_buf.ring_offsets),
                            polygon_bounds_ptr,
                            poly_logical_rows_ptr,
                            logical_source_rows_ptr,
                            side_input.row_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    kernel_name = _logical_coverage_kernel_name(
                        "polygon",
                        logical_row_count=side_input.row_count,
                        coordinate_count=poly_coordinate_count,
                        physical_polygon_count=poly_count,
                    )
                    stage_name = (
                        f"overlay.faces.coverage.{side_name}."
                        f"{kernel_name.removeprefix('label_face_coverage_')}"
                    )
                else:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(poly_buf.x),
                            ptr(poly_buf.y),
                            ptr(poly_buf.geometry_offsets),
                            ptr(poly_buf.ring_offsets),
                            polygon_bounds_ptr,
                            poly_rows_ptr,
                            poly_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    kernel_name = _physical_coverage_kernel_name(
                        "polygon",
                        coordinate_count=poly_coordinate_count,
                        physical_polygon_count=poly_count,
                        physical_geometry_count=poly_count,
                        has_bounds=d_polygon_bounds is not None,
                    )
                    if kernel_name.endswith("_block"):
                        params = (
                            params[0][:8]
                            + (polygon_ring_bounds_ptr,)
                            + params[0][8:],
                            params[1][:8]
                            + (KERNEL_PARAM_PTR,)
                            + params[1][8:],
                        )
                    stage_name = (
                        f"overlay.faces.coverage.{side_name}."
                        f"{kernel_name.removeprefix('label_face_coverage_')}"
                    )
                with hotpath_stage(stage_name, category="refine"):
                    poly_grid, poly_block = _coverage_launch_config(
                        runtime,
                        kernels[kernel_name],
                        kernel_name,
                        face_count,
                    )
                    runtime.launch(
                        kernels[kernel_name], grid=poly_grid, block=poly_block, params=params
                    )
                    coverage_launches.append((kernel_name, poly_grid, poly_block, params))
                    _sync_hotpath(runtime)
            if launch_mpoly:
                if use_mpoly_same_row:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(mp_buf.x),
                            ptr(mp_buf.y),
                            ptr(mp_buf.geometry_offsets),
                            ptr(mp_buf.part_offsets),
                            ptr(mp_buf.ring_offsets),
                            ptr(d_mpoly_polygon_bounds),
                            mp_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    stage_name = f"overlay.faces.coverage.{side_name}.multipolygon_same_row"
                    kernel_name = "label_face_coverage_multipolygon_same_row"
                elif use_mpoly_logical_rows:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(mp_buf.x),
                            ptr(mp_buf.y),
                            ptr(mp_buf.geometry_offsets),
                            ptr(mp_buf.part_offsets),
                            ptr(mp_buf.ring_offsets),
                            mpoly_logical_rows_ptr,
                            logical_source_rows_ptr,
                            side_input.row_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_I32,
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    kernel_name = _logical_coverage_kernel_name(
                        "multipolygon",
                        logical_row_count=side_input.row_count,
                        coordinate_count=mp_coordinate_count,
                        physical_polygon_count=mp_polygon_count,
                    )
                    stage_name = (
                        f"overlay.faces.coverage.{side_name}."
                        f"{kernel_name.removeprefix('label_face_coverage_')}"
                    )
                else:
                    params = (
                        (
                            ptr(label_x),
                            ptr(label_y),
                            face_rows_ptr,
                            ptr(mp_buf.x),
                            ptr(mp_buf.y),
                            ptr(mp_buf.geometry_offsets),
                            ptr(mp_buf.part_offsets),
                            ptr(mp_buf.ring_offsets),
                            mp_rows_ptr,
                            mp_count,
                            ptr(out_covered),
                            face_count,
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
                            KERNEL_PARAM_PTR,
                            KERNEL_PARAM_I32,
                        ),
                    )
                    kernel_name = _physical_coverage_kernel_name(
                        "multipolygon",
                        coordinate_count=mp_coordinate_count,
                        physical_polygon_count=mp_polygon_count,
                        physical_geometry_count=mp_count,
                        has_bounds=False,
                    )
                    stage_name = (
                        f"overlay.faces.coverage.{side_name}."
                        f"{kernel_name.removeprefix('label_face_coverage_')}"
                    )
                with hotpath_stage(stage_name, category="refine"):
                    mp_grid, mp_block = _coverage_launch_config(
                        runtime,
                        kernels[kernel_name],
                        kernel_name,
                        face_count,
                    )
                    runtime.launch(
                        kernels[kernel_name], grid=mp_grid, block=mp_block, params=params
                    )
                    coverage_launches.append((kernel_name, mp_grid, mp_block, params))
                    _sync_hotpath(runtime)
        if refine_face_mask is not None and coverage_launches:
            # Inactive and already-covered lanes are seeded to one. Every
            # coverage kernel treats that value as complete, so the refinement
            # remains face-capacity shaped without compacting a sparse rowset.
            refined_covered = cp.where(
                refine_face_mask,
                out_covered,
                cp.int8(1),
            ).astype(cp.int8, copy=False)
            with hotpath_stage(
                f"overlay.faces.coverage.{side_name}.collapsed_triangle_refine",
                category="refine",
            ):
                for kernel_name, launch_grid, launch_block, primary_params in (
                    coverage_launches
                ):
                    primary_values, primary_types = primary_params
                    refine_values = (
                        ptr(refine_label_x),
                        ptr(refine_label_y),
                        *primary_values[2:-2],
                        ptr(refined_covered),
                        primary_values[-1],
                    )
                    runtime.launch(
                        kernels[kernel_name],
                        grid=launch_grid,
                        block=launch_block,
                        params=(refine_values, primary_types),
                    )
                _sync_hotpath(runtime)
            out_covered[...] = cp.where(
                refine_face_mask,
                refined_covered,
                out_covered,
            )
    return left_covered, right_covered


def _overlay_face_selection_mask_gpu(
    faces: OverlayFaceTable,
    *,
    operation: str,
) -> cp.ndarray:
    """Return the face-capacity operation mask without compacting topology."""
    ds = faces.device_state
    d_bounded = cp.asarray(ds.bounded_mask)
    d_left = cp.asarray(ds.left_covered)
    d_right = cp.asarray(ds.right_covered)

    if operation == "intersection":
        d_mask = (d_left != 0) & (d_right != 0)
    elif operation == "union":
        d_mask = (d_left != 0) | (d_right != 0)
    elif operation == "difference":
        d_mask = (d_left != 0) & (d_right == 0)
    elif operation == "right_difference":
        d_mask = (d_left == 0) & (d_right != 0)
    elif operation == "symmetric_difference":
        d_mask = d_left != d_right
    elif operation == "identity":
        d_mask = d_left != 0
    elif operation == "polygonize":
        # Topology-repair consumers select every positively oriented bounded
        # face. Coverage labels are intentionally irrelevant for invalid source
        # rings; split/half-edge topology is the compatibility contract.
        d_mask = (d_bounded != 0) & (cp.asarray(ds.signed_area) > 0)
    else:
        raise ValueError(f"unsupported overlay operation: {operation}")

    return d_mask


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

    return cp.flatnonzero(_overlay_face_selection_mask_gpu(faces, operation=operation)).astype(
        cp.int32
    )


def _selected_face_indices_to_host(d_selected_face_indices: cp.ndarray) -> np.ndarray:
    """Export selected faces only at the admitted CPU face-assembly boundary."""
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    item_count = int(getattr(d_selected_face_indices, "size", len(d_selected_face_indices)))
    itemsize = int(getattr(getattr(d_selected_face_indices, "dtype", None), "itemsize", 0))
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
    """Assemble selected overlay faces through the admitted device path.

    Accepts device-resident (CuPy) face indices from
    ``_select_overlay_face_indices_gpu``. The physical shape is dynamic-output
    polygon assembly: selected face rows, half-edge topology, rings, output
    rows, and output bytes stay device-shaped. The separate host bridge remains
    available to explicit debug/export callers, never as an execution fallback.
    """
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
        (
            d_face_offsets,
            d_face_edge_ids,
            d_bounded_mask,
            d_signed_area,
            d_centroid_x,
            d_centroid_y,
            d_label_x,
            d_label_y,
            face_count,
        ) = _gpu_face_walk(half_edge_graph)

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
                right_geometry_broadcast=right_geometry_broadcast,
                # An exact arrangement can contain a triangular face whose two
                # computed vertices differ by only a few fp64 representable
                # steps. Its inward edge probe may have no stable representable
                # interior point. The centered probe is evaluated in the same
                # prepared coverage pass only for that structural ULP envelope.
                refine_label_x=d_centroid_x,
                refine_label_y=d_centroid_y,
                refine_face_mask=_collapsed_triangle_face_mask_gpu(
                    half_edge_graph,
                    d_face_offsets,
                    d_face_edge_ids,
                    d_bounded_mask,
                ),
            )
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
        bounded_mask_values.append(1 if signed_area > 0.0 else 0)

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
            d_bounded_mask = runtime.from_host(bounded_mask)
            # Host copies deferred -- lazily materialised by property accessor.
            left_covered = None
            right_covered = None
            _gpu_coverage = True
        else:
            left_covered, right_covered = _label_face_coverage(left, right, label_x, label_y)
            left_covered = left_covered.astype(np.int8, copy=False)
            right_covered = right_covered.astype(np.int8, copy=False)

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
    return GeometryFamily.POLYGON in geom.families or GeometryFamily.MULTIPOLYGON in geom.families
