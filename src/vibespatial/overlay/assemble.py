"""Polygon output assembly from overlay face data.

This module contains the GPU and utility functions for assembling polygon
output from half-edge graph faces produced by the overlay pipeline.
Extracted from ``overlay/gpu.py`` to reduce module size; see ADR-0016.
"""

from __future__ import annotations

import logging

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import (
    exclusive_sum,
    segmented_reduce_sum,
    sort_pairs,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.residency import Residency

from .types import (
    HalfEdgeGraph,
    OverlayFaceTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)


def _has_polygonal_families(geom: OwnedGeometryArray) -> bool:
    """Return True if the geometry array has POLYGON or MULTIPOLYGON families."""
    return (
        GeometryFamily.POLYGON in geom.families
        or GeometryFamily.MULTIPOLYGON in geom.families
    )


def _empty_polygon_output(runtime_selection: RuntimeSelection) -> OwnedGeometryArray:
    residency = Residency.DEVICE if cp is not None else Residency.HOST
    empty_validity = np.asarray([], dtype=bool)
    empty_tags = np.asarray([], dtype=np.int8)
    empty_offsets = np.asarray([], dtype=np.int32)
    device_state = None
    if residency is Residency.DEVICE:
        try:
            rt = get_cuda_runtime()
            device_state = OwnedGeometryDeviceState(
                validity=rt.from_host(empty_validity),
                tags=rt.from_host(empty_tags),
                family_row_offsets=rt.from_host(empty_offsets),
                families={},
            )
        except Exception:
            residency = Residency.HOST
            device_state = None
    return OwnedGeometryArray(
        validity=empty_validity,
        tags=empty_tags,
        family_row_offsets=empty_offsets,
        families={},
        residency=residency,
        runtime_history=[runtime_selection],
        device_state=device_state,
    )


def _build_device_backed_fixed_polygon_output(
    device_x: DeviceArray,
    device_y: DeviceArray,
    *,
    row_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    runtime = get_cuda_runtime()
    geometry_offsets = np.arange(row_count + 1, dtype=np.int32)
    ring_offsets = np.arange(0, (row_count + 1) * 5, 5, dtype=np.int32)
    empty_mask = np.zeros(row_count, dtype=bool)
    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(row_count, dtype=np.int32)
    polygon_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POLYGON,
        schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
        row_count=row_count,
        x=np.empty(0, dtype=np.float64),
        y=np.empty(0, dtype=np.float64),
        geometry_offsets=geometry_offsets,
        empty_mask=empty_mask,
        ring_offsets=ring_offsets,
        bounds=None,
        host_materialized=False,
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
        runtime_history=[runtime_selection],
        device_state=OwnedGeometryDeviceState(
            validity=runtime.from_host(validity),
            tags=runtime.from_host(tags),
            family_row_offsets=runtime.from_host(family_row_offsets),
            families={
                GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                    family=GeometryFamily.POLYGON,
                    x=device_x,
                    y=device_y,
                    geometry_offsets=runtime.from_host(geometry_offsets),
                    empty_mask=runtime.from_host(empty_mask),
                    ring_offsets=runtime.from_host(ring_offsets),
                    bounds=None,
                )
            },
        ),
    )


def _axis_aligned_box_bounds(values: OwnedGeometryArray) -> np.ndarray | None:
    if set(values.families) != {GeometryFamily.POLYGON}:
        return None
    polygon_buffer = values.families[GeometryFamily.POLYGON]
    row_count = polygon_buffer.row_count
    if row_count == 0 or row_count != values.row_count:
        return None
    if polygon_buffer.ring_offsets is None:
        return None
    if not np.array_equal(polygon_buffer.geometry_offsets, np.arange(row_count + 1, dtype=np.int32)):
        return None
    if not np.array_equal(polygon_buffer.ring_offsets, np.arange(0, (row_count + 1) * 5, 5, dtype=np.int32)):
        return None
    if np.any(polygon_buffer.empty_mask):
        return None

    x = polygon_buffer.x.reshape(row_count, 5)
    y = polygon_buffer.y.reshape(row_count, 5)
    if not (np.allclose(x[:, 0], x[:, 4]) and np.allclose(y[:, 0], y[:, 4])):
        return None

    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    axis_aligned = ((np.abs(dx) < 1e-12) ^ (np.abs(dy) < 1e-12))
    if not np.all(axis_aligned):
        return None
    return np.column_stack(
        (
            np.min(x[:, :4], axis=1),
            np.min(y[:, :4], axis=1),
            np.max(x[:, :4], axis=1),
            np.max(y[:, :4], axis=1),
        )
    ).astype(np.float64, copy=False)


def _build_polygon_output_from_faces_gpu(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    selected_face_indices: np.ndarray | cp.ndarray,
) -> OwnedGeometryArray | None:
    """GPU face-to-polygon assembly (Phase 11: GPU boundary cycle detection).

    Full GPU pipeline:
      Steps 1-2: Edge-to-face mapping and face selection via CuPy scatter.
      Step 3: Boundary edge identification via NVRTC kernel.
      Step 4: Boundary next-edge computation via NVRTC kernel.
      Step 5: Cycle detection via GPU pointer jumping + list ranking;
              per-cycle area/centroid via segmented reduction.
      Steps 6-7: Coordinate offset computation and ring scatter via GPU.
      Steps 7b-8: Hole ring extraction and merge on device.
      Step 8b: GPU nesting depth for boundary rings (even depth = exterior).
      Step 9: Hole-to-exterior assignment via GPU PIP kernel.
      Step 9b: GPU sibling hole nesting depth (even = valid hole, odd = skip).
      Step 10: GPU output assembly with device-side sorting, grouping,
              and host-side row_polygons construction; D->H transfer at the
              ADR-0005 materialization boundary.

    Returns None if GPU is unavailable (caller falls back to CPU path).
    """
    if cp is None or half_edge_graph.device_state is None or faces.device_state is None:
        return None
    if selected_face_indices.size == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # Lazy import: kernel compile functions stay in gpu.py to avoid
    # circular imports (they depend on gpu_kernels module-level state).
    from vibespatial.overlay.gpu import (
        _overlay_face_assembly_kernels,
        _overlay_face_walk_kernels,
    )

    runtime = get_cuda_runtime()
    kernels = _overlay_face_assembly_kernels()
    walk_kernels = _overlay_face_walk_kernels()
    kernels.update(walk_kernels)
    ptr = runtime.pointer
    edge_count = half_edge_graph.edge_count
    face_count = faces.face_count
    block = (256, 1, 1)
    edge_grid = (max(1, (edge_count + 255) // 256), 1, 1)

    device = half_edge_graph.device_state
    face_device = faces.device_state

    # --- Step 1: Map edges to faces (Tier 2: CuPy vectorised scatter) ---
    # Build edge_face_ids on device: for each edge, which face does it belong to?
    d_edge_face_ids = cp.full(edge_count, -1, dtype=cp.int32)
    d_face_offsets = cp.asarray(face_device.face_offsets)
    d_face_edge_ids = cp.asarray(face_device.face_edge_ids)
    total_face_edges = int(d_face_edge_ids.size)
    if total_face_edges > 0:
        # For each slot in face_edge_ids, find which face it belongs to
        slot_ids = cp.arange(total_face_edges, dtype=cp.int32)
        slot_face = cp.searchsorted(d_face_offsets[1:], slot_ids, side='right').astype(cp.int32)
        d_edge_face_ids[d_face_edge_ids] = slot_face

    # --- Step 2: Build face selection mask on device ---
    d_face_selected = cp.zeros(face_count, dtype=cp.int8)
    d_face_selected[cp.asarray(selected_face_indices)] = 1

    # --- Step 3: Identify boundary edges via GPU kernel ---
    d_is_boundary = cp.empty(edge_count, dtype=cp.int8)
    runtime.launch(
        kernels["compute_boundary_edges"],
        grid=edge_grid, block=block,
        params=(
            (ptr(d_edge_face_ids), ptr(d_face_selected), ptr(device.next_edge_ids),
             ptr(d_is_boundary), edge_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    boundary_count = int(cp.sum(d_is_boundary != 0))
    if boundary_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # --- Step 4: Compute boundary next pointers via GPU kernel ---
    d_boundary_next = cp.full(edge_count, -1, dtype=cp.int32)
    max_steps = edge_count
    runtime.launch(
        kernels["compute_boundary_next"],
        grid=edge_grid, block=block,
        params=(
            (ptr(d_edge_face_ids), ptr(d_face_selected), ptr(device.next_edge_ids),
             ptr(d_is_boundary), ptr(d_boundary_next), edge_count, max_steps),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    # --- Step 5: Detect boundary cycles via GPU pointer jumping ---
    boundary_edge_indices = cp.flatnonzero(d_is_boundary != 0).astype(cp.int32, copy=False)
    boundary_count = int(boundary_edge_indices.size)
    # Phase 25 memory: d_is_boundary, d_edge_face_ids, d_face_selected,
    # and the Step 1 face_offsets/edge_ids copies are dead.
    del d_is_boundary, d_edge_face_ids, d_face_selected
    del d_face_offsets, d_face_edge_ids

    # Build compact boundary-local next array
    edge_to_compact = cp.full(edge_count, -1, dtype=cp.int32)
    edge_to_compact[boundary_edge_indices] = cp.arange(boundary_count, dtype=cp.int32)
    compact_next = edge_to_compact[d_boundary_next[boundary_edge_indices]]
    # Phase 25 memory: edge_to_compact is dead after compact_next is built.
    del edge_to_compact

    # Pointer jumping to find cycle labels (minimum compact index in cycle)
    cycle_label = cp.arange(boundary_count, dtype=cp.int32)
    jump_b = compact_next.copy()
    max_iter_b = max(1, int(np.ceil(np.log2(max(1, boundary_count)))))
    for _ in range(max_iter_b):
        cycle_label = cp.minimum(cycle_label, cycle_label[jump_b])
        jump_b = jump_b[jump_b]
    # Phase 25 memory: jump_b is dead after pointer jumping.
    del jump_b

    # List ranking: compute position within each cycle using NVRTC kernel
    d_rank_b = cp.empty(boundary_count, dtype=cp.int32)
    d_compact_next_i64 = compact_next.astype(cp.int64)
    boundary_grid = (max(1, (boundary_count + 255) // 256), 1, 1)
    rank_params = (
        (ptr(cycle_label), ptr(d_compact_next_i64),
         ptr(d_rank_b), boundary_count, boundary_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    runtime.launch(
        kernels["list_rank_within_cycle"],
        grid=boundary_grid, block=block, params=rank_params,
    )

    # Sort by (cycle_label, rank) to get cycle-ordered boundary edges
    packed_b = cycle_label.astype(cp.int64) * int(boundary_count) + d_rank_b.astype(cp.int64)
    # Phase 25 memory: d_rank_b and d_compact_next_i64 are dead.
    del d_rank_b, d_compact_next_i64
    b_sort = sort_pairs(packed_b, cp.arange(boundary_count, dtype=cp.int32), synchronize=False)
    del packed_b
    sorted_compact_ids = b_sort.values
    sorted_labels = cycle_label[sorted_compact_ids]
    # Phase 25 memory: cycle_label consumed; compact_next dead.
    del cycle_label, compact_next

    # Find unique cycles and their segment boundaries
    b_start_mask = cp.empty(boundary_count, dtype=cp.bool_)
    b_start_mask[0] = True
    if boundary_count > 1:
        b_start_mask[1:] = sorted_labels[1:] != sorted_labels[:-1]
    cycle_starts = cp.flatnonzero(b_start_mask).astype(cp.int32, copy=False)
    cycle_ends = cp.concatenate((cycle_starts[1:], cp.asarray([boundary_count], dtype=cp.int32)))
    cycle_lengths = cycle_ends - cycle_starts
    del b_start_mask, sorted_labels

    # Filter cycles with >= 3 edges
    valid_cycle_mask = cycle_lengths >= 3
    valid_cycle_indices = cp.flatnonzero(valid_cycle_mask).astype(cp.int32, copy=False)
    ring_count = int(valid_cycle_indices.size)
    del valid_cycle_mask

    if ring_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    valid_cycle_starts = cycle_starts[valid_cycle_indices]
    valid_cycle_ends = cycle_ends[valid_cycle_indices]
    valid_cycle_lengths = cycle_lengths[valid_cycle_indices]
    del cycle_starts, cycle_ends, cycle_lengths, valid_cycle_indices

    # Map sorted compact ids back to full edge ids for the valid cycles
    sorted_full_edge_ids = boundary_edge_indices[sorted_compact_ids]
    del sorted_compact_ids

    # Compute per-boundary-edge shoelace contributions on device
    d_src_x_b = cp.asarray(device.src_x)
    d_src_y_b = cp.asarray(device.src_y)
    b_x0 = d_src_x_b[sorted_full_edge_ids]
    b_y0 = d_src_y_b[sorted_full_edge_ids]
    b_next_edges = d_boundary_next[sorted_full_edge_ids]
    b_x1 = d_src_x_b[b_next_edges]
    b_y1 = d_src_y_b[b_next_edges]
    del b_next_edges
    b_cross = b_x0 * b_y1 - b_x1 * b_y0
    b_cx_contrib = (b_x0 + b_x1) * b_cross
    b_cy_contrib = (b_y0 + b_y1) * b_cross
    # Phase 25 memory: boundary coordinate arrays consumed by cross products.
    del b_x0, b_y0, b_x1, b_y1

    # Segmented reduce for per-cycle area and centroid
    cross_sums_b = segmented_reduce_sum(b_cross, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    cx_sums_b = segmented_reduce_sum(b_cx_contrib, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    cy_sums_b = segmented_reduce_sum(b_cy_contrib, valid_cycle_starts, valid_cycle_ends, num_segments=ring_count).values
    del b_cross, b_cx_contrib, b_cy_contrib

    d_ring_area = cross_sums_b * 0.5
    safe_twice_b = cp.where(cross_sums_b == 0.0, 1.0, cross_sums_b)
    factor_b = 1.0 / (3.0 * safe_twice_b)
    d_ring_centroid_x = cx_sums_b * factor_b
    d_ring_centroid_y = cy_sums_b * factor_b
    del cross_sums_b, cx_sums_b, cy_sums_b, safe_twice_b, factor_b

    # Ring edge starts (first full edge id of each valid cycle) and counts
    d_ring_edge_starts = sorted_full_edge_ids[valid_cycle_starts]
    d_ring_edge_counts = valid_cycle_lengths
    del sorted_full_edge_ids, valid_cycle_starts, valid_cycle_ends, valid_cycle_lengths

    # Source row per cycle: take the row_index of the first edge (device-resident
    # until Step 10 materialization boundary per ADR-0005).
    # Read from device_state directly to avoid D->H->D round-trip.
    d_row_indices = cp.asarray(device.row_indices)
    d_cycle_source_rows = d_row_indices[d_ring_edge_starts].astype(cp.int32)

    # --- Step 6: Compute ring coordinate offsets (Tier 3a: exclusive_scan) ---
    # Each ring needs edge_count + 1 coordinates (for closure)
    ring_coord_counts = d_ring_edge_counts + 1
    d_ring_coord_offsets = exclusive_sum(ring_coord_counts.astype(cp.int32, copy=False))
    # Single scalar D->H read: total_coords = last_offset + last_count.
    total_coords = int(cp.asnumpy(d_ring_coord_offsets[-1:] + ring_coord_counts[-1:])[0])  # zcopy:ok(allocation-fence: need total_coords to size d_out_x/d_out_y output buffers)

    # --- Step 7: Scatter ring coordinates via GPU kernel ---
    # Zero-fill to prevent denormalized garbage in unwritten positions if the
    # scatter kernel skips any coordinate slot due to an out-of-bounds
    # boundary_next index.  cp.empty() would recycle a pool block containing
    # stale int32 metadata, which reinterprets as denormalized float64 values
    # (e.g. 4e-316) that crash GEOS with TopologyException.
    d_out_x = cp.zeros(total_coords, dtype=cp.float64)
    d_out_y = cp.zeros(total_coords, dtype=cp.float64)
    ring_grid = (max(1, (ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["scatter_ring_coordinates"],
        grid=ring_grid, block=block,
        params=(
            (ptr(device.src_x), ptr(device.src_y),
             ptr(d_ring_edge_starts), ptr(d_ring_coord_offsets),
             ptr(d_ring_edge_counts), ptr(d_boundary_next),
             ptr(d_out_x), ptr(d_out_y), ring_count,
             edge_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
             KERNEL_PARAM_I32),
        ),
    )

    # Phase 25 memory: d_ring_edge_starts, d_ring_coord_offsets, and
    # d_ring_edge_counts are still needed for Step 8 merge.

    # --- Step 7b: Extract hole rings from unselected bounded faces ---
    # Per ADR-0016, holes are unselected bounded faces whose ring coordinates
    # form interior rings of the output polygons.
    # GPU filter: hole faces are unselected bounded faces (Tier 2: CuPy).
    d_bounded_mask_dev = cp.asarray(face_device.bounded_mask)
    d_face_selected_bool = cp.zeros(face_count, dtype=cp.bool_)
    d_face_selected_bool[cp.asarray(selected_face_indices)] = True
    d_hole_mask = (d_bounded_mask_dev != 0) & (~d_face_selected_bool)
    del d_bounded_mask_dev, d_face_selected_bool
    d_hole_fi = cp.flatnonzero(d_hole_mask).astype(cp.int32)
    del d_hole_mask

    # Extract hole ring coordinates using scatter_ring_coordinates kernel.
    # Hole faces use next_edge_ids (not boundary_next) for edge traversal.
    d_face_offsets_dev = cp.asarray(face_device.face_offsets)
    d_face_edge_ids_dev = cp.asarray(face_device.face_edge_ids)
    d_next_i32 = cp.asarray(device.next_edge_ids).astype(cp.int32)

    if d_hole_fi.size > 0:
        d_hole_starts_in_face = d_face_offsets_dev[d_hole_fi]
        d_hole_ends_in_face = d_face_offsets_dev[d_hole_fi + 1]
        d_hole_lengths = d_hole_ends_in_face - d_hole_starts_in_face

        # Filter to holes with >= 3 edges
        d_valid_hole_mask = d_hole_lengths >= 3
        d_valid_hole_idx = cp.flatnonzero(d_valid_hole_mask).astype(cp.int32)
        n_valid_holes = int(d_valid_hole_idx.size)

        if n_valid_holes > 0:
            d_vh_starts = d_hole_starts_in_face[d_valid_hole_idx]
            d_vh_lengths = d_hole_lengths[d_valid_hole_idx]

            # First edge of each hole face (the canonical cycle start)
            d_hole_edge_starts = d_face_edge_ids_dev[d_vh_starts]

            # Scatter coordinates via GPU kernel
            d_hole_coord_counts = d_vh_lengths + 1  # +1 for ring closure
            d_hole_coord_offsets_partial = exclusive_sum(d_hole_coord_counts.astype(cp.int32))
            # exclusive_sum returns [0, c0, c0+c1, ...] of size n_valid_holes.
            # Append the total to form a proper (n+1) offset array so that
            # offsets[:-1] gives n start indices and offsets[1:] gives n ends.
            _last_offset = d_hole_coord_offsets_partial[-1:] + d_hole_coord_counts[-1:]
            d_hole_coord_offsets = cp.concatenate([d_hole_coord_offsets_partial, _last_offset])
            total_hole_coords = int(cp.asnumpy(_last_offset)[0])  # hygiene:ok(allocation-fence: need total_hole_coords to size d_hole_x/d_hole_y)

            # Zero-fill: same rationale as boundary ring output above.
            d_hole_x = cp.zeros(total_hole_coords, dtype=cp.float64)
            d_hole_y = cp.zeros(total_hole_coords, dtype=cp.float64)
            hole_grid = (max(1, (n_valid_holes + 255) // 256), 1, 1)
            runtime.launch(
                kernels["scatter_ring_coordinates"],
                grid=hole_grid, block=block,
                params=(
                    (ptr(device.src_x), ptr(device.src_y),
                     ptr(d_hole_edge_starts), ptr(d_hole_coord_offsets),
                     ptr(d_vh_lengths), ptr(d_next_i32),
                     ptr(d_hole_x), ptr(d_hole_y), n_valid_holes,
                     edge_count),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                     KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                     KERNEL_PARAM_I32),
                ),
            )

            # Compute per-hole-ring shoelace area/centroid on device
            # Phase 25 memory: removed unused hole_slot_ids searchsorted result.
            # Shoelace: cross product per edge (skip last coord = closure)
            h_x0 = d_hole_x[:-1] if total_hole_coords > 1 else d_hole_x
            h_y0 = d_hole_y[:-1] if total_hole_coords > 1 else d_hole_y
            h_x1 = d_hole_x[1:] if total_hole_coords > 1 else d_hole_x
            h_y1 = d_hole_y[1:] if total_hole_coords > 1 else d_hole_y

            # Per-ring starts/ends for segmented reduce (use coord offsets)
            h_cross = h_x0 * h_y1 - h_x1 * h_y0
            h_cx_c = (h_x0 + h_x1) * h_cross
            h_cy_c = (h_y0 + h_y1) * h_cross
            del h_x0, h_y0, h_x1, h_y1

            # Segmented reduce -- use coord-based segments (each ring's coords
            # are contiguous, last coord is closure vertex)
            hole_seg_starts = d_hole_coord_offsets[:-1]
            hole_seg_ends = d_hole_coord_offsets[1:] - 1  # exclude closure vertex from cross products
            h_area = segmented_reduce_sum(h_cross, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values * 0.5
            h_cx_sum = segmented_reduce_sum(h_cx_c, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values
            h_cy_sum = segmented_reduce_sum(h_cy_c, hole_seg_starts, hole_seg_ends, num_segments=n_valid_holes).values
            del h_cross, h_cx_c, h_cy_c, hole_seg_starts, hole_seg_ends
            h_safe_twice = cp.where(h_area == 0.0, 0.5, h_area * 2.0)
            h_factor = 1.0 / (3.0 * h_safe_twice)
            h_centroid_x = h_cx_sum * h_factor
            h_centroid_y = h_cy_sum * h_factor
            del h_cx_sum, h_cy_sum, h_safe_twice, h_factor

            # Filter degenerate holes on device (|area| < 1e-12)
            d_nondegenerate = cp.abs(h_area) >= 1e-12
            d_nondegen_idx = cp.flatnonzero(d_nondegenerate).astype(cp.int32)
            n_valid_nondegenerate = int(d_nondegen_idx.size)
            del d_nondegenerate

            if n_valid_nondegenerate > 0:
                # Keep only non-degenerate hole rings (device-resident)
                _d_hole_area = h_area[d_nondegen_idx]
                _d_hole_cx = h_centroid_x[d_nondegen_idx]
                _d_hole_cy = h_centroid_y[d_nondegen_idx]
                _d_hole_lengths = d_vh_lengths[d_nondegen_idx]
                _d_hole_starts = d_hole_coord_offsets[:-1][d_nondegen_idx]
                # These device arrays are passed directly to Step 8
                _hole_device_data = (_d_hole_area, _d_hole_cx, _d_hole_cy,
                                     _d_hole_lengths, _d_hole_starts,
                                     d_hole_x, d_hole_y, d_hole_coord_offsets,
                                     n_valid_nondegenerate)
            else:
                _hole_device_data = None
        else:
            _hole_device_data = None
    else:
        _hole_device_data = None

    # --- Step 8: Merge boundary + hole ring data on device ---
    boundary_ring_count = ring_count

    if _hole_device_data is not None:
        (_d_hole_area, _d_hole_cx, _d_hole_cy, _d_hole_lengths,
         _d_hole_starts, _d_hole_all_x, _d_hole_all_y,
         _d_hole_coord_offsets_full, n_holes) = _hole_device_data

        # Build compact hole coordinate buffer: gather non-degenerate hole
        # ring coords into a contiguous buffer. Use per-hole coord counts.
        d_hole_coord_counts = _d_hole_lengths + 1  # +1 for ring closure
        d_hole_offsets_compact = exclusive_sum(d_hole_coord_counts.astype(cp.int32))
        # exclusive_sum returns [0, c0, c0+c1, ...] of size n_holes; total
        # is last_offset + last_count (not just last_offset).
        total_hole_compact = int(cp.asnumpy(d_hole_offsets_compact[-1:])[0]) + int(cp.asnumpy(d_hole_coord_counts[-1:])[0])  # hygiene:ok(allocation-fence: need total_hole_compact to size compact gather buffers)

        # Gather coords from the original hole buffer using starts + lengths
        if total_hole_compact > 0:
            slot_ids_h = cp.arange(total_hole_compact, dtype=cp.int32)
            slot_ring_h = cp.searchsorted(d_hole_offsets_compact[1:], slot_ids_h, side='right').astype(cp.int32)
            slot_local_h = slot_ids_h - d_hole_offsets_compact[slot_ring_h]
            slot_src_h = _d_hole_starts[slot_ring_h] + slot_local_h
            d_hole_compact_x = _d_hole_all_x[slot_src_h]
            d_hole_compact_y = _d_hole_all_y[slot_src_h]
        else:
            d_hole_compact_x = cp.empty(0, dtype=cp.float64)
            d_hole_compact_y = cp.empty(0, dtype=cp.float64)

        d_hole_edge_counts = _d_hole_lengths

        # Merge: boundary rings [0..ring_count), hole rings [ring_count..total)
        # Hole faces are traversed counterclockwise (positive area) in the
        # overlay half-edge graph, but they represent interior rings that must
        # have negative (clockwise) area so assign_holes_to_exteriors treats
        # them as holes rather than self-assigned exteriors.
        d_all_area = cp.concatenate((d_ring_area, -cp.abs(_d_hole_area)))
        d_all_cx = cp.concatenate((d_ring_centroid_x, _d_hole_cx))
        d_all_cy = cp.concatenate((d_ring_centroid_y, _d_hole_cy))
        d_all_x = cp.concatenate((d_out_x, d_hole_compact_x))
        d_all_y = cp.concatenate((d_out_y, d_hole_compact_y))
        # Merged coordinate offsets: boundary offsets + compact hole offsets shifted.
        # Use total_coords (not boundary_total) because total_coords is the
        # actual byte count of boundary ring coordinates in d_out_x/d_out_y,
        # i.e. where the compact hole coords start in d_all_x/d_all_y.
        d_hole_offsets_shifted = d_hole_offsets_compact + total_coords
        d_all_coord_offsets = cp.concatenate((d_ring_coord_offsets, d_hole_offsets_shifted))
        d_all_edge_counts = cp.concatenate((d_ring_edge_counts, d_hole_edge_counts))
        # Phase 25 memory: pre-merge ring arrays consumed by concatenation.
        del d_ring_area, d_ring_centroid_x, d_ring_centroid_y
        del d_out_x, d_out_y, d_ring_coord_offsets, d_ring_edge_counts
        del d_hole_compact_x, d_hole_compact_y, d_hole_offsets_compact
        del _d_hole_area, _d_hole_cx, _d_hole_cy, d_hole_edge_counts
        del d_hole_offsets_shifted
    else:
        n_holes = 0
        d_all_area = d_ring_area
        d_all_cx = d_ring_centroid_x
        d_all_cy = d_ring_centroid_y
        d_all_x = d_out_x
        d_all_y = d_out_y
        d_all_coord_offsets = d_ring_coord_offsets
        d_all_edge_counts = d_ring_edge_counts

    # Phase 25 memory: boundary_next, face data copies, and hole-detection
    # arrays are dead after the merge.
    del d_boundary_next, d_face_offsets_dev, d_face_edge_ids_dev, d_next_i32
    del d_hole_fi, d_src_x_b, d_src_y_b
    total_ring_count = boundary_ring_count + n_holes

    # --- Source rows for ALL rings on device ---
    # Boundary rings: from cycle walk; holes: inherit from exterior later.
    d_all_source_rows = cp.full(total_ring_count, -1, dtype=cp.int32)
    d_all_source_rows[:boundary_ring_count] = d_cycle_source_rows

    # --- Step 8b: GPU nesting depth for boundary rings ---
    # Boundary rings with positive area might be nested inside other
    # positive-area boundary rings from the same source row. Count
    # containment depth: even → true exterior, odd → nested interior.
    d_is_boundary_flag = cp.zeros(total_ring_count, dtype=cp.bool_)
    d_is_boundary_flag[:boundary_ring_count] = True
    d_pos_area_boundary = d_is_boundary_flag & (d_all_area > 0.0)
    n_pos_boundary = int(cp.sum(d_pos_area_boundary))

    if n_pos_boundary > 1:
        # Launch nesting depth kernel only when there are multiple
        # positive-area boundary rings (single ring is trivially exterior)
        d_boundary_depth = cp.zeros(boundary_ring_count, dtype=cp.int32)
        boundary_depth_grid = (max(1, (boundary_ring_count + 255) // 256), 1, 1)
        runtime.launch(
            kernels["count_boundary_nesting_depth"],
            grid=boundary_depth_grid, block=block,
            params=(
                (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
                 ptr(d_all_source_rows), ptr(d_all_coord_offsets),
                 ptr(d_all_edge_counts), ptr(d_all_x), ptr(d_all_y),
                 ptr(d_boundary_depth), boundary_ring_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )
        # True exteriors: positive area + even nesting depth
        d_even_depth = (d_boundary_depth % 2) == 0
        d_exterior_mask = (d_all_area[:boundary_ring_count] > 0.0) & d_even_depth
        # Extend to full ring array (hole rings are never exteriors)
        d_exterior_mask_full = cp.zeros(total_ring_count, dtype=cp.bool_)
        d_exterior_mask_full[:boundary_ring_count] = d_exterior_mask
    else:
        # Single or zero positive-area boundary rings: no nesting possible
        d_exterior_mask_full = cp.zeros(total_ring_count, dtype=cp.bool_)
        d_exterior_mask_full[:boundary_ring_count] = (
            d_all_area[:boundary_ring_count] > 0.0
        )

    d_exterior_indices = cp.flatnonzero(d_exterior_mask_full).astype(cp.int32)
    exterior_count = int(d_exterior_indices.size)

    if exterior_count == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # --- Step 9: Assign holes to exteriors via GPU kernel ---
    d_exterior_id = cp.full(total_ring_count, -1, dtype=cp.int32)
    ring_grid_all = (max(1, (total_ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["assign_holes_to_exteriors"],
        grid=ring_grid_all, block=block,
        params=(
            (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
             ptr(d_all_coord_offsets), ptr(d_all_edge_counts),
             ptr(d_all_x), ptr(d_all_y),
             ptr(d_exterior_indices), exterior_count,
             ptr(d_exterior_id), total_ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # --- Step 9b: GPU sibling hole nesting depth ---
    # For each hole assigned to an exterior, count how many sibling holes
    # (same exterior, larger |area|) contain its centroid. Even local
    # depth -> valid direct hole; odd -> nested inside another hole (skip).
    d_sibling_depth = cp.zeros(total_ring_count, dtype=cp.int32)

    # Check if there are any holes assigned
    d_is_hole = (d_exterior_id >= 0) & (d_exterior_id != cp.arange(total_ring_count, dtype=cp.int32))
    n_assigned_holes = int(cp.sum(d_is_hole))

    if n_assigned_holes > 1:
        runtime.launch(
            kernels["count_sibling_hole_depth"],
            grid=ring_grid_all, block=block,
            params=(
                (ptr(d_all_cx), ptr(d_all_cy), ptr(d_all_area),
                 ptr(d_exterior_id), ptr(d_all_coord_offsets),
                 ptr(d_all_edge_counts), ptr(d_all_x), ptr(d_all_y),
                 ptr(d_sibling_depth), total_ring_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )

    # --- Step 10: Device-resident output assembly ---
    # Build GeoArrow-format polygon output entirely on device.
    # Valid holes: assigned to an exterior AND even sibling depth.
    d_valid_hole_mask = d_is_hole & ((d_sibling_depth % 2) == 0)

    # Also filter: only keep holes whose assigned exterior is a true exterior
    d_ext_is_valid = d_exterior_mask_full[d_exterior_id.clip(0, total_ring_count - 1)]
    d_valid_hole_mask = d_valid_hole_mask & d_ext_is_valid

    # Propagate source rows from exterior to holes on device
    d_hole_ext_ids = d_exterior_id.clip(0, total_ring_count - 1)
    d_all_source_rows = cp.where(
        d_valid_hole_mask,
        d_all_source_rows[d_hole_ext_ids],
        d_all_source_rows,
    )

    # Build the ordered ring list: for each exterior, gather its holes.
    # Create a sort key: (source_row, exterior_id, is_exterior_flag desc,
    # ring_id) so rings within a polygon group sort as
    # [exterior, hole_0, hole_1, ...].
    d_is_output_ring = d_exterior_mask_full | d_valid_hole_mask
    d_output_ring_ids = cp.flatnonzero(d_is_output_ring).astype(cp.int32)
    n_output_rings = int(d_output_ring_ids.size)

    if n_output_rings == 0:
        return _empty_polygon_output(faces.runtime_selection)

    # For each output ring, its exterior id (exteriors map to themselves)
    d_out_ext_id = cp.where(
        d_exterior_mask_full[d_output_ring_ids],
        d_output_ring_ids,
        d_exterior_id[d_output_ring_ids],
    )
    d_out_source_row = d_all_source_rows[d_output_ring_ids]
    d_out_is_ext = d_exterior_mask_full[d_output_ring_ids].astype(cp.int32)

    # Sort by (source_row, exterior_id, NOT is_exterior, ring_id) so
    # exteriors come first within each polygon group.
    # Use CuPy lexsort (Tier 2) for arbitrary value ranges.
    d_sort_order = cp.lexsort(cp.stack((
        d_output_ring_ids,           # last key = least significant
        1 - d_out_is_ext,            # exteriors (0) before holes (1)
        d_out_ext_id,                # group by exterior
        d_out_source_row,            # primary key: source row
    )))
    d_sorted_output_ids = d_output_ring_ids[d_sort_order]

    # Identify polygon boundaries: a new polygon starts at each exterior ring
    d_sorted_is_ext = d_exterior_mask_full[d_sorted_output_ids]
    d_sorted_source_row = d_all_source_rows[d_sorted_output_ids]

    # Each exterior starts a new polygon group
    # Count rings per polygon (for ring_offsets)
    d_poly_start_mask = d_sorted_is_ext
    d_poly_starts = cp.flatnonzero(d_poly_start_mask).astype(cp.int32)
    n_polygons = int(d_poly_starts.size)

    if n_polygons == 0:
        return _empty_polygon_output(faces.runtime_selection)

    d_poly_ends = cp.concatenate((d_poly_starts[1:], cp.asarray([n_output_rings], dtype=cp.int32)))
    d_rings_per_poly = d_poly_ends - d_poly_starts

    # Source row per polygon (from the exterior ring)
    d_poly_source_row = d_sorted_source_row[d_poly_starts]

    # Polygons are already sorted by source row (primary sort key of the
    # ring sort). Detect source-row group boundaries directly.
    d_row_change = cp.empty(n_polygons, dtype=cp.bool_)
    d_row_change[0] = True
    if n_polygons > 1:
        d_row_change[1:] = d_poly_source_row[1:] != d_poly_source_row[:-1]
    d_row_starts = cp.flatnonzero(d_row_change).astype(cp.int32)
    d_row_ends = cp.concatenate((d_row_starts[1:], cp.asarray([n_polygons], dtype=cp.int32)))
    d_polys_per_row = d_row_ends - d_row_starts
    n_output_rows = int(d_row_starts.size)

    # Transfer only small O(ring_count) structural metadata to host.
    # Coordinate arrays d_all_x/d_all_y stay on device (Phase 14 zero-copy).
    # ADR-0005 materialization boundary: structural offsets are O(ring_count)
    # integers; coordinate data stays device-resident.
    h_sorted_output_ids = cp.asnumpy(d_sorted_output_ids)  # hygiene:ok(ADR-0005 materialization boundary: O(ring_count) structural metadata)
    h_rings_per_poly = cp.asnumpy(d_rings_per_poly)  # hygiene:ok(ADR-0005 materialization boundary)
    h_polys_per_row = cp.asnumpy(d_polys_per_row)  # hygiene:ok(ADR-0005 materialization boundary)
    h_row_source_ids = cp.asnumpy(d_poly_source_row[d_row_starts])  # hygiene:ok(ADR-0005 materialization boundary)
    h_poly_starts = cp.asnumpy(d_poly_starts)  # hygiene:ok(ADR-0005 materialization boundary)
    h_all_coord_offsets = cp.asnumpy(d_all_coord_offsets)  # hygiene:ok(ADR-0005 materialization boundary)
    h_all_edge_counts = cp.asnumpy(d_all_edge_counts)  # hygiene:ok(ADR-0005 materialization boundary)

    return _build_device_resident_polygon_output(
        d_all_x=d_all_x,
        d_all_y=d_all_y,
        h_all_coord_offsets=h_all_coord_offsets,
        h_all_edge_counts=h_all_edge_counts,
        h_sorted_output_ids=h_sorted_output_ids,
        h_rings_per_poly=h_rings_per_poly,
        h_polys_per_row=h_polys_per_row,
        h_row_source_ids=h_row_source_ids,
        h_poly_starts=h_poly_starts,
        n_output_rows=n_output_rows,
        runtime_selection=faces.runtime_selection,
    )


def _build_device_resident_polygon_output(
    *,
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    h_all_coord_offsets: np.ndarray,
    h_all_edge_counts: np.ndarray,
    h_sorted_output_ids: np.ndarray,
    h_rings_per_poly: np.ndarray,
    h_polys_per_row: np.ndarray,
    h_row_source_ids: np.ndarray,
    h_poly_starts: np.ndarray,
    n_output_rows: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    """Build device-resident OwnedGeometryArray from GPU face assembly results.

    Accepts GPU-computed ring grouping (Phase 12 sibling hole nesting) and
    builds GeoArrow offset arrays on host via vectorised NumPy, then gathers
    coordinates on device via CuPy fancy indexing.

    Phase 14 (ADR-0005): eliminates dominant D->H coordinate transfer.
    Phase 26: Vectorised output assembly -- eliminates per-row and per-ring
    Python loops that dominated wall time for large polygon counts.
    """
    runtime = get_cuda_runtime()

    # --- Vectorised classification: polygon vs multipolygon per row ---
    # A row with exactly 1 polygon is a POLYGON; otherwise MULTIPOLYGON.
    is_polygon_row = h_polys_per_row == 1
    output_row_count = n_output_rows
    if output_row_count == 0:
        return _empty_polygon_output(runtime_selection)

    validity = np.ones(output_row_count, dtype=bool)
    tags = np.full(output_row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(output_row_count, -1, dtype=np.int32)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]

    # Total polygon count: sum of polys_per_row for polygon-type rows (== 1).
    # Total multipolygon count: number of multipolygon rows.
    n_total_polys = int(h_rings_per_poly.size)  # total output polygons

    # Expand polygon index -> its output rings (all sorted_output_ids entries
    # are already in [exterior, hole0, hole1, ...] order from GPU sort).
    # Flat ring order for ALL polygons (both POLYGON and MULTIPOLYGON families):
    # ring_indices[i] = the i-th ring in output order across all polygons.
    # Use vectorised expansion: repeat each polygon's ring range.
    ring_counts_per_poly = h_rings_per_poly.astype(np.int32)
    total_output_rings = int(ring_counts_per_poly.sum())
    if total_output_rings == 0:
        return _empty_polygon_output(runtime_selection)

    # Build per-polygon -> row mapping: polygon p belongs to the row
    # determined by cumulative polys_per_row.
    # poly_to_row[p] = which output row polygon p belongs to.
    poly_to_row = np.repeat(np.arange(output_row_count, dtype=np.int32), h_polys_per_row)

    # For each polygon, the rings in sorted_output_ids
    # are at positions [poly_starts[p], poly_starts[p]+rings_per_poly[p]).
    # Vectorised expansion: for each ring slot, compute which polygon it
    # belongs to and its offset within that polygon's ring list.
    ring_poly_ids = np.repeat(np.arange(n_total_polys, dtype=np.int32), ring_counts_per_poly)
    poly_ring_offsets_prefix = np.zeros(n_total_polys + 1, dtype=np.int32)
    np.cumsum(ring_counts_per_poly, out=poly_ring_offsets_prefix[1:])
    ring_local_offsets = np.arange(total_output_rings, dtype=np.int32) - poly_ring_offsets_prefix[ring_poly_ids]
    # Global ring indices in the all-rings coordinate arrays
    all_ring_order = h_sorted_output_ids[h_poly_starts[ring_poly_ids] + ring_local_offsets]

    # Coord counts per ring: edge_count + 1
    all_ring_coord_counts = h_all_edge_counts[all_ring_order].astype(np.int32) + 1

    # --- Separate POLYGON and MULTIPOLYGON families ---
    # Polygon family: rows with exactly 1 polygon
    poly_row_mask = is_polygon_row
    mpoly_row_mask = ~is_polygon_row

    # Map rows to polygon indices (within poly_to_row)
    # A polygon belongs to the polygon family if its row is a polygon row.
    poly_mask_per_polygon = poly_row_mask[poly_to_row]
    mpoly_mask_per_polygon = mpoly_row_mask[poly_to_row]

    # Ring-level masks: a ring belongs to the polygon family if its polygon does.
    ring_is_poly = poly_mask_per_polygon[ring_poly_ids]
    ring_is_mpoly = mpoly_mask_per_polygon[ring_poly_ids]

    # Assign family tags and family_row_offsets (vectorised, no Python loop)
    tags[poly_row_mask] = poly_tag
    tags[mpoly_row_mask] = mpoly_tag
    polygon_count = int(poly_row_mask.sum())
    multipolygon_count = int(mpoly_row_mask.sum())
    # Family-local sequential index: polygon rows get 0..polygon_count-1,
    # multipolygon rows get 0..multipolygon_count-1.
    if polygon_count > 0:
        family_row_offsets[poly_row_mask] = np.arange(polygon_count, dtype=np.int32)
    if multipolygon_count > 0:
        family_row_offsets[mpoly_row_mask] = np.arange(multipolygon_count, dtype=np.int32)

    device_families: dict[GeometryFamily, DeviceFamilyGeometryBuffer] = {}

    if polygon_count > 0:
        # Extract polygon rings
        poly_ring_indices = np.flatnonzero(ring_is_poly)
        poly_ring_order = all_ring_order[poly_ring_indices]
        poly_coord_counts = all_ring_coord_counts[poly_ring_indices]
        # Ring offsets = cumulative coord counts
        h_poly_ring_offsets = np.zeros(len(poly_ring_indices) + 1, dtype=np.int32)
        np.cumsum(poly_coord_counts, out=h_poly_ring_offsets[1:])
        # Geometry offsets: each polygon has rings_per_poly rings.
        # Since polygon rows have exactly 1 polygon each, geometry offsets
        # are just the cumulative ring-per-polygon counts for polygon family.
        rings_per_family_poly = ring_counts_per_poly[poly_mask_per_polygon]
        h_poly_geom_offsets = np.zeros(polygon_count + 1, dtype=np.int32)
        np.cumsum(rings_per_family_poly, out=h_poly_geom_offsets[1:])
        # Vectorised coordinate gather
        d_poly_x, d_poly_y = _gather_coords_vectorised(
            d_all_x, d_all_y, h_all_coord_offsets, poly_ring_order, poly_coord_counts,
        )
        device_families[GeometryFamily.POLYGON] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=d_poly_x,
            y=d_poly_y,
            geometry_offsets=runtime.from_host(h_poly_geom_offsets),
            empty_mask=runtime.from_host(np.zeros(polygon_count, dtype=np.bool_)),
            ring_offsets=runtime.from_host(h_poly_ring_offsets),
            bounds=None,
        )

    if multipolygon_count > 0:
        # Extract multipolygon rings
        mpoly_ring_indices = np.flatnonzero(ring_is_mpoly)
        mpoly_ring_order = all_ring_order[mpoly_ring_indices]
        mpoly_coord_counts = all_ring_coord_counts[mpoly_ring_indices]
        # Ring offsets
        h_mpoly_ring_offsets = np.zeros(len(mpoly_ring_indices) + 1, dtype=np.int32)
        np.cumsum(mpoly_coord_counts, out=h_mpoly_ring_offsets[1:])
        # Part offsets: each polygon (part) within a multipolygon
        # has rings_per_poly rings. Cumulative within each multipolygon row.
        mpoly_polygons = np.flatnonzero(mpoly_mask_per_polygon)
        rings_per_mpoly_part = ring_counts_per_poly[mpoly_polygons]
        h_mpoly_part_offsets = np.zeros(len(mpoly_polygons) + 1, dtype=np.int32)
        np.cumsum(rings_per_mpoly_part, out=h_mpoly_part_offsets[1:])
        # Geometry offsets: parts per multipolygon row
        mpoly_row_indices = poly_to_row[mpoly_polygons]
        # Map row indices to family-local multipolygon index
        mpoly_row_to_family = np.full(output_row_count, -1, dtype=np.int32)
        mpoly_rows = np.flatnonzero(mpoly_row_mask)
        mpoly_row_to_family[mpoly_rows] = np.arange(multipolygon_count, dtype=np.int32)
        mpoly_family_ids = mpoly_row_to_family[mpoly_row_indices]
        parts_per_geom = np.bincount(mpoly_family_ids, minlength=multipolygon_count).astype(np.int32)
        h_mpoly_geom_offsets = np.zeros(multipolygon_count + 1, dtype=np.int32)
        np.cumsum(parts_per_geom, out=h_mpoly_geom_offsets[1:])
        # Vectorised coordinate gather
        d_mpoly_x, d_mpoly_y = _gather_coords_vectorised(
            d_all_x, d_all_y, h_all_coord_offsets, mpoly_ring_order, mpoly_coord_counts,
        )
        device_families[GeometryFamily.MULTIPOLYGON] = DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            x=d_mpoly_x,
            y=d_mpoly_y,
            geometry_offsets=runtime.from_host(h_mpoly_geom_offsets),
            empty_mask=runtime.from_host(
                np.zeros(multipolygon_count, dtype=np.bool_),
            ),
            part_offsets=runtime.from_host(h_mpoly_part_offsets),
            ring_offsets=runtime.from_host(h_mpoly_ring_offsets),
            bounds=None,
        )

    result = build_device_resident_owned(
        device_families=device_families,
        row_count=output_row_count,
        tags=tags,
        validity=validity,
        family_row_offsets=family_row_offsets,
    )
    result.runtime_history.append(runtime_selection)
    return result


def _gather_coords_vectorised(
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    h_all_coord_offsets: np.ndarray,
    ring_order: np.ndarray,
    coord_counts: np.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Vectorised coordinate gather: build flat index array without Python loops.

    For each ring in ring_order, gathers coord_counts[i] consecutive
    coordinates starting at h_all_coord_offsets[ring_order[i]].
    """
    if ring_order.size == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    total_coords = int(coord_counts.sum())
    # Build per-ring start positions in source coordinate array
    ring_starts = h_all_coord_offsets[ring_order].astype(np.int64)
    # Expand: for each coordinate slot, compute source index.
    # slot_ring[i] = which ring does coordinate slot i belong to?
    ring_offsets = np.zeros(len(ring_order) + 1, dtype=np.int64)
    np.cumsum(coord_counts, out=ring_offsets[1:])
    slot_ring = np.repeat(np.arange(len(ring_order), dtype=np.int32), coord_counts)
    slot_local = np.arange(total_coords, dtype=np.int64) - ring_offsets[slot_ring]
    h_gather = ring_starts[slot_ring] + slot_local
    d_gather = cp.asarray(h_gather)
    return d_all_x[d_gather], d_all_y[d_gather]


def _overlay_intersection_rectangles_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    requested: ExecutionMode,
) -> OwnedGeometryArray | None:
    if cp is None or left.row_count != right.row_count:
        return None
    left_bounds = _axis_aligned_box_bounds(left)
    right_bounds = _axis_aligned_box_bounds(right)
    if left_bounds is None or right_bounds is None:
        return None

    runtime_selection = RuntimeSelection(
        requested=requested,
        selected=ExecutionMode.GPU,
        reason="GPU rectangle intersection fast path selected",
    )
    left_device = cp.asarray(left_bounds)
    right_device = cp.asarray(right_bounds)
    xmin = cp.maximum(left_device[:, 0], right_device[:, 0])
    ymin = cp.maximum(left_device[:, 1], right_device[:, 1])
    xmax = cp.minimum(left_device[:, 2], right_device[:, 2])
    ymax = cp.minimum(left_device[:, 3], right_device[:, 3])
    keep = (xmin < xmax) & (ymin < ymax)
    keep_rows = cp.flatnonzero(keep).astype(cp.int32, copy=False)
    if int(keep_rows.size) == 0:
        return _empty_polygon_output(runtime_selection)

    xmin = xmin[keep_rows]
    ymin = ymin[keep_rows]
    xmax = xmax[keep_rows]
    ymax = ymax[keep_rows]
    row_count = int(keep_rows.size)
    out_x = cp.empty((row_count * 5,), dtype=cp.float64)
    out_y = cp.empty((row_count * 5,), dtype=cp.float64)
    out_x[0::5] = xmin
    out_y[0::5] = ymin
    out_x[1::5] = xmax
    out_y[1::5] = ymin
    out_x[2::5] = xmax
    out_y[2::5] = ymax
    out_x[3::5] = xmin
    out_y[3::5] = ymax
    out_x[4::5] = xmin
    out_y[4::5] = ymin
    return _build_device_backed_fixed_polygon_output(
        out_x,
        out_y,
        row_count=row_count,
        runtime_selection=runtime_selection,
    )
