"""Polygon output assembly from overlay face data.

This module contains the GPU and utility functions for assembling polygon
output from half-edge graph faces produced by the overlay pipeline.
Extracted from ``overlay/gpu.py`` to reduce module size; see ADR-0016.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    DeviceArray,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum, sort_pairs
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    seed_all_validity_cache,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.runtime.residency import Residency

from .graph import _stable_radix_order_pass
from .types import (
    HalfEdgeGraph,
    IndexedComponentContainmentDeviceState,
    OverlayFaceTable,
)

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FaceBoundaryRings:
    """Capacity-backed boundary cycles for one device face selection."""

    area: DeviceArray
    edge_counts: DeviceArray
    coord_offsets: DeviceArray
    x: DeviceArray
    y: DeviceArray
    active: DeviceArray
    source_rows: DeviceArray
    face_ids: DeviceArray | None
    component_ids: DeviceArray | None
    ring_count: int
    coord_capacity: int


def _sync_hotpath(runtime) -> None:
    if hotpath_trace_enabled():
        runtime.synchronize()


def _device_int_scalar(value, *, reason: str) -> int:
    host = get_cuda_runtime().copy_device_to_host(value, reason=reason)
    return int(np.asarray(host).reshape(-1)[0])


def _device_bool_scalar(value, *, reason: str) -> bool:
    host = get_cuda_runtime().copy_device_to_host(value, reason=reason)
    return bool(np.asarray(host).reshape(-1)[0])


def _has_polygonal_families(geom: OwnedGeometryArray) -> bool:
    """Return True if the geometry array has POLYGON or MULTIPOLYGON families."""
    return GeometryFamily.POLYGON in geom.families or GeometryFamily.MULTIPOLYGON in geom.families


def _empty_polygon_output(
    runtime_selection: RuntimeSelection,
    *,
    row_count: int = 0,
) -> OwnedGeometryArray:
    residency = Residency.DEVICE if cp is not None else Residency.HOST
    empty_validity = np.zeros(row_count, dtype=bool)
    empty_tags = np.full(row_count, -1, dtype=np.int8)
    empty_offsets = np.full(row_count, -1, dtype=np.int32)
    if residency is Residency.DEVICE:
        result = build_device_resident_owned(
            device_families={},
            row_count=row_count,
            tags=cp.asarray(empty_tags),
            validity=cp.asarray(empty_validity),
            family_row_offsets=cp.asarray(empty_offsets),
            execution_mode="gpu",
        )
        result.runtime_history.append(runtime_selection)
        return result
    return OwnedGeometryArray(
        validity=empty_validity,
        tags=empty_tags,
        family_row_offsets=empty_offsets,
        families={},
        residency=residency,
        runtime_history=[runtime_selection],
    )


def _build_device_backed_fixed_polygon_output(
    device_x: DeviceArray,
    device_y: DeviceArray,
    *,
    row_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    bounds = cp.column_stack(
        (
            device_x[0::5],
            device_y[0::5],
            device_x[2::5],
            device_y[2::5],
        )
    )
    result = build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=device_x,
                y=device_y,
                geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
                empty_mask=cp.zeros(row_count, dtype=cp.bool_),
                ring_offsets=cp.arange(0, (row_count + 1) * 5, 5, dtype=cp.int32),
                bounds=bounds,
                dense_single_ring_width=5,
                axis_aligned_rectangles=True,
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=cp.ones(row_count, dtype=cp.bool_),
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )
    result.runtime_history.append(runtime_selection)
    seed_all_validity_cache(result)
    return result


def _axis_aligned_box_bounds(values: OwnedGeometryArray) -> np.ndarray | None:
    if set(values.families) != {GeometryFamily.POLYGON}:
        return None
    polygon_buffer = values.families[GeometryFamily.POLYGON]
    row_count = polygon_buffer.row_count
    if row_count == 0 or row_count != values.row_count:
        return None
    if polygon_buffer.ring_offsets is None:
        return None
    if not np.array_equal(
        polygon_buffer.geometry_offsets, np.arange(row_count + 1, dtype=np.int32)
    ):
        return None
    if not np.array_equal(
        polygon_buffer.ring_offsets, np.arange(0, (row_count + 1) * 5, 5, dtype=np.int32)
    ):
        return None
    if np.any(polygon_buffer.empty_mask):
        return None
    if not polygon_buffer.host_materialized:
        # Rectangle detection is the host-only fast path that actually needs
        # x/y. Keep the D->H transfer local instead of forcing it at overlay
        # entry.
        values._ensure_host_state()
        polygon_buffer = values.families[GeometryFamily.POLYGON]

    x = polygon_buffer.x.reshape(row_count, 5)
    y = polygon_buffer.y.reshape(row_count, 5)
    if not (np.allclose(x[:, 0], x[:, 4]) and np.allclose(y[:, 0], y[:, 4])):
        return None

    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    axis_aligned = (np.abs(dx) < SPATIAL_EPSILON) ^ (np.abs(dy) < SPATIAL_EPSILON)
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


def _axis_aligned_box_bounds_device(values: OwnedGeometryArray):
    """Return device bounds for dense rectangle polygons without host x/y export."""
    if cp is None or values.device_state is None:
        return None
    if set(values.families) != {GeometryFamily.POLYGON}:
        return None
    device_buffer = values.device_state.families.get(GeometryFamily.POLYGON)
    if device_buffer is None:
        return None
    row_count = int(values.row_count)
    if row_count == 0:
        return None
    if int(device_buffer.geometry_offsets.size) != row_count + 1:
        return None
    if int(device_buffer.x.size) != row_count * 5 or int(device_buffer.y.size) != row_count * 5:
        return None
    if device_buffer.dense_single_ring_width != 5:
        return None
    if bool(getattr(device_buffer, "axis_aligned_rectangles", False)):
        if device_buffer.bounds is not None:
            bounds = cp.asarray(device_buffer.bounds)
            if tuple(int(dim) for dim in bounds.shape) == (row_count, 4):
                return bounds
        x = cp.asarray(device_buffer.x).reshape(row_count, 5)
        y = cp.asarray(device_buffer.y).reshape(row_count, 5)
        return cp.column_stack(
            (
                cp.min(x[:, :4], axis=1),
                cp.min(y[:, :4], axis=1),
                cp.max(x[:, :4], axis=1),
                cp.max(y[:, :4], axis=1),
            )
        )

    x = cp.asarray(device_buffer.x).reshape(row_count, 5)
    y = cp.asarray(device_buffer.y).reshape(row_count, 5)
    closed = (cp.abs(x[:, 0] - x[:, 4]) <= SPATIAL_EPSILON) & (
        cp.abs(y[:, 0] - y[:, 4]) <= SPATIAL_EPSILON
    )
    dx = cp.diff(x, axis=1)
    dy = cp.diff(y, axis=1)
    axis_aligned_edges = (cp.abs(dx) < SPATIAL_EPSILON) ^ (cp.abs(dy) < SPATIAL_EPSILON)
    bounds = cp.column_stack(
        (
            cp.min(x[:, :4], axis=1),
            cp.min(y[:, :4], axis=1),
            cp.max(x[:, :4], axis=1),
            cp.max(y[:, :4], axis=1),
        )
    )
    nondegenerate = (bounds[:, 0] < bounds[:, 2]) & (bounds[:, 1] < bounds[:, 3])
    if not _device_bool_scalar(
        cp.all(closed & cp.all(axis_aligned_edges, axis=1) & nondegenerate),
        reason="overlay rectangle fast-path box certification scalar fence",
    ):
        return None
    return bounds


def _extract_face_boundary_rings_gpu(
    half_edge_graph: HalfEdgeGraph,
    *,
    d_face_offsets,
    d_face_edge_ids,
    d_face_selected,
    edge_face_ids=None,
    d_face_component=None,
    kernels,
    runtime,
    area_epsilon: float,
    stage: str,
) -> _FaceBoundaryRings | None:
    """Extract collective boundary cycles for a device face-selection mask."""
    from vibespatial.api._native_rowset import NativeDeviceSelection

    device = half_edge_graph.device_state
    if device is None:
        return None
    ptr = runtime.pointer
    edge_count = half_edge_graph.edge_count
    face_count = int(d_face_selected.size)
    if edge_count < 3 or face_count == 0:
        return None

    with hotpath_stage(f"{stage}.edge_selection", category="setup"):
        d_edge_selected = cp.zeros(edge_count, dtype=cp.int8)
        runtime.launch(
            kernels["scatter_edge_face_selection"],
            grid=(max(face_count, 1), 1, 1),
            block=(256, 1, 1),
            params=(
                (
                    ptr(d_face_offsets),
                    ptr(d_face_edge_ids),
                    ptr(d_face_selected),
                    ptr(d_edge_selected),
                    face_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    with hotpath_stage(f"{stage}.boundary_edges", category="refine"):
        d_is_boundary = cp.empty(edge_count, dtype=cp.int8)
        edge_grid, edge_block = runtime.launch_config(
            kernels["compute_boundary_edges"],
            edge_count,
        )
        runtime.launch(
            kernels["compute_boundary_edges"],
            grid=edge_grid,
            block=edge_block,
            params=(
                (ptr(d_edge_selected), ptr(d_is_boundary), edge_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
            ),
        )
        _sync_hotpath(runtime)

    with hotpath_stage(f"{stage}.boundary_cycles", category="sort"):
        boundary_selection = NativeDeviceSelection.from_mask(d_is_boundary != 0)
        boundary_edge_indices = boundary_selection.partition_capacity_positions().astype(
            cp.int32,
            copy=False,
        )
        d_boundary_active = boundary_selection.active_capacity_mask()
        boundary_capacity = boundary_selection.capacity
        d_boundary_next_full = cp.empty(boundary_capacity, dtype=cp.int32)
        boundary_next_grid, boundary_next_block = runtime.launch_config(
            kernels["compute_boundary_next"],
            boundary_capacity,
        )
        runtime.launch(
            kernels["compute_boundary_next"],
            grid=boundary_next_grid,
            block=boundary_next_block,
            params=(
                (
                    ptr(boundary_edge_indices),
                    ptr(d_boundary_active),
                    ptr(device.next_edge_ids),
                    ptr(d_is_boundary),
                    ptr(d_boundary_next_full),
                    boundary_capacity,
                    edge_count,
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
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        d_boundary_inverse = cp.empty(edge_count, dtype=cp.int32)
        d_boundary_inverse[boundary_edge_indices] = cp.where(
            d_boundary_active,
            cp.arange(boundary_capacity, dtype=cp.int32),
            cp.int32(-1),
        )
        compact_next = cp.where(
            d_boundary_active,
            d_boundary_inverse[d_boundary_next_full],
            cp.arange(boundary_capacity, dtype=cp.int32),
        ).astype(cp.int32, copy=False)

        cycle_label = cp.arange(boundary_capacity, dtype=cp.int32)
        jump = compact_next.copy()
        max_iter = max(1, int(np.ceil(np.log2(max(1, boundary_capacity)))))
        for _ in range(max_iter):
            cycle_label = cp.minimum(cycle_label, cycle_label[jump])
            jump = jump[jump]

        sorted_cycles = sort_pairs(
            cycle_label,
            cp.arange(boundary_capacity, dtype=cp.int32),
            synchronize=False,
        )
        sorted_compact_ids = sorted_cycles.values
        sorted_labels = sorted_cycles.keys
        cycle_start_mask = cp.empty(boundary_capacity, dtype=cp.bool_)
        cycle_start_mask[0] = True
        if boundary_capacity > 1:
            cycle_start_mask[1:] = sorted_labels[1:] != sorted_labels[:-1]
        cycle_selection = NativeDeviceSelection.from_mask(cycle_start_mask)
        cycle_starts = cycle_selection.partition_capacity_positions().astype(
            cp.int32,
            copy=False,
        )
        d_cycle_active = cycle_selection.active_capacity_mask()
        d_cycle_lanes = cp.arange(boundary_capacity, dtype=cp.int64)
        d_has_next_cycle = (
            d_cycle_lanes + 1 < cp.asarray(cycle_selection.logical_count, dtype=cp.int64)[0]
        )
        cycle_ends = cp.where(
            d_cycle_active & d_has_next_cycle,
            cycle_starts[cp.minimum(d_cycle_lanes + 1, boundary_capacity - 1)],
            cp.int32(boundary_capacity),
        )
        cycle_lengths = cp.where(
            d_cycle_active,
            cycle_ends - cycle_starts,
            cp.int32(0),
        )
        valid_cycles = NativeDeviceSelection.from_mask(d_cycle_active & (cycle_lengths >= 3))
        ring_count = boundary_capacity // 3
        if ring_count == 0:
            return None
        valid_cycles = NativeDeviceSelection(
            positions=valid_cycles.positions[:ring_count],
            logical_count=valid_cycles.logical_count,
            source_row_count=boundary_capacity,
        )
        d_ring_active = valid_cycles.active_capacity_mask()
        valid_cycle_starts = valid_cycles.gather_capacity(
            cycle_starts,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        valid_cycle_ends = valid_cycles.gather_capacity(
            cycle_ends,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ring_edge_counts = valid_cycles.gather_capacity(
            cycle_lengths,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ring_edge_starts = sorted_compact_ids[valid_cycle_starts]

    with hotpath_stage(f"{stage}.boundary_metrics", category="refine"):
        d_ring_area = cp.empty(ring_count, dtype=cp.float64)
        area_block_size = min(
            256,
            runtime.optimal_block_size(kernels["compute_centered_boundary_ring_areas"]),
        )
        area_block_size = 1 << (max(1, int(area_block_size)).bit_length() - 1)
        runtime.launch(
            kernels["compute_centered_boundary_ring_areas"],
            grid=(ring_count, 1, 1),
            block=(area_block_size, 1, 1),
            params=(
                (
                    ptr(device.src_x),
                    ptr(device.src_y),
                    ptr(boundary_edge_indices),
                    ptr(sorted_compact_ids),
                    ptr(valid_cycle_starts),
                    ptr(valid_cycle_ends),
                    ptr(d_ring_active),
                    ptr(d_boundary_next_full),
                    ptr(d_ring_area),
                    ring_count,
                    boundary_capacity,
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
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        _sync_hotpath(runtime)

    d_row_indices = cp.asarray(device.row_indices)
    d_cycle_source_rows = cp.where(
        d_ring_active,
        d_row_indices[boundary_edge_indices[d_ring_edge_starts]],
        cp.int32(-1),
    ).astype(cp.int32, copy=False)
    d_cycle_face_ids = None
    d_cycle_component_ids = None
    if edge_face_ids is not None and d_face_component is not None:
        d_cycle_face_ids = cp.where(
            d_ring_active,
            cp.asarray(edge_face_ids, dtype=cp.int32)[
                boundary_edge_indices[d_ring_edge_starts]
            ],
            cp.int32(-1),
        ).astype(cp.int32, copy=False)
        d_cycle_component_ids = cp.where(
            d_ring_active,
            cp.asarray(d_face_component, dtype=cp.int32)[
                d_cycle_face_ids.clip(0, face_count - 1)
            ],
            cp.int32(-1),
        ).astype(cp.int32, copy=False)

    with hotpath_stage(f"{stage}.boundary_scatter", category="emit"):
        ring_coord_counts = cp.where(
            d_ring_active,
            d_ring_edge_counts + 1,
            cp.int32(0),
        )
        d_ring_coord_offsets = exclusive_sum(ring_coord_counts.astype(cp.int32, copy=False))
        coord_capacity = int(boundary_capacity + ring_count)
        d_out_x = cp.zeros(coord_capacity, dtype=cp.float64)
        d_out_y = cp.zeros(coord_capacity, dtype=cp.float64)
        ring_grid, ring_block = runtime.launch_config(
            kernels["scatter_boundary_ring_coordinates"],
            ring_count,
        )
        runtime.launch(
            kernels["scatter_boundary_ring_coordinates"],
            grid=ring_grid,
            block=ring_block,
            params=(
                (
                    ptr(device.src_x),
                    ptr(device.src_y),
                    ptr(boundary_edge_indices),
                    ptr(d_ring_edge_starts),
                    ptr(d_ring_coord_offsets),
                    ptr(d_ring_edge_counts),
                    ptr(d_ring_active),
                    ptr(compact_next),
                    ptr(d_out_x),
                    ptr(d_out_y),
                    ring_count,
                    boundary_capacity,
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
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        _sync_hotpath(runtime)

    # Bounded cycles come from the exact half-edge arrangement. Their output
    # dimension is the exact sign of the represented fp64 ring, not an error
    # interval around how an intersection coordinate was constructed.
    d_nondegenerate = d_ring_active & (cp.abs(d_ring_area) > area_epsilon)
    return _FaceBoundaryRings(
        area=cp.where(d_nondegenerate, d_ring_area, 0.0),
        edge_counts=cp.where(d_nondegenerate, d_ring_edge_counts, cp.int32(0)),
        coord_offsets=d_ring_coord_offsets,
        x=d_out_x,
        y=d_out_y,
        active=d_nondegenerate,
        source_rows=cp.where(d_nondegenerate, d_cycle_source_rows, cp.int32(-1)),
        face_ids=(
            None
            if d_cycle_face_ids is None
            else cp.where(d_nondegenerate, d_cycle_face_ids, cp.int32(-1))
        ),
        component_ids=(
            None
            if d_cycle_component_ids is None
            else cp.where(d_nondegenerate, d_cycle_component_ids, cp.int32(-1))
        ),
        ring_count=ring_count,
        coord_capacity=coord_capacity,
    )


def _build_polygon_output_from_faces_gpu(
    half_edge_graph: HalfEdgeGraph,
    faces: OverlayFaceTable,
    selected_face_indices,
    *,
    preserve_row_count: int | None = None,
    d_valid_empty_rows=None,
    area_epsilon: float = 0.0,
    component_nesting: IndexedComponentContainmentDeviceState | None = None,
) -> OwnedGeometryArray | None:
    """GPU face-to-polygon assembly (Phase 11: GPU boundary cycle detection).

    Full GPU pipeline:
      Steps 1-2: Face selection scattered to an edge-selection bit carrier.
      Step 3: Boundary edge identification via NVRTC kernel.
      Step 4: Boundary next-edge computation via NVRTC kernel.
      Step 5: Compact boundary cycle detection via GPU pointer jumping;
              per-cycle area via segmented reduction.
      Steps 6-7: Coordinate offset computation and ring scatter via GPU.
      Step 8: Classify oriented boundary cycles into shells and holes.
      Step 9: Assign holes to exteriors via GPU PIP kernel.
      Step 10: GPU output assembly with device-side sorting and grouping.

    Returns None if GPU is unavailable (caller falls back to CPU path).
    """
    if cp is None or half_edge_graph.device_state is None or faces.device_state is None:
        return None

    def _empty_preserved_output() -> OwnedGeometryArray:
        if preserve_row_count is not None and d_valid_empty_rows is not None:
            from vibespatial.geometry.owned import build_empty_polygon_rows_device

            result = build_empty_polygon_rows_device(
                preserve_row_count,
                validity=cp.asarray(d_valid_empty_rows, dtype=cp.bool_),
            )
            result.runtime_history.append(faces.runtime_selection)
            return result
        return _empty_polygon_output(
            faces.runtime_selection,
            row_count=preserve_row_count or 0,
        )

    selection_capacity = getattr(selected_face_indices, "capacity", None)
    if (
        selection_capacity is None and int(selected_face_indices.size) == 0
    ) or selection_capacity == 0:
        return _empty_preserved_output()

    # Lazy import: kernel compile functions stay in gpu.py to avoid
    # circular imports (they depend on gpu_kernels module-level state).
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.overlay.gpu import (
        _overlay_face_assembly_kernels,
        _overlay_face_walk_kernels,
    )

    runtime = get_cuda_runtime()
    kernels = _overlay_face_assembly_kernels()
    walk_kernels = _overlay_face_walk_kernels()
    kernels.update(walk_kernels)
    ptr = runtime.pointer
    face_count = faces.face_count

    face_device = faces.device_state

    d_face_offsets = cp.asarray(face_device.face_offsets)
    d_face_edge_ids = cp.asarray(face_device.face_edge_ids)
    if selection_capacity is not None:
        if int(selection_capacity) != face_count:
            raise ValueError("selected face capacity must match overlay face count")
        d_face_selected = selected_face_indices.source_mask().astype(
            cp.int8,
            copy=False,
        )
    else:
        d_face_selected = cp.zeros(face_count, dtype=cp.int8)
        d_face_selected[cp.asarray(selected_face_indices)] = 1

    selected_rings = _extract_face_boundary_rings_gpu(
        half_edge_graph,
        d_face_offsets=d_face_offsets,
        d_face_edge_ids=d_face_edge_ids,
        d_face_selected=d_face_selected,
        edge_face_ids=(
            face_device.edge_face_ids if component_nesting is not None else None
        ),
        d_face_component=(
            component_nesting.face_component
            if component_nesting is not None
            else None
        ),
        kernels=kernels,
        runtime=runtime,
        area_epsilon=area_epsilon,
        stage="overlay.assemble.selected",
    )
    if selected_rings is None:
        return _empty_preserved_output()

    d_ring_area = selected_rings.area
    d_ring_edge_counts = selected_rings.edge_counts
    d_ring_coord_offsets = selected_rings.coord_offsets
    d_out_x = selected_rings.x
    d_out_y = selected_rings.y
    d_ring_active = selected_rings.active
    d_cycle_source_rows = selected_rings.source_rows
    d_cycle_face_ids = selected_rings.face_ids
    d_cycle_component_ids = selected_rings.component_ids
    ring_count = selected_rings.ring_count

    # Selected-region cycles already carry every selected/excluded interface:
    # positive cycles are exteriors and negative cycles are holes. Extracting
    # the inverse excluded-face set duplicates those edges and inflates ring
    # capacity to the full face table without adding output topology.
    del d_face_selected
    total_ring_count = ring_count
    d_all_area = d_ring_area
    d_all_x = d_out_x
    d_all_y = d_out_y
    d_all_coord_offsets = d_ring_coord_offsets
    d_all_edge_counts = d_ring_edge_counts
    d_all_ring_active = d_ring_active

    # Boundary rings get their source row from the selected face. Holes inherit
    # the exterior row after assignment below.
    d_all_source_rows = cp.where(
        d_ring_active,
        d_cycle_source_rows,
        cp.int32(-1),
    )

    d_ring_active_i8 = d_all_ring_active.astype(cp.int8, copy=False)
    if component_nesting is not None:
        # Boundary-only topology already classified disconnected components
        # through the indexed exact relation used by face propagation. Consume
        # those structural hints directly instead of repeating ring F² PIP.
        if (
            component_nesting.component_depth is None
            or component_nesting.component_parent is None
            or d_cycle_face_ids is None
            or d_cycle_component_ids is None
        ):
            raise ValueError("boundary component nesting hints are incomplete")
        d_safe_components = d_cycle_component_ids.clip(0, face_count - 1)
        d_containment_depth = cp.where(
            d_all_ring_active,
            cp.asarray(component_nesting.component_depth)[d_safe_components],
            cp.int32(0),
        ).astype(cp.int32, copy=False)
        d_exterior_mask_full = d_all_ring_active & (
            (d_containment_depth & np.int32(1)) == 0
        )
        d_face_to_ring = cp.full(face_count, -1, dtype=cp.int32)
        d_ring_ids = cp.arange(total_ring_count, dtype=cp.int32)
        d_face_to_ring[d_cycle_face_ids[d_all_ring_active]] = d_ring_ids[
            d_all_ring_active
        ]
        d_parent_faces = cp.asarray(component_nesting.component_parent)[
            d_safe_components
        ]
        d_parent_rings = d_face_to_ring[d_parent_faces.clip(0, face_count - 1)]
        d_exterior_id = cp.where(
            d_exterior_mask_full,
            d_ring_ids,
            cp.where(d_all_ring_active, d_parent_rings, cp.int32(-1)),
        ).astype(cp.int32, copy=False)
    else:
        # Generic overlay assembly retains source-row ring semantics. It has no
        # component carrier, so classify and assign its output rings locally.
        with hotpath_stage("overlay.assemble.sample_points", category="refine"):
            d_ring_sample_x = cp.empty(total_ring_count, dtype=cp.float64)
            d_ring_sample_y = cp.empty(total_ring_count, dtype=cp.float64)
            d_ring_bounds = cp.empty((total_ring_count, 4), dtype=cp.float64)
            sample_grid, sample_block = runtime.launch_config(
                kernels["compute_ring_sample_points"],
                total_ring_count,
            )
            runtime.launch(
                kernels["compute_ring_sample_points"],
                grid=sample_grid,
                block=sample_block,
                params=(
                    (
                        ptr(d_all_coord_offsets),
                        ptr(d_all_edge_counts),
                        ptr(d_all_ring_active),
                        ptr(d_all_x),
                        ptr(d_all_y),
                        ptr(d_ring_sample_x),
                        ptr(d_ring_sample_y),
                        ptr(d_ring_bounds),
                        total_ring_count,
                    ),
                    (KERNEL_PARAM_PTR,) * 8 + (KERNEL_PARAM_I32,),
                ),
            )
            _sync_hotpath(runtime)

        d_containment_order = cp.arange(total_ring_count, dtype=cp.int32)
        d_containment_group_key = cp.where(
            d_all_ring_active,
            d_all_source_rows,
            cp.int32(np.iinfo(np.int32).max),
        ).astype(cp.int32, copy=False)
        d_containment_order = _stable_radix_order_pass(
            d_containment_order,
            d_containment_group_key,
        )
        d_sorted_group_keys = d_containment_group_key[d_containment_order]
        d_containment_group_start = cp.empty(total_ring_count, dtype=cp.int32)
        d_containment_group_end = cp.empty(total_ring_count, dtype=cp.int32)
        group_span_grid, group_span_block = runtime.launch_config(
            kernels["locate_boundary_ring_group_spans"],
            total_ring_count,
        )
        runtime.launch(
            kernels["locate_boundary_ring_group_spans"],
            grid=group_span_grid,
            block=group_span_block,
            params=(
                (
                    ptr(d_sorted_group_keys),
                    ptr(d_containment_order),
                    ptr(d_ring_active_i8),
                    ptr(d_containment_group_start),
                    ptr(d_containment_group_end),
                    total_ring_count,
                ),
                (KERNEL_PARAM_PTR,) * 5 + (KERNEL_PARAM_I32,),
            ),
        )
        d_containment_depth = cp.zeros(total_ring_count, dtype=cp.int32)
        containment_grid, containment_block = runtime.launch_config(
            kernels["count_boundary_ring_containment_depth"],
            total_ring_count,
        )
        runtime.launch(
            kernels["count_boundary_ring_containment_depth"],
            grid=containment_grid,
            block=containment_block,
            params=(
                (
                    ptr(d_ring_sample_x),
                    ptr(d_ring_sample_y),
                    ptr(d_all_area),
                    ptr(d_ring_active_i8),
                    ptr(d_containment_order),
                    ptr(d_containment_group_start),
                    ptr(d_containment_group_end),
                    ptr(d_all_coord_offsets),
                    ptr(d_all_edge_counts),
                    ptr(d_all_x),
                    ptr(d_all_y),
                    ptr(d_ring_bounds),
                    ptr(d_containment_depth),
                    total_ring_count,
                ),
                (KERNEL_PARAM_PTR,) * 13 + (KERNEL_PARAM_I32,),
            ),
        )
        _sync_hotpath(runtime)
        d_exterior_mask_full = d_all_ring_active & (
            (d_containment_depth & np.int32(1)) == 0
        )
        with hotpath_stage("overlay.assemble.hole_assignment", category="refine"):
            d_exterior_id = cp.full(total_ring_count, -1, dtype=cp.int32)
            ring_grid_all, ring_block_all = runtime.launch_config(
                kernels["assign_holes_to_exteriors"],
                total_ring_count,
            )
            runtime.launch(
                kernels["assign_holes_to_exteriors"],
                grid=ring_grid_all,
                block=ring_block_all,
                params=(
                    (
                        ptr(d_ring_sample_x),
                        ptr(d_ring_sample_y),
                        ptr(d_all_area),
                        ptr(d_exterior_mask_full.astype(cp.int8, copy=False)),
                        ptr(d_ring_active_i8),
                        ptr(d_containment_order),
                        ptr(d_containment_group_start),
                        ptr(d_containment_group_end),
                        ptr(d_all_coord_offsets),
                        ptr(d_all_edge_counts),
                        ptr(d_all_x),
                        ptr(d_all_y),
                        ptr(d_ring_bounds),
                        ptr(d_exterior_id),
                        total_ring_count,
                    ),
                    (KERNEL_PARAM_PTR,) * 14 + (KERNEL_PARAM_I32,),
                ),
            )
            _sync_hotpath(runtime)

    exterior_selection = NativeDeviceSelection.from_mask(d_exterior_mask_full)
    d_exterior_indices = exterior_selection.partition_capacity_positions().astype(
        cp.int32,
        copy=False,
    )

    # Odd-depth rings are holes. Even-depth islands were admitted as exteriors
    # above and remain concrete polygons in the same output row.
    d_is_hole = (
        d_all_ring_active
        & (d_exterior_id >= 0)
        & ((d_containment_depth % 2) != 0)
    )

    with hotpath_stage("overlay.assemble.output_grouping", category="sort"):
        # --- Step 10: Device-resident output assembly ---
        # Build GeoArrow-format polygon output entirely on device.
        d_valid_hole_mask = d_is_hole

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

        d_explicit_polygon_output_rows = None
        d_explicit_polygon_active = None
        if preserve_row_count is not None:
            polygon_capacity = total_ring_count
            d_polygon_lanes = cp.arange(polygon_capacity, dtype=cp.int32)
            d_polygon_active = exterior_selection.active_capacity_mask()
            d_exterior_ring_ids = d_exterior_indices.astype(cp.int32, copy=False)
            d_polygon_output_rows = cp.where(
                d_polygon_active,
                d_all_source_rows[d_exterior_ring_ids],
                cp.int32(0),
            )

            d_exterior_lane_extended = cp.full(
                total_ring_count + polygon_capacity,
                -1,
                dtype=cp.int32,
            )
            d_exterior_lane_destinations = cp.where(
                d_polygon_active,
                d_exterior_ring_ids,
                cp.int32(total_ring_count) + d_polygon_lanes,
            )
            d_exterior_lane_extended[d_exterior_lane_destinations] = cp.where(
                d_polygon_active,
                d_polygon_lanes,
                cp.int32(-1),
            )
            d_exterior_lanes = d_exterior_lane_extended[:total_ring_count]
            d_hole_polygon_lanes = d_exterior_lanes[d_exterior_id.clip(0, total_ring_count - 1)]
            d_count_destinations = cp.where(
                d_valid_hole_mask & (d_hole_polygon_lanes >= 0),
                d_hole_polygon_lanes,
                cp.int32(polygon_capacity),
            )
            d_holes_per_polygon = cp.bincount(
                d_count_destinations,
                weights=d_valid_hole_mask.astype(cp.int32, copy=False),
                minlength=polygon_capacity + 1,
            )[:polygon_capacity].astype(cp.int32, copy=False)
            d_rings_per_poly = cp.where(
                d_polygon_active,
                d_holes_per_polygon + cp.int32(1),
                cp.int32(0),
            )
            d_polygon_ring_offsets = cp.zeros(
                polygon_capacity + 1,
                dtype=cp.int64,
            )
            cp.cumsum(
                d_rings_per_poly,
                dtype=cp.int64,
                out=d_polygon_ring_offsets[1:],
            )
            d_poly_starts = d_polygon_ring_offsets[:-1]

            ring_capacity = total_ring_count
            d_output_ring_extended = cp.zeros(
                ring_capacity + polygon_capacity,
                dtype=cp.int32,
            )
            d_exterior_destinations = cp.where(
                d_polygon_active,
                d_polygon_ring_offsets[:-1],
                np.int64(ring_capacity) + d_polygon_lanes.astype(cp.int64),
            )
            d_output_ring_extended[d_exterior_destinations] = cp.where(
                d_polygon_active,
                d_exterior_ring_ids,
                cp.int32(0),
            )
            d_sorted_output_ids = d_output_ring_extended[:ring_capacity]
            d_hole_counters = cp.zeros(polygon_capacity, dtype=cp.int32)
            hole_grid, hole_block = runtime.launch_config(
                kernels["scatter_output_holes"],
                total_ring_count,
            )
            runtime.launch(
                kernels["scatter_output_holes"],
                grid=hole_grid,
                block=hole_block,
                params=(
                    (
                        ptr(d_valid_hole_mask.astype(cp.uint8, copy=False)),
                        ptr(d_hole_polygon_lanes),
                        ptr(d_polygon_ring_offsets),
                        ptr(d_hole_counters),
                        ptr(d_sorted_output_ids),
                        total_ring_count,
                    ),
                    (
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32,
                    ),
                ),
            )
            _sync_hotpath(runtime)

            d_row_count_destinations = cp.where(
                d_polygon_active,
                d_polygon_output_rows,
                cp.int32(preserve_row_count),
            )
            d_polys_per_row = cp.bincount(
                d_row_count_destinations,
                weights=d_polygon_active.astype(cp.int32, copy=False),
                minlength=preserve_row_count + 1,
            )[:preserve_row_count].astype(cp.int32, copy=False)
            n_output_rows = preserve_row_count
            d_output_source_rows = cp.arange(
                preserve_row_count,
                dtype=cp.int32,
            )
            d_explicit_polygon_output_rows = d_polygon_output_rows
            d_explicit_polygon_active = d_polygon_active
        else:
            # Dynamic public cardinality still crosses the explicit Owned-array
            # physicalization boundary. Native callers pass preserve_row_count.
            d_is_output_ring = d_exterior_mask_full | d_valid_hole_mask
            d_output_ring_ids = cp.flatnonzero(d_is_output_ring).astype(cp.int32)
            n_output_rings = int(d_output_ring_ids.size)
            if n_output_rings == 0:
                return _empty_polygon_output(faces.runtime_selection)

            d_out_ext_id = cp.where(
                d_exterior_mask_full[d_output_ring_ids],
                d_output_ring_ids,
                d_exterior_id[d_output_ring_ids],
            )
            d_out_source_row = d_all_source_rows[d_output_ring_ids]
            d_out_is_ext = d_exterior_mask_full[d_output_ring_ids].astype(cp.int32)

            d_sort_order = cp.arange(n_output_rings, dtype=cp.int32)
            for key in (
                d_output_ring_ids,
                1 - d_out_is_ext,
                d_out_ext_id,
                d_out_source_row,
            ):
                d_sort_order = _stable_radix_order_pass(d_sort_order, key)
            d_sorted_output_ids = d_output_ring_ids[d_sort_order]
            d_sorted_is_ext = d_exterior_mask_full[d_sorted_output_ids]
            d_sorted_source_row = d_all_source_rows[d_sorted_output_ids]
            d_poly_starts = cp.flatnonzero(d_sorted_is_ext).astype(cp.int32)
            n_polygons = int(d_poly_starts.size)
            if n_polygons == 0:
                return _empty_polygon_output(faces.runtime_selection)
            d_poly_ends = cp.concatenate(
                (
                    d_poly_starts[1:],
                    cp.asarray([n_output_rings], dtype=cp.int32),
                )
            )
            d_rings_per_poly = d_poly_ends - d_poly_starts
            d_poly_source_row = d_sorted_source_row[d_poly_starts]
            d_row_change = cp.empty(n_polygons, dtype=cp.bool_)
            d_row_change[0] = True
            if n_polygons > 1:
                d_row_change[1:] = d_poly_source_row[1:] != d_poly_source_row[:-1]
            d_row_starts = cp.flatnonzero(d_row_change).astype(cp.int32)
            d_row_ends = cp.concatenate(
                (
                    d_row_starts[1:],
                    cp.asarray([n_polygons], dtype=cp.int32),
                )
            )
            d_polys_per_row = d_row_ends - d_row_starts
            n_output_rows = int(d_row_starts.size)
            d_output_source_rows = d_poly_source_row[d_row_starts]

    with hotpath_stage("overlay.assemble.output_materialize", category="emit"):
        return _build_device_resident_polygon_output(
            d_all_x=d_all_x,
            d_all_y=d_all_y,
            d_all_coord_offsets=d_all_coord_offsets,
            d_all_edge_counts=d_all_edge_counts,
            d_sorted_output_ids=d_sorted_output_ids,
            d_rings_per_poly=d_rings_per_poly,
            d_polys_per_row=d_polys_per_row,
            d_poly_starts=d_poly_starts,
            d_output_source_rows=d_output_source_rows,
            n_output_rows=n_output_rows,
            runtime_selection=faces.runtime_selection,
            preserve_row_count=preserve_row_count,
            d_valid_empty_rows=d_valid_empty_rows,
            d_explicit_polygon_output_rows=d_explicit_polygon_output_rows,
            d_explicit_polygon_active=d_explicit_polygon_active,
        )


def _build_device_resident_polygon_output(
    *,
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    d_all_coord_offsets: cp.ndarray,
    d_all_edge_counts: cp.ndarray | None,
    d_sorted_output_ids: cp.ndarray,
    d_rings_per_poly: cp.ndarray,
    d_polys_per_row: cp.ndarray,
    d_poly_starts: cp.ndarray,
    d_output_source_rows: cp.ndarray,
    n_output_rows: int,
    runtime_selection: RuntimeSelection,
    preserve_row_count: int | None = None,
    d_valid_empty_rows: cp.ndarray | None = None,
    coord_capacity: int | None = None,
    d_explicit_polygon_output_rows: cp.ndarray | None = None,
    d_explicit_polygon_active: cp.ndarray | None = None,
    d_sorted_output_edge_counts: cp.ndarray | None = None,
) -> OwnedGeometryArray:
    """Build row-capacity Polygon/MultiPolygon buffers without compaction.

    Polygon and MultiPolygon family buffers each retain public-row capacity.
    Ring and part membership is packed into capacity arrays with device logical
    prefixes; tags and validity select the active family for every public row.
    """
    from vibespatial.api._native_rowset import NativeDeviceSelection

    output_row_count = preserve_row_count if preserve_row_count is not None else n_output_rows
    if output_row_count == 0:
        return _empty_polygon_output(
            runtime_selection,
            row_count=preserve_row_count or 0,
        )

    d_compact_row_ids = (
        cp.asarray(d_output_source_rows, dtype=cp.int32)
        if preserve_row_count is not None
        else cp.arange(n_output_rows, dtype=cp.int32)
    )
    d_full_polys_per_row = cp.zeros(output_row_count, dtype=cp.int32)
    d_full_polys_per_row[d_compact_row_ids] = cp.asarray(
        d_polys_per_row,
        dtype=cp.int32,
    )
    d_nonempty_polygon_rows = d_full_polys_per_row == 1
    d_multipolygon_rows = d_full_polys_per_row > 1
    d_valid_empty = (
        cp.zeros(output_row_count, dtype=cp.bool_)
        if d_valid_empty_rows is None
        else cp.asarray(d_valid_empty_rows, dtype=cp.bool_).copy()
    )
    if int(d_valid_empty.size) != output_row_count:
        raise ValueError("valid-empty row mask must match polygon output rows")
    d_valid_empty &= d_full_polys_per_row == 0
    d_polygon_rows = d_nonempty_polygon_rows | d_valid_empty
    d_validity = d_polygon_rows | d_multipolygon_rows

    polygon_capacity = int(d_rings_per_poly.size)
    ring_capacity = int(d_sorted_output_ids.size)
    if polygon_capacity == 0 or ring_capacity == 0:
        from vibespatial.geometry.owned import build_empty_polygon_rows_device

        result = build_empty_polygon_rows_device(
            output_row_count,
            validity=d_valid_empty,
        )
        result.runtime_history.append(runtime_selection)
        return result

    explicit_mapping = d_explicit_polygon_output_rows is not None
    if explicit_mapping != (d_explicit_polygon_active is not None):
        raise ValueError("explicit polygon output rows and activity must be provided together")
    if explicit_mapping:
        d_polygon_output_rows = cp.asarray(
            d_explicit_polygon_output_rows,
            dtype=cp.int32,
        )
        d_active_polygons = cp.asarray(
            d_explicit_polygon_active,
            dtype=cp.bool_,
        )
        if (
            int(d_polygon_output_rows.size) != polygon_capacity
            or int(d_active_polygons.size) != polygon_capacity
        ):
            raise ValueError("explicit polygon output mapping must match polygon capacity")
    else:
        d_polygon_offsets = cp.zeros(
            int(d_polys_per_row.size) + 1,
            dtype=cp.int64,
        )
        cp.cumsum(
            cp.asarray(d_polys_per_row, dtype=cp.int64),
            out=d_polygon_offsets[1:],
        )
        d_polygon_logical_count = d_polygon_offsets[-1:]
        d_polygon_slots = cp.arange(polygon_capacity, dtype=cp.int64)
        d_active_polygons = d_polygon_slots < d_polygon_logical_count[0]
        d_safe_polygon_slots = cp.minimum(
            d_polygon_slots,
            cp.maximum(d_polygon_logical_count[0] - 1, 0),
        )
        d_polygon_compact_rows = cp.searchsorted(
            d_polygon_offsets[1:],
            d_safe_polygon_slots,
            side="right",
        ).astype(cp.int32, copy=False)
        d_polygon_compact_rows = cp.minimum(
            d_polygon_compact_rows,
            max(int(d_polys_per_row.size) - 1, 0),
        )
        d_polygon_output_rows = d_compact_row_ids[d_polygon_compact_rows]
    d_multipolygon_family_mask = d_active_polygons & d_multipolygon_rows[d_polygon_output_rows]

    d_ring_offsets_by_polygon = cp.zeros(polygon_capacity + 1, dtype=cp.int64)
    cp.cumsum(
        cp.asarray(d_rings_per_poly, dtype=cp.int64),
        out=d_ring_offsets_by_polygon[1:],
    )
    d_ring_logical_count = d_ring_offsets_by_polygon[-1:]
    d_ring_slots = cp.arange(ring_capacity, dtype=cp.int64)
    d_active_rings = d_ring_slots < d_ring_logical_count[0]
    d_safe_ring_slots = cp.minimum(
        d_ring_slots,
        cp.maximum(d_ring_logical_count[0] - 1, 0),
    )
    d_ring_polygon_ids = cp.searchsorted(
        d_ring_offsets_by_polygon[1:],
        d_safe_ring_slots,
        side="right",
    ).astype(cp.int32, copy=False)
    d_ring_polygon_ids = cp.minimum(
        d_ring_polygon_ids,
        polygon_capacity - 1,
    )
    d_ring_local = d_safe_ring_slots - d_ring_offsets_by_polygon[d_ring_polygon_ids]
    d_ring_lookup = cp.asarray(d_poly_starts, dtype=cp.int64)[d_ring_polygon_ids] + d_ring_local
    d_ring_lookup = cp.minimum(
        d_ring_lookup,
        max(ring_capacity - 1, 0),
    )
    d_all_ring_order = cp.asarray(
        d_sorted_output_ids,
        dtype=cp.int32,
    )[d_ring_lookup]
    d_all_ring_order = cp.where(d_active_rings, d_all_ring_order, cp.int32(0))
    if d_sorted_output_edge_counts is None:
        if d_all_edge_counts is None:
            raise ValueError("polygon output requires source- or rowset-aligned edge counts")
        d_all_ring_coord_counts = cp.where(
            d_active_rings,
            cp.asarray(d_all_edge_counts, dtype=cp.int32)[d_all_ring_order] + 1,
            cp.int32(0),
        )
    else:
        d_rowset_edge_counts = cp.asarray(d_sorted_output_edge_counts, dtype=cp.int32)
        if int(d_rowset_edge_counts.size) != ring_capacity:
            raise ValueError("rowset-aligned edge counts must match ring capacity")
        d_all_ring_coord_counts = cp.where(
            d_active_rings,
            d_rowset_edge_counts[d_ring_lookup] + 1,
            cp.int32(0),
        )

    d_rings_per_output_row = cp.zeros(output_row_count, dtype=cp.int32)
    cp.add.at(
        d_rings_per_output_row,
        d_polygon_output_rows,
        cp.where(
            d_active_polygons,
            cp.asarray(d_rings_per_poly, dtype=cp.int32),
            cp.int32(0),
        ),
    )

    output_coord_capacity = int(d_all_x.size) if coord_capacity is None else int(coord_capacity)
    if output_coord_capacity < 0:
        raise ValueError("polygon output coordinate capacity must be nonnegative")

    def _family_ring_storage(d_family_ring_mask):
        selection = NativeDeviceSelection.from_mask(d_family_ring_mask)
        d_ring_order = selection.gather_capacity(
            d_all_ring_order,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_coord_counts = selection.gather_capacity(
            d_all_ring_coord_counts,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ring_offsets = cp.zeros(ring_capacity + 1, dtype=cp.int32)
        cp.cumsum(d_coord_counts, out=d_ring_offsets[1:])
        d_x, d_y = _gather_coords_vectorised(
            d_all_x,
            d_all_y,
            d_all_coord_offsets,
            d_ring_order,
            d_coord_counts,
            total_capacity=output_coord_capacity,
        )
        return d_x, d_y, d_ring_offsets

    d_ring_polygon_rows = d_polygon_output_rows[d_ring_polygon_ids]
    d_polygon_ring_mask = d_active_rings & d_polygon_rows[d_ring_polygon_rows]
    d_multipolygon_ring_mask = d_active_rings & d_multipolygon_rows[d_ring_polygon_rows]
    d_poly_x, d_poly_y, d_poly_ring_offsets = _family_ring_storage(
        d_polygon_ring_mask,
    )
    d_mpoly_x, d_mpoly_y, d_mpoly_ring_offsets = _family_ring_storage(
        d_multipolygon_ring_mask,
    )

    d_poly_geometry_offsets = cp.zeros(output_row_count + 1, dtype=cp.int32)
    cp.cumsum(
        cp.where(
            d_polygon_rows,
            d_rings_per_output_row,
            cp.int32(0),
        ),
        out=d_poly_geometry_offsets[1:],
    )

    multipolygon_selection = NativeDeviceSelection.from_mask(
        d_multipolygon_family_mask,
    )
    d_mpoly_part_ring_counts = multipolygon_selection.gather_capacity(
        cp.asarray(d_rings_per_poly, dtype=cp.int32),
        fill_value=0,
    ).astype(cp.int32, copy=False)
    d_mpoly_part_offsets = cp.zeros(polygon_capacity + 1, dtype=cp.int32)
    cp.cumsum(d_mpoly_part_ring_counts, out=d_mpoly_part_offsets[1:])
    d_mpoly_geometry_offsets = cp.zeros(output_row_count + 1, dtype=cp.int32)
    cp.cumsum(
        cp.where(
            d_multipolygon_rows,
            d_full_polys_per_row,
            cp.int32(0),
        ),
        out=d_mpoly_geometry_offsets[1:],
    )

    device_families = {
        GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=d_poly_x,
            y=d_poly_y,
            geometry_offsets=d_poly_geometry_offsets,
            empty_mask=~d_nonempty_polygon_rows,
            ring_offsets=d_poly_ring_offsets,
            bounds=None,
        ),
        GeometryFamily.MULTIPOLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            x=d_mpoly_x,
            y=d_mpoly_y,
            geometry_offsets=d_mpoly_geometry_offsets,
            empty_mask=~d_multipolygon_rows,
            part_offsets=d_mpoly_part_offsets,
            ring_offsets=d_mpoly_ring_offsets,
            bounds=None,
        ),
    }
    d_tags = cp.where(
        d_polygon_rows,
        cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON]),
        cp.where(
            d_multipolygon_rows,
            cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]),
            cp.int8(-1),
        ),
    )
    d_family_row_offsets = cp.where(
        d_validity,
        cp.arange(output_row_count, dtype=cp.int32),
        cp.int32(-1),
    )
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=output_row_count,
        tags=d_tags,
        validity=d_validity,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    result.device_state.trusted_unique_family_rows = True
    result.runtime_history.append(runtime_selection)
    result._cached_is_valid_mask = None
    return result


def classify_grouped_polygonal_complement_parts_gpu(
    left: OwnedGeometryArray,
    right_parts: OwnedGeometryArray,
    grouped,
):
    """Return a device mask for right parts properly contained by their left row."""
    if cp is None or not getattr(grouped, "is_device", False):
        return None
    row_count = int(left.row_count)
    part_count = int(right_parts.row_count)
    resolved_group_count = int(
        getattr(
            grouped,
            "resolved_group_count",
            getattr(grouped, "group_count", -1),
        )
    )
    if row_count <= 0 or part_count <= 0 or row_count != resolved_group_count:
        return None

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right_parts._ensure_device_state(preserve_indexed_view=True)
    if left_state.families.get(GeometryFamily.POLYGON) is None:
        return None
    if right_state.families.get(GeometryFamily.POLYGON) is None:
        return None
    left_polygon = left_state.families[GeometryFamily.POLYGON]
    right_polygon = right_state.families[GeometryFamily.POLYGON]
    if (
        int(left_polygon.geometry_offsets.size) <= 1
        or int(right_polygon.geometry_offsets.size) <= 1
    ):
        return cp.zeros(part_count, dtype=cp.bool_)
    d_part_group_codes = cp.asarray(grouped.group_codes, dtype=cp.int32)
    if int(d_part_group_codes.size) != part_count:
        return None
    selection = getattr(grouped, "selection", None)
    d_part_active = (
        cp.asarray(selection.active_capacity_mask(), dtype=cp.bool_)
        if selection is not None
        else cp.ones(part_count, dtype=cp.bool_)
    )
    d_safe_part_group_codes = cp.where(
        d_part_active,
        d_part_group_codes,
        cp.int32(0),
    )

    from vibespatial.predicates.binary import binary_predicate_expression

    pair_left = left._device_indexed_take(
        d_safe_part_group_codes.astype(cp.int64, copy=False),
    )
    proper = binary_predicate_expression(
        "contains_properly",
        pair_left,
        right_parts,
        dispatch_mode=ExecutionMode.GPU,
        operation="overlay.grouped_difference.polygonal_complement_admission",
    )
    if proper is None:
        return None
    d_part_supported = cp.asarray(proper.values, dtype=cp.bool_)
    if int(d_part_supported.size) != part_count:
        return None
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_family_rows = cp.asarray(left_state.family_row_offsets, dtype=cp.int64)
    d_right_family_rows = cp.asarray(right_state.family_row_offsets, dtype=cp.int64)
    d_part_supported &= d_part_active & (
        cp.asarray(left_state.validity, dtype=cp.bool_)[d_safe_part_group_codes]
        & cp.asarray(right_state.validity, dtype=cp.bool_)
        & (cp.asarray(left_state.tags, dtype=cp.int8)[d_safe_part_group_codes] == polygon_tag)
        & (cp.asarray(right_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_left_family_rows[d_safe_part_group_codes] >= 0)
        & (d_right_family_rows >= 0)
    )
    return d_part_supported


def classify_grouped_polygonal_complement_groups_gpu(
    left: OwnedGeometryArray,
    right_parts: OwnedGeometryArray,
    grouped,
):
    """Return a device mask for groups admitted by complement assembly."""
    d_part_supported = classify_grouped_polygonal_complement_parts_gpu(
        left,
        right_parts,
        grouped,
    )
    if d_part_supported is None:
        return None
    row_count = int(left.row_count)
    part_count = int(right_parts.row_count)
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right_parts._ensure_device_state(preserve_indexed_view=True)
    d_part_group_codes = cp.asarray(grouped.group_codes, dtype=cp.int32)
    selection = getattr(grouped, "selection", None)
    d_part_active = (
        cp.asarray(selection.active_capacity_mask(), dtype=cp.bool_)
        if selection is not None
        else cp.ones(part_count, dtype=cp.bool_)
    )
    d_safe_part_group_codes = cp.where(
        d_part_active,
        d_part_group_codes,
        cp.int32(0),
    )

    d_group_part_counts = cp.zeros(row_count, dtype=cp.int32)
    d_group_supported_counts = cp.zeros(row_count, dtype=cp.int32)
    d_group_structural_counts = cp.zeros(row_count, dtype=cp.int32)
    cp.add.at(
        d_group_part_counts,
        d_safe_part_group_codes,
        d_part_active.astype(cp.int32, copy=False),
    )
    cp.add.at(
        d_group_supported_counts,
        d_safe_part_group_codes,
        d_part_supported.astype(cp.int32, copy=False),
    )
    left_polygon = left_state.families[GeometryFamily.POLYGON]
    right_polygon = right_state.families[GeometryFamily.POLYGON]
    d_left_validity = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_validity = cp.asarray(right_state.validity, dtype=cp.bool_)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_family_rows = cp.asarray(
        left_state.family_row_offsets,
        dtype=cp.int64,
    )
    d_right_family_rows = cp.asarray(
        right_state.family_row_offsets,
        dtype=cp.int64,
    )
    d_left_polygon_rows = (
        d_left_validity
        & (cp.asarray(left_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_left_family_rows >= 0)
    )
    d_right_polygon_rows = (
        d_right_validity
        & (cp.asarray(right_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_right_family_rows >= 0)
    )
    d_left_rows = cp.where(
        d_left_polygon_rows,
        d_left_family_rows,
        cp.int64(0),
    )
    d_right_rows = cp.where(
        d_right_polygon_rows,
        d_right_family_rows,
        cp.int64(0),
    )
    d_left_geometry_offsets = cp.asarray(
        left_polygon.geometry_offsets,
        dtype=cp.int32,
    )
    d_right_geometry_offsets = cp.asarray(
        right_polygon.geometry_offsets,
        dtype=cp.int32,
    )
    d_left_ring_counts = (
        d_left_geometry_offsets[d_left_rows + 1] - d_left_geometry_offsets[d_left_rows]
    )
    d_right_ring_counts = (
        d_right_geometry_offsets[d_right_rows + 1] - d_right_geometry_offsets[d_right_rows]
    )
    d_left_ring_counts = cp.where(
        d_left_polygon_rows,
        d_left_ring_counts,
        cp.int32(0),
    )
    d_right_ring_counts = cp.where(
        d_right_polygon_rows,
        d_right_ring_counts,
        cp.int32(0),
    )
    cp.add.at(
        d_group_structural_counts,
        d_safe_part_group_codes,
        (d_part_active & (d_right_ring_counts > 0)).astype(
            cp.int32,
            copy=False,
        ),
    )
    return (
        (d_group_part_counts > 0)
        & (d_group_supported_counts == d_group_part_counts)
        & (d_group_structural_counts == d_group_part_counts)
        & (d_left_ring_counts > 0)
    )


def assemble_grouped_polygonal_complement_gpu(
    left: OwnedGeometryArray,
    right_parts: OwnedGeometryArray,
    grouped,
    *,
    support_mask=None,
    right_ring_capacity: int | None = None,
    right_coord_capacity: int | None = None,
) -> OwnedGeometryArray | None:
    """Assemble ``left - grouped_union(right_parts)`` from polygon rings.

    Physical shape: one logical Polygon row on the left per group, a
    row-indirected Polygon-part capacity on the right, and a device
    ``NativeGrouped`` carrier mapping those parts back to left rows. Every right
    exterior is a subtraction boundary. Every right interior is retained as an
    island, and nested right parts are attached to the nearest containing island
    so valid MultiPolygon nesting survives without rowwise constructive fallback.
    """
    if cp is None or not getattr(grouped, "is_device", False):
        return None
    if left.row_count <= 0 or right_parts.row_count <= 0:
        return None
    resolved_group_count = int(
        getattr(
            grouped,
            "resolved_group_count",
            getattr(grouped, "group_count", -1),
        )
    )
    if left.row_count != resolved_group_count:
        return None
    selection = getattr(grouped, "selection", None)
    if selection is None and getattr(grouped, "all_groups_observed", None) is not True:
        return None
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right_parts._ensure_device_state(preserve_indexed_view=True)
    left_polygon = left_state.families.get(GeometryFamily.POLYGON)
    right_polygon = right_state.families.get(GeometryFamily.POLYGON)
    if left_polygon is None or right_polygon is None:
        return None
    if left_polygon.ring_offsets is None or right_polygon.ring_offsets is None:
        return None

    row_count = int(left.row_count)
    part_count = int(right_parts.row_count)
    d_part_group_codes = cp.asarray(grouped.group_codes, dtype=cp.int32)
    if int(d_part_group_codes.size) != part_count:
        return None
    if selection is None:
        d_part_active = cp.ones(part_count, dtype=cp.bool_)
        d_group_offsets = cp.asarray(grouped.group_offsets, dtype=cp.int32)
        d_group_ids = cp.asarray(grouped.group_ids, dtype=cp.int32)
        d_sorted_part_rows = cp.asarray(grouped.sorted_order, dtype=cp.int64)
        if (
            int(d_group_offsets.size) != row_count + 1
            or int(d_group_ids.size) != row_count
            or int(d_sorted_part_rows.size) != part_count
        ):
            return None
    else:
        d_part_active = cp.asarray(
            selection.active_capacity_mask(),
            dtype=cp.bool_,
        )
        d_sort_codes = cp.where(
            d_part_active,
            d_part_group_codes,
            cp.int32(row_count),
        )
        d_sorted_part_rows = cp.argsort(d_sort_codes).astype(
            cp.int64,
            copy=False,
        )
        d_group_counts = cp.bincount(
            cp.where(d_part_active, d_part_group_codes, cp.int32(row_count)),
            weights=d_part_active.astype(cp.int32, copy=False),
            minlength=row_count + 1,
        )[:row_count].astype(cp.int32, copy=False)
        d_group_offsets = cp.zeros(row_count + 1, dtype=cp.int32)
        cp.cumsum(d_group_counts, out=d_group_offsets[1:])
        d_group_ids = cp.arange(row_count, dtype=cp.int32)

    d_left_validity = cp.asarray(left_state.validity, dtype=cp.bool_)
    d_right_validity = cp.asarray(right_state.validity, dtype=cp.bool_)
    polygon_tag = cp.int8(FAMILY_TAGS[GeometryFamily.POLYGON])
    d_left_family_rows = cp.asarray(
        left_state.family_row_offsets,
        dtype=cp.int64,
    )
    d_right_family_rows = cp.asarray(
        right_state.family_row_offsets,
        dtype=cp.int64,
    )
    d_left_polygon_rows = (
        d_left_validity
        & (cp.asarray(left_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_left_family_rows >= 0)
    )
    d_right_polygon_rows = (
        d_right_validity
        & (cp.asarray(right_state.tags, dtype=cp.int8) == polygon_tag)
        & (d_right_family_rows >= 0)
    )
    d_left_rows = cp.where(
        d_left_polygon_rows,
        d_left_family_rows,
        cp.int64(0),
    )
    d_right_rows = cp.where(
        d_right_polygon_rows,
        d_right_family_rows,
        cp.int64(0),
    )
    d_left_geometry_offsets = cp.asarray(
        left_polygon.geometry_offsets,
        dtype=cp.int32,
    )
    d_right_geometry_offsets = cp.asarray(
        right_polygon.geometry_offsets,
        dtype=cp.int32,
    )
    d_left_ring_offsets = cp.asarray(left_polygon.ring_offsets, dtype=cp.int32)
    d_right_ring_offsets = cp.asarray(right_polygon.ring_offsets, dtype=cp.int32)
    d_left_ring_starts = d_left_geometry_offsets[d_left_rows]
    d_left_ring_counts = (d_left_geometry_offsets[d_left_rows + 1] - d_left_ring_starts).astype(
        cp.int32, copy=False
    )
    d_left_ring_counts = cp.where(
        d_left_polygon_rows,
        d_left_ring_counts,
        cp.int32(0),
    )
    d_sorted_right_rows = d_right_rows[d_sorted_part_rows]
    d_sorted_part_active = (d_part_active & d_right_polygon_rows)[d_sorted_part_rows]
    d_part_ring_starts = d_right_geometry_offsets[d_sorted_right_rows]
    d_part_ring_counts = (
        d_right_geometry_offsets[d_sorted_right_rows + 1] - d_part_ring_starts
    ).astype(cp.int32, copy=False)
    d_part_ring_counts = cp.where(
        d_sorted_part_active,
        d_part_ring_counts,
        cp.int32(0),
    )
    if support_mask is None:
        support_mask = classify_grouped_polygonal_complement_groups_gpu(
            left,
            right_parts,
            grouped,
        )
    if support_mask is None:
        return None
    d_support_mask = cp.asarray(support_mask, dtype=cp.bool_)
    if int(d_support_mask.size) != row_count:
        return None

    d_sorted_part_positions = cp.arange(part_count, dtype=cp.int32)
    d_part_logical_count = d_group_offsets[-1]
    d_safe_sorted_part_positions = cp.minimum(
        d_sorted_part_positions,
        cp.maximum(d_part_logical_count - 1, 0),
    )
    d_part_group_rows = cp.searchsorted(
        d_group_offsets[1:],
        d_safe_sorted_part_positions,
        side="right",
    ).astype(cp.int32, copy=False)
    d_part_group_rows = cp.minimum(d_part_group_rows, row_count - 1)
    d_part_interior_counts = cp.maximum(d_part_ring_counts - 1, 0)
    d_part_interior_offsets = cp.zeros(part_count + 1, dtype=cp.int32)
    cp.cumsum(d_part_interior_counts, out=d_part_interior_offsets[1:])

    physical_left_ring_capacity = max(int(d_left_ring_offsets.size) - 1, 0)
    physical_left_coord_capacity = int(left_polygon.x.size)
    if left.is_indexed_view and left_state.trusted_unique_family_rows is not True:
        fixed_size = getattr(left_polygon, "fixed_size", None)
        fixed_ring_count = getattr(
            fixed_size,
            "first_level_count_per_row",
            None,
        )
        fixed_coord_count = getattr(fixed_size, "coord_count_per_row", None)
        if fixed_ring_count is None and left_polygon.dense_single_ring_width is not None:
            fixed_ring_count = 1
            fixed_coord_count = int(left_polygon.dense_single_ring_width)
        if fixed_ring_count is None or fixed_coord_count is None:
            return None
        left_ring_capacity = row_count * int(fixed_ring_count)
        left_coord_capacity = row_count * int(fixed_coord_count)
    else:
        left_ring_capacity = physical_left_ring_capacity
        left_coord_capacity = physical_left_coord_capacity
    physical_right_ring_capacity = max(int(d_right_ring_offsets.size) - 1, 0)
    right_ring_capacity = (
        physical_right_ring_capacity if right_ring_capacity is None else int(right_ring_capacity)
    )
    right_coord_capacity = (
        int(right_polygon.x.size) if right_coord_capacity is None else int(right_coord_capacity)
    )
    if right_ring_capacity < 0 or right_coord_capacity < 0:
        return None
    interior_capacity = right_ring_capacity

    if interior_capacity > 0:
        d_interior_positions = cp.arange(interior_capacity, dtype=cp.int32)
        d_interior_logical_count = d_part_interior_offsets[-1]
        d_interior_active = d_interior_positions < d_interior_logical_count
        d_safe_interior_positions = cp.minimum(
            d_interior_positions,
            cp.maximum(d_interior_logical_count - 1, 0),
        )
        d_interior_part_rows = cp.searchsorted(
            d_part_interior_offsets[1:],
            d_safe_interior_positions,
            side="right",
        ).astype(cp.int32, copy=False)
        d_interior_part_rows = cp.minimum(d_interior_part_rows, part_count - 1)
        d_interior_local = d_safe_interior_positions - d_part_interior_offsets[d_interior_part_rows]
        d_interior_ring_ids = (
            d_part_ring_starts[d_interior_part_rows] + 1 + d_interior_local
        ).astype(cp.int32, copy=False)
        d_interior_ring_ids = cp.where(
            d_interior_active,
            d_interior_ring_ids,
            cp.int32(0),
        )
        d_interior_group_rows = d_part_group_rows[d_interior_part_rows]
        d_group_interior_counts = cp.bincount(
            d_interior_group_rows,
            weights=d_interior_active.astype(cp.int32, copy=False),
            minlength=row_count,
        ).astype(cp.int32, copy=False)
    else:
        d_interior_positions = cp.empty(0, dtype=cp.int32)
        d_interior_active = cp.empty(0, dtype=cp.bool_)
        d_interior_part_rows = cp.empty(0, dtype=cp.int32)
        d_interior_ring_ids = cp.empty(0, dtype=cp.int32)
        d_interior_group_rows = cp.empty(0, dtype=cp.int32)
        d_group_interior_counts = cp.zeros(row_count, dtype=cp.int32)

    d_group_interior_offsets = cp.zeros(row_count + 1, dtype=cp.int32)
    cp.cumsum(d_group_interior_counts, out=d_group_interior_offsets[1:])
    d_parent_interior = cp.full(part_count, -1, dtype=cp.int32)
    if interior_capacity > 0:
        from vibespatial.overlay.gpu import _overlay_face_assembly_kernels

        runtime = get_cuda_runtime()
        kernels = _overlay_face_assembly_kernels()
        ptr = runtime.pointer
        d_interior_abs_area = cp.empty(interior_capacity, dtype=cp.float64)
        d_interior_bounds = cp.empty((interior_capacity, 4), dtype=cp.float64)
        metrics = kernels["grouped_complement_hole_metrics"]
        grid, block = runtime.launch_config(metrics, interior_capacity)
        runtime.launch(
            metrics,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(cp.asarray(right_polygon.x, dtype=cp.float64)),
                    ptr(cp.asarray(right_polygon.y, dtype=cp.float64)),
                    ptr(d_right_ring_offsets),
                    ptr(d_interior_ring_ids),
                    ptr(d_interior_abs_area),
                    ptr(d_interior_bounds),
                    interior_capacity,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        assign = kernels["assign_grouped_complement_exterior_parents"]
        grid, block = runtime.launch_config(assign, part_count)
        runtime.launch(
            assign,
            grid=grid,
            block=block,
            params=(
                (
                    ptr(cp.asarray(right_polygon.x, dtype=cp.float64)),
                    ptr(cp.asarray(right_polygon.y, dtype=cp.float64)),
                    ptr(d_right_ring_offsets),
                    ptr(d_part_ring_starts.astype(cp.int32, copy=False)),
                    ptr(d_part_group_rows),
                    ptr(d_interior_ring_ids),
                    ptr(d_group_interior_offsets),
                    ptr(d_interior_abs_area),
                    ptr(d_interior_bounds),
                    ptr(d_parent_interior),
                    part_count,
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
                ),
            ),
        )

    d_parts_per_group = (1 + d_group_interior_counts).astype(cp.int32, copy=False)
    d_geometry_offsets = cp.zeros(row_count + 1, dtype=cp.int32)
    cp.cumsum(d_parts_per_group, out=d_geometry_offsets[1:])
    total_part_capacity = row_count + interior_capacity
    d_main_part_rows = d_geometry_offsets[:-1]
    if interior_capacity > 0:
        d_interior_output_parts = (
            d_main_part_rows[d_interior_group_rows]
            + 1
            + cp.minimum(
                d_interior_positions,
                cp.maximum(d_part_interior_offsets[-1] - 1, 0),
            )
            - d_group_interior_offsets[d_interior_group_rows]
        ).astype(cp.int32, copy=False)
        d_interior_output_parts = cp.where(
            d_interior_active,
            d_interior_output_parts,
            cp.int32(0),
        )
    else:
        d_interior_output_parts = cp.empty(0, dtype=cp.int32)

    d_left_ring_positions = cp.arange(left_ring_capacity, dtype=cp.int32)
    d_left_logical_offsets = cp.zeros(row_count + 1, dtype=cp.int32)
    cp.cumsum(d_left_ring_counts, out=d_left_logical_offsets[1:])
    d_left_ring_active = d_left_ring_positions < d_left_logical_offsets[-1]
    d_safe_left_ring_positions = cp.minimum(
        d_left_ring_positions,
        cp.maximum(d_left_logical_offsets[-1] - 1, 0),
    )
    d_left_ring_groups = cp.searchsorted(
        d_left_logical_offsets[1:],
        d_safe_left_ring_positions,
        side="right",
    ).astype(cp.int32, copy=False)
    d_left_ring_groups = cp.minimum(d_left_ring_groups, row_count - 1)
    d_left_local_rings = d_safe_left_ring_positions - d_left_logical_offsets[d_left_ring_groups]
    d_left_source_rings = (d_left_ring_starts[d_left_ring_groups] + d_left_local_rings).astype(
        cp.int32, copy=False
    )
    d_left_source_rings = cp.where(
        d_left_ring_active,
        d_left_source_rings,
        cp.int32(0),
    )
    d_left_output_parts = d_main_part_rows[d_left_ring_groups]
    d_left_ring_kind = cp.where(
        d_left_local_rings == 0,
        cp.zeros((), dtype=cp.int32),
        cp.ones((), dtype=cp.int32),
    )

    d_right_ring_positions = cp.arange(right_ring_capacity, dtype=cp.int32)
    d_part_logical_ring_offsets = cp.zeros(part_count + 1, dtype=cp.int32)
    cp.cumsum(d_part_ring_counts, out=d_part_logical_ring_offsets[1:])
    d_right_ring_active = d_right_ring_positions < d_part_logical_ring_offsets[-1]
    d_safe_right_ring_positions = cp.minimum(
        d_right_ring_positions,
        cp.maximum(d_part_logical_ring_offsets[-1] - 1, 0),
    )
    d_right_ring_parts = cp.searchsorted(
        d_part_logical_ring_offsets[1:],
        d_safe_right_ring_positions,
        side="right",
    ).astype(cp.int32, copy=False)
    d_right_ring_parts = cp.minimum(d_right_ring_parts, part_count - 1)
    d_right_local_rings = (
        d_safe_right_ring_positions - d_part_logical_ring_offsets[d_right_ring_parts]
    )
    d_right_source_rings = (d_part_ring_starts[d_right_ring_parts] + d_right_local_rings).astype(
        cp.int32, copy=False
    )
    d_right_source_rings = cp.where(
        d_right_ring_active,
        d_right_source_rings,
        cp.int32(0),
    )
    d_right_ring_groups = d_part_group_rows[d_right_ring_parts]
    d_right_is_exterior = d_right_local_rings == 0
    if interior_capacity > 0:
        d_safe_interior_positions = cp.maximum(
            d_part_interior_offsets[d_right_ring_parts] + d_right_local_rings - 1,
            0,
        ).astype(cp.int32, copy=False)
        d_safe_interior_positions = cp.minimum(
            d_safe_interior_positions,
            interior_capacity - 1,
        )
        d_safe_parent = cp.maximum(d_parent_interior[d_right_ring_parts], 0)
        d_safe_parent = cp.minimum(d_safe_parent, interior_capacity - 1)
        d_exterior_output_parts = cp.where(
            d_parent_interior[d_right_ring_parts] >= 0,
            d_interior_output_parts[d_safe_parent],
            d_main_part_rows[d_right_ring_groups],
        )
        d_right_output_parts = cp.where(
            d_right_is_exterior,
            d_exterior_output_parts,
            d_interior_output_parts[d_safe_interior_positions],
        ).astype(cp.int32, copy=False)
    else:
        d_right_output_parts = d_main_part_rows[d_right_ring_groups]
    d_right_output_parts = cp.where(
        d_right_ring_active,
        d_right_output_parts,
        cp.int32(0),
    )
    d_right_ring_kind = cp.where(
        d_right_is_exterior,
        cp.full((), 2, dtype=cp.int32),
        cp.zeros((), dtype=cp.int32),
    )

    total_ring_capacity = left_ring_capacity + right_ring_capacity
    d_ring_active = cp.concatenate((d_left_ring_active, d_right_ring_active))
    d_ring_side = cp.concatenate(
        (
            cp.zeros(left_ring_capacity, dtype=cp.bool_),
            cp.ones(right_ring_capacity, dtype=cp.bool_),
        ),
    )
    d_ring_output_parts = cp.concatenate(
        (d_left_output_parts, d_right_output_parts),
    )
    d_ring_kind = cp.concatenate((d_left_ring_kind, d_right_ring_kind))
    d_left_sources = cp.concatenate(
        (d_left_source_rings, cp.zeros(right_ring_capacity, dtype=cp.int32)),
    )
    d_right_sources = cp.concatenate(
        (cp.zeros(left_ring_capacity, dtype=cp.int32), d_right_source_rings),
    )
    d_ring_output_sort_key = cp.where(
        d_ring_active,
        d_ring_output_parts,
        cp.int32(total_part_capacity),
    )
    d_stable_ring_order = cp.arange(total_ring_capacity, dtype=cp.int32)
    d_ring_order = cp.arange(total_ring_capacity, dtype=cp.int32)
    for key in (d_stable_ring_order, d_ring_kind, d_ring_output_sort_key):
        d_ring_order = _stable_radix_order_pass(d_ring_order, key)
    d_ring_active = d_ring_active[d_ring_order]
    d_ring_side = d_ring_side[d_ring_order]
    d_left_sources = d_left_sources[d_ring_order]
    d_right_sources = d_right_sources[d_ring_order]
    d_ring_output_parts = d_ring_output_parts[d_ring_order]

    d_rings_per_part = cp.bincount(
        cp.where(d_ring_active, d_ring_output_parts, cp.int32(0)),
        weights=d_ring_active.astype(cp.int32, copy=False),
        minlength=total_part_capacity,
    )[:total_part_capacity].astype(cp.int32, copy=False)
    d_part_offsets = cp.zeros(total_part_capacity + 1, dtype=cp.int32)
    cp.cumsum(d_rings_per_part, out=d_part_offsets[1:])
    d_left_lengths = d_left_ring_offsets[1:] - d_left_ring_offsets[:-1]
    d_right_lengths = d_right_ring_offsets[1:] - d_right_ring_offsets[:-1]
    d_ring_lengths = cp.where(
        d_ring_active,
        cp.where(
            d_ring_side,
            d_right_lengths[d_right_sources],
            d_left_lengths[d_left_sources],
        ),
        cp.int32(0),
    ).astype(cp.int32, copy=False)
    d_output_ring_offsets = cp.zeros(total_ring_capacity + 1, dtype=cp.int32)
    cp.cumsum(d_ring_lengths, out=d_output_ring_offsets[1:])

    total_coord_capacity = left_coord_capacity + right_coord_capacity
    d_coord_positions = cp.arange(total_coord_capacity, dtype=cp.int64)
    d_coord_active = d_coord_positions < d_output_ring_offsets[-1]
    d_safe_coord_positions = cp.minimum(
        d_coord_positions,
        cp.maximum(d_output_ring_offsets[-1] - 1, 0),
    )
    d_coord_rings = cp.searchsorted(
        d_output_ring_offsets[1:],
        d_safe_coord_positions,
        side="right",
    ).astype(cp.int32, copy=False)
    d_coord_rings = cp.minimum(d_coord_rings, total_ring_capacity - 1)
    d_coord_local = d_safe_coord_positions - d_output_ring_offsets[d_coord_rings]
    d_coord_side = d_ring_side[d_coord_rings]
    d_coord_left_rings = d_left_sources[d_coord_rings]
    d_coord_right_rings = d_right_sources[d_coord_rings]
    d_left_coord_rows = (d_left_ring_offsets[d_coord_left_rings] + d_coord_local).astype(
        cp.int64, copy=False
    )
    d_right_coord_rows = (d_right_ring_offsets[d_coord_right_rings + 1] - 1 - d_coord_local).astype(
        cp.int64, copy=False
    )
    d_x = cp.zeros(total_coord_capacity, dtype=cp.float64)
    d_y = cp.zeros(total_coord_capacity, dtype=cp.float64)
    d_left_coord_mask = d_coord_active & ~d_coord_side
    d_x[d_left_coord_mask] = cp.asarray(left_polygon.x, dtype=cp.float64)[
        d_left_coord_rows[d_left_coord_mask]
    ]
    d_y[d_left_coord_mask] = cp.asarray(left_polygon.y, dtype=cp.float64)[
        d_left_coord_rows[d_left_coord_mask]
    ]
    d_right_coord_mask = d_coord_active & d_coord_side
    d_x[d_right_coord_mask] = cp.asarray(right_polygon.x, dtype=cp.float64)[
        d_right_coord_rows[d_right_coord_mask]
    ]
    d_y[d_right_coord_mask] = cp.asarray(right_polygon.y, dtype=cp.float64)[
        d_right_coord_rows[d_right_coord_mask]
    ]

    result = _build_device_resident_polygon_output(
        d_all_x=d_x,
        d_all_y=d_y,
        d_all_coord_offsets=d_output_ring_offsets[:-1],
        d_all_edge_counts=d_ring_lengths - 1,
        d_sorted_output_ids=cp.arange(total_ring_capacity, dtype=cp.int32),
        d_rings_per_poly=d_rings_per_part,
        d_polys_per_row=d_parts_per_group,
        d_poly_starts=d_part_offsets[:-1],
        d_output_source_rows=cp.arange(row_count, dtype=cp.int32),
        n_output_rows=row_count,
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason=(
                "grouped polygonal complement assembled from part, ring, and coordinate carriers"
            ),
        ),
        preserve_row_count=row_count,
    )
    if result.device_state is not None:
        cached_bounds = getattr(left_state, "row_bounds", None)
        if cached_bounds is not None and int(cached_bounds.shape[0]) == row_count:
            result.device_state.row_bounds = cached_bounds
    if result.device_state is not None:
        result_state = result.device_state
        d_result_validity = cp.asarray(result_state.validity, dtype=cp.bool_)
        d_result_validity &= d_support_mask
        result_state.validity = d_result_validity
        result_state.tags = cp.where(
            d_result_validity,
            cp.asarray(result_state.tags, dtype=cp.int8),
            cp.int8(-1),
        )
        result_state.family_row_offsets = cp.where(
            d_result_validity,
            cp.asarray(result_state.family_row_offsets, dtype=cp.int32),
            cp.int32(-1),
        )
        for device_buffer in result_state.families.values():
            if int(device_buffer.empty_mask.size) == row_count:
                device_buffer.empty_mask = (
                    cp.asarray(
                        device_buffer.empty_mask,
                        dtype=cp.bool_,
                    )
                    | ~d_support_mask
                )
        result_state.trusted_all_valid = None
        result_state.trusted_all_non_empty = None
    result._cached_is_valid_mask = None
    return result


def _gather_coords_vectorised(
    d_all_x: cp.ndarray,
    d_all_y: cp.ndarray,
    d_all_coord_offsets: cp.ndarray,
    ring_order: cp.ndarray,
    coord_counts: cp.ndarray,
    *,
    total_capacity: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Vectorised coordinate gather: build flat index array without Python loops.

    For each ring in ring_order, gathers coord_counts[i] consecutive
    coordinates starting at d_all_coord_offsets[ring_order[i]].
    """
    if ring_order.size == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    total_coords = int(total_capacity)
    if total_coords == 0:
        return cp.empty(0, dtype=cp.float64), cp.empty(0, dtype=cp.float64)
    if int(d_all_x.size) == 0:
        return (
            cp.zeros(total_coords, dtype=cp.float64),
            cp.zeros(total_coords, dtype=cp.float64),
        )
    ring_starts = d_all_coord_offsets[ring_order].astype(cp.int64, copy=False)
    # Expand: for each coordinate slot, compute source index.
    # slot_ring[i] = which ring does coordinate slot i belong to?
    ring_offsets = cp.zeros(int(ring_order.size) + 1, dtype=cp.int64)
    cp.cumsum(coord_counts.astype(cp.int64, copy=False), out=ring_offsets[1:])
    slots = cp.arange(total_coords, dtype=cp.int64)
    d_logical_total = ring_offsets[-1]
    slots = cp.minimum(slots, cp.maximum(d_logical_total - 1, 0))
    slot_ring = _expand_by_counts(
        coord_counts.astype(cp.int32, copy=False),
        total=total_coords,
        slots=slots,
    )
    slot_ring = cp.minimum(slot_ring, int(ring_order.size) - 1)
    slot_local = slots - ring_offsets[slot_ring]
    d_gather = cp.minimum(
        ring_starts[slot_ring] + slot_local,
        int(d_all_x.size) - 1,
    )
    d_active = cp.arange(total_coords, dtype=cp.int64) < d_logical_total
    return (
        cp.where(d_active, d_all_x[d_gather], cp.float64(0.0)),
        cp.where(d_active, d_all_y[d_gather], cp.float64(0.0)),
    )


def _expand_by_counts(
    counts: cp.ndarray,
    *,
    total: int,
    slots: cp.ndarray | None = None,
) -> cp.ndarray:
    """Return the repeated source index for each slot implied by ``counts``."""
    if counts.size == 0:
        return cp.empty(0, dtype=cp.int32)
    total = int(total)
    if total == 0:
        return cp.empty(0, dtype=cp.int32)
    offsets = cp.zeros(int(counts.size) + 1, dtype=cp.int64)
    cp.cumsum(counts.astype(cp.int64, copy=False), out=offsets[1:])
    if slots is None:
        slots = cp.arange(total, dtype=cp.int64)
    expanded = cp.searchsorted(offsets[1:], slots, side="right")
    return cp.minimum(expanded, int(counts.size) - 1).astype(
        cp.int32,
        copy=False,
    )


def _overlay_intersection_rectangles_gpu(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    requested: ExecutionMode,
) -> OwnedGeometryArray | None:
    if cp is None or left.row_count != right.row_count:
        return None
    left_bounds = _axis_aligned_box_bounds_device(left)
    right_bounds = _axis_aligned_box_bounds_device(right)
    if left_bounds is None:
        left_bounds = _axis_aligned_box_bounds(left)
    if right_bounds is None:
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
