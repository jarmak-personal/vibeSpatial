"""GPU-resident make_valid batch repair pipeline (ADR-0019 + ADR-0033).

Phase 16: Batch repolygonization. All invalid polygons are collected into one
contiguous batch and processed through the GPU repair pipeline together,
eliminating the per-polygon Python loop. shapely.polygonize and
shapely.make_valid are no longer used in the primary GPU path.

Pipeline stages (batched across invalid polygon rows):
  Phase A: capacity-backed ring closure, duplicate removal, and orientation
  Phase B: canonical OGC device validity rowset
  Phase C: shared overlay sweep, streamed split events, and complete-row paging
  Phase D: positive bounded-face polygonization and sparse repaired-row scatter
  Phase E: native area/lower-dimensional composition for terminal export

All kernels use fp64 compute (CONSTRUCTIVE class per ADR-0002).
Tier 1: NVRTC for geometry-specific work.
Tier 3a: CCCL for scan/sort/compact.
Tier 2: CuPy for element-wise ops.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from vibespatial.constructive.make_valid_gpu_kernels import (
    _REPAIR_KERNEL_NAMES,
    _REPAIR_KERNEL_SOURCE,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    exclusive_sum,
    segmented_reduce_sum,
)

request_warmup(
    [
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "select_i32",
        "select_i64",
        "radix_sort_i32_i32",
        "radix_sort_u64_i32",
        "segmented_reduce_sum_f64",
    ]
)
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    DeviceFixedGeometrySizeMetadata,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.crossover import estimate_segment_pair_work_from_owned  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup(
    [
        ("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES),
    ]
)


def _compile_repair_kernels():
    return compile_kernel_group("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES)


def _planned_make_valid_runtime_selection(
    *,
    kernel_name: str,
    owned: OwnedGeometryArray,
    selected_row_count: int,
    reason: str,
) -> RuntimeSelection:
    selection = plan_dispatch_selection(
        kernel_name=kernel_name,
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=selected_row_count,
        work_estimate=estimate_segment_pair_work_from_owned(
            owned,
            selected_row_count=selected_row_count,
            output_row_count=selected_row_count,
            primary_unit_name="make-valid-repair-segment-pair",
        ),
        requested_mode=ExecutionMode.GPU,
        gpu_available=cp is not None,
    )
    return replace(selection.runtime_selection, reason=reason)


# ---------------------------------------------------------------------------
# Phase B: Simple repair operations
# ---------------------------------------------------------------------------


def _gpu_close_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Close unclosed rings by appending first vertex. Returns new x, y, ring_offsets.

    Closure check via check_ring_closure NVRTC kernel (Tier 1, 1 thread/ring).
    Copy+append via close_rings NVRTC kernel.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- Step 1: NVRTC kernel checks which rings need closure (Tier 1) ---
    d_needs_close = cp.empty(ring_count, dtype=cp.int32)
    grid, block = runtime.launch_config(kernels["check_ring_closure"], ring_count)
    runtime.launch(
        kernels["check_ring_closure"],
        grid=grid,
        block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_close), ring_count),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- Step 2: Compute new ring sizes and offsets on device (Tier 2: CuPy) ---
    d_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_new_sizes = d_sizes + d_needs_close
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)

    # --- Step 3: Copy into a capacity-backed ring carrier (Tier 1) ---
    # Every ring can add at most one coordinate. The device offsets carry the
    # logical coordinate length, so allocation never needs a host total.
    coordinate_capacity = int(d_x.size) + ring_count
    d_out_x = cp.empty(coordinate_capacity, dtype=cp.float64)
    d_out_y = cp.empty(coordinate_capacity, dtype=cp.float64)

    grid, block = runtime.launch_config(kernels["close_rings"], ring_count)
    runtime.launch(
        kernels["close_rings"],
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_x),
                ptr(d_y),
                ptr(d_ring_offsets),
                ptr(d_needs_close),
                ptr(d_new_offsets),
                ptr(d_out_x),
                ptr(d_out_y),
                ring_count,
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
            ),
        ),
    )
    return d_out_x, d_out_y, d_new_offsets


def _gpu_remove_duplicate_vertices(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Remove consecutive duplicate vertices within each ring.

    Tier 1: flag_duplicate_vertices NVRTC kernel (1 thread/vertex).
    Tier 2: CuPy searchsorted for vertex-to-ring mapping, fancy indexing for compaction.
    Tier 3a: CCCL scan plus segmented_reduce_sum for capacity-backed compaction
    for per-ring kept counts.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    vertex_count = int(d_x.shape[0])
    if vertex_count == 0:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident vertex-to-ring mapping (Tier 2: CuPy searchsorted) ---
    d_vertex_ids = cp.arange(vertex_count, dtype=cp.int32)
    d_vertex_ring_ids = cp.searchsorted(d_ring_offsets[1:], d_vertex_ids, side="right").astype(
        cp.int32
    )

    # --- Flag duplicates via NVRTC kernel (Tier 1) ---
    d_keep = cp.empty(vertex_count, dtype=cp.uint8)
    grid, block = runtime.launch_config(kernels["flag_duplicate_vertices"], vertex_count)
    runtime.launch(
        kernels["flag_duplicate_vertices"],
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_x),
                ptr(d_y),
                ptr(d_ring_offsets),
                ptr(d_vertex_ring_ids),
                ptr(d_keep),
                ring_count,
                vertex_count,
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
    # --- Capacity-backed compaction (Tier 3a scan + Tier 1 scatter) ---
    # Retain coordinate capacity and let rebuilt ring offsets define the active
    # prefix. This avoids a kept-count allocation fence and remains consumable
    # by downstream offset-shaped geometry kernels.
    d_keep_i32 = d_keep.astype(cp.int32)
    d_keep_positions = exclusive_sum(d_keep_i32, synchronize=False)
    new_x = cp.empty(vertex_count, dtype=cp.float64)
    new_y = cp.empty(vertex_count, dtype=cp.float64)
    scatter_kernel = kernels["scatter_kept_vertices"]
    scatter_grid, scatter_block = runtime.launch_config(
        scatter_kernel,
        vertex_count,
    )
    runtime.launch(
        scatter_kernel,
        grid=scatter_grid,
        block=scatter_block,
        params=(
            (
                ptr(d_x),
                ptr(d_y),
                ptr(d_keep),
                ptr(d_keep_positions),
                ptr(new_x),
                ptr(new_y),
                vertex_count,
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

    # --- GPU-resident ring offset rebuild ---
    # Per-ring kept count via segmented reduce (Tier 3a: CCCL)
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_keep_i32,
        d_starts,
        d_ends,
        num_segments=ring_count,
        synchronize=False,
    )
    d_new_sizes = seg_result.values.astype(cp.int32)
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)

    return new_x, new_y, d_new_offsets


def _gpu_fix_ring_orientation(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Fix ring orientation: exterior rings CCW (positive area), holes CW (negative area).

    Tier 1: compute_ring_shoelace NVRTC kernel for per-vertex cross products.
    Tier 3a: segmented_reduce_sum (CCCL) for per-ring signed area.
    Tier 2: CuPy element-wise for exterior/hole classification and reversal mask.
    Tier 1: reverse_ring_coords NVRTC kernel for coordinate reversal.
    All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    vertex_count = int(d_x.shape[0])
    if vertex_count < 3 or ring_count == 0:
        return d_x, d_y

    # --- Step 1: Compute per-vertex shoelace cross products (Tier 1: NVRTC) ---
    # compute_ring_shoelace writes x[v]*y[v+1] - x[v+1]*y[v] for each vertex.
    # Launch for vertex_count-1 to avoid out-of-bounds read on the last vertex.
    # Zero-init so last-vertex-per-ring contributions are automatically 0.
    d_cross = cp.zeros(vertex_count, dtype=cp.float64)
    safe_count = vertex_count - 1
    if safe_count > 0:
        grid, block = runtime.launch_config(kernels["compute_ring_shoelace"], safe_count)
        runtime.launch(
            kernels["compute_ring_shoelace"],
            grid=grid,
            block=block,
            params=(
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_cross), ring_count, safe_count),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    # Zero out cross products at ring boundaries to prevent cross-ring contamination
    d_last_verts = d_ring_offsets[1:] - 1
    d_cross[d_last_verts] = 0.0

    # --- Step 2: Per-ring signed area via segmented reduce (Tier 3a: CCCL) ---
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_cross,
        d_starts,
        d_ends,
        num_segments=ring_count,
        synchronize=False,
    )
    d_ring_areas = seg_result.values * 0.5

    # --- Step 3: Classify exterior vs hole and build reversal mask (Tier 2: CuPy) ---
    d_ring_ids = cp.arange(ring_count, dtype=cp.int32)
    d_poly_of_ring = cp.searchsorted(d_geom_offsets[1:], d_ring_ids, side="right").astype(cp.int32)
    d_first_ring_of_poly = d_geom_offsets[d_poly_of_ring]
    d_is_exterior = d_ring_ids == d_first_ring_of_poly

    # Exterior should be CCW (positive area); hole should be CW (negative area)
    d_needs_reverse = (
        (d_is_exterior & (d_ring_areas < 0)) | (~d_is_exterior & (d_ring_areas > 0))
    ).astype(cp.uint8)

    # --- Step 4: Reverse only flagged rings (Tier 1: NVRTC) ---
    # The kernel already treats the mask as its admission carrier. Launching it
    # over ring rows is cheaper than synchronizing a device reduction merely to
    # discover that the mask is empty.
    grid, block = runtime.launch_config(kernels["reverse_ring_coords"], ring_count)
    runtime.launch(
        kernels["reverse_ring_coords"],
        grid=grid,
        block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_reverse), ring_count),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    # No explicit sync needed -- caller will either launch more same-stream
    # work or trigger implicit sync via CuPy/D2H read.
    return d_x, d_y


# ---------------------------------------------------------------------------
# Phase A + C: Self-intersection detection and splitting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase D + E: Re-polygonization and output assembly
# ---------------------------------------------------------------------------


def _repolygonize_owned_rows_via_overlay(
    source: OwnedGeometryArray,
) -> OwnedGeometryArray | None:
    """Polygonize invalid rows through the shared bounded overlay topology plan."""
    if source.row_count == 0 or source.device_state is None:
        return None

    from vibespatial.geometry.owned import build_empty_polygon_rows_device
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    # A whole-batch segment count is a conservative per-row span proof. It may
    # choose one complete logical row per topology page, but never requires a
    # row-shaped max-span reduction or host metadata export.
    max_segment_span = 0
    state = source._ensure_device_state(preserve_indexed_view=True)
    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        buffer = state.families.get(family)
        if buffer is None or buffer.ring_offsets is None:
            continue
        max_segment_span += max(
            int(buffer.x.size) - (int(buffer.ring_offsets.size) - 1),
            0,
        )
    if max_segment_span == 0:
        return None

    empty_right = build_empty_polygon_rows_device(source.row_count)
    plan = _build_overlay_execution_plan(
        source,
        empty_right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _use_same_row_fast_path=True,
        _same_row_single_group=source.row_count == 1,
        _same_row_span_summary=(
            max_segment_span,
            0,
            source.row_count - 1,
        ),
        _include_same_side_splits=True,
    )
    result, selected = _materialize_overlay_execution_plan(
        plan,
        operation="polygonize",
        requested=ExecutionMode.GPU,
        preserve_row_count=source.row_count,
    )
    if selected is not ExecutionMode.GPU:
        raise RuntimeError("make-valid polygonize topology left GPU execution")
    return result


# ---------------------------------------------------------------------------
# Main GPU repair entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPURepairResult:
    """Complete device-resident result of GPU make_valid repair."""

    repaired_owned: OwnedGeometryArray | None  # device-resident merged result
    repaired_count: int
    gpu_phases_used: tuple[str, ...]


def _extract_batch_coords_device(
    d_buffer,
    invalid_family_rows,
    *,
    host_buffer: FamilyGeometryBuffer | None = None,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    DeviceFixedGeometrySizeMetadata | None,
] | None:
    """Device-side extraction of invalid polygon coordinates into contiguous batch.

    Like _extract_batch_coords but operates entirely on device arrays via
    _device_take_family_buffer. Host structural offsets, when still available,
    provide allocation sizes for the nested gather without D->H scalar fences.
    """
    from vibespatial.geometry.owned import _device_take_family_buffer

    if invalid_family_rows.size == 0:
        return None

    d_rows = cp.asarray(invalid_family_rows, dtype=cp.int32)
    host_family_rows = (
        np.asarray(invalid_family_rows, dtype=np.int32)
        if isinstance(invalid_family_rows, np.ndarray)
        else None
    )
    d_sorted_rows = cp.sort(d_rows)
    d_unique_starts = cp.empty(d_sorted_rows.size, dtype=cp.bool_)
    d_unique_starts[0] = True
    d_unique_starts[1:] = d_sorted_rows[1:] != d_sorted_rows[:-1]

    from vibespatial.api._native_rowset import NativeDeviceSelection

    unique_selection = NativeDeviceSelection.from_mask(d_unique_starts)
    d_unique_active = unique_selection.active_capacity_mask()
    d_unique_rows = unique_selection.gather_capacity(
        d_sorted_rows,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    taken = _device_take_family_buffer(
        d_buffer,
        GeometryFamily.POLYGON,
        d_unique_rows,
        host_buffer=host_buffer,
        host_family_rows=host_family_rows,
        assume_unique_indices=True,
        active_row_count=unique_selection.logical_count,
        active_row_mask=d_unique_active,
    )

    if taken.x.size == 0:
        return None

    repaired_fixed_size = None
    if taken.fixed_size is not None:
        ring_bound = taken.fixed_size.max_first_level_count_per_row
        coord_bound = taken.fixed_size.max_coord_count_per_row
        if ring_bound is not None and coord_bound is not None:
            repaired_fixed_size = DeviceFixedGeometrySizeMetadata(
                max_first_level_count_per_row=int(ring_bound),
                max_coord_count_per_row=int(coord_bound) + int(ring_bound),
            )

    source_row_count = int(d_buffer.geometry_offsets.size) - 1
    capacity = int(d_rows.size)
    d_lanes = cp.arange(capacity, dtype=cp.int64)
    d_lookup_destinations = cp.where(
        d_unique_active,
        d_unique_rows.astype(cp.int64, copy=False),
        cp.int64(source_row_count) + d_lanes,
    )
    d_unique_lanes_by_source = cp.full(
        source_row_count + capacity,
        -1,
        dtype=cp.int64,
    )
    d_unique_lanes_by_source[d_lookup_destinations] = d_lanes
    d_original_to_unique = d_unique_lanes_by_source[d_rows.astype(cp.int64, copy=False)]

    return (
        taken.x,
        taken.y,
        taken.ring_offsets,
        taken.geometry_offsets,
        d_original_to_unique,
        repaired_fixed_size,
    )


def _build_batch_repaired_device(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
    fixed_size: DeviceFixedGeometrySizeMetadata | None = None,
) -> OwnedGeometryArray | None:
    """Build a device-resident OwnedGeometryArray from batch device coordinates.

    Filters degenerate rings (< 4 vertices) on device and emits the repaired
    carrier directly through ``build_device_resident_owned``.
    """
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import (
        DeviceFamilyGeometryBuffer,
        build_device_resident_owned,
    )

    if d_x.size == 0 or polygon_count == 0:
        return None

    # Filter degenerate rings on device (< 4 vertices)
    d_ring_lens = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_valid_rings = d_ring_lens >= 4
    ring_selection = NativeDeviceSelection.from_mask(d_valid_rings)
    d_valid_ring_indices = ring_selection.safe_capacity_positions().astype(
        cp.int32,
        copy=False,
    )
    d_active_rings = ring_selection.active_capacity_mask()

    # Gather valid ring coordinate ranges and build new ring offsets
    from vibespatial.geometry.owned import _device_gather_xy_offset_slices

    new_x, new_y, new_ring_offsets = _device_gather_xy_offset_slices(
        d_x,
        d_y,
        d_ring_offsets,
        d_valid_ring_indices,
        allocation_capacity=int(d_x.size),
        active_row_count=ring_selection.logical_count,
    )

    # Build new geom offsets: count valid rings per polygon
    d_source_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:],
        cp.arange(ring_count, dtype=cp.int32),
        side="right",
    ).astype(cp.int32)
    d_poly_of_ring = ring_selection.gather_capacity(
        d_source_poly_of_ring,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    d_rings_per_poly = cp.bincount(
        d_poly_of_ring,
        weights=d_active_rings.astype(cp.int32),
        minlength=polygon_count,
    ).astype(cp.int32)
    new_geom_offsets = cp.zeros(polygon_count + 1, dtype=cp.int32)
    cp.cumsum(d_rings_per_poly, out=new_geom_offsets[1:])

    # Validity: polygons with zero valid rings are invalid
    d_poly_valid = d_rings_per_poly > 0
    d_tags = cp.full(polygon_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8)
    d_family_row_offsets = cp.arange(polygon_count, dtype=cp.int32)

    device_families = {
        GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            x=new_x,
            y=new_y,
            geometry_offsets=new_geom_offsets,
            empty_mask=~d_poly_valid,
            ring_offsets=new_ring_offsets,
            bounds=None,
            fixed_size=fixed_size,
        ),
    }
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=polygon_count,
        tags=d_tags,
        validity=d_poly_valid,
        family_row_offsets=d_family_row_offsets,
        execution_mode="gpu",
    )
    result.runtime_history.append(runtime_selection)
    if fixed_size is not None and fixed_size.max_coord_count_per_row is not None:
        result._active_family_row_segment_capacity_bound = int(
            fixed_size.max_coord_count_per_row
        )
    return result


def _build_polygon_rows_from_ring_selection_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_ring_rows: cp.ndarray,
    *,
    logical_count,
    d_active_rows: cp.ndarray,
    kernels: dict,
    orient_as_exterior: bool,
    coord_capacity_per_row: int | None = None,
) -> OwnedGeometryArray | None:
    """Build one single-ring polygon row per selection-capacity lane."""
    from vibespatial.geometry.owned import _device_gather_xy_offset_slices

    d_ring_rows = cp.asarray(d_ring_rows, dtype=cp.int64)
    d_active_rows = cp.asarray(d_active_rows, dtype=cp.bool_)
    row_count = int(d_ring_rows.size)
    if row_count == 0:
        return None
    if int(d_active_rows.size) != row_count:
        raise ValueError("make-valid ring positions and active mask must align")

    coord_capacity = (
        int(d_x.size)
        if coord_capacity_per_row is None
        else row_count * int(coord_capacity_per_row)
    )
    new_x, new_y, new_ring_offsets = _device_gather_xy_offset_slices(
        d_x,
        d_y,
        d_ring_offsets,
        d_ring_rows,
        allocation_capacity=coord_capacity,
        active_row_count=logical_count,
    )
    new_geom_offsets = cp.arange(row_count + 1, dtype=cp.int32)
    if orient_as_exterior:
        new_x, new_y = _gpu_fix_ring_orientation(
            new_x,
            new_y,
            new_ring_offsets,
            new_geom_offsets,
            row_count,
            row_count,
            kernels,
        )

    return build_device_resident_owned(
        device_families={
            GeometryFamily.POLYGON: DeviceFamilyGeometryBuffer(
                family=GeometryFamily.POLYGON,
                x=new_x,
                y=new_y,
                geometry_offsets=new_geom_offsets,
                empty_mask=~d_active_rows,
                ring_offsets=new_ring_offsets,
                bounds=None,
                fixed_size=DeviceFixedGeometrySizeMetadata(
                    max_first_level_count_per_row=1,
                    max_coord_count_per_row=(
                        coord_capacity_per_row
                        if coord_capacity_per_row is not None
                        else int(d_x.size)
                    ),
                ),
            )
        },
        row_count=row_count,
        tags=cp.full(row_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=cp.int8),
        validity=d_active_rows,
        family_row_offsets=cp.arange(row_count, dtype=cp.int32),
        execution_mode="gpu",
    )


def _build_hole_ring_polygons_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    problem_selection,
    *,
    kernels: dict,
    coord_capacity_per_row: int | None = None,
) -> tuple[OwnedGeometryArray, cp.ndarray] | None:
    """Explode selected polygon holes at physical ring capacity."""
    from vibespatial.api._native_rowset import NativeDeviceSelection

    if not isinstance(problem_selection, NativeDeviceSelection):
        raise TypeError("make-valid hole builder requires NativeDeviceSelection")
    if problem_selection.capacity == 0:
        return None

    ring_count = int(d_ring_offsets.size) - 1
    polygon_count = int(d_geom_offsets.size) - 1
    if ring_count <= 0 or polygon_count <= 0:
        return None
    if problem_selection.source_row_count != polygon_count:
        raise ValueError("make-valid problem selection source domain mismatch")

    # Project the compact-prefix lane id back to the polygon source domain.
    # Inactive lanes target scratch rows so duplicate safe positions cannot
    # overwrite an active source mapping.
    problem_capacity = problem_selection.capacity
    d_problem_active = problem_selection.active_capacity_mask()
    d_problem_lanes = cp.arange(problem_capacity, dtype=cp.int32)
    d_problem_destinations = cp.where(
        d_problem_active,
        problem_selection.safe_capacity_positions(),
        cp.int64(polygon_count) + d_problem_lanes.astype(cp.int64),
    )
    d_problem_local_extended = cp.full(
        polygon_count + problem_capacity,
        -1,
        dtype=cp.int32,
    )
    d_problem_local_extended[d_problem_destinations] = d_problem_lanes
    d_problem_local_rows = d_problem_local_extended[:polygon_count]

    d_ring_rows = cp.arange(ring_count, dtype=cp.int32)
    d_polygon_rows = cp.searchsorted(
        d_geom_offsets[1:],
        d_ring_rows,
        side="right",
    ).astype(cp.int32, copy=False)
    d_ring_source_rows = d_problem_local_rows[d_polygon_rows]
    d_is_hole = d_ring_rows > d_geom_offsets[d_polygon_rows]
    hole_selection = NativeDeviceSelection.from_mask((d_ring_source_rows >= 0) & d_is_hole)
    d_source_rows = hole_selection.gather_capacity(
        d_ring_source_rows,
        fill_value=0,
    ).astype(
        cp.int32,
        copy=False,
    )

    hole_owned = _build_polygon_rows_from_ring_selection_gpu(
        d_x,
        d_y,
        d_ring_offsets,
        hole_selection.safe_capacity_positions(),
        logical_count=hole_selection.logical_count,
        d_active_rows=hole_selection.active_capacity_mask(),
        kernels=kernels,
        orient_as_exterior=True,
        coord_capacity_per_row=coord_capacity_per_row,
    )
    if hole_owned is None:
        return None
    return hole_owned, d_source_rows


def _repair_touching_hole_rings_gpu(
    batch_result: OwnedGeometryArray,
    *,
    kernels: dict,
) -> tuple[OwnedGeometryArray, bool]:
    """Canonicalize invalid polygons whose hole rings touch or overlap.

    Physical shape: selected post-repair invalid polygon rows -> ring capacity ->
    grouped hole coverage union -> row-aligned exterior-minus-holes difference.
    This repairs inter-ring topology without materializing Shapely geometries.
    """
    if cp is None or batch_result.device_state is None or batch_result.row_count == 0:
        return batch_result, False

    state = batch_result._ensure_device_state()
    if len(state.families) != 1:
        return batch_result, False
    poly_buf = state.families.get(GeometryFamily.POLYGON)
    if poly_buf is None or poly_buf.ring_offsets is None:
        return batch_result, False
    if int(poly_buf.geometry_offsets.size) - 1 != int(batch_result.row_count):
        return batch_result, False

    from vibespatial.constructive.validity import validity_expression_owned

    d_valid_flags = cp.asarray(
        validity_expression_owned(batch_result, exact_collinearity=True).values,
        dtype=cp.bool_,
    )
    d_geom_offsets = cp.asarray(poly_buf.geometry_offsets, dtype=cp.int32)
    d_ring_counts = d_geom_offsets[1:] - d_geom_offsets[:-1]
    d_problem_mask = (
        (~d_valid_flags) & cp.asarray(state.validity, dtype=cp.bool_) & (d_ring_counts > 1)
    )
    from vibespatial.api._native_rowset import NativeDeviceSelection

    problem_selection = NativeDeviceSelection.from_mask(d_problem_mask)
    problem_capacity = problem_selection.capacity
    fixed_size = poly_buf.fixed_size
    coord_capacity_per_row = (
        None
        if fixed_size is None
        else fixed_size.max_coord_count_per_row
    )

    d_exterior_ring_rows = d_geom_offsets[problem_selection.safe_capacity_positions()].astype(
        cp.int64, copy=False
    )
    exterior_owned = _build_polygon_rows_from_ring_selection_gpu(
        cp.asarray(poly_buf.x),
        cp.asarray(poly_buf.y),
        cp.asarray(poly_buf.ring_offsets, dtype=cp.int32),
        d_exterior_ring_rows,
        logical_count=problem_selection.logical_count,
        d_active_rows=problem_selection.active_capacity_mask(),
        kernels=kernels,
        orient_as_exterior=True,
        coord_capacity_per_row=coord_capacity_per_row,
    )
    if exterior_owned is None:
        return batch_result, False

    hole_parts = _build_hole_ring_polygons_gpu(
        cp.asarray(poly_buf.x),
        cp.asarray(poly_buf.y),
        cp.asarray(poly_buf.ring_offsets, dtype=cp.int32),
        d_geom_offsets,
        problem_selection,
        kernels=kernels,
        coord_capacity_per_row=coord_capacity_per_row,
    )
    if hole_parts is None:
        return batch_result, False
    hole_owned, d_hole_source_rows = hole_parts

    from vibespatial.constructive.binary_constructive import (
        _dispatch_grouped_polygon_known_coverage_union_gpu,
        binary_constructive_owned,
    )

    hole_union = _dispatch_grouped_polygon_known_coverage_union_gpu(
        hole_owned,
        d_hole_source_rows,
        output_row_count=problem_capacity,
        dispatch_mode=ExecutionMode.GPU,
        assume_all_valid=True,
        assume_source_rows_valid=True,
    )
    if hole_union is None or hole_union.row_count != problem_capacity:
        return batch_result, False

    repaired = binary_constructive_owned(
        "difference",
        exterior_owned,
        hole_union,
        dispatch_mode=ExecutionMode.GPU,
    )
    if repaired is None or repaired.row_count != problem_capacity:
        return batch_result, False

    from vibespatial.constructive.measurement import _area_gpu_device_fp64

    d_input_area = problem_selection.gather_capacity(
        _area_gpu_device_fp64(batch_result),
        fill_value=0.0,
    )
    d_repaired_area = _area_gpu_device_fp64(repaired)
    d_area_tolerance = cp.maximum(cp.abs(d_input_area) * 1.0e-9, 1.0e-9)
    d_repaired_valid_flags = cp.asarray(
        validity_expression_owned(repaired, exact_collinearity=True).values,
        dtype=cp.bool_,
    )
    d_accepted = problem_selection.active_capacity_mask() & (
        (cp.abs(d_input_area - d_repaired_area) <= d_area_tolerance) & d_repaired_valid_flags
    )

    from vibespatial.geometry.owned import device_scatter_owned_capacity_selection

    scattered = device_scatter_owned_capacity_selection(
        batch_result,
        repaired,
        problem_selection,
        active_mask=d_accepted,
    )
    return scattered, True


def _device_scatter_repaired(
    original_owned: OwnedGeometryArray,
    repaired_batch: OwnedGeometryArray,
    family_name: str,
    invalid_family_rows,
    invalid_global_rows,
) -> OwnedGeometryArray:
    """Scatter an aligned repaired batch through native row indirection."""
    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.geometry.owned import device_scatter_owned_capacity_selection

    d_invalid_family_rows = cp.asarray(invalid_family_rows, dtype=cp.int32)
    d_invalid_global_rows = cp.asarray(invalid_global_rows, dtype=cp.int64)
    invalid_count = int(d_invalid_family_rows.size)
    if int(d_invalid_global_rows.size) != invalid_count:
        raise ValueError("make_valid family/global repair rowsets must align")
    if repaired_batch.row_count != invalid_count:
        raise RuntimeError(
            "make_valid repaired batch lost input row alignment "
            f"({repaired_batch.row_count} repaired rows for "
            f"{invalid_count} invalid source rows)"
        )
    selection = NativeDeviceSelection(
        positions=d_invalid_global_rows,
        logical_count=cp.asarray([invalid_count], dtype=cp.int64),
        source_row_count=original_owned.row_count,
        ordered=True,
        unique=True,
    )
    return device_scatter_owned_capacity_selection(
        original_owned,
        repaired_batch,
        selection,
    )


def _scatter_valid_repaired_batch(
    original_owned: OwnedGeometryArray,
    repaired_batch: OwnedGeometryArray | None,
    family_name: str,
    invalid_family_rows,
    invalid_global_rows,
) -> tuple[OwnedGeometryArray, object] | None:
    """Validate and scatter one aligned repair batch at input-row capacity."""
    if repaired_batch is None:
        return None

    from vibespatial.api._native_rowset import NativeDeviceSelection
    from vibespatial.constructive.validity import validity_expression_owned
    from vibespatial.geometry.owned import device_scatter_owned_capacity_selection

    d_batch_valid = cp.asarray(
        validity_expression_owned(
            repaired_batch,
            exact_collinearity=True,
        ).values,
        dtype=cp.bool_,
    )
    d_invalid_family_rows = cp.asarray(invalid_family_rows, dtype=cp.int32)
    d_invalid_global_rows = cp.asarray(invalid_global_rows, dtype=cp.int64)
    capacity = int(d_invalid_global_rows.size)
    if int(d_invalid_family_rows.size) != capacity:
        raise ValueError("make_valid family/global repair rowsets must align")
    if repaired_batch.row_count != capacity:
        raise RuntimeError(
            "make_valid repaired batch lost input row capacity "
            f"({repaired_batch.row_count} repaired rows for {capacity} lanes)"
        )

    selection = NativeDeviceSelection(
        positions=d_invalid_global_rows,
        logical_count=cp.asarray([capacity], dtype=cp.int64),
        source_row_count=original_owned.row_count,
        ordered=True,
        unique=True,
    )

    return (
        device_scatter_owned_capacity_selection(
            original_owned,
            repaired_batch,
            selection,
            active_mask=d_batch_valid,
        ),
        selection.source_mask(active_mask=d_batch_valid),
    )


def _repair_multipolygon_rows_grouped_device(
    owned: OwnedGeometryArray,
    invalid_global_rows,
) -> OwnedGeometryArray | None:
    """Repair multipart polygon rows as grouped polygon-part topology."""
    d_invalid_global_rows = cp.asarray(invalid_global_rows, dtype=cp.int64)
    if int(d_invalid_global_rows.size) == 0:
        return None

    from vibespatial.api._native_grouped import NativeGroupedSelection
    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygon_capacity_gpu,
    )
    from vibespatial.constructive.make_valid_pipeline import make_valid_owned
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs
    from vibespatial.geometry.owned import build_empty_polygon_rows_device
    from vibespatial.kernels.constructive.segmented_union import (
        segmented_union_all_device_grouped,
    )
    from vibespatial.runtime.precision import (
        CompensationMode,
        KernelClass,
        PrecisionMode,
        PrecisionPlan,
        RefinementMode,
    )

    selected = owned.device_take(d_invalid_global_rows)
    polygon_parts = _explode_polygonal_rows_to_polygon_capacity_gpu(selected)
    if polygon_parts is None or polygon_parts.capacity == 0:
        return None

    part_repair = make_valid_owned(
        owned=polygon_parts.geometry,
        method="structure",
        keep_collapsed=True,
        dispatch_mode=ExecutionMode.GPU,
    )
    repaired_parts = part_repair.owned
    if repaired_parts is None or repaired_parts.row_count != polygon_parts.capacity:
        return None

    part_capacity = polygon_parts.capacity
    d_part_source_rows = cp.asarray(polygon_parts.source_rows, dtype=cp.int32)
    grouped_parts = NativeGroupedSelection(
        selection=polygon_parts.selection,
        group_codes=d_part_source_rows,
        group_count=selected.row_count,
    )
    d_group_sizes = grouped_parts.reduce_numeric(
        cp.ones(part_capacity, dtype=cp.int32),
        "count",
    ).values.astype(cp.int64, copy=False)
    d_active = polygon_parts.selection.active_capacity_mask()
    # The grouped reducer accepts observed groups only. Capacity lanes after the
    # active part prefix are invalid geometry rows, so place them at the tail of
    # group zero instead of declaring a potentially zero-length sentinel group.
    # Seed selection ignores invalid rows and every declared group remains
    # structurally nonempty.
    d_sort_groups = cp.where(
        d_active,
        d_part_source_rows,
        cp.int32(0),
    ).astype(cp.uint64, copy=False)
    d_sort_keys = (d_sort_groups << cp.uint64(32)) | cp.arange(part_capacity, dtype=cp.uint64)
    d_order = sort_pairs(
        d_sort_keys,
        cp.arange(part_capacity, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)

    d_group_sizes[0] += (
        cp.int64(part_capacity) - cp.asarray(polygon_parts.logical_count, dtype=cp.int64)[0]
    )
    d_group_offsets = cp.empty(selected.row_count + 1, dtype=cp.int64)
    d_group_offsets[0] = 0
    cp.cumsum(d_group_sizes, out=d_group_offsets[1:])
    ordered_parts = repaired_parts._device_indexed_take(d_order)
    empty_output = build_empty_polygon_rows_device(selected.row_count)
    grouped_result = segmented_union_all_device_grouped(
        ordered_parts,
        d_group_offsets,
        cp.arange(selected.row_count, dtype=cp.int64),
        output_row_count=selected.row_count,
        precision_plan=PrecisionPlan(
            storage_precision=PrecisionMode.FP64,
            compute_precision=PrecisionMode.FP64,
            kernel_class=KernelClass.CONSTRUCTIVE,
            compensation=CompensationMode.NONE,
            refinement=RefinementMode.NONE,
            center_coordinates=False,
            reason="multipart make-valid uses grouped constructive fp64",
        ),
        empty_output=empty_output,
        all_groups_observed=False,
    )
    if grouped_result is None or grouped_result.row_count != selected.row_count:
        return None
    return grouped_result


def gpu_repair_invalid_polygons(
    owned: OwnedGeometryArray,
    invalid_rows: object,
    geometries: np.ndarray | None = None,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
) -> GPURepairResult | None:
    """GPU-resident batch repair of invalid polygon geometries (Phase 16).

    Implements the full make_valid pipeline on GPU with batch processing:
    1. Collect invalid polygon coordinates into a device repair batch.
    2. Normalize rings through capacity-backed closure/dedup/orientation.
    3. Select still-invalid rows with the canonical OGC validity expression.
    4. Polygonize those rows through shared paged overlay topology.
    5. Merge valid repaired rows back into original logical order on device.

    When ``owned.device_state`` is available, the coordinate pipeline and its
    compact logical/family rowsets stay device-resident. The result carries a
    complete ``repaired_owned`` array so callers can stay on device (ADR-0005).

    Returns None if GPU repair is not applicable (no GPU, no polygon families,
    or CuPy not available).

    Parameters
    ----------
    owned : OwnedGeometryArray with device_state
    invalid_rows : indices of invalid rows to repair
    geometries : optional shapely geometry array (unused in device path)
    method : repair method (only "linework" supported on GPU)
    keep_collapsed : whether to keep collapsed geometries
    """
    if cp is None:
        return None

    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    if invalid_rows.size == 0:
        return GPURepairResult(
            repaired_owned=owned,
            repaired_count=0,
            gpu_phases_used=(),
        )

    # Only polygon families have a native repair implementation. Unsupported
    # invalid families make the complete native operation inapplicable.
    polygon_families = set()
    for family_name in ("polygon", "multipolygon"):
        if family_name in owned.families:
            polygon_families.add(family_name)

    if not polygon_families:
        return None

    # Compilation/runtime failures after native admission are unexpected and
    # propagate atomically. Structural inapplicability returns above.
    kernels = _compile_repair_kernels()

    # GPU repair has one physical input contract. Host-owned callers upload once
    # here; all family routing and nested gathers below consume device metadata.
    device_state = owned._ensure_device_state(preserve_indexed_view=True)

    runtime_sel = _planned_make_valid_runtime_selection(
        kernel_name="make_valid_gpu_batch_repair",
        owned=owned,
        selected_row_count=int(invalid_rows.size),
        reason="make_valid GPU batch repair pipeline (Phase 16)",
    )

    phases_used: list[str] = []
    d_repaired_global_mask = cp.zeros(owned.row_count, dtype=cp.bool_)
    # Start with the original owned; each family merge updates it
    merged_owned = owned

    for family_name in polygon_families:
        family = GeometryFamily(family_name)
        buffer = owned.families[family]
        d_buf = device_state.families.get(family)
        if d_buf is None or d_buf.ring_offsets is None or d_buf.geometry_offsets is None:
            continue
        polygon_count = int(d_buf.geometry_offsets.size) - 1
        if polygon_count <= 0:
            continue

        # Map compact invalid rows to aligned logical and family device rowsets.
        # Do not touch owned.tags/family_row_offsets here; those lazily
        # materialize full host metadata on Native* paths.
        family_tag = FAMILY_TAGS[family]
        d_invalid_rows = cp.asarray(invalid_rows, dtype=cp.int32)
        d_tags = cp.asarray(device_state.tags)
        d_family_row_offsets = cp.asarray(device_state.family_row_offsets)
        d_family_mask = d_tags[d_invalid_rows] == np.int8(family_tag)
        d_global_invalid = d_invalid_rows[d_family_mask].astype(cp.int32, copy=False)
        if int(d_global_invalid.size) == 0:
            continue
        d_fam_row_offsets = d_family_row_offsets[d_global_invalid].astype(
            cp.int32,
            copy=False,
        )
        d_valid_fro = (d_fam_row_offsets >= 0) & (d_fam_row_offsets < polygon_count)
        d_global_invalid = d_global_invalid[d_valid_fro].astype(cp.int32, copy=False)
        d_fam_row_offsets = d_fam_row_offsets[d_valid_fro].astype(cp.int32, copy=False)
        if int(d_global_invalid.size) == 0:
            continue

        d_invalid_family_rows = d_fam_row_offsets.astype(
            cp.int32,
            copy=False,
        )

        if family is GeometryFamily.MULTIPOLYGON:
            batch_result = _repair_multipolygon_rows_grouped_device(
                merged_owned,
                d_global_invalid,
            )
            scattered = _scatter_valid_repaired_batch(
                merged_owned,
                batch_result,
                family_name,
                d_invalid_family_rows,
                d_global_invalid,
            )
            if scattered is not None:
                merged_owned, d_scattered_global_mask = scattered
                d_repaired_global_mask |= d_scattered_global_mask
                phases_used.append("grouped_multipart_topology")
            continue

        batch = _extract_batch_coords_device(
            d_buf,
            d_invalid_family_rows,
            host_buffer=buffer,
        )
        if batch is None:
            continue
        (
            d_x,
            d_y,
            d_ring_offsets,
            d_geom_offsets,
            d_original_to_unique,
            repaired_fixed_size,
        ) = batch

        batch_ring_count = d_ring_offsets.size - 1
        batch_poly_count = d_geom_offsets.size - 1

        if batch_ring_count == 0 or batch_poly_count == 0:
            continue

        # --- Step 2: Phase B — batched simple repair ---
        d_x, d_y, d_ring_offsets = _gpu_close_rings(
            d_x,
            d_y,
            d_ring_offsets,
            batch_ring_count,
            kernels,
        )
        phases_used.append("close_rings")

        d_x, d_y, d_ring_offsets = _gpu_remove_duplicate_vertices(
            d_x,
            d_y,
            d_ring_offsets,
            batch_ring_count,
            kernels,
        )
        phases_used.append("remove_duplicates")

        d_x, d_y = _gpu_fix_ring_orientation(
            d_x,
            d_y,
            d_ring_offsets,
            d_geom_offsets,
            batch_ring_count,
            batch_poly_count,
            kernels,
        )
        phases_used.append("fix_orientation")

        batch_result = _build_batch_repaired_device(
            d_x,
            d_y,
            d_ring_offsets,
            d_geom_offsets,
            batch_ring_count,
            batch_poly_count,
            runtime_sel,
            repaired_fixed_size,
        )

        # Preserve the original shell/hole semantics before generic face
        # polygonization. Polygonizing adjacent invalid holes can emit only
        # the exterior face, after which the hole rings are no longer
        # available for the grouped union and exterior-difference repair.
        if batch_result is not None:
            batch_result, repaired_touching_holes = _repair_touching_hole_rings_gpu(
                batch_result,
                kernels=kernels,
            )
            if repaired_touching_holes:
                phases_used.append("repair_touching_holes")

        # Invalid simple-repair rows use the same sweep, streamed split-event,
        # half-edge, and complete-row paging carriers as overlay. This replaces
        # the old make-valid-specific quadratic pair matrix and contiguous
        # split/rebuild allocation.
        if batch_result is not None:
            from vibespatial.api._native_rowset import NativeDeviceSelection
            from vibespatial.constructive.validity import validity_expression_owned
            from vibespatial.geometry.owned import device_mask_owned_capacity

            batch_state = batch_result._ensure_device_state(preserve_indexed_view=True)
            d_topology_mask = (
                ~cp.asarray(
                    validity_expression_owned(
                        batch_result,
                        exact_collinearity=True,
                    ).values,
                    dtype=cp.bool_,
                )
            ) & cp.asarray(batch_state.validity, dtype=cp.bool_)
            topology_selection = NativeDeviceSelection.from_mask(d_topology_mask)
            topology_count = get_cuda_runtime().copy_device_to_host(
                topology_selection.logical_count,
                reason="make-valid topology plan admission scalar fence",
            )
            if int(np.asarray(topology_count, dtype=np.int64)[0]) > 0:
                topology_source = device_mask_owned_capacity(
                    batch_result,
                    d_topology_mask,
                )
                topology_repaired = _repolygonize_owned_rows_via_overlay(topology_source)
                topology_scattered = _scatter_valid_repaired_batch(
                    batch_result,
                    topology_repaired,
                    GeometryFamily.POLYGON.value,
                    cp.arange(batch_result.row_count, dtype=cp.int32),
                    cp.arange(batch_result.row_count, dtype=cp.int64),
                )
                if topology_scattered is not None:
                    batch_result, _ = topology_scattered
                    phases_used.append("overlay_topology_polygonize")

        # --- Step 5: Merge repaired batch back into owned on device ---
        if batch_result is not None:
            batch_result = batch_result._device_indexed_take(
                d_original_to_unique,
                assume_unique_indices=False,
            )
        scattered = _scatter_valid_repaired_batch(
            merged_owned,
            batch_result,
            family_name,
            d_invalid_family_rows,
            d_global_invalid,
        )
        if scattered is not None:
            merged_owned, d_scattered_global_mask = scattered
            d_repaired_global_mask |= d_scattered_global_mask

    # A repair carrier is atomic: never expose a partly repaired array for a
    # caller to patch row-by-row on the host. The public dispatcher owns the
    # observable whole-operation decline when any requested row remains.
    d_invalid_rows = cp.asarray(invalid_rows, dtype=cp.int64)
    d_requested_mask = cp.zeros(owned.row_count, dtype=cp.bool_)
    d_requested_mask[d_invalid_rows] = True
    d_complete = cp.asarray(
        cp.all((~d_requested_mask) | d_repaired_global_mask),
        dtype=cp.bool_,
    ).reshape(1)
    complete = get_cuda_runtime().copy_device_to_host(
        d_complete,
        reason="make-valid atomic repair completion admission scalar fence",
    )
    if not bool(np.asarray(complete, dtype=bool)[0]):
        return None

    repaired_count = int(invalid_rows.size)
    return GPURepairResult(
        repaired_owned=merged_owned,
        repaired_count=repaired_count,
        gpu_phases_used=tuple(set(phases_used)),
    )
