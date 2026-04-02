"""GPU-accelerated extract_unique_points: per-geometry coordinate deduplication.

For each geometry row, extracts all coordinates, deduplicates (x, y) pairs,
and returns a MultiPoint containing only the unique coordinates.

ADR-0033 tiers:
    Tier 1 NVRTC — count_coords_per_row, scatter_coords, mark_unique_coords
    Tier 3a CCCL — segmented_sort (sort x within row segments),
                   exclusive_sum (prefix scan for offsets),
                   compact_indices (gather unique coords)
    Tier 2 CuPy  — element-wise gather, boolean indexing

ADR-0002: CONSTRUCTIVE class — fp64 uniform precision.  Coordinates are
exact subsets of input (no arithmetic), so the precision plan is wired
through for observability but stays fp64.

ADR-0034: NVRTC and CCCL warmup registered at module scope.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.constructive.extract_unique_points_cpu import (
    _extract_unique_points_cpu,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    exclusive_sum,
    segmented_sort,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    OwnedGeometryArray,
    build_device_resident_owned,
    from_shapely_geometries,
)
from vibespatial.kernels.constructive.extract_unique_points import (
    KERNEL_SOURCE,
    _get_kernel_names,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
)

# ADR-0034: request warmup for CCCL primitives used in this module
request_warmup([
    "exclusive_scan_i32",
    "segmented_sort_asc_f64",
])

_KERNEL_NAMES = _get_kernel_names()

request_nvrtc_warmup([
    ("extract-unique-points-fp64", KERNEL_SOURCE, _KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

def _compile_kernels():
    """Compile and cache NVRTC kernels."""
    return compile_kernel_group(
        "extract-unique-points-fp64",
        KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


@register_kernel_variant(
    "extract_unique_points",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "extract_unique_points"),
)
def _extract_unique_points_gpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """GPU extract_unique_points — returns device-resident MultiPoint OGA.

    Algorithm:
        1. Count coordinates per geometry row (NVRTC count_coords_per_row)
        2. Exclusive prefix sum for write offsets (CCCL exclusive_sum)
        3. Scatter all coordinates into flat SoA arrays (NVRTC scatter_coords)
        4. Sort x within row-segments (CCCL segmented_sort), carrying y and
           original indices
        5. Mark unique (x, y) pairs within each segment (NVRTC mark_unique_coords)
        6. Count unique per row (CCCL exclusive_sum on unique counts)
        7. Compact unique coordinates (CuPy boolean gather)
        8. Build output MultiPoint OGA
    """
    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count

    # Identify valid rows (non-null)
    validity = owned.validity
    tags = owned.tags

    valid_mask = validity.copy()
    valid_rows_host = np.flatnonzero(valid_mask).astype(np.int32, copy=False)

    if valid_rows_host.size == 0:
        return _build_empty_multipoint_output(row_count, validity)

    n_valid = valid_rows_host.size
    kernels = _compile_kernels()
    ptr = runtime.pointer

    # Tags are int8 on device; kernels read int32
    d_family_codes = d_state.tags.astype(cp.int32) if d_state.tags.dtype != cp.int32 else d_state.tags
    d_family_row_off = d_state.family_row_offsets

    # Build per-family offset pointers and empty masks.
    # We process families one at a time, launching count + scatter per family,
    # then merge.  But the kernel is designed to handle all families in a
    # single launch using family dispatch.  To do that we need unified
    # offset arrays.  The segment_primitives pattern processes per-family,
    # but for extract_unique_points we can use a simpler approach: since
    # the kernel branches on family code, we need the offsets from each
    # family's device buffer accessible via the family_row_offset indirection.
    #
    # Strategy: process per-family (up to 6 families, 6 launches for count
    # + 6 for scatter).  This is cleanest and avoids building unified
    # offset tables.  Each launch processes only rows of one family.

    # Accumulate per-family results
    family_data = []

    for family_enum in GeometryFamily:
        family_tag = FAMILY_TAGS[family_enum]
        if family_enum not in d_state.families:
            continue

        d_buf = d_state.families[family_enum]

        # Valid rows for this family
        fam_valid_mask = validity & (tags == family_tag)
        fam_valid_rows = np.flatnonzero(fam_valid_mask).astype(np.int32, copy=False)
        if fam_valid_rows.size == 0:
            continue

        n_fam = fam_valid_rows.size
        d_fam_valid = runtime.from_host(fam_valid_rows)

        # Build empty mask on device
        if family_enum in owned.families:
            host_buf = owned.families[family_enum]
            fam_local_rows = owned.family_row_offsets[fam_valid_rows]
            d_empty_host = cp.asarray(host_buf.empty_mask.astype(np.uint8))
            d_fam_local = cp.asarray(fam_local_rows)
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)
            safe_idx = cp.minimum(d_fam_local, max(int(d_empty_host.size) - 1, 0))
            valid_fr = d_fam_local < d_empty_host.size
            d_fam_empty[valid_fr] = d_empty_host[safe_idx[valid_fr]]
        else:
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)

        # Offset arrays (use dummy zeros if not present)
        d_geom_off = d_buf.geometry_offsets
        d_part_off = d_buf.part_offsets if d_buf.part_offsets is not None else cp.zeros(1, dtype=cp.int32)
        d_ring_off = d_buf.ring_offsets if d_buf.ring_offsets is not None else cp.zeros(1, dtype=cp.int32)

        # --- Pass 1: Count coordinates per row ---
        d_counts = runtime.allocate((n_fam,), np.int32, zero=True)

        params = (
            (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_off),
             ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
             ptr(d_fam_empty), ptr(d_counts), n_fam),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernels["count_coords_per_row"], n_fam)
        runtime.launch(kernels["count_coords_per_row"], grid=grid, block=block, params=params)

        family_data.append((
            family_enum, n_fam, d_fam_valid, d_fam_empty,
            d_geom_off, d_part_off, d_ring_off, d_counts, d_buf,
        ))

    if not family_data:
        return _build_empty_multipoint_output(row_count, validity)

    # Merge all family counts into a single array indexed by valid_row position.
    # Build a mapping: for each valid row, find which family_data entry it belongs to
    # and what its count is.
    # Simpler approach: allocate a full-sized count array for all valid rows,
    # then fill in from each family's counts.
    d_all_counts = cp.zeros(n_valid, dtype=cp.int32)

    # Build mapping from valid_rows_host position to family position
    for family_enum, n_fam, d_fam_valid, d_fam_empty, \
            d_geom_off, d_part_off, d_ring_off, d_counts, d_buf in family_data:
        family_tag = FAMILY_TAGS[family_enum]
        fam_valid_mask = validity & (tags == family_tag)
        fam_valid_rows = np.flatnonzero(fam_valid_mask).astype(np.int32, copy=False)

        # Map family valid rows to positions in valid_rows_host
        # valid_rows_host is sorted, fam_valid_rows is a subset
        # Use searchsorted to find positions
        positions = np.searchsorted(valid_rows_host, fam_valid_rows).astype(np.int32)
        d_positions = cp.asarray(positions)
        # Scatter counts
        d_all_counts[d_positions] = d_counts

    # --- Prefix sum for scatter offsets ---
    d_coord_offsets = exclusive_sum(d_all_counts, synchronize=False)

    # Get total coordinate count
    total_coords = count_scatter_total(runtime, d_all_counts, d_coord_offsets)

    if total_coords == 0:
        return _build_empty_multipoint_output(row_count, validity)

    # --- Allocate output coordinate buffers ---
    d_x_flat = runtime.allocate((total_coords,), np.float64)
    d_y_flat = runtime.allocate((total_coords,), np.float64)
    d_row_ids = runtime.allocate((total_coords,), np.int32)

    # --- Pass 2: Scatter coordinates ---
    for family_enum, n_fam, d_fam_valid, d_fam_empty, \
            d_geom_off, d_part_off, d_ring_off, d_counts, d_buf in family_data:
        family_tag = FAMILY_TAGS[family_enum]

        # We need the per-family coord_offsets.  The offsets in d_coord_offsets
        # are indexed by valid-row position.  We need to extract the offsets
        # for this family's rows.
        fam_valid_mask = validity & (tags == family_tag)
        fam_valid_rows = np.flatnonzero(fam_valid_mask).astype(np.int32, copy=False)
        positions = np.searchsorted(valid_rows_host, fam_valid_rows).astype(np.int32)
        d_positions = cp.asarray(positions)
        d_fam_offsets = d_coord_offsets[d_positions]

        # row_id_map: maps family-local tid -> merged valid-row position
        d_row_id_map = d_positions

        params = (
            (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_off),
             ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
             ptr(d_fam_empty),
             ptr(d_buf.x), ptr(d_buf.y),
             ptr(d_fam_offsets),
             ptr(d_row_id_map),
             ptr(d_x_flat), ptr(d_y_flat), ptr(d_row_ids),
             n_fam),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernels["scatter_coords"], n_fam)
        runtime.launch(kernels["scatter_coords"], grid=grid, block=block, params=params)

    # --- Lexicographic sort (x, y) within row-segments ---
    # CCCL segmented_sort uses radix sort (stable).  Two-pass stable sort:
    #   Pass A: sort by y (secondary key) within segments
    #   Pass B: sort by x (primary key) within segments
    # After both stable passes, coordinates are lexicographically sorted
    # by (x, y), so exact duplicate (x, y) pairs are adjacent.
    d_seg_starts = d_coord_offsets.astype(cp.int32)
    d_seg_ends = (d_coord_offsets + d_all_counts).astype(cp.int32)

    d_indices = cp.arange(total_coords, dtype=cp.int32)

    # Pass A: sort by y (secondary key), carrying indices
    sort_y = segmented_sort(
        keys=d_y_flat,
        values=d_indices,
        starts=d_seg_starts,
        ends=d_seg_ends,
        num_segments=n_valid,
    )
    # Reorder x by the y-sort permutation
    d_x_after_ysort = d_x_flat[sort_y.values]

    # Pass B: sort by x (primary key), carrying the y-sort indices
    sort_x = segmented_sort(
        keys=d_x_after_ysort,
        values=sort_y.values,
        starts=d_seg_starts,
        ends=d_seg_ends,
        num_segments=n_valid,
    )

    # Final sorted arrays: apply the composed permutation
    d_x_sorted = sort_x.keys
    d_y_sorted = d_y_flat[sort_x.values]
    d_row_ids_sorted = d_row_ids[sort_x.values]

    # --- Mark unique (x, y) pairs within each segment (NVRTC) ---
    d_unique_mask = runtime.allocate((total_coords,), np.uint8, zero=True)

    params = (
        (ptr(d_x_sorted), ptr(d_y_sorted), ptr(d_row_ids_sorted),
         ptr(d_unique_mask), total_coords),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernels["mark_unique_coords"], total_coords)
    runtime.launch(kernels["mark_unique_coords"], grid=grid, block=block, params=params)

    # --- Count unique per row ---
    # Convert unique_mask to int32 for prefix sum per row
    d_unique_i32 = d_unique_mask.astype(cp.int32)

    # Sum unique counts per valid row using the row_ids and unique mask
    # Strategy: scatter-add unique_mask into per-row unique counts
    d_unique_counts = cp.zeros(n_valid, dtype=cp.int32)
    # Use cupyx scatter_add for the per-row reduction
    cp.add.at(d_unique_counts, d_row_ids_sorted, d_unique_i32)

    # Build output geometry offsets via exclusive prefix sum
    d_out_offsets = exclusive_sum(d_unique_counts, synchronize=False)
    total_unique = count_scatter_total(runtime, d_unique_counts, d_out_offsets)

    if total_unique == 0:
        return _build_empty_multipoint_output(row_count, validity)

    # --- Compact unique coordinates ---
    # Gather unique coordinates using boolean mask
    d_unique_bool = d_unique_mask.astype(cp.bool_)
    d_compact_indices = cp.flatnonzero(d_unique_bool).astype(cp.int32)
    cp.cuda.Stream.null.synchronize()

    d_x_unique = d_x_sorted[d_compact_indices]
    d_y_unique = d_y_sorted[d_compact_indices]

    # --- Build output MultiPoint OGA ---
    # Output geometry_offsets: prefix sums of unique counts per valid row,
    # but we need offsets for ALL rows (including nulls).
    # Build full-size geometry offsets on device: null/invalid rows get
    # zero-length spans.  No D2H round-trip.
    d_valid_rows = cp.asarray(valid_rows_host)
    d_full_counts = cp.zeros(row_count + 1, dtype=cp.int32)
    d_full_counts[d_valid_rows + 1] = d_unique_counts
    cp.cumsum(d_full_counts, out=d_full_counts)
    d_out_geom_offsets = d_full_counts

    # Empty mask on device: rows with zero unique coords are empty
    d_mp_empty = cp.zeros(row_count, dtype=cp.uint8)
    d_zero_count = d_unique_counts == 0
    if int(d_zero_count.any()):
        d_mp_empty[d_valid_rows[d_zero_count]] = 1

    # Build output metadata arrays
    out_validity = validity.copy()
    out_tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.MULTIPOINT], dtype=np.int8)
    out_tags[~validity] = -1  # null rows
    out_family_row_offsets = np.arange(row_count, dtype=np.int32)
    out_family_row_offsets[~validity] = -1

    device_families = {
        GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=d_x_unique,
            y=d_y_unique,
            geometry_offsets=d_out_geom_offsets,
            empty_mask=d_mp_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
    )


def _build_empty_multipoint_output(
    row_count: int,
    validity: np.ndarray,
) -> OwnedGeometryArray:
    """Build an all-empty MultiPoint OGA for degenerate cases."""
    runtime = get_cuda_runtime()
    out_validity = validity.copy()
    out_tags = np.full(row_count, FAMILY_TAGS[GeometryFamily.MULTIPOINT], dtype=np.int8)
    out_tags[~validity] = -1
    out_family_row_offsets = np.arange(row_count, dtype=np.int32)
    out_family_row_offsets[~validity] = -1

    d_x = runtime.allocate((0,), np.float64)
    d_y = runtime.allocate((0,), np.float64)
    d_geom_off = runtime.from_host(np.zeros(row_count + 1, dtype=np.int32))
    d_empty = runtime.from_host(np.ones(row_count, dtype=np.uint8))

    device_families = {
        GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=d_x,
            y=d_y,
            geometry_offsets=d_geom_off,
            empty_mask=d_empty,
        ),
    }

    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=out_validity,
        family_row_offsets=out_family_row_offsets,
    )


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def extract_unique_points_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Extract unique coordinates from each geometry as MultiPoint.

    For each row, flattens all coordinates, deduplicates (x, y) pairs,
    and returns a MultiPoint containing the unique coordinates.

    Parameters
    ----------
    owned : OwnedGeometryArray
        Input geometries (any family).
    dispatch_mode : ExecutionMode or str, default AUTO
        Execution mode hint.
    precision : PrecisionMode or str, default AUTO
        Precision mode.  CONSTRUCTIVE class stays fp64 per ADR-0002;
        wired for observability.

    Returns
    -------
    OwnedGeometryArray
        MultiPoint geometries with unique coordinates per row.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="extract_unique_points",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        requested_precision=precision,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _extract_unique_points_gpu(owned)
        record_dispatch_event(
            surface="geopandas.array.extract_unique_points",
            operation="extract_unique_points",
            implementation="extract_unique_points_gpu_nvrtc",
            reason=selection.reason,
            detail=(
                f"rows={row_count}, "
                f"precision={precision_plan.compute_precision.value}"
            ),
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    result = _extract_unique_points_cpu(owned)
    record_dispatch_event(
        surface="geopandas.array.extract_unique_points",
        operation="extract_unique_points",
        implementation="extract_unique_points_cpu_shapely",
        reason=selection.reason,
        detail=f"rows={row_count}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    return result
