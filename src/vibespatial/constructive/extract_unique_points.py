"""GPU-accelerated extract_unique_points: per-geometry coordinate deduplication.

For each geometry row, extracts all coordinates, deduplicates (x, y) pairs,
and returns a MultiPoint containing only the unique coordinates.

ADR-0033 tiers:
    Tier 1 NVRTC - row-family count/scatter, unique marking, capacity scatter
    Tier 3a CCCL - segmented_sort and exclusive_sum
    Tier 2 CuPy - capacity allocation and device metadata transforms

ADR-0002: CONSTRUCTIVE class - fp64 uniform precision. Coordinates are exact
subsets of input (no arithmetic), so the precision plan is wired through for
observability but stays fp64.

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
    DeviceFixedGeometrySizeMetadata,
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
from vibespatial.runtime.crossover import estimate_physical_work_from_owned
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

request_warmup(
    [
        "exclusive_scan_i32",
        "segmented_sort_asc_f64",
    ]
)

_KERNEL_NAMES = _get_kernel_names()

request_nvrtc_warmup(
    [
        ("extract-unique-points-fp64", KERNEL_SOURCE, _KERNEL_NAMES),
    ]
)


def _compile_kernels():
    """Compile and cache NVRTC kernels."""
    return compile_kernel_group(
        "extract-unique-points-fp64",
        KERNEL_SOURCE,
        _KERNEL_NAMES,
    )


def _family_offsets(device_buffer, dummy):
    part_offsets = device_buffer.part_offsets if device_buffer.part_offsets is not None else dummy
    ring_offsets = device_buffer.ring_offsets if device_buffer.ring_offsets is not None else dummy
    return part_offsets, ring_offsets


def _deduplicate_row_candidate_capacity(
    *,
    runtime,
    kernels,
    d_x,
    d_y,
    d_row_ids,
    d_counts,
    d_offsets,
    row_count: int,
    candidate_capacity: int,
):
    """Deduplicate row-segmented candidate coordinates at fixed capacity."""
    ptr = runtime.pointer
    d_active_total = d_offsets[-1:] + d_counts[-1:]
    d_segment_starts = d_offsets.astype(cp.int32, copy=False)
    d_segment_ends = (d_offsets + d_counts).astype(cp.int32, copy=False)
    d_positions = cp.arange(candidate_capacity, dtype=cp.int32)
    d_active_positions = d_positions < d_active_total[0]

    sort_y = segmented_sort(
        keys=d_y,
        values=d_positions,
        starts=d_segment_starts,
        ends=d_segment_ends,
        num_segments=row_count,
        synchronize=False,
    )
    d_y_permutation = cp.where(d_active_positions, sort_y.values, 0)
    sort_x = segmented_sort(
        keys=d_x[d_y_permutation],
        values=d_y_permutation,
        starts=d_segment_starts,
        ends=d_segment_ends,
        num_segments=row_count,
        synchronize=False,
    )
    d_final_permutation = cp.where(d_active_positions, sort_x.values, 0)
    d_x_sorted = d_x[d_final_permutation]
    d_y_sorted = d_y[d_final_permutation]
    d_row_ids_sorted = d_row_ids[d_final_permutation]

    d_unique_mask = cp.zeros(candidate_capacity, dtype=cp.uint8)
    params = (
        (
            ptr(d_x_sorted),
            ptr(d_y_sorted),
            ptr(d_row_ids_sorted),
            ptr(d_unique_mask),
            ptr(d_active_total),
            candidate_capacity,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(
        kernels["mark_unique_coords"],
        candidate_capacity,
    )
    runtime.launch(
        kernels["mark_unique_coords"],
        grid=grid,
        block=block,
        params=params,
    )

    d_unique_prefix = cp.cumsum(d_unique_mask, dtype=cp.int32)
    d_end_positions = cp.maximum(d_segment_ends - 1, 0)
    d_start_positions = cp.maximum(d_segment_starts - 1, 0)
    d_end_totals = cp.where(
        d_segment_ends > 0,
        d_unique_prefix[d_end_positions],
        0,
    )
    d_start_totals = cp.where(
        d_segment_starts > 0,
        d_unique_prefix[d_start_positions],
        0,
    )
    d_unique_counts = (d_end_totals - d_start_totals).astype(
        cp.int32,
        copy=False,
    )
    d_unique_offsets = cp.empty(row_count + 1, dtype=cp.int32)
    d_unique_offsets[0] = 0
    cp.cumsum(d_unique_counts, out=d_unique_offsets[1:])
    d_x_unique = cp.empty(candidate_capacity, dtype=cp.float64)
    d_y_unique = cp.empty(candidate_capacity, dtype=cp.float64)
    params = (
        (
            ptr(d_x_sorted),
            ptr(d_y_sorted),
            ptr(d_unique_mask),
            ptr(d_unique_prefix),
            ptr(d_x_unique),
            ptr(d_y_unique),
            candidate_capacity,
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
    )
    grid, block = runtime.launch_config(
        kernels["scatter_unique_coords"],
        candidate_capacity,
    )
    runtime.launch(
        kernels["scatter_unique_coords"],
        grid=grid,
        block=block,
        params=params,
    )
    return d_x_unique, d_y_unique, d_unique_offsets, d_unique_counts


@register_kernel_variant(
    "extract_unique_points",
    "gpu-cuda-python",
    kernel_class=KernelClass.CONSTRUCTIVE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point",
        "multipoint",
        "linestring",
        "multilinestring",
        "polygon",
        "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "constructive", "extract_unique_points"),
)
def _extract_unique_points_gpu(
    owned: OwnedGeometryArray,
    active_mask=None,
) -> OwnedGeometryArray:
    """Return unique coordinates using row indirection and device capacity.

    Indexed inputs transform their physical base once and preserve the index
    map. Physical inputs retain input-coordinate capacity through segmented
    sort and unique scatter, so neither active cardinality crosses the host.
    """
    if (
        owned.is_indexed_view
        and owned._base is not None
        and owned._index_map is not None
        and hasattr(owned._index_map, "__cuda_array_interface__")
    ):
        base_active = None
        if active_mask is not None:
            base_active_u8 = cp.zeros(owned._base.row_count, dtype=cp.uint8)
            cp.maximum.at(
                base_active_u8,
                owned._index_map,
                cp.asarray(active_mask, dtype=cp.uint8),
            )
            base_active = base_active_u8.astype(cp.bool_, copy=False)
        extracted_base = _extract_unique_points_gpu(
            owned._base,
            active_mask=base_active,
        )
        return OwnedGeometryArray._indexed_view(extracted_base, owned._index_map)

    runtime = get_cuda_runtime()
    d_state = owned._ensure_device_state()
    row_count = owned.row_count
    d_validity = cp.asarray(d_state.validity, dtype=cp.bool_)
    if active_mask is not None:
        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != row_count:
            raise ValueError("active mask must be one-dimensional and row-aligned")
        d_validity = d_validity & d_active
    if row_count == 0:
        return _build_empty_multipoint_output(row_count, d_validity)

    kernels = _compile_kernels()
    ptr = runtime.pointer
    d_family_codes = cp.asarray(d_state.tags, dtype=cp.int32)
    d_family_row_offsets = cp.asarray(
        d_state.family_row_offsets,
        dtype=cp.int32,
    )
    family_data = tuple(d_state.families.items())
    coordinate_capacity = sum(int(buffer.x.size) for _, buffer in family_data)
    if coordinate_capacity > np.iinfo(np.int32).max:
        raise OverflowError("extract_unique_points coordinate capacity exceeds int32")

    d_counts = cp.zeros(row_count, dtype=cp.int32)
    d_dummy_offsets = cp.zeros(1, dtype=cp.int32)
    for family, buffer in family_data:
        part_offsets, ring_offsets = _family_offsets(buffer, d_dummy_offsets)
        params = (
            (
                ptr(d_validity),
                ptr(d_family_codes),
                ptr(d_family_row_offsets),
                ptr(buffer.geometry_offsets),
                ptr(part_offsets),
                ptr(ring_offsets),
                ptr(buffer.empty_mask),
                ptr(d_counts),
                FAMILY_TAGS[family],
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
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(
            kernels["count_coords_per_row"],
            row_count,
        )
        runtime.launch(
            kernels["count_coords_per_row"],
            grid=grid,
            block=block,
            params=params,
        )

    if coordinate_capacity == 0:
        return _build_empty_multipoint_output(row_count, d_validity)

    d_coord_offsets = exclusive_sum(d_counts, synchronize=False)
    d_x_flat = cp.empty(coordinate_capacity, dtype=cp.float64)
    d_y_flat = cp.empty(coordinate_capacity, dtype=cp.float64)
    d_row_ids = cp.full(coordinate_capacity, row_count, dtype=cp.int32)

    for family, buffer in family_data:
        part_offsets, ring_offsets = _family_offsets(buffer, d_dummy_offsets)
        params = (
            (
                ptr(d_validity),
                ptr(d_family_codes),
                ptr(d_family_row_offsets),
                ptr(buffer.geometry_offsets),
                ptr(part_offsets),
                ptr(ring_offsets),
                ptr(buffer.empty_mask),
                ptr(buffer.x),
                ptr(buffer.y),
                ptr(d_coord_offsets),
                ptr(d_x_flat),
                ptr(d_y_flat),
                ptr(d_row_ids),
                FAMILY_TAGS[family],
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernels["scatter_coords"], row_count)
        runtime.launch(
            kernels["scatter_coords"],
            grid=grid,
            block=block,
            params=params,
        )

    d_x_unique, d_y_unique, d_geometry_offsets, d_unique_counts = (
        _deduplicate_row_candidate_capacity(
            runtime=runtime,
            kernels=kernels,
            d_x=d_x_flat,
            d_y=d_y_flat,
            d_row_ids=d_row_ids,
            d_counts=d_counts,
            d_offsets=d_coord_offsets,
            row_count=row_count,
            candidate_capacity=coordinate_capacity,
        )
    )

    multipoint_tag = cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
    out_tags = cp.where(d_validity, multipoint_tag, cp.int8(-1))
    out_family_row_offsets = cp.where(
        d_validity,
        cp.arange(row_count, dtype=cp.int32),
        cp.int32(-1),
    )
    device_families = {
        GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=d_x_unique,
            y=d_y_unique,
            geometry_offsets=d_geometry_offsets,
            empty_mask=(d_unique_counts == 0).astype(cp.uint8),
        ),
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _build_empty_multipoint_output(row_count: int, validity) -> OwnedGeometryArray:
    """Build an all-empty device MultiPoint without a scalar metadata probe."""
    d_validity = cp.asarray(validity, dtype=cp.bool_)
    out_tags = cp.where(
        d_validity,
        cp.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT]),
        cp.int8(-1),
    )
    out_family_row_offsets = cp.where(
        d_validity,
        cp.arange(row_count, dtype=cp.int32),
        cp.int32(-1),
    )
    device_families = {
        GeometryFamily.MULTIPOINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOINT,
            x=cp.empty(0, dtype=cp.float64),
            y=cp.empty(0, dtype=cp.float64),
            geometry_offsets=cp.zeros(row_count + 1, dtype=cp.int32),
            empty_mask=cp.ones(row_count, dtype=cp.uint8),
        ),
    }
    return build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=out_tags,
        validity=d_validity,
        family_row_offsets=out_family_row_offsets,
        execution_mode="gpu",
    )


def _build_capacity_point_output(d_x, d_y, d_validity) -> OwnedGeometryArray:
    """Build a row-aligned Point carrier with null inactive lanes."""
    row_count = int(d_validity.size)
    d_validity = cp.asarray(d_validity, dtype=cp.bool_)
    d_x = cp.where(d_validity, d_x, cp.nan)
    d_y = cp.where(d_validity, d_y, cp.nan)
    device_families = {
        GeometryFamily.POINT: DeviceFamilyGeometryBuffer(
            family=GeometryFamily.POINT,
            x=d_x,
            y=d_y,
            geometry_offsets=cp.arange(row_count + 1, dtype=cp.int32),
            empty_mask=(~d_validity).astype(cp.uint8),
            fixed_size=DeviceFixedGeometrySizeMetadata(coord_count_per_row=1),
        ),
    }
    result = build_device_resident_owned(
        device_families=device_families,
        row_count=row_count,
        tags=cp.where(
            d_validity,
            cp.int8(FAMILY_TAGS[GeometryFamily.POINT]),
            cp.int8(-1),
        ),
        validity=d_validity,
        family_row_offsets=cp.where(
            d_validity,
            cp.arange(row_count, dtype=cp.int32),
            cp.int32(-1),
        ),
        execution_mode="gpu",
    )
    if result.device_state is not None:
        result.device_state.trusted_homogeneous_family = GeometryFamily.POINT
    return result


def degenerate_line_centroids_owned_capacity(
    owned: OwnedGeometryArray,
    active_mask,
) -> OwnedGeometryArray:
    """Reduce collapsed line parts to exact row-aligned centroid points.

    A zero-length LineString contributes its first coordinate. A zero-length
    MultiLineString contributes the first coordinate of every nonempty part,
    deduplicates those points per row, then averages the unique points. This is
    the GEOS-compatible repair semantics previously expressed through sparse
    ``extract_unique_points`` plus generic centroid dispatch.
    """
    d_active = cp.asarray(active_mask, dtype=cp.bool_)
    if d_active.ndim != 1 or int(d_active.size) != owned.row_count:
        raise ValueError("active mask must be one-dimensional and row-aligned")
    if (
        owned.is_indexed_view
        and owned._base is not None
        and owned._index_map is not None
        and hasattr(owned._index_map, "__cuda_array_interface__")
    ):
        base_active_u8 = cp.zeros(owned._base.row_count, dtype=cp.uint8)
        cp.maximum.at(
            base_active_u8,
            owned._index_map,
            d_active.astype(cp.uint8, copy=False),
        )
        repaired_base = degenerate_line_centroids_owned_capacity(
            owned._base,
            base_active_u8.astype(cp.bool_, copy=False),
        )
        return OwnedGeometryArray._indexed_view(repaired_base, owned._index_map)

    runtime = get_cuda_runtime()
    kernels = _compile_kernels()
    ptr = runtime.pointer
    state = owned._ensure_device_state()
    row_count = owned.row_count
    d_tags = cp.asarray(state.tags, dtype=cp.int32)
    d_family_rows = cp.asarray(state.family_row_offsets, dtype=cp.int32)
    d_validity = cp.asarray(state.validity, dtype=cp.bool_) & d_active
    line_data = tuple(
        (family, state.families[family])
        for family in (
            GeometryFamily.LINESTRING,
            GeometryFamily.MULTILINESTRING,
        )
        if family in state.families
    )
    candidate_capacity = 0
    for family, buffer in line_data:
        if family is GeometryFamily.LINESTRING:
            candidate_capacity += max(int(buffer.geometry_offsets.size) - 1, 0)
        else:
            candidate_capacity += max(int(buffer.part_offsets.size) - 1, 0)
    if candidate_capacity > np.iinfo(np.int32).max:
        raise OverflowError("degenerate line candidate capacity exceeds int32")
    if row_count == 0 or candidate_capacity == 0:
        return _build_capacity_point_output(
            cp.empty(row_count, dtype=cp.float64),
            cp.empty(row_count, dtype=cp.float64),
            cp.zeros(row_count, dtype=cp.bool_),
        )

    d_counts = cp.zeros(row_count, dtype=cp.int32)
    d_dummy_parts = cp.zeros(1, dtype=cp.int32)
    for family, buffer in line_data:
        part_offsets = buffer.part_offsets if buffer.part_offsets is not None else d_dummy_parts
        params = (
            (
                ptr(d_validity),
                ptr(d_tags),
                ptr(d_family_rows),
                ptr(buffer.geometry_offsets),
                ptr(part_offsets),
                ptr(buffer.empty_mask),
                ptr(d_counts),
                FAMILY_TAGS[family],
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
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(
            kernels["count_degenerate_line_candidates"],
            row_count,
        )
        runtime.launch(
            kernels["count_degenerate_line_candidates"],
            grid=grid,
            block=block,
            params=params,
        )

    d_offsets = exclusive_sum(d_counts, synchronize=False)
    d_x_candidates = cp.empty(candidate_capacity, dtype=cp.float64)
    d_y_candidates = cp.empty(candidate_capacity, dtype=cp.float64)
    d_row_ids = cp.full(candidate_capacity, row_count, dtype=cp.int32)
    for family, buffer in line_data:
        part_offsets = buffer.part_offsets if buffer.part_offsets is not None else d_dummy_parts
        params = (
            (
                ptr(d_validity),
                ptr(d_tags),
                ptr(d_family_rows),
                ptr(buffer.geometry_offsets),
                ptr(part_offsets),
                ptr(buffer.empty_mask),
                ptr(buffer.x),
                ptr(buffer.y),
                ptr(d_offsets),
                ptr(d_x_candidates),
                ptr(d_y_candidates),
                ptr(d_row_ids),
                FAMILY_TAGS[family],
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(
            kernels["scatter_degenerate_line_candidates"],
            row_count,
        )
        runtime.launch(
            kernels["scatter_degenerate_line_candidates"],
            grid=grid,
            block=block,
            params=params,
        )

    d_x_unique, d_y_unique, d_unique_offsets, d_unique_counts = _deduplicate_row_candidate_capacity(
        runtime=runtime,
        kernels=kernels,
        d_x=d_x_candidates,
        d_y=d_y_candidates,
        d_row_ids=d_row_ids,
        d_counts=d_counts,
        d_offsets=d_offsets,
        row_count=row_count,
        candidate_capacity=candidate_capacity,
    )
    d_repaired_validity = d_validity & (d_unique_counts > 0)
    d_x_mean = cp.full(row_count, cp.nan, dtype=cp.float64)
    d_y_mean = cp.full(row_count, cp.nan, dtype=cp.float64)
    params = (
        (
            ptr(d_x_unique),
            ptr(d_y_unique),
            ptr(d_unique_offsets),
            ptr(d_repaired_validity),
            ptr(d_x_mean),
            ptr(d_y_mean),
            row_count,
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
    )
    grid, block = runtime.launch_config(kernels["mean_unique_coords"], row_count)
    runtime.launch(
        kernels["mean_unique_coords"],
        grid=grid,
        block=block,
        params=params,
    )
    return _build_capacity_point_output(
        d_x_mean,
        d_y_mean,
        d_repaired_validity,
    )


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
        Precision mode. CONSTRUCTIVE class stays fp64 per ADR-0002;
        wired for observability.

    Returns
    -------
    OwnedGeometryArray
        MultiPoint geometries with unique coordinates per row.
    """
    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    source_work = estimate_physical_work_from_owned(owned)
    selection = plan_dispatch_selection(
        kernel_name="extract_unique_points",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=row_count,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=row_count,
            output_byte_count=source_work.coordinate_count * 16,
            temporary_byte_count=source_work.coordinate_count * 24,
            primary_unit_count=max(row_count, source_work.coordinate_count),
            primary_unit_name="extract-unique-points-coordinate",
        ),
        requested_mode=dispatch_mode,
        requested_precision=precision,
        current_residency=owned.residency,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = selection.precision_plan
        result = _extract_unique_points_gpu(owned)
        record_dispatch_event(
            surface="geopandas.array.extract_unique_points",
            operation="extract_unique_points",
            implementation="extract_unique_points_gpu_nvrtc",
            reason=selection.reason,
            detail=(f"rows={row_count}, precision={precision_plan.compute_precision.value}"),
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


def extract_unique_points_native_tabular_result(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, object] | None = None,
):
    from vibespatial.api._native_results import (
        _unary_constructive_owned_to_native_tabular_result,
    )

    result = extract_unique_points_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="extract_unique_points",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )
