"""GPU-accelerated geometry normalization.

Canonicalizes geometries by:
- Rotating polygon rings to the lexicographically smallest cyclic coordinate order
- Reversing linestrings so the smaller endpoint comes first
- Sorting multi-geometry parts by their first vertex

ADR-0033: Tier 1 NVRTC for ring rotation + lex-min scan, Tier 3a CCCL
for multi-part sorting via segmented_sort.
ADR-0002: COARSE class, dual fp32/fp64. Storage reads stay double,
lex comparison uses compute_t after coordinate centering.
"""

from __future__ import annotations

import numpy as np

from vibespatial.constructive.measurement import _coord_stats_from_owned
from vibespatial.constructive.normalize_cpu import _normalize_cpu
from vibespatial.constructive.normalize_kernels import (
    _LINE_KERNEL_NAMES,
    _LINE_KERNEL_SOURCE,
    _RING_KERNEL_NAMES,
    _RING_KERNEL_SOURCE,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup as _request_nvrtc
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    default_crossover_policy,
    estimate_physical_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

for _ct in ("float", "double"):
    _ring_src = _RING_KERNEL_SOURCE.format(compute_type=_ct)
    _request_nvrtc([(f"normalize-ring-{_ct}", _ring_src, _RING_KERNEL_NAMES)])
    _line_src = _LINE_KERNEL_SOURCE.format(compute_type=_ct)
    _request_nvrtc([(f"normalize-linestring-{_ct}", _line_src, _LINE_KERNEL_NAMES)])


# ---------------------------------------------------------------------------
# Dispatch entry point
# ---------------------------------------------------------------------------


def normalize_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Normalize geometries to canonical form.

    GPU path uses NVRTC kernels for ring rotation and linestring reversal,
    CCCL segmented_sort for multi-part ordering. Falls back to Shapely
    for small inputs or when GPU is unavailable.
    """
    row_count = owned.row_count
    if row_count == 0:
        return owned

    selection = plan_dispatch_selection(
        kernel_name="normalize",
        kernel_class=KernelClass.COARSE,
        row_count=row_count,
        requested_mode=dispatch_mode,
        current_residency=owned.residency,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=row_count,
            primary_unit_name="normalize-coordinate",
        ),
    )

    if selection.selected is ExecutionMode.GPU:
        result = _normalize_gpu(owned, precision=precision)
        if result is not None:
            record_dispatch_event(
                surface="normalize",
                operation="normalize",
                implementation="gpu_nvrtc_ring_rotate",
                reason="GPU ring rotation + linestring reversal",
                detail=f"rows={row_count}",
                selected=ExecutionMode.GPU,
            )
            return result

    return _normalize_cpu(owned)


def normalize_native_tabular_result(
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

    result = normalize_owned(
        owned,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    return _unary_constructive_owned_to_native_tabular_result(
        result,
        operation="normalize",
        crs=crs,
        geometry_name=geometry_name,
        source_rows=source_rows,
        source_tokens=source_tokens,
        attrs=attrs,
    )


@register_kernel_variant(
    "normalize",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "polygon",
        "multipolygon",
        "linestring",
        "multilinestring",
        "point",
        "multipoint",
    ),
    supports_mixed=True,
    min_rows=default_crossover_policy("normalize", KernelClass.COARSE).auto_min_rows,
    tags=("cuda-python", "ring-rotation", "linestring-reversal"),
)
def _normalize_gpu(
    owned: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray | None:
    """GPU path: NVRTC ring rotation + linestring reversal."""
    from vibespatial.runtime import has_gpu_runtime
    from vibespatial.runtime.precision import CoordinateStats

    if not has_gpu_runtime():
        return None

    # Normalization rewrites complete family coordinate spans and therefore
    # requires row-contiguous family buffers.  Indexed carriers may also have
    # independently materialized host stubs after a terminal Shapely export;
    # consuming those stubs alongside the shared device root would mix two
    # different physical layouts.  Resolve the row map on device once before
    # reading either structure.
    if owned.is_indexed_view:
        owned = owned.physicalize_device_rows(allow_capacity_allocation=True)

    max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0

    selection = plan_dispatch_selection(
        kernel_name="normalize",
        kernel_class=KernelClass.COARSE,
        row_count=owned.row_count,
        requested_mode=ExecutionMode.GPU,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        current_residency=owned.residency,
        work_estimate=estimate_physical_work_from_owned(
            owned,
            output_row_count=owned.row_count,
            primary_unit_name="normalize-coordinate",
        ),
    )
    precision_plan = selection.precision_plan
    compute_type = "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan.center_coordinates:
        center_x = float((coord_min + coord_max) * 0.5) if np.isfinite(coord_min) else 0.0
        center_y = center_x  # symmetric centering

    # Build new coordinate buffers for output
    new_families = {}
    for family_key, buf in owned.families.items():
        if buf.row_count == 0:
            new_families[family_key] = buf
            continue
        device_buffer = None
        if owned.device_state is not None and family_key in owned.device_state.families:
            device_buffer = owned.device_state.families[family_key]
            owned._ensure_host_family_structure(family_key)
            buf = owned.families[family_key]

        if family_key in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            new_buf = _normalize_polygon_family_gpu(
                buf,
                family_key,
                compute_type,
                center_x,
                center_y,
                device_buffer=device_buffer,
            )
        elif family_key in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
            new_buf = _normalize_linestring_family_gpu(
                buf,
                family_key,
                compute_type,
                center_x,
                center_y,
                device_buffer=device_buffer,
            )
        elif family_key in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
            new_buf = _normalize_point_family_gpu(
                buf,
                family_key,
                device_buffer=device_buffer,
            )
        else:
            new_buf = buf

        new_families[family_key] = new_buf

    return OwnedGeometryArray(
        validity=owned.validity.copy(),
        tags=owned.tags.copy(),
        family_row_offsets=owned.family_row_offsets.copy(),
        families=new_families,
    )


def _normalize_polygon_family_gpu(
    buf,
    family,
    compute_type,
    center_x,
    center_y,
    *,
    device_buffer=None,
):
    """Rotate all rings in a polygon family to start at lex-smallest vertex."""
    runtime = get_cuda_runtime()
    needs_free = device_buffer is None
    if device_buffer is not None:
        if (
            device_buffer.ring_offsets is None
            or buf.ring_offsets is None
            or int(buf.ring_offsets.size) < 2
        ):
            return buf
        active_geometry_count = int(buf.row_count)
        active_part_count = None
        if family is GeometryFamily.POLYGON:
            active_ring_count = int(buf.geometry_offsets[active_geometry_count])
        else:
            if device_buffer.part_offsets is None or buf.part_offsets is None:
                return buf
            active_part_count = int(buf.geometry_offsets[active_geometry_count])
            active_ring_count = int(buf.part_offsets[active_part_count])
        active_coord_count = int(buf.ring_offsets[active_ring_count])
        d_x = device_buffer.x[:active_coord_count]
        d_y = device_buffer.y[:active_coord_count]
        d_ring_offsets = device_buffer.ring_offsets[: active_ring_count + 1]
        d_geometry_offsets_source = device_buffer.geometry_offsets[: active_geometry_count + 1]
        d_part_offsets_source = None
        if family is GeometryFamily.MULTIPOLYGON:
            d_part_offsets_source = device_buffer.part_offsets[: active_part_count + 1]
        total_rings = active_ring_count
        total_coords = active_coord_count
    else:
        if buf.ring_offsets is None or len(buf.ring_offsets) < 2:
            return buf
        ring_offsets = buf.ring_offsets.astype(np.int32)
        total_rings = len(ring_offsets) - 1
        total_coords = len(buf.x)

    if total_rings <= 0:
        return buf

    # Compile kernels for chosen precision
    ring_src = _RING_KERNEL_SOURCE.format(compute_type=compute_type)
    kernels = compile_kernel_group(f"normalize-ring-{compute_type}", ring_src, _RING_KERNEL_NAMES)

    # Allocate output coordinate buffers
    d_x_out = runtime.allocate((total_coords,), np.float64)
    d_y_out = runtime.allocate((total_coords,), np.float64)
    hierarchy_inputs_to_free = []
    hierarchy_outputs_to_free = []
    d_is_exterior = None

    try:
        if needs_free:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring_offsets = runtime.from_host(ring_offsets)
        if device_buffer is not None:
            d_geometry_offsets_input = d_geometry_offsets_source
            d_part_offsets_input = d_part_offsets_source
        else:
            d_geometry_offsets_input = runtime.from_host(
                buf.geometry_offsets.astype(np.int32, copy=False)
            )
            hierarchy_inputs_to_free.append(d_geometry_offsets_input)
            d_part_offsets_input = None
            if family is GeometryFamily.MULTIPOLYGON:
                d_part_offsets_input = runtime.from_host(
                    buf.part_offsets.astype(np.int32, copy=False)
                )
                hierarchy_inputs_to_free.append(d_part_offsets_input)

        import cupy as cp

        d_is_exterior = cp.zeros(total_rings, dtype=cp.uint8)
        d_shell_offsets = (
            d_geometry_offsets_input if family is GeometryFamily.POLYGON else d_part_offsets_input
        )
        d_is_exterior[cp.asarray(d_shell_offsets, dtype=cp.int64)[:-1]] = 1
        ptr = runtime.pointer

        params = (
            (
                ptr(d_x),
                ptr(d_y),
                ptr(d_x_out),
                ptr(d_y_out),
                ptr(d_ring_offsets),
                ptr(d_is_exterior),
                center_x,
                center_y,
                total_rings,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernels["normalize_ring_rotate"], total_rings)
        runtime.launch(kernels["normalize_ring_rotate"], grid=grid, block=block, params=params)

        d_rotated_x = d_x_out
        d_rotated_y = d_y_out
        (
            d_canonical_x,
            d_canonical_y,
            d_geometry_offsets_out,
            d_part_offsets_out,
            d_ring_offsets_out,
        ) = _canonicalize_polygon_hierarchy_device(
            family,
            d_rotated_x,
            d_rotated_y,
            d_ring_offsets,
            d_geometry_offsets_input,
            d_part_offsets_input,
            total_coords=total_coords,
        )
        if d_canonical_x is not d_rotated_x:
            runtime.free(d_rotated_x)
        if d_canonical_y is not d_rotated_y:
            runtime.free(d_rotated_y)
        d_x_out = d_canonical_x
        d_y_out = d_canonical_y
        for output, source in (
            (d_geometry_offsets_out, d_geometry_offsets_input),
            (d_part_offsets_out, d_part_offsets_input),
            (d_ring_offsets_out, d_ring_offsets),
        ):
            if output is not None and output is not source:
                hierarchy_outputs_to_free.append(output)

        x_out = runtime.copy_device_to_host(
            d_x_out,
            reason=f"normalize {buf.family.value} x-coordinate host export",
        )
        y_out = runtime.copy_device_to_host(
            d_y_out,
            reason=f"normalize {buf.family.value} y-coordinate host export",
        )
        geometry_offsets_out = runtime.copy_device_to_host(
            d_geometry_offsets_out,
            reason=f"normalize {buf.family.value} geometry offsets host export",
        ).astype(np.int32, copy=False)
        part_offsets_out = (
            None
            if d_part_offsets_out is None
            else runtime.copy_device_to_host(
                d_part_offsets_out,
                reason=f"normalize {buf.family.value} part offsets host export",
            ).astype(np.int32, copy=False)
        )
        ring_offsets_out = runtime.copy_device_to_host(
            d_ring_offsets_out,
            reason=f"normalize {buf.family.value} ring offsets host export",
        ).astype(np.int32, copy=False)
    finally:
        if needs_free:
            for d in (d_x, d_y, d_ring_offsets):
                runtime.free(d)
        for d in hierarchy_inputs_to_free:
            runtime.free(d)
        for d in hierarchy_outputs_to_free:
            runtime.free(d)
        runtime.free(d_is_exterior)
        for d in (d_x_out, d_y_out):
            runtime.free(d)

    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=geometry_offsets_out,
        empty_mask=buf.empty_mask.copy(),
        part_offsets=part_offsets_out,
        ring_offsets=ring_offsets_out,
    )


def _canonicalize_polygon_hierarchy_device(
    family,
    d_x,
    d_y,
    d_ring_offsets,
    d_geometry_offsets,
    d_part_offsets,
    *,
    total_coords: int,
):
    """Order polygon rings and parts in GEOS canonical descending order.

    Rings have already been rotated and direction-normalized. Valid polygon
    components cannot share their first directed edge, so the first two
    vertices are an exact ordering key for valid inputs; original position is
    retained only for structurally identical/invalid ties.
    """
    import cupy as cp

    from vibespatial.geometry.owned import _device_gather_xy_offset_slices

    d_ring_offsets = cp.asarray(d_ring_offsets, dtype=cp.int64)
    d_geometry_offsets = cp.asarray(d_geometry_offsets, dtype=cp.int32)
    ring_count = int(d_ring_offsets.size) - 1
    if ring_count <= 1:
        return d_x, d_y, d_geometry_offsets, d_part_offsets, d_ring_offsets.astype(cp.int32)

    d_ring_ids = cp.arange(ring_count, dtype=cp.int32)
    d_ring_starts = d_ring_offsets[:-1]
    d_ring_lengths = d_ring_offsets[1:] - d_ring_starts
    d_second = d_ring_starts + cp.minimum(cp.maximum(d_ring_lengths - 2, 0), 1)
    d_first_x = cp.asarray(d_x)[d_ring_starts]
    d_first_y = cp.asarray(d_y)[d_ring_starts]
    d_second_x = cp.asarray(d_x)[d_second]
    d_second_y = cp.asarray(d_y)[d_second]

    if family is GeometryFamily.POLYGON:
        d_ring_owner = cp.searchsorted(
            d_geometry_offsets[1:],
            d_ring_ids,
            side="right",
        ).astype(cp.int32, copy=False)
        d_shell = d_ring_ids == d_geometry_offsets[d_ring_owner]
        d_ring_order = cp.lexsort(
            cp.stack(
                (
                    d_ring_ids,
                    -d_second_y,
                    -d_second_x,
                    -d_first_y,
                    -d_first_x,
                    (~d_shell).astype(cp.int8),
                    d_ring_owner,
                )
            )
        ).astype(cp.int32, copy=False)
        d_new_x, d_new_y, d_new_ring_offsets = _device_gather_xy_offset_slices(
            d_x,
            d_y,
            d_ring_offsets,
            d_ring_order,
            precomputed_total=total_coords,
        )
        return (
            d_new_x,
            d_new_y,
            d_geometry_offsets,
            None,
            d_new_ring_offsets,
        )

    d_part_offsets = cp.asarray(d_part_offsets, dtype=cp.int32)
    part_count = int(d_part_offsets.size) - 1
    if part_count <= 0:
        return d_x, d_y, d_geometry_offsets, d_part_offsets, d_ring_offsets.astype(cp.int32)
    d_part_ids = cp.arange(part_count, dtype=cp.int32)
    d_part_owner = cp.searchsorted(
        d_geometry_offsets[1:],
        d_part_ids,
        side="right",
    ).astype(cp.int32, copy=False)
    d_shell_rings = d_part_offsets[:-1]
    d_part_order = cp.lexsort(
        cp.stack(
            (
                d_part_ids,
                -d_second_y[d_shell_rings],
                -d_second_x[d_shell_rings],
                -d_first_y[d_shell_rings],
                -d_first_x[d_shell_rings],
                d_part_owner,
            )
        )
    ).astype(cp.int32, copy=False)
    d_part_rank = cp.empty(part_count, dtype=cp.int32)
    d_part_rank[d_part_order] = cp.arange(part_count, dtype=cp.int32)

    d_ring_owner = cp.searchsorted(
        d_part_offsets[1:],
        d_ring_ids,
        side="right",
    ).astype(cp.int32, copy=False)
    d_shell = d_ring_ids == d_part_offsets[d_ring_owner]
    d_ring_order = cp.lexsort(
        cp.stack(
            (
                d_ring_ids,
                -d_second_y,
                -d_second_x,
                -d_first_y,
                -d_first_x,
                (~d_shell).astype(cp.int8),
                d_part_rank[d_ring_owner],
            )
        )
    ).astype(cp.int32, copy=False)
    d_new_x, d_new_y, d_new_ring_offsets = _device_gather_xy_offset_slices(
        d_x,
        d_y,
        d_ring_offsets,
        d_ring_order,
        precomputed_total=total_coords,
    )
    d_ring_counts = cp.diff(d_part_offsets)[d_part_order]
    d_new_part_offsets = cp.empty(part_count + 1, dtype=cp.int32)
    d_new_part_offsets[0] = 0
    d_new_part_offsets[1:] = cp.cumsum(d_ring_counts, dtype=cp.int32)
    return (
        d_new_x,
        d_new_y,
        d_geometry_offsets,
        d_new_part_offsets,
        d_new_ring_offsets,
    )


def _normalize_linestring_family_gpu(
    buf,
    family,
    compute_type,
    center_x,
    center_y,
    *,
    device_buffer=None,
):
    """Normalize line direction and MultiLineString component ordering."""
    import cupy as cp

    runtime = get_cuda_runtime()

    needs_free = device_buffer is None
    geometry_count = int(buf.row_count)
    if family is GeometryFamily.MULTILINESTRING:
        part_count = int(buf.geometry_offsets[geometry_count])
        coord_count = int(buf.part_offsets[part_count])
        max_part_coords = (
            int(np.diff(buf.part_offsets[: part_count + 1]).max())
            if part_count > 0
            else 0
        )
    else:
        part_count = geometry_count
        coord_count = int(buf.geometry_offsets[geometry_count])
        max_part_coords = 0

    if part_count <= 0:
        return buf

    line_src = _LINE_KERNEL_SOURCE.format(compute_type=compute_type)
    kernels = compile_kernel_group(
        f"normalize-linestring-{compute_type}", line_src, _LINE_KERNEL_NAMES
    )

    hierarchy_inputs_to_free = []
    hierarchy_outputs_to_free = []
    if device_buffer is not None:
        d_x = cp.asarray(device_buffer.x[:coord_count]).copy()
        d_y = cp.asarray(device_buffer.y[:coord_count]).copy()
        d_geometry_offsets = device_buffer.geometry_offsets[: geometry_count + 1]
        d_part_offsets = (
            device_buffer.part_offsets[: part_count + 1]
            if family is GeometryFamily.MULTILINESTRING
            else None
        )
    else:
        d_x = runtime.from_host(buf.x[:coord_count])
        d_y = runtime.from_host(buf.y[:coord_count])
        d_geometry_offsets = runtime.from_host(
            buf.geometry_offsets[: geometry_count + 1].astype(np.int32, copy=False)
        )
        hierarchy_inputs_to_free.append(d_geometry_offsets)
        d_part_offsets = None
        if family is GeometryFamily.MULTILINESTRING:
            d_part_offsets = runtime.from_host(
                buf.part_offsets[: part_count + 1].astype(np.int32, copy=False)
            )
            hierarchy_inputs_to_free.append(d_part_offsets)

    d_line_offsets = (
        d_part_offsets
        if family is GeometryFamily.MULTILINESTRING
        else d_geometry_offsets
    )

    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_line_offsets), center_x, center_y, part_count),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(
            kernels["normalize_linestring_reverse"],
            part_count,
        )
        runtime.launch(
            kernels["normalize_linestring_reverse"], grid=grid, block=block, params=params
        )

        d_part_offsets_out = d_part_offsets
        if family is GeometryFamily.MULTILINESTRING:
            d_x_out, d_y_out, d_part_offsets_out = (
                _canonicalize_multilinestring_hierarchy_device(
                    d_x,
                    d_y,
                    d_geometry_offsets,
                    d_part_offsets,
                    total_coords=coord_count,
                    max_part_coords=max_part_coords,
                )
            )
            if d_x_out is not d_x:
                runtime.free(d_x)
                d_x = d_x_out
            if d_y_out is not d_y:
                runtime.free(d_y)
                d_y = d_y_out
            if d_part_offsets_out is not d_part_offsets:
                hierarchy_outputs_to_free.append(d_part_offsets_out)

        x_out = runtime.copy_device_to_host(
            d_x,
            reason=f"normalize {family.value} x-coordinate host export",
        )
        y_out = runtime.copy_device_to_host(
            d_y,
            reason=f"normalize {family.value} y-coordinate host export",
        )
        part_offsets_out = (
            None
            if d_part_offsets_out is None
            else runtime.copy_device_to_host(
                d_part_offsets_out,
                reason=f"normalize {family.value} part offsets host export",
            ).astype(np.int32, copy=False)
        )
    finally:
        runtime.free(d_x)
        runtime.free(d_y)
        for allocation in hierarchy_outputs_to_free:
            runtime.free(allocation)
        if needs_free:
            for allocation in hierarchy_inputs_to_free:
                runtime.free(allocation)

    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=buf.geometry_offsets[: geometry_count + 1].copy(),
        empty_mask=buf.empty_mask.copy(),
        part_offsets=part_offsets_out,
        ring_offsets=buf.ring_offsets.copy() if buf.ring_offsets is not None else None,
    )


def _dense_lexicographic_ranks_device(d_first, d_second):
    """Return dense ascending ranks for exact pairs of device values."""
    import cupy as cp

    item_count = int(d_first.size)
    if item_count == 0:
        return cp.empty(0, dtype=cp.int32)
    d_ids = cp.arange(item_count, dtype=cp.int32)
    d_order = cp.lexsort(cp.stack((d_ids, d_second, d_first))).astype(
        cp.int32,
        copy=False,
    )
    d_sorted_first = d_first[d_order]
    d_sorted_second = d_second[d_order]
    d_boundaries = cp.empty(item_count, dtype=cp.int32)
    d_boundaries[0] = 1
    if item_count > 1:
        d_boundaries[1:] = (
            (d_sorted_first[1:] != d_sorted_first[:-1])
            | (d_sorted_second[1:] != d_sorted_second[:-1])
        )
    d_sorted_ranks = cp.cumsum(d_boundaries, dtype=cp.int32) - 1
    d_ranks = cp.empty(item_count, dtype=cp.int32)
    d_ranks[d_order] = d_sorted_ranks
    return d_ranks


def _line_coordinate_sequence_ranks_device(
    d_x,
    d_y,
    d_part_offsets,
    *,
    max_part_coords: int,
):
    """Rank complete variable-length coordinate sequences exactly on device."""
    import cupy as cp

    d_offsets = cp.asarray(d_part_offsets, dtype=cp.int64)
    coord_count = int(d_x.size)
    if coord_count == 0:
        return cp.empty(int(d_offsets.size) - 1, dtype=cp.int32)
    d_coord_ids = cp.arange(coord_count, dtype=cp.int64)
    d_owner = cp.searchsorted(d_offsets[1:], d_coord_ids, side="right")
    d_position = d_coord_ids - d_offsets[d_owner]
    d_lengths = cp.diff(d_offsets)
    d_ranks = _dense_lexicographic_ranks_device(
        cp.asarray(d_x),
        cp.asarray(d_y),
    )

    width = 1
    while width < max_part_coords:
        d_has_second = d_position + width < d_lengths[d_owner]
        d_second_index = cp.minimum(d_coord_ids + width, coord_count - 1)
        d_second_rank = cp.where(d_has_second, d_ranks[d_second_index], -1)
        d_ranks = _dense_lexicographic_ranks_device(d_ranks, d_second_rank)
        width *= 2
    return d_ranks[d_offsets[:-1]]


def _canonicalize_multilinestring_hierarchy_device(
    d_x,
    d_y,
    d_geometry_offsets,
    d_part_offsets,
    *,
    total_coords: int,
    max_part_coords: int,
):
    """Order MultiLineString parts by the exact GEOS comparison contract."""
    import cupy as cp

    from vibespatial.geometry.owned import _device_gather_xy_offset_slices

    d_geometry_offsets = cp.asarray(d_geometry_offsets, dtype=cp.int32)
    d_part_offsets = cp.asarray(d_part_offsets, dtype=cp.int64)
    part_count = int(d_part_offsets.size) - 1
    if part_count <= 1:
        return d_x, d_y, d_part_offsets.astype(cp.int32)

    d_part_ids = cp.arange(part_count, dtype=cp.int32)
    d_owner = cp.searchsorted(
        d_geometry_offsets[1:],
        d_part_ids,
        side="right",
    ).astype(cp.int32, copy=False)
    d_lengths = cp.diff(d_part_offsets).astype(cp.int32, copy=False)
    d_sequence_ranks = _line_coordinate_sequence_ranks_device(
        d_x,
        d_y,
        d_part_offsets,
        max_part_coords=max_part_coords,
    )
    d_part_order = cp.lexsort(
        cp.stack(
            (
                d_part_ids,
                -d_sequence_ranks,
                -d_lengths,
                d_owner,
            )
        )
    ).astype(cp.int32, copy=False)
    return _device_gather_xy_offset_slices(
        d_x,
        d_y,
        d_part_offsets,
        d_part_order,
        precomputed_total=total_coords,
    )


def _normalize_point_family_gpu(buf, family, *, device_buffer=None):
    """Physicalize points and order MultiPoint members canonically."""
    import cupy as cp

    runtime = get_cuda_runtime()
    geometry_count = int(buf.row_count)
    coord_count = int(buf.geometry_offsets[geometry_count])

    if device_buffer is not None:
        d_x = cp.asarray(device_buffer.x[:coord_count])
        d_y = cp.asarray(device_buffer.y[:coord_count])
        d_geometry_offsets = cp.asarray(
            device_buffer.geometry_offsets[: geometry_count + 1],
            dtype=cp.int32,
        )
    else:
        d_x = cp.asarray(buf.x[:coord_count])
        d_y = cp.asarray(buf.y[:coord_count])
        d_geometry_offsets = cp.asarray(
            buf.geometry_offsets[: geometry_count + 1],
            dtype=cp.int32,
        )
    d_order = cp.arange(coord_count, dtype=cp.int32)
    if family is GeometryFamily.MULTIPOINT and coord_count > 1:
        d_owner = cp.searchsorted(
            d_geometry_offsets[1:],
            d_order,
            side="right",
        ).astype(cp.int32, copy=False)
        d_order = cp.lexsort(
            cp.stack((d_order, -d_y, -d_x, d_owner))
        ).astype(cp.int32, copy=False)
    x_out = runtime.copy_device_to_host(
        d_x[d_order],
        reason=f"normalize {family.value} x-coordinate host export",
    )
    y_out = runtime.copy_device_to_host(
        d_y[d_order],
        reason=f"normalize {family.value} y-coordinate host export",
    )
    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=buf.geometry_offsets[: geometry_count + 1].copy(),
        empty_mask=buf.empty_mask.copy(),
        part_offsets=None,
        ring_offsets=None,
    )
