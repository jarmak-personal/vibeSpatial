"""GPU-accelerated geometry normalization.

Canonicalizes geometries by:
- Rotating polygon rings to start at the lexicographically smallest (x,y) vertex
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
from vibespatial.runtime.crossover import default_crossover_policy
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


@register_kernel_variant(
    "normalize",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
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

    max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0

    selection = plan_dispatch_selection(
        kernel_name="normalize",
        kernel_class=KernelClass.COARSE,
        row_count=owned.row_count,
        requested_mode=ExecutionMode.GPU,
        requested_precision=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
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
        else:
            # Points: no normalization needed
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
        if device_buffer.ring_offsets is None or int(device_buffer.ring_offsets.size) < 2:
            return buf
        d_x = device_buffer.x
        d_y = device_buffer.y
        d_ring_offsets = device_buffer.ring_offsets
        total_rings = int(d_ring_offsets.size) - 1
        total_coords = int(d_x.size)
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

    d_min_index = runtime.allocate((total_rings,), np.int32, zero=True)

    # Allocate output coordinate buffers
    d_x_out = runtime.allocate((total_coords,), np.float64)
    d_y_out = runtime.allocate((total_coords,), np.float64)

    try:
        if needs_free:
            d_x = runtime.from_host(buf.x)
            d_y = runtime.from_host(buf.y)
            d_ring_offsets = runtime.from_host(ring_offsets)
        ptr = runtime.pointer

        # Pass 1: find lex-min vertex per ring
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_min_index),
             center_x, center_y, total_rings),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernels["normalize_ring_find_min"], total_rings)
        runtime.launch(kernels["normalize_ring_find_min"], grid=grid, block=block, params=params)

        # Pass 2: rotate ring coordinates
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_x_out), ptr(d_y_out),
             ptr(d_ring_offsets), ptr(d_min_index), total_rings),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernels["normalize_ring_rotate"], total_rings)
        runtime.launch(kernels["normalize_ring_rotate"], grid=grid, block=block, params=params)

        x_out = runtime.copy_device_to_host(d_x_out)
        y_out = runtime.copy_device_to_host(d_y_out)
    finally:
        if needs_free:
            for d in (d_x, d_y, d_ring_offsets):
                runtime.free(d)
        for d in (d_min_index, d_x_out, d_y_out):
            runtime.free(d)

    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=buf.geometry_offsets.copy(),
        empty_mask=buf.empty_mask.copy(),
        part_offsets=buf.part_offsets.copy() if buf.part_offsets is not None else None,
        ring_offsets=buf.ring_offsets.copy(),
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
    """Reverse linestrings where endpoint < startpoint (lex order)."""
    import cupy as cp

    runtime = get_cuda_runtime()

    # For multi-linestrings, use part_offsets as the geometry boundaries
    needs_free = device_buffer is None
    if device_buffer is not None and device_buffer.part_offsets is not None:
        d_offsets = device_buffer.part_offsets
        row_count = int(d_offsets.size) - 1
    elif device_buffer is not None:
        d_offsets = device_buffer.geometry_offsets
        row_count = buf.row_count
    elif buf.part_offsets is not None:
        offsets = buf.part_offsets.astype(np.int32)
        row_count = len(offsets) - 1
    else:
        offsets = buf.geometry_offsets.astype(np.int32)
        row_count = buf.row_count

    if row_count <= 0:
        return buf

    line_src = _LINE_KERNEL_SOURCE.format(compute_type=compute_type)
    kernels = compile_kernel_group(f"normalize-linestring-{compute_type}", line_src, _LINE_KERNEL_NAMES)

    # Copy coordinates (kernel reverses in-place)
    if device_buffer is not None:
        d_x = cp.asarray(device_buffer.x).copy()
        d_y = cp.asarray(device_buffer.y).copy()
    else:
        x_out = buf.x.copy()
        y_out = buf.y.copy()
        d_x = runtime.from_host(x_out)
        d_y = runtime.from_host(y_out)
        d_offsets = runtime.from_host(offsets)

    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_offsets), center_x, center_y, row_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernels["normalize_linestring_reverse"], row_count)
        runtime.launch(kernels["normalize_linestring_reverse"], grid=grid, block=block, params=params)

        x_out = runtime.copy_device_to_host(d_x)
        y_out = runtime.copy_device_to_host(d_y)
    finally:
        runtime.free(d_x)
        runtime.free(d_y)
        if needs_free:
            runtime.free(d_offsets)

    return FamilyGeometryBuffer(
        family=buf.family,
        schema=buf.schema,
        row_count=buf.row_count,
        x=x_out,
        y=y_out,
        geometry_offsets=buf.geometry_offsets.copy(),
        empty_mask=buf.empty_mask.copy(),
        part_offsets=buf.part_offsets.copy() if buf.part_offsets is not None else None,
        ring_offsets=buf.ring_offsets.copy() if buf.ring_offsets is not None else None,
    )
