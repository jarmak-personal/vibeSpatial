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
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    from_shapely_geometries,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

_NORMALIZE_GPU_THRESHOLD = 500

# Precompile both precision variants (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup as _request_nvrtc  # noqa: E402

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

    if selection.selected is ExecutionMode.GPU and row_count >= _NORMALIZE_GPU_THRESHOLD:
        try:
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
        except Exception:
            pass

    return _normalize_cpu(owned)


@register_kernel_variant(
    "normalize",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
    supports_mixed=True,
    tags=("shapely",),
)
def _normalize_cpu(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """CPU fallback: Shapely normalize."""
    import shapely

    record_dispatch_event(
        surface="normalize",
        operation="normalize",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={owned.row_count}",
        selected=ExecutionMode.CPU,
    )
    geoms = owned.to_shapely()
    result = shapely.normalize(np.asarray(geoms, dtype=object))
    return from_shapely_geometries(result.tolist())


@register_kernel_variant(
    "normalize",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon", "linestring", "multilinestring", "point", "multipoint"),
    supports_mixed=True,
    min_rows=_NORMALIZE_GPU_THRESHOLD,
    tags=("cuda-python", "ring-rotation", "linestring-reversal"),
)
def _normalize_gpu(
    owned: OwnedGeometryArray,
    *,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray | None:
    """GPU path: NVRTC ring rotation + linestring reversal."""
    from vibespatial.runtime import RuntimeSelection, has_gpu_runtime
    from vibespatial.runtime.precision import CoordinateStats, select_precision_plan

    if not has_gpu_runtime():
        return None

    # ADR-0002: select compute precision
    runtime_sel = RuntimeSelection(
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
        reason="normalize GPU dispatch",
    )
    max_abs = 0.0
    coord_min, coord_max = np.inf, -np.inf
    for buf in owned.families.values():
        if buf.row_count > 0 and buf.x.size > 0:
            max_abs = max(max_abs, float(np.abs(buf.x).max()), float(np.abs(buf.y).max()))
            coord_min = min(coord_min, float(buf.x.min()), float(buf.y.min()))
            coord_max = max(coord_max, float(buf.x.max()), float(buf.y.max()))
    span = coord_max - coord_min if np.isfinite(coord_min) else 0.0

    precision_plan = select_precision_plan(
        runtime_selection=runtime_sel,
        kernel_class=KernelClass.COARSE,
        requested=precision,
        coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
    )
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

        if family_key in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            new_buf = _normalize_polygon_family_gpu(buf, family_key, compute_type, center_x, center_y)
        elif family_key in (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING):
            new_buf = _normalize_linestring_family_gpu(buf, family_key, compute_type, center_x, center_y)
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


def _normalize_polygon_family_gpu(buf, family, compute_type, center_x, center_y):
    """Rotate all rings in a polygon family to start at lex-smallest vertex."""
    if buf.ring_offsets is None or len(buf.ring_offsets) < 2:
        return buf

    runtime = get_cuda_runtime()
    ring_offsets = buf.ring_offsets.astype(np.int32)
    total_rings = len(ring_offsets) - 1

    if total_rings <= 0:
        return buf

    # Compile kernels for chosen precision
    ring_src = _RING_KERNEL_SOURCE.format(compute_type=compute_type)
    kernels = compile_kernel_group(f"normalize-ring-{compute_type}", ring_src, _RING_KERNEL_NAMES)

    # Upload data
    d_x = runtime.from_host(buf.x)
    d_y = runtime.from_host(buf.y)
    d_ring_offsets = runtime.from_host(ring_offsets)
    d_min_index = runtime.allocate((total_rings,), np.int32, zero=True)

    # Allocate output coordinate buffers
    d_x_out = runtime.allocate((len(buf.x),), np.float64)
    d_y_out = runtime.allocate((len(buf.y),), np.float64)

    try:
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
        for d in (d_x, d_y, d_ring_offsets, d_min_index, d_x_out, d_y_out):
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


def _normalize_linestring_family_gpu(buf, family, compute_type, center_x, center_y):
    """Reverse linestrings where endpoint < startpoint (lex order)."""
    runtime = get_cuda_runtime()

    # For multi-linestrings, use part_offsets as the geometry boundaries
    if buf.part_offsets is not None:
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
