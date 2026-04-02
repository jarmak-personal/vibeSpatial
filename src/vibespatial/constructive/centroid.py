"""GPU-accelerated centroid computation for all geometry types.

Tier 1 NVRTC kernels (ADR-0033) for computing geometric centroids
directly from OwnedGeometryArray coordinate buffers.  ADR-0002 METRIC class
precision dispatch: fp32 + Kahan + coordinate centering on consumer GPUs,
native fp64 on datacenter GPUs.

Covers Point, MultiPoint, LineString, MultiLineString, Polygon, and
MultiPolygon families.  Polygon/MultiPolygon kernel source is imported
from polygon_constructive.py (not duplicated).

Zero host/device transfers mid-process.  When data is already device-resident
(vibeFrame path), kernels read directly from DeviceFamilyGeometryBuffer
pointers with no copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    _compile_precision_kernel,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

from .measurement import _coord_stats_from_owned
from .polygon import (
    _POLYGON_CENTROID_KERNEL_NAMES,
    _polygon_centroids_cpu,
)

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan

from vibespatial.constructive.centroid_kernels import (
    _LINESTRING_CENTROID_FP32,
    _LINESTRING_CENTROID_FP64,
    _MULTILINESTRING_CENTROID_FP32,
    _MULTILINESTRING_CENTROID_FP64,
    _MULTIPOINT_CENTROID_FP32,
    _MULTIPOINT_CENTROID_FP64,
    _POINT_CENTROID_FP32,
    _POINT_CENTROID_FP64,
    _POLYGON_CENTROID_COOPERATIVE_FP32,
    _POLYGON_CENTROID_COOPERATIVE_FP64,
    _POLYGON_CENTROID_FP32,
    _POLYGON_CENTROID_FP64,
)

# ---------------------------------------------------------------------------
# Kernel names and precompiled source variants
# ---------------------------------------------------------------------------

_POINT_CENTROID_NAMES = ("point_centroid",)
_MULTIPOINT_CENTROID_NAMES = ("multipoint_centroid",)
_LINESTRING_CENTROID_NAMES = ("linestring_centroid",)
_MULTILINESTRING_CENTROID_NAMES = ("multilinestring_centroid",)
_POLYGON_CENTROID_COOPERATIVE_NAMES = ("polygon_centroid_cooperative",)


# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("centroid-point-fp64", _POINT_CENTROID_FP64, _POINT_CENTROID_NAMES),
    ("centroid-point-fp32", _POINT_CENTROID_FP32, _POINT_CENTROID_NAMES),
    ("centroid-multipoint-fp64", _MULTIPOINT_CENTROID_FP64, _MULTIPOINT_CENTROID_NAMES),
    ("centroid-multipoint-fp32", _MULTIPOINT_CENTROID_FP32, _MULTIPOINT_CENTROID_NAMES),
    ("centroid-linestring-fp64", _LINESTRING_CENTROID_FP64, _LINESTRING_CENTROID_NAMES),
    ("centroid-linestring-fp32", _LINESTRING_CENTROID_FP32, _LINESTRING_CENTROID_NAMES),
    ("centroid-multilinestring-fp64", _MULTILINESTRING_CENTROID_FP64, _MULTILINESTRING_CENTROID_NAMES),
    ("centroid-multilinestring-fp32", _MULTILINESTRING_CENTROID_FP32, _MULTILINESTRING_CENTROID_NAMES),
    ("centroid-polygon-cooperative-fp64", _POLYGON_CENTROID_COOPERATIVE_FP64, _POLYGON_CENTROID_COOPERATIVE_NAMES),
    ("centroid-polygon-cooperative-fp32", _POLYGON_CENTROID_COOPERATIVE_FP32, _POLYGON_CENTROID_COOPERATIVE_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _compile_kernel(name_prefix: str, fp64_source: str, fp32_source: str,
                    kernel_names: tuple[str, ...], compute_type: str = "double"):
    return _compile_precision_kernel(
        name_prefix,
        fp64_source,
        fp32_source,
        kernel_names,
        compute_type,
    )


def _compile_polygon_centroid_kernel(compute_type: str = "double"):
    source = _POLYGON_CENTROID_FP64 if compute_type == "double" else _POLYGON_CENTROID_FP32
    suffix = "fp64" if compute_type == "double" else "fp32"
    return compile_kernel_group(f"polygon-centroid-{suffix}", source, _POLYGON_CENTROID_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# GPU implementation: Centroid
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_centroid",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "centroid", "kahan", "centered"),
)
def _centroid_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> OwnedGeometryArray:
    """GPU-accelerated centroid for all geometry families.

    Returns device-resident Point OwnedGeometryArray.
    Respects ADR-0002 precision dispatch.
    """
    from vibespatial.runtime.precision import PrecisionMode

    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            for buf in owned.families.values():
                if buf.row_count > 0 and buf.x.size > 0:
                    center_x = float((buf.x.min() + buf.x.max()) * 0.5)
                    center_y = float((buf.y.min() + buf.y.max()) * 0.5)
                    break

    runtime = get_cuda_runtime()
    row_count = owned.row_count

    # Allocate global device output arrays initialized to NaN
    d_cx = runtime.allocate((row_count,), np.float64)
    d_cy = runtime.allocate((row_count,), np.float64)
    d_cx[:] = cp.nan
    d_cy[:] = cp.nan

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # --- Point family ---
    _launch_point_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
    )

    # --- MultiPoint family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
        GeometryFamily.MULTIPOINT, "multipoint_centroid",
        _MULTIPOINT_CENTROID_FP64, _MULTIPOINT_CENTROID_FP32,
        _MULTIPOINT_CENTROID_NAMES, "centroid-multipoint",
        has_part_offsets=False,
    )

    # --- LineString family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
        GeometryFamily.LINESTRING, "linestring_centroid",
        _LINESTRING_CENTROID_FP64, _LINESTRING_CENTROID_FP32,
        _LINESTRING_CENTROID_NAMES, "centroid-linestring",
        has_part_offsets=False,
    )

    # --- MultiLineString family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
        GeometryFamily.MULTILINESTRING, "multilinestring_centroid",
        _MULTILINESTRING_CENTROID_FP64, _MULTILINESTRING_CENTROID_FP32,
        _MULTILINESTRING_CENTROID_NAMES, "centroid-multilinestring",
        has_part_offsets=True,
    )

    # --- Polygon family ---
    _launch_polygon_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
        GeometryFamily.POLYGON,
    )

    # --- MultiPolygon family ---
    _launch_polygon_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        d_cx, d_cy, center_x, center_y, compute_type,
        GeometryFamily.MULTIPOLYGON,
    )

    # Mask invalid rows as NaN on device
    d_validity = cp.asarray(owned.validity)
    d_cx[~d_validity] = cp.nan
    d_cy[~d_validity] = cp.nan

    # Build device-resident Point OwnedGeometryArray
    from vibespatial.constructive.point import _build_device_backed_point_output

    return _build_device_backed_point_output(d_cx, d_cy, row_count=row_count)


def _launch_point_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    d_out_cx, d_out_cy, center_x, center_y, compute_type,
):
    """Launch point centroid kernel (identity copy).

    Scatters results directly into device output arrays ``d_out_cx``/``d_out_cy``.
    """
    tag = FAMILY_TAGS[GeometryFamily.POINT]
    mask = tags == tag
    if not np.any(mask) or not owned.family_has_rows(GeometryFamily.POINT):
        return
    buf = owned.families[GeometryFamily.POINT]

    kernels = _compile_kernel(
        "centroid-point", _POINT_CENTROID_FP64, _POINT_CENTROID_FP32,
        _POINT_CENTROID_NAMES, compute_type,
    )
    kernel = kernels["point_centroid"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = device_state is None or GeometryFamily.POINT not in (device_state.families if device_state else {})
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.POINT]
        d_x, d_y = ds.x, ds.y
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_geom])

    d_cx = runtime.allocate((n,), np.float64)
    d_cy = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_geom),
             ptr(d_cx), ptr(d_cy), center_x, center_y, n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        # Device-side scatter into global output arrays
        d_global = cp.asarray(global_rows)
        d_family = cp.asarray(family_rows)
        d_out_cx[d_global] = d_cx[d_family]
        d_out_cy[d_global] = d_cy[d_family]
    finally:
        runtime.free(d_cx)
        runtime.free(d_cy)
        for d in allocated:
            runtime.free(d)


def _launch_simple_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    d_out_cx, d_out_cy, center_x, center_y, compute_type,
    family, kernel_name, fp64_source, fp32_source, kernel_names, prefix,
    has_part_offsets,
):
    """Launch centroid kernel for MultiPoint, LineString, or MultiLineString.

    Scatters results directly into device output arrays ``d_out_cx``/``d_out_cy``.
    """
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or not owned.family_has_rows(family):
        return
    buf = owned.families[family]
    if has_part_offsets:
        has_parts = (
            (device_state is not None and family in device_state.families
             and device_state.families[family].part_offsets is not None)
            or buf.part_offsets is not None
        )
        if not has_parts:
            return

    kernels = _compile_kernel(prefix, fp64_source, fp32_source, kernel_names, compute_type)
    kernel = kernels[kernel_name]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = device_state is None or family not in (device_state.families if device_state else {})
    allocated = []
    if not needs_free:
        ds = device_state.families[family]
        d_x, d_y = ds.x, ds.y
        d_geom = ds.geometry_offsets
        d_part = ds.part_offsets if has_part_offsets else None
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_geom])
        if has_part_offsets:
            d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
            allocated.append(d_part)
        else:
            d_part = None

    d_cx = runtime.allocate((n,), np.float64)
    d_cy = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        if has_part_offsets:
            params = (
                (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom),
                 ptr(d_cx), ptr(d_cy), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32),
            )
        else:
            params = (
                (ptr(d_x), ptr(d_y), ptr(d_geom),
                 ptr(d_cx), ptr(d_cy), center_x, center_y, n),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32),
            )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        # Device-side scatter into global output arrays
        d_global = cp.asarray(global_rows)
        d_family = cp.asarray(family_rows)
        d_out_cx[d_global] = d_cx[d_family]
        d_out_cy[d_global] = d_cy[d_family]
    finally:
        runtime.free(d_cx)
        runtime.free(d_cy)
        for d in allocated:
            runtime.free(d)


def _launch_polygon_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    d_out_cx, d_out_cy, center_x, center_y, compute_type,
    family,
):
    """Launch polygon centroid kernel (Polygon or MultiPolygon).

    Scatters results directly into device output arrays ``d_out_cx``/``d_out_cy``.
    """
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or not owned.family_has_rows(family):
        return
    buf = owned.families[family]

    n = buf.row_count

    # Choose cooperative vs simple kernel based on avg vertex count.
    # Cooperative kernel only applies to Polygon family (not MultiPolygon,
    # which uses the same per-thread polygon_centroid kernel).
    # Read coordinate count from authoritative side (device stubs have x.size==0).
    if device_state is not None and family in device_state.families:
        coord_count = int(device_state.families[family].x.size)
    else:
        coord_count = buf.x.size
    avg_verts = coord_count / max(n, 1) if n > 0 else 0
    use_cooperative = avg_verts >= 64 and family is GeometryFamily.POLYGON

    if use_cooperative:
        coop_kernels = _compile_kernel(
            "centroid-polygon-cooperative",
            _POLYGON_CENTROID_COOPERATIVE_FP64, _POLYGON_CENTROID_COOPERATIVE_FP32,
            _POLYGON_CENTROID_COOPERATIVE_NAMES, compute_type,
        )
        kernel = coop_kernels["polygon_centroid_cooperative"]
    else:
        kernels = _compile_polygon_centroid_kernel(compute_type)
        kernel = kernels["polygon_centroid"]

    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]

    needs_free = device_state is None or family not in (device_state.families if device_state else {})
    allocated = []
    if not needs_free:
        ds = device_state.families[family]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_geom])

    d_cx = runtime.allocate((n,), np.float64)
    d_cy = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom),
             ptr(d_cx), ptr(d_cy), center_x, center_y, n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
             KERNEL_PARAM_I32),
        )
        if use_cooperative:
            # 1 block per geometry; fixed at 256 to match __launch_bounds__(256, 4)
            # and shared memory sized for 8 warps (256 / 32).
            grid = (n, 1, 1)
            block = (256, 1, 1)
        else:
            grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        # Device-side scatter into global output arrays
        d_global = cp.asarray(global_rows)
        d_family = cp.asarray(family_rows)
        d_out_cx[d_global] = d_cx[d_family]
        d_out_cy[d_global] = d_cy[d_family]
    finally:
        runtime.free(d_cx)
        runtime.free(d_cy)
        for d in allocated:
            runtime.free(d)


# ---------------------------------------------------------------------------
# CPU fallback: Centroid (NumPy, delegates to polygon_constructive for polygons)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "geometry_centroid",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("point", "multipoint", "linestring", "multilinestring", "polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "centroid"),
)
def _centroid_cpu(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU centroid for all geometry families using NumPy."""
    # Ensure host buffers are populated -- device_take creates stub host
    # buffers with host_materialized=False that must be hydrated before
    # any host-side computation.
    owned._ensure_host_state()

    row_count = owned.row_count
    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    # --- Point ---
    _centroid_cpu_points(owned, cx, cy, tags, family_row_offsets)

    # --- MultiPoint ---
    _centroid_cpu_multipoints(owned, cx, cy, tags, family_row_offsets)

    # --- LineString ---
    _centroid_cpu_linestrings(owned, cx, cy, tags, family_row_offsets)

    # --- MultiLineString ---
    _centroid_cpu_multilinestrings(owned, cx, cy, tags, family_row_offsets)

    # --- Polygon / MultiPolygon: delegate to existing CPU implementation ---
    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    has_polys = np.any(tags == poly_tag) or np.any(tags == mpoly_tag)
    if has_polys:
        poly_cx, poly_cy = _polygon_centroids_cpu(owned)
        poly_mask = np.isin(tags, [poly_tag, mpoly_tag])
        cx[poly_mask] = poly_cx[poly_mask]
        cy[poly_mask] = poly_cy[poly_mask]

    return cx, cy


def _centroid_cpu_points(owned, cx, cy, tags, family_row_offsets):
    tag = FAMILY_TAGS[GeometryFamily.POINT]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.POINT not in owned.families:
        return
    buf = owned.families[GeometryFamily.POINT]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    geom_offsets = buf.geometry_offsets
    for gi, fr in zip(global_rows, family_rows):
        idx = geom_offsets[fr]
        cx[gi] = buf.x[idx]
        cy[gi] = buf.y[idx]


def _centroid_cpu_multipoints(owned, cx, cy, tags, family_row_offsets):
    tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTIPOINT not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTIPOINT]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    geom_offsets = buf.geometry_offsets
    for gi, fr in zip(global_rows, family_rows):
        cs = geom_offsets[fr]
        ce = geom_offsets[fr + 1]
        if ce <= cs:
            continue  # stays NaN
        cx[gi] = float(np.mean(buf.x[cs:ce]))
        cy[gi] = float(np.mean(buf.y[cs:ce]))


def _centroid_cpu_linestrings(owned, cx, cy, tags, family_row_offsets):
    tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.LINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.LINESTRING]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    geom_offsets = buf.geometry_offsets
    for gi, fr in zip(global_rows, family_rows):
        cs = geom_offsets[fr]
        ce = geom_offsets[fr + 1]
        n = ce - cs
        if n == 0:
            continue
        if n == 1:
            cx[gi] = buf.x[cs]
            cy[gi] = buf.y[cs]
            continue
        lx, ly = _length_weighted_centroid(buf.x, buf.y, cs, ce)
        cx[gi] = lx
        cy[gi] = ly


def _centroid_cpu_multilinestrings(owned, cx, cy, tags, family_row_offsets):
    tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTILINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTILINESTRING]
    if buf.row_count == 0:
        return
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets
    for gi, fr in zip(global_rows, family_rows):
        fp = geom_offsets[fr]
        lp = geom_offsets[fr + 1]
        if fp == lp:
            continue
        total_len = 0.0
        wt_x = 0.0
        wt_y = 0.0
        for p in range(fp, lp):
            cs = part_offsets[p]
            ce = part_offsets[p + 1]
            n = ce - cs
            if n < 2:
                continue
            seg_x = buf.x[cs:ce]
            seg_y = buf.y[cs:ce]
            dx = np.diff(seg_x)
            dy = np.diff(seg_y)
            seg_lens = np.sqrt(dx * dx + dy * dy)
            mid_x = (seg_x[:-1] + seg_x[1:]) * 0.5
            mid_y = (seg_y[:-1] + seg_y[1:]) * 0.5
            total_len += float(np.sum(seg_lens))
            wt_x += float(np.sum(mid_x * seg_lens))
            wt_y += float(np.sum(mid_y * seg_lens))
        if total_len < 1e-30:
            cs = part_offsets[fp]
            cx[gi] = buf.x[cs]
            cy[gi] = buf.y[cs]
        else:
            cx[gi] = wt_x / total_len
            cy[gi] = wt_y / total_len


def _length_weighted_centroid(x, y, cs, ce):
    """Length-weighted centroid of a single coordinate span."""
    seg_x = x[cs:ce]
    seg_y = y[cs:ce]
    dx = np.diff(seg_x)
    dy = np.diff(seg_y)
    seg_lens = np.sqrt(dx * dx + dy * dy)
    total = float(np.sum(seg_lens))
    if total < 1e-30:
        return float(x[cs]), float(y[cs])
    mid_x = (seg_x[:-1] + seg_x[1:]) * 0.5
    mid_y = (seg_y[:-1] + seg_y[1:]) * 0.5
    return float(np.sum(mid_x * seg_lens) / total), float(np.sum(mid_y * seg_lens) / total)


# ---------------------------------------------------------------------------
# Public dispatch API
# ---------------------------------------------------------------------------

def centroid_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> OwnedGeometryArray:
    """Compute centroid directly from OwnedGeometryArray coordinate buffers.

    Supports all geometry families: Point, MultiPoint, LineString,
    MultiLineString, Polygon, MultiPolygon.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns a device-resident Point OwnedGeometryArray (GPU path) or
    host-resident Point OwnedGeometryArray (CPU path).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.geometry.owned import from_shapely_geometries
    from vibespatial.runtime.dispatch import record_dispatch_event
    from vibespatial.runtime.precision import CoordinateStats

    row_count = owned.row_count
    if row_count == 0:
        return from_shapely_geometries([])

    selection = plan_dispatch_selection(
        kernel_name="geometry_centroid",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        max_abs, coord_min, coord_max = _coord_stats_from_owned(owned)
        span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
        # The centroid shoelace formula involves products of coordinates
        # (xi*yi1 - xi1*yi) which require constructive-level precision.
        # fp32 introduces unacceptable absolute errors even with Kahan
        # summation and coordinate centering (observed >1 unit error for
        # coordinate ranges [0, 1000]).  Request CONSTRUCTIVE precision
        # planning to guarantee fp64 compute on consumer GPUs.
        selection = plan_dispatch_selection(
            kernel_name="geometry_centroid",
            kernel_class=KernelClass.METRIC,
            row_count=row_count,
            requested_mode=dispatch_mode,
            requested_precision=precision,
            precision_kernel_class=KernelClass.CONSTRUCTIVE,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        precision_plan = selection.precision_plan
        result = _centroid_gpu(owned, precision_plan=precision_plan)
        record_dispatch_event(
            surface="geopandas.array.centroid",
            operation="centroid",
            implementation="gpu_nvrtc_centroid",
            reason=selection.reason,
            detail=f"rows={row_count}, precision={precision_plan.compute_precision}",
            requested=selection.requested,
            selected=ExecutionMode.GPU,
        )
        return result

    cx, cy = _centroid_cpu(owned)
    cx[~owned.validity] = np.nan
    cy[~owned.validity] = np.nan
    record_dispatch_event(
        surface="geopandas.array.centroid",
        operation="centroid",
        implementation="numpy",
        reason="CPU selected by dispatch planner",
        detail=f"rows={row_count}",
        requested=dispatch_mode,
        selected=ExecutionMode.CPU,
    )
    # Build host-resident Point OwnedGeometryArray from CPU results.
    from vibespatial.constructive.point import point_owned_from_xy

    return point_owned_from_xy(cx, cy)
