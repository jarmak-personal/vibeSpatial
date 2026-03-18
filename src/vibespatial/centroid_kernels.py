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

from vibespatial.adaptive_runtime import plan_dispatch_selection
from vibespatial.cuda_runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.kernel_registry import register_kernel_variant
from vibespatial.measurement_kernels import _PRECISION_PREAMBLE
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.polygon_constructive import (
    _POLYGON_CENTROID_KERNEL_NAMES,
    _POLYGON_CENTROID_KERNEL_SOURCE,
    _polygon_centroids_cpu,
)
from vibespatial.precision import KernelClass
from vibespatial.runtime import ExecutionMode

if TYPE_CHECKING:
    from vibespatial.precision import PrecisionMode, PrecisionPlan

# ---------------------------------------------------------------------------
# Point centroid kernel: identity copy (x, y -> cx, cy)
# ---------------------------------------------------------------------------

_POINT_CENTROID_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void point_centroid(
    const double* x,
    const double* y,
    const int* geometry_offsets,
    double* cx,
    double* cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int idx = geometry_offsets[row];
    cx[row] = x[idx];
    cy[row] = y[idx];
}}
"""

# ---------------------------------------------------------------------------
# MultiPoint centroid kernel: Kahan mean of all points in each geometry
# ---------------------------------------------------------------------------

_MULTIPOINT_CENTROID_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void multipoint_centroid(
    const double* x,
    const double* y,
    const int* geometry_offsets,
    double* cx,
    double* cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int n = ce - cs;

    if (n == 0) {{
        cx[row] = 0.0 / 0.0;  /* NaN */
        cy[row] = 0.0 / 0.0;
        return;
    }}

    compute_t sum_x = (compute_t)0.0;
    compute_t sum_y = (compute_t)0.0;
    compute_t c_x = (compute_t)0.0;
    compute_t c_y = (compute_t)0.0;

    for (int i = cs; i < ce; i++) {{
        KAHAN_ADD(sum_x, CX(x[i]), c_x);
        KAHAN_ADD(sum_y, CY(y[i]), c_y);
    }}

    cx[row] = (double)(sum_x / (compute_t)n) + center_x;
    cy[row] = (double)(sum_y / (compute_t)n) + center_y;
}}
"""

# ---------------------------------------------------------------------------
# LineString centroid kernel: length-weighted segment midpoints
# ---------------------------------------------------------------------------

_LINESTRING_CENTROID_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void linestring_centroid(
    const double* x,
    const double* y,
    const int* geometry_offsets,
    double* cx,
    double* cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int n = ce - cs;

    if (n == 0) {{
        cx[row] = 0.0 / 0.0;  /* NaN */
        cy[row] = 0.0 / 0.0;
        return;
    }}

    if (n == 1) {{
        cx[row] = x[cs];
        cy[row] = y[cs];
        return;
    }}

    compute_t total_len = (compute_t)0.0;
    compute_t wt_x = (compute_t)0.0;
    compute_t wt_y = (compute_t)0.0;
    compute_t c_len = (compute_t)0.0;
    compute_t c_x = (compute_t)0.0;
    compute_t c_y = (compute_t)0.0;

    for (int i = cs; i < ce - 1; i++) {{
        const compute_t x0 = CX(x[i]);
        const compute_t y0 = CY(y[i]);
        const compute_t x1 = CX(x[i + 1]);
        const compute_t y1 = CY(y[i + 1]);
        const compute_t dx = x1 - x0;
        const compute_t dy = y1 - y0;
        const compute_t seg_len = sqrt(dx * dx + dy * dy);

        KAHAN_ADD(total_len, seg_len, c_len);
        const compute_t mid_x = (x0 + x1) * (compute_t)0.5;
        const compute_t mid_y = (y0 + y1) * (compute_t)0.5;
        KAHAN_ADD(wt_x, mid_x * seg_len, c_x);
        KAHAN_ADD(wt_y, mid_y * seg_len, c_y);
    }}

    if (total_len < (compute_t)1e-30) {{
        /* Degenerate: all points coincident, use first point. */
        cx[row] = x[cs];
        cy[row] = y[cs];
    }} else {{
        cx[row] = (double)(wt_x / total_len) + center_x;
        cy[row] = (double)(wt_y / total_len) + center_y;
    }}
}}
"""

# ---------------------------------------------------------------------------
# MultiLineString centroid kernel: length-weighted across all parts
# ---------------------------------------------------------------------------

_MULTILINESTRING_CENTROID_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
extern "C" __global__ void multilinestring_centroid(
    const double* x,
    const double* y,
    const int* part_offsets,
    const int* geometry_offsets,
    double* cx,
    double* cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    if (first_part == last_part) {{
        cx[row] = 0.0 / 0.0;  /* NaN */
        cy[row] = 0.0 / 0.0;
        return;
    }}

    compute_t total_len = (compute_t)0.0;
    compute_t wt_x = (compute_t)0.0;
    compute_t wt_y = (compute_t)0.0;
    compute_t c_len = (compute_t)0.0;
    compute_t c_x = (compute_t)0.0;
    compute_t c_y = (compute_t)0.0;

    for (int part = first_part; part < last_part; part++) {{
        const int cs = part_offsets[part];
        const int ce = part_offsets[part + 1];
        const int n = ce - cs;
        if (n < 2) continue;

        for (int i = cs; i < ce - 1; i++) {{
            const compute_t x0 = CX(x[i]);
            const compute_t y0 = CY(y[i]);
            const compute_t x1 = CX(x[i + 1]);
            const compute_t y1 = CY(y[i + 1]);
            const compute_t dx = x1 - x0;
            const compute_t dy = y1 - y0;
            const compute_t seg_len = sqrt(dx * dx + dy * dy);

            KAHAN_ADD(total_len, seg_len, c_len);
            const compute_t mid_x = (x0 + x1) * (compute_t)0.5;
            const compute_t mid_y = (y0 + y1) * (compute_t)0.5;
            KAHAN_ADD(wt_x, mid_x * seg_len, c_x);
            KAHAN_ADD(wt_y, mid_y * seg_len, c_y);
        }}
    }}

    if (total_len < (compute_t)1e-30) {{
        /* Degenerate: use first coordinate. */
        const int cs = part_offsets[first_part];
        cx[row] = x[cs];
        cy[row] = y[cs];
    }} else {{
        cx[row] = (double)(wt_x / total_len) + center_x;
        cy[row] = (double)(wt_y / total_len) + center_y;
    }}
}}
"""

# ---------------------------------------------------------------------------
# Kernel names and precompiled source variants
# ---------------------------------------------------------------------------

_POINT_CENTROID_NAMES = ("point_centroid",)
_MULTIPOINT_CENTROID_NAMES = ("multipoint_centroid",)
_LINESTRING_CENTROID_NAMES = ("linestring_centroid",)
_MULTILINESTRING_CENTROID_NAMES = ("multilinestring_centroid",)

_POINT_CENTROID_FP64 = _POINT_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_POINT_CENTROID_FP32 = _POINT_CENTROID_KERNEL_SOURCE.format(compute_type="float")
_MULTIPOINT_CENTROID_FP64 = _MULTIPOINT_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_MULTIPOINT_CENTROID_FP32 = _MULTIPOINT_CENTROID_KERNEL_SOURCE.format(compute_type="float")
_LINESTRING_CENTROID_FP64 = _LINESTRING_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_LINESTRING_CENTROID_FP32 = _LINESTRING_CENTROID_KERNEL_SOURCE.format(compute_type="float")
_MULTILINESTRING_CENTROID_FP64 = _MULTILINESTRING_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_MULTILINESTRING_CENTROID_FP32 = _MULTILINESTRING_CENTROID_KERNEL_SOURCE.format(compute_type="float")
_POLYGON_CENTROID_FP64 = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_CENTROID_FP32 = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="float")

# Background precompilation (ADR-0034)
from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("centroid-point-fp64", _POINT_CENTROID_FP64, _POINT_CENTROID_NAMES),
    ("centroid-point-fp32", _POINT_CENTROID_FP32, _POINT_CENTROID_NAMES),
    ("centroid-multipoint-fp64", _MULTIPOINT_CENTROID_FP64, _MULTIPOINT_CENTROID_NAMES),
    ("centroid-multipoint-fp32", _MULTIPOINT_CENTROID_FP32, _MULTIPOINT_CENTROID_NAMES),
    ("centroid-linestring-fp64", _LINESTRING_CENTROID_FP64, _LINESTRING_CENTROID_NAMES),
    ("centroid-linestring-fp32", _LINESTRING_CENTROID_FP32, _LINESTRING_CENTROID_NAMES),
    ("centroid-multilinestring-fp64", _MULTILINESTRING_CENTROID_FP64, _MULTILINESTRING_CENTROID_NAMES),
    ("centroid-multilinestring-fp32", _MULTILINESTRING_CENTROID_FP32, _MULTILINESTRING_CENTROID_NAMES),
])


# ---------------------------------------------------------------------------
# Kernel compilation helpers
# ---------------------------------------------------------------------------

def _compile_kernel(name_prefix: str, fp64_source: str, fp32_source: str,
                    kernel_names: tuple[str, ...], compute_type: str = "double"):
    source = fp64_source if compute_type == "double" else fp32_source
    suffix = "fp64" if compute_type == "double" else "fp32"
    return compile_kernel_group(f"{name_prefix}-{suffix}", source, kernel_names)


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
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated centroid for all geometry families.

    Returns (cx, cy) tuple of float64 arrays of shape (row_count,).
    Respects ADR-0002 precision dispatch.
    """
    from vibespatial.precision import PrecisionMode

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
    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # --- Point family ---
    _launch_point_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
    )

    # --- MultiPoint family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
        GeometryFamily.MULTIPOINT, "multipoint_centroid",
        _MULTIPOINT_CENTROID_FP64, _MULTIPOINT_CENTROID_FP32,
        _MULTIPOINT_CENTROID_NAMES, "centroid-multipoint",
        has_part_offsets=False,
    )

    # --- LineString family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
        GeometryFamily.LINESTRING, "linestring_centroid",
        _LINESTRING_CENTROID_FP64, _LINESTRING_CENTROID_FP32,
        _LINESTRING_CENTROID_NAMES, "centroid-linestring",
        has_part_offsets=False,
    )

    # --- MultiLineString family ---
    _launch_simple_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
        GeometryFamily.MULTILINESTRING, "multilinestring_centroid",
        _MULTILINESTRING_CENTROID_FP64, _MULTILINESTRING_CENTROID_FP32,
        _MULTILINESTRING_CENTROID_NAMES, "centroid-multilinestring",
        has_part_offsets=True,
    )

    # --- Polygon family ---
    _launch_polygon_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
        GeometryFamily.POLYGON,
    )

    # --- MultiPolygon family ---
    _launch_polygon_centroid(
        owned, runtime, tags, family_row_offsets, device_state,
        cx, cy, center_x, center_y, compute_type,
        GeometryFamily.MULTIPOLYGON,
    )

    return cx, cy


def _launch_point_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    cx, cy, center_x, center_y, compute_type,
):
    """Launch point centroid kernel (identity copy)."""
    tag = FAMILY_TAGS[GeometryFamily.POINT]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.POINT not in owned.families:
        return
    buf = owned.families[GeometryFamily.POINT]
    if buf.row_count == 0 or len(buf.geometry_offsets) < 2:
        return

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
        family_cx = runtime.copy_device_to_host(d_cx)
        family_cy = runtime.copy_device_to_host(d_cy)
        cx[global_rows] = family_cx[family_rows]
        cy[global_rows] = family_cy[family_rows]
    finally:
        runtime.free(d_cx)
        runtime.free(d_cy)
        for d in allocated:
            runtime.free(d)


def _launch_simple_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    cx, cy, center_x, center_y, compute_type,
    family, kernel_name, fp64_source, fp32_source, kernel_names, prefix,
    has_part_offsets,
):
    """Launch centroid kernel for MultiPoint, LineString, or MultiLineString."""
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0 or len(buf.geometry_offsets) < 2:
        return
    if has_part_offsets and buf.part_offsets is None:
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
        family_cx = runtime.copy_device_to_host(d_cx)
        family_cy = runtime.copy_device_to_host(d_cy)
        cx[global_rows] = family_cx[family_rows]
        cy[global_rows] = family_cy[family_rows]
    finally:
        runtime.free(d_cx)
        runtime.free(d_cy)
        for d in allocated:
            runtime.free(d)


def _launch_polygon_centroid(
    owned, runtime, tags, family_row_offsets, device_state,
    cx, cy, center_x, center_y, compute_type,
    family,
):
    """Launch polygon centroid kernel (Polygon or MultiPolygon)."""
    tag = FAMILY_TAGS[family]
    mask = tags == tag
    if not np.any(mask) or family not in owned.families:
        return
    buf = owned.families[family]
    if buf.row_count == 0 or buf.ring_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = _compile_polygon_centroid_kernel(compute_type)
    kernel = kernels["polygon_centroid"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

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
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_cx = runtime.copy_device_to_host(d_cx)
        family_cy = runtime.copy_device_to_host(d_cy)
        cx[global_rows] = family_cx[family_rows]
        cy[global_rows] = family_cy[family_rows]
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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid directly from OwnedGeometryArray coordinate buffers.

    Supports all geometry families: Point, MultiPoint, LineString,
    MultiLineString, Polygon, MultiPolygon.

    GPU path uses ADR-0002 METRIC-class precision dispatch.
    Returns (cx, cy) tuple of float64 arrays of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.precision import CoordinateStats, select_precision_plan
    from vibespatial.runtime import RuntimeSelection

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="geometry_centroid",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        max_abs = 0.0
        coord_min = np.inf
        coord_max = -np.inf
        for buf in owned.families.values():
            if buf.row_count > 0 and buf.x.size > 0:
                max_abs = max(max_abs, float(np.abs(buf.x).max()), float(np.abs(buf.y).max()))
                coord_min = min(coord_min, float(buf.x.min()), float(buf.y.min()))
                coord_max = max(coord_max, float(buf.x.max()), float(buf.y.max()))
        span = coord_max - coord_min if np.isfinite(coord_min) else 0.0
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="geometry_centroid GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        try:
            cx, cy = _centroid_gpu(owned, precision_plan=precision_plan)
            cx[~owned.validity] = np.nan
            cy[~owned.validity] = np.nan
            return cx, cy
        except Exception:
            pass  # fall through to CPU

    cx, cy = _centroid_cpu(owned)
    cx[~owned.validity] = np.nan
    cy[~owned.validity] = np.nan
    return cx, cy
