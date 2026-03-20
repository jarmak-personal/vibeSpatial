from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total_with_transfer,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import exclusive_sum
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan


_POLYGON_BUFFER_KERNEL_SOURCE = r"""
#define PI 3.14159265358979323846
#define EPSILON 1e-12

#define JOIN_ROUND 0
#define JOIN_MITRE 1
#define JOIN_BEVEL 2

/* ------------------------------------------------------------------ */
/*  Device helpers                                                     */
/* ------------------------------------------------------------------ */

__device__ void normalize_2d(double dx, double dy,
                             double* nx, double* ny, double* len) {
    double l = sqrt(dx * dx + dy * dy);
    *len = l;
    if (l < EPSILON) { *nx = 0.0; *ny = 0.0; return; }
    *nx = dx / l;
    *ny = dy / l;
}

__device__ double cross_2d(double ax, double ay, double bx, double by) {
    return ax * by - ay * bx;
}

__device__ int arc_vertex_count(double sweep_angle, int quad_segs) {
    double step = PI / (2.0 * quad_segs);
    int n = (int)ceil(fabs(sweep_angle) / step);
    return n < 1 ? 1 : n;
}

__device__ void emit_arc_verts(double cx, double cy, double radius,
                               double start_angle, double sweep,
                               int n_steps,
                               double* out_x, double* out_y, int* pos) {
    for (int i = 1; i <= n_steps; i++) {
        double angle = start_angle + sweep * ((double)i / (double)n_steps);
        out_x[*pos] = cx + radius * cos(angle);
        out_y[*pos] = cy + radius * sin(angle);
        (*pos)++;
    }
}

/* winding_sign: +1.0 if ring is CCW, -1.0 if ring is CW.
   d_eff: effective distance accounting for winding and hole status.
   d_eff = d * winding_sign * (is_hole ? -1 : 1)
   sf (sign_factor) = winding_sign — used for convex/concave classification. */

__device__ int convex_join_count_poly(
    double prev_dx, double prev_dy, double curr_dx, double curr_dy,
    int join_style, int quad_segs, double mitre_limit,
    double d_eff, double sf
) {
    if (join_style == JOIN_ROUND) {
        double prev_nx = -prev_dy, prev_ny = prev_dx;
        double curr_nx = -curr_dy, curr_ny = curr_dx;
        double effective_d = sf * d_eff;
        double d_sign = (effective_d >= 0.0) ? 1.0 : -1.0;
        double start_a = atan2(-d_sign * prev_ny, -d_sign * prev_nx);
        double end_a   = atan2(-d_sign * curr_ny, -d_sign * curr_nx);
        double sweep = end_a - start_a;
        if (effective_d >= 0.0) { if (sweep < 0.0) sweep += 2.0 * PI; }  /* CCW */
        else                    { if (sweep > 0.0) sweep -= 2.0 * PI; }  /* CW  */
        return 1 + arc_vertex_count(sweep, quad_segs);  /* arrival + arc */
    }
    if (join_style == JOIN_MITRE) {
        double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
        if ((1.0 - dot_d) * mitre_limit * mitre_limit < 2.0) {
            return 2;  /* bevel fallback */
        }
        return 1;  /* mitre intersection */
    }
    /* JOIN_BEVEL */
    return 2;  /* arrival + departure */
}

/* ------------------------------------------------------------------ */
/*  Count kernel: one thread per ring                                  */
/*                                                                     */
/*  ring_winding: +1.0 if CCW, -1.0 if CW (from signed area).         */
/*  ring_is_hole: 1 if hole ring, 0 if exterior.                       */
/*  d_eff = d * winding * (is_hole ? -1 : 1) — effective distance.     */
/*  sf = winding — sign_factor for classification.                     */
/* ------------------------------------------------------------------ */

extern "C" __global__ void polygon_buffer_ring_count(
    const int* ring_to_row,
    const int* ring_is_hole,
    const double* ring_winding,
    const int* poly_ring_offsets,
    const double* poly_x,
    const double* poly_y,
    const double* distances,
    int quad_segs,
    int join_style,
    double mitre_limit,
    int* out_ring_counts,
    int total_rings
) {
    const int ring_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring_idx >= total_rings) return;

    const int coord_start = poly_ring_offsets[ring_idx];
    const int coord_end   = poly_ring_offsets[ring_idx + 1];
    const int N = coord_end - coord_start - 1;

    if (N < 3) {
        out_ring_counts[ring_idx] = 0;
        return;
    }

    const int row = ring_to_row[ring_idx];
    const double d = distances[row];
    const double w = ring_winding[ring_idx];
    const double hole_sign = (ring_is_hole[ring_idx]) ? -1.0 : 1.0;
    const double d_eff = d * w * hole_sign;
    const double sf = w;
    const int q = quad_segs;
    double seg_len;
    double prev_dx, prev_dy, curr_dx, curr_dy;

    int total = 0;

    normalize_2d(poly_x[coord_start] - poly_x[coord_start + N - 1],
                 poly_y[coord_start] - poly_y[coord_start + N - 1],
                 &prev_dx, &prev_dy, &seg_len);

    for (int i = 0; i < N; i++) {
        int next = (i + 1 < N) ? i + 1 : 0;
        normalize_2d(poly_x[coord_start + next] - poly_x[coord_start + i],
                     poly_y[coord_start + next] - poly_y[coord_start + i],
                     &curr_dx, &curr_dy, &seg_len);

        double cross = cross_2d(prev_dx, prev_dy, curr_dx, curr_dy);
        double ec = sf * cross;
        int is_convex = (d_eff >= 0.0 && ec > EPSILON) || (d_eff < 0.0 && ec < -EPSILON);
        if (is_convex) {
            total += convex_join_count_poly(
                prev_dx, prev_dy, curr_dx, curr_dy,
                join_style, q, mitre_limit, d_eff, sf);
        } else {
            total += 1;
        }

        prev_dx = curr_dx;
        prev_dy = curr_dy;
    }

    total += 1;  /* closing vertex */
    out_ring_counts[ring_idx] = total;
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel: one thread per ring                                */
/* ------------------------------------------------------------------ */

__device__ void emit_convex_join_poly(
    double vx, double vy, double d_eff, double sf,
    double prev_dx, double prev_dy, double curr_dx, double curr_dy,
    double cross,
    int join_style, int quad_segs, double mitre_limit,
    double* out_x, double* out_y, int* pos
) {
    double prev_nx = -prev_dy, prev_ny = prev_dx;
    double curr_nx = -curr_dy, curr_ny = curr_dx;

    if (join_style == JOIN_ROUND) {
        out_x[*pos] = vx - d_eff * prev_nx;
        out_y[*pos] = vy - d_eff * prev_ny;
        (*pos)++;
        double abs_d = fabs(d_eff);
        double effective_d = sf * d_eff;
        double d_sign = (effective_d >= 0.0) ? 1.0 : -1.0;
        double start_a = atan2(-d_sign * prev_ny, -d_sign * prev_nx);
        double end_a   = atan2(-d_sign * curr_ny, -d_sign * curr_nx);
        double sweep = end_a - start_a;
        if (effective_d >= 0.0) { if (sweep < 0.0) sweep += 2.0 * PI; }
        else                    { if (sweep > 0.0) sweep -= 2.0 * PI; }
        int n_steps = arc_vertex_count(sweep, quad_segs);
        emit_arc_verts(vx, vy, abs_d, start_a, sweep, n_steps,
                       out_x, out_y, pos);
        return;
    }
    if (join_style == JOIN_MITRE) {
        double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
        if ((1.0 - dot_d) * mitre_limit * mitre_limit < 2.0) {
            out_x[*pos] = vx - d_eff * prev_nx;
            out_y[*pos] = vy - d_eff * prev_ny;
            (*pos)++;
            out_x[*pos] = vx - d_eff * curr_nx;
            out_y[*pos] = vy - d_eff * curr_ny;
            (*pos)++;
            return;
        }
        double t = d_eff * (1.0 - dot_d) / cross;
        out_x[*pos] = vx - d_eff * prev_nx + t * prev_dx;
        out_y[*pos] = vy - d_eff * prev_ny + t * prev_dy;
        (*pos)++;
        return;
    }
    /* JOIN_BEVEL */
    out_x[*pos] = vx - d_eff * prev_nx;
    out_y[*pos] = vy - d_eff * prev_ny;
    (*pos)++;
    out_x[*pos] = vx - d_eff * curr_nx;
    out_y[*pos] = vy - d_eff * curr_ny;
    (*pos)++;
}

extern "C" __global__ void polygon_buffer_ring_scatter(
    const int* ring_to_row,
    const int* ring_is_hole,
    const double* ring_winding,
    const int* poly_ring_offsets,
    const double* poly_x,
    const double* poly_y,
    const double* distances,
    int quad_segs,
    int join_style,
    double mitre_limit,
    const int* output_ring_offsets,
    double* out_x,
    double* out_y,
    int total_rings
) {
    const int ring_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring_idx >= total_rings) return;

    const int coord_start = poly_ring_offsets[ring_idx];
    const int coord_end   = poly_ring_offsets[ring_idx + 1];
    const int N = coord_end - coord_start - 1;

    if (N < 3) return;

    const int row = ring_to_row[ring_idx];
    const double d = distances[row];
    const double w = ring_winding[ring_idx];
    const double hole_sign = (ring_is_hole[ring_idx]) ? -1.0 : 1.0;
    const double d_eff = d * w * hole_sign;
    const double sf = w;
    const int q = quad_segs;

    int pos = output_ring_offsets[ring_idx];

    double seg_len;
    double prev_dx, prev_dy, curr_dx, curr_dy;

    normalize_2d(poly_x[coord_start] - poly_x[coord_start + N - 1],
                 poly_y[coord_start] - poly_y[coord_start + N - 1],
                 &prev_dx, &prev_dy, &seg_len);

    for (int i = 0; i < N; i++) {
        int next = (i + 1 < N) ? i + 1 : 0;
        normalize_2d(poly_x[coord_start + next] - poly_x[coord_start + i],
                     poly_y[coord_start + next] - poly_y[coord_start + i],
                     &curr_dx, &curr_dy, &seg_len);

        double vx = poly_x[coord_start + i];
        double vy = poly_y[coord_start + i];
        double cross = cross_2d(prev_dx, prev_dy, curr_dx, curr_dy);
        double ec = sf * cross;
        int is_convex = (d_eff >= 0.0 && ec > EPSILON) || (d_eff < 0.0 && ec < -EPSILON);

        if (is_convex) {
            emit_convex_join_poly(vx, vy, d_eff, sf,
                prev_dx, prev_dy, curr_dx, curr_dy,
                cross, join_style, q, mitre_limit,
                out_x, out_y, &pos);
        } else if (fabs(cross) > EPSILON) {
            double prev_nx = -prev_dy, prev_ny = prev_dx;
            double dot_d = prev_dx * curr_dx + prev_dy * curr_dy;
            double t = d_eff * (1.0 - dot_d) / cross;
            out_x[pos] = vx - d_eff * prev_nx + t * prev_dx;
            out_y[pos] = vy - d_eff * prev_ny + t * prev_dy;
            pos++;
        } else {
            double prev_nx = -prev_dy, prev_ny = prev_dx;
            out_x[pos] = vx - d_eff * prev_nx;
            out_y[pos] = vy - d_eff * prev_ny;
            pos++;
        }

        prev_dx = curr_dx;
        prev_dy = curr_dy;
    }

    /* Closing vertex = copy of first emitted vertex */
    out_x[pos] = out_x[output_ring_offsets[ring_idx]];
    out_y[pos] = out_y[output_ring_offsets[ring_idx]];
    pos++;
}
"""

POLYGON_BUFFER_GPU_THRESHOLD = 50_000

_POLYGON_BUFFER_KERNEL_NAMES = ("polygon_buffer_ring_count", "polygon_buffer_ring_scatter")

_RING_WINDING_KERNEL_SOURCE = r"""
extern "C" __global__ void compute_ring_winding(
    const double* x,
    const double* y,
    const int* ring_offsets,
    double* ring_winding,
    int total_rings
) {
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= total_rings) return;

    const int coord_start = ring_offsets[ring];
    const int coord_end = ring_offsets[ring + 1];
    const int n = coord_end - coord_start;
    if (n < 3) {
        ring_winding[ring] = 1.0;
        return;
    }

    /* Shoelace formula: sum of (x[i]*y[i+1] - x[i+1]*y[i]).
       Exclude closing vertex (last == first). */
    double area = 0.0;
    const int last = coord_end - 1;
    for (int i = coord_start; i < last; ++i) {
        int next = i + 1;
        if (next >= last) next = coord_start;
        area += x[i] * y[next] - x[next] * y[i];
    }
    ring_winding[ring] = area > 0.0 ? 1.0 : -1.0;
}
"""

_RING_WINDING_KERNEL_NAMES = ("compute_ring_winding",)

# ---------------------------------------------------------------------------
# GPU Polygon Centroid Kernel (NVRTC Tier 1)
# ---------------------------------------------------------------------------
# One thread per polygon.  Each thread traverses the exterior ring and
# computes the centroid via the shoelace formula:
#   cross_i = x_i * y_{i+1} - x_{i+1} * y_i
#   signed_area = sum(cross_i) / 2
#   cx = sum((x_i + x_{i+1}) * cross_i) / (6 * signed_area)
#   cy = sum((y_i + y_{i+1}) * cross_i) / (6 * signed_area)

_POLYGON_CENTROID_KERNEL_SOURCE = r"""
typedef {compute_type} compute_t;

/* Centered coordinate read: subtract center in fp64, then cast to compute_t.
   When compute_t is double, this is a no-op identity.  When compute_t is float,
   centering reduces absolute magnitude before the lossy cast. */
#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

/* Kahan summation helper — add `val` to `sum` with compensation `c`.
   When compute_t is double, the compiler can optimize this away if it
   proves the compensation term is always zero (it won't, but the cost
   is negligible at fp64 throughput). */
#define KAHAN_ADD(sum, val, c) do {{ \
    const compute_t _y = (val) - (c); \
    const compute_t _t = (sum) + _y; \
    (c) = (_t - (sum)) - _y; \
    (sum) = _t; \
}} while(0)

extern "C" __global__ void polygon_centroid(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* geometry_offsets,
    double* cx,
    double* cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    /* Exterior ring is the first ring of the polygon. */
    const int first_ring = geometry_offsets[row];
    const int coord_start = ring_offsets[first_ring];
    const int coord_end = ring_offsets[first_ring + 1];
    int n = coord_end - coord_start;

    /* Strip closure vertex if present. */
    if (n >= 2) {{
        double dx = x[coord_start] - x[coord_end - 1];
        double dy = y[coord_start] - y[coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) n--;
    }}

    if (n < 3) {{
        cx[row] = 0.0 / 0.0;  /* NaN */
        cy[row] = 0.0 / 0.0;
        return;
    }}

    compute_t sum_area = (compute_t)0.0;
    compute_t sum_cx = (compute_t)0.0;
    compute_t sum_cy = (compute_t)0.0;
    /* Kahan compensation terms */
    compute_t c_area = (compute_t)0.0;
    compute_t c_cx = (compute_t)0.0;
    compute_t c_cy = (compute_t)0.0;

    for (int i = 0; i < n; i++) {{
        const int cur = coord_start + i;
        const int nxt = coord_start + ((i + 1) % n);
        const compute_t xi  = CX(x[cur]);
        const compute_t yi  = CY(y[cur]);
        const compute_t xi1 = CX(x[nxt]);
        const compute_t yi1 = CY(y[nxt]);

        const compute_t cross = xi * yi1 - xi1 * yi;
        KAHAN_ADD(sum_area, cross, c_area);
        KAHAN_ADD(sum_cx, (xi + xi1) * cross, c_cx);
        KAHAN_ADD(sum_cy, (yi + yi1) * cross, c_cy);
    }}

    const compute_t signed_area = sum_area * (compute_t)0.5;

    if (fabs(signed_area) < (compute_t)1e-30) {{
        /* Degenerate polygon -- fall back to coordinate mean. */
        compute_t mx = (compute_t)0.0, my = (compute_t)0.0;
        for (int i = 0; i < n; i++) {{
            mx += CX(x[coord_start + i]);
            my += CY(y[coord_start + i]);
        }}
        /* Un-center: add back center to get absolute coordinates. */
        cx[row] = (double)(mx / (compute_t)n) + center_x;
        cy[row] = (double)(my / (compute_t)n) + center_y;
    }} else {{
        /* Un-center: centroid of centered coords + center offset. */
        cx[row] = (double)(sum_cx / ((compute_t)6.0 * signed_area)) + center_x;
        cy[row] = (double)(sum_cy / ((compute_t)6.0 * signed_area)) + center_y;
    }}
}}
"""

_POLYGON_CENTROID_KERNEL_NAMES = ("polygon_centroid",)

_POLYGON_CENTROID_FP64_SOURCE = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_CENTROID_FP32_SOURCE = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="float")

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("polygon-buffer", _POLYGON_BUFFER_KERNEL_SOURCE, _POLYGON_BUFFER_KERNEL_NAMES),
    ("ring-winding", _RING_WINDING_KERNEL_SOURCE, _RING_WINDING_KERNEL_NAMES),
    ("polygon-centroid-fp64", _POLYGON_CENTROID_FP64_SOURCE, _POLYGON_CENTROID_KERNEL_NAMES),
    ("polygon-centroid-fp32", _POLYGON_CENTROID_FP32_SOURCE, _POLYGON_CENTROID_KERNEL_NAMES),
])

from vibespatial.cuda.cccl_precompile import request_warmup  # noqa: E402

request_warmup(["exclusive_scan_i32"])

_CAP_STYLE_MAP = {"round": 0, "flat": 1, "square": 2}
_JOIN_STYLE_MAP = {"round": 0, "mitre": 1, "bevel": 2}


def _ring_winding_kernels():
    return compile_kernel_group("ring-winding", _RING_WINDING_KERNEL_SOURCE, _RING_WINDING_KERNEL_NAMES)


def _polygon_centroid_kernels(compute_type: str = "double"):
    source = _POLYGON_CENTROID_FP64_SOURCE if compute_type == "double" else _POLYGON_CENTROID_FP32_SOURCE
    prefix = f"polygon-centroid-{compute_type[:2]}"  # fp64 / fl (float)
    return compile_kernel_group(prefix, source, _POLYGON_CENTROID_KERNEL_NAMES)


@register_kernel_variant(
    "polygon_centroid",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "centroid", "kahan", "centered"),
)
def _polygon_centroids_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GPU-accelerated polygon centroid computation via NVRTC shoelace kernel.

    Returns (cx, cy) arrays of shape (row_count,) or None if GPU is
    unavailable or no polygon families are present.  Respects ADR-0002
    precision dispatch: fp32 with Kahan summation + coordinate centering
    on consumer GPUs, native fp64 on datacenter GPUs.
    """
    from vibespatial.runtime.precision import PrecisionMode
    compute_type = "double"
    center_x, center_y = 0.0, 0.0
    if precision_plan is not None and precision_plan.compute_precision is PrecisionMode.FP32:
        compute_type = "float"
        if precision_plan.center_coordinates:
            # Compute coordinate centroid for centering
            for buf in owned.families.values():
                if buf.row_count > 0 and buf.x.size > 0:
                    center_x = float((buf.x.min() + buf.x.max()) * 0.5)
                    center_y = float((buf.y.min() + buf.y.max()) * 0.5)
                    break

    runtime = get_cuda_runtime()
    kernels = _polygon_centroid_kernels(compute_type)
    kernel = kernels["polygon_centroid"]

    row_count = owned.row_count
    cx = np.full(row_count, np.nan, dtype=np.float64)
    cy = np.full(row_count, np.nan, dtype=np.float64)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for tag, family_key in ((poly_tag, GeometryFamily.POLYGON), (mpoly_tag, GeometryFamily.MULTIPOLYGON)):
        row_mask = tags == tag
        if not np.any(row_mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0 or buf.geometry_offsets is None or len(buf.geometry_offsets) < 2:
            continue

        family_rows_count = buf.row_count
        global_rows = np.flatnonzero(row_mask)
        family_rows = family_row_offsets[global_rows]

        # Build a compact mapping: for each family row that appears in this
        # batch, we launch the kernel once with family_rows_count items.
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring_offsets = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_geom_offsets = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        d_cx = runtime.allocate((family_rows_count,), np.float64)
        d_cy = runtime.allocate((family_rows_count,), np.float64)

        try:
            ptr = runtime.pointer
            params = (
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_geom_offsets),
                 ptr(d_cx), ptr(d_cy), center_x, center_y, family_rows_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32),
            )
            grid, block = runtime.launch_config(kernel, family_rows_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)

            family_cx = runtime.copy_device_to_host(d_cx)
            family_cy = runtime.copy_device_to_host(d_cy)

            # Scatter family results back to global row positions
            cx[global_rows] = family_cx[family_rows]
            cy[global_rows] = family_cy[family_rows]
        finally:
            runtime.free(d_x)
            runtime.free(d_y)
            runtime.free(d_ring_offsets)
            runtime.free(d_geom_offsets)
            runtime.free(d_cx)
            runtime.free(d_cy)

    return cx, cy


def _polygon_buffer_kernels():
    return compile_kernel_group("polygon-buffer", _POLYGON_BUFFER_KERNEL_SOURCE, _POLYGON_BUFFER_KERNEL_NAMES)


def _build_device_backed_polygon_output_variable(
    device_x,
    device_y,
    *,
    row_count: int,
    geometry_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    bounds: np.ndarray | None = None,
) -> OwnedGeometryArray:
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
        bounds=bounds,
        host_materialized=False,
    )
    runtime = get_cuda_runtime()
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POLYGON: polygon_buffer},
        residency=Residency.DEVICE,
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
                    bounds=None if bounds is None else runtime.from_host(bounds),
                )
            },
        ),
    )


def polygon_buffer_owned_array(
    polygons: OwnedGeometryArray,
    distance: float | np.ndarray,
    *,
    quad_segs: int = 8,
    join_style: str = "round",
    mitre_limit: float = 5.0,
    dispatch_mode: ExecutionMode = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    if GeometryFamily.POLYGON not in polygons.families or len(polygons.families) != 1:
        raise ValueError("polygon_buffer_owned_array requires a polygon-only OwnedGeometryArray")
    if not np.all(polygons.validity):
        raise ValueError("polygon_buffer_owned_array requires non-null rows only")
    if np.any(polygons.tags != FAMILY_TAGS[GeometryFamily.POLYGON]):
        raise ValueError("polygon_buffer_owned_array requires polygon-only rows")

    poly_buffer = polygons.families[GeometryFamily.POLYGON]
    if np.any(poly_buffer.empty_mask):
        raise ValueError("polygon_buffer_owned_array requires non-empty rows only")

    radii = (
        np.full(polygons.row_count, float(distance), dtype=np.float64)
        if np.isscalar(distance)
        else np.asarray(distance, dtype=np.float64)
    )
    if radii.shape != (polygons.row_count,):
        raise ValueError("distance must be a scalar or length-matched vector")

    join_int = _JOIN_STYLE_MAP.get(join_style, 0)

    selected_mode = plan_dispatch_selection(
        kernel_name="polygon_buffer",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=polygons.row_count,
        requested_mode=dispatch_mode,
    ).selected

    if selected_mode is not ExecutionMode.GPU:
        return _build_polygon_buffers_cpu(
            polygons, radii, quad_segs=quad_segs,
            join_style=join_style, mitre_limit=mitre_limit,
        )

    return _build_polygon_buffers_gpu(
        polygons, radii, quad_segs=quad_segs,
        join_style=join_int, mitre_limit=mitre_limit,
    )


def _build_polygon_buffers_cpu(
    polygons: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    join_style: str = "round",
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    import shapely
    polygons._ensure_host_state()
    shapely_geoms = polygons.to_shapely()
    result_geoms = shapely.buffer(
        np.asarray(shapely_geoms, dtype=object),
        radii,
        quad_segs=quad_segs,
        join_style=join_style,
        mitre_limit=mitre_limit,
    )
    from vibespatial.geometry.owned import from_shapely_geometries
    return from_shapely_geometries(list(result_geoms))


def _build_polygon_buffers_gpu(
    polygons: OwnedGeometryArray,
    radii: np.ndarray,
    *,
    quad_segs: int,
    join_style: int = 0,
    mitre_limit: float = 5.0,
) -> OwnedGeometryArray:
    polygons.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygon_buffer_owned_array selected GPU execution",
    )
    runtime = get_cuda_runtime()
    state = polygons._ensure_device_state()
    poly_buf = state.families[GeometryFamily.POLYGON]

    # Build ring-level mapping arrays on host
    host_poly_buf = polygons.families[GeometryFamily.POLYGON]
    input_geo_offsets = host_poly_buf.geometry_offsets
    ring_counts_per_poly = np.diff(input_geo_offsets).astype(np.int32)
    total_rings = int(input_geo_offsets[-1])

    if total_rings == 0:
        device_x = runtime.allocate((0,), np.float64)
        device_y = runtime.allocate((0,), np.float64)
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=polygons.row_count,
            geometry_offsets=input_geo_offsets.copy(),
            ring_offsets=np.zeros(1, dtype=np.int32),
        )

    ring_to_row = np.repeat(
        np.arange(polygons.row_count, dtype=np.int32), ring_counts_per_poly
    )
    ring_is_hole = np.ones(total_rings, dtype=np.int32)
    ring_is_hole[input_geo_offsets[:-1]] = 0

    # Compute actual winding direction per ring via signed area (shoelace)
    # on GPU — one thread per ring, no Python loop.
    winding_kernels = _ring_winding_kernels()
    device_ring_winding = runtime.allocate((total_rings,), np.float64)
    winding_grid, winding_block = runtime.launch_config(
        winding_kernels["compute_ring_winding"], total_rings,
    )
    winding_params = (
        (
            runtime.pointer(poly_buf.x),
            runtime.pointer(poly_buf.y),
            runtime.pointer(poly_buf.ring_offsets),
            runtime.pointer(device_ring_winding),
            total_rings,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    runtime.launch(
        winding_kernels["compute_ring_winding"],
        grid=winding_grid, block=winding_block, params=winding_params,
    )

    device_radii = runtime.from_host(radii)
    device_ring_to_row = runtime.from_host(ring_to_row)
    device_ring_is_hole = runtime.from_host(ring_is_hole)
    device_ring_counts = runtime.allocate((total_rings,), np.int32)
    device_ring_offsets = None
    device_x = None
    device_y = None
    success = False

    try:
        kernels = _polygon_buffer_kernels()
        ptr = runtime.pointer

        # Pass 1: count output vertices per ring
        count_params = (
            (
                ptr(device_ring_to_row),
                ptr(device_ring_is_hole),
                ptr(device_ring_winding),
                ptr(poly_buf.ring_offsets),
                ptr(poly_buf.x),
                ptr(poly_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                mitre_limit,
                ptr(device_ring_counts),
                total_rings,
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
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(kernels["polygon_buffer_ring_count"], total_rings)
        runtime.launch(kernels["polygon_buffer_ring_count"],
                       grid=count_grid, block=count_block, params=count_params)

        # Compute exclusive prefix sum for scatter offsets
        device_ring_offsets = exclusive_sum(device_ring_counts)

        # Get total and start async full-counts D2H transfer.  The
        # transfer runs on a dedicated stream, overlapping with the
        # scatter kernel on the null stream.
        total_verts, xfer_stream, pinned_counts = count_scatter_total_with_transfer(
            runtime, device_ring_counts, device_ring_offsets,
        )

        if total_verts == 0:
            xfer_stream.synchronize()
            runtime.destroy_stream(xfer_stream)
            device_x = runtime.allocate((0,), np.float64)
            device_y = runtime.allocate((0,), np.float64)
            out_ring_offsets = np.zeros(total_rings + 1, dtype=np.int32)
            success = True
            return _build_device_backed_polygon_output_variable(
                device_x, device_y,
                row_count=polygons.row_count,
                geometry_offsets=input_geo_offsets.copy(),
                ring_offsets=out_ring_offsets,
            )

        # Allocate output coordinate arrays
        device_x = runtime.allocate((total_verts,), np.float64)
        device_y = runtime.allocate((total_verts,), np.float64)

        # Pass 2: scatter vertices
        scatter_params = (
            (
                ptr(device_ring_to_row),
                ptr(device_ring_is_hole),
                ptr(device_ring_winding),
                ptr(poly_buf.ring_offsets),
                ptr(poly_buf.x),
                ptr(poly_buf.y),
                ptr(device_radii),
                quad_segs,
                join_style,
                mitre_limit,
                ptr(device_ring_offsets),
                ptr(device_x),
                ptr(device_y),
                total_rings,
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
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["polygon_buffer_ring_scatter"], total_rings)
        runtime.launch(kernels["polygon_buffer_ring_scatter"],
                       grid=scatter_grid, block=scatter_block, params=scatter_params)
        runtime.synchronize()

        # Counts D2H transfer was started before the scatter kernel;
        # wait for it now (should already be done).
        xfer_stream.synchronize()
        runtime.destroy_stream(xfer_stream)
        host_ring_counts = pinned_counts
        out_ring_offsets = np.empty(total_rings + 1, dtype=np.int32)
        out_ring_offsets[0] = 0
        np.cumsum(host_ring_counts, out=out_ring_offsets[1:])

        # geometry_offsets mirrors input (same ring structure per polygon)
        out_geometry_offsets = input_geo_offsets.copy()

        success = True
        return _build_device_backed_polygon_output_variable(
            device_x, device_y,
            row_count=polygons.row_count,
            geometry_offsets=out_geometry_offsets,
            ring_offsets=out_ring_offsets,
        )
    finally:
        runtime.free(device_radii)
        runtime.free(device_ring_to_row)
        runtime.free(device_ring_is_hole)
        runtime.free(device_ring_winding)
        runtime.free(device_ring_counts)
        if device_ring_offsets is not None:
            runtime.free(device_ring_offsets)
        if not success:
            if device_x is not None:
                runtime.free(device_x)
            if device_y is not None:
                runtime.free(device_y)


_CENTROID_GPU_THRESHOLD = 500


def polygon_centroids_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute polygon centroids directly from OwnedGeometryArray coordinate buffers.

    Uses the shoelace formula on ring coordinates (exterior ring only):
      cx = sum((x_i + x_{i+1}) * cross_i) / (6 * area)
      cy = sum((y_i + y_{i+1}) * cross_i) / (6 * area)
      where cross_i = x_i * y_{i+1} - x_{i+1} * y_i

    Returns (cx, cy) arrays of shape (row_count,).  GPU path uses
    ADR-0002 precision dispatch: fp32 with Kahan summation + coordinate
    centering on consumer GPUs, native fp64 on datacenter GPUs.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    # Dispatch decision
    selection = plan_dispatch_selection(
        kernel_name="polygon_centroid",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )
    if selection.selected is ExecutionMode.GPU and row_count >= _CENTROID_GPU_THRESHOLD:
        from vibespatial.runtime.precision import CoordinateStats
        # Compute coordinate stats for centering decision
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
                reason="polygon_centroid GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
            coordinate_stats=CoordinateStats(max_abs_coord=max_abs, span=span),
        )
        try:
            result = _polygon_centroids_gpu(owned, precision_plan=precision_plan)
            if result is not None:
                return result
        except Exception:
            pass  # fall through to CPU

    return _polygon_centroids_cpu(owned)


@register_kernel_variant(
    "polygon_centroid",
    "cpu",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("polygon", "multipolygon"),
    supports_mixed=True,
    tags=("numpy", "metric", "centroid"),
)
def _polygon_centroids_cpu(
    owned: OwnedGeometryArray,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU fallback: NumPy shoelace per polygon (Python loop)."""
    row_count = owned.row_count
    cx = np.empty(row_count, dtype=np.float64)
    cy = np.empty(row_count, dtype=np.float64)

    poly_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mpoly_tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for tag, family_key in ((poly_tag, GeometryFamily.POLYGON), (mpoly_tag, GeometryFamily.MULTIPOLYGON)):
        row_mask = tags == tag
        if not np.any(row_mask):
            continue
        if family_key not in owned.families:
            continue
        buf = owned.families[family_key]
        if buf.row_count == 0 or buf.geometry_offsets is None or len(buf.geometry_offsets) < 2:
            continue
        x = buf.x
        y = buf.y
        ring_offsets = buf.ring_offsets
        geom_offsets = buf.geometry_offsets

        global_rows = np.flatnonzero(row_mask)
        family_rows = family_row_offsets[global_rows]

        for i, (gr, fr) in enumerate(zip(global_rows, family_rows)):
            ring_start_idx = int(geom_offsets[fr])
            coord_start = int(ring_offsets[ring_start_idx])
            coord_end = int(ring_offsets[ring_start_idx + 1])

            xs = x[coord_start:coord_end]
            ys = y[coord_start:coord_end]

            if len(xs) < 3:
                cx[gr] = np.nan
                cy[gr] = np.nan
                continue

            xs_next = np.roll(xs, -1)
            ys_next = np.roll(ys, -1)
            cross = xs * ys_next - xs_next * ys
            signed_area = cross.sum() * 0.5

            if abs(signed_area) < 1e-30:
                cx[gr] = float(xs.mean())
                cy[gr] = float(ys.mean())
            else:
                cx[gr] = float(((xs + xs_next) * cross).sum() / (6.0 * signed_area))
                cy[gr] = float(((ys + ys_next) * cross).sum() / (6.0 * signed_area))

    # Handle any point/line rows that somehow got here (shouldn't happen, but safe)
    remaining = ~np.isin(tags, [poly_tag, mpoly_tag])
    if np.any(remaining):
        cx[remaining] = np.nan
        cy[remaining] = np.nan

    return cx, cy
