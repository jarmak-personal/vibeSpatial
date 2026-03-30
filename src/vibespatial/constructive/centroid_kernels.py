"""NVRTC kernel sources for centroid."""

from __future__ import annotations

from vibespatial.constructive.polygon_kernels import _POLYGON_CENTROID_KERNEL_SOURCE
from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# Cooperative polygon centroid kernel: 1 block per geometry (for complex polygons)
# ---------------------------------------------------------------------------

_POLYGON_CENTROID_COOPERATIVE_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ __launch_bounds__(256, 4)
void polygon_centroid_cooperative(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ cx,
    double* __restrict__ cy,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x;
    if (row >= row_count) return;

    /* Polygon centroid uses only the exterior ring (first ring). */
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
        if (threadIdx.x == 0) {{
            cx[row] = 0.0 / 0.0;  /* NaN */
            cy[row] = 0.0 / 0.0;
        }}
        return;
    }}

    /* Shared memory for inter-warp Kahan reduction (up to 256/32 = 8 warps) */
    __shared__ compute_t warp_area[8];
    __shared__ compute_t warp_area_c[8];
    __shared__ compute_t warp_cx[8];
    __shared__ compute_t warp_cx_c[8];
    __shared__ compute_t warp_cy[8];
    __shared__ compute_t warp_cy_c[8];

    /* Each thread accumulates partial sums with Kahan */
    compute_t partial_area = (compute_t)0.0;
    compute_t partial_area_c = (compute_t)0.0;
    compute_t partial_cx = (compute_t)0.0;
    compute_t partial_cx_c = (compute_t)0.0;
    compute_t partial_cy = (compute_t)0.0;
    compute_t partial_cy_c = (compute_t)0.0;

    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {{
        const int cur = coord_start + i;
        const int nxt = coord_start + ((i + 1) % n);
        const compute_t xi  = CX(x[cur]);
        const compute_t yi  = CY(y[cur]);
        const compute_t xi1 = CX(x[nxt]);
        const compute_t yi1 = CY(y[nxt]);

        const compute_t cross = xi * yi1 - xi1 * yi;
        KAHAN_ADD(partial_area, cross, partial_area_c);
        KAHAN_ADD(partial_cx, (xi + xi1) * cross, partial_cx_c);
        KAHAN_ADD(partial_cy, (yi + yi1) * cross, partial_cy_c);
    }}

    /* Warp-level reduction via __shfl_down_sync */
    const unsigned int FULL_MASK = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1) {{
        compute_t s_area = __shfl_down_sync(FULL_MASK, partial_area, offset);
        compute_t s_area_c = __shfl_down_sync(FULL_MASK, partial_area_c, offset);
        KAHAN_ADD(partial_area, s_area - s_area_c, partial_area_c);

        compute_t s_cx = __shfl_down_sync(FULL_MASK, partial_cx, offset);
        compute_t s_cx_c = __shfl_down_sync(FULL_MASK, partial_cx_c, offset);
        KAHAN_ADD(partial_cx, s_cx - s_cx_c, partial_cx_c);

        compute_t s_cy = __shfl_down_sync(FULL_MASK, partial_cy, offset);
        compute_t s_cy_c = __shfl_down_sync(FULL_MASK, partial_cy_c, offset);
        KAHAN_ADD(partial_cy, s_cy - s_cy_c, partial_cy_c);
    }}

    /* Block-level reduction via shared memory */
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) {{
        warp_area[warp_id] = partial_area;
        warp_area_c[warp_id] = partial_area_c;
        warp_cx[warp_id] = partial_cx;
        warp_cx_c[warp_id] = partial_cx_c;
        warp_cy[warp_id] = partial_cy;
        warp_cy_c[warp_id] = partial_cy_c;
    }}
    __syncthreads();

    /* Thread 0 combines across warps with Kahan */
    if (threadIdx.x == 0) {{
        compute_t sum_area = (compute_t)0.0;
        compute_t c_area = (compute_t)0.0;
        compute_t sum_cx = (compute_t)0.0;
        compute_t c_cx = (compute_t)0.0;
        compute_t sum_cy = (compute_t)0.0;
        compute_t c_cy = (compute_t)0.0;
        const int num_warps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < num_warps; ++w) {{
            KAHAN_ADD(sum_area, warp_area[w] - warp_area_c[w], c_area);
            KAHAN_ADD(sum_cx, warp_cx[w] - warp_cx_c[w], c_cx);
            KAHAN_ADD(sum_cy, warp_cy[w] - warp_cy_c[w], c_cy);
        }}

        const compute_t signed_area = sum_area * (compute_t)0.5;

        if (fabs(signed_area) < (compute_t)1e-30) {{
            /* Degenerate polygon -- fall back to coordinate mean. */
            compute_t mx = (compute_t)0.0, my = (compute_t)0.0;
            for (int i = 0; i < n; i++) {{
                mx += CX(x[coord_start + i]);
                my += CY(y[coord_start + i]);
            }}
            cx[row] = (double)(mx / (compute_t)n) + center_x;
            cy[row] = (double)(my / (compute_t)n) + center_y;
        }} else {{
            cx[row] = (double)(sum_cx / ((compute_t)6.0 * signed_area)) + center_x;
            cy[row] = (double)(sum_cy / ((compute_t)6.0 * signed_area)) + center_y;
        }}
    }}
}}
"""
# ---------------------------------------------------------------------------
# Point centroid kernel: identity copy (x, y -> cx, cy)
# ---------------------------------------------------------------------------

_POINT_CENTROID_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
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

_MULTIPOINT_CENTROID_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
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

_LINESTRING_CENTROID_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
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

_MULTILINESTRING_CENTROID_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
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
_POLYGON_CENTROID_COOPERATIVE_FP64 = _POLYGON_CENTROID_COOPERATIVE_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_CENTROID_COOPERATIVE_FP32 = _POLYGON_CENTROID_COOPERATIVE_KERNEL_SOURCE.format(compute_type="float")
