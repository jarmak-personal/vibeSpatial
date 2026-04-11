"""NVRTC kernel sources for polygon."""

from __future__ import annotations

from vibespatial.cuda.device_functions.signed_area import SIGNED_AREA_DEVICE
from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE
from vibespatial.cuda.preamble import PRECISION_PREAMBLE, SPATIAL_TOLERANCE_PREAMBLE

_POLYGON_BUFFER_KERNEL_SOURCE = SPATIAL_TOLERANCE_PREAMBLE + r"""
#define PI 3.14159265358979323846
#define EPSILON VS_SPATIAL_EPSILON

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
_RING_WINDING_KERNEL_SOURCE = SIGNED_AREA_DEVICE + r"""
extern "C" __global__ void compute_ring_winding(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    double* __restrict__ ring_winding,
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

    const double area2 = vs_ring_signed_area_2x(x, y, coord_start, coord_end);
    ring_winding[ring] = area2 > 0.0 ? 1.0 : -1.0;
}
"""
_RING_WINDING_KERNEL_NAMES = ("compute_ring_winding",)
# ---------------------------------------------------------------------------
# GPU Polygon Centroid Kernel (NVRTC Tier 1)
# ---------------------------------------------------------------------------
# One thread per polygonal row.  The row centroid must incorporate every
# polygon part and subtract hole contributions, not just the first ring.
# Ring centroid coordinates are orientation-invariant; we therefore weight the
# first ring of each polygon part by +abs(area) and later rings by -abs(area).

_POLYGON_CENTROID_KERNEL_SOURCE = PRECISION_PREAMBLE + STRIP_CLOSURE_DEVICE + r"""
__device__ int polygon_centroid_ring_terms(
    const double* x,
    const double* y,
    int coord_start,
    int coord_end,
    double center_x,
    double center_y,
    compute_t* out_abs_area,
    compute_t* out_cx,
    compute_t* out_cy,
    compute_t* out_mean_x,
    compute_t* out_mean_y,
    int* out_mean_n
) {{
    int n = coord_end - coord_start;
    n = vs_strip_closure(x, y, coord_start, coord_end, n, 1e-24);
    if (n < 3) {{
        *out_abs_area = (compute_t)0.0;
        *out_cx = (compute_t)0.0;
        *out_cy = (compute_t)0.0;
        *out_mean_x = (compute_t)0.0;
        *out_mean_y = (compute_t)0.0;
        *out_mean_n = 0;
        return 0;
    }}

    compute_t sum_area = (compute_t)0.0;
    compute_t sum_cx = (compute_t)0.0;
    compute_t sum_cy = (compute_t)0.0;
    compute_t c_area = (compute_t)0.0;
    compute_t c_cx = (compute_t)0.0;
    compute_t c_cy = (compute_t)0.0;

    for (int i = 0; i < n; i++) {{
        const int cur = coord_start + i;
        const int nxt = coord_start + ((i + 1) % n);
        const compute_t xi  = (compute_t)(x[cur] - center_x);
        const compute_t yi  = (compute_t)(y[cur] - center_y);
        const compute_t xi1 = (compute_t)(x[nxt] - center_x);
        const compute_t yi1 = (compute_t)(y[nxt] - center_y);

        const compute_t cross = xi * yi1 - xi1 * yi;
        KAHAN_ADD(sum_area, cross, c_area);
        KAHAN_ADD(sum_cx, (xi + xi1) * cross, c_cx);
        KAHAN_ADD(sum_cy, (yi + yi1) * cross, c_cy);
    }}

    const compute_t signed_area = sum_area * (compute_t)0.5;
    if (fabs(signed_area) < (compute_t)1e-30) {{
        compute_t mx = (compute_t)0.0;
        compute_t my = (compute_t)0.0;
        for (int i = 0; i < n; i++) {{
            mx += (compute_t)(x[coord_start + i] - center_x);
            my += (compute_t)(y[coord_start + i] - center_y);
        }}
        *out_abs_area = (compute_t)0.0;
        *out_cx = (compute_t)0.0;
        *out_cy = (compute_t)0.0;
        *out_mean_x = mx;
        *out_mean_y = my;
        *out_mean_n = n;
        return 1;
    }}

    *out_abs_area = fabs(signed_area);
    *out_cx = sum_cx / ((compute_t)6.0 * signed_area);
    *out_cy = sum_cy / ((compute_t)6.0 * signed_area);
    *out_mean_x = (compute_t)0.0;
    *out_mean_y = (compute_t)0.0;
    *out_mean_n = 0;
    return 1;
}}

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

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];
    if (first_ring == last_ring) {{
        cx[row] = 0.0 / 0.0;
        cy[row] = 0.0 / 0.0;
        return;
    }}

    compute_t total_area = (compute_t)0.0;
    compute_t total_cx = (compute_t)0.0;
    compute_t total_cy = (compute_t)0.0;
    compute_t c_area = (compute_t)0.0;
    compute_t c_cx = (compute_t)0.0;
    compute_t c_cy = (compute_t)0.0;
    compute_t mean_x = (compute_t)0.0;
    compute_t mean_y = (compute_t)0.0;
    int mean_n = 0;

    for (int ring = first_ring; ring < last_ring; ++ring) {{
        compute_t ring_abs_area;
        compute_t ring_cx;
        compute_t ring_cy;
        compute_t ring_mean_x;
        compute_t ring_mean_y;
        int ring_mean_n;
        const int coord_start = ring_offsets[ring];
        const int coord_end = ring_offsets[ring + 1];
        if (!polygon_centroid_ring_terms(
            x, y, coord_start, coord_end,
            center_x, center_y,
            &ring_abs_area, &ring_cx, &ring_cy,
            &ring_mean_x, &ring_mean_y, &ring_mean_n
        )) {{
            continue;
        }}
        if (ring_mean_n > 0) {{
            mean_x += ring_mean_x;
            mean_y += ring_mean_y;
            mean_n += ring_mean_n;
            continue;
        }}
        const compute_t weight = (ring == first_ring) ? ring_abs_area : -ring_abs_area;
        KAHAN_ADD(total_area, weight, c_area);
        KAHAN_ADD(total_cx, ring_cx * weight, c_cx);
        KAHAN_ADD(total_cy, ring_cy * weight, c_cy);
    }}

    if (fabs(total_area) < (compute_t)1e-30) {{
        if (mean_n == 0) {{
            cx[row] = 0.0 / 0.0;
            cy[row] = 0.0 / 0.0;
            return;
        }}
        cx[row] = (double)(mean_x / (compute_t)mean_n) + center_x;
        cy[row] = (double)(mean_y / (compute_t)mean_n) + center_y;
    }} else {{
        cx[row] = (double)(total_cx / total_area) + center_x;
        cy[row] = (double)(total_cy / total_area) + center_y;
    }}
}}

extern "C" __global__ void multipolygon_centroid(
    const double* x,
    const double* y,
    const int* ring_offsets,
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

    const int first_polygon = geometry_offsets[row];
    const int last_polygon = geometry_offsets[row + 1];
    if (first_polygon == last_polygon) {{
        cx[row] = 0.0 / 0.0;
        cy[row] = 0.0 / 0.0;
        return;
    }}

    compute_t total_area = (compute_t)0.0;
    compute_t total_cx = (compute_t)0.0;
    compute_t total_cy = (compute_t)0.0;
    compute_t c_area = (compute_t)0.0;
    compute_t c_cx = (compute_t)0.0;
    compute_t c_cy = (compute_t)0.0;
    compute_t mean_x = (compute_t)0.0;
    compute_t mean_y = (compute_t)0.0;
    int mean_n = 0;

    for (int polygon = first_polygon; polygon < last_polygon; ++polygon) {{
        const int first_ring = part_offsets[polygon];
        const int last_ring = part_offsets[polygon + 1];
        for (int ring = first_ring; ring < last_ring; ++ring) {{
            compute_t ring_abs_area;
            compute_t ring_cx;
            compute_t ring_cy;
            compute_t ring_mean_x;
            compute_t ring_mean_y;
            int ring_mean_n;
            const int coord_start = ring_offsets[ring];
            const int coord_end = ring_offsets[ring + 1];
            if (!polygon_centroid_ring_terms(
                x, y, coord_start, coord_end,
                center_x, center_y,
                &ring_abs_area, &ring_cx, &ring_cy,
                &ring_mean_x, &ring_mean_y, &ring_mean_n
            )) {{
                continue;
            }}
            if (ring_mean_n > 0) {{
                mean_x += ring_mean_x;
                mean_y += ring_mean_y;
                mean_n += ring_mean_n;
                continue;
            }}
            const compute_t weight = (ring == first_ring) ? ring_abs_area : -ring_abs_area;
            KAHAN_ADD(total_area, weight, c_area);
            KAHAN_ADD(total_cx, ring_cx * weight, c_cx);
            KAHAN_ADD(total_cy, ring_cy * weight, c_cy);
        }}
    }}

    if (fabs(total_area) < (compute_t)1e-30) {{
        if (mean_n == 0) {{
            cx[row] = 0.0 / 0.0;
            cy[row] = 0.0 / 0.0;
            return;
        }}
        cx[row] = (double)(mean_x / (compute_t)mean_n) + center_x;
        cy[row] = (double)(mean_y / (compute_t)mean_n) + center_y;
    }} else {{
        cx[row] = (double)(total_cx / total_area) + center_x;
        cy[row] = (double)(total_cy / total_area) + center_y;
    }}
}}
"""
_POLYGON_CENTROID_KERNEL_NAMES = ("polygon_centroid", "multipolygon_centroid")
_POLYGON_CENTROID_FP64_SOURCE = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="double")
_POLYGON_CENTROID_FP32_SOURCE = _POLYGON_CENTROID_KERNEL_SOURCE.format(compute_type="float")
