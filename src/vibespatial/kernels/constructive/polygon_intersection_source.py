"""CUDA kernel source for GPU polygon intersection.

Contains the Sutherland-Hodgman polygon clipping NVRTC kernel source
(count + scatter passes) and kernel name tuple.

Extracted from polygon_intersection.py -- dispatch logic remains there.
"""
from __future__ import annotations

from vibespatial.cuda.device_functions.signed_area import SIGNED_AREA_DEVICE

_MAX_CLIP_VERTS = 64  # 4 buffers * 64 * 8 bytes = 2KB per thread (vs 8KB at 256)

_POLYGON_INTERSECTION_KERNEL_SOURCE = SIGNED_AREA_DEVICE + r"""
#define MAX_CLIP_VERTS """ + str(_MAX_CLIP_VERTS) + r"""

/* ------------------------------------------------------------------ */
/*  Sutherland-Hodgman: clip a polygon by a single edge               */
/*                                                                     */
/*  clip_edge defined by points (ex0,ey0) -> (ex1,ey1).               */
/*  "Inside" is the left side of the directed edge.                    */
/* ------------------------------------------------------------------ */

__device__ double cross_sign(
    double px, double py,
    double ex0, double ey0,
    double ex1, double ey1
) {
    return (ex1 - ex0) * (py - ey0) - (ey1 - ey0) * (px - ex0);
}

__device__ void line_intersect(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    double dx, double dy,
    double* ix, double* iy
) {
    /* Intersection of line (a->b) with line (c->d). */
    double a1 = by - ay;
    double b1 = ax - bx;
    double c1 = a1 * ax + b1 * ay;

    double a2 = dy - cy;
    double b2 = cx - dx;
    double c2 = a2 * cx + b2 * cy;

    double det = a1 * b2 - a2 * b1;
    if (fabs(det) < 1e-15) {
        /* Parallel lines -- use midpoint of the shared segment. */
        *ix = (ax + bx) * 0.5;
        *iy = (ay + by) * 0.5;
    } else {
        *ix = (c1 * b2 - c2 * b1) / det;
        *iy = (a1 * c2 - a2 * c1) / det;
    }
}

/* ------------------------------------------------------------------ */
/*  Count kernel: compute output vertex count per pair                 */
/*                                                                     */
/*  One thread per geometry pair.  Runs Sutherland-Hodgman in          */
/*  registers/local memory to count output vertices.                   */
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void polygon_intersection_count(
    /* Left (subject) polygon buffers */
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    /* Validity masks (1=valid, 0=null/empty) */
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    /* Output */
    int* __restrict__ out_counts,
    int* __restrict__ out_valid,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* Invalid inputs -> empty output */
    if (!left_valid[idx] || !right_valid[idx]) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[idx];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[idx];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present (last == first). */
    if (l_n >= 2) {
        double dx = left_x[l_coord_start] - left_x[l_coord_end - 1];
        double dy = left_y[l_coord_start] - left_y[l_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) l_n--;
    }
    if (r_n >= 2) {
        double dx = right_x[r_coord_start] - right_x[r_coord_end - 1];
        double dy = right_y[r_coord_start] - right_y[r_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) r_n--;
    }

    /* Degenerate inputs -> empty */
    if (l_n < 3 || r_n < 3) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }

    /* Detect winding direction of the clip (right) polygon.
       Sutherland-Hodgman assumes the clip polygon is CCW (interior on
       the left side of each directed edge).  If the clip polygon is CW,
       we negate the cross_sign results so the inside/outside test still
       works correctly.  This handles arbitrary input winding without
       requiring a pre-normalization step. */
    const double clip_area2 = vs_ring_signed_area_2x_open(
        right_x, right_y, r_coord_start, r_n);
    const double wsign = (clip_area2 >= 0.0) ? 1.0 : -1.0;

    /* Local workspace for Sutherland-Hodgman.
       We alternate between buf_a and buf_b. */
    double buf_ax[MAX_CLIP_VERTS], buf_ay[MAX_CLIP_VERTS];
    double buf_bx[MAX_CLIP_VERTS], buf_by[MAX_CLIP_VERTS];

    /* Initialize buf_a with the subject polygon vertices. */
    int input_count;
    if (l_n > MAX_CLIP_VERTS) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }
    for (int i = 0; i < l_n; i++) {
        buf_ax[i] = left_x[l_coord_start + i];
        buf_ay[i] = left_y[l_coord_start + i];
    }
    input_count = l_n;

    /* For each edge of the clip polygon, clip the current polygon. */
    double* in_x = buf_ax;
    double* in_y = buf_ay;
    double* out_x = buf_bx;
    double* out_y = buf_by;

    for (int e = 0; e < r_n; e++) {
        double ex0 = right_x[r_coord_start + e];
        double ey0 = right_y[r_coord_start + e];
        double ex1 = right_x[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];
        double ey1 = right_y[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];

        int out_count = 0;

        if (input_count == 0) break;

        for (int i = 0; i < input_count; i++) {
            int j = i + 1 < input_count ? i + 1 : 0;

            double sx = in_x[i], sy = in_y[i];
            double px = in_x[j], py = in_y[j];

            double s_side = wsign * cross_sign(sx, sy, ex0, ey0, ex1, ey1);
            double p_side = wsign * cross_sign(px, py, ex0, ey0, ex1, ey1);

            if (s_side >= 0.0) {
                /* S is inside */
                if (out_count < MAX_CLIP_VERTS) {
                    out_x[out_count] = sx;
                    out_y[out_count] = sy;
                    out_count++;
                }
                if (p_side < 0.0) {
                    /* S inside, P outside -> emit intersection */
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_x[out_count] = ix;
                        out_y[out_count] = iy;
                        out_count++;
                    }
                }
            } else {
                /* S is outside */
                if (p_side >= 0.0) {
                    /* S outside, P inside -> emit intersection then P */
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_x[out_count] = ix;
                        out_y[out_count] = iy;
                        out_count++;
                    }
                }
            }
        }

        /* Swap buffers for next edge */
        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_x;
        in_y = out_y;
        out_x = tmp_x;
        out_y = tmp_y;
        input_count = out_count;
    }

    if (input_count < 3) {
        /* Degenerate result (point or line) -> empty */
        out_counts[idx] = 0;
        out_valid[idx] = 0;
    } else {
        /* +1 for closing vertex */
        out_counts[idx] = input_count + 1;
        out_valid[idx] = 1;
    }
}

/* ------------------------------------------------------------------ */
/*  Scatter kernel: write clipped polygon vertices to output           */
/*                                                                     */
/*  Re-runs Sutherland-Hodgman (same as count pass) and writes         */
/*  the result vertices at the pre-computed offsets.                    */
/* ------------------------------------------------------------------ */

extern "C" __global__ __launch_bounds__(256, 2) void polygon_intersection_scatter(
    /* Left (subject) polygon buffers */
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    /* Validity masks */
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    /* Scatter targets */
    const int* __restrict__ output_offsets,
    const int* __restrict__ output_valid,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!output_valid[idx]) return;

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[idx];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[idx];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present. */
    if (l_n >= 2) {
        double dx = left_x[l_coord_start] - left_x[l_coord_end - 1];
        double dy = left_y[l_coord_start] - left_y[l_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) l_n--;
    }
    if (r_n >= 2) {
        double dx = right_x[r_coord_start] - right_x[r_coord_end - 1];
        double dy = right_y[r_coord_start] - right_y[r_coord_end - 1];
        if (dx * dx + dy * dy < 1e-24) r_n--;
    }

    /* Detect winding direction of the clip polygon (same as count pass). */
    const double clip_area2 = vs_ring_signed_area_2x_open(
        right_x, right_y, r_coord_start, r_n);
    const double wsign = (clip_area2 >= 0.0) ? 1.0 : -1.0;

    /* Local workspace for Sutherland-Hodgman. */
    double buf_ax[MAX_CLIP_VERTS], buf_ay[MAX_CLIP_VERTS];
    double buf_bx[MAX_CLIP_VERTS], buf_by[MAX_CLIP_VERTS];

    int input_count;
    if (l_n > MAX_CLIP_VERTS) return;
    for (int i = 0; i < l_n; i++) {
        buf_ax[i] = left_x[l_coord_start + i];
        buf_ay[i] = left_y[l_coord_start + i];
    }
    input_count = l_n;

    double* in_x = buf_ax;
    double* in_y = buf_ay;
    double* out_bx = buf_bx;
    double* out_by = buf_by;

    for (int e = 0; e < r_n; e++) {
        double ex0 = right_x[r_coord_start + e];
        double ey0 = right_y[r_coord_start + e];
        double ex1 = right_x[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];
        double ey1 = right_y[r_coord_start + (e + 1 < r_n ? e + 1 : 0)];

        int out_count = 0;
        if (input_count == 0) break;

        for (int i = 0; i < input_count; i++) {
            int j = i + 1 < input_count ? i + 1 : 0;

            double sx = in_x[i], sy = in_y[i];
            double px = in_x[j], py = in_y[j];

            double s_side = wsign * cross_sign(sx, sy, ex0, ey0, ex1, ey1);
            double p_side = wsign * cross_sign(px, py, ex0, ey0, ex1, ey1);

            if (s_side >= 0.0) {
                if (out_count < MAX_CLIP_VERTS) {
                    out_bx[out_count] = sx;
                    out_by[out_count] = sy;
                    out_count++;
                }
                if (p_side < 0.0) {
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_bx[out_count] = ix;
                        out_by[out_count] = iy;
                        out_count++;
                    }
                }
            } else {
                if (p_side >= 0.0) {
                    double ix, iy;
                    line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy);
                    if (out_count < MAX_CLIP_VERTS) {
                        out_bx[out_count] = ix;
                        out_by[out_count] = iy;
                        out_count++;
                    }
                }
            }
        }

        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_bx;
        in_y = out_by;
        out_bx = tmp_x;
        out_by = tmp_y;
        input_count = out_count;
    }

    /* Write clipped vertices at the pre-computed offset. */
    int pos = output_offsets[idx];
    for (int i = 0; i < input_count; i++) {
        out_x[pos + i] = in_x[i];
        out_y[pos + i] = in_y[i];
    }
    /* Closing vertex: first vertex repeated. */
    out_x[pos + input_count] = in_x[0];
    out_y[pos + input_count] = in_y[0];
}
"""

_KERNEL_NAMES = ("polygon_intersection_count", "polygon_intersection_scatter")
