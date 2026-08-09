"""CUDA kernel source for GPU polygon intersection.

Contains the Sutherland-Hodgman polygon clipping NVRTC kernel source
(count + scatter passes) and kernel name tuple.

Extracted from polygon_intersection.py -- dispatch logic remains there.
"""

from __future__ import annotations

from vibespatial.cuda.device_functions.constructed_orientation import (
    CONSTRUCTED_ORIENTATION_DEVICE,
)
from vibespatial.cuda.device_functions.intersection_point import (
    INTERSECTION_POINT_DEVICE,
)
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE

_MAX_CLIP_VERTS = 128  # 4 buffers * 128 * 8 bytes = 4KB per thread; covers common buffered polygons

_POLYGON_INTERSECTION_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + CONSTRUCTED_ORIENTATION_DEVICE
    + INTERSECTION_POINT_DEVICE
    + STRIP_CLOSURE_DEVICE
    + r"""
#define MAX_CLIP_VERTS """
    + str(_MAX_CLIP_VERTS)
    + r"""

/* ------------------------------------------------------------------ */
/*  Sutherland-Hodgman: clip a polygon by a single edge               */
/*                                                                     */
/*  clip_edge defined by points (ex0,ey0) -> (ex1,ey1).               */
/*  "Inside" is the left side of the directed edge.                    */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ int classify_side(
    double px, double py,
    double ex0, double ey0,
    double ex1, double ey1
) {
    return vs_constructed_orient(ex0, ey0, ex1, ey1, px, py);
}

__device__ int line_intersect(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    double dx, double dy,
    double* ix, double* iy
) {
    return vs_proper_intersection_point_dd(
        ax, ay, bx, by, cx, cy, dx, dy, ix, iy
    );
}

__device__ __forceinline__ int approx_same_vertex(
    double ax, double ay,
    double bx, double by
) {
    return ax == bx && ay == by;
}

__device__ __forceinline__ int append_unique_vertex(
    double* out_x,
    double* out_y,
    int* out_count,
    double x,
    double y
) {
    if (*out_count > 0) {
        const int prev = *out_count - 1;
        if (approx_same_vertex(out_x[prev], out_y[prev], x, y)) {
            return 1;
        }
    }
    if (*out_count >= MAX_CLIP_VERTS) return 0;
    out_x[*out_count] = x;
    out_y[*out_count] = y;
    (*out_count)++;
    return 1;
}

__device__ __forceinline__ int collapse_redundant_vertices(
    double* xs,
    double* ys,
    int count
) {
    if (count <= 1) {
        return count;
    }

    int write = 1;
    for (int i = 1; i < count; i++) {
        if (approx_same_vertex(xs[write - 1], ys[write - 1], xs[i], ys[i])) {
            continue;
        }
        xs[write] = xs[i];
        ys[write] = ys[i];
        write++;
    }

    if (write > 1 && approx_same_vertex(xs[0], ys[0], xs[write - 1], ys[write - 1])) {
        write--;
    }
    return write;
}

__device__ __forceinline__ int constructed_ring_has_area_dimension(
    const double* xs,
    const double* ys,
    int count
) {
    if (count < 3) return 0;
    for (int i = 1; i + 1 < count; ++i) {
        if (vs_constructed_orient(
                xs[0], ys[0], xs[i], ys[i], xs[i + 1], ys[i + 1]) != 0) {
            return 1;
        }
    }
    return 0;
}

__device__ __forceinline__ int append_intersection_vertex(
    double* out_x,
    double* out_y,
    int* out_count,
    double sx, double sy,
    double px, double py,
    double ex0, double ey0,
    double ex1, double ey1,
    int s_side,
    int p_side
) {
    if (s_side == 0) {
        return append_unique_vertex(out_x, out_y, out_count, sx, sy);
    }
    if (p_side == 0) {
        return append_unique_vertex(out_x, out_y, out_count, px, py);
    }

    double ix, iy;
    if (!line_intersect(sx, sy, px, py, ex0, ey0, ex1, ey1, &ix, &iy)) {
        return 0;
    }
    return append_unique_vertex(out_x, out_y, out_count, ix, iy);
}

__device__ __forceinline__ int source_pair_has_uncertain_incidence(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    int left_start,
    int left_n,
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    int right_start,
    int right_n
) {
    for (int li = 0; li < left_n; ++li) {
        const double px = left_x[left_start + li];
        const double py = left_y[left_start + li];
        for (int ri = 0; ri < right_n; ++ri) {
            const int rj = ri == 0 ? right_n - 1 : ri - 1;
            if (vs_source_incidence_is_uncertain(
                    right_x[right_start + rj], right_y[right_start + rj],
                    right_x[right_start + ri], right_y[right_start + ri],
                    px, py)) {
                return 1;
            }
        }
    }
    for (int ri = 0; ri < right_n; ++ri) {
        const double px = right_x[right_start + ri];
        const double py = right_y[right_start + ri];
        for (int li = 0; li < left_n; ++li) {
            const int lj = li == 0 ? left_n - 1 : li - 1;
            if (vs_source_incidence_is_uncertain(
                    left_x[left_start + lj], left_y[left_start + lj],
                    left_x[left_start + li], left_y[left_start + li],
                    px, py)) {
                return 1;
            }
        }
    }
    return 0;
}

__device__ __forceinline__ int convex_open_ring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int coord_start,
    const int n
) {
    if (n < 3) {
        return 0;
    }
    int sign = 0;
    for (int i = 0; i < n; ++i) {
        const int prev = (i == 0) ? n - 1 : i - 1;
        const int next = (i + 1 == n) ? 0 : i + 1;
        const double px = x[coord_start + prev];
        const double py = y[coord_start + prev];
        const double cx = x[coord_start + i];
        const double cy = y[coord_start + i];
        const double nx = x[coord_start + next];
        const double ny = y[coord_start + next];
        const int current = vs_orient2d(px, py, cx, cy, nx, ny);
        if (current == 0) continue;
        if (sign == 0) {
            sign = current;
        } else if (sign != current) {
            return 0;
        }
    }
    return sign != 0;
}

__device__ __forceinline__ int open_ring_winding(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int start,
    const int count
) {
    const double origin_x = x[start];
    const double origin_y = y[start];
    double area = 0.0;
    double compensation = 0.0;
    for (int i = 0; i < count; ++i) {
        const int j = i + 1 < count ? i + 1 : 0;
        const double term =
            (x[start + i] - origin_x) * (y[start + j] - origin_y) -
            (x[start + j] - origin_x) * (y[start + i] - origin_y);
        const double corrected = term - compensation;
        const double next = area + corrected;
        compensation = (next - area) - corrected;
        area = next;
    }
    return (area > 0.0) - (area < 0.0);
}

extern "C" __global__ __launch_bounds__(256, 4) void polygon_intersection_sh_eligible(
    /* Logical row metadata */
    const unsigned char* __restrict__ left_valid,
    const signed char* __restrict__ left_tags,
    const int* __restrict__ left_family_rows,
    const unsigned char* __restrict__ right_valid,
    const signed char* __restrict__ right_tags,
    const int* __restrict__ right_family_rows,
    /* Left polygon buffers */
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_ring_offsets,
    const int* __restrict__ left_geom_offsets,
    const unsigned char* __restrict__ left_empty,
    int left_polygon_rows,
    /* Right polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const unsigned char* __restrict__ right_empty,
    int right_polygon_rows,
    int polygon_tag,
    int n,
    unsigned char* __restrict__ eligible
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    eligible[idx] = 0;

    const int l_row = left_family_rows[idx];
    const int r_row = right_family_rows[idx];
    if (
        !left_valid[idx] || !right_valid[idx] ||
        ((int)left_tags[idx]) != polygon_tag ||
        ((int)right_tags[idx]) != polygon_tag ||
        l_row < 0 || r_row < 0 ||
        l_row >= left_polygon_rows ||
        r_row >= right_polygon_rows ||
        left_empty[l_row] || right_empty[r_row]
    ) {
        return;
    }

    const int l_geom_start = left_geom_offsets[l_row];
    const int l_geom_end = left_geom_offsets[l_row + 1];
    const int r_geom_start = right_geom_offsets[r_row];
    const int r_geom_end = right_geom_offsets[r_row + 1];
    if ((l_geom_end - l_geom_start) != 1 || (r_geom_end - r_geom_start) != 1) {
        return;
    }

    const int l_coord_start = left_ring_offsets[l_geom_start];
    const int l_coord_end = left_ring_offsets[l_geom_start + 1];
    const int r_coord_start = right_ring_offsets[r_geom_start];
    const int r_coord_end = right_ring_offsets[r_geom_start + 1];
    int l_n = l_coord_end - l_coord_start;
    int r_n = r_coord_end - r_coord_start;
    l_n = vs_strip_closure(left_x, left_y, l_coord_start, l_coord_end, l_n, 0.0);
    r_n = vs_strip_closure(right_x, right_y, r_coord_start, r_coord_end, r_n, 0.0);
    if (
        l_n < 3 || r_n < 3 ||
        l_n > MAX_CLIP_VERTS ||
        r_n > MAX_CLIP_VERTS ||
        (l_n + r_n) > MAX_CLIP_VERTS
    ) {
        return;
    }
    if (
        !convex_open_ring(left_x, left_y, l_coord_start, l_n) ||
        !convex_open_ring(right_x, right_y, r_coord_start, r_n)
    ) {
        return;
    }
    if (source_pair_has_uncertain_incidence(
            left_x, left_y, l_coord_start, l_n,
            right_x, right_y, r_coord_start, r_n)) {
        return;
    }
    eligible[idx] = 1;
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
    const int* __restrict__ left_family_rows,
    int left_polygon_rows,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const int* __restrict__ right_family_rows,
    int right_polygon_rows,
    /* Validity masks (1=valid, 0=null/empty) */
    const int* __restrict__ left_valid,
    const int* __restrict__ right_valid,
    /* Output */
    int* __restrict__ out_counts,
    int* __restrict__ out_valid,
    int* __restrict__ out_supported,
    int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out_supported[idx] = 0;

    /* Invalid inputs -> empty output */
    if (!left_valid[idx] || !right_valid[idx]) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = 1;
        return;
    }

    const int l_row = left_family_rows[idx];
    const int r_row = right_family_rows[idx];
    if (
        l_row < 0 || r_row < 0 ||
        l_row >= left_polygon_rows ||
        r_row >= right_polygon_rows
    ) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = 1;
        return;
    }

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[l_row];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[r_row];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present (last == first). */
    l_n = vs_strip_closure(left_x, left_y, l_coord_start, l_coord_end, l_n, 0.0);
    r_n = vs_strip_closure(right_x, right_y, r_coord_start, r_coord_end, r_n, 0.0);

    /* Degenerate inputs -> empty */
    if (l_n < 3 || r_n < 3) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        return;
    }
    if (source_pair_has_uncertain_incidence(
            left_x, left_y, l_coord_start, l_n,
            right_x, right_y, r_coord_start, r_n)) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = 0;
        return;
    }

    const int clip_winding = open_ring_winding(
        right_x, right_y, r_coord_start, r_n
    );
    if (clip_winding == 0) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = 0;
        return;
    }
    const double wsign = (double)clip_winding;

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
        int stage_supported = 1;

        if (input_count == 0) break;

        for (int i = 0; i < input_count; i++) {
            int j = i + 1 < input_count ? i + 1 : 0;

            double sx = in_x[i], sy = in_y[i];
            double px = in_x[j], py = in_y[j];

            int s_side = classify_side(sx, sy, ex0, ey0, ex1, ey1);
            int p_side = classify_side(px, py, ex0, ey0, ex1, ey1);
            s_side = (int)wsign * s_side;
            p_side = (int)wsign * p_side;

            if (s_side >= 0) {
                /* S is inside */
                stage_supported = append_unique_vertex(
                    out_x, out_y, &out_count, sx, sy
                );
                if (p_side < 0) {
                    /* S inside, P outside -> emit intersection */
                    stage_supported = stage_supported && append_intersection_vertex(
                        out_x, out_y, &out_count,
                        sx, sy, px, py,
                        ex0, ey0, ex1, ey1,
                        s_side, p_side
                    );
                }
            } else {
                /* S is outside */
                if (p_side >= 0) {
                    /* S outside, P inside -> emit intersection then P */
                    stage_supported = append_intersection_vertex(
                        out_x, out_y, &out_count,
                        sx, sy, px, py,
                        ex0, ey0, ex1, ey1,
                        s_side, p_side
                    );
                }
            }
            if (!stage_supported) break;
        }

        if (!stage_supported) {
            out_counts[idx] = 0;
            out_valid[idx] = 0;
            out_supported[idx] = 0;
            return;
        }

        out_count = collapse_redundant_vertices(out_x, out_y, out_count);

        /* Swap buffers for next edge */
        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_x;
        in_y = out_y;
        out_x = tmp_x;
        out_y = tmp_y;
        input_count = out_count;
    }

    input_count = collapse_redundant_vertices(in_x, in_y, input_count);

    if (input_count < 3) {
        /* A point/line contact needs the exact mixed-dimensional carrier. */
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = input_count == 0 ? 1 : 0;
    } else if (!constructed_ring_has_area_dimension(in_x, in_y, input_count)) {
        out_counts[idx] = 0;
        out_valid[idx] = 0;
        out_supported[idx] = 0;
    } else {
        /* +1 for closing vertex */
        out_counts[idx] = input_count + 1;
        out_valid[idx] = 1;
        out_supported[idx] = 1;
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
    const int* __restrict__ left_family_rows,
    int left_polygon_rows,
    /* Right (clip) polygon buffers */
    const double* __restrict__ right_x,
    const double* __restrict__ right_y,
    const int* __restrict__ right_ring_offsets,
    const int* __restrict__ right_geom_offsets,
    const int* __restrict__ right_family_rows,
    int right_polygon_rows,
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

    const int l_row = left_family_rows[idx];
    const int r_row = right_family_rows[idx];
    if (
        l_row < 0 || r_row < 0 ||
        l_row >= left_polygon_rows ||
        r_row >= right_polygon_rows
    ) {
        return;
    }

    /* Get exterior ring bounds for left (subject) polygon */
    const int l_first_ring = left_geom_offsets[l_row];
    const int l_coord_start = left_ring_offsets[l_first_ring];
    const int l_coord_end = left_ring_offsets[l_first_ring + 1];
    int l_n = l_coord_end - l_coord_start;

    /* Get exterior ring bounds for right (clip) polygon */
    const int r_first_ring = right_geom_offsets[r_row];
    const int r_coord_start = right_ring_offsets[r_first_ring];
    const int r_coord_end = right_ring_offsets[r_first_ring + 1];
    int r_n = r_coord_end - r_coord_start;

    /* Strip closing vertex if present. */
    l_n = vs_strip_closure(left_x, left_y, l_coord_start, l_coord_end, l_n, 0.0);
    r_n = vs_strip_closure(right_x, right_y, r_coord_start, r_coord_end, r_n, 0.0);

    const int clip_winding = open_ring_winding(
        right_x, right_y, r_coord_start, r_n
    );
    if (clip_winding == 0) return;
    const double wsign = (double)clip_winding;

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

            int s_side = classify_side(sx, sy, ex0, ey0, ex1, ey1);
            int p_side = classify_side(px, py, ex0, ey0, ex1, ey1);
            s_side = (int)wsign * s_side;
            p_side = (int)wsign * p_side;

            if (s_side >= 0) {
                if (!append_unique_vertex(out_bx, out_by, &out_count, sx, sy)) return;
                if (p_side < 0) {
                    if (!append_intersection_vertex(
                        out_bx, out_by, &out_count,
                        sx, sy, px, py,
                        ex0, ey0, ex1, ey1,
                        s_side, p_side
                    )) return;
                }
            } else {
                if (p_side >= 0) {
                    if (!append_intersection_vertex(
                        out_bx, out_by, &out_count,
                        sx, sy, px, py,
                        ex0, ey0, ex1, ey1,
                        s_side, p_side
                    )) return;
                }
            }
        }

        out_count = collapse_redundant_vertices(out_bx, out_by, out_count);

        double* tmp_x = in_x;
        double* tmp_y = in_y;
        in_x = out_bx;
        in_y = out_by;
        out_bx = tmp_x;
        out_by = tmp_y;
        input_count = out_count;
    }

    input_count = collapse_redundant_vertices(in_x, in_y, input_count);
    if (input_count < 3) return;

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
)

_KERNEL_NAMES = (
    "polygon_intersection_count",
    "polygon_intersection_scatter",
    "polygon_intersection_sh_eligible",
)
