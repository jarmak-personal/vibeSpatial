"""NVRTC kernel sources for minimum_rotated_rectangle."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVRTC kernel source: rotating calipers minimum-area bounding rectangle
#
# One thread per geometry.  Input is the convex hull (closed ring, Polygon
# family).  For each edge of the hull, the kernel rotates all hull vertices
# into the edge-aligned coordinate system, computes the axis-aligned
# bounding box, and tracks the minimum-area rectangle.  The 4 corners of
# the best rectangle are un-rotated back and written as a 5-vertex closed
# polygon.
#
# ADR-0033 Tier 1: geometry-specific inner loop (rotating calipers).
# ADR-0002: CONSTRUCTIVE class, always fp64.
# ---------------------------------------------------------------------------

_MIN_RECT_KERNEL_SOURCE = r"""
extern "C" __global__ void minimum_rotated_rectangle(
    const double* __restrict__ hull_x,
    const double* __restrict__ hull_y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geom_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    /* Determine the hull ring span for this geometry.
       Polygon layout: geom_offsets[row]..geom_offsets[row+1] are ring indices,
       ring_offsets[ring_idx]..ring_offsets[ring_idx+1] are coord indices.
       Convex hull output always has exactly 1 ring per polygon. */
    const int ring_idx = geom_offsets[row];
    const int cs = ring_offsets[ring_idx];
    const int ce = ring_offsets[ring_idx + 1];
    const int n_coords = ce - cs;  /* includes closing vertex */

    /* Output base: 5 vertices per geometry */
    const int out_base = row * 5;

    /* Degenerate: 0 or 1 coords -> point at origin or single coord */
    if (n_coords <= 1) {
        double px = (n_coords == 1) ? hull_x[cs] : 0.0;
        double py = (n_coords == 1) ? hull_y[cs] : 0.0;
        for (int i = 0; i < 5; i++) {
            out_x[out_base + i] = px;
            out_y[out_base + i] = py;
        }
        return;
    }

    /* Number of unique hull vertices (closed ring: n_coords - 1) */
    const int n_verts = n_coords - 1;

    /* Degenerate: 2 unique vertices -> line segment, rectangle is the segment */
    if (n_verts <= 2) {
        double x0 = hull_x[cs];
        double y0 = hull_y[cs];
        double x1 = hull_x[cs + 1];
        double y1 = hull_y[cs + 1];
        out_x[out_base + 0] = x0;  out_y[out_base + 0] = y0;
        out_x[out_base + 1] = x1;  out_y[out_base + 1] = y1;
        out_x[out_base + 2] = x1;  out_y[out_base + 2] = y1;
        out_x[out_base + 3] = x0;  out_y[out_base + 3] = y0;
        out_x[out_base + 4] = x0;  out_y[out_base + 4] = y0;
        return;
    }

    /* Rotating calipers: for each edge, rotate into edge-aligned frame,
       compute AABB, track minimum area. */
    double best_area = 1.0e300;  /* sentinel: larger than any real area */
    const double px0 = hull_x[cs];
    const double py0 = hull_y[cs];
    double best_c0x = px0, best_c0y = py0, best_c1x = px0, best_c1y = py0;
    double best_c2x = px0, best_c2y = py0, best_c3x = px0, best_c3y = py0;

    for (int e = 0; e < n_verts; e++) {
        /* Edge from hull[e] to hull[(e+1) % n_verts] */
        const int i0 = cs + e;
        const int i1 = cs + ((e + 1) % n_verts);
        const double ex = hull_x[i1] - hull_x[i0];
        const double ey = hull_y[i1] - hull_y[i0];

        /* Edge length and normalized direction */
        const double edge_len = sqrt(ex * ex + ey * ey);
        if (edge_len < 1.0e-15) continue;  /* degenerate zero-length edge */

        const double dx = ex / edge_len;
        const double dy = ey / edge_len;

        /* Rotate all hull vertices into edge-aligned frame:
           u = x*dx + y*dy   (along edge)
           v = -x*dy + y*dx  (perpendicular to edge) */
        double min_u = 1.0e300, max_u = -1.0e300;
        double min_v = 1.0e300, max_v = -1.0e300;

        for (int j = 0; j < n_verts; j++) {
            const double vx = hull_x[cs + j];
            const double vy = hull_y[cs + j];
            const double u = vx * dx + vy * dy;
            const double v = -vx * dy + vy * dx;

            if (u < min_u) min_u = u;
            if (u > max_u) max_u = u;
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }

        const double area = (max_u - min_u) * (max_v - min_v);

        if (area < best_area) {
            best_area = area;

            /* Un-rotate the 4 corners back to original space:
               x = u*dx - v*dy
               y = u*dy + v*dx */
            best_c0x = min_u * dx - min_v * dy;
            best_c0y = min_u * dy + min_v * dx;

            best_c1x = max_u * dx - min_v * dy;
            best_c1y = max_u * dy + min_v * dx;

            best_c2x = max_u * dx - max_v * dy;
            best_c2y = max_u * dy + max_v * dx;

            best_c3x = min_u * dx - max_v * dy;
            best_c3y = min_u * dy + max_v * dx;
        }
    }

    /* Write the 5-vertex closed polygon */
    out_x[out_base + 0] = best_c0x;  out_y[out_base + 0] = best_c0y;
    out_x[out_base + 1] = best_c1x;  out_y[out_base + 1] = best_c1y;
    out_x[out_base + 2] = best_c2x;  out_y[out_base + 2] = best_c2y;
    out_x[out_base + 3] = best_c3x;  out_y[out_base + 3] = best_c3y;
    out_x[out_base + 4] = best_c0x;  out_y[out_base + 4] = best_c0y;  /* close ring */
}
"""
_MIN_RECT_KERNEL_NAMES = ("minimum_rotated_rectangle",)
