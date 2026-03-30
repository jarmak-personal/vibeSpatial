"""NVRTC kernel sources for minimum_bounding_circle."""

from __future__ import annotations

_N_TESSELLATION_VERTS = 128  # number of vertices on the circle (before closure)
# ---------------------------------------------------------------------------
# NVRTC kernel source: Ritter's bounding circle (1 thread per geometry)
#
# ADR-0033 Tier 1: geometry-specific inner loop (Ritter's iteration).
# ADR-0002: CONSTRUCTIVE/METRIC class -- fp64 only.
# ---------------------------------------------------------------------------

_RITTER_KERNEL_SOURCE = r"""
/* Ritter's 2D bounding circle: one thread per geometry.
 *
 * Inputs:
 *   x, y:            SoA coordinate arrays (all geometries concatenated)
 *   coord_offsets:    per-geometry coordinate range [row] .. [row+1]
 *   out_cx, out_cy:   output circle center
 *   out_radius:       output circle radius
 *   n_geometries:     number of geometries
 *
 * Degenerate cases:
 *   0 coords -> center = (0,0), radius = 0
 *   1 coord  -> center = that point, radius = 0
 *   2 coords -> center = midpoint, radius = half-distance
 */

extern "C" __global__ void ritter_bounding_circle(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ coord_offsets,
    double* __restrict__ out_cx,
    double* __restrict__ out_cy,
    double* __restrict__ out_radius,
    int n_geometries
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_geometries) return;

    const int cs = coord_offsets[row];
    const int ce = coord_offsets[row + 1];
    const int n = ce - cs;

    /* --- Degenerate: 0 coordinates --- */
    if (n == 0) {
        out_cx[row] = 0.0;
        out_cy[row] = 0.0;
        out_radius[row] = 0.0;
        return;
    }

    /* --- Degenerate: 1 coordinate --- */
    if (n == 1) {
        out_cx[row] = x[cs];
        out_cy[row] = y[cs];
        out_radius[row] = 0.0;
        return;
    }

    /* --- Step 1: Compute centroid --- */
    double cx_acc = 0.0;
    double cy_acc = 0.0;
    for (int i = cs; i < ce; i++) {
        cx_acc += x[i];
        cy_acc += y[i];
    }
    double mean_x = cx_acc / (double)n;
    double mean_y = cy_acc / (double)n;

    /* --- Step 2: Find point P furthest from centroid --- */
    int p_idx = cs;
    double max_dist_sq = 0.0;
    for (int i = cs; i < ce; i++) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        double d2 = dx * dx + dy * dy;
        if (d2 > max_dist_sq) {
            max_dist_sq = d2;
            p_idx = i;
        }
    }

    /* --- Step 3: Find point Q furthest from P --- */
    double px = x[p_idx];
    double py = y[p_idx];
    int q_idx = cs;
    max_dist_sq = 0.0;
    for (int i = cs; i < ce; i++) {
        double dx = x[i] - px;
        double dy = y[i] - py;
        double d2 = dx * dx + dy * dy;
        if (d2 > max_dist_sq) {
            max_dist_sq = d2;
            q_idx = i;
        }
    }
    double qx = x[q_idx];
    double qy = y[q_idx];

    /* --- Step 4: Initial circle from P-Q diameter --- */
    double ccx = (px + qx) * 0.5;
    double ccy = (py + qy) * 0.5;
    double dx_pq = qx - px;
    double dy_pq = qy - py;
    double radius = sqrt(dx_pq * dx_pq + dy_pq * dy_pq) * 0.5;

    /* --- Step 5-6: Expand circle (2 passes) --- */
    for (int pass = 0; pass < 2; pass++) {
        for (int i = cs; i < ce; i++) {
            double dx = x[i] - ccx;
            double dy = y[i] - ccy;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist > radius) {
                /* Expand: new diameter includes old circle + outside point */
                double new_radius = (radius + dist) * 0.5;
                double shift = new_radius - radius;
                /* Move center toward the outside point */
                double inv_dist = 1.0 / dist;
                ccx += dx * inv_dist * shift;
                ccy += dy * inv_dist * shift;
                radius = new_radius;
            }
        }
    }

    out_cx[row] = ccx;
    out_cy[row] = ccy;
    out_radius[row] = radius;
}
"""
_RITTER_KERNEL_NAMES = ("ritter_bounding_circle",)
_TESSELLATE_KERNEL_SOURCE = (
    "\n#define N_VERTS " + str(_N_TESSELLATION_VERTS) + "\n"
) + r"""
/* Tessellate circles to closed polygon rings.
 *
 * Each thread handles one geometry.  Output ring has N_VERTS+1 coordinates
 * (first == last for closure).
 *
 * For radius == 0 (degenerate point), all N_VERTS+1 coordinates are the
 * center point, producing a valid degenerate closed ring.
 */

extern "C" __global__ void tessellate_circle(
    const double* __restrict__ cx,
    const double* __restrict__ cy,
    const double* __restrict__ radius,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int n_geometries
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_geometries) return;

    const double c_x = cx[row];
    const double c_y = cy[row];
    const double r = radius[row];

    const int base = row * (N_VERTS + 1);

    /* 2*pi / N_VERTS */
    const double angle_step = 6.283185307179586 / (double)N_VERTS;

    for (int i = 0; i < N_VERTS; i++) {
        double angle = (double)i * angle_step;
        double s, c;
        sincos(angle, &s, &c);
        out_x[base + i] = c_x + r * c;
        out_y[base + i] = c_y + r * s;
    }

    /* Closure: first == last */
    out_x[base + N_VERTS] = out_x[base];
    out_y[base + N_VERTS] = out_y[base];
}
"""
_TESSELLATE_KERNEL_NAMES = ("tessellate_circle",)
