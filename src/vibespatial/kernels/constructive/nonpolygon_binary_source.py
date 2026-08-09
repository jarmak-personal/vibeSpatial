"""CUDA kernel source for non-polygon binary constructive operations.

Contains NVRTC kernel source strings for:
- Point-LineString: point-on-segment test

Extracted from nonpolygon_binary.py -- dispatch logic remains there.
"""

from __future__ import annotations

_POINT_LINESTRING_KERNEL_SOURCE = r"""
extern "C" __global__ __launch_bounds__(256, 4) void point_linestring_on_line(
    const double* __restrict__ pt_x,
    const double* __restrict__ pt_y,
    const int* __restrict__ pt_geom_offsets,
    const double* __restrict__ ls_x,
    const double* __restrict__ ls_y,
    const int* __restrict__ ls_geom_offsets,
    const int* __restrict__ valid_mask,
    int* __restrict__ out_on_line,
    const int n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (!valid_mask[idx]) {
        out_on_line[idx] = 0;
        return;
    }

    /* Point coordinates */
    const int pt_start = pt_geom_offsets[idx];
    const int pt_end = pt_geom_offsets[idx + 1];
    if (pt_end <= pt_start) {
        out_on_line[idx] = 0;
        return;
    }
    const double px = pt_x[pt_start];
    const double py = pt_y[pt_start];

    /* LineString coordinates */
    const int ls_start = ls_geom_offsets[idx];
    const int ls_end = ls_geom_offsets[idx + 1];
    const int seg_count = ls_end - ls_start - 1;
    if (seg_count < 1) {
        out_on_line[idx] = 0;
        return;
    }

    /* Check each segment for point-on-segment */
    const double tol = 1e-8;

    for (int s = 0; s < seg_count; s++) {
        const double ax = ls_x[ls_start + s];
        const double ay = ls_y[ls_start + s];
        const double bx = ls_x[ls_start + s + 1];
        const double by = ls_y[ls_start + s + 1];

        /* Cross product to check collinearity */
        const double dx_ab = bx - ax;
        const double dy_ab = by - ay;
        const double cross = (px - ax) * dy_ab - (py - ay) * dx_ab;
        const double seg_len_sq = dx_ab * dx_ab + dy_ab * dy_ab;

        if (seg_len_sq < 1e-30) {
            /* Degenerate segment (zero length): check if point == segment point */
            if ((px - ax) * (px - ax) + (py - ay) * (py - ay) < tol * tol) {
                out_on_line[idx] = 1;
                return;
            }
            continue;
        }

        /* Relative cross product magnitude */
        if (cross * cross > tol * tol * seg_len_sq) {
            continue;  /* not collinear */
        }

        /* Project point onto segment to check if within bounds */
        const double t = ((px - ax) * dx_ab + (py - ay) * dy_ab) / seg_len_sq;
        if (t >= -tol && t <= 1.0 + tol) {
            out_on_line[idx] = 1;
            return;
        }
    }

    out_on_line[idx] = 0;
}
"""

_POINT_LINESTRING_KERNEL_NAMES = ("point_linestring_on_line",)
