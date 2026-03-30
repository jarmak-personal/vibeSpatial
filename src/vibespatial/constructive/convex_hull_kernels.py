"""NVRTC kernel sources for convex_hull."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel source: Andrew's monotone chain convex hull (2-pass)
#
# Both count and write kernels use the same monotone_chain device function.
# Each thread processes one geometry.  Local-memory stacks (spill to L1/L2)
# handle up to MAX_HULL_VERTS coordinates per geometry.
#
# ADR-0033 Tier 1: geometry-specific inner loop (monotone chain traversal).
# ---------------------------------------------------------------------------

_MAX_HULL_VERTS = 2048
_CONVEX_HULL_KERNEL_SOURCE = PRECISION_PREAMBLE + (
    "\n#define MAX_HULL_VERTS " + str(_MAX_HULL_VERTS) + "\n"
) + r"""
/* ---------- shared device helper: monotone chain ----------
 *
 * Runs Andrew's monotone chain on x-sorted coordinates sx[0..n-1], sy[0..n-1].
 * If out_x/out_y are non-NULL, writes hull coordinates there.
 * Returns the number of hull vertices (including the closing vertex).
 *
 * The algorithm requires input sorted by x (ties broken arbitrarily).
 * The cross-product test uses <= 0 to exclude collinear and right-turning
 * points, which is correct even without strict (x,y) lexicographic sort.
 */
__device__ int monotone_chain(
    const double* __restrict__ sx,
    const double* __restrict__ sy,
    int n,
    double* out_x,
    double* out_y
) {{
    /* Degenerate: 0 coordinates -> 4-vertex degenerate polygon at origin. */
    if (n == 0) {{
        if (out_x) {{
            out_x[0] = 0.0; out_y[0] = 0.0;
            out_x[1] = 0.0; out_y[1] = 0.0;
            out_x[2] = 0.0; out_y[2] = 0.0;
            out_x[3] = 0.0; out_y[3] = 0.0;
        }}
        return 4;
    }}

    /* Degenerate: 1 coordinate -> 4-vertex degenerate polygon. */
    if (n == 1) {{
        if (out_x) {{
            out_x[0] = sx[0]; out_y[0] = sy[0];
            out_x[1] = sx[0]; out_y[1] = sy[0];
            out_x[2] = sx[0]; out_y[2] = sy[0];
            out_x[3] = sx[0]; out_y[3] = sy[0];
        }}
        return 4;
    }}

    /* Degenerate: 2 coordinates -> 5-vertex degenerate polygon. */
    if (n == 2) {{
        if (out_x) {{
            out_x[0] = sx[0]; out_y[0] = sy[0];
            out_x[1] = sx[1]; out_y[1] = sy[1];
            out_x[2] = sx[1]; out_y[2] = sy[1];
            out_x[3] = sx[0]; out_y[3] = sy[0];
            out_x[4] = sx[0]; out_y[4] = sy[0];
        }}
        return 5;
    }}

    /* Clamp to stack capacity.  For n > MAX_HULL_VERTS the kernel cannot
       run the full chain; emit all coords + closure as a safe fallback. */
    if (n > MAX_HULL_VERTS) {{
        if (out_x) {{
            for (int i = 0; i < n; i++) {{
                out_x[i] = sx[i];
                out_y[i] = sy[i];
            }}
            out_x[n] = sx[0];
            out_y[n] = sy[0];
        }}
        return n + 1;
    }}

    /* Lower hull */
    int lower[MAX_HULL_VERTS];  /* indices into sorted coords */
    int lower_size = 0;
    for (int i = 0; i < n; i++) {{
        while (lower_size >= 2) {{
            int a = lower[lower_size - 2];
            int b = lower[lower_size - 1];
            double cross = (sx[b] - sx[a]) * (sy[i] - sy[a])
                         - (sy[b] - sy[a]) * (sx[i] - sx[a]);
            if (cross <= 0.0)
                lower_size--;
            else
                break;
        }}
        lower[lower_size++] = i;
    }}

    /* Upper hull */
    int upper[MAX_HULL_VERTS];
    int upper_size = 0;
    for (int i = n - 1; i >= 0; i--) {{
        while (upper_size >= 2) {{
            int a = upper[upper_size - 2];
            int b = upper[upper_size - 1];
            double cross = (sx[b] - sx[a]) * (sy[i] - sy[a])
                         - (sy[b] - sy[a]) * (sx[i] - sx[a]);
            if (cross <= 0.0)
                upper_size--;
            else
                break;
        }}
        upper[upper_size++] = i;
    }}

    /* Total hull = lower[:-1] + upper[:-1] + closure */
    int hull_count = (lower_size - 1) + (upper_size - 1) + 1;

    if (out_x) {{
        int pos = 0;
        for (int i = 0; i < lower_size - 1; i++) {{
            out_x[pos] = sx[lower[i]];
            out_y[pos] = sy[lower[i]];
            pos++;
        }}
        for (int i = 0; i < upper_size - 1; i++) {{
            out_x[pos] = sx[upper[i]];
            out_y[pos] = sy[upper[i]];
            pos++;
        }}
        /* Closure: repeat first vertex */
        out_x[pos] = out_x[0];
        out_y[pos] = out_y[0];
    }}

    return hull_count;
}}


/* ---------- Pass 1: count hull vertices per geometry ---------- */

extern "C" __global__ void convex_hull_count(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ coord_offsets,
    int* __restrict__ hull_counts,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = coord_offsets[row];
    const int ce = coord_offsets[row + 1];
    const int n = ce - cs;

    hull_counts[row] = monotone_chain(x + cs, y + cs, n, (double*)0, (double*)0);
}}


/* ---------- Pass 2: write hull vertices at pre-computed offsets ---------- */

extern "C" __global__ void convex_hull_write(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ coord_offsets,
    const int* __restrict__ hull_offsets,
    double* __restrict__ ox,
    double* __restrict__ oy,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = coord_offsets[row];
    const int ce = coord_offsets[row + 1];
    const int n = ce - cs;

    const int out_start = hull_offsets[row];
    monotone_chain(x + cs, y + cs, n, ox + out_start, oy + out_start);
}}
"""
_CONVEX_HULL_KERNEL_NAMES = ("convex_hull_count", "convex_hull_write")
_CONVEX_HULL_FP64 = _CONVEX_HULL_KERNEL_SOURCE.format(compute_type="double")
