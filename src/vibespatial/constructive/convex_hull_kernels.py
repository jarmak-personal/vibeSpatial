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
 * The algorithm requires input sorted by x.  Equal-x runs may be in arbitrary
 * order because the public pipeline uses segmented sort on x only.  For each
 * equal-x run, the lower chain consumes only the minimum-y point and the upper
 * chain consumes only the maximum-y point; all other same-x points are interior
 * to the vertical slice and cannot be hull vertices.
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

    /* Fast degeneracy for vertical-only input.  X-sorted input guarantees all
       points share one x iff first and last x match. */
    if (sx[0] == sx[n - 1]) {{
        int min_idx = 0;
        int max_idx = 0;
        for (int i = 1; i < n; i++) {{
            if (sy[i] < sy[min_idx]) min_idx = i;
            if (sy[i] > sy[max_idx]) max_idx = i;
        }}
        if (sy[min_idx] == sy[max_idx]) {{
            if (out_x) {{
                out_x[0] = sx[min_idx]; out_y[0] = sy[min_idx];
                out_x[1] = sx[min_idx]; out_y[1] = sy[min_idx];
                out_x[2] = sx[min_idx]; out_y[2] = sy[min_idx];
                out_x[3] = sx[min_idx]; out_y[3] = sy[min_idx];
            }}
            return 4;
        }}
        if (out_x) {{
            out_x[0] = sx[min_idx]; out_y[0] = sy[min_idx];
            out_x[1] = sx[max_idx]; out_y[1] = sy[max_idx];
            out_x[2] = sx[max_idx]; out_y[2] = sy[max_idx];
            out_x[3] = sx[min_idx]; out_y[3] = sy[min_idx];
            out_x[4] = sx[min_idx]; out_y[4] = sy[min_idx];
        }}
        return 5;
    }}

    /* Lower hull */
    int lower[MAX_HULL_VERTS];  /* indices into sorted coords */
    int lower_size = 0;
    for (int run_start = 0; run_start < n; ) {{
        int run_end = run_start + 1;
        int i = run_start;
        while (run_end < n && sx[run_end] == sx[run_start]) {{
            if (sy[run_end] < sy[i]) i = run_end;
            run_end++;
        }}
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
        run_start = run_end;
    }}

    /* Upper hull */
    int upper[MAX_HULL_VERTS];
    int upper_size = 0;
    for (int run_end = n; run_end > 0; ) {{
        int run_start = run_end - 1;
        int i = run_start;
        while (run_start > 0 && sx[run_start - 1] == sx[run_end - 1]) {{
            run_start--;
            if (sy[run_start] > sy[i]) i = run_start;
        }}
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
        run_end = run_start;
    }}

    /* Total hull = lower + upper with only true duplicate endpoints removed.
       With equal-x run collapsing, lower may end at (max_x, min_y) while upper
       starts at (max_x, max_y); those are both real hull vertices on a vertical
       edge and must not be dropped just because they share x. */
    int skip_upper_first = 0;
    int skip_upper_last = 0;
    if (lower_size > 0 && upper_size > 0) {{
        int lower_last = lower[lower_size - 1];
        int upper_first = upper[0];
        skip_upper_first = (
            sx[lower_last] == sx[upper_first] &&
            sy[lower_last] == sy[upper_first]
        );
        int lower_first = lower[0];
        int upper_last = upper[upper_size - 1];
        skip_upper_last = (
            sx[lower_first] == sx[upper_last] &&
            sy[lower_first] == sy[upper_last]
        );
    }}
    int hull_count = lower_size + upper_size - skip_upper_first - skip_upper_last + 1;

    if (out_x) {{
        int pos = 0;
        for (int i = 0; i < lower_size; i++) {{
            out_x[pos] = sx[lower[i]];
            out_y[pos] = sy[lower[i]];
            pos++;
        }}
        int upper_stop = upper_size - skip_upper_last;
        for (int i = skip_upper_first; i < upper_stop; i++) {{
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
