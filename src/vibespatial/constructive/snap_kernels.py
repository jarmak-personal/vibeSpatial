"""NVRTC kernel sources for snap."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVRTC kernel source: GEOS-compatible snap
# ---------------------------------------------------------------------------
# GEOS snap algorithm: for each B vertex, find the nearest A vertex within
# tolerance and snap it to B.  This means each B vertex claims at most one
# A vertex, and each A vertex can be claimed by at most one B vertex (the
# nearest B vertex that claims it wins).
#
# Implementation:
# Phase 1 kernel: one thread per B vertex.  For each B vertex, find the
#   nearest A vertex within tolerance.  Atomically write the snap target
#   into a per-A-vertex array (the closest B vertex wins via atomicMin on
#   distance).
# Phase 2 kernel: one thread per span (ring/linestring).  Sequential scan
#   applying the snap map and producing output coordinates with keep flags.
#   Ring closure is handled here.

_SNAP_PHASE1_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
snap_find_targets(
    /* Geometry A (source) -- vertices that may be snapped */
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,

    /* Geometry B (target) -- each B vertex claims nearest A vertex */
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,
    int b_total_verts,

    /* Per-B-vertex: which A coordinate range to search */
    const int* __restrict__ a_range_start,
    const int* __restrict__ a_range_end,

    /* Output: per-A-vertex snap target.
       snap_target_idx[a_idx] = B vertex index to snap to (-1 = no snap).
       snap_target_dist[a_idx] = squared distance (for tie-breaking).
       Updated atomically: closest B vertex wins. */
    int* __restrict__ snap_target_idx,
    unsigned long long* __restrict__ snap_target_dist_bits,

    double tolerance_sq
) {{
    const int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= b_total_verts) return;

    const double bvx = b_x[bid];
    const double bvy = b_y[bid];
    const int as = a_range_start[bid];
    const int ae = a_range_end[bid];

    /* Find nearest A vertex within tolerance */
    double best_d2 = tolerance_sq + 1.0;
    int best_a = -1;
    for (int ai = as; ai < ae; ai++) {{
        const double dx = bvx - a_x[ai];
        const double dy = bvy - a_y[ai];
        const double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {{
            best_d2 = d2;
            best_a = ai;
        }}
    }}

    if (best_a < 0 || best_d2 > tolerance_sq) return;  /* No A vertex within tolerance */

    /* Atomically claim this A vertex if we are closer than any previous
       B vertex.  We use atomicMin on the distance (encoded as uint64)
       to resolve ties.  The encoding packs distance bits and B index
       into a single uint64 for atomic comparison. */
    unsigned long long dist_bits = __double_as_longlong(best_d2);
    unsigned long long old_bits = atomicMin(&snap_target_dist_bits[best_a], dist_bits);
    if (dist_bits <= old_bits) {{
        snap_target_idx[best_a] = bid;
    }}
}}


extern "C" __global__ void __launch_bounds__(256, 4)
snap_apply(
    /* Geometry A (source) */
    const double* __restrict__ a_x,
    const double* __restrict__ a_y,
    const int* __restrict__ span_offsets,
    int span_count,

    /* B coordinates for looking up snap targets */
    const double* __restrict__ b_x,
    const double* __restrict__ b_y,

    /* Per-A-vertex snap target from phase 1 */
    const int* __restrict__ snap_target_idx,

    /* Output */
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int* __restrict__ keep_flags,

    int is_polygon_ring
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n = end - start;
    if (n <= 0) return;

    /* Process first vertex: always keep */
    int tidx = snap_target_idx[start];
    double out_vx = (tidx >= 0) ? b_x[tidx] : a_x[start];
    double out_vy = (tidx >= 0) ? b_y[tidx] : a_y[start];

    out_x[start] = out_vx;
    out_y[start] = out_vy;
    keep_flags[start] = 1;

    double last_x = out_vx;
    double last_y = out_vy;

    if (n == 1) return;

    /* Sequential scan: apply snap targets, skip consecutive duplicates */
    for (int i = start + 1; i < end; i++) {{
        tidx = snap_target_idx[i];
        out_vx = (tidx >= 0) ? b_x[tidx] : a_x[i];
        out_vy = (tidx >= 0) ? b_y[tidx] : a_y[i];

        if (out_vx != last_x || out_vy != last_y) {{
            out_x[i] = out_vx;
            out_y[i] = out_vy;
            keep_flags[i] = 1;
            last_x = out_vx;
            last_y = out_vy;
        }} else {{
            out_x[i] = a_x[i];
            out_y[i] = a_y[i];
            keep_flags[i] = 0;
        }}
    }}

    /* Ring closure */
    if (is_polygon_ring && n >= 2) {{
        keep_flags[end - 1] = 1;
        out_x[end - 1] = out_x[start];
        out_y[end - 1] = out_y[start];
    }}
}}
"""
_SNAP_KERNEL_NAMES = ("snap_find_targets", "snap_apply")
_SNAP_FP64 = _SNAP_PHASE1_SOURCE.format()
