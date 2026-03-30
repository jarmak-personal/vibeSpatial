"""NVRTC kernel sources for remove_repeated_points."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# NVRTC kernel source: mark keep/remove flags per ring
# ---------------------------------------------------------------------------
# One thread per ring/linestring.  Sequential scan within each span:
# compare each coordinate to the previous *kept* coordinate.
#
# Ring closure: the last point is always kept if it equals the first
# point (preserves polygon ring closure and linestring minimum count).

_REMOVE_REPEATED_KERNEL_SOURCE = r"""
extern "C" __global__ void remove_repeated_points_mark(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    int* __restrict__ keep_flags,
    double tolerance_sq,
    int span_count
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n = end - start;

    if (n <= 0) return;

    /* Always keep the first point */
    keep_flags[start] = 1;
    double last_kept_x = x[start];
    double last_kept_y = y[start];

    if (n == 1) return;

    /* Sequential scan: compare to previous *kept* point */
    for (int i = start + 1; i < end; i++) {{
        const double dx = x[i] - last_kept_x;
        const double dy = y[i] - last_kept_y;
        if (dx * dx + dy * dy > tolerance_sq) {{
            keep_flags[i] = 1;
            last_kept_x = x[i];
            last_kept_y = y[i];
        }} else {{
            keep_flags[i] = 0;
        }}
    }}

    /* For polygon rings: always keep the last point if it equals the
       first point (ring closure preservation). For linestrings, the
       minimum vertex count (>= 2) is enforced by the host-side fixup. */
    if (n >= 2) {{
        const double fx = x[start];
        const double fy = y[start];
        const double lx = x[end - 1];
        const double ly = y[end - 1];
        if (fx == lx && fy == ly) {{
            keep_flags[end - 1] = 1;
        }}
    }}
}}
"""
_REMOVE_REPEATED_KERNEL_NAMES = ("remove_repeated_points_mark",)
_REMOVE_REPEATED_FP64 = _REMOVE_REPEATED_KERNEL_SOURCE.format()
