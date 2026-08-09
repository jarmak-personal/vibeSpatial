"""NVRTC kernel sources for coordinate-capacity segmentize."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

_SEGMENTIZE_COUNT_KERNEL_SOURCE = (
    PRECISION_PREAMBLE
    + r"""
extern "C" __global__ void segmentize_count(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    long long* __restrict__ out_counts,
    unsigned char* __restrict__ out_terminal,
    double max_seg_len,
    int coord_capacity,
    int span_count
) {{
    /* One lane per physical input coordinate, not one serial lane per span. */
    const int coord = blockIdx.x * blockDim.x + threadIdx.x;
    if (coord >= coord_capacity) return;

    const int logical_coord_count = span_offsets[span_count];
    if (coord >= logical_coord_count) {{
        out_counts[coord] = 0;
        out_terminal[coord] = 0;
        return;
    }}

    int lo = 0;
    int hi = span_count;
    while (lo < hi) {{
        const int mid = (lo + hi) >> 1;
        if (span_offsets[mid + 1] <= coord) lo = mid + 1;
        else hi = mid;
    }}
    const int span = lo;
    const int terminal = coord == span_offsets[span + 1] - 1;
    out_terminal[coord] = (unsigned char)terminal;
    if (terminal) {{
        out_counts[coord] = 1;
        return;
    }}

    const double dx = x[coord + 1] - x[coord];
    const double dy = y[coord + 1] - y[coord];
    const double raw_splits = ceil(sqrt(dx * dx + dy * dy) / max_seg_len);
    long long splits = raw_splits > 2147483647.0
        ? 2147483647LL
        : (long long)raw_splits;
    if (splits < 1) splits = 1;
    out_counts[coord] = splits;
}}
"""
)

_SEGMENTIZE_SCATTER_KERNEL_SOURCE = (
    PRECISION_PREAMBLE
    + r"""
extern "C" __global__ void segmentize_scatter(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const long long* __restrict__ coord_offsets,
    const long long* __restrict__ coord_counts,
    const unsigned char* __restrict__ terminal,
    double* __restrict__ ox,
    double* __restrict__ oy,
    int coord_capacity,
    long long total_out
) {{
    const long long tid =
        (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    int lo = 0;
    int hi = coord_capacity;
    while (lo < hi) {{
        const int mid = (lo + hi) >> 1;
        if (coord_offsets[mid + 1] <= tid) lo = mid + 1;
        else hi = mid;
    }}
    const int coord = lo;
    if (terminal[coord]) {{
        ox[tid] = x[coord];
        oy[tid] = y[coord];
        return;
    }}

    const long long local = tid - coord_offsets[coord];
    const long long splits = coord_counts[coord];
    const double t = (double)local / (double)splits;
    ox[tid] = x[coord] + t * (x[coord + 1] - x[coord]);
    oy[tid] = y[coord] + t * (y[coord + 1] - y[coord]);
}}
"""
)

_SEGMENTIZE_COUNT_KERNEL_NAMES = ("segmentize_count",)
_SEGMENTIZE_SCATTER_KERNEL_NAMES = ("segmentize_scatter",)
_SEGMENTIZE_COUNT_FP64 = _SEGMENTIZE_COUNT_KERNEL_SOURCE.format(compute_type="double")
_SEGMENTIZE_SCATTER_FP64 = _SEGMENTIZE_SCATTER_KERNEL_SOURCE.format(compute_type="double")
