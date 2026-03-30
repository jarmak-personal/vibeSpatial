"""NVRTC kernel sources for segmentize."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel source: two-pass segmentize
#
# Pass 1 (segmentize_count): 1 thread per input segment.  Computes the
#   number of output points contributed by each segment:
#   splits = ceil(seg_len / max_seg_len), clamped to >= 1.
#   Output count for segment i = splits (last segment of span adds +1
#   for the final endpoint).
#
# Pass 2 (segmentize_scatter): 1 thread per output coordinate.  Each
#   thread determines which input segment it belongs to (via binary
#   search on a prefix-sum of split counts), then interpolates the
#   coordinate within that segment.
#
# Both kernels operate on a flat span-offsets array that indexes into
# the coordinate arrays (geometry_offsets for LineStrings, ring_offsets
# for Polygons, part_offsets for MultiLineStrings).
# ---------------------------------------------------------------------------

_SEGMENTIZE_COUNT_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void segmentize_count(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    int* __restrict__ out_counts,
    double max_seg_len,
    int span_count
) {{
    /* 1 thread per span. Counts total output coordinates for each span. */
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n = end - start;

    if (n <= 1) {{
        out_counts[span] = n;
        return;
    }}

    int total = 1;  /* first point always emitted */
    for (int i = start; i < end - 1; i++) {{
        double dx = x[i + 1] - x[i];
        double dy = y[i + 1] - y[i];
        double seg_len = sqrt(dx * dx + dy * dy);
        int splits = (int)ceil(seg_len / max_seg_len);
        if (splits < 1) splits = 1;
        total += splits;
    }}
    out_counts[span] = total;
}}
"""
_SEGMENTIZE_SCATTER_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void segmentize_scatter(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    const int* __restrict__ out_offsets,
    double* __restrict__ ox,
    double* __restrict__ oy,
    double max_seg_len,
    int span_count
) {{
    /* 1 thread per output coordinate (across all spans). */
    const int total_out = out_offsets[span_count];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_out) return;

    /* Binary search: find which span this output coord belongs to. */
    int lo = 0, hi = span_count;
    while (lo < hi) {{
        int mid = (lo + hi) >> 1;
        if (out_offsets[mid + 1] <= tid) lo = mid + 1;
        else hi = mid;
    }}
    const int span = lo;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n = end - start;
    const int local = tid - out_offsets[span];

    if (n <= 1) {{
        /* Degenerate span: single point or empty. */
        if (local == 0 && n == 1) {{
            ox[tid] = x[start];
            oy[tid] = y[start];
        }}
        return;
    }}

    /* Walk segments to find which input segment this output coord falls in. */
    int seg_out_start = 0;
    for (int i = start; i < end - 1; i++) {{
        double dx = x[i + 1] - x[i];
        double dy = y[i + 1] - y[i];
        double seg_len = sqrt(dx * dx + dy * dy);
        int splits = (int)ceil(seg_len / max_seg_len);
        if (splits < 1) splits = 1;

        if (local < seg_out_start + splits) {{
            /* This output point falls in segment [i, i+1]. */
            int k = local - seg_out_start;
            double t = (double)k / (double)splits;
            ox[tid] = x[i] + t * dx;
            oy[tid] = y[i] + t * dy;
            return;
        }}
        seg_out_start += splits;
    }}

    /* Last point in span = endpoint of final segment. */
    ox[tid] = x[end - 1];
    oy[tid] = y[end - 1];
}}
"""
_SEGMENTIZE_COUNT_KERNEL_NAMES = ("segmentize_count",)
_SEGMENTIZE_SCATTER_KERNEL_NAMES = ("segmentize_scatter",)
_SEGMENTIZE_COUNT_FP64 = _SEGMENTIZE_COUNT_KERNEL_SOURCE.format(compute_type="double")
_SEGMENTIZE_SCATTER_FP64 = _SEGMENTIZE_SCATTER_KERNEL_SOURCE.format(compute_type="double")
