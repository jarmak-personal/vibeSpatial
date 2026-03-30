"""NVRTC kernel sources for GPU pattern matching and span detection."""

from __future__ import annotations

_PATTERN_MATCH_NAMES = ("pattern_match_kernel",)

_SPAN_BOUNDARIES_SOURCE = r"""
extern "C" __global__ void span_boundaries_kernel(
    const int* __restrict__ depth,
    const long long* __restrict__ starts,
    long long* __restrict__ ends,
    int n_spans,
    long long n_bytes,
    int skip_bytes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spans) return;

    // Start scanning after skip_bytes past the start position
    long long pos = starts[idx] + (long long)skip_bytes;
    // Skip whitespace: advance while depth does not change
    while (pos < n_bytes && depth[pos] == depth[pos - 1]) {
        pos++;
    }
    if (pos >= n_bytes) {
        ends[idx] = n_bytes;
        return;
    }
    int start_depth = depth[pos];
    // Scan forward until depth drops below start_depth
    pos++;
    while (pos < n_bytes && depth[pos] >= start_depth) {
        pos++;
    }
    ends[idx] = pos;
}
"""

_SPAN_BOUNDARIES_NAMES = ("span_boundaries_kernel",)

_MARK_SPANS_SOURCE = r"""
extern "C" __global__ void mark_spans_kernel(
    const long long* __restrict__ starts,
    const long long* __restrict__ ends,
    unsigned char* __restrict__ mask,
    int n_spans
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_spans) return;

    long long start = starts[idx];
    long long end = ends[idx];
    for (long long i = start; i < end; i++) {
        mask[i] = 1;
    }
}
"""

_MARK_SPANS_NAMES = ("mark_spans_kernel",)
