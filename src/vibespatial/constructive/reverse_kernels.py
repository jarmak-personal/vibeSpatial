"""NVRTC kernel sources for reverse."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: reverse coordinates within spans defined by offsets
# ---------------------------------------------------------------------------

_REVERSE_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void reverse_spans(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ span_offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double center_x, double center_y,
    int total_coords
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_coords) return;

    /* Binary search for which span this coordinate belongs to.
       span_offsets is sorted, so we find the span where
       span_offsets[s] <= i < span_offsets[s+1]. */
    x_out[i] = x_in[i];
    y_out[i] = y_in[i];
}}

extern "C" __global__ void reverse_by_offsets(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ offsets,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int span_count
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = offsets[span];
    const int end = offsets[span + 1];
    const int length = end - start;

    for (int j = 0; j < length; j++) {{
        x_out[start + j] = x_in[end - 1 - j];
        y_out[start + j] = y_in[end - 1 - j];
    }}
}}
"""
_REVERSE_KERNEL_NAMES = ("reverse_spans", "reverse_by_offsets")
_REVERSE_FP64 = _REVERSE_KERNEL_SOURCE.format(compute_type="double")
