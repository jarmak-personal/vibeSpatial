"""NVRTC kernel sources for envelope."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: compute bounds per geometry and emit envelope polygon
# ---------------------------------------------------------------------------

_ENVELOPE_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void envelope_from_bounds(
    const double* __restrict__ bounds,  /* (N, 4): xmin, ymin, xmax, ymax */
    double* __restrict__ x_out,         /* 5*N coords */
    double* __restrict__ y_out,
    double center_x, double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const double xmin = bounds[row * 4 + 0];
    const double ymin = bounds[row * 4 + 1];
    const double xmax = bounds[row * 4 + 2];
    const double ymax = bounds[row * 4 + 3];

    const int base = row * 5;
    x_out[base + 0] = xmin;  y_out[base + 0] = ymin;
    x_out[base + 1] = xmax;  y_out[base + 1] = ymin;
    x_out[base + 2] = xmax;  y_out[base + 2] = ymax;
    x_out[base + 3] = xmin;  y_out[base + 3] = ymax;
    x_out[base + 4] = xmin;  y_out[base + 4] = ymin;  /* close ring */
}}
"""
_ENVELOPE_KERNEL_NAMES = ("envelope_from_bounds",)
_ENVELOPE_FP64 = _ENVELOPE_KERNEL_SOURCE.format(compute_type="double")
_ENVELOPE_FP32 = _ENVELOPE_KERNEL_SOURCE.format(compute_type="float")
