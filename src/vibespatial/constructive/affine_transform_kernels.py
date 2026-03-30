"""NVRTC kernel sources for affine_transform."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel source
# ---------------------------------------------------------------------------

_AFFINE_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void affine_transform_coords(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    double a, double b, double xoff,
    double d, double e, double yoff,
    double center_x, double center_y,
    int coord_count
) {{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= coord_count) return;

    const double xi = x_in[i];
    const double yi = y_in[i];
    x_out[i] = a * xi + b * yi + xoff;
    y_out[i] = d * xi + e * yi + yoff;
}}
"""
_AFFINE_KERNEL_NAMES = ("affine_transform_coords",)
_AFFINE_FP64 = _AFFINE_KERNEL_SOURCE.format(compute_type="double")
_AFFINE_FP32 = _AFFINE_KERNEL_SOURCE.format(compute_type="float")
