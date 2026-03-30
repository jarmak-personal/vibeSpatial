"""NVRTC kernel sources for simplify."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: compute effective area per vertex (Visvalingam-Whyatt)
# ---------------------------------------------------------------------------

_VW_AREA_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void vw_effective_area(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ offsets,
    double* __restrict__ areas,
    double center_x, double center_y,
    int span_count
) {{
    const int span = blockIdx.x * blockDim.x + threadIdx.x;
    if (span >= span_count) return;

    const int start = offsets[span];
    const int end = offsets[span + 1];
    const int n = end - start;

    /* First and last vertices always kept (infinite area) */
    if (n <= 2) {{
        for (int i = start; i < end; i++) {{
            areas[i] = 1e308;  /* effectively infinite */
        }}
        return;
    }}

    areas[start] = 1e308;
    areas[end - 1] = 1e308;

    for (int i = start + 1; i < end - 1; i++) {{
        /* Triangle area = 0.5 * |cross product of (p[i-1]->p[i]) x (p[i-1]->p[i+1])| */
        double ax = x[i] - x[i - 1];
        double ay = y[i] - y[i - 1];
        double bx = x[i + 1] - x[i - 1];
        double by = y[i + 1] - y[i - 1];
        double cross = ax * by - ay * bx;
        areas[i] = 0.5 * (cross < 0 ? -cross : cross);
    }}
}}
"""
_VW_AREA_KERNEL_NAMES = ("vw_effective_area",)
_VW_AREA_FP64 = _VW_AREA_KERNEL_SOURCE.format(compute_type="double")
