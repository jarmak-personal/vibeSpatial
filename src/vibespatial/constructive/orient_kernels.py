"""NVRTC kernel sources for orient."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: orient rings by shoelace signed area
# ---------------------------------------------------------------------------

_ORIENT_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void orient_rings(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ is_exterior,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    int exterior_ccw,   /* 1 = exterior CCW (default), 0 = exterior CW */
    int ring_count
) {{
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;

    const int start = ring_offsets[ring];
    const int end = ring_offsets[ring + 1];
    const int length = end - start;

    /* Degenerate ring: fewer than 3 coords, just copy through. */
    if (length < 3) {{
        for (int j = 0; j < length; j++) {{
            x_out[start + j] = x_in[start + j];
            y_out[start + j] = y_in[start + j];
        }}
        return;
    }}

    /* Shoelace signed area (2x): positive = CCW, negative = CW. */
    double area2 = 0.0;
    for (int j = start; j < end - 1; j++) {{
        area2 += x_in[j] * y_in[j + 1] - x_in[j + 1] * y_in[j];
    }}

    /* Determine desired orientation:
       - Exterior rings: CCW when exterior_ccw=1, CW when exterior_ccw=0
       - Interior rings (holes): opposite of exterior
       want_positive_area = true means we want CCW (area2 > 0). */
    const int ext = is_exterior[ring];
    int want_positive;
    if (ext) {{
        want_positive = exterior_ccw;       /* exterior: CCW if exterior_ccw=1 */
    }} else {{
        want_positive = 1 - exterior_ccw;   /* interior: opposite of exterior */
    }}

    /* Check if current orientation matches desired; reverse if not. */
    int need_reverse = 0;
    if (want_positive && area2 < 0.0) {{
        need_reverse = 1;  /* want CCW but ring is CW */
    }} else if (!want_positive && area2 > 0.0) {{
        need_reverse = 1;  /* want CW but ring is CCW */
    }}

    if (need_reverse) {{
        for (int j = 0; j < length; j++) {{
            x_out[start + j] = x_in[end - 1 - j];
            y_out[start + j] = y_in[end - 1 - j];
        }}
    }} else {{
        for (int j = 0; j < length; j++) {{
            x_out[start + j] = x_in[start + j];
            y_out[start + j] = y_in[start + j];
        }}
    }}
}}
"""
_ORIENT_KERNEL_NAMES = ("orient_rings",)
_ORIENT_FP64 = _ORIENT_KERNEL_SOURCE.format(compute_type="double")
