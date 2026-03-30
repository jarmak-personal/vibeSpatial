"""Shared CUDA device function: ring signed area (2x, shoelace formula).

COARSE/PREDICATE-class: winding sign detection only -- no Kahan, no centering.
Do NOT use for METRIC-class area computation (measurement_kernels.py uses
Kahan+centering and is structurally different).
"""

from __future__ import annotations

__all__ = ["SIGNED_AREA_DEVICE"]

SIGNED_AREA_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Ring signed area (2x) -- shoelace formula for winding detection     */
/* Shared via vibespatial.cuda.device_functions.signed_area            */
/* ------------------------------------------------------------------ */

/* Compute 2x signed area of a CLOSED ring (last vertex == first vertex).
   Positive = CCW, negative = CW.
   Ring coords at x[start..end-1], y[start..end-1].
   The ring MUST be closed (vertex at end-1 == vertex at start). */
__device__ inline double vs_ring_signed_area_2x(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start, int end
) {{
    double area2 = 0.0;
    for (int i = start; i < end - 1; i++) {{
        area2 += x[i] * y[i + 1] - x[i + 1] * y[i];
    }}
    return area2;
}}

/* Compute 2x signed area of an OPEN ring (no closure vertex).
   Positive = CCW, negative = CW.
   Ring has `count` distinct vertices at x[start..start+count-1].
   Uses modular indexing to wrap the last edge back to vertex 0. */
__device__ inline double vs_ring_signed_area_2x_open(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start, int count
) {{
    double area2 = 0.0;
    for (int i = 0; i < count; i++) {{
        int j = (i + 1 < count) ? i + 1 : 0;
        area2 += x[start + i] * y[start + j] - x[start + j] * y[start + i];
    }}
    return area2;
}}
"""
