"""Shared CUDA device function: strip closing ring vertex when present."""

from __future__ import annotations

__all__ = ["STRIP_CLOSURE_DEVICE"]

STRIP_CLOSURE_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Strip trailing ring closure vertex when the last coord repeats the  */
/* first coord within a caller-provided squared tolerance.             */
/* Shared via vibespatial.cuda.device_functions.strip_closure          */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ int vs_strip_closure(
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int end,
    int count,
    double tol_sq
) {{
    if (count >= 2) {{
        double dx = x[start] - x[end - 1];
        double dy = y[start] - y[end - 1];
        if (dx * dx + dy * dy < tol_sq) count--;
    }}
    return count;
}}
"""
