"""Shared CUDA C++ source preambles for NVRTC compilation.

Canonical location for precision-aware macros (compute_t typedef,
centered coordinate reads, Kahan summation) used across all Tier 1
NVRTC kernels.
"""

from __future__ import annotations

__all__ = ["PRECISION_PREAMBLE"]

PRECISION_PREAMBLE = r"""
typedef {compute_type} compute_t;

/* Centered coordinate read: subtract center in fp64, then cast to compute_t.
   When compute_t is double, this is a no-op identity.  When compute_t is float,
   centering reduces absolute magnitude before the lossy cast. */
#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

/* Kahan summation helper -- add `val` to `sum` with compensation `c`. */
#define KAHAN_ADD(sum, val, c) do {{ \
    const compute_t _y = (val) - (c); \
    const compute_t _t = (sum) + _y; \
    (c) = (_t - (sum)) - _y; \
    (sum) = _t; \
}} while(0)
"""
