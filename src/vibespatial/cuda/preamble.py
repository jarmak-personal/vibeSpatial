"""Shared CUDA C++ source preambles for NVRTC compilation.

Canonical location for precision-aware macros (compute_t typedef,
centered coordinate reads, Kahan summation) used across all Tier 1
NVRTC kernels.
"""

from __future__ import annotations

from vibespatial.runtime.config import SPATIAL_EPSILON

__all__ = ["PRECISION_PREAMBLE", "SPATIAL_TOLERANCE_PREAMBLE"]

SPATIAL_TOLERANCE_PREAMBLE = f"#define VS_SPATIAL_EPSILON {SPATIAL_EPSILON:.17g}\n"

PRECISION_PREAMBLE = SPATIAL_TOLERANCE_PREAMBLE + r"""
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

/* Warp-level Kahan reduction for a (sum, compensation) pair. */
#define VS_WARP_FULL_MASK 0xFFFFFFFFu
#define WARP_KAHAN_REDUCE(sum, c) do {{ \
    for (int _vs_offset = 16; _vs_offset > 0; _vs_offset >>= 1) {{ \
        const compute_t _vs_shfl_sum = __shfl_down_sync(VS_WARP_FULL_MASK, (sum), _vs_offset); \
        const compute_t _vs_shfl_c = __shfl_down_sync(VS_WARP_FULL_MASK, (c), _vs_offset); \
        KAHAN_ADD((sum), _vs_shfl_sum - _vs_shfl_c, (c)); \
    }} \
}} while(0)
"""
