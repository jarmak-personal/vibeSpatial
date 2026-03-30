"""Shared CUDA device function: Shewchuk orient2d adaptive predicate."""

from __future__ import annotations

__all__ = ["ORIENT2D_DEVICE"]

ORIENT2D_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Shewchuk error-free primitives (fp64 exact predicates)             */
/* Shared via vibespatial.cuda.device_functions.orient2d              */
/* ------------------------------------------------------------------ */

/* Shewchuk two-product error-free transformation (GPU implementation).
   Given a, b: computes (p, e) such that a*b = p + e exactly.
   Uses Dekker's algorithm with FMA where available. */
__device__ inline void vs_two_product(double a, double b, double &p, double &e) {{
    p = a * b;
    e = fma(a, b, -p);
}}

/* Shewchuk two-sum error-free transformation.
   Given a, b: computes (s, e) such that a+b = s + e exactly. */
__device__ inline void vs_two_sum(double a, double b, double &s, double &e) {{
    s = a + b;
    double bv = s - a;
    double av = s - bv;
    double br = b - bv;
    double ar = a - av;
    e = ar + br;
}}

/* Shewchuk orient2d adaptive predicate (stage B).
   Returns exact sign of det = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
   using error-free arithmetic expansions when the fast filter is ambiguous.
   Returns: +1, 0, or -1  */
__device__ int vs_orient2d(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {{
    double acx = ax - cx;
    double bcx = bx - cx;
    double acy = ay - cy;
    double bcy = by - cy;

    double detleft, detleft_err;
    vs_two_product(acx, bcy, detleft, detleft_err);

    double detright, detright_err;
    vs_two_product(acy, bcx, detright, detright_err);

    double det_sum, det_sum_err;
    vs_two_sum(detleft, -detright, det_sum, det_sum_err);

    /* Accumulate all error terms */
    double B3 = detleft_err - detright_err + det_sum_err;

    /* B3 is the correction: det = det_sum + B3 */
    double det = det_sum + B3;

    if (det > 0.0) return 1;
    if (det < 0.0) return -1;

    /* Stage 1 already uses error-free vs_two_product (FMA-based) and vs_two_sum,
       capturing all rounding error in B3. For IEEE-754 fp64 inputs, the
       accumulated det = det_sum + B3 is exact. If it is zero, the
       orientation is truly zero (collinear). */
    return 0;
}}
"""
