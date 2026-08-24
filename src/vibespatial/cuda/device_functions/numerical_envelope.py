"""Shared CUDA helpers for conservative fp32 decision envelopes.

The helpers in this module never make an approximate topology decision.  They
return an authoritative sign only when a conservative fp32 error envelope
excludes zero; ambiguous determinants are refined by the caller with the
existing adaptive fp64 predicate.
"""

from __future__ import annotations

__all__ = ["ORIENT2D_FP32_ENVELOPE_DEVICE"]


ORIENT2D_FP32_ENVELOPE_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Conservative centered-fp32 orient2d decision envelope              */
/* Requires: vs_orient2d (from ORIENT2D_DEVICE)                       */
/* ------------------------------------------------------------------ */

#define VS_NUMERICAL_SIGN_AMBIGUOUS 2

/*
 * Filter the sign of the determinant evaluated from fp64 coordinate
 * differences cast to fp32.  Subtracting before the cast centers every
 * orientation locally and avoids losing small deltas at large CRS offsets.
 *
 * The bound covers four operand roundings, two products, and the final
 * subtraction.  Eight fp32 epsilons is deliberately conservative.  CUDA may
 * flush fp32 subnormals to zero, so a nonzero fp64 difference or product that
 * is not representable as a normal fp32 value is never certified: it returns
 * AMBIGUOUS and must be refined with vs_orient2d.
 */
__device__ inline int vs_orient2d_fp32_envelope(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    float* absolute_error
) {
    const double acx64 = ax - cx;
    const double bcx64 = bx - cx;
    const double acy64 = ay - cy;
    const double bcy64 = by - cy;
    const float fp32_min_normal = 1.1754943508222875e-38f;
    if (
        (acx64 != 0.0 && fabs(acx64) < (double)fp32_min_normal) ||
        (bcx64 != 0.0 && fabs(bcx64) < (double)fp32_min_normal) ||
        (acy64 != 0.0 && fabs(acy64) < (double)fp32_min_normal) ||
        (bcy64 != 0.0 && fabs(bcy64) < (double)fp32_min_normal)
    ) {
        *absolute_error = 3.402823466e+38f;
        return VS_NUMERICAL_SIGN_AMBIGUOUS;
    }

    const float acx = (float)acx64;
    const float bcx = (float)bcx64;
    const float acy = (float)acy64;
    const float bcy = (float)bcy64;
    if (!isfinite(acx) || !isfinite(bcx) || !isfinite(acy) || !isfinite(bcy)) {
        *absolute_error = 3.402823466e+38f;
        return VS_NUMERICAL_SIGN_AMBIGUOUS;
    }

    const float left = acx * bcy;
    const float right = acy * bcx;
    if (
        (acx64 != 0.0 && bcy64 != 0.0 &&
         (left == 0.0f || fabsf(left) < fp32_min_normal)) ||
        (acy64 != 0.0 && bcx64 != 0.0 &&
         (right == 0.0f || fabsf(right) < fp32_min_normal))
    ) {
        *absolute_error = 3.402823466e+38f;
        return VS_NUMERICAL_SIGN_AMBIGUOUS;
    }
    const float determinant = left - right;
    if (!isfinite(left) || !isfinite(right) || !isfinite(determinant)) {
        *absolute_error = 3.402823466e+38f;
        return VS_NUMERICAL_SIGN_AMBIGUOUS;
    }

    const float magnitude = fabsf(left) + fabsf(right);
    const float bound =
        9.5367431640625e-7f * magnitude + 9.403954806578300e-38f;
    *absolute_error = bound;
    if (determinant > bound) return 1;
    if (determinant < -bound) return -1;
    return VS_NUMERICAL_SIGN_AMBIGUOUS;
}

__device__ inline int vs_orient2d_selective_fp32(
    double ax, double ay,
    double bx, double by,
    double cx, double cy,
    bool* refined
) {
    float absolute_error = 0.0f;
    const int coarse = vs_orient2d_fp32_envelope(
        ax, ay, bx, by, cx, cy, &absolute_error);
    (void)absolute_error;
    if (coarse != VS_NUMERICAL_SIGN_AMBIGUOUS) {
        *refined = false;
        return coarse;
    }
    *refined = true;
    return vs_orient2d(ax, ay, bx, by, cx, cy);
}
"""
