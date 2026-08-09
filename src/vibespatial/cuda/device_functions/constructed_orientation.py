"""Shared CUDA device predicate for coordinates produced by construction."""

from __future__ import annotations

__all__ = ["CONSTRUCTED_ORIENTATION_DEVICE"]


CONSTRUCTED_ORIENTATION_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Orientation with propagated fp64 construction error                */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ double vs_construction_coordinate_error(
    double x,
    double y
) {
    const double scale = fmax(1.0, fmax(fabs(x), fabs(y)));
    return 8.0 * 2.2204460492503131e-16 * scale;
}

__device__ __forceinline__ int vs_constructed_orient(
    double ax, double ay,
    double bx, double by,
    double px, double py
) {
    const int exact_coordinate_sign = vs_orient2d(ax, ay, bx, by, px, py);
    const double abx = bx - ax;
    const double aby = by - ay;
    const double apx = px - ax;
    const double apy = py - ay;
    const double left_product = abx * apy;
    const double right_product = aby * apx;
    const double determinant = left_product - right_product;
    const double a_error = vs_construction_coordinate_error(ax, ay);
    const double b_error = vs_construction_coordinate_error(bx, by);
    const double p_error = vs_construction_coordinate_error(px, py);
    const double ab_error = a_error + b_error;
    const double ap_error = a_error + p_error;
    const double arithmetic_error =
        16.0 * 2.2204460492503131e-16 *
        fmax(1.0, fabs(left_product) + fabs(right_product));
    const double input_rounding_error =
        fabs(abx) * ap_error + fabs(apy) * ab_error +
        ab_error * ap_error +
        fabs(aby) * ap_error + fabs(apx) * ab_error +
        ab_error * ap_error;
    const double error = arithmetic_error + input_rounding_error;
    if (fabs(determinant) > error) return exact_coordinate_sign;
    return 0;
}

__device__ __forceinline__ int vs_source_incidence_is_uncertain(
    double ax, double ay,
    double bx, double by,
    double px, double py
) {
    const int source_sign = vs_orient2d(ax, ay, bx, by, px, py);
    return source_sign != 0 &&
        vs_constructed_orient(ax, ay, bx, by, px, py) == 0;
}
"""
