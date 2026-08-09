"""Shared CUDA device function for compensated segment intersections."""

from __future__ import annotations

__all__ = ["INTERSECTION_POINT_DEVICE"]

INTERSECTION_POINT_DEVICE: str = r"""
/* ------------------------------------------------------------------ */
/* Compensated proper-segment intersection point                      */
/* Requires vs_two_sum/vs_two_product from ORIENT2D_DEVICE.           */
/* ------------------------------------------------------------------ */

typedef struct {
    double hi;
    double lo;
} vs_dd;

__device__ inline vs_dd vs_dd_normalize(double hi, double lo) {
    double s, e;
    vs_two_sum(hi, lo, s, e);
    vs_dd out;
    out.hi = s;
    out.lo = e;
    return out;
}

__device__ inline vs_dd vs_dd_sub(vs_dd a, vs_dd b) {
    double s, e;
    vs_two_sum(a.hi, -b.hi, s, e);
    return vs_dd_normalize(s, a.lo - b.lo + e);
}

__device__ inline vs_dd vs_dd_mul_double(vs_dd a, double b) {
    double p, e;
    vs_two_product(a.hi, b, p, e);
    return vs_dd_normalize(p, e + a.lo * b);
}

__device__ inline vs_dd vs_dd_mul_diff(
    double ax, double by, double ay, double bx
) {
    double p1, e1, p2, e2, s, e;
    vs_two_product(ax, by, p1, e1);
    vs_two_product(ay, bx, p2, e2);
    vs_two_sum(p1, -p2, s, e);
    return vs_dd_normalize(s, e1 - e2 + e);
}

__device__ inline double vs_dd_div(vs_dd num, vs_dd den) {
    const double q1 = num.hi / den.hi;
    vs_dd rem1 = vs_dd_sub(num, vs_dd_mul_double(den, q1));
    const double q2 = rem1.hi / den.hi;
    vs_dd rem2 = vs_dd_sub(rem1, vs_dd_mul_double(den, q2));
    const double q3 = rem2.hi / den.hi;
    return (q1 + q2) + q3;
}

__device__ inline int vs_proper_intersection_point_dd(
    double ax,
    double ay,
    double bx,
    double by,
    double cx,
    double cy,
    double dx,
    double dy,
    double* out_x,
    double* out_y
) {
    /*
     * A proper intersection is symmetric in its two supports, but evaluating
     * ``A + t(B-A)`` is not bitwise symmetric.  Normalize endpoint direction
     * and support order so repeated topology work over the same two fp64
     * segments emits one canonical coordinate without tolerance snapping.
     */
    if ((bx < ax) || (bx == ax && by < ay)) {
        const double tx = ax;
        const double ty = ay;
        ax = bx;
        ay = by;
        bx = tx;
        by = ty;
    }
    if ((dx < cx) || (dx == cx && dy < cy)) {
        const double tx = cx;
        const double ty = cy;
        cx = dx;
        cy = dy;
        dx = tx;
        dy = ty;
    }
    const int second_support_first =
        (cx < ax) ||
        (cx == ax && cy < ay) ||
        (cx == ax && cy == ay && dx < bx) ||
        (cx == ax && cy == ay && dx == bx && dy < by);
    if (second_support_first) {
        const double tax = ax;
        const double tay = ay;
        const double tbx = bx;
        const double tby = by;
        ax = cx;
        ay = cy;
        bx = dx;
        by = dy;
        cx = tax;
        cy = tay;
        dx = tbx;
        dy = tby;
    }
    const double rx = bx - ax;
    const double ry = by - ay;
    const double sx = dx - cx;
    const double sy = dy - cy;
    if (rx == 0.0 && sy == 0.0) {
        *out_x = ax;
        *out_y = cy;
        return 1;
    }
    if (ry == 0.0 && sx == 0.0) {
        *out_x = cx;
        *out_y = ay;
        return 1;
    }
    const vs_dd denominator = vs_dd_mul_diff(
        ax - bx, cy - dy, ay - by, cx - dx
    );
    if (denominator.hi == 0.0 && denominator.lo == 0.0) {
        return 0;
    }
    const vs_dd left_det = vs_dd_mul_diff(ax, by, ay, bx);
    const vs_dd right_det = vs_dd_mul_diff(cx, dy, cy, dx);
    const vs_dd numerator_x = vs_dd_sub(
        vs_dd_mul_double(left_det, cx - dx),
        vs_dd_mul_double(right_det, ax - bx)
    );
    const vs_dd numerator_y = vs_dd_sub(
        vs_dd_mul_double(left_det, cy - dy),
        vs_dd_mul_double(right_det, ay - by)
    );
    *out_x = vs_dd_div(numerator_x, denominator);
    *out_y = vs_dd_div(numerator_y, denominator);
    return 1;
}
"""
