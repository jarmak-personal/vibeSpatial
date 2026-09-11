"""Shared precision-parametric point/segment metric with relative residuals."""

POINT_SEGMENT_DISTANCE_DEVICE = r"""
#ifndef VIBESPATIAL_POINT_SEGMENT_DISTANCE
#define VIBESPATIAL_POINT_SEGMENT_DISTANCE
template <typename scalar_t>
__device__ inline scalar_t point_segment_sq_distance(
    scalar_t px, scalar_t py, scalar_t ax, scalar_t ay, scalar_t bx, scalar_t by
) {
  const scalar_t dx = bx - ax, dy = by - ay;
  const scalar_t len_sq = dx * dx + dy * dy;
  const scalar_t vx = px - ax, vy = py - ay;
  const scalar_t dot = vx * dx + vy * dy;
  if (len_sq == (scalar_t)0.0 || dot <= (scalar_t)0.0) {
    return vx * vx + vy * vy;
  }
  if (dot >= len_sq) {
    const scalar_t ex = px - bx, ey = py - by;
    return ex * ex + ey * ey;
  }
  const scalar_t t = dot / len_sq;
  // World-coordinate reconstruction erases small real improvements near an
  // endpoint. Keep the residual relative to the segment origin instead.
  const scalar_t ex = vx - t * dx, ey = vy - t * dy;
  return ex * ex + ey * ey;
}
#endif
"""
