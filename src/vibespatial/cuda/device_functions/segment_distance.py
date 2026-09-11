"""Shared FP64 segment metric and polygonal containment device functions."""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.point_in_ring import (
    POINT_IN_RING_BOUNDARY_DEVICE,
)
from vibespatial.cuda.device_functions.point_on_segment import POINT_ON_SEGMENT_DEVICE
from vibespatial.cuda.device_functions.point_segment_distance import POINT_SEGMENT_DISTANCE_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

SEGMENT_DISTANCE_DEVICE = (
    POINT_SEGMENT_DISTANCE_DEVICE
    + ORIENT2D_DEVICE
    + POINT_ON_SEGMENT_DEVICE
    + POINT_IN_RING_BOUNDARY_DEVICE
    + SPATIAL_TOLERANCE_PREAMBLE
    + r"""
#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// ===================================================================
// Level 0: segment-segment squared distance in 2D
// ===================================================================
// Exact contact classification followed by four endpoint projections.
// Returns squared Euclidean distance; callers take sqrt() once at the end.
extern "C" __device__ inline double segment_segment_sq_dist(
    const double p1x, const double p1y, const double p2x, const double p2y,
    const double q1x, const double q1y, const double q2x, const double q2y
) {
  // In 2D an intersection has exactly zero distance. Closest-approach
  // arithmetic can leave a positive rounding residual, which incorrectly
  // drops this pair when another intersecting feature produces exact zero.
  // Certify contact geometrically; never turn a positive gap into a tie by
  // rounding small metric values to zero. Most disjoint pairs skip orient2d.
  if (fmax(p1x, p2x) >= fmin(q1x, q2x) &&
      fmax(q1x, q2x) >= fmin(p1x, p2x) &&
      fmax(p1y, p2y) >= fmin(q1y, q2y) &&
      fmax(q1y, q2y) >= fmin(p1y, p2y)) {
    const int o1 = vs_orient2d(p1x, p1y, p2x, p2y, q1x, q1y);
    const int o2 = vs_orient2d(p1x, p1y, p2x, p2y, q2x, q2y);
    if (o1 * o2 <= 0) {
      const int o3 = vs_orient2d(q1x, q1y, q2x, q2y, p1x, p1y);
      const int o4 = vs_orient2d(q1x, q1y, q2x, q2y, p2x, p2y);
      if (o3 * o4 <= 0) return 0.0;
    }
  }
  // For disjoint 2D segments a closest pair contains an endpoint. This
  // avoids cancellation in the closest-approach determinant for nearly
  // parallel segments and never collapses nonzero short edges into points.
  return fmin(
      fmin(point_segment_sq_distance(p1x,p1y,q1x,q1y,q2x,q2y),
           point_segment_sq_distance(p2x,p2y,q1x,q1y,q2x,q2y)),
      fmin(point_segment_sq_distance(q1x,q1y,p1x,p1y,p2x,p2y),
           point_segment_sq_distance(q2x,q2y,p1x,p1y,p2x,p2y)));
}

// ===================================================================
// Level 1a: min sq distance between all segment pairs in two coord ranges
// ===================================================================
extern "C" __device__ inline double coords_coords_min_sq_dist(
    const double* __restrict__ x1, const double* __restrict__ y1, int cs1, int ce1,
    const double* __restrict__ x2, const double* __restrict__ y2, int cs2, int ce2
) {
  double best = INFINITY;
  for (int i = cs1 + 1; i < ce1; ++i) {
    for (int j = cs2 + 1; j < ce2; ++j) {
      const double d = segment_segment_sq_dist(
          x1[i - 1], y1[i - 1], x1[i], y1[i],
          x2[j - 1], y2[j - 1], x2[j], y2[j]);
      if (d < best) best = d;
      if (best <= 0.0) return 0.0;
    }
  }
  return best;
}

// ===================================================================
// Level 1b: even-odd point-in-rings containment check
// ===================================================================
extern "C" __device__ inline bool seg_point_in_rings(
    const double px, const double py,
    const double* __restrict__ x, const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    int ring_start, int ring_end
) {
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    if ((ce - cs) < 2) continue;
    bool on_boundary = false;
    // Distance must not collapse a positive gap into boundary contact. The
    // fp64 metric path therefore uses exact on-segment classification.
    bool ring_inside = vs_ring_contains_point_with_boundary(
        px, py, x, y, cs, ce, 0.0, &on_boundary);
    if (on_boundary) return true;
    if (ring_inside) inside = !inside;
  }
  return inside;
}

extern "C" __device__ inline int boundary_range_count(
    int kind, int row,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets
) {
  if (kind == 0) return 1;
  if (kind == 1 || kind == 2)
    return geometry_offsets[row + 1] - geometry_offsets[row];
  const int polygon_start = geometry_offsets[row];
  const int polygon_end = geometry_offsets[row + 1];
  return part_offsets[polygon_end] - part_offsets[polygon_start];
}

extern "C" __device__ inline void boundary_coord_range(
    int kind, int row, int range_index,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    int* coord_start, int* coord_end
) {
  if (kind == 0) {
    *coord_start = geometry_offsets[row];
    *coord_end = geometry_offsets[row + 1];
    return;
  }
  if (kind == 1) {
    const int part = geometry_offsets[row] + range_index;
    *coord_start = part_offsets[part];
    *coord_end = part_offsets[part + 1];
    return;
  }
  const int ring = kind == 2
      ? geometry_offsets[row] + range_index
      : part_offsets[geometry_offsets[row]] + range_index;
  *coord_start = ring_offsets[ring];
  *coord_end = ring_offsets[ring + 1];
}

extern "C" __device__ inline bool point_in_polygonal_family(
    double px, double py, int kind, int row,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ x,
    const double* __restrict__ y
) {
  if (kind == 2) {
    return seg_point_in_rings(
        px, py, x, y, ring_offsets,
        geometry_offsets[row], geometry_offsets[row + 1]);
  }
  for (int polygon = geometry_offsets[row]; polygon < geometry_offsets[row + 1]; ++polygon) {
    if (seg_point_in_rings(
            px, py, x, y, ring_offsets,
            part_offsets[polygon], part_offsets[polygon + 1])) return true;
  }
  return false;
}

"""
)
