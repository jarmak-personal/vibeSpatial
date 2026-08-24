"""NVRTC kernel sources for polygon/line DE-9IM predicates."""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.point_in_ring import POINT_IN_RING_KIND_DEVICE
from vibespatial.cuda.device_functions.point_on_segment import (
    POINT_ON_SEGMENT_KIND_DEVICE,
)
from vibespatial.cuda.device_functions.segment_crossing import SEGMENT_CROSSING_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

_POLYGON_PREDICATES_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + SEGMENT_CROSSING_DEVICE
    + SPATIAL_TOLERANCE_PREAMBLE
    + POINT_ON_SEGMENT_KIND_DEVICE
    + POINT_IN_RING_KIND_DEVICE
    + """
#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// ===================================================================
// DE-9IM bitmask constants (must match Python-side layout)
// ===================================================================
#define DE9IM_II (1 << 0)
#define DE9IM_IB (1 << 1)
#define DE9IM_IE (1 << 2)
#define DE9IM_BI (1 << 3)
#define DE9IM_BB (1 << 4)
#define DE9IM_BE (1 << 5)
#define DE9IM_EI (1 << 6)
#define DE9IM_EB (1 << 7)
#define DE9IM_EE (1 << 8)

// ===================================================================
// Device helpers
// ===================================================================

// Even-odd point-in-rings classification.
// Returns 0 = outside, 1 = on boundary, 2 = interior.
extern "C" __device__ inline unsigned char de9im_point_in_rings(
    double px, double py,
    const double* x, const double* y,
    const int* ring_offsets,
    int ring_start, int ring_end
) {
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    if ((ce - cs) < 2) continue;
    const unsigned char kind = vs_ring_point_classify(
        px, py, x, y, cs, ce, VS_SPATIAL_EPSILON);
    if (kind == 1) return 1;
    if (kind == 2) inside = !inside;
  }
  return inside ? 2 : 0;
}

extern "C" __device__ inline unsigned char de9im_point_in_polygons(
    double px, double py,
    const double* x, const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  unsigned char best_loc = 0;
  for (int p = 0; p < n_polys; ++p) {
    const unsigned char loc = de9im_point_in_rings(
        px, py, x, y, ring_offsets, poly_ring_starts[p], poly_ring_ends[p]);
    if (loc > best_loc) best_loc = loc;
    if (best_loc == 2) break;
  }
  return best_loc;
}

extern "C" __device__ inline bool polygonal_mask_is_single_convex_no_holes(
    const double* x, const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  if (n_polys != 1) return false;
  if (poly_ring_ends[0] - poly_ring_starts[0] != 1) return false;

  const int ring = poly_ring_starts[0];
  const int cs = ring_offsets[ring];
  const int ce = ring_offsets[ring + 1];
  if (ce - cs < 4) return false;

  const int last = ce - 1;
  if (x[cs] != x[last] || y[cs] != y[last]) return false;
  const int nverts = last - cs;
  if (nverts < 3) return false;

  int sign = 0;
  for (int local = 0; local < nverts; ++local) {
    const int i0 = cs + local;
    const int i1 = cs + ((local + 1) % nverts);
    const int i2 = cs + ((local + 2) % nverts);
    const double x0 = x[i0], y0 = y[i0];
    const double x1 = x[i1], y1 = y[i1];
    const double x2 = x[i2], y2 = y[i2];
    if (!isfinite(x0) || !isfinite(y0)
        || !isfinite(x1) || !isfinite(y1)
        || !isfinite(x2) || !isfinite(y2)) return false;
    const int current_sign = vs_orient2d(x0, y0, x1, y1, x2, y2);
    if (current_sign == 0) continue;
    if (sign == 0) {
      sign = current_sign;
    } else if (current_sign != sign) {
      return false;
    }
  }
  /* A consistent turn sign is not sufficient for self-intersecting stars.
     Reject every non-adjacent edge contact with exact orientation signs. */
  for (int first = 0; first < nverts; ++first) {
    const int first_next = (first + 1) % nverts;
    const double ax = x[cs + first], ay = y[cs + first];
    const double bx = x[cs + first_next], by = y[cs + first_next];
    for (int second = first + 1; second < nverts; ++second) {
      const int second_next = (second + 1) % nverts;
      if (second == first_next || second_next == first) continue;
      const double cx = x[cs + second], cy = y[cs + second];
      const double dx = x[cs + second_next], dy = y[cs + second_next];
      if (fmax(ax, bx) < fmin(cx, dx) || fmax(cx, dx) < fmin(ax, bx)
          || fmax(ay, by) < fmin(cy, dy) || fmax(cy, dy) < fmin(ay, by)) {
        continue;
      }
      const int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
      const int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
      const int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
      const int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);
      if ((o1 * o2 < 0 && o3 * o4 < 0)
          || (o1 == 0 && vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by))
          || (o2 == 0 && vs_point_on_segment_collinear(dx, dy, ax, ay, bx, by))
          || (o3 == 0 && vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy))
          || (o4 == 0 && vs_point_on_segment_collinear(bx, by, cx, cy, dx, dy))) {
        return false;
      }
    }
  }
  return sign != 0;
}

extern "C" __device__ inline bool polygonal_source_ring_is_simple_nonzero(
    const double* x, const double* y,
    const int* ring_offsets,
    int ring_start, int ring_end
) {
  if (ring_end - ring_start != 1) return false;
  const int cs = ring_offsets[ring_start];
  const int ce = ring_offsets[ring_start + 1];
  if (ce - cs < 4) return false;
  const int last = ce - 1;
  if (x[cs] != x[last] || y[cs] != y[last]) return false;
  const int nverts = last - cs;
  bool nonzero = false;
  for (int local = 0; local < nverts; ++local) {
    const int i0 = cs + local;
    const int i1 = cs + ((local + 1) % nverts);
    const int i2 = cs + ((local + 2) % nverts);
    const double x0 = x[i0], y0 = y[i0];
    const double x1 = x[i1], y1 = y[i1];
    const double x2 = x[i2], y2 = y[i2];
    if (!isfinite(x0) || !isfinite(y0)
        || !isfinite(x1) || !isfinite(y1)
        || !isfinite(x2) || !isfinite(y2)) return false;
    nonzero = nonzero || vs_orient2d(x0, y0, x1, y1, x2, y2) != 0;
  }
  if (!nonzero) return false;

  for (int first = 0; first < nverts; ++first) {
    const int first_next = (first + 1) % nverts;
    const double ax = x[cs + first], ay = y[cs + first];
    const double bx = x[cs + first_next], by = y[cs + first_next];
    for (int second = first + 1; second < nverts; ++second) {
      const int second_next = (second + 1) % nverts;
      if (second == first_next || second_next == first) continue;
      const double cx = x[cs + second], cy = y[cs + second];
      const double dx = x[cs + second_next], dy = y[cs + second_next];
      if (fmax(ax, bx) < fmin(cx, dx) || fmax(cx, dx) < fmin(ax, bx)
          || fmax(ay, by) < fmin(cy, dy) || fmax(cy, dy) < fmin(ay, by)) {
        continue;
      }
      const int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
      const int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
      const int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
      const int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);
      if ((o1 * o2 < 0 && o3 * o4 < 0)
          || (o1 == 0 && vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by))
          || (o2 == 0 && vs_point_on_segment_collinear(dx, dy, ax, ay, bx, by))
          || (o3 == 0 && vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy))
          || (o4 == 0 && vs_point_on_segment_collinear(bx, by, cx, cy, dx, dy))) {
        return false;
      }
    }
  }
  return true;
}

extern "C" __device__ inline bool polygonal_source_is_single_simple_no_holes(
    const double* x, const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  return n_polys == 1 && polygonal_source_ring_is_simple_nonzero(
      x, y, ring_offsets, poly_ring_starts[0], poly_ring_ends[0]);
}

extern "C" __device__ inline bool polygonal_source_is_single_collapsed_ring(
    const double* x, const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  if (n_polys != 1 || poly_ring_ends[0] - poly_ring_starts[0] != 1) return false;
  const int ring = poly_ring_starts[0];
  const int cs = ring_offsets[ring];
  const int ce = ring_offsets[ring + 1];
  if (ce - cs < 4) return false;
  const int last = ce - 1;
  if (x[cs] != x[last] || y[cs] != y[last]) return false;
  const int nverts = last - cs;
  for (int local = 0; local < nverts; ++local) {
    const int i0 = cs + local;
    const int i1 = cs + ((local + 1) % nverts);
    const int i2 = cs + ((local + 2) % nverts);
    if (!isfinite(x[i0]) || !isfinite(y[i0])) return false;
    if (vs_orient2d(x[i0], y[i0], x[i1], y[i1], x[i2], y[i2]) != 0) {
      return false;
    }
  }
  return true;
}

extern "C" __device__ inline unsigned char de9im_mask_is_covered_by(unsigned short mask) {
  const bool has_contact = (mask & (DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB)) != 0;
  const bool left_outside_right = (mask & (DE9IM_IE | DE9IM_BE)) != 0;
  return (has_contact && !left_outside_right) ? 1 : 0;
}

extern "C" __device__ inline bool de9im_segments_intersect_or_touch(
    double ax, double ay, double bx, double by,
    double cx, double cy, double dx, double dy
) {
  const double aminx = ax < bx ? ax : bx;
  const double amaxx = ax > bx ? ax : bx;
  const double aminy = ay < by ? ay : by;
  const double amaxy = ay > by ? ay : by;
  const double bminx = cx < dx ? cx : dx;
  const double bmaxx = cx > dx ? cx : dx;
  const double bminy = cy < dy ? cy : dy;
  const double bmaxy = cy > dy ? cy : dy;
  if (amaxx < bminx || bmaxx < aminx || amaxy < bminy || bmaxy < aminy) {
    return false;
  }

  const int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
  const int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
  const int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
  const int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);

  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0))
      && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) {
    return true;
  }
  if (o1 == 0 && vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by)) return true;
  if (o2 == 0 && vs_point_on_segment_collinear(dx, dy, ax, ay, bx, by)) return true;
  if (o3 == 0 && vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy)) return true;
  if (o4 == 0 && vs_point_on_segment_collinear(bx, by, cx, cy, dx, dy)) return true;
  return false;
}

extern "C" __device__ __noinline__ unsigned char polygonal_intersects_polygonal(
    const double* ax, const double* ay,
    const int* a_ring_offsets,
    const int* a_poly_ring_starts,
    const int* a_poly_ring_ends,
    int n_a_polys,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys
) {
  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      for (int ai = acs + 1; ai < ace; ++ai) {
        const double a1x = ax[ai - 1], a1y = ay[ai - 1];
        const double a2x = ax[ai],     a2y = ay[ai];
        for (int bp = 0; bp < n_b_polys; ++bp) {
          const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
          for (int br = brs; br < bre; ++br) {
            const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
            for (int bi = bcs + 1; bi < bce; ++bi) {
              if (de9im_segments_intersect_or_touch(
                      a1x, a1y, a2x, a2y,
                      bx[bi - 1], by[bi - 1], bx[bi], by[bi])) {
                return 1;
              }
            }
          }
        }
      }
    }
  }

  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      const int vlast = (ace > acs + 1) ? ace - 1 : ace;
      for (int vi = acs; vi < vlast; ++vi) {
        const unsigned char loc = de9im_point_in_polygons(
            ax[vi], ay[vi], bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
        if (loc != 0) return 1;
      }
    }
  }

  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      const int vlast = (bce > bcs + 1) ? bce - 1 : bce;
      for (int vi = bcs; vi < vlast; ++vi) {
        const unsigned char loc = de9im_point_in_polygons(
            bx[vi], by[vi], ax, ay, a_ring_offsets,
            a_poly_ring_starts, a_poly_ring_ends, n_a_polys);
        if (loc != 0) return 1;
      }
    }
  }

  return 0;
}

extern "C" __device__ inline bool rect_contains_point_inclusive(
    double px, double py,
    double xmin, double ymin,
    double xmax, double ymax
) {
  const double scale = fmax(fmax(fabs(xmax - xmin), fabs(ymax - ymin)), 1.0);
  const double eps = VS_SPATIAL_EPSILON * scale;
  return px >= xmin - eps && px <= xmax + eps
      && py >= ymin - eps && py <= ymax + eps;
}

extern "C" __device__ inline bool rect_contains_point_strict(
    double px, double py,
    double xmin, double ymin,
    double xmax, double ymax
) {
  const double scale = fmax(fmax(fabs(xmax - xmin), fabs(ymax - ymin)), 1.0);
  const double eps = VS_SPATIAL_EPSILON * scale;
  return px > xmin + eps && px < xmax - eps
      && py > ymin + eps && py < ymax - eps;
}

extern "C" __device__ __noinline__ bool segment_intersects_rect_open_interior(
    double x0, double y0,
    double x1, double y1,
    double xmin, double ymin,
    double xmax, double ymax
) {
  if (rect_contains_point_strict(x0, y0, xmin, ymin, xmax, ymax)
      || rect_contains_point_strict(x1, y1, xmin, ymin, xmax, ymax)) {
    return true;
  }

  double t0 = 0.0;
  double t1 = 1.0;
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  const double p[4] = {-dx, dx, -dy, dy};
  const double q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};

  for (int edge = 0; edge < 4; ++edge) {
    const double pe = p[edge];
    const double qe = q[edge];
    if (fabs(pe) <= VS_SPATIAL_EPSILON) {
      if (qe < 0.0) return false;
      continue;
    }
    const double r = qe / pe;
    if (pe < 0.0) {
      if (r > t1) return false;
      if (r > t0) t0 = r;
    } else {
      if (r < t0) return false;
      if (r < t1) t1 = r;
    }
  }

  if (t1 - t0 <= VS_SPATIAL_EPSILON) return false;
  const double tm = (t0 + t1) * 0.5;
  return rect_contains_point_strict(
      x0 + dx * tm, y0 + dy * tm, xmin, ymin, xmax, ymax);
}

extern "C" __device__ __noinline__ bool ring_vertex_in_rect_inclusive(
    const double* x,
    const double* y,
    const int* ring_offsets,
    int ring_start,
    int ring_end,
    double xmin, double ymin,
    double xmax, double ymax
) {
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    const int vlast = (ce > cs + 1) ? ce - 1 : ce;
    for (int vi = cs; vi < vlast; ++vi) {
      if (rect_contains_point_inclusive(x[vi], y[vi], xmin, ymin, xmax, ymax)) {
        return true;
      }
    }
  }
  return false;
}

extern "C" __device__ __noinline__ bool ring_segment_touches_rect_boundary(
    const double* x,
    const double* y,
    const int* ring_offsets,
    int ring_start,
    int ring_end,
    double xmin, double ymin,
    double xmax, double ymax
) {
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    for (int vi = cs + 1; vi < ce; ++vi) {
      const double ax = x[vi - 1], ay = y[vi - 1];
      const double bx = x[vi],     by = y[vi];
      if (de9im_segments_intersect_or_touch(ax, ay, bx, by, xmin, ymin, xmax, ymin)
          || de9im_segments_intersect_or_touch(ax, ay, bx, by, xmax, ymin, xmax, ymax)
          || de9im_segments_intersect_or_touch(ax, ay, bx, by, xmax, ymax, xmin, ymax)
          || de9im_segments_intersect_or_touch(ax, ay, bx, by, xmin, ymax, xmin, ymin)) {
        return true;
      }
    }
  }
  return false;
}

extern "C" __device__ __noinline__ bool ring_segment_crosses_rect_open_interior(
    const double* x,
    const double* y,
    const int* ring_offsets,
    int ring_start,
    int ring_end,
    double xmin, double ymin,
    double xmax, double ymax
) {
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    for (int vi = cs + 1; vi < ce; ++vi) {
      if (segment_intersects_rect_open_interior(
              x[vi - 1], y[vi - 1], x[vi], y[vi],
              xmin, ymin, xmax, ymax)) {
        return true;
      }
    }
  }
  return false;
}

extern "C" __device__ __noinline__ bool rect_corners_inside_polygonal(
    double xmin, double ymin,
    double xmax, double ymax,
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  return de9im_point_in_polygons(
             xmin, ymin, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      && de9im_point_in_polygons(
             xmax, ymin, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      && de9im_point_in_polygons(
             xmax, ymax, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      && de9im_point_in_polygons(
             xmin, ymax, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0;
}

extern "C" __device__ __noinline__ unsigned char rect_intersects_polygonal(
    double xmin, double ymin,
    double xmax, double ymax,
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  if (de9im_point_in_polygons(
          xmin, ymin, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      || de9im_point_in_polygons(
          xmax, ymin, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      || de9im_point_in_polygons(
          xmax, ymax, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0
      || de9im_point_in_polygons(
          xmin, ymax, x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys) != 0) {
    return 1;
  }

  for (int p = 0; p < n_polys; ++p) {
    const int rs = poly_ring_starts[p], re = poly_ring_ends[p];
    if (ring_vertex_in_rect_inclusive(x, y, ring_offsets, rs, re, xmin, ymin, xmax, ymax)
        || ring_segment_touches_rect_boundary(x, y, ring_offsets, rs, re, xmin, ymin, xmax, ymax)) {
      return 1;
    }
  }
  return 0;
}

extern "C" __device__ __noinline__ unsigned char rect_covered_by_polygonal(
    double xmin, double ymin,
    double xmax, double ymax,
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* poly_ring_starts,
    const int* poly_ring_ends,
    int n_polys
) {
  if (!rect_corners_inside_polygonal(
          xmin, ymin, xmax, ymax,
          x, y, ring_offsets, poly_ring_starts, poly_ring_ends, n_polys)) {
    return 0;
  }

  for (int p = 0; p < n_polys; ++p) {
    const int rs = poly_ring_starts[p], re = poly_ring_ends[p];
    if (ring_segment_crosses_rect_open_interior(
            x, y, ring_offsets, rs, re, xmin, ymin, xmax, ymax)) {
      return 0;
    }
  }
  return 1;
}

#define VS_DE9IM_MAX_SEGMENT_SPLITS 128

extern "C" __device__ inline void de9im_insert_segment_split_t(
    double* split_t,
    int* split_count,
    double t
) {
  const double eps = 1.0e-12;
  if (t <= eps || t >= 1.0 - eps) return;
  for (int i = 0; i < *split_count; ++i) {
    if (fabs(split_t[i] - t) <= eps) return;
  }
  if (*split_count >= VS_DE9IM_MAX_SEGMENT_SPLITS) return;
  int pos = *split_count;
  while (pos > 0 && split_t[pos - 1] > t) {
    split_t[pos] = split_t[pos - 1];
    --pos;
  }
  split_t[pos] = t;
  *split_count += 1;
}

extern "C" __device__ __noinline__ void de9im_classify_segment_interval_samples(
    double a0x, double a0y,
    double a1x, double a1y,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys,
    bool* any_inside_b,
    bool* any_boundary_b,
    bool* any_outside_b
) {
  const double adx = a1x - a0x;
  const double ady = a1y - a0y;
  const double denom = adx * adx + ady * ady;
  if (denom <= 0.0) return;

  double split_t[VS_DE9IM_MAX_SEGMENT_SPLITS];
  int split_count = 2;
  split_t[0] = 0.0;
  split_t[1] = 1.0;

  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      const int vlast = (bce > bcs + 1) ? bce - 1 : bce;
      for (int vi = bcs; vi < vlast; ++vi) {
        const double px = bx[vi], py = by[vi];
        if (vs_orient2d(a0x, a0y, a1x, a1y, px, py) == 0
            && vs_point_on_segment_collinear(px, py, a0x, a0y, a1x, a1y)) {
          const double t = ((px - a0x) * adx + (py - a0y) * ady) / denom;
          de9im_insert_segment_split_t(split_t, &split_count, t);
        }
      }
      for (int bi = bcs + 1; bi < bce; ++bi) {
        const double b0x = bx[bi - 1], b0y = by[bi - 1];
        const double b1x = bx[bi],     b1y = by[bi];
        const double bdx = b1x - b0x;
        const double bdy = b1y - b0y;
        const double cross = adx * bdy - ady * bdx;
        const double scale = fabs(adx) + fabs(ady) + fabs(bdx) + fabs(bdy) + 1.0;
        if (fabs(cross) > VS_SPATIAL_EPSILON * scale * scale) {
          const double t = ((b0x - a0x) * bdy - (b0y - a0y) * bdx) / cross;
          de9im_insert_segment_split_t(split_t, &split_count, t);
        }
      }
    }
  }

  for (int i = 0; i + 1 < split_count; ++i) {
    const double t0 = split_t[i], t1 = split_t[i + 1];
    if (t1 - t0 <= 1.0e-12) continue;

    // Prove coincident boundary intervals from source segment parameters.
    // Reclassifying a computed midpoint can move it one ulp off an exactly
    // coincident edge, especially for large translated coordinates.
    bool interval_on_boundary = false;
    for (int bp = 0; bp < n_b_polys && !interval_on_boundary; ++bp) {
      const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
      for (int br = brs; br < bre && !interval_on_boundary; ++br) {
        const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
        for (int bi = bcs + 1; bi < bce; ++bi) {
          const double b0x = bx[bi - 1], b0y = by[bi - 1];
          const double b1x = bx[bi],     b1y = by[bi];
          if (vs_orient2d(a0x, a0y, a1x, a1y, b0x, b0y) != 0
              || vs_orient2d(a0x, a0y, a1x, a1y, b1x, b1y) != 0) {
            continue;
          }
          const double bt0 = ((b0x - a0x) * adx + (b0y - a0y) * ady) / denom;
          const double bt1 = ((b1x - a0x) * adx + (b1y - a0y) * ady) / denom;
          const double blo = bt0 < bt1 ? bt0 : bt1;
          const double bhi = bt0 > bt1 ? bt0 : bt1;
          if (t0 >= blo - 1.0e-12 && t1 <= bhi + 1.0e-12) {
            interval_on_boundary = true;
            break;
          }
        }
      }
    }
    if (interval_on_boundary) {
      *any_boundary_b = true;
      continue;
    }

    const double t = (t0 + t1) * 0.5;
    const double sx = a0x + adx * t;
    const double sy = a0y + ady * t;
    const unsigned char loc = de9im_point_in_polygons(
        sx, sy, bx, by, b_ring_offsets,
        b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
    if (loc == 2) {
      *any_inside_b = true;
    } else if (loc == 1) {
      *any_boundary_b = true;
    } else {
      *any_outside_b = true;
    }
  }
}

extern "C" __device__ __noinline__ bool de9im_classify_polygon_interior_samples(
    const double* ax, const double* ay,
    const int* a_ring_offsets,
    const int* a_poly_ring_starts,
    const int* a_poly_ring_ends,
    int n_a_polys,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys,
    bool* any_inside_b,
    bool* any_outside_b
) {
  bool found_sample = false;
  *any_inside_b = false;
  *any_outside_b = false;

  for (int ap = 0; ap < n_a_polys; ++ap) {
    if (a_poly_ring_starts[ap] >= a_poly_ring_ends[ap]) continue;
    const int exterior_ring = a_poly_ring_starts[ap];
    const int acs = a_ring_offsets[exterior_ring];
    const int ace = a_ring_offsets[exterior_ring + 1];
    const int nverts = (ace > acs + 1) ? (ace - acs - 1) : (ace - acs);
    if (nverts < 3) continue;

    for (int local = 0; local < nverts; ++local) {
      const int i0 = acs + local;
      const int i1 = acs + ((local + 1) % nverts);
      const int i2 = acs + ((local + 2) % nverts);
      const double x0 = ax[i0], y0 = ay[i0];
      const double x1 = ax[i1], y1 = ay[i1];
      const double x2 = ax[i2], y2 = ay[i2];
      const double area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
      const double scale = fabs(x1 - x0) + fabs(y1 - y0)
                         + fabs(x2 - x1) + fabs(y2 - y1)
                         + fabs(x0 - x2) + fabs(y0 - y2) + 1.0;
      if (fabs(area2) <= VS_SPATIAL_EPSILON * scale * scale) continue;

      const double sx = (x0 + x1 + x2) / 3.0;
      const double sy = (y0 + y1 + y2) / 3.0;
      const unsigned char in_a = de9im_point_in_rings(
          sx, sy, ax, ay, a_ring_offsets,
          a_poly_ring_starts[ap], a_poly_ring_ends[ap]);
      if (in_a != 2) continue;

      const unsigned char loc = de9im_point_in_polygons(
          sx, sy, bx, by, b_ring_offsets,
          b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
      if (loc == 2) {
        *any_inside_b = true;
      } else if (loc == 0) {
        *any_outside_b = true;
      }
      found_sample = true;
      break;
    }
  }

  return found_sample;
}

// ===================================================================
// DE-9IM bitmask computation for a single polygon × polygon pair.
// Handles both POLYGON and MULTIPOLYGON via sub-polygon iteration.
// ===================================================================
extern "C" __device__ inline unsigned short de9im_polygon_polygon(
    // Geometry A: coordinates and offsets for all rings.
    const double* ax, const double* ay,
    const int* a_ring_offsets,
    // Sub-polygon ranges for A.  For POLYGON, n_a_polys=1 and
    // a_poly_ring_starts[0]/a_poly_ring_ends[0] span all rings.
    const int* a_poly_ring_starts,
    const int* a_poly_ring_ends,
    int n_a_polys,
    // Geometry B (symmetric).
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys
) {
  unsigned short mask = DE9IM_EE;  // EE always non-empty.

  // ---- Phase 1: Segment crossing detection ----
  // Any proper crossing between A's boundary and B's boundary implies
  // II, IB, BI, BB are all non-empty (the crossing creates interior
  // overlap and each boundary passes through the other's interior).
  for (int ap = 0; ap < n_a_polys && !(mask & DE9IM_II); ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are && !(mask & DE9IM_II); ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      for (int ai = acs + 1; ai < ace && !(mask & DE9IM_II); ++ai) {
        const double p1x = ax[ai - 1], p1y = ay[ai - 1];
        const double p2x = ax[ai],     p2y = ay[ai];
        for (int bp = 0; bp < n_b_polys; ++bp) {
          const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
          for (int br = brs; br < bre; ++br) {
            const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
            for (int bi = bcs + 1; bi < bce; ++bi) {
              if (vs_segments_properly_cross(
                      p1x, p1y, p2x, p2y,
                      bx[bi - 1], by[bi - 1], bx[bi], by[bi])) {
                mask |= DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB;
                goto crossing_done;
              }
            }
          }
        }
      }
    }
  }
  crossing_done:;

  // ---- Phase 2: Classify vertices of A w.r.t. B ----
  bool any_a_outside_b = false;
  bool any_a_vertex_outside_b = false;
  bool any_a_vertex_inside_b = false;
  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      // Skip duplicate closing vertex of each ring.
      const int vlast = (ace > acs + 1) ? ace - 1 : ace;
      for (int vi = acs; vi < vlast; ++vi) {
        const double vx = ax[vi], vy = ay[vi];
        // Check this vertex against ALL sub-polygons of B and take the
        // "deepest" classification (inside > boundary > outside).
        unsigned char best_loc = de9im_point_in_polygons(
            vx, vy, bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
        if (best_loc == 2) {
          any_a_vertex_inside_b = true;
          mask |= DE9IM_II | DE9IM_BI;
        } else if (best_loc == 1) {
          mask |= DE9IM_BB;
        } else {
          any_a_vertex_outside_b = true;
          any_a_outside_b = true;
          mask |= DE9IM_IE | DE9IM_BE;
        }
        const int vnext = (vi + 1 < vlast) ? vi + 1 : acs;
        bool segment_inside_b = false;
        bool segment_boundary_b = false;
        bool segment_outside_b = false;
        de9im_classify_segment_interval_samples(
            vx, vy, ax[vnext], ay[vnext],
            bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys,
            &segment_inside_b, &segment_boundary_b, &segment_outside_b);
        if (segment_inside_b) {
          any_a_vertex_inside_b = true;
          mask |= DE9IM_II | DE9IM_BI;
        }
        if (segment_boundary_b) {
          mask |= DE9IM_BB;
        }
        if (segment_outside_b) {
          any_a_outside_b = true;
          mask |= DE9IM_IE | DE9IM_BE;
        }
      }
    }
  }

  // ---- Phase 3: Classify vertices of B w.r.t. A (symmetric) ----
  bool any_b_outside_a = false;
  bool any_b_vertex_outside_a = false;
  bool any_b_vertex_inside_a = false;
  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      const int vlast = (bce > bcs + 1) ? bce - 1 : bce;
      for (int vi = bcs; vi < vlast; ++vi) {
        const double vx = bx[vi], vy = by[vi];
        unsigned char best_loc = de9im_point_in_polygons(
            vx, vy, ax, ay, a_ring_offsets,
            a_poly_ring_starts, a_poly_ring_ends, n_a_polys);
        if (best_loc == 2) {
          any_b_vertex_inside_a = true;
          mask |= DE9IM_II | DE9IM_IB;
        } else if (best_loc == 1) {
          mask |= DE9IM_BB;
        } else {
          any_b_vertex_outside_a = true;
          any_b_outside_a = true;
          mask |= DE9IM_EI | DE9IM_EB;
        }
        const int vnext = (vi + 1 < vlast) ? vi + 1 : bcs;
        bool segment_inside_a = false;
        bool segment_boundary_a = false;
        bool segment_outside_a = false;
        de9im_classify_segment_interval_samples(
            vx, vy, bx[vnext], by[vnext],
            ax, ay, a_ring_offsets,
            a_poly_ring_starts, a_poly_ring_ends, n_a_polys,
            &segment_inside_a, &segment_boundary_a, &segment_outside_a);
        if (segment_inside_a) {
          any_b_vertex_inside_a = true;
          mask |= DE9IM_II | DE9IM_IB;
        }
        if (segment_boundary_a) {
          mask |= DE9IM_BB;
        }
        if (segment_outside_a) {
          any_b_outside_a = true;
          mask |= DE9IM_EI | DE9IM_EB;
        }
      }
    }
  }

  // ---- Phase 4: Containment inference ----
  // If all vertices of A are inside-or-on-boundary of B (none outside),
  // then A is contained in closure(B), so Int(A) ⊂ Int(B) ∪ Bnd(B).
  // For a non-degenerate polygon A, Int(A) is non-empty and must overlap
  // Int(B) → II = T.  Symmetric for B ⊂ closure(A).
  // Boundary interval probes are floating-point witnesses for the boundary
  // DE-9IM cells.  They must not veto containment: even the rounded midpoint
  // of two exactly coincident vertices can fall one ulp off the source edge.
  // Exact vertex classifications plus an interior sample provide the
  // containment proof when every vertex lies in closure(B).
  if (!any_a_vertex_outside_b) {
    if (any_a_vertex_inside_b) {
      mask |= DE9IM_II;
    } else {
      bool sample_inside_b = false;
      bool sample_outside_b = false;
      const bool sampled = de9im_classify_polygon_interior_samples(
          ax, ay, a_ring_offsets, a_poly_ring_starts, a_poly_ring_ends, n_a_polys,
          bx, by, b_ring_offsets, b_poly_ring_starts, b_poly_ring_ends, n_b_polys,
          &sample_inside_b, &sample_outside_b);
      if (!sampled || sample_inside_b) {
        mask |= DE9IM_II;
      }
      if (sample_outside_b) {
        mask |= DE9IM_IE;
      }
    }
  }
  if (!any_b_vertex_outside_a) {
    if (any_b_vertex_inside_a) {
      mask |= DE9IM_II;
    } else {
      bool sample_inside_a = false;
      bool sample_outside_a = false;
      const bool sampled = de9im_classify_polygon_interior_samples(
          bx, by, b_ring_offsets, b_poly_ring_starts, b_poly_ring_ends, n_b_polys,
          ax, ay, a_ring_offsets, a_poly_ring_starts, a_poly_ring_ends, n_a_polys,
          &sample_inside_a, &sample_outside_a);
      if (!sampled || sample_inside_a) {
        mask |= DE9IM_II;
      }
      if (sample_outside_a) {
        mask |= DE9IM_EI;
      }
    }
  }

  return mask;
}

// Exact covered_by probe for polygonal A against a single polygonal B.
//
// Convex no-hole masks use the cheaper one-sided proof. Concave, multipart,
// and hole-bearing masks stay on device and fall through to full DE-9IM.
extern "C" __device__ inline unsigned char polygonal_covered_by_no_holes_mask(
    const double* ax, const double* ay,
    const int* a_ring_offsets,
    const int* a_poly_ring_starts,
    const int* a_poly_ring_ends,
    int n_a_polys,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys
) {
  const bool convex_mask = polygonal_mask_is_single_convex_no_holes(
      bx, by, b_ring_offsets, b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
  const bool simple_source = polygonal_source_is_single_simple_no_holes(
      ax, ay, a_ring_offsets, a_poly_ring_starts, a_poly_ring_ends, n_a_polys);
  if (convex_mask && polygonal_source_is_single_collapsed_ring(
          ax, ay, a_ring_offsets, a_poly_ring_starts, a_poly_ring_ends, n_a_polys)) {
    const int cs = a_ring_offsets[a_poly_ring_starts[0]];
    const int ce = a_ring_offsets[a_poly_ring_starts[0] + 1] - 1;
    bool any_vertex = false;
    for (int vertex = cs; vertex < ce; ++vertex) {
      const unsigned char loc = de9im_point_in_polygons(
          ax[vertex], ay[vertex], bx, by, b_ring_offsets,
          b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
      if (loc != 2) return 0;
      any_vertex = true;
    }
    return any_vertex ? 1 : 0;
  }
  if (!convex_mask || !simple_source) {
    const unsigned short mask = de9im_polygon_polygon(
        ax, ay, a_ring_offsets, a_poly_ring_starts, a_poly_ring_ends, n_a_polys,
        bx, by, b_ring_offsets, b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
    return de9im_mask_is_covered_by(mask);
  }

  bool any_contact = false;

  // Boundary of A must not properly cross boundary of B.  Touches and
  // collinear boundary overlaps are allowed for covered_by.
  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      for (int ai = acs + 1; ai < ace; ++ai) {
        const double a1x = ax[ai - 1], a1y = ay[ai - 1];
        const double a2x = ax[ai],     a2y = ay[ai];
        const double aminx = a1x < a2x ? a1x : a2x;
        const double amaxx = a1x > a2x ? a1x : a2x;
        const double aminy = a1y < a2y ? a1y : a2y;
        const double amaxy = a1y > a2y ? a1y : a2y;
        for (int bp = 0; bp < n_b_polys; ++bp) {
          const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
          for (int br = brs; br < bre; ++br) {
            const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
            for (int bi = bcs + 1; bi < bce; ++bi) {
              const double b1x = bx[bi - 1], b1y = by[bi - 1];
              const double b2x = bx[bi],     b2y = by[bi];
              const double bminx = b1x < b2x ? b1x : b2x;
              const double bmaxx = b1x > b2x ? b1x : b2x;
              const double bminy = b1y < b2y ? b1y : b2y;
              const double bmaxy = b1y > b2y ? b1y : b2y;
              if (amaxx < bminx || bmaxx < aminx || amaxy < bminy || bmaxy < aminy) {
                continue;
              }
              if (vs_segments_properly_cross(
                      a1x, a1y, a2x, a2y,
                      b1x, b1y, b2x, b2y)) {
                return 0;
              }
            }
          }
        }

        // Endpoint-only boundary contacts can hide an outside chord across a
        // concavity.  Classifying the segment midpoint catches that case while
        // proper crossings catch interior boundary exits/re-entries.
        const double mx = (a1x + a2x) * 0.5;
        const double my = (a1y + a2y) * 0.5;
        const unsigned char mid_loc = de9im_point_in_polygons(
            mx, my, bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
        if (mid_loc == 0) return 0;
        any_contact = true;
      }
    }
  }

  // Every boundary vertex of A must lie in the interior or boundary of B.
  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      const int vlast = (ace > acs + 1) ? ace - 1 : ace;
      for (int vi = acs; vi < vlast; ++vi) {
        const unsigned char loc = de9im_point_in_polygons(
            ax[vi], ay[vi], bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
        if (loc == 0) return 0;
        any_contact = true;
      }
    }
  }

  return any_contact ? 1 : 0;
}

extern "C" __device__ __noinline__ bool de9im_segment_on_polygon_boundary(
    double a0x, double a0y,
    double a1x, double a1y,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys
) {
  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      for (int bi = bcs + 1; bi < bce; ++bi) {
        const double b0x = bx[bi - 1], b0y = by[bi - 1];
        const double b1x = bx[bi],     b1y = by[bi];
        if (vs_orient2d(b0x, b0y, b1x, b1y, a0x, a0y) == 0
            && vs_orient2d(b0x, b0y, b1x, b1y, a1x, a1y) == 0
            && vs_point_on_segment_collinear(a0x, a0y, b0x, b0y, b1x, b1y)
            && vs_point_on_segment_collinear(a1x, a1y, b0x, b0y, b1x, b1y)) {
          return true;
        }
      }
    }
  }
  return false;
}

extern "C" __device__ inline unsigned char polygonal_covered_by_mask_coop(
    const double* ax, const double* ay,
    const int* a_ring_offsets,
    const int* a_poly_ring_starts,
    const int* a_poly_ring_ends,
    int n_a_polys,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts,
    const int* b_poly_ring_ends,
    int n_b_polys
) {
  __shared__ int fail;
  __shared__ int contact;
  __shared__ int convex_mask;
  __shared__ int source_kind;
  if (threadIdx.x == 0) {
    fail = 0;
    contact = 0;
    convex_mask = polygonal_mask_is_single_convex_no_holes(
        bx, by, b_ring_offsets,
        b_poly_ring_starts, b_poly_ring_ends, n_b_polys) ? 1 : 0;
    source_kind = polygonal_source_is_single_simple_no_holes(
        ax, ay, a_ring_offsets,
        a_poly_ring_starts, a_poly_ring_ends, n_a_polys) ? 1 :
        (polygonal_source_is_single_collapsed_ring(
            ax, ay, a_ring_offsets,
            a_poly_ring_starts, a_poly_ring_ends, n_a_polys) ? 2 : 0);
  }
  __syncthreads();

  const int tid = threadIdx.x;
  const int stride = blockDim.x;

  /*
   * For a certified convex mask, containment is exactly vertex shaped: every
   * straight source edge and polygonal interior lies in the convex hull of
   * the source vertices.  Classify vertices in parallel and reduce directly
   * to the pair result.  Concave, holed, multipart, or uncertain masks retain
   * the complete DE-9IM-compatible path below.
   */
  if (convex_mask && source_kind == 2) {
    int collapsed_vertex_ordinal = 0;
    for (int ap = 0; ap < n_a_polys; ++ap) {
      const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
      for (int ar = ars; ar < are; ++ar) {
        const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
        const int vlast = (ace > acs + 1) ? ace - 1 : ace;
        for (int vi = acs; vi < vlast; ++vi, ++collapsed_vertex_ordinal) {
          if ((collapsed_vertex_ordinal % stride) != tid) continue;
          const unsigned char loc = de9im_point_in_polygons(
              ax[vi], ay[vi], bx, by, b_ring_offsets,
              b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
          if (loc != 2) atomicExch(&fail, 1);
          atomicExch(&contact, 1);
        }
      }
    }
    __syncthreads();
    return (contact != 0 && fail == 0) ? 1 : 0;
  }

  if (convex_mask && source_kind == 1) {
    int convex_vertex_ordinal = 0;
    for (int ap = 0; ap < n_a_polys; ++ap) {
      const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
      for (int ar = ars; ar < are; ++ar) {
        const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
        const int vlast = (ace > acs + 1) ? ace - 1 : ace;
        for (int vi = acs; vi < vlast; ++vi, ++convex_vertex_ordinal) {
          if ((convex_vertex_ordinal % stride) != tid) continue;
          const unsigned char loc = de9im_point_in_polygons(
              ax[vi], ay[vi], bx, by, b_ring_offsets,
              b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
          if (loc == 0) {
            atomicExch(&fail, 1);
          } else {
            atomicExch(&contact, 1);
          }
        }
      }
    }
    __syncthreads();
    return (contact != 0 && fail == 0) ? 1 : 0;
  }

  int vertex_ordinal = 0;
  int segment_ordinal = 0;
  for (int ap = 0; ap < n_a_polys; ++ap) {
    const int ars = a_poly_ring_starts[ap], are = a_poly_ring_ends[ap];
    for (int ar = ars; ar < are; ++ar) {
      const int acs = a_ring_offsets[ar], ace = a_ring_offsets[ar + 1];
      const int vlast = (ace > acs + 1) ? ace - 1 : ace;
      for (int vi = acs; vi < vlast; ++vi, ++vertex_ordinal) {
        if ((vertex_ordinal % stride) != tid) continue;
        const unsigned char loc = de9im_point_in_polygons(
            ax[vi], ay[vi], bx, by, b_ring_offsets,
            b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
        if (loc == 0) {
          atomicExch(&fail, 1);
        } else {
          atomicExch(&contact, 1);
        }
      }

      for (int ai = acs + 1; ai < ace; ++ai, ++segment_ordinal) {
        if ((segment_ordinal % stride) != tid) continue;
        const double a1x = ax[ai - 1], a1y = ay[ai - 1];
        const double a2x = ax[ai],     a2y = ay[ai];
        const double aminx = a1x < a2x ? a1x : a2x;
        const double amaxx = a1x > a2x ? a1x : a2x;
        const double aminy = a1y < a2y ? a1y : a2y;
        const double amaxy = a1y > a2y ? a1y : a2y;
        for (int bp = 0; bp < n_b_polys; ++bp) {
          const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
          for (int br = brs; br < bre; ++br) {
            const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
            for (int bi = bcs + 1; bi < bce; ++bi) {
              const double b1x = bx[bi - 1], b1y = by[bi - 1];
              const double b2x = bx[bi],     b2y = by[bi];
              const double bminx = b1x < b2x ? b1x : b2x;
              const double bmaxx = b1x > b2x ? b1x : b2x;
              const double bminy = b1y < b2y ? b1y : b2y;
              const double bmaxy = b1y > b2y ? b1y : b2y;
              if (amaxx < bminx || bmaxx < aminx || amaxy < bminy || bmaxy < aminy) {
                continue;
              }
              if (vs_segments_properly_cross(
                      a1x, a1y, a2x, a2y,
                      b1x, b1y, b2x, b2y)) {
                atomicExch(&fail, 1);
              }
            }
          }
        }

        if (de9im_segment_on_polygon_boundary(
                a1x, a1y, a2x, a2y,
                bx, by, b_ring_offsets,
                b_poly_ring_starts, b_poly_ring_ends, n_b_polys)) {
          atomicExch(&contact, 1);
        } else {
          const double mx = a1x + (a2x - a1x) * 0.5;
          const double my = a1y + (a2y - a1y) * 0.5;
          const unsigned char mid_loc = de9im_point_in_polygons(
              mx, my, bx, by, b_ring_offsets,
              b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
          if (mid_loc == 0) {
            atomicExch(&fail, 1);
          } else {
            atomicExch(&contact, 1);
          }
        }
      }
    }
  }

  int hole_ordinal = 0;
  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs + 1; br < bre; ++br, ++hole_ordinal) {
      if ((hole_ordinal % stride) != tid) continue;
      const int hcs = b_ring_offsets[br], hce = b_ring_offsets[br + 1];
      const int nverts = (hce > hcs + 1) ? (hce - hcs - 1) : (hce - hcs);
      bool found_sample = false;
      double sx = 0.0, sy = 0.0;
      for (int local = 0; local < nverts; ++local) {
        const int i0 = hcs + local;
        const int i1 = hcs + ((local + 1) % nverts);
        const int i2 = hcs + ((local + 2) % nverts);
        const double x0 = bx[i0], y0 = by[i0];
        const double x1 = bx[i1], y1 = by[i1];
        const double x2 = bx[i2], y2 = by[i2];
        const double area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        const double scale = fabs(x1 - x0) + fabs(y1 - y0)
                           + fabs(x2 - x1) + fabs(y2 - y1)
                           + fabs(x0 - x2) + fabs(y0 - y2) + 1.0;
        if (fabs(area2) <= VS_SPATIAL_EPSILON * scale * scale) continue;
        const double tx = (x0 + x1 + x2) / 3.0;
        const double ty = (y0 + y1 + y2) / 3.0;
        if (de9im_point_in_rings(tx, ty, bx, by, b_ring_offsets, br, br + 1) == 2) {
          sx = tx;
          sy = ty;
          found_sample = true;
          break;
        }
      }
      if (!found_sample && nverts > 0) {
        sx = bx[hcs];
        sy = by[hcs];
        found_sample = true;
      }
      if (found_sample) {
        const unsigned char loc = de9im_point_in_polygons(
            sx, sy, ax, ay, a_ring_offsets,
            a_poly_ring_starts, a_poly_ring_ends, n_a_polys);
        if (loc != 0) {
          atomicExch(&fail, 1);
        }
      }
    }
  }

  __syncthreads();
  return (contact != 0 && fail == 0) ? 1 : 0;
}

// ===================================================================
// Global kernels
// ===================================================================

extern "C" __global__ void certify_single_polygon_convex_no_holes(
    const int* geometry_offsets,
    const int* ring_offsets,
    const unsigned char* empty_mask,
    const double* x,
    const double* y,
    int polygon_row,
    unsigned char* out
) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (empty_mask[polygon_row]) { out[0] = 0; return; }
  const int ring_start = geometry_offsets[polygon_row];
  const int ring_end = geometry_offsets[polygon_row + 1];
  out[0] = polygonal_mask_is_single_convex_no_holes(
      x, y, ring_offsets, &ring_start, &ring_end, 1) ? 1 : 0;
}

extern "C" __global__ void certify_single_multipolygon_convex_no_holes(
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    const unsigned char* empty_mask,
    const double* x,
    const double* y,
    int multipolygon_row,
    unsigned char* out
) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (empty_mask[multipolygon_row]) { out[0] = 0; return; }
  const int polygon_start = geometry_offsets[multipolygon_row];
  const int polygon_end = geometry_offsets[multipolygon_row + 1];
  out[0] = polygonal_mask_is_single_convex_no_holes(
      x, y, ring_offsets,
      part_offsets + polygon_start,
      part_offsets + polygon_start + 1,
      polygon_end - polygon_start) ? 1 : 0;
}

extern "C" __global__ void certify_polygon_sources_simple_no_holes(
    const int* geometry_offsets,
    const int* ring_offsets,
    const unsigned char* empty_mask,
    const double* x,
    const double* y,
    int row_count,
    unsigned char* row_certificates,
    int* all_certified
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) return;
  const bool certified = !empty_mask[row] && polygonal_source_ring_is_simple_nonzero(
      x, y, ring_offsets, geometry_offsets[row], geometry_offsets[row + 1]);
  row_certificates[row] = certified ? 1u : 0u;
  if (!certified) atomicExch(all_certified, 0);
}

extern "C" __global__ void certify_multipolygon_sources_simple_no_holes(
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    const unsigned char* empty_mask,
    const double* x,
    const double* y,
    int row_count,
    unsigned char* row_certificates,
    int* all_certified
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) return;
  const int polygon_start = geometry_offsets[row];
  const int polygon_end = geometry_offsets[row + 1];
  const bool certified = !empty_mask[row]
      && polygon_end - polygon_start == 1
      && polygonal_source_ring_is_simple_nonzero(
          x, y, ring_offsets,
          part_offsets[polygon_start], part_offsets[polygon_start + 1]);
  row_certificates[row] = certified ? 1u : 0u;
  if (!certified) atomicExch(all_certified, 0);
}

extern "C" __global__ void polygon_coordinate_offsets_i64(
    const int* geometry_offsets,
    const int* ring_offsets,
    long long* coordinate_offsets,
    int offset_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= offset_count) return;
  coordinate_offsets[index] = (long long)ring_offsets[geometry_offsets[index]];
}

extern "C" __global__ void multipolygon_coordinate_offsets_i64(
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    long long* coordinate_offsets,
    int offset_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= offset_count) return;
  coordinate_offsets[index] = (long long)ring_offsets[part_offsets[geometry_offsets[index]]];
}

// ---- Polygon × Polygon DE-9IM bitmask ----
extern "C" __global__ void polygon_polygon_de9im_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned short*      out_mask,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; continue; }

  // Single polygon: 1 sub-polygon whose rings span the full geometry_offsets range.
  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro,
      &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      &r_ring_start, &r_ring_end, 1);
  }
}

extern "C" __global__ void polygon_polygon_intersects_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; continue; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out[i] = polygonal_intersects_polygonal(
      left_x, left_y, left_ro,
      &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      &r_ring_start, &r_ring_end, 1);
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
rect_bounds_polygon_mask_predicates(
    const double*        rect_bounds,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    unsigned char*       out_intersects,
    unsigned char*       out_covered_by,
    int                  row_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= row_count) return;

  out_intersects[i] = 0;
  out_covered_by[i] = 0;
  if (!right_validity[0] || right_tags[0] != right_tag) return;

  const int rr = right_fro[0];
  if (rr < 0 || right_em[rr]) return;

  const double xmin = rect_bounds[i * 4 + 0];
  const double ymin = rect_bounds[i * 4 + 1];
  const double xmax = rect_bounds[i * 4 + 2];
  const double ymax = rect_bounds[i * 4 + 3];
  if (!(isfinite(xmin) && isfinite(ymin) && isfinite(xmax) && isfinite(ymax))
      || xmin >= xmax || ymin >= ymax) {
    return;
  }

  const int ring_start = right_go[rr], ring_end = right_go[rr + 1];
  out_intersects[i] = rect_intersects_polygonal(
      xmin, ymin, xmax, ymax,
      right_x, right_y, right_ro, &ring_start, &ring_end, 1);
  if (out_intersects[i]) {
    out_covered_by[i] = rect_covered_by_polygonal(
        xmin, ymin, xmax, ymax,
        right_x, right_y, right_ro, &ring_start, &ring_end, 1);
  }
}

// ---- MultiPolygon × MultiPolygon DE-9IM bitmask ----
extern "C" __global__ void multipolygon_multipolygon_de9im_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned short*      out_mask,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; continue; }

  // MultiPolygon: geometry_offsets gives polygon range, part_offsets gives ring range per polygon.
  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int n_l = l_poly_end - l_poly_start;
  const int n_r = r_poly_end - r_poly_start;

  const int* l_ring_starts = left_po + l_poly_start;
  const int* l_ring_ends = l_ring_starts + 1;
  const int* r_ring_starts = right_po + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro, l_ring_starts, l_ring_ends, n_l,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, n_r);
  }
}

extern "C" __global__ void multipolygon_multipolygon_intersects_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; continue; }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int n_l = l_poly_end - l_poly_start;
  const int n_r = r_poly_end - r_poly_start;
  const int* l_ring_starts = left_po + l_poly_start;
  const int* l_ring_ends = l_ring_starts + 1;
  const int* r_ring_starts = right_po + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;

  out[i] = polygonal_intersects_polygonal(
      left_x, left_y, left_ro, l_ring_starts, l_ring_ends, n_l,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, n_r);
  }
}

// ---- Polygon × MultiPolygon DE-9IM bitmask ----
extern "C" __global__ void polygon_multipolygon_de9im_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned short*      out_mask,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; continue; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];

  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start;
  const int* r_ring_starts = right_po + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, nr);
  }
}

extern "C" __global__ void polygon_multipolygon_intersects_from_owned(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const long long*     pair_offset,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  const int offset = pair_offset ? (int)*pair_offset : 0;
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) {
  const int i = offset + local_i;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out[i] = 0; continue; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out[i] = 0; continue; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; continue; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start;
  const int* r_ring_starts = right_po + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;

  out[i] = polygonal_intersects_polygonal(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, nr);
  }
}

// ---- Polygon × single Polygon mask covered_by probe ----
extern "C" __global__ void polygon_polygon_covered_by_mask_no_holes(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) { out[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) { out[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; return; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out[i] = polygonal_covered_by_no_holes_mask(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
}

// ---- MultiPolygon × single Polygon mask covered_by probe ----
extern "C" __global__ void multipolygon_polygon_covered_by_mask_no_holes(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) { out[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) { out[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; return; }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out[i] = polygonal_covered_by_no_holes_mask(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
}

// ---- Polygon × single MultiPolygon mask covered_by probe ----
extern "C" __global__ void polygon_multipolygon_covered_by_mask_no_holes(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) { out[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) { out[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; return; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  out[i] = polygonal_covered_by_no_holes_mask(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
}

// ---- MultiPolygon × single MultiPolygon mask covered_by probe ----
extern "C" __global__ void multipolygon_multipolygon_covered_by_mask_no_holes(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) { out[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) { out[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out[i] = 0; return; }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  out[i] = polygonal_covered_by_no_holes_mask(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
}

extern "C" __global__ void polygon_polygon_covered_by_mask_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void multipolygon_polygon_covered_by_mask_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void polygon_multipolygon_covered_by_mask_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void multipolygon_multipolygon_covered_by_mask_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count,
    int                  right_row
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  if (!left_validity[li] || !right_validity[right_row]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[right_row] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[right_row];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void polygon_polygon_covered_by_pair_rows_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void multipolygon_polygon_covered_by_pair_rows_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro, &r_ring_start, &r_ring_end, 1);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void polygon_multipolygon_covered_by_pair_rows_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
  if (threadIdx.x == 0) out[i] = value;
}

extern "C" __global__ void multipolygon_multipolygon_covered_by_pair_rows_no_holes_coop(
    const unsigned char* left_validity,
    const signed char*   left_tags,
    const int*           left_fro,
    const int*           left_go,
    const int*           left_po,
    const int*           left_ro,
    const unsigned char* left_em,
    const double*        left_x,
    const double*        left_y,
    int                  left_tag,
    const unsigned char* right_validity,
    const signed char*   right_tags,
    const int*           right_fro,
    const int*           right_go,
    const int*           right_po,
    const int*           right_ro,
    const unsigned char* right_em,
    const double*        right_x,
    const double*        right_y,
    int                  right_tag,
    const int*           left_idx,
    const int*           right_idx,
    unsigned char*       out,
    const int*           pair_count_device,
    int                  pair_count
) {
  const int i = blockIdx.x;
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count;
  if (i >= effective_pair_count) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) {
    if (threadIdx.x == 0) out[i] = 0;
    return;
  }

  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const unsigned char value = polygonal_covered_by_mask_coop(
      left_x, left_y, left_ro,
      left_po + l_poly_start, left_po + l_poly_start + 1, l_poly_end - l_poly_start,
      right_x, right_y, right_ro,
      right_po + r_poly_start, right_po + r_poly_start + 1, r_poly_end - r_poly_start);
  if (threadIdx.x == 0) out[i] = value;
}

// ===================================================================
// Line-family DE-9IM helpers
// ===================================================================

// Classify point w.r.t. a set of line parts.
// Returns: 0 = exterior, 1 = on boundary (linestring endpoint), 2 = on interior.
extern "C" __device__ inline unsigned char de9im_point_on_line(
    double px, double py,
    const double* lx, const double* ly,
    const int* part_starts, const int* part_ends, int n_parts
) {
  for (int p = 0; p < n_parts; ++p) {
    const int cs = part_starts[p], ce = part_ends[p];
    if (ce - cs < 2) continue;
    for (int c = cs + 1; c < ce; ++c) {
      const double ax = lx[c - 1], ay = ly[c - 1];
      const double bx = lx[c],     by = ly[c];
      const double cross_val = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
      const double scale = fabs(bx - ax) + fabs(by - ay) + 1.0;
      if (fabs(cross_val) <= VS_SPATIAL_EPSILON * scale) {
        const double minx = ax < bx ? ax : bx, maxx = ax > bx ? ax : bx;
        const double miny = ay < by ? ay : by, maxy = ay > by ? ay : by;
        if (px >= minx - VS_SPATIAL_EPSILON && px <= maxx + VS_SPATIAL_EPSILON &&
            py >= miny - VS_SPATIAL_EPSILON && py <= maxy + VS_SPATIAL_EPSILON) {
          // On this segment — is it at a linestring endpoint?
          if (
              (fabs(px - lx[cs]) < VS_SPATIAL_EPSILON && fabs(py - ly[cs]) < VS_SPATIAL_EPSILON)
              || (fabs(px - lx[ce - 1]) < VS_SPATIAL_EPSILON && fabs(py - ly[ce - 1]) < VS_SPATIAL_EPSILON)
          ) {
            return 1;  // boundary
          }
          return 2;  // interior
        }
      }
    }
  }
  return 0;  // exterior
}

// ===================================================================
// DE-9IM for Line × Polygon
// ===================================================================
// A = Line (interior = open segments, boundary = endpoints)
// B = Polygon (interior = area, boundary = rings)
extern "C" __device__ inline unsigned short de9im_line_polygon(
    const double* ax, const double* ay,
    const int* a_part_starts, const int* a_part_ends, int n_a_parts,
    const double* bx, const double* by,
    const int* b_ring_offsets,
    const int* b_poly_ring_starts, const int* b_poly_ring_ends, int n_b_polys
) {
  unsigned short mask = DE9IM_EE;
  // EI: polygon 2D interior is never fully covered by 1D line.
  mask |= DE9IM_EI;
  // EB: polygon boundary is never fully covered by a line (in practice).
  mask |= DE9IM_EB;
  bool any_line_outside = false;

  // Phase 1: Segment crossings (line × polygon boundary).
  for (int ap = 0; ap < n_a_parts && !(mask & DE9IM_II); ++ap) {
    const int acs = a_part_starts[ap], ace = a_part_ends[ap];
    for (int ai = acs + 1; ai < ace && !(mask & DE9IM_II); ++ai) {
      const double l1x = ax[ai - 1], l1y = ay[ai - 1];
      const double l2x = ax[ai],     l2y = ay[ai];
      for (int bp = 0; bp < n_b_polys; ++bp) {
        const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
        for (int br = brs; br < bre; ++br) {
          const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
          for (int bi = bcs + 1; bi < bce; ++bi) {
            if (vs_segments_properly_cross(l1x, l1y, l2x, l2y,
                    bx[bi - 1], by[bi - 1], bx[bi], by[bi])) {
              // A proper crossing toggles polygon membership. For valid
              // polygonal components, the line therefore has both interior
              // and exterior portions around the boundary event.
              mask |= DE9IM_II | DE9IM_IB | DE9IM_IE;
              any_line_outside = true;
              goto lp_crossing_done;
            }
            // Collinear positive-length overlap with a polygon boundary
            // segment contributes Interior(line) ∩ Boundary(polygon).
            // Vertex-only classification misses this when a single line
            // segment straddles the polygon edge with both endpoints outside.
            const double q1x = bx[bi - 1], q1y = by[bi - 1];
            const double q2x = bx[bi],     q2y = by[bi];
            const double d1 = (q2x - q1x) * (l1y - q1y) - (q2y - q1y) * (l1x - q1x);
            const double d2 = (q2x - q1x) * (l2y - q1y) - (q2y - q1y) * (l2x - q1x);
            const double scale = fabs(q2x - q1x) + fabs(q2y - q1y)
                               + fabs(l2x - l1x) + fabs(l2y - l1y) + 1.0;
            if (fabs(d1) <= VS_SPATIAL_EPSILON * scale && fabs(d2) <= VS_SPATIAL_EPSILON * scale) {
              double llo, lhi, qlo, qhi;
              if (fabs(l2x - l1x) + fabs(q2x - q1x)
                  >= fabs(l2y - l1y) + fabs(q2y - q1y)) {
                llo = l1x < l2x ? l1x : l2x; lhi = l1x > l2x ? l1x : l2x;
                qlo = q1x < q2x ? q1x : q2x; qhi = q1x > q2x ? q1x : q2x;
              } else {
                llo = l1y < l2y ? l1y : l2y; lhi = l1y > l2y ? l1y : l2y;
                qlo = q1y < q2y ? q1y : q2y; qhi = q1y > q2y ? q1y : q2y;
              }
              const double overlap_lo = llo > qlo ? llo : qlo;
              const double overlap_hi = lhi < qhi ? lhi : qhi;
              const double axis_scale = fabs(lhi - llo) + fabs(qhi - qlo) + 1.0;
              if (overlap_hi - overlap_lo > VS_SPATIAL_EPSILON * axis_scale) {
                mask |= DE9IM_IB;
              }
            }
          }
        }
      }
    }
  }

  // Phase 2a: classify every segment interior. Endpoint-only inference is
  // insufficient when both endpoints lie on a hole ring: the open segment
  // can be entirely in polygon exterior. Split constructive paths feed atomic
  // intervals here, while general predicates also gain the correct midpoint
  // evidence for boundary-to-boundary segments.
  for (int ap = 0; ap < n_a_parts; ++ap) {
    const int acs = a_part_starts[ap], ace = a_part_ends[ap];
    for (int ai = acs + 1; ai < ace; ++ai) {
      const double mx = ax[ai - 1] + (ax[ai] - ax[ai - 1]) * 0.5;
      const double my = ay[ai - 1] + (ay[ai] - ay[ai - 1]) * 0.5;
      const unsigned char loc = de9im_point_in_polygons(
          mx, my, bx, by, b_ring_offsets,
          b_poly_ring_starts, b_poly_ring_ends, n_b_polys);
      if (loc == 2) {
        mask |= DE9IM_II;
      } else if (loc == 1) {
        mask |= DE9IM_IB;
      } else {
        mask |= DE9IM_IE;
        any_line_outside = true;
      }
    }
  }
  lp_crossing_done:;

  // Phase 2: Classify line vertices w.r.t. polygon.
  for (int ap = 0; ap < n_a_parts; ++ap) {
    const int acs = a_part_starts[ap], ace = a_part_ends[ap];
    if (ace - acs < 2) continue;
    for (int vi = acs; vi < ace; ++vi) {
      const bool is_boundary = (vi == acs || vi == ace - 1);
      unsigned char best_loc = 0;
      for (int bp = 0; bp < n_b_polys; ++bp) {
        const unsigned char loc = de9im_point_in_rings(
            ax[vi], ay[vi], bx, by, b_ring_offsets,
            b_poly_ring_starts[bp], b_poly_ring_ends[bp]);
        if (loc > best_loc) best_loc = loc;
        if (best_loc == 2) break;
      }
      if (best_loc == 2) {  // inside polygon
        if (is_boundary) mask |= DE9IM_BI;
        else             mask |= DE9IM_II;
      } else if (best_loc == 1) {  // on polygon boundary
        if (is_boundary) mask |= DE9IM_BB;
        else             mask |= DE9IM_IB;
      } else {  // outside polygon
        any_line_outside = true;
        if (is_boundary) mask |= DE9IM_BE;
        else             mask |= DE9IM_IE;
      }
    }
  }

  // Phase 2b: Classify polygon boundary vertices w.r.t. the line.
  // This catches the case where the line traverses polygon interior via
  // polygon vertices, so proper-crossing never fires but the line still
  // contacts polygon boundary along its interior.
  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      const int vlast = (bce > bcs + 1) ? bce - 1 : bce;
      for (int vi = bcs; vi < vlast; ++vi) {
        unsigned char best_kind = 0;
        for (int ap = 0; ap < n_a_parts && best_kind < 2; ++ap) {
          const int acs = a_part_starts[ap], ace = a_part_ends[ap];
          if (ace - acs < 2) continue;
          for (int ai = acs + 1; ai < ace; ++ai) {
            const unsigned char kind = vs_point_on_segment_kind(
                bx[vi], by[vi],
                ax[ai - 1], ay[ai - 1], ax[ai], ay[ai],
                VS_SPATIAL_EPSILON);
            if (kind > best_kind) best_kind = kind;
            if (best_kind == 2) break;
          }
        }
        if (best_kind == 2) {
          mask |= DE9IM_IB;
        } else if (best_kind == 1) {
          mask |= DE9IM_BB;
        }
      }
    }
  }

  return mask;
}

// ===================================================================
// DE-9IM for Line × Line
// ===================================================================
extern "C" __device__ inline unsigned short de9im_line_line(
    const double* ax, const double* ay,
    const int* a_part_starts, const int* a_part_ends, int n_a_parts,
    const double* bx, const double* by,
    const int* b_part_starts, const int* b_part_ends, int n_b_parts
) {
  unsigned short mask = DE9IM_EE;

  // Phase 1: Segment crossings (proper interior-interior crossings).
  for (int ap = 0; ap < n_a_parts && !(mask & DE9IM_II); ++ap) {
    const int acs = a_part_starts[ap], ace = a_part_ends[ap];
    for (int ai = acs + 1; ai < ace && !(mask & DE9IM_II); ++ai) {
      const double p1x = ax[ai - 1], p1y = ay[ai - 1];
      const double p2x = ax[ai],     p2y = ay[ai];
      for (int bp = 0; bp < n_b_parts; ++bp) {
        const int bcs = b_part_starts[bp], bce = b_part_ends[bp];
        for (int bi = bcs + 1; bi < bce; ++bi) {
          if (vs_segments_properly_cross(p1x, p1y, p2x, p2y,
                  bx[bi - 1], by[bi - 1], bx[bi], by[bi])) {
            mask |= DE9IM_II;
            goto ll_crossing_done;
          }
        }
      }
    }
  }
  ll_crossing_done:;

  // Phase 2: Classify vertices of A w.r.t. B.
  for (int ap = 0; ap < n_a_parts; ++ap) {
    const int acs = a_part_starts[ap], ace = a_part_ends[ap];
    if (ace - acs < 2) continue;
    for (int vi = acs; vi < ace; ++vi) {
      const bool a_is_bdy = (vi == acs || vi == ace - 1);
      const unsigned char loc = de9im_point_on_line(
          ax[vi], ay[vi], bx, by, b_part_starts, b_part_ends, n_b_parts);
      if (loc == 2) {  // on B interior
        if (a_is_bdy) mask |= DE9IM_BI;
        else           mask |= DE9IM_II;
      } else if (loc == 1) {  // on B boundary
        if (a_is_bdy) mask |= DE9IM_BB;
        else           mask |= DE9IM_IB;
      } else {  // B exterior
        if (a_is_bdy) mask |= DE9IM_BE;
        else           mask |= DE9IM_IE;
      }
    }
  }

  // Phase 3: Classify vertices of B w.r.t. A (symmetric).
  for (int bp = 0; bp < n_b_parts; ++bp) {
    const int bcs = b_part_starts[bp], bce = b_part_ends[bp];
    if (bce - bcs < 2) continue;
    for (int vi = bcs; vi < bce; ++vi) {
      const bool b_is_bdy = (vi == bcs || vi == bce - 1);
      const unsigned char loc = de9im_point_on_line(
          bx[vi], by[vi], ax, ay, a_part_starts, a_part_ends, n_a_parts);
      if (loc == 2) {  // on A interior
        if (b_is_bdy) mask |= DE9IM_IB;
        else           mask |= DE9IM_II;
      } else if (loc == 1) {  // on A boundary
        if (b_is_bdy) mask |= DE9IM_BB;
        else           mask |= DE9IM_BI;
      } else {  // A exterior
        if (b_is_bdy) mask |= DE9IM_EB;
        else           mask |= DE9IM_EI;
      }
    }
  }

  // Phase 4: Collinear overlap detection.
  // If segments are collinear and overlap in their interiors, II is set.
  if (!(mask & DE9IM_II)) {
    for (int ap = 0; ap < n_a_parts && !(mask & DE9IM_II); ++ap) {
      const int acs = a_part_starts[ap], ace = a_part_ends[ap];
      for (int ai = acs + 1; ai < ace && !(mask & DE9IM_II); ++ai) {
        const double p1x = ax[ai - 1], p1y = ay[ai - 1];
        const double p2x = ax[ai],     p2y = ay[ai];
        for (int bp = 0; bp < n_b_parts; ++bp) {
          const int bcs = b_part_starts[bp], bce = b_part_ends[bp];
          for (int bi = bcs + 1; bi < bce && !(mask & DE9IM_II); ++bi) {
            const double q1x = bx[bi - 1], q1y = by[bi - 1];
            const double q2x = bx[bi],     q2y = by[bi];
            // Check collinearity.
            const double d1 = (q2x - q1x) * (p1y - q1y) - (q2y - q1y) * (p1x - q1x);
            const double d2 = (q2x - q1x) * (p2y - q1y) - (q2y - q1y) * (p2x - q1x);
            const double scale = fabs(q2x - q1x) + fabs(q2y - q1y) + fabs(p2x - p1x) + fabs(p2y - p1y) + 1.0;
            if (fabs(d1) <= VS_SPATIAL_EPSILON * scale && fabs(d2) <= VS_SPATIAL_EPSILON * scale) {
              // Collinear — check overlap on the dominant axis.
              double plo, phi, qlo, qhi;
              if (fabs(p2x - p1x) + fabs(q2x - q1x) >= fabs(p2y - p1y) + fabs(q2y - q1y)) {
                plo = p1x < p2x ? p1x : p2x; phi = p1x > p2x ? p1x : p2x;
                qlo = q1x < q2x ? q1x : q2x; qhi = q1x > q2x ? q1x : q2x;
              } else {
                plo = p1y < p2y ? p1y : p2y; phi = p1y > p2y ? p1y : p2y;
                qlo = q1y < q2y ? q1y : q2y; qhi = q1y > q2y ? q1y : q2y;
              }
              const double olo = plo > qlo ? plo : qlo;
              const double ohi = phi < qhi ? phi : qhi;
              if (ohi > olo + VS_SPATIAL_EPSILON) {
                mask |= DE9IM_II;
              }
            }
          }
        }
      }
    }
  }

  return mask;
}

// ===================================================================
// Line-family global kernels
// ===================================================================

// Preamble for line-family kernels.
#define LINE_PREAMBLE(lt, rt) \\
  const int lane = blockIdx.x * blockDim.x + threadIdx.x; \\
  const int stride = blockDim.x * gridDim.x; \\
  const int effective_pair_count = pair_count_device ? *pair_count_device : pair_count; \\
  const int offset = pair_offset ? (int)*pair_offset : 0; \\
  for (int local_i = lane; local_i < effective_pair_count; local_i += stride) { \\
  const int i = offset + local_i; \\
  const int li = left_idx[i], ri = right_idx[i]; \\
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; continue; } \\
  if (left_tags[li] != (lt) || right_tags[ri] != (rt)) { out_mask[i] = 0; continue; } \\
  const int lr = left_fro[li], rr = right_fro[ri]; \\
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; continue; }

#define LINE_POSTAMBLE }

// ---- LineString × LineString ----
extern "C" __global__ void ls_ls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int rcs = right_go[rr], rce = right_go[rr + 1];
  out_mask[i] = de9im_line_line(left_x, left_y, &lcs, &lce, 1,
                                 right_x, right_y, &rcs, &rce, 1);
  LINE_POSTAMBLE
}

// ---- LineString × MultiLineString ----
extern "C" __global__ void ls_mls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int ps = right_go[rr], pe = right_go[rr + 1];
  const int np = pe - ps;
  const int* r_starts = right_po + ps;
  const int* r_ends = r_starts + 1;
  out_mask[i] = de9im_line_line(left_x, left_y, &lcs, &lce, 1,
                                 right_x, right_y, r_starts, r_ends, np);
  LINE_POSTAMBLE
}

// ---- MultiLineString × MultiLineString ----
extern "C" __global__ void mls_mls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps;
  const int* l_starts = left_po + lps;
  const int* l_ends = l_starts + 1;

  const int rps = right_go[rr], rpe = right_go[rr + 1];
  const int nr = rpe - rps;
  const int* r_starts = right_po + rps;
  const int* r_ends = r_starts + 1;

  out_mask[i] = de9im_line_line(left_x, left_y, l_starts, l_ends, nl,
                                 right_x, right_y, r_starts, r_ends, nr);
  LINE_POSTAMBLE
}

// ---- LineString × Polygon ----
extern "C" __global__ void ls_pg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out_mask[i] = de9im_line_polygon(left_x, left_y, &lcs, &lce, 1,
                                    right_x, right_y, right_ro,
                                    &r_ring_start, &r_ring_end, 1);
  LINE_POSTAMBLE
}

// ---- LineString × MultiPolygon ----
extern "C" __global__ void ls_mpg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start;
  const int* r_ring_starts = right_po + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;
  out_mask[i] = de9im_line_polygon(left_x, left_y, &lcs, &lce, 1,
                                    right_x, right_y, right_ro,
                                    r_ring_starts, r_ring_ends, nr);
  LINE_POSTAMBLE
}

// ---- MultiLineString × Polygon ----
extern "C" __global__ void mls_pg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps;
  const int* l_starts = left_po + lps;
  const int* l_ends = l_starts + 1;
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out_mask[i] = de9im_line_polygon(left_x, left_y, l_starts, l_ends, nl,
                                    right_x, right_y, right_ro,
                                    &r_ring_start, &r_ring_end, 1);
  LINE_POSTAMBLE
}

// ---- MultiLineString × MultiPolygon ----
extern "C" __global__ void mls_mpg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po_r, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask,
    const long long* pair_offset,
    const int* pair_count_device, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps;
  const int* l_starts = left_po + lps;
  const int* l_ends = l_starts + 1;

  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start;
  const int* r_ring_starts = right_po_r + r_poly_start;
  const int* r_ring_ends = r_ring_starts + 1;
  out_mask[i] = de9im_line_polygon(left_x, left_y, l_starts, l_ends, nl,
                                    right_x, right_y, right_ro,
                                    r_ring_starts, r_ring_ends, nr);
  LINE_POSTAMBLE
}
"""
)


_POLYGON_PREDICATES_KERNEL_NAMES = (
    "certify_single_polygon_convex_no_holes",
    "certify_single_multipolygon_convex_no_holes",
    "certify_polygon_sources_simple_no_holes",
    "certify_multipolygon_sources_simple_no_holes",
    "polygon_coordinate_offsets_i64",
    "multipolygon_coordinate_offsets_i64",
    "polygon_polygon_de9im_from_owned",
    "multipolygon_multipolygon_de9im_from_owned",
    "polygon_multipolygon_de9im_from_owned",
    "polygon_polygon_intersects_from_owned",
    "rect_bounds_polygon_mask_predicates",
    "multipolygon_multipolygon_intersects_from_owned",
    "polygon_multipolygon_intersects_from_owned",
    "polygon_polygon_covered_by_mask_no_holes",
    "multipolygon_polygon_covered_by_mask_no_holes",
    "polygon_multipolygon_covered_by_mask_no_holes",
    "multipolygon_multipolygon_covered_by_mask_no_holes",
    "polygon_polygon_covered_by_mask_no_holes_coop",
    "multipolygon_polygon_covered_by_mask_no_holes_coop",
    "polygon_multipolygon_covered_by_mask_no_holes_coop",
    "multipolygon_multipolygon_covered_by_mask_no_holes_coop",
    "polygon_polygon_covered_by_pair_rows_no_holes_coop",
    "multipolygon_polygon_covered_by_pair_rows_no_holes_coop",
    "polygon_multipolygon_covered_by_pair_rows_no_holes_coop",
    "multipolygon_multipolygon_covered_by_pair_rows_no_holes_coop",
    "ls_ls_de9im_from_owned",
    "ls_mls_de9im_from_owned",
    "mls_mls_de9im_from_owned",
    "ls_pg_de9im_from_owned",
    "ls_mpg_de9im_from_owned",
    "mls_pg_de9im_from_owned",
    "mls_mpg_de9im_from_owned",
)
