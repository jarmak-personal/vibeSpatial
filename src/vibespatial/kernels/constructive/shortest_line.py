"""NVRTC kernel source for shortest_line (binary constructive).

Tier 1 NVRTC (ADR-0033) -- geometry-specific inner loops that iterate all
segment pairs across two geometries and track the closest point pair.

ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
PrecisionPlan is wired through dispatch for observability.

The kernel outputs four coordinate arrays (ax, ay, bx, by) representing
the closest point on geometry A and the closest point on geometry B.
The dispatch layer assembles these into 2-point LineString OGAs.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Segment-segment closest-point pair (Ericson's closest-point algorithm)
#
# Unlike segment_distance which only returns the squared distance, this
# version returns the actual closest point coordinates on both segments.
# ---------------------------------------------------------------------------

_SHORTEST_LINE_KERNEL_SOURCE = r"""
#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// ===================================================================
// Level 0: segment-segment closest points (Ericson's algorithm)
// ===================================================================
// Computes the closest pair of points between two 2-D line segments.
// Returns squared distance; writes closest points to (out_px, out_py)
// on segment P and (out_qx, out_qy) on segment Q.
extern "C" __device__ inline double segment_segment_closest_points(
    const double p1x, const double p1y, const double p2x, const double p2y,
    const double q1x, const double q1y, const double q2x, const double q2y,
    double* out_px, double* out_py, double* out_qx, double* out_qy
) {
  const double d1x = p2x - p1x, d1y = p2y - p1y;
  const double d2x = q2x - q1x, d2y = q2y - q1y;
  const double rx  = p1x - q1x, ry  = p1y - q1y;

  const double a = d1x * d1x + d1y * d1y;   // |d1|^2
  const double e = d2x * d2x + d2y * d2y;   // |d2|^2
  const double f = d2x * rx  + d2y * ry;    // d2 . r

  double s, t;

  if (a <= 1e-30 && e <= 1e-30) {
    // Both degenerate to points.
    s = 0.0; t = 0.0;
  } else if (a <= 1e-30) {
    s = 0.0;
    t = f / e;
    if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
  } else {
    const double c = d1x * rx + d1y * ry;  // d1 . r
    if (e <= 1e-30) {
      t = 0.0;
      s = -c / a;
      if (s < 0.0) s = 0.0; else if (s > 1.0) s = 1.0;
    } else {
      const double b = d1x * d2x + d1y * d2y;  // d1 . d2
      const double denom = a * e - b * b;

      if (denom > 1e-30) {
        s = (b * f - c * e) / denom;
        if (s < 0.0) s = 0.0; else if (s > 1.0) s = 1.0;
      } else {
        s = 0.0;  // Nearly parallel -- pick arbitrary s, solve for t.
      }

      t = (b * s + f) / e;

      if (t < 0.0) {
        t = 0.0;
        s = -c / a;
        if (s < 0.0) s = 0.0; else if (s > 1.0) s = 1.0;
      } else if (t > 1.0) {
        t = 1.0;
        s = (b - c) / a;
        if (s < 0.0) s = 0.0; else if (s > 1.0) s = 1.0;
      }
    }
  }

  *out_px = p1x + s * d1x;
  *out_py = p1y + s * d1y;
  *out_qx = q1x + t * d2x;
  *out_qy = q1y + t * d2y;

  const double dpx = rx + s * d1x - t * d2x;
  const double dpy = ry + s * d1y - t * d2y;
  return dpx * dpx + dpy * dpy;
}

// ===================================================================
// Level 1: closest point pair across all segments in two coord ranges
// ===================================================================
extern "C" __device__ inline double coords_coords_closest_points(
    const double* __restrict__ x1, const double* __restrict__ y1, int cs1, int ce1,
    const double* __restrict__ x2, const double* __restrict__ y2, int cs2, int ce2,
    double* best_ax, double* best_ay, double* best_bx, double* best_by
) {
  double best = INFINITY;
  // Handle degenerate case: one side has only a single point (no segments)
  if (ce1 - cs1 < 2 && ce2 - cs2 < 2) {
    // Point vs point
    if (ce1 > cs1 && ce2 > cs2) {
      *best_ax = x1[cs1]; *best_ay = y1[cs1];
      *best_bx = x2[cs2]; *best_by = y2[cs2];
      const double dx = x1[cs1] - x2[cs2], dy = y1[cs1] - y2[cs2];
      return dx * dx + dy * dy;
    }
    return INFINITY;
  }
  if (ce1 - cs1 < 2) {
    // Point vs segments
    double px = x1[cs1], py = y1[cs1];
    for (int j = cs2 + 1; j < ce2; ++j) {
      double cpx, cpy, cqx, cqy;
      // Treat point as a degenerate segment (p,p) vs (q1,q2)
      double d = segment_segment_closest_points(
          px, py, px, py,
          x2[j - 1], y2[j - 1], x2[j], y2[j],
          &cpx, &cpy, &cqx, &cqy);
      if (d < best) {
        best = d;
        *best_ax = cpx; *best_ay = cpy;
        *best_bx = cqx; *best_by = cqy;
      }
      if (best <= 0.0) return 0.0;
    }
    return best;
  }
  if (ce2 - cs2 < 2) {
    // Segments vs point
    double qx = x2[cs2], qy = y2[cs2];
    for (int i = cs1 + 1; i < ce1; ++i) {
      double cpx, cpy, cqx, cqy;
      double d = segment_segment_closest_points(
          x1[i - 1], y1[i - 1], x1[i], y1[i],
          qx, qy, qx, qy,
          &cpx, &cpy, &cqx, &cqy);
      if (d < best) {
        best = d;
        *best_ax = cpx; *best_ay = cpy;
        *best_bx = cqx; *best_by = cqy;
      }
      if (best <= 0.0) return 0.0;
    }
    return best;
  }

  for (int i = cs1 + 1; i < ce1; ++i) {
    for (int j = cs2 + 1; j < ce2; ++j) {
      double cpx, cpy, cqx, cqy;
      const double d = segment_segment_closest_points(
          x1[i - 1], y1[i - 1], x1[i], y1[i],
          x2[j - 1], y2[j - 1], x2[j], y2[j],
          &cpx, &cpy, &cqx, &cqy);
      if (d < best) {
        best = d;
        *best_ax = cpx; *best_ay = cpy;
        *best_bx = cqx; *best_by = cqy;
      }
      if (best <= 0.0) return 0.0;
    }
  }
  return best;
}

// ===================================================================
// Level 1b: even-odd point-in-rings containment check
// ===================================================================
extern "C" __device__ inline bool sl_point_in_rings(
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
    for (int c = cs + 1; c < ce; ++c) {
      const double ax = x[c - 1], ay = y[c - 1];
      const double bx = x[c],     by = y[c];
      const double cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
      const double scale = fabs(bx - ax) + fabs(by - ay) + 1.0;
      if (fabs(cross_val) <= (1e-12 * scale)) {
        const double minx = ax < bx ? ax : bx;
        const double maxx = ax > bx ? ax : bx;
        const double miny = ay < by ? ay : by;
        const double maxy = ay > by ? ay : by;
        if (px >= (minx - 1e-12) && px <= (maxx + 1e-12) &&
            py >= (miny - 1e-12) && py <= (maxy + 1e-12)) {
          return true;
        }
      }
      if (((ay > py) != (by > py)) &&
          (px <= (((bx - ax) * (py - ay)) / ((by - ay) + 0.0)) + ax)) {
        inside = !inside;
      }
    }
  }
  return inside;
}

// ===================================================================
// Level 2: coord range vs polygon closest points
// ===================================================================
extern "C" __device__ inline double ls_polygon_closest_points(
    const double* __restrict__ lx, const double* __restrict__ ly, int lcs, int lce,
    const double* __restrict__ px, const double* __restrict__ py,
    const int* __restrict__ geom_offsets, const int* __restrict__ ring_offsets, int prow,
    double* best_ax, double* best_ay, double* best_bx, double* best_by
) {
  const int ring_start = geom_offsets[prow];
  const int ring_end   = geom_offsets[prow + 1];

  // Containment: first vertex of left coord range inside polygon -> distance 0
  if (lce > lcs) {
    if (sl_point_in_rings(lx[lcs], ly[lcs], px, py, ring_offsets,
                           ring_start, ring_end)) {
      *best_ax = lx[lcs]; *best_ay = ly[lcs];
      *best_bx = lx[lcs]; *best_by = ly[lcs];
      return 0.0;
    }
  }

  double best = INFINITY;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int rcs = ring_offsets[ring];
    const int rce = ring_offsets[ring + 1];
    double ax, ay, bx, by;
    const double d = coords_coords_closest_points(
        lx, ly, lcs, lce, px, py, rcs, rce,
        &ax, &ay, &bx, &by);
    if (d < best) {
      best = d;
      *best_ax = ax; *best_ay = ay;
      *best_bx = bx; *best_by = by;
    }
    if (best <= 0.0) return 0.0;
  }
  return best;
}

// ===================================================================
// Level 2b: polygon vs polygon closest points
// ===================================================================
extern "C" __device__ inline double pg_pg_closest_points(
    const double* __restrict__ x1, const double* __restrict__ y1,
    const int* __restrict__ go1, const int* __restrict__ ro1, int r1,
    const double* __restrict__ x2, const double* __restrict__ y2,
    const int* __restrict__ go2, const int* __restrict__ ro2, int r2,
    double* best_ax, double* best_ay, double* best_bx, double* best_by
) {
  const int rs1 = go1[r1], re1 = go1[r1 + 1];
  const int rs2 = go2[r2], re2 = go2[r2 + 1];

  // Containment: first vertex of poly1 inside poly2
  if (rs1 < re1) {
    const int cs = ro1[rs1];
    if (sl_point_in_rings(x1[cs], y1[cs], x2, y2, ro2, rs2, re2)) {
      *best_ax = x1[cs]; *best_ay = y1[cs];
      *best_bx = x1[cs]; *best_by = y1[cs];
      return 0.0;
    }
  }
  // Containment: first vertex of poly2 inside poly1
  if (rs2 < re2) {
    const int cs = ro2[rs2];
    if (sl_point_in_rings(x2[cs], y2[cs], x1, y1, ro1, rs1, re1)) {
      *best_ax = x2[cs]; *best_ay = y2[cs];
      *best_bx = x2[cs]; *best_by = y2[cs];
      return 0.0;
    }
  }

  double best = INFINITY;
  for (int ring1 = rs1; ring1 < re1; ++ring1) {
    const int c1s = ro1[ring1], c1e = ro1[ring1 + 1];
    for (int ring2 = rs2; ring2 < re2; ++ring2) {
      const int c2s = ro2[ring2], c2e = ro2[ring2 + 1];
      double ax, ay, bx, by;
      const double d = coords_coords_closest_points(
          x1, y1, c1s, c1e, x2, y2, c2s, c2e,
          &ax, &ay, &bx, &by);
      if (d < best) {
        best = d;
        *best_ax = ax; *best_ay = ay;
        *best_bx = bx; *best_by = by;
      }
      if (best <= 0.0) return 0.0;
    }
  }
  return best;
}

// ===================================================================
// Preamble macro (matches segment_distance pattern)
// ===================================================================
#define SL_PREAMBLE(LEFT_TAG_VAR, RIGHT_TAG_VAR)                           \
  const int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
  if (i >= pair_count) return;                                             \
  const int li = left_idx[i], ri = right_idx[i];                           \
  if (!left_validity[li] || !right_validity[ri])                           \
    { out_ax[i] = INFINITY; out_ay[i] = INFINITY;                         \
      out_bx[i] = INFINITY; out_by[i] = INFINITY; return; }               \
  if (left_tags[li] != LEFT_TAG_VAR || right_tags[ri] != RIGHT_TAG_VAR)   \
    { out_ax[i] = INFINITY; out_ay[i] = INFINITY;                         \
      out_bx[i] = INFINITY; out_by[i] = INFINITY; return; }               \
  const int lr = left_fro[li], rr = right_fro[ri];                         \
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr])                    \
    { out_ax[i] = INFINITY; out_ay[i] = INFINITY;                         \
      out_bx[i] = INFINITY; out_by[i] = INFINITY; return; }

// ===================================================================
// Point helpers: extract single coordinate from a point family row
// ===================================================================
#define PT_COORD(side, row_var) \
  const int side##_coord = side##_go[row_var]; \
  const double side##_px = side##_x[side##_coord]; \
  const double side##_py = side##_y[side##_coord];

// ===================================================================
// Global kernels: all family combinations
// ===================================================================

// ---- PT x PT ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pt_pt(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  PT_COORD(left, lr)
  PT_COORD(right, rr)
  out_ax[i] = left_px; out_ay[i] = left_py;
  out_bx[i] = right_px; out_by[i] = right_py;
}

// ---- PT x LS ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pt_ls(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcoord = left_go[lr];
  const int rcs = right_go[rr], rce = right_go[rr + 1];
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  // Point is a 1-coord range [lcoord, lcoord+1) -- handled by degenerate branch
  coords_coords_closest_points(
      left_x, left_y, lcoord, lcoord + 1,
      right_x, right_y, rcs, rce,
      &bax, &bay, &bbx, &bby);
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- PT x MLS ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pt_mls(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcoord = left_go[lr];
  const int ps = right_go[rr], pe = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int p = ps; p < pe; ++p) {
    double ax, ay, bx, by;
    double d = coords_coords_closest_points(
        left_x, left_y, lcoord, lcoord + 1,
        right_x, right_y, right_po[p], right_po[p + 1],
        &ax, &ay, &bx, &by);
    if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
    if (best <= 0.0) break;
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- PT x PG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pt_pg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcoord = left_go[lr];
  const double lpx = left_x[lcoord], lpy = left_y[lcoord];
  // Check containment: point inside polygon -> shortest line is point to itself
  const int ring_start = right_go[rr], ring_end = right_go[rr + 1];
  if (sl_point_in_rings(lpx, lpy, right_x, right_y, right_ro, ring_start, ring_end)) {
    out_ax[i] = lpx; out_ay[i] = lpy;
    out_bx[i] = lpx; out_by[i] = lpy;
    return;
  }
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int rcs = right_ro[ring], rce = right_ro[ring + 1];
    double ax, ay, bx, by;
    double d = coords_coords_closest_points(
        left_x, left_y, lcoord, lcoord + 1,
        right_x, right_y, rcs, rce,
        &ax, &ay, &bx, &by);
    if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
    if (best <= 0.0) break;
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- PT x MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pt_mpg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcoord = left_go[lr];
  const double lpx = left_x[lcoord], lpy = left_y[lcoord];
  const int poly_start = right_go[rr], poly_end = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int poly = poly_start; poly < poly_end; ++poly) {
    const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
    // Containment check
    if (sl_point_in_rings(lpx, lpy, right_x, right_y, right_ro, ring_start, ring_end)) {
      out_ax[i] = lpx; out_ay[i] = lpy;
      out_bx[i] = lpx; out_by[i] = lpy;
      return;
    }
    for (int ring = ring_start; ring < ring_end; ++ring) {
      const int rcs = right_ro[ring], rce = right_ro[ring + 1];
      double ax, ay, bx, by;
      double d = coords_coords_closest_points(
          left_x, left_y, lcoord, lcoord + 1,
          right_x, right_y, rcs, rce,
          &ax, &ay, &bx, &by);
      if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
      if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- LS x LS ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_ls_ls(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  double bax, bay, bbx, bby;
  coords_coords_closest_points(
      left_x, left_y, left_go[lr], left_go[lr + 1],
      right_x, right_y, right_go[rr], right_go[rr + 1],
      &bax, &bay, &bbx, &bby);
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- LS x MLS ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_ls_mls(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int ps = right_go[rr], pe = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int p = ps; p < pe; ++p) {
    double ax, ay, bx, by;
    double d = coords_coords_closest_points(
        left_x, left_y, lcs, lce,
        right_x, right_y, right_po[p], right_po[p + 1],
        &ax, &ay, &bx, &by);
    if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
    if (best <= 0.0) break;
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- LS x PG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_ls_pg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  double bax, bay, bbx, bby;
  ls_polygon_closest_points(
      left_x, left_y, left_go[lr], left_go[lr + 1],
      right_x, right_y, right_go, right_ro, rr,
      &bax, &bay, &bbx, &bby);
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- LS x MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_ls_mpg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int poly_start = right_go[rr], poly_end = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int poly = poly_start; poly < poly_end; ++poly) {
    const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
    // Containment check
    if (lce > lcs && sl_point_in_rings(left_x[lcs], left_y[lcs],
            right_x, right_y, right_ro, ring_start, ring_end)) {
      out_ax[i] = left_x[lcs]; out_ay[i] = left_y[lcs];
      out_bx[i] = left_x[lcs]; out_by[i] = left_y[lcs];
      return;
    }
    for (int ring = ring_start; ring < ring_end; ++ring) {
      double ax, ay, bx, by;
      const double d = coords_coords_closest_points(
          left_x, left_y, lcs, lce,
          right_x, right_y, right_ro[ring], right_ro[ring + 1],
          &ax, &ay, &bx, &by);
      if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
      if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- MLS x MLS ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_mls_mls(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int rps = right_go[rr], rpe = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    for (int rp = rps; rp < rpe; ++rp) {
      double ax, ay, bx, by;
      const double d = coords_coords_closest_points(
          left_x, left_y, left_po[lp], left_po[lp + 1],
          right_x, right_y, right_po[rp], right_po[rp + 1],
          &ax, &ay, &bx, &by);
      if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
      if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- MLS x PG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_mls_pg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    double ax, ay, bx, by;
    double d = ls_polygon_closest_points(
        left_x, left_y, left_po[lp], left_po[lp + 1],
        right_x, right_y, right_go, right_ro, rr,
        &ax, &ay, &bx, &by);
    if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
    if (best <= 0.0) break;
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- MLS x MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_mls_mpg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go2, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int poly_start = right_go2[rr], poly_end = right_go2[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    const int lcs = left_po[lp], lce = left_po[lp + 1];
    for (int poly = poly_start; poly < poly_end; ++poly) {
      const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
      // Containment
      if (lce > lcs && sl_point_in_rings(left_x[lcs], left_y[lcs],
              right_x, right_y, right_ro, ring_start, ring_end)) {
        out_ax[i] = left_x[lcs]; out_ay[i] = left_y[lcs];
        out_bx[i] = left_x[lcs]; out_by[i] = left_y[lcs];
        return;
      }
      for (int ring = ring_start; ring < ring_end; ++ring) {
        double ax, ay, bx, by;
        const double d = coords_coords_closest_points(
            left_x, left_y, lcs, lce,
            right_x, right_y, right_ro[ring], right_ro[ring + 1],
            &ax, &ay, &bx, &by);
        if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
        if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
      }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- PG x PG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pg_pg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  double bax, bay, bbx, bby;
  pg_pg_closest_points(
      left_x, left_y, left_go, left_ro, lr,
      right_x, right_y, right_go, right_ro, rr,
      &bax, &bay, &bbx, &bby);
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- PG x MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_pg_mpg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go2, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lrs = left_go[lr], lre = left_go[lr + 1];
  const int poly_start = right_go2[rr], poly_end = right_go2[rr + 1];

  double first_x, first_y;
  bool have_first = false;
  if (lrs < lre) {
    const int cs = left_ro[lrs];
    first_x = left_x[cs]; first_y = left_y[cs];
    have_first = true;
  }

  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int poly = poly_start; poly < poly_end; ++poly) {
    const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
    if (have_first && sl_point_in_rings(first_x, first_y,
            right_x, right_y, right_ro, ring_start, ring_end)) {
      out_ax[i] = first_x; out_ay[i] = first_y;
      out_bx[i] = first_x; out_by[i] = first_y;
      return;
    }
    if (ring_start < ring_end) {
      const int cs = right_ro[ring_start];
      if (sl_point_in_rings(right_x[cs], right_y[cs],
              left_x, left_y, left_ro, lrs, lre)) {
        out_ax[i] = right_x[cs]; out_ay[i] = right_y[cs];
        out_bx[i] = right_x[cs]; out_by[i] = right_y[cs];
        return;
      }
    }
    for (int lr2 = lrs; lr2 < lre; ++lr2) {
      const int c1s = left_ro[lr2], c1e = left_ro[lr2 + 1];
      for (int rr2 = ring_start; rr2 < ring_end; ++rr2) {
        const int c2s = right_ro[rr2], c2e = right_ro[rr2 + 1];
        double ax, ay, bx, by;
        const double d = coords_coords_closest_points(
            left_x, left_y, c1s, c1e, right_x, right_y, c2s, c2e,
            &ax, &ay, &bx, &by);
        if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
        if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
      }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}

// ---- MPG x MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void shortest_line_mpg_mpg(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx,
    double* __restrict__ out_ax, double* __restrict__ out_ay, double* __restrict__ out_bx, double* __restrict__ out_by,
    int pair_count
) {
  SL_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int rps = right_go[rr], rpe = right_go[rr + 1];
  double best = INFINITY;
  double bax, bay, bbx, bby;
  bax = bay = bbx = bby = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    const int lrs = left_po[lp], lre = left_po[lp + 1];
    for (int rp = rps; rp < rpe; ++rp) {
      const int rrs = right_po[rp], rre = right_po[rp + 1];
      // Containment checks
      if (lrs < lre) {
        const int cs = left_ro[lrs];
        if (sl_point_in_rings(left_x[cs], left_y[cs],
                right_x, right_y, right_ro, rrs, rre)) {
          out_ax[i] = left_x[cs]; out_ay[i] = left_y[cs];
          out_bx[i] = left_x[cs]; out_by[i] = left_y[cs];
          return;
        }
      }
      if (rrs < rre) {
        const int cs = right_ro[rrs];
        if (sl_point_in_rings(right_x[cs], right_y[cs],
                left_x, left_y, left_ro, lrs, lre)) {
          out_ax[i] = right_x[cs]; out_ay[i] = right_y[cs];
          out_bx[i] = right_x[cs]; out_by[i] = right_y[cs];
          return;
        }
      }
      for (int lr2 = lrs; lr2 < lre; ++lr2) {
        const int c1s = left_ro[lr2], c1e = left_ro[lr2 + 1];
        for (int rr2 = rrs; rr2 < rre; ++rr2) {
          const int c2s = right_ro[rr2], c2e = right_ro[rr2 + 1];
          double ax, ay, bx, by;
          const double d = coords_coords_closest_points(
              left_x, left_y, c1s, c1e, right_x, right_y, c2s, c2e,
              &ax, &ay, &bx, &by);
          if (d < best) { best = d; bax = ax; bay = ay; bbx = bx; bby = by; }
          if (best <= 0.0) { out_ax[i] = bax; out_ay[i] = bay; out_bx[i] = bbx; out_by[i] = bby; return; }
        }
      }
    }
  }
  out_ax[i] = bax; out_ay[i] = bay;
  out_bx[i] = bbx; out_by[i] = bby;
}
"""

SHORTEST_LINE_KERNEL_NAMES = (
    "shortest_line_pt_pt",
    "shortest_line_pt_ls",
    "shortest_line_pt_mls",
    "shortest_line_pt_pg",
    "shortest_line_pt_mpg",
    "shortest_line_ls_ls",
    "shortest_line_ls_mls",
    "shortest_line_ls_pg",
    "shortest_line_ls_mpg",
    "shortest_line_mls_mls",
    "shortest_line_mls_pg",
    "shortest_line_mls_mpg",
    "shortest_line_pg_pg",
    "shortest_line_pg_mpg",
    "shortest_line_mpg_mpg",
)
