from __future__ import annotations

from vibespatial.cuda_runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray

# ---------------------------------------------------------------------------
# Family ordering for canonical-pair normalisation (lower value = "left").
# ---------------------------------------------------------------------------
_FAMILY_ORDER: dict[GeometryFamily, int] = {
    GeometryFamily.LINESTRING: 0,
    GeometryFamily.MULTILINESTRING: 1,
    GeometryFamily.POLYGON: 2,
    GeometryFamily.MULTIPOLYGON: 3,
}


_SEGMENT_DISTANCE_KERNEL_SOURCE = """
#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// ===================================================================
// Level 0: segment–segment squared distance (Ericson's algorithm)
// ===================================================================
// Parametric closest-approach between two 2-D line segments.
// Returns squared Euclidean distance; callers take sqrt() once at the end.
extern "C" __device__ inline double segment_segment_sq_dist(
    const double p1x, const double p1y, const double p2x, const double p2y,
    const double q1x, const double q1y, const double q2x, const double q2y
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
    return rx * rx + ry * ry;
  }
  if (a <= 1e-30) {
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
        s = 0.0;  // Nearly parallel — pick arbitrary s, solve for t.
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

  const double dpx = rx + s * d1x - t * d2x;
  const double dpy = ry + s * d1y - t * d2y;
  return dpx * dpx + dpy * dpy;
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
    for (int c = cs + 1; c < ce; ++c) {
      const double ax = x[c - 1], ay = y[c - 1];
      const double bx = x[c],     by = y[c];
      // Boundary check (collinear + in bbox).
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
// Level 2: composed distance helpers
// ===================================================================

// Linestring (coord range) → single polygon distance.
// Returns 0 if linestring touches/crosses/is-inside the polygon.
extern "C" __device__ inline double ls_polygon_sq_dist(
    const double* __restrict__ lx, const double* __restrict__ ly, int lcs, int lce,
    const double* __restrict__ px, const double* __restrict__ py,
    const int* __restrict__ geom_offsets, const int* __restrict__ ring_offsets, int prow
) {
  const int ring_start = geom_offsets[prow];
  const int ring_end   = geom_offsets[prow + 1];

  // Containment: first vertex of linestring inside polygon → 0.
  if (lce > lcs) {
    if (seg_point_in_rings(lx[lcs], ly[lcs], px, py, ring_offsets,
                            ring_start, ring_end)) {
      return 0.0;
    }
  }

  // Min segment-segment distance across all rings.
  double best = INFINITY;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int rcs = ring_offsets[ring];
    const int rce = ring_offsets[ring + 1];
    const double d = coords_coords_min_sq_dist(lx, ly, lcs, lce, px, py, rcs, rce);
    if (d < best) best = d;
    if (best <= 0.0) return 0.0;
  }
  return best;
}

// Polygon → polygon distance (single polygon rows on each side).
extern "C" __device__ inline double pg_pg_sq_dist(
    const double* __restrict__ x1, const double* __restrict__ y1,
    const int* __restrict__ go1, const int* __restrict__ ro1, int r1,
    const double* __restrict__ x2, const double* __restrict__ y2,
    const int* __restrict__ go2, const int* __restrict__ ro2, int r2
) {
  const int rs1 = go1[r1], re1 = go1[r1 + 1];
  const int rs2 = go2[r2], re2 = go2[r2 + 1];

  // Containment: first vertex of poly1 inside poly2.
  if (rs1 < re1) {
    const int cs = ro1[rs1];
    if (seg_point_in_rings(x1[cs], y1[cs], x2, y2, ro2, rs2, re2)) {
      return 0.0;
    }
  }
  // Containment: first vertex of poly2 inside poly1.
  if (rs2 < re2) {
    const int cs = ro2[rs2];
    if (seg_point_in_rings(x2[cs], y2[cs], x1, y1, ro1, rs1, re1)) {
      return 0.0;
    }
  }

  // Min segment-segment distance across all ring pairs.
  double best = INFINITY;
  for (int ring1 = rs1; ring1 < re1; ++ring1) {
    const int c1s = ro1[ring1], c1e = ro1[ring1 + 1];
    for (int ring2 = rs2; ring2 < re2; ++ring2) {
      const int c2s = ro2[ring2], c2e = ro2[ring2 + 1];
      const double d = coords_coords_min_sq_dist(x1, y1, c1s, c1e, x2, y2, c2s, c2e);
      if (d < best) best = d;
      if (best <= 0.0) return 0.0;
    }
  }
  return best;
}

// ===================================================================
// Boilerplate macros for the from_owned kernel preamble
// ===================================================================
// Every kernel starts with the same thread-check / validity / tag / row
// extraction.  Using macros keeps the 10 kernel bodies short.

#define DIST_PREAMBLE(LEFT_TAG_VAR, RIGHT_TAG_VAR)                       \\
  const int i = blockIdx.x * blockDim.x + threadIdx.x;                   \\
  if (i >= pair_count) return;                                            \\
  const int li = left_idx[i], ri = right_idx[i];                         \\
  if (!left_validity[li] || !right_validity[ri])                         \\
    { out[i] = INFINITY; return; }                                       \\
  if (left_tags[li] != LEFT_TAG_VAR || right_tags[ri] != RIGHT_TAG_VAR)  \\
    { out[i] = INFINITY; return; }                                       \\
  const int lr = left_fro[li], rr = right_fro[ri];                       \\
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr])                  \\
    { out[i] = INFINITY; return; }

// ===================================================================
// Global kernels — 10 canonical (left_family, right_family) pairs
// ===================================================================

// ---- LS × LS ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_ls_ls_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  out[i] = sqrt(coords_coords_min_sq_dist(
      left_x, left_y, left_go[lr], left_go[lr + 1],
      right_x, right_y, right_go[rr], right_go[rr + 1]));
}

// ---- LS × MLS ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_ls_mls_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int ps = right_go[rr], pe = right_go[rr + 1];
  double best = INFINITY;
  for (int p = ps; p < pe; ++p) {
    const double d = coords_coords_min_sq_dist(
        left_x, left_y, lcs, lce,
        right_x, right_y, right_po[p], right_po[p + 1]);
    if (d < best) best = d;
    if (best <= 0.0) break;
  }
  out[i] = sqrt(best);
}

// ---- LS × PG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_ls_pg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  out[i] = sqrt(ls_polygon_sq_dist(
      left_x, left_y, left_go[lr], left_go[lr + 1],
      right_x, right_y, right_go, right_ro, rr));
}

// ---- LS × MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_ls_mpg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int poly_start = right_go[rr], poly_end = right_go[rr + 1];
  double best = INFINITY;
  for (int poly = poly_start; poly < poly_end; ++poly) {
    // Treat each sub-polygon as a single polygon with its own ring range.
    const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
    // Containment check.
    if (lce > lcs && seg_point_in_rings(left_x[lcs], left_y[lcs],
            right_x, right_y, right_ro, ring_start, ring_end)) {
      out[i] = 0.0; return;
    }
    for (int ring = ring_start; ring < ring_end; ++ring) {
      const double d = coords_coords_min_sq_dist(
          left_x, left_y, lcs, lce,
          right_x, right_y, right_ro[ring], right_ro[ring + 1]);
      if (d < best) best = d;
      if (best <= 0.0) { out[i] = 0.0; return; }
    }
  }
  out[i] = sqrt(best);
}

// ---- MLS × MLS ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_mls_mls_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int rps = right_go[rr], rpe = right_go[rr + 1];
  double best = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    for (int rp = rps; rp < rpe; ++rp) {
      const double d = coords_coords_min_sq_dist(
          left_x, left_y, left_po[lp], left_po[lp + 1],
          right_x, right_y, right_po[rp], right_po[rp + 1]);
      if (d < best) best = d;
      if (best <= 0.0) { out[i] = 0.0; return; }
    }
  }
  out[i] = sqrt(best);
}

// ---- MLS × PG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_mls_pg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  double best = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    const double d = ls_polygon_sq_dist(
        left_x, left_y, left_po[lp], left_po[lp + 1],
        right_x, right_y, right_go, right_ro, rr);
    if (d < best) best = d;
    if (best <= 0.0) { out[i] = 0.0; return; }
  }
  out[i] = sqrt(best);
}

// ---- MLS × MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_mls_mpg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go2, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int poly_start = right_go2[rr], poly_end = right_go2[rr + 1];
  double best = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    const int lcs = left_po[lp], lce = left_po[lp + 1];
    for (int poly = poly_start; poly < poly_end; ++poly) {
      const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
      // Containment.
      if (lce > lcs && seg_point_in_rings(left_x[lcs], left_y[lcs],
              right_x, right_y, right_ro, ring_start, ring_end)) {
        out[i] = 0.0; return;
      }
      for (int ring = ring_start; ring < ring_end; ++ring) {
        const double d = coords_coords_min_sq_dist(
            left_x, left_y, lcs, lce,
            right_x, right_y, right_ro[ring], right_ro[ring + 1]);
        if (d < best) best = d;
        if (best <= 0.0) { out[i] = 0.0; return; }
      }
    }
  }
  out[i] = sqrt(best);
}

// ---- PG × PG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_pg_pg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  out[i] = sqrt(pg_pg_sq_dist(
      left_x, left_y, left_go, left_ro, lr,
      right_x, right_y, right_go, right_ro, rr));
}

// ---- PG × MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_pg_mpg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go2, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lrs = left_go[lr], lre = left_go[lr + 1];
  const int poly_start = right_go2[rr], poly_end = right_go2[rr + 1];

  // Containment: first vertex of polygon inside any sub-polygon of multipolygon.
  double first_x, first_y;
  bool have_first = false;
  if (lrs < lre) {
    const int cs = left_ro[lrs];
    first_x = left_x[cs]; first_y = left_y[cs];
    have_first = true;
  }

  double best = INFINITY;
  for (int poly = poly_start; poly < poly_end; ++poly) {
    const int ring_start = right_po[poly], ring_end = right_po[poly + 1];
    // Check left polygon's first vertex inside this sub-polygon.
    if (have_first && seg_point_in_rings(first_x, first_y,
            right_x, right_y, right_ro, ring_start, ring_end)) {
      out[i] = 0.0; return;
    }
    // Check this sub-polygon's first vertex inside left polygon.
    if (ring_start < ring_end) {
      const int cs = right_ro[ring_start];
      if (seg_point_in_rings(right_x[cs], right_y[cs],
              left_x, left_y, left_ro, lrs, lre)) {
        out[i] = 0.0; return;
      }
    }
    // Segment-segment across all ring pairs.
    for (int lr2 = lrs; lr2 < lre; ++lr2) {
      const int c1s = left_ro[lr2], c1e = left_ro[lr2 + 1];
      for (int rr2 = ring_start; rr2 < ring_end; ++rr2) {
        const int c2s = right_ro[rr2], c2e = right_ro[rr2 + 1];
        const double d = coords_coords_min_sq_dist(
            left_x, left_y, c1s, c1e, right_x, right_y, c2s, c2e);
        if (d < best) best = d;
        if (best <= 0.0) { out[i] = 0.0; return; }
      }
    }
  }
  out[i] = sqrt(best);
}

// ---- MPG × MPG ----
extern "C" __global__ __launch_bounds__(256, 4) void distance_mpg_mpg_from_owned(
    const unsigned char* __restrict__ left_validity, const signed char* __restrict__ left_tags, const int* __restrict__ left_fro,
    const int* __restrict__ left_go, const int* __restrict__ left_po, const int* __restrict__ left_ro, const unsigned char* __restrict__ left_em, const double* __restrict__ left_x, const double* __restrict__ left_y, int left_tag,
    const unsigned char* __restrict__ right_validity, const signed char* __restrict__ right_tags, const int* __restrict__ right_fro,
    const int* __restrict__ right_go, const int* __restrict__ right_po, const int* __restrict__ right_ro, const unsigned char* __restrict__ right_em, const double* __restrict__ right_x, const double* __restrict__ right_y, int right_tag,
    const int* __restrict__ left_idx, const int* __restrict__ right_idx, double* __restrict__ out, int exclusive, int pair_count
) {
  DIST_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int rps = right_go[rr], rpe = right_go[rr + 1];
  double best = INFINITY;
  for (int lp = lps; lp < lpe; ++lp) {
    const int lrs = left_po[lp], lre = left_po[lp + 1];
    for (int rp = rps; rp < rpe; ++rp) {
      const int rrs = right_po[rp], rre = right_po[rp + 1];
      // Containment checks.
      if (lrs < lre) {
        const int cs = left_ro[lrs];
        if (seg_point_in_rings(left_x[cs], left_y[cs],
                right_x, right_y, right_ro, rrs, rre)) {
          out[i] = 0.0; return;
        }
      }
      if (rrs < rre) {
        const int cs = right_ro[rrs];
        if (seg_point_in_rings(right_x[cs], right_y[cs],
                left_x, left_y, left_ro, lrs, lre)) {
          out[i] = 0.0; return;
        }
      }
      // Segment-segment across all ring pairs.
      for (int lr2 = lrs; lr2 < lre; ++lr2) {
        const int c1s = left_ro[lr2], c1e = left_ro[lr2 + 1];
        for (int rr2 = rrs; rr2 < rre; ++rr2) {
          const int c2s = right_ro[rr2], c2e = right_ro[rr2 + 1];
          const double d = coords_coords_min_sq_dist(
              left_x, left_y, c1s, c1e, right_x, right_y, c2s, c2e);
          if (d < best) best = d;
          if (best <= 0.0) { out[i] = 0.0; return; }
        }
      }
    }
  }
  out[i] = sqrt(best);
}
"""

_SEGMENT_DISTANCE_KERNEL_NAMES = (
    "distance_ls_ls_from_owned",
    "distance_ls_mls_from_owned",
    "distance_ls_pg_from_owned",
    "distance_ls_mpg_from_owned",
    "distance_mls_mls_from_owned",
    "distance_mls_pg_from_owned",
    "distance_mls_mpg_from_owned",
    "distance_pg_pg_from_owned",
    "distance_pg_mpg_from_owned",
    "distance_mpg_mpg_from_owned",
)

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES),
])


def _segment_distance_kernels():
    return compile_kernel_group("segment-distance", _SEGMENT_DISTANCE_KERNEL_SOURCE, _SEGMENT_DISTANCE_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Canonical pair table: (left_family, right_family) → kernel name
# ---------------------------------------------------------------------------
_LS = GeometryFamily.LINESTRING
_MLS = GeometryFamily.MULTILINESTRING
_PG = GeometryFamily.POLYGON
_MPG = GeometryFamily.MULTIPOLYGON

_CANONICAL_KERNELS: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    (_LS, _LS): "distance_ls_ls_from_owned",
    (_LS, _MLS): "distance_ls_mls_from_owned",
    (_LS, _PG): "distance_ls_pg_from_owned",
    (_LS, _MPG): "distance_ls_mpg_from_owned",
    (_MLS, _MLS): "distance_mls_mls_from_owned",
    (_MLS, _PG): "distance_mls_pg_from_owned",
    (_MLS, _MPG): "distance_mls_mpg_from_owned",
    (_PG, _PG): "distance_pg_pg_from_owned",
    (_PG, _MPG): "distance_pg_mpg_from_owned",
    (_MPG, _MPG): "distance_mpg_mpg_from_owned",
}


def _family_args(state, family, runtime):
    """Build (args, arg_types) for one side of a from_owned kernel."""
    ptr = runtime.pointer
    P = KERNEL_PARAM_PTR
    buf = state.families[family]

    # Common prefix: validity, tags, family_row_offsets.
    args = [ptr(state.validity), ptr(state.tags), ptr(state.family_row_offsets)]
    types = [P, P, P]

    # geometry_offsets (always present).
    args.append(ptr(buf.geometry_offsets))
    types.append(P)

    # Family-specific extra offset arrays.
    if family in (_MLS, _MPG):
        args.append(ptr(buf.part_offsets))
        types.append(P)
    if family in (_PG, _MPG):
        args.append(ptr(buf.ring_offsets))
        types.append(P)

    # empty_mask, x, y.
    args.extend([ptr(buf.empty_mask), ptr(buf.x), ptr(buf.y)])
    types.extend([P, P, P])

    # tag value.
    args.append(FAMILY_TAGS[family])
    types.append(KERNEL_PARAM_I32)

    return args, types


def compute_segment_distance_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    exclusive: bool = False,
) -> bool:
    """Compute geometry-geometry distance on device.

    Covers all combinations of LINESTRING, MULTILINESTRING, POLYGON,
    MULTIPOLYGON on both sides.  Uses canonical-pair normalisation with
    index swapping for symmetric pairs.

    Writes results into *d_distances*.  Returns True on success, False
    if the family pair is not supported.
    """
    # Canonical ordering — lower _FAMILY_ORDER value goes on left.
    q_ord = _FAMILY_ORDER.get(query_family)
    t_ord = _FAMILY_ORDER.get(tree_family)
    if q_ord is None or t_ord is None:
        return False

    if q_ord <= t_ord:
        canonical = (query_family, tree_family)
        left_owned, right_owned = query_owned, tree_owned
        eff_left, eff_right = d_left, d_right
    else:
        canonical = (tree_family, query_family)
        left_owned, right_owned = tree_owned, query_owned
        eff_left, eff_right = d_right, d_left

    kernel_name = _CANONICAL_KERNELS.get(canonical)
    if kernel_name is None:
        return False

    from vibespatial.residency import Residency, TransferTrigger

    left_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"segment_distance GPU kernel: left {canonical[0].name}",
    )
    right_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"segment_distance GPU kernel: right {canonical[1].name}",
    )

    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _segment_distance_kernels()

    left_args, left_types = _family_args(left_state, canonical[0], runtime)
    right_args, right_types = _family_args(right_state, canonical[1], runtime)

    # Tail: left_idx, right_idx, out, exclusive, pair_count.
    tail_args = [ptr(eff_left), ptr(eff_right), ptr(d_distances),
                 1 if exclusive else 0, pair_count]
    tail_types = [KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                  KERNEL_PARAM_I32, KERNEL_PARAM_I32]

    all_args = tuple(left_args + right_args + tail_args)
    all_types = tuple(left_types + right_types + tail_types)

    grid, block = runtime.launch_config(kernels[kernel_name], pair_count)
    runtime.launch(
        kernels[kernel_name],
        grid=grid,
        block=block,
        params=(all_args, all_types),
    )
    runtime.synchronize()
    return True


def supported_segment_distance_families() -> frozenset[GeometryFamily]:
    """Return the set of geometry families supported by segment-distance kernels."""
    return frozenset(_FAMILY_ORDER.keys())
