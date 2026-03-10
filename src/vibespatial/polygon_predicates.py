from __future__ import annotations

import numpy as np

from vibespatial.cuda_runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray


# ---------------------------------------------------------------------------
# DE-9IM bitmask layout
# ---------------------------------------------------------------------------
#   bit 0: II  (Interior ∩ Interior)
#   bit 1: IB  (Interior ∩ Boundary)
#   bit 2: IE  (Interior ∩ Exterior)
#   bit 3: BI  (Boundary ∩ Interior)
#   bit 4: BB  (Boundary ∩ Boundary)
#   bit 5: BE  (Boundary ∩ Exterior)
#   bit 6: EI  (Exterior ∩ Interior)
#   bit 7: EB  (Exterior ∩ Boundary)
#   bit 8: EE  (Exterior ∩ Exterior)

DE9IM_II = 1 << 0
DE9IM_IB = 1 << 1
DE9IM_IE = 1 << 2
DE9IM_BI = 1 << 3
DE9IM_BB = 1 << 4
DE9IM_BE = 1 << 5
DE9IM_EI = 1 << 6
DE9IM_EB = 1 << 7
DE9IM_EE = 1 << 8


_POLYGON_PREDICATES_KERNEL_SOURCE = """
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

// Proper crossing test: returns true if segments P1→P2 and Q1→Q2
// cross at a point that is interior to both segments.
extern "C" __device__ inline bool segments_properly_cross(
    double p1x, double p1y, double p2x, double p2y,
    double q1x, double q1y, double q2x, double q2y
) {
  const double d1 = (q2x - q1x) * (p1y - q1y) - (q2y - q1y) * (p1x - q1x);
  const double d2 = (q2x - q1x) * (p2y - q1y) - (q2y - q1y) * (p2x - q1x);
  const double d3 = (p2x - p1x) * (q1y - p1y) - (p2y - p1y) * (q1x - p1x);
  const double d4 = (p2x - p1x) * (q2y - p1y) - (p2y - p1y) * (q2x - p1x);
  if (((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) &&
      ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))) {
    return true;
  }
  return false;
}

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
    for (int c = cs + 1; c < ce; ++c) {
      const double ax = x[c - 1], ay = y[c - 1];
      const double bx = x[c],     by = y[c];
      // Boundary check.
      const double cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
      const double scale = fabs(bx - ax) + fabs(by - ay) + 1.0;
      if (fabs(cross_val) <= (1e-12 * scale)) {
        const double minx = ax < bx ? ax : bx;
        const double maxx = ax > bx ? ax : bx;
        const double miny = ay < by ? ay : by;
        const double maxy = ay > by ? ay : by;
        if (px >= (minx - 1e-12) && px <= (maxx + 1e-12) &&
            py >= (miny - 1e-12) && py <= (maxy + 1e-12)) {
          return 1;  // boundary
        }
      }
      if (((ay > py) != (by > py)) &&
          (px <= (((bx - ax) * (py - ay)) / ((by - ay) + 0.0)) + ax)) {
        inside = !inside;
      }
    }
  }
  return inside ? 2 : 0;
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
              if (segments_properly_cross(
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
        unsigned char best_loc = 0;
        for (int bp = 0; bp < n_b_polys; ++bp) {
          const unsigned char loc = de9im_point_in_rings(
              vx, vy, bx, by, b_ring_offsets,
              b_poly_ring_starts[bp], b_poly_ring_ends[bp]);
          if (loc > best_loc) best_loc = loc;
          if (best_loc == 2) break;
        }
        if (best_loc == 2) {
          mask |= DE9IM_II | DE9IM_BI;
        } else if (best_loc == 1) {
          mask |= DE9IM_BB;
        } else {
          any_a_outside_b = true;
          mask |= DE9IM_IE | DE9IM_BE;
        }
      }
    }
  }

  // ---- Phase 3: Classify vertices of B w.r.t. A (symmetric) ----
  bool any_b_outside_a = false;
  for (int bp = 0; bp < n_b_polys; ++bp) {
    const int brs = b_poly_ring_starts[bp], bre = b_poly_ring_ends[bp];
    for (int br = brs; br < bre; ++br) {
      const int bcs = b_ring_offsets[br], bce = b_ring_offsets[br + 1];
      const int vlast = (bce > bcs + 1) ? bce - 1 : bce;
      for (int vi = bcs; vi < vlast; ++vi) {
        const double vx = bx[vi], vy = by[vi];
        unsigned char best_loc = 0;
        for (int ap = 0; ap < n_a_polys; ++ap) {
          const unsigned char loc = de9im_point_in_rings(
              vx, vy, ax, ay, a_ring_offsets,
              a_poly_ring_starts[ap], a_poly_ring_ends[ap]);
          if (loc > best_loc) best_loc = loc;
          if (best_loc == 2) break;
        }
        if (best_loc == 2) {
          mask |= DE9IM_II | DE9IM_IB;
        } else if (best_loc == 1) {
          mask |= DE9IM_BB;
        } else {
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
  if (!any_a_outside_b) {
    mask |= DE9IM_II;
  }
  if (!any_b_outside_a) {
    mask |= DE9IM_II;
  }

  return mask;
}

// ===================================================================
// Global kernels
// ===================================================================

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
    int                  pair_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; return; }

  // Single polygon: 1 sub-polygon whose rings span the full geometry_offsets range.
  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro,
      &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro,
      &r_ring_start, &r_ring_end, 1);
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
    int                  pair_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; return; }

  // MultiPolygon: geometry_offsets gives polygon range, part_offsets gives ring range per polygon.
  const int l_poly_start = left_go[lr], l_poly_end = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int n_l = l_poly_end - l_poly_start;
  const int n_r = r_poly_end - r_poly_start;

  // Build ring-start / ring-end arrays from part_offsets.
  // Stack-allocate for geometries with up to 32 sub-polygons per side.
  int l_ring_starts[32], l_ring_ends[32];
  int r_ring_starts[32], r_ring_ends[32];

  const int nl = n_l < 32 ? n_l : 32;
  const int nr = n_r < 32 ? n_r : 32;
  for (int p = 0; p < nl; ++p) {
    l_ring_starts[p] = left_po[l_poly_start + p];
    l_ring_ends[p]   = left_po[l_poly_start + p + 1];
  }
  for (int p = 0; p < nr; ++p) {
    r_ring_starts[p] = right_po[r_poly_start + p];
    r_ring_ends[p]   = right_po[r_poly_start + p + 1];
  }

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro, l_ring_starts, l_ring_ends, nl,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, nr);
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
    int                  pair_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

  const int li = left_idx[i], ri = right_idx[i];
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; return; }
  if (left_tags[li] != left_tag || right_tags[ri] != right_tag) { out_mask[i] = 0; return; }
  const int lr = left_fro[li], rr = right_fro[ri];
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; return; }

  const int l_ring_start = left_go[lr], l_ring_end = left_go[lr + 1];

  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start;
  int r_ring_starts[32], r_ring_ends[32];
  const int nr_capped = nr < 32 ? nr : 32;
  for (int p = 0; p < nr_capped; ++p) {
    r_ring_starts[p] = right_po[r_poly_start + p];
    r_ring_ends[p]   = right_po[r_poly_start + p + 1];
  }

  out_mask[i] = de9im_polygon_polygon(
      left_x, left_y, left_ro, &l_ring_start, &l_ring_end, 1,
      right_x, right_y, right_ro, r_ring_starts, r_ring_ends, nr_capped);
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
      if (fabs(cross_val) <= 1e-12 * scale) {
        const double minx = ax < bx ? ax : bx, maxx = ax > bx ? ax : bx;
        const double miny = ay < by ? ay : by, maxy = ay > by ? ay : by;
        if (px >= minx - 1e-12 && px <= maxx + 1e-12 &&
            py >= miny - 1e-12 && py <= maxy + 1e-12) {
          // On this segment — is it at a linestring endpoint?
          if ((fabs(px - lx[cs]) < 1e-12 && fabs(py - ly[cs]) < 1e-12) ||
              (fabs(px - lx[ce - 1]) < 1e-12 && fabs(py - ly[ce - 1]) < 1e-12)) {
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
            if (segments_properly_cross(l1x, l1y, l2x, l2y,
                    bx[bi - 1], by[bi - 1], bx[bi], by[bi])) {
              mask |= DE9IM_II | DE9IM_IB;
              goto lp_crossing_done;
            }
          }
        }
      }
    }
  }
  lp_crossing_done:;

  // Phase 2: Classify line vertices w.r.t. polygon.
  bool any_line_outside = false;
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

  // Phase 3: Containment inference.
  if (!any_line_outside) {
    mask |= DE9IM_II;
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
          if (segments_properly_cross(p1x, p1y, p2x, p2y,
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
        else           mask |= DE9IM_EI;
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
            if (fabs(d1) <= 1e-12 * scale && fabs(d2) <= 1e-12 * scale) {
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
              if (ohi > olo + 1e-12) {
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
  const int i = blockIdx.x * blockDim.x + threadIdx.x; \\
  if (i >= pair_count) return; \\
  const int li = left_idx[i], ri = right_idx[i]; \\
  if (!left_validity[li] || !right_validity[ri]) { out_mask[i] = 0; return; } \\
  if (left_tags[li] != (lt) || right_tags[ri] != (rt)) { out_mask[i] = 0; return; } \\
  const int lr = left_fro[li], rr = right_fro[ri]; \\
  if (lr < 0 || rr < 0 || left_em[lr] || right_em[rr]) { out_mask[i] = 0; return; }

// ---- LineString × LineString ----
extern "C" __global__ void ls_ls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int rcs = right_go[rr], rce = right_go[rr + 1];
  out_mask[i] = de9im_line_line(left_x, left_y, &lcs, &lce, 1,
                                 right_x, right_y, &rcs, &rce, 1);
}

// ---- LineString × MultiLineString ----
extern "C" __global__ void ls_mls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int ps = right_go[rr], pe = right_go[rr + 1];
  const int np = pe - ps < 32 ? pe - ps : 32;
  int r_starts[32], r_ends[32];
  for (int p = 0; p < np; ++p) { r_starts[p] = right_po[ps + p]; r_ends[p] = right_po[ps + p + 1]; }
  out_mask[i] = de9im_line_line(left_x, left_y, &lcs, &lce, 1,
                                 right_x, right_y, r_starts, r_ends, np);
}

// ---- MultiLineString × MultiLineString ----
extern "C" __global__ void mls_mls_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps < 32 ? lpe - lps : 32;
  int l_starts[32], l_ends[32];
  for (int p = 0; p < nl; ++p) { l_starts[p] = left_po[lps + p]; l_ends[p] = left_po[lps + p + 1]; }

  const int rps = right_go[rr], rpe = right_go[rr + 1];
  const int nr = rpe - rps < 32 ? rpe - rps : 32;
  int r_starts[32], r_ends[32];
  for (int p = 0; p < nr; ++p) { r_starts[p] = right_po[rps + p]; r_ends[p] = right_po[rps + p + 1]; }

  out_mask[i] = de9im_line_line(left_x, left_y, l_starts, l_ends, nl,
                                 right_x, right_y, r_starts, r_ends, nr);
}

// ---- LineString × Polygon ----
extern "C" __global__ void ls_pg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out_mask[i] = de9im_line_polygon(left_x, left_y, &lcs, &lce, 1,
                                    right_x, right_y, right_ro,
                                    &r_ring_start, &r_ring_end, 1);
}

// ---- LineString × MultiPolygon ----
extern "C" __global__ void ls_mpg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lcs = left_go[lr], lce = left_go[lr + 1];
  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start < 32 ? r_poly_end - r_poly_start : 32;
  int r_ring_starts[32], r_ring_ends[32];
  for (int p = 0; p < nr; ++p) {
    r_ring_starts[p] = right_po[r_poly_start + p];
    r_ring_ends[p]   = right_po[r_poly_start + p + 1];
  }
  out_mask[i] = de9im_line_polygon(left_x, left_y, &lcs, &lce, 1,
                                    right_x, right_y, right_ro,
                                    r_ring_starts, r_ring_ends, nr);
}

// ---- MultiLineString × Polygon ----
extern "C" __global__ void mls_pg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps < 32 ? lpe - lps : 32;
  int l_starts[32], l_ends[32];
  for (int p = 0; p < nl; ++p) { l_starts[p] = left_po[lps + p]; l_ends[p] = left_po[lps + p + 1]; }
  const int r_ring_start = right_go[rr], r_ring_end = right_go[rr + 1];
  out_mask[i] = de9im_line_polygon(left_x, left_y, l_starts, l_ends, nl,
                                    right_x, right_y, right_ro,
                                    &r_ring_start, &r_ring_end, 1);
}

// ---- MultiLineString × MultiPolygon ----
extern "C" __global__ void mls_mpg_de9im_from_owned(
    const unsigned char* left_validity, const signed char* left_tags, const int* left_fro,
    const int* left_go, const int* left_po, const unsigned char* left_em, const double* left_x, const double* left_y, int left_tag,
    const unsigned char* right_validity, const signed char* right_tags, const int* right_fro,
    const int* right_go, const int* right_po_r, const int* right_ro, const unsigned char* right_em, const double* right_x, const double* right_y, int right_tag,
    const int* left_idx, const int* right_idx, unsigned short* out_mask, int pair_count
) {
  LINE_PREAMBLE(left_tag, right_tag)
  const int lps = left_go[lr], lpe = left_go[lr + 1];
  const int nl = lpe - lps < 32 ? lpe - lps : 32;
  int l_starts[32], l_ends[32];
  for (int p = 0; p < nl; ++p) { l_starts[p] = left_po[lps + p]; l_ends[p] = left_po[lps + p + 1]; }

  const int r_poly_start = right_go[rr], r_poly_end = right_go[rr + 1];
  const int nr = r_poly_end - r_poly_start < 32 ? r_poly_end - r_poly_start : 32;
  int r_ring_starts[32], r_ring_ends[32];
  for (int p = 0; p < nr; ++p) {
    r_ring_starts[p] = right_po_r[r_poly_start + p];
    r_ring_ends[p]   = right_po_r[r_poly_start + p + 1];
  }
  out_mask[i] = de9im_line_polygon(left_x, left_y, l_starts, l_ends, nl,
                                    right_x, right_y, right_ro,
                                    r_ring_starts, r_ring_ends, nr);
}
"""


_POLYGON_PREDICATES_KERNEL_NAMES = (
    "polygon_polygon_de9im_from_owned",
    "multipolygon_multipolygon_de9im_from_owned",
    "polygon_multipolygon_de9im_from_owned",
    "ls_ls_de9im_from_owned",
    "ls_mls_de9im_from_owned",
    "mls_mls_de9im_from_owned",
    "ls_pg_de9im_from_owned",
    "ls_mpg_de9im_from_owned",
    "mls_pg_de9im_from_owned",
    "mls_mpg_de9im_from_owned",
)

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402
request_nvrtc_warmup([
    ("polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES),
])


def _polygon_predicates_kernels():
    return compile_kernel_group("polygon-predicates", _POLYGON_PREDICATES_KERNEL_SOURCE, _POLYGON_PREDICATES_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Predicate evaluation from DE-9IM bitmask
# ---------------------------------------------------------------------------
# Each predicate is defined by a (required_set, required_unset) pair of
# bitmasks.  The predicate is TRUE when:
#   (mask & required_set) == required_set  AND  (mask & required_unset) == 0

_PREDICATE_RULES: dict[str, tuple[int, int]] = {
    # intersects: at least one of II, IB, BI, BB is set
    "intersects": (0, 0),  # handled specially below
    # contains: II set, EI and EB unset
    "contains": (DE9IM_II, DE9IM_EI | DE9IM_EB),
    # within: II set, IE and BE unset
    "within": (DE9IM_II, DE9IM_IE | DE9IM_BE),
    # covers: at least one of II/IB/BI/BB set, EI and EB unset
    "covers": (0, DE9IM_EI | DE9IM_EB),  # handled specially
    # covered_by: at least one of II/IB/BI/BB set, IE and BE unset
    "covered_by": (0, DE9IM_IE | DE9IM_BE),  # handled specially
    # touches: II unset, at least one of IB/BI/BB set
    "touches": (0, 0),  # handled specially
    # overlaps (same-dim = 2D polygon): II, IE, EI all set
    "overlaps": (DE9IM_II | DE9IM_IE | DE9IM_EI, 0),
    # disjoint: II, IB, BI, BB all unset
    "disjoint": (0, DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB),
    # contains_properly: contains (II set, EI/EB unset) AND BB unset
    "contains_properly": (DE9IM_II, DE9IM_EI | DE9IM_EB | DE9IM_BB),
}

_CONTACT_MASK = DE9IM_II | DE9IM_IB | DE9IM_BI | DE9IM_BB


def evaluate_predicate_from_de9im(masks: np.ndarray, predicate: str) -> np.ndarray:
    """Evaluate a spatial predicate from DE-9IM bitmasks.

    Parameters
    ----------
    masks : uint16 array of DE-9IM bitmasks
    predicate : one of the supported predicate names

    Returns
    -------
    bool array
    """
    m = masks.astype(np.uint16, copy=False)

    if predicate == "intersects":
        return (m & _CONTACT_MASK).astype(bool)

    if predicate == "touches":
        has_contact = (m & (DE9IM_IB | DE9IM_BI | DE9IM_BB)).astype(bool)
        no_ii = ~(m & DE9IM_II).astype(bool)
        return has_contact & no_ii

    if predicate == "covers":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_EI | DE9IM_EB)).astype(bool)
        return has_contact & no_ext

    if predicate == "covered_by":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_IE | DE9IM_BE)).astype(bool)
        return has_contact & no_ext

    rule = _PREDICATE_RULES.get(predicate)
    if rule is None:
        raise ValueError(f"Unsupported predicate for DE-9IM evaluation: {predicate}")

    required_set, required_unset = rule
    result = np.ones(len(m), dtype=bool)
    if required_set:
        result &= (m & required_set) == required_set
    if required_unset:
        result &= (m & required_unset) == 0
    return result


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------

# Maps (left_family, right_family) → kernel name.
_KERNEL_MAP: dict[tuple[GeometryFamily, GeometryFamily], str] = {
    # Polygon × Polygon
    (GeometryFamily.POLYGON, GeometryFamily.POLYGON): "polygon_polygon_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON): "multipolygon_multipolygon_de9im_from_owned",
    (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON): "polygon_multipolygon_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON): "polygon_multipolygon_de9im_from_owned",
    # Line × Line
    (GeometryFamily.LINESTRING, GeometryFamily.LINESTRING): "ls_ls_de9im_from_owned",
    (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING): "ls_mls_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.LINESTRING): "ls_mls_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTILINESTRING): "mls_mls_de9im_from_owned",
    # Line × Polygon
    (GeometryFamily.LINESTRING, GeometryFamily.POLYGON): "ls_pg_de9im_from_owned",
    (GeometryFamily.LINESTRING, GeometryFamily.MULTIPOLYGON): "ls_mpg_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON): "mls_pg_de9im_from_owned",
    (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON): "mls_mpg_de9im_from_owned",
    # Polygon × Line (dispatched by swapping to Line × Polygon)
    (GeometryFamily.POLYGON, GeometryFamily.LINESTRING): "ls_pg_de9im_from_owned",
    (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING): "mls_pg_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.LINESTRING): "ls_mpg_de9im_from_owned",
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTILINESTRING): "mls_mpg_de9im_from_owned",
}


_LINE_FAMILIES = frozenset({GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING})
_POLYGON_FAMILIES = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})

# Pairs that require swapping (A,B) to (B,A) before kernel dispatch.
# The kernel is written with a specific left/right layout, so we swap
# and transpose the DE-9IM result.
_SWAP_PAIRS: dict[tuple[GeometryFamily, GeometryFamily], tuple[GeometryFamily, GeometryFamily]] = {
    # MPG×PG → PG×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON):
        (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
    # MLS×LS → LS×MLS
    (GeometryFamily.MULTILINESTRING, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING),
    # PG×LS → LS×PG
    (GeometryFamily.POLYGON, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.POLYGON),
    # PG×MLS → MLS×PG
    (GeometryFamily.POLYGON, GeometryFamily.MULTILINESTRING):
        (GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON),
    # MPG×LS → LS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.LINESTRING):
        (GeometryFamily.LINESTRING, GeometryFamily.MULTIPOLYGON),
    # MPG×MLS → MLS×MPG
    (GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTILINESTRING):
        (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON),
}


def _build_side_args(ptr, state, buf, family):
    """Build kernel args + types for one side of a DE-9IM kernel call."""
    P = KERNEL_PARAM_PTR
    I32 = KERNEL_PARAM_I32

    args = [
        ptr(state.validity), ptr(state.tags), ptr(state.family_row_offsets),
        ptr(buf.geometry_offsets),
    ]
    types = [P, P, P, P]

    # Multi-families need part_offsets before ring/coord offsets.
    if family in (GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON):
        args.append(ptr(buf.part_offsets))
        types.append(P)

    # Polygon families need ring_offsets.
    if family in _POLYGON_FAMILIES:
        args.append(ptr(buf.ring_offsets))
        types.append(P)

    args.extend([ptr(buf.empty_mask), ptr(buf.x), ptr(buf.y), FAMILY_TAGS[family]])
    types.extend([P, P, P, I32])
    return args, types


def compute_polygon_de9im_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    *,
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
    d_left: object | None = None,
    d_right: object | None = None,
) -> np.ndarray | None:
    """Compute DE-9IM bitmasks for geometry candidate pairs on GPU.

    Supports all combinations of LINESTRING, MULTILINESTRING, POLYGON,
    and MULTIPOLYGON families.

    When *d_left* / *d_right* are provided (device-resident CuPy int32
    arrays), they are used directly instead of uploading *left_indices* /
    *right_indices* from host — avoiding a redundant host→device transfer
    when candidates are already on device.

    Returns uint16 array of DE-9IM bitmasks, or None if the family pair is
    not supported.
    """
    key = (query_family, tree_family)
    swap = False
    if key in _SWAP_PAIRS:
        swap = True
        key = _SWAP_PAIRS[key]

    kernel_name = _KERNEL_MAP.get(key)
    if kernel_name is None:
        return None

    from vibespatial.residency import Residency, TransferTrigger

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"de9im GPU: query {query_family.name}",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"de9im GPU: tree {tree_family.name}",
    )

    if swap:
        eff_query_owned, eff_tree_owned = tree_owned, query_owned
        eff_query_family, eff_tree_family = tree_family, query_family
        eff_left, eff_right = right_indices, left_indices
        eff_d_left, eff_d_right = d_right, d_left
    else:
        eff_query_owned, eff_tree_owned = query_owned, tree_owned
        eff_query_family, eff_tree_family = query_family, tree_family
        eff_left, eff_right = left_indices, right_indices
        eff_d_left, eff_d_right = d_left, d_right

    query_state = eff_query_owned._ensure_device_state()
    tree_state = eff_tree_owned._ensure_device_state()
    query_buf = query_state.families[eff_query_family]
    tree_buf = tree_state.families[eff_tree_family]

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    pair_count = left_indices.size

    # Use device-resident arrays when provided; otherwise upload from host.
    own_d_left = eff_d_left is None
    own_d_right = eff_d_right is None
    if own_d_left:
        eff_d_left = runtime.from_host(np.ascontiguousarray(eff_left, dtype=np.int32))
    if own_d_right:
        eff_d_right = runtime.from_host(np.ascontiguousarray(eff_right, dtype=np.int32))
    d_mask = runtime.allocate((pair_count,), np.uint16)

    try:
        kernels = _polygon_predicates_kernels()
        P = KERNEL_PARAM_PTR
        I32 = KERNEL_PARAM_I32

        left_args, left_types = _build_side_args(ptr, query_state, query_buf, eff_query_family)
        right_args, right_types = _build_side_args(ptr, tree_state, tree_buf, eff_tree_family)
        tail_args = [ptr(eff_d_left), ptr(eff_d_right), ptr(d_mask), pair_count]
        tail_types = [P, P, P, I32]

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

        h_mask = np.empty(pair_count, dtype=np.uint16)
        runtime.copy_device_to_host(d_mask, h_mask)

        if swap:
            h_mask = _transpose_de9im(h_mask)

        return h_mask

    finally:
        if own_d_left:
            runtime.free(eff_d_left)
        if own_d_right:
            runtime.free(eff_d_right)
        runtime.free(d_mask)


def _transpose_de9im(masks: np.ndarray) -> np.ndarray:
    """Transpose DE-9IM bitmasks (swap A and B roles)."""
    m = masks.astype(np.uint16, copy=True)
    out = np.zeros_like(m)
    # II stays, EE stays.
    out |= (m & DE9IM_II)
    out |= (m & DE9IM_EE)
    # Swap IB ↔ BI.
    out |= np.where(m & DE9IM_IB, DE9IM_BI, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_BI, DE9IM_IB, 0).astype(np.uint16)
    # Swap IE ↔ EI.
    out |= np.where(m & DE9IM_IE, DE9IM_EI, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_EI, DE9IM_IE, 0).astype(np.uint16)
    # Swap BE ↔ EB.
    out |= np.where(m & DE9IM_BE, DE9IM_EB, 0).astype(np.uint16)
    out |= np.where(m & DE9IM_EB, DE9IM_BE, 0).astype(np.uint16)
    # BB stays.
    out |= (m & DE9IM_BB)
    return out


def supported_predicate_families() -> frozenset[tuple[GeometryFamily, GeometryFamily]]:
    """Return the set of family pairs supported by polygon predicate kernels."""
    return frozenset(_KERNEL_MAP.keys())
