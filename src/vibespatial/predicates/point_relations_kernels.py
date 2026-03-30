"""NVRTC kernel sources for point binary relations and multipoint relations."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CUDA kernel source -- shared device helpers
# ---------------------------------------------------------------------------
# These helper functions (vibespatial_abs, point_on_segment_kind,
# ring_contains_even_odd_kind, polygon_point_location,
# multipolygon_point_location) are used by both single-point and multipoint
# kernels.  Previously they were duplicated across two kernel source strings
# (~140 lines each).  We now define them once and compose via concatenation.

_SHARED_DEVICE_HELPERS = r"""
extern "C" __device__ inline double vibespatial_abs(double value) {
  return value < 0.0 ? -value : value;
}

extern "C" __device__ inline unsigned char point_on_segment_kind(
    double px,
    double py,
    double ax,
    double ay,
    double bx,
    double by
) {
  const double dx = bx - ax;
  const double dy = by - ay;
  const double cross = ((px - ax) * dy) - ((py - ay) * dx);
  const double scale = vibespatial_abs(dx) + vibespatial_abs(dy) + 1.0;
  if (vibespatial_abs(cross) > (1e-12 * scale)) {
    return 0;
  }
  const double minx = ax < bx ? ax : bx;
  const double maxx = ax > bx ? ax : bx;
  const double miny = ay < by ? ay : by;
  const double maxy = ay > by ? ay : by;
  if (px < (minx - 1e-12) || px > (maxx + 1e-12) || py < (miny - 1e-12) || py > (maxy + 1e-12)) {
    return 0;
  }
  const bool endpoint =
      (vibespatial_abs(px - ax) <= 1e-12 && vibespatial_abs(py - ay) <= 1e-12) ||
      (vibespatial_abs(px - bx) <= 1e-12 && vibespatial_abs(py - by) <= 1e-12);
  return endpoint ? 1 : 2;
}

extern "C" __device__ inline unsigned char ring_contains_even_odd_kind(
    double px,
    double py,
    const double* x,
    const double* y,
    int coord_start,
    int coord_end
) {
  bool inside = false;
  if ((coord_end - coord_start) < 2) {
    return 0;
  }
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {
    const double ax = x[coord - 1];
    const double ay = y[coord - 1];
    const double bx = x[coord];
    const double by = y[coord];
    const unsigned char segment_kind = point_on_segment_kind(px, py, ax, ay, bx, by);
    if (segment_kind != 0) {
      return 1;
    }
    const bool intersects = ((ay > py) != (by > py)) &&
        (px <= (((bx - ax) * (py - ay)) / ((by - ay) + 0.0)) + ax);
    if (intersects) {
      inside = !inside;
    }
  }
  return inside ? 2 : 0;
}

extern "C" __device__ inline unsigned char polygon_point_location(
    double px,
    double py,
    const double* x,
    const double* y,
    const int* geometry_offsets,
    const int* ring_offsets,
    int polygon_row
) {
  const int ring_start = geometry_offsets[polygon_row];
  const int ring_end = geometry_offsets[polygon_row + 1];
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int coord_start = ring_offsets[ring];
    const int coord_end = ring_offsets[ring + 1];
    const unsigned char ring_kind = ring_contains_even_odd_kind(px, py, x, y, coord_start, coord_end);
    if (ring_kind == 1) {
      return 1;
    }
    if (ring_kind == 2) {
      inside = !inside;
    }
  }
  return inside ? 2 : 0;
}

extern "C" __device__ inline unsigned char multipolygon_point_location(
    double px,
    double py,
    const double* x,
    const double* y,
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    int multipolygon_row
) {
  const int polygon_start = geometry_offsets[multipolygon_row];
  const int polygon_end = geometry_offsets[multipolygon_row + 1];
  for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
    bool inside = false;
    const int ring_start = part_offsets[polygon];
    const int ring_end = part_offsets[polygon + 1];
    for (int ring = ring_start; ring < ring_end; ++ring) {
      const int coord_start = ring_offsets[ring];
      const int coord_end = ring_offsets[ring + 1];
      const unsigned char ring_kind = ring_contains_even_odd_kind(px, py, x, y, coord_start, coord_end);
      if (ring_kind == 1) {
        return 1;
      }
      if (ring_kind == 2) {
        inside = !inside;
      }
    }
    if (inside) {
      return 2;
    }
  }
  return 0;
}
"""

# ---------------------------------------------------------------------------
# Single-point kernel __global__ functions (appended after shared helpers)
# ---------------------------------------------------------------------------

_POINT_KERNEL_GLOBALS = r"""
extern "C" __global__ void point_equals_compacted(
    const int* candidate_rows,
    const int* left_row_offsets,
    const int* left_geometry_offsets,
    const unsigned char* left_empty_mask,
    const double* left_x,
    const double* left_y,
    const int* right_row_offsets,
    const int* right_geometry_offsets,
    const unsigned char* right_empty_mask,
    const double* right_x,
    const double* right_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const int row = candidate_rows[index];
  const int left_row = left_row_offsets[row];
  const int right_row = right_row_offsets[row];
  if (left_row < 0 || right_row < 0 || left_empty_mask[left_row] || right_empty_mask[right_row]) {
    out[index] = 0;
    return;
  }
  const int left_coord = left_geometry_offsets[left_row];
  const int right_coord = right_geometry_offsets[right_row];
  const bool same =
      vibespatial_abs(left_x[left_coord] - right_x[right_coord]) <= 1e-12 &&
      vibespatial_abs(left_y[left_coord] - right_y[right_coord]) <= 1e-12;
  out[index] = same ? 2 : 0;
}

extern "C" __global__ void point_on_linestring_compacted(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* line_row_offsets,
    const int* line_geometry_offsets,
    const unsigned char* line_empty_mask,
    const double* line_x,
    const double* line_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int line_row = line_row_offsets[row];
  if (point_row < 0 || line_row < 0 || point_empty_mask[point_row] || line_empty_mask[line_row]) {
    out[index] = 0;
    return;
  }
  const int point_coord = point_geometry_offsets[point_row];
  const double px = point_x[point_coord];
  const double py = point_y[point_coord];
  const int coord_start = line_geometry_offsets[line_row];
  const int coord_end = line_geometry_offsets[line_row + 1];
  unsigned char best = 0;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {
    const unsigned char kind = point_on_segment_kind(
        px,
        py,
        line_x[coord - 1],
        line_y[coord - 1],
        line_x[coord],
        line_y[coord]
    );
    if (kind == 2) {
      out[index] = 2;
      return;
    }
    if (kind == 1) {
      best = 1;
    }
  }
  out[index] = best;
}

extern "C" __global__ void point_on_multilinestring_compacted(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* line_row_offsets,
    const int* line_geometry_offsets,
    const int* line_part_offsets,
    const unsigned char* line_empty_mask,
    const double* line_x,
    const double* line_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int line_row = line_row_offsets[row];
  if (point_row < 0 || line_row < 0 || point_empty_mask[point_row] || line_empty_mask[line_row]) {
    out[index] = 0;
    return;
  }
  const int point_coord = point_geometry_offsets[point_row];
  const double px = point_x[point_coord];
  const double py = point_y[point_coord];
  const int part_start = line_geometry_offsets[line_row];
  const int part_end = line_geometry_offsets[line_row + 1];
  unsigned char best = 0;
  for (int part = part_start; part < part_end; ++part) {
    const int coord_start = line_part_offsets[part];
    const int coord_end = line_part_offsets[part + 1];
    for (int coord = coord_start + 1; coord < coord_end; ++coord) {
      const unsigned char kind = point_on_segment_kind(
          px,
          py,
          line_x[coord - 1],
          line_y[coord - 1],
          line_x[coord],
          line_y[coord]
      );
      if (kind == 2) {
        out[index] = 2;
        return;
      }
      if (kind == 1) {
        best = 1;
      }
    }
  }
  out[index] = best;
}

extern "C" __global__ void point_in_polygon_polygon_compacted_state(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int polygon_row = polygon_row_offsets[row];
  if (point_row < 0 || polygon_row < 0 || point_empty_mask[point_row] || polygon_empty_mask[polygon_row]) {
    out[index] = 0;
    return;
  }
  const int point_coord = point_geometry_offsets[point_row];
  out[index] = polygon_point_location(
      point_x[point_coord],
      point_y[point_coord],
      polygon_x,
      polygon_y,
      polygon_geometry_offsets,
      polygon_ring_offsets,
      polygon_row
  );
}

extern "C" __global__ void point_in_polygon_multipolygon_compacted_state(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_part_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {
    return;
  }
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int polygon_row = polygon_row_offsets[row];
  if (point_row < 0 || polygon_row < 0 || point_empty_mask[point_row] || polygon_empty_mask[polygon_row]) {
    out[index] = 0;
    return;
  }
  const int point_coord = point_geometry_offsets[point_row];
  out[index] = multipolygon_point_location(
      point_x[point_coord],
      point_y[point_coord],
      polygon_x,
      polygon_y,
      polygon_geometry_offsets,
      polygon_part_offsets,
      polygon_ring_offsets,
      polygon_row
  );
}
"""

# ---------------------------------------------------------------------------
# Multipoint kernel __global__ functions -- iterate over multipoint
# coordinates and aggregate per-point location codes into packed bit flags
# per candidate pair.
#
# Output bits: bit 0 = any_outside, bit 1 = any_boundary, bit 2 = any_interior
# Tier 1 per ADR-0033: geometry-specific inner loops.
# ---------------------------------------------------------------------------

_MULTIPOINT_KERNEL_GLOBALS = r"""
/* Pack a location code (0/1/2) into the output bits. */
extern "C" __device__ inline unsigned char loc_to_bit(unsigned char loc) {
  if (loc == 0) return 1;  /* any_outside */
  if (loc == 1) return 2;  /* any_boundary */
  return 4;                /* any_interior */
}

/* MULTIPOINT x POINT -- check if any MP coordinate equals the target point. */
extern "C" __global__ void multipoint_point_relation_compacted(
    const int* candidate_rows,
    const int* mp_row_offsets,
    const int* mp_geometry_offsets,
    const unsigned char* mp_empty_mask,
    const double* mp_x,
    const double* mp_y,
    const int* pt_row_offsets,
    const int* pt_geometry_offsets,
    const unsigned char* pt_empty_mask,
    const double* pt_x,
    const double* pt_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int mp_row = mp_row_offsets[row];
  const int pt_row = pt_row_offsets[row];
  if (mp_row < 0 || pt_row < 0 || mp_empty_mask[mp_row] || pt_empty_mask[pt_row]) {
    out[index] = 0; return;
  }
  const int pt_coord = pt_geometry_offsets[pt_row];
  const double px = pt_x[pt_coord];
  const double py = pt_y[pt_coord];
  const int start = mp_geometry_offsets[mp_row];
  const int end = mp_geometry_offsets[mp_row + 1];
  unsigned char bits = 0;
  for (int c = start; c < end; ++c) {
    const bool same = vibespatial_abs(mp_x[c] - px) <= 1e-12 &&
                      vibespatial_abs(mp_y[c] - py) <= 1e-12;
    bits |= same ? 4 : 1;
  }
  out[index] = bits;
}

/* MULTIPOINT x LINESTRING */
extern "C" __global__ void multipoint_linestring_relation_compacted(
    const int* candidate_rows,
    const int* mp_row_offsets,
    const int* mp_geometry_offsets,
    const unsigned char* mp_empty_mask,
    const double* mp_x,
    const double* mp_y,
    const int* line_row_offsets,
    const int* line_geometry_offsets,
    const unsigned char* line_empty_mask,
    const double* line_x,
    const double* line_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int mp_row = mp_row_offsets[row];
  const int line_row = line_row_offsets[row];
  if (mp_row < 0 || line_row < 0 || mp_empty_mask[mp_row] || line_empty_mask[line_row]) {
    out[index] = 0; return;
  }
  const int mp_start = mp_geometry_offsets[mp_row];
  const int mp_end = mp_geometry_offsets[mp_row + 1];
  const int ls = line_geometry_offsets[line_row];
  const int le = line_geometry_offsets[line_row + 1];
  unsigned char bits = 0;
  for (int c = mp_start; c < mp_end; ++c) {
    const double px = mp_x[c];
    const double py = mp_y[c];
    unsigned char best = 0;
    for (int s = ls + 1; s < le; ++s) {
      const unsigned char kind = point_on_segment_kind(px, py, line_x[s - 1], line_y[s - 1], line_x[s], line_y[s]);
      if (kind == 2) { best = 2; break; }
      if (kind == 1) best = 1;
    }
    bits |= loc_to_bit(best);
  }
  out[index] = bits;
}

/* MULTIPOINT x MULTILINESTRING */
extern "C" __global__ void multipoint_multilinestring_relation_compacted(
    const int* candidate_rows,
    const int* mp_row_offsets,
    const int* mp_geometry_offsets,
    const unsigned char* mp_empty_mask,
    const double* mp_x,
    const double* mp_y,
    const int* mls_row_offsets,
    const int* mls_geometry_offsets,
    const int* mls_part_offsets,
    const unsigned char* mls_empty_mask,
    const double* mls_x,
    const double* mls_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int mp_row = mp_row_offsets[row];
  const int mls_row = mls_row_offsets[row];
  if (mp_row < 0 || mls_row < 0 || mp_empty_mask[mp_row] || mls_empty_mask[mls_row]) {
    out[index] = 0; return;
  }
  const int mp_start = mp_geometry_offsets[mp_row];
  const int mp_end = mp_geometry_offsets[mp_row + 1];
  const int part_start = mls_geometry_offsets[mls_row];
  const int part_end = mls_geometry_offsets[mls_row + 1];
  unsigned char bits = 0;
  for (int c = mp_start; c < mp_end; ++c) {
    const double px = mp_x[c];
    const double py = mp_y[c];
    unsigned char best = 0;
    for (int part = part_start; part < part_end; ++part) {
      const int cs = mls_part_offsets[part];
      const int ce = mls_part_offsets[part + 1];
      for (int s = cs + 1; s < ce; ++s) {
        const unsigned char kind = point_on_segment_kind(px, py, mls_x[s - 1], mls_y[s - 1], mls_x[s], mls_y[s]);
        if (kind == 2) { best = 2; break; }
        if (kind == 1) best = 1;
      }
      if (best == 2) break;
    }
    bits |= loc_to_bit(best);
  }
  out[index] = bits;
}

/* MULTIPOINT x POLYGON */
extern "C" __global__ void multipoint_polygon_relation_compacted(
    const int* candidate_rows,
    const int* mp_row_offsets,
    const int* mp_geometry_offsets,
    const unsigned char* mp_empty_mask,
    const double* mp_x,
    const double* mp_y,
    const int* pg_row_offsets,
    const unsigned char* pg_empty_mask,
    const int* pg_geometry_offsets,
    const int* pg_ring_offsets,
    const double* pg_x,
    const double* pg_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int mp_row = mp_row_offsets[row];
  const int pg_row = pg_row_offsets[row];
  if (mp_row < 0 || pg_row < 0 || mp_empty_mask[mp_row] || pg_empty_mask[pg_row]) {
    out[index] = 0; return;
  }
  const int mp_start = mp_geometry_offsets[mp_row];
  const int mp_end = mp_geometry_offsets[mp_row + 1];
  unsigned char bits = 0;
  for (int c = mp_start; c < mp_end; ++c) {
    bits |= loc_to_bit(polygon_point_location(mp_x[c], mp_y[c], pg_x, pg_y, pg_geometry_offsets, pg_ring_offsets, pg_row));
  }
  out[index] = bits;
}

/* MULTIPOINT x MULTIPOLYGON */
extern "C" __global__ void multipoint_multipolygon_relation_compacted(
    const int* candidate_rows,
    const int* mp_row_offsets,
    const int* mp_geometry_offsets,
    const unsigned char* mp_empty_mask,
    const double* mp_x,
    const double* mp_y,
    const int* mpg_row_offsets,
    const unsigned char* mpg_empty_mask,
    const int* mpg_geometry_offsets,
    const int* mpg_part_offsets,
    const int* mpg_ring_offsets,
    const double* mpg_x,
    const double* mpg_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int mp_row = mp_row_offsets[row];
  const int mpg_row = mpg_row_offsets[row];
  if (mp_row < 0 || mpg_row < 0 || mp_empty_mask[mp_row] || mpg_empty_mask[mpg_row]) {
    out[index] = 0; return;
  }
  const int mp_start = mp_geometry_offsets[mp_row];
  const int mp_end = mp_geometry_offsets[mp_row + 1];
  unsigned char bits = 0;
  for (int c = mp_start; c < mp_end; ++c) {
    bits |= loc_to_bit(multipolygon_point_location(mp_x[c], mp_y[c], mpg_x, mpg_y, mpg_geometry_offsets, mpg_part_offsets, mpg_ring_offsets, mpg_row));
  }
  out[index] = bits;
}

/* MULTIPOINT x MULTIPOINT -- check pairwise coordinate matches. */
extern "C" __global__ void multipoint_multipoint_relation_compacted(
    const int* candidate_rows,
    const int* left_row_offsets,
    const int* left_geometry_offsets,
    const unsigned char* left_empty_mask,
    const double* left_x,
    const double* left_y,
    const int* right_row_offsets,
    const int* right_geometry_offsets,
    const unsigned char* right_empty_mask,
    const double* right_x,
    const double* right_y,
    unsigned char* out,
    int candidate_count
) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const int row = candidate_rows[index];
  const int lr = left_row_offsets[row];
  const int rr = right_row_offsets[row];
  if (lr < 0 || rr < 0 || left_empty_mask[lr] || right_empty_mask[rr]) {
    out[index] = 0; return;
  }
  const int l_start = left_geometry_offsets[lr];
  const int l_end = left_geometry_offsets[lr + 1];
  const int r_start = right_geometry_offsets[rr];
  const int r_end = right_geometry_offsets[rr + 1];
  /* For each left point, check if it matches any right point. */
  unsigned char bits = 0;
  for (int lc = l_start; lc < l_end; ++lc) {
    const double lx = left_x[lc];
    const double ly = left_y[lc];
    bool matched = false;
    for (int rc = r_start; rc < r_end; ++rc) {
      if (vibespatial_abs(lx - right_x[rc]) <= 1e-12 &&
          vibespatial_abs(ly - right_y[rc]) <= 1e-12) {
        matched = true;
        break;
      }
    }
    bits |= matched ? 4 : 1;  /* interior (matched) or outside */
  }
  out[index] = bits;
}
"""

# ---------------------------------------------------------------------------
# Compose full kernel source strings from shared helpers + kernel globals.
# This eliminates the previous duplication of ~140 lines of identical device
# helper code that was copy-pasted into both kernel source strings.
# ---------------------------------------------------------------------------

_POINT_BINARY_RELATIONS_KERNEL_SOURCE = _SHARED_DEVICE_HELPERS + _POINT_KERNEL_GLOBALS
_MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE = _SHARED_DEVICE_HELPERS + _MULTIPOINT_KERNEL_GLOBALS

_MULTIPOINT_KERNEL_NAMES = (
    "multipoint_point_relation_compacted",
    "multipoint_linestring_relation_compacted",
    "multipoint_multilinestring_relation_compacted",
    "multipoint_polygon_relation_compacted",
    "multipoint_multipolygon_relation_compacted",
    "multipoint_multipoint_relation_compacted",
)

_POINT_BINARY_RELATIONS_KERNEL_NAMES = (
    "point_equals_compacted",
    "point_on_linestring_compacted",
    "point_on_multilinestring_compacted",
    "point_in_polygon_polygon_compacted_state",
    "point_in_polygon_multipolygon_compacted_state",
)

