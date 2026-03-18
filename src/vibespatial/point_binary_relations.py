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

POINT_LOCATION_OUTSIDE = np.uint8(0)
POINT_LOCATION_BOUNDARY = np.uint8(1)
POINT_LOCATION_INTERIOR = np.uint8(2)


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

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("point-binary-relations", _POINT_BINARY_RELATIONS_KERNEL_SOURCE, _POINT_BINARY_RELATIONS_KERNEL_NAMES),
    ("multipoint-binary-relations", _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE, _MULTIPOINT_KERNEL_NAMES),
])


def _point_binary_relation_kernels():
    return compile_kernel_group("point-binary-relations", _POINT_BINARY_RELATIONS_KERNEL_SOURCE, _POINT_BINARY_RELATIONS_KERNEL_NAMES)


def _multipoint_relation_kernels():
    return compile_kernel_group("multipoint-binary-relations", _MULTIPOINT_BINARY_RELATIONS_KERNEL_SOURCE, _MULTIPOINT_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Unified kernel launch -- replaces the three nearly-identical functions
# _launch_rows_kernel, _launch_indexed_kernel, _launch_indexed_mp_kernel.
# ---------------------------------------------------------------------------

def _launch_kernel(
    kernel_dict_fn,
    kernel_name: str,
    candidate_rows: np.ndarray,
    args: tuple[int, ...],
    arg_types: tuple[object, ...],
    *,
    extra_device_allocs: list | None = None,
) -> np.ndarray:
    """Launch a point or multipoint binary-relation kernel.

    Parameters
    ----------
    kernel_dict_fn : callable
        One of ``_point_binary_relation_kernels`` or ``_multipoint_relation_kernels``.
    kernel_name : str
        Name of the CUDA kernel to launch.
    candidate_rows : np.ndarray
        Row indices (int32) to pass as the first kernel argument.
    args : tuple
        Device pointer / scalar arguments between candidate_rows and (out, count).
    arg_types : tuple
        KERNEL_PARAM_* type tags matching *args*.
    extra_device_allocs : list or None
        Additional device allocations to free after launch (e.g. uploaded
        mapped FRO arrays).  The device_rows and device_out are always freed.
    """
    n_items = candidate_rows.size
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    device_rows = runtime.from_host(candidate_rows.astype(np.int32, copy=False))
    device_out = runtime.allocate((n_items,), np.uint8)
    try:
        kernel = kernel_dict_fn()[kernel_name]
        params = (
            (ptr(device_rows), *args, ptr(device_out), n_items),
            (KERNEL_PARAM_PTR, *arg_types, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n_items)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        runtime.synchronize()
        out = np.empty(n_items, dtype=np.uint8)
        runtime.copy_device_to_host(device_out, out)
        return out
    finally:
        runtime.free(device_rows)
        runtime.free(device_out)
        if extra_device_allocs:
            for alloc in extra_device_allocs:
                runtime.free(alloc)


# ---------------------------------------------------------------------------
# Non-indexed public API -- use candidate_rows and the owned array's
# device-side family_row_offsets directly.
# ---------------------------------------------------------------------------

def classify_point_equals_gpu(
    candidate_rows: np.ndarray,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted",
        candidate_rows,
        (
            ptr(left_state.family_row_offsets),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(right_state.family_row_offsets),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
    )


def classify_point_line_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    lines: OwnedGeometryArray,
    *,
    line_family: GeometryFamily,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state()
    line_state = lines._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(line_state.family_row_offsets),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        candidate_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
    )


def classify_point_region_gpu(
    candidate_rows: np.ndarray,
    points: OwnedGeometryArray,
    regions: OwnedGeometryArray,
    *,
    region_family: GeometryFamily,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=np.uint8)
    point_state = points._ensure_device_state()
    region_state = regions._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    args = [
        ptr(point_state.family_row_offsets),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(region_state.family_row_offsets),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        candidate_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
    )


# ---------------------------------------------------------------------------
# Indexed variants: separate left/right index arrays into original owned
# geometry arrays.  Avoids the expensive take() buffer copy by pre-gathering
# family_row_offsets on host and uploading the mapped arrays.
# ---------------------------------------------------------------------------

_POINT_TAG_INDEXED = FAMILY_TAGS[GeometryFamily.POINT]
_LINE_FAMILIES_INDEXED = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_REGION_FAMILIES_INDEXED = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_LINE_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _LINE_FAMILIES_INDEXED)
_REGION_TAGS_INDEXED = tuple(FAMILY_TAGS[f] for f in _REGION_FAMILIES_INDEXED)


def _prepare_indexed_fro(owned, indices, runtime):
    """Map indices through family_row_offsets and upload to device."""
    mapped = owned.family_row_offsets[indices].astype(np.int32, copy=False)
    return runtime.from_host(mapped)


def _classify_indexed_point_equals(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    n = left_indices.size
    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.POINT]
    right_buffer = right_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro = _prepare_indexed_fro(left_owned, left_indices, runtime)
    device_right_fro = _prepare_indexed_fro(right_owned, right_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    return _launch_kernel(
        _point_binary_relation_kernels,
        "point_equals_compacted", identity_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_left_fro, device_right_fro],
    )


def _classify_indexed_point_line(
    point_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
) -> np.ndarray:
    n = point_indices.size
    point_state = point_owned._ensure_device_state()
    line_state = line_owned._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro = _prepare_indexed_fro(point_owned, point_indices, runtime)
    device_line_fro = _prepare_indexed_fro(line_owned, line_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "point_on_linestring_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "point_on_multilinestring_compacted"
    )
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_point_fro, device_line_fro],
    )


def _classify_indexed_point_region(
    point_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    point_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
) -> np.ndarray:
    n = point_indices.size
    point_state = point_owned._ensure_device_state()
    region_state = region_owned._ensure_device_state()
    point_buffer = point_state.families[GeometryFamily.POINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_point_fro = _prepare_indexed_fro(point_owned, point_indices, runtime)
    device_region_fro = _prepare_indexed_fro(region_owned, region_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "point_in_polygon_polygon_compacted_state"
        if region_family is GeometryFamily.POLYGON
        else "point_in_polygon_multipolygon_compacted_state"
    )
    args = [
        ptr(device_point_fro),
        ptr(point_buffer.geometry_offsets),
        ptr(point_buffer.empty_mask),
        ptr(point_buffer.x),
        ptr(point_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _point_binary_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_point_fro, device_region_fro],
    )


def classify_point_predicates_indexed(
    predicate: str,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate point-family predicates using indexed access into original owned arrays.

    Avoids the expensive take() buffer copy by pre-gathering family_row_offsets
    on the host and passing them directly to existing GPU kernels.

    Returns a boolean array of length ``left_indices.size``.
    """
    from vibespatial.binary_predicates import (
        _apply_relation_rows,
        _point_equals_to_predicate,
        _point_relation_to_predicate,
    )

    n = left_indices.size
    if n == 0:
        return np.empty(0, dtype=bool)

    out = np.zeros(n, dtype=bool)
    left_tags = left_owned.tags[left_indices]
    right_tags = right_owned.tags[right_indices]

    # Point x point
    pp_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == _POINT_TAG_INDEXED)
    if pp_mask.any():
        idx = np.flatnonzero(pp_mask)
        relation = _classify_indexed_point_equals(
            left_owned, right_owned, left_indices[idx], right_indices[idx],
        )
        _apply_relation_rows(out, idx, _point_equals_to_predicate(predicate, relation))

    # Point x line and line x point
    for line_family, line_tag in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        pl_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == line_tag)
        if pl_mask.any():
            idx = np.flatnonzero(pl_mask)
            relation = _classify_indexed_point_line(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                line_family=line_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True))

        lp_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG_INDEXED)
        if lp_mask.any():
            idx = np.flatnonzero(lp_mask)
            relation = _classify_indexed_point_line(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                line_family=line_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False))

    # Point x region and region x point
    for region_family, region_tag in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        pr_mask = (left_tags == _POINT_TAG_INDEXED) & (right_tags == region_tag)
        if pr_mask.any():
            idx = np.flatnonzero(pr_mask)
            relation = _classify_indexed_point_region(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                region_family=region_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=True))

        rp_mask = (region_tag == left_tags) & (right_tags == _POINT_TAG_INDEXED)
        if rp_mask.any():
            idx = np.flatnonzero(rp_mask)
            relation = _classify_indexed_point_region(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                region_family=region_family,
            )
            _apply_relation_rows(out, idx, _point_relation_to_predicate(predicate, relation, point_on_left=False))

    # Multipoint x anything and anything x multipoint
    mp_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
    mp_left_mask = left_tags == mp_tag
    mp_right_mask = right_tags == mp_tag

    if mp_left_mask.any() or mp_right_mask.any():
        _dispatch_multipoint_pairs(
            predicate, out,
            left_owned, right_owned,
            left_indices, right_indices,
            left_tags, right_tags,
            mp_left_mask, mp_right_mask,
            _apply_relation_rows,
        )

    return out


# ---------------------------------------------------------------------------
# Multipoint support -- launch helpers, predicate conversion, and dispatch.
# Tier 1 per ADR-0033: geometry-specific inner loops (multipoint coord iteration).
# ---------------------------------------------------------------------------

# Bit flags in multipoint kernel output
_MP_ANY_OUTSIDE = np.uint8(1)
_MP_ANY_BOUNDARY = np.uint8(2)
_MP_ANY_INTERIOR = np.uint8(4)


def _multipoint_bits_to_predicate(
    predicate: str,
    bits: np.ndarray,
    *,
    mp_on_left: bool,
    target_family: GeometryFamily | None = None,
) -> np.ndarray:
    """Convert packed multipoint relation bits to boolean predicate results.

    Bits: 0x1 = any_outside, 0x2 = any_boundary, 0x4 = any_interior.
    Each bit records how the multipoint's coordinates relate to the target.

    **Key asymmetry:**  The bits tell us *"for each MP coord, its location in
    the target."*  This directly gives ``within`` / ``covered_by`` (all MP
    coords inside target) and the symmetric predicates (intersects / disjoint
    / touches).  For ``contains`` / ``covers`` / ``contains_properly`` we need
    the *reverse* -- whether the target fits inside the multipoint.

    * MP x Point ``contains``: at least one MP coord equals the point ->
      ``any_interior`` (the kernel records "equal" as interior).
    * MP x Line/Polygon ``contains``: always False (0-D cannot contain >=1-D).
    * MP x MP ``contains``: handled by the dispatch running the kernel in
      reverse and calling this function with a swapped predicate.
    """
    any_outside = (bits & _MP_ANY_OUTSIDE).astype(bool)
    any_boundary = (bits & _MP_ANY_BOUNDARY).astype(bool)
    any_interior = (bits & _MP_ANY_INTERIOR).astype(bool)
    any_hit = any_boundary | any_interior
    n = bits.shape[0]

    # --- Symmetric predicates ---
    if predicate == "intersects":
        return any_hit
    if predicate == "disjoint":
        return ~any_hit
    if predicate == "touches":
        return any_boundary & ~any_interior

    # --- within / covered_by: is the MP inside the target? ---
    # Condition: every MP coord must be inside (or on boundary of) the target.
    if mp_on_left:
        if predicate == "within":
            return any_interior & ~any_outside
        if predicate == "covered_by":
            return any_hit & ~any_outside

        # contains / covers / contains_properly: is the target inside the MP?
        if predicate in {"contains", "covers", "contains_properly"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                # MP contains point iff point matches at least one MP coord.
                # For MPxMP, the dispatch handles the reverse check.
                return any_interior
            # MP can't contain a line or polygon -- 0-D vs >=1-D.
            return np.zeros(n, dtype=bool)

    else:
        # MP is on the right (tree side), target on left.

        # contains / covers: does the target contain every MP coord?
        if predicate == "contains":
            return any_interior & ~any_outside
        if predicate == "covers":
            return any_hit & ~any_outside
        if predicate == "contains_properly":
            return any_interior & ~any_outside

        # within / covered_by: is the target within the MP?
        if predicate in {"within", "covered_by"}:
            tf = target_family
            if tf is GeometryFamily.POINT or tf is GeometryFamily.MULTIPOINT:
                return any_interior
            return np.zeros(n, dtype=bool)

    return np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# Indexed multipoint classify functions
# ---------------------------------------------------------------------------

def _classify_indexed_mp_point(
    mp_owned: OwnedGeometryArray,
    pt_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    pt_indices: np.ndarray,
) -> np.ndarray:
    """MULTIPOINT x POINT relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    pt_state = pt_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    pt_buffer = pt_state.families[GeometryFamily.POINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_pt_fro = _prepare_indexed_fro(pt_owned, pt_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_point_relation_compacted", identity_rows,
        (
            ptr(device_mp_fro),
            ptr(mp_buffer.geometry_offsets),
            ptr(mp_buffer.empty_mask),
            ptr(mp_buffer.x),
            ptr(mp_buffer.y),
            ptr(device_pt_fro),
            ptr(pt_buffer.geometry_offsets),
            ptr(pt_buffer.empty_mask),
            ptr(pt_buffer.x),
            ptr(pt_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_mp_fro, device_pt_fro],
    )


def _classify_indexed_mp_line(
    mp_owned: OwnedGeometryArray,
    line_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    line_indices: np.ndarray,
    *,
    line_family: GeometryFamily,
) -> np.ndarray:
    """MULTIPOINT x LINESTRING/MULTILINESTRING relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    line_state = line_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    line_buffer = line_state.families[line_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_line_fro = _prepare_indexed_fro(line_owned, line_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "multipoint_linestring_relation_compacted"
        if line_family is GeometryFamily.LINESTRING
        else "multipoint_multilinestring_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_line_fro),
        ptr(line_buffer.geometry_offsets),
    ]
    if line_family is not GeometryFamily.LINESTRING:
        args.append(ptr(line_buffer.part_offsets))
    args.extend([
        ptr(line_buffer.empty_mask),
        ptr(line_buffer.x),
        ptr(line_buffer.y),
    ])
    return _launch_kernel(
        _multipoint_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_mp_fro, device_line_fro],
    )


def _classify_indexed_mp_region(
    mp_owned: OwnedGeometryArray,
    region_owned: OwnedGeometryArray,
    mp_indices: np.ndarray,
    region_indices: np.ndarray,
    *,
    region_family: GeometryFamily,
) -> np.ndarray:
    """MULTIPOINT x POLYGON/MULTIPOLYGON relation bits."""
    n = mp_indices.size
    mp_state = mp_owned._ensure_device_state()
    region_state = region_owned._ensure_device_state()
    mp_buffer = mp_state.families[GeometryFamily.MULTIPOINT]
    region_buffer = region_state.families[region_family]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_mp_fro = _prepare_indexed_fro(mp_owned, mp_indices, runtime)
    device_region_fro = _prepare_indexed_fro(region_owned, region_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    kernel_name = (
        "multipoint_polygon_relation_compacted"
        if region_family is GeometryFamily.POLYGON
        else "multipoint_multipolygon_relation_compacted"
    )
    args = [
        ptr(device_mp_fro),
        ptr(mp_buffer.geometry_offsets),
        ptr(mp_buffer.empty_mask),
        ptr(mp_buffer.x),
        ptr(mp_buffer.y),
        ptr(device_region_fro),
        ptr(region_buffer.empty_mask),
        ptr(region_buffer.geometry_offsets),
    ]
    if region_family is not GeometryFamily.POLYGON:
        args.append(ptr(region_buffer.part_offsets))
    args.extend([
        ptr(region_buffer.ring_offsets),
        ptr(region_buffer.x),
        ptr(region_buffer.y),
    ])
    return _launch_kernel(
        _multipoint_relation_kernels, kernel_name,
        identity_rows, tuple(args), (KERNEL_PARAM_PTR,) * len(args),
        extra_device_allocs=[device_mp_fro, device_region_fro],
    )


def _classify_indexed_mp_mp(
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    """MULTIPOINT x MULTIPOINT relation bits (left MP vs right MP)."""
    n = left_indices.size
    left_state = left_owned._ensure_device_state()
    right_state = right_owned._ensure_device_state()
    left_buffer = left_state.families[GeometryFamily.MULTIPOINT]
    right_buffer = right_state.families[GeometryFamily.MULTIPOINT]
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    device_left_fro = _prepare_indexed_fro(left_owned, left_indices, runtime)
    device_right_fro = _prepare_indexed_fro(right_owned, right_indices, runtime)
    identity_rows = np.arange(n, dtype=np.int32)
    return _launch_kernel(
        _multipoint_relation_kernels,
        "multipoint_multipoint_relation_compacted", identity_rows,
        (
            ptr(device_left_fro),
            ptr(left_buffer.geometry_offsets),
            ptr(left_buffer.empty_mask),
            ptr(left_buffer.x),
            ptr(left_buffer.y),
            ptr(device_right_fro),
            ptr(right_buffer.geometry_offsets),
            ptr(right_buffer.empty_mask),
            ptr(right_buffer.x),
            ptr(right_buffer.y),
        ),
        (KERNEL_PARAM_PTR,) * 10,
        extra_device_allocs=[device_left_fro, device_right_fro],
    )


def _dispatch_multipoint_pairs(
    predicate: str,
    out: np.ndarray,
    left_owned: OwnedGeometryArray,
    right_owned: OwnedGeometryArray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    left_tags: np.ndarray,
    right_tags: np.ndarray,
    mp_left_mask: np.ndarray,
    mp_right_mask: np.ndarray,
    _apply_relation_rows,
) -> None:
    """Dispatch multipoint pairs to the appropriate GPU kernel and convert results."""
    pt_tag = _POINT_TAG_INDEXED

    # --- MULTIPOINT on left ---

    # MP x point
    mask = mp_left_mask & (right_tags == pt_tag)
    if mask.any():
        idx = np.flatnonzero(mask)
        bits = _classify_indexed_mp_point(
            left_owned, right_owned, left_indices[idx], right_indices[idx],
        )
        _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
            predicate, bits, mp_on_left=True, target_family=GeometryFamily.POINT,
        ))

    # MP x line families
    for lf, lt in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        mask = mp_left_mask & (right_tags == lt)
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_line(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                line_family=lf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=True, target_family=lf,
            ))

    # MP x region families
    for rf, rt in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        mask = mp_left_mask & (right_tags == rt)
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_region(
                left_owned, right_owned, left_indices[idx], right_indices[idx],
                region_family=rf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=True, target_family=rf,
            ))

    # MP x MP: the kernel checks each LEFT-MP coord against RIGHT-MP.
    # For contains/covers/contains_properly we also need the reverse check
    # (each RIGHT-MP coord against LEFT-MP) to verify the right side is a
    # subset of the left side.
    mask = mp_left_mask & mp_right_mask
    if mask.any():
        idx = np.flatnonzero(mask)
        li, ri = left_indices[idx], right_indices[idx]
        bits_fwd = _classify_indexed_mp_mp(left_owned, right_owned, li, ri)

        if predicate in {"contains", "covers", "contains_properly"}:
            # Reverse: check each right-MP coord against left-MP.
            bits_rev = _classify_indexed_mp_mp(right_owned, left_owned, ri, li)
            # "contains" = right subset of left = ~any_outside in reverse
            result = _multipoint_bits_to_predicate(
                predicate, bits_rev, mp_on_left=False,
                target_family=GeometryFamily.MULTIPOINT,
            )
        elif predicate in {"within", "covered_by"}:
            # Forward bits already tell us if left is subset of right.
            result = _multipoint_bits_to_predicate(
                predicate, bits_fwd, mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        else:
            # Symmetric predicates -- forward bits suffice.
            result = _multipoint_bits_to_predicate(
                predicate, bits_fwd, mp_on_left=True,
                target_family=GeometryFamily.MULTIPOINT,
            )
        _apply_relation_rows(out, idx, result)

    # --- MULTIPOINT on right only (left is not multipoint) ---

    # point x MP
    mask = (left_tags == pt_tag) & mp_right_mask
    if mask.any():
        idx = np.flatnonzero(mask)
        bits = _classify_indexed_mp_point(
            right_owned, left_owned, right_indices[idx], left_indices[idx],
        )
        _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
            predicate, bits, mp_on_left=False, target_family=GeometryFamily.POINT,
        ))

    # line x MP
    for lf, lt in zip(_LINE_FAMILIES_INDEXED, _LINE_TAGS_INDEXED, strict=True):
        mask = (left_tags == lt) & mp_right_mask
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_line(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                line_family=lf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=False, target_family=lf,
            ))

    # region x MP
    for rf, rt in zip(_REGION_FAMILIES_INDEXED, _REGION_TAGS_INDEXED, strict=True):
        mask = (left_tags == rt) & mp_right_mask
        if mask.any():
            idx = np.flatnonzero(mask)
            bits = _classify_indexed_mp_region(
                right_owned, left_owned, right_indices[idx], left_indices[idx],
                region_family=rf,
            )
            _apply_relation_rows(out, idx, _multipoint_bits_to_predicate(
                predicate, bits, mp_on_left=False, target_family=rf,
            ))
