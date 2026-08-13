"""NVRTC kernel sources for point-to-geometry distance computation."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

_POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE = (
    PRECISION_PREAMBLE
    + """

#if !defined(INFINITY)
#define INFINITY __longlong_as_double(0x7FF0000000000000LL)
#endif

// ---------------------------------------------------------------------------
// Tier 1 NVRTC: point-to-segment squared distance (device helper)
// ---------------------------------------------------------------------------
extern "C" __device__ inline compute_t point_segment_sq_distance(
    compute_t px, compute_t py,
    compute_t ax, compute_t ay,
    compute_t bx, compute_t by
) {{
  const compute_t dx = bx - ax;
  const compute_t dy = by - ay;
  const compute_t len_sq = dx * dx + dy * dy;
  compute_t t;
  if (len_sq < (compute_t)1e-30) {{
    t = (compute_t)0.0;
  }} else {{
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
    if (t < (compute_t)0.0) t = (compute_t)0.0;
    else if (t > (compute_t)1.0) t = (compute_t)1.0;
  }}
  const compute_t cx = ax + t * dx;
  const compute_t cy = ay + t * dy;
  const compute_t ex = px - cx;
  const compute_t ey = py - cy;
  return ex * ex + ey * ey;
}}

// ---------------------------------------------------------------------------
// Tier 1 NVRTC: min squared distance from a point to a coordinate range
// ---------------------------------------------------------------------------
extern "C" __device__ inline compute_t point_coords_min_sq_distance(
    compute_t px, compute_t py,
    const double* __restrict__ x, const double* __restrict__ y,
    double center_x, double center_y,
    int coord_start, int coord_end
) {{
  compute_t best = (compute_t)INFINITY;
  for (int c = coord_start + 1; c < coord_end; ++c) {{
    const compute_t d = point_segment_sq_distance(
        px, py, CX(x[c - 1]), CY(y[c - 1]), CX(x[c]), CY(y[c]));
    if (d < best) best = d;
    // Early exit: point is ON this edge -- distance can't improve.
    if (best <= (compute_t)0.0) return best;
  }}
  return best;
}}

// ---------------------------------------------------------------------------
// Winding-number point-in-polygon test (even-odd rule).
// This test uses centered coordinates for consistency but the boolean
// result is not precision-sensitive for well-separated geometries.
// ---------------------------------------------------------------------------
extern "C" __device__ inline bool point_inside_polygon(
    compute_t px, compute_t py,
    const double* __restrict__ x, const double* __restrict__ y,
    double center_x, double center_y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ ring_offsets,
    int polygon_row
) {{
  const int ring_start = geometry_offsets[polygon_row];
  const int ring_end   = geometry_offsets[polygon_row + 1];
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];
    if ((ce - cs) < 2) continue;
    for (int c = cs + 1; c < ce; ++c) {{
      const compute_t ax = CX(x[c - 1]), ay = CY(y[c - 1]);
      const compute_t bx = CX(x[c]),     by = CY(y[c]);
      const compute_t cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
      if (cross_val == (compute_t)0.0) {{
        const compute_t minx = ax < bx ? ax : bx;
        const compute_t maxx = ax > bx ? ax : bx;
        const compute_t miny = ay < by ? ay : by;
        const compute_t maxy = ay > by ? ay : by;
        if (px >= minx && px <= maxx && py >= miny && py <= maxy) {{
          return true;
        }}
      }}
      if (((ay > py) != (by > py)) &&
          (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax)) {{
        inside = !inside;
      }}
    }}
  }}
  return inside;
}}

// ---------------------------------------------------------------------------
// Tier 1 NVRTC kernels: point distance to linestring / polygon families
// ---------------------------------------------------------------------------

extern "C" __global__ __launch_bounds__(256, 4) void point_linestring_distance_from_owned(
    const unsigned char* __restrict__ query_validity,
    const signed char*   __restrict__ query_tags,
    const int*           __restrict__ query_family_row_offsets,
    const int*           __restrict__ query_geometry_offsets,
    const unsigned char* __restrict__ query_empty_mask,
    const double*        __restrict__ query_x,
    const double*        __restrict__ query_y,
    int                  query_point_tag,
    const unsigned char* __restrict__ tree_validity,
    const signed char*   __restrict__ tree_tags,
    const int*           __restrict__ tree_family_row_offsets,
    const int*           __restrict__ tree_geometry_offsets,
    const unsigned char* __restrict__ tree_empty_mask,
    const double*        __restrict__ tree_x,
    const double*        __restrict__ tree_y,
    int                  tree_line_tag,
    const int*           __restrict__ left_idx,
    const int*           __restrict__ right_idx,
    double*              __restrict__ out_distances,
    int                  exclusive,
    const long long*     __restrict__ logical_count,
    int                  pair_capacity,
    const double*        __restrict__ center
) {{
  const double center_x = center[0], center_y = center[1];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_capacity || (logical_count != 0 && i >= logical_count[0])) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_line_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int coord_start = tree_geometry_offsets[trow];
  const int coord_end   = tree_geometry_offsets[trow + 1];

  const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                     center_x, center_y,
                                                     coord_start, coord_end);
  out_distances[i] = (double)sqrt((double)sq);
}}

extern "C" __global__ __launch_bounds__(256, 4) void point_multilinestring_distance_from_owned(
    const unsigned char* __restrict__ query_validity,
    const signed char*   __restrict__ query_tags,
    const int*           __restrict__ query_family_row_offsets,
    const int*           __restrict__ query_geometry_offsets,
    const unsigned char* __restrict__ query_empty_mask,
    const double*        __restrict__ query_x,
    const double*        __restrict__ query_y,
    int                  query_point_tag,
    const unsigned char* __restrict__ tree_validity,
    const signed char*   __restrict__ tree_tags,
    const int*           __restrict__ tree_family_row_offsets,
    const int*           __restrict__ tree_geometry_offsets,
    const int*           __restrict__ tree_part_offsets,
    const unsigned char* __restrict__ tree_empty_mask,
    const double*        __restrict__ tree_x,
    const double*        __restrict__ tree_y,
    int                  tree_multiline_tag,
    const int*           __restrict__ left_idx,
    const int*           __restrict__ right_idx,
    double*              __restrict__ out_distances,
    int                  exclusive,
    const long long*     __restrict__ logical_count,
    int                  pair_capacity,
    const double*        __restrict__ center
) {{
  const double center_x = center[0], center_y = center[1];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_capacity || (logical_count != 0 && i >= logical_count[0])) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_multiline_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int part_start = tree_geometry_offsets[trow];
  const int part_end   = tree_geometry_offsets[trow + 1];

  compute_t best = (compute_t)INFINITY;
  for (int part = part_start; part < part_end; ++part) {{
    const int cs = tree_part_offsets[part];
    const int ce = tree_part_offsets[part + 1];
    const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                       center_x, center_y, cs, ce);
    if (sq < best) best = sq;
    if (best <= (compute_t)0.0) break;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}

extern "C" __global__ __launch_bounds__(256, 4) void point_polygon_distance_from_owned(
    const unsigned char* __restrict__ query_validity,
    const signed char*   __restrict__ query_tags,
    const int*           __restrict__ query_family_row_offsets,
    const int*           __restrict__ query_geometry_offsets,
    const unsigned char* __restrict__ query_empty_mask,
    const double*        __restrict__ query_x,
    const double*        __restrict__ query_y,
    int                  query_point_tag,
    const unsigned char* __restrict__ tree_validity,
    const signed char*   __restrict__ tree_tags,
    const int*           __restrict__ tree_family_row_offsets,
    const int*           __restrict__ tree_polygon_geometry_offsets,
    const int*           __restrict__ tree_ring_offsets,
    const unsigned char* __restrict__ tree_empty_mask,
    const double*        __restrict__ tree_x,
    const double*        __restrict__ tree_y,
    int                  tree_polygon_tag,
    const int*           __restrict__ left_idx,
    const int*           __restrict__ right_idx,
    double*              __restrict__ out_distances,
    int                  exclusive,
    const long long*     __restrict__ logical_count,
    int                  pair_capacity,
    const double*        __restrict__ center
) {{
  const double center_x = center[0], center_y = center[1];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_capacity || (logical_count != 0 && i >= logical_count[0])) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_polygon_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  if (point_inside_polygon(px, py, tree_x, tree_y, center_x, center_y,
                            tree_polygon_geometry_offsets, tree_ring_offsets, trow)) {{
    out_distances[i] = 0.0;
    return;
  }}

  const int ring_start = tree_polygon_geometry_offsets[trow];
  const int ring_end   = tree_polygon_geometry_offsets[trow + 1];
  compute_t best = (compute_t)INFINITY;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    const int cs = tree_ring_offsets[ring];
    const int ce = tree_ring_offsets[ring + 1];
    const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                       center_x, center_y, cs, ce);
    if (sq < best) best = sq;
    if (best <= (compute_t)0.0) break;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}

extern "C" __global__ __launch_bounds__(256, 4) void point_multipolygon_distance_from_owned(
    const unsigned char* __restrict__ query_validity,
    const signed char*   __restrict__ query_tags,
    const int*           __restrict__ query_family_row_offsets,
    const int*           __restrict__ query_geometry_offsets,
    const unsigned char* __restrict__ query_empty_mask,
    const double*        __restrict__ query_x,
    const double*        __restrict__ query_y,
    int                  query_point_tag,
    const unsigned char* __restrict__ tree_validity,
    const signed char*   __restrict__ tree_tags,
    const int*           __restrict__ tree_family_row_offsets,
    const int*           __restrict__ tree_geometry_offsets,
    const int*           __restrict__ tree_part_offsets,
    const int*           __restrict__ tree_ring_offsets,
    const unsigned char* __restrict__ tree_empty_mask,
    const double*        __restrict__ tree_x,
    const double*        __restrict__ tree_y,
    int                  tree_multipolygon_tag,
    const int*           __restrict__ left_idx,
    const int*           __restrict__ right_idx,
    double*              __restrict__ out_distances,
    int                  exclusive,
    const long long*     __restrict__ logical_count,
    int                  pair_capacity,
    const double*        __restrict__ center
) {{
  const double center_x = center[0], center_y = center[1];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_capacity || (logical_count != 0 && i >= logical_count[0])) return;

  const int li = left_idx[i];
  const int ri = right_idx[i];

  if (!query_validity[li] || !tree_validity[ri]) {{
    out_distances[i] = INFINITY; return;
  }}
  if (query_tags[li] != query_point_tag || tree_tags[ri] != tree_multipolygon_tag) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qrow = query_family_row_offsets[li];
  const int trow = tree_family_row_offsets[ri];
  if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
    out_distances[i] = INFINITY; return;
  }}

  const int qcoord = query_geometry_offsets[qrow];
  const double raw_px = query_x[qcoord];
  const double raw_py = query_y[qcoord];
  if (isnan(raw_px) || isnan(raw_py)) {{ out_distances[i] = INFINITY; return; }}

  const compute_t px = CX(raw_px);
  const compute_t py = CY(raw_py);

  const int polygon_start = tree_geometry_offsets[trow];
  const int polygon_end   = tree_geometry_offsets[trow + 1];

  compute_t best = (compute_t)INFINITY;
  for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {{
    const int ring_start = tree_part_offsets[polygon];
    const int ring_end   = tree_part_offsets[polygon + 1];
    bool inside = false;
    compute_t poly_best = (compute_t)INFINITY;
    for (int ring = ring_start; ring < ring_end; ++ring) {{
      const int cs = tree_ring_offsets[ring];
      const int ce = tree_ring_offsets[ring + 1];
      if ((ce - cs) < 2) continue;
      bool ring_inside = false;
      bool on_boundary = false;
      for (int c = cs + 1; c < ce; ++c) {{
        const compute_t ax = CX(tree_x[c - 1]), ay = CY(tree_y[c - 1]);
        const compute_t bx = CX(tree_x[c]),     by = CY(tree_y[c]);
        const compute_t cross_val = ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
        if (cross_val == (compute_t)0.0) {{
          const compute_t minx = ax < bx ? ax : bx;
          const compute_t maxx = ax > bx ? ax : bx;
          const compute_t miny = ay < by ? ay : by;
          const compute_t maxy = ay > by ? ay : by;
          if (px >= minx && px <= maxx && py >= miny && py <= maxy) {{
            on_boundary = true;
          }}
        }}
        if (((ay > py) != (by > py)) &&
            (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax)) {{
          ring_inside = !ring_inside;
        }}
      }}
      if (on_boundary) {{ out_distances[i] = 0.0; return; }}
      if (ring_inside) inside = !inside;
      const compute_t sq = point_coords_min_sq_distance(px, py, tree_x, tree_y,
                                                         center_x, center_y, cs, ce);
      if (sq < poly_best) poly_best = sq;
    }}
    if (inside) {{ out_distances[i] = 0.0; return; }}
    if (poly_best < best) best = poly_best;
  }}
  out_distances[i] = (double)sqrt((double)best);
}}

extern "C" __device__ inline bool point_inside_ring_span(
    compute_t px, compute_t py,
    const double* __restrict__ x, const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    int ring_start, int ring_end,
    double center_x, double center_y
) {{
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    const int cs = ring_offsets[ring], ce = ring_offsets[ring + 1];
    bool ring_inside = false;
    for (int c = cs + 1; c < ce; ++c) {{
      const compute_t ax = CX(x[c - 1]), ay = CY(y[c - 1]);
      const compute_t bx = CX(x[c]), by = CY(y[c]);
      const compute_t cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
      if (cross == (compute_t)0.0) {{
        const compute_t minx = ax < bx ? ax : bx, maxx = ax > bx ? ax : bx;
        const compute_t miny = ay < by ? ay : by, maxy = ay > by ? ay : by;
        if (px >= minx && px <= maxx && py >= miny && py <= maxy) return true;
      }}
      if (((ay > py) != (by > py)) &&
          px <= ((bx - ax) * (py - ay)) / (by - ay) + ax) ring_inside = !ring_inside;
    }}
    if (ring_inside) inside = !inside;
  }}
  return inside;
}}

extern "C" __device__ inline compute_t point_family_sq_distance(
    compute_t px, compute_t py,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int row, int target_kind,
    double center_x, double center_y
) {{
  compute_t best = (compute_t)INFINITY;
  if (target_kind <= 1) {{
    const int cs = geometry_offsets[row], ce = geometry_offsets[row + 1];
    for (int c = cs; c < ce; ++c) {{
      const compute_t dx = px - CX(x[c]), dy = py - CY(y[c]);
      const compute_t sq = dx * dx + dy * dy;
      if (sq < best) best = sq;
    }}
    return best;
  }}
  if (target_kind == 2) {{
    return point_coords_min_sq_distance(
        px, py, x, y, center_x, center_y,
        geometry_offsets[row], geometry_offsets[row + 1]);
  }}
  if (target_kind == 3) {{
    for (int part = geometry_offsets[row]; part < geometry_offsets[row + 1]; ++part) {{
      const compute_t sq = point_coords_min_sq_distance(
          px, py, x, y, center_x, center_y, part_offsets[part], part_offsets[part + 1]);
      if (sq < best) best = sq;
      if (best <= (compute_t)0.0) return best;
    }}
    return best;
  }}
  if (target_kind == 4) {{
    if (point_inside_polygon(
            px, py, x, y, center_x, center_y, geometry_offsets, ring_offsets, row))
      return (compute_t)0.0;
    for (int ring = geometry_offsets[row]; ring < geometry_offsets[row + 1]; ++ring) {{
      const compute_t sq = point_coords_min_sq_distance(
          px, py, x, y, center_x, center_y, ring_offsets[ring], ring_offsets[ring + 1]);
      if (sq < best) best = sq;
      if (best <= (compute_t)0.0) return best;
    }}
    return best;
  }}
  for (int polygon = geometry_offsets[row]; polygon < geometry_offsets[row + 1]; ++polygon) {{
    const int ring_start = part_offsets[polygon], ring_end = part_offsets[polygon + 1];
    if (point_inside_ring_span(
            px, py, x, y, ring_offsets, ring_start, ring_end, center_x, center_y))
      return (compute_t)0.0;
    for (int ring = ring_start; ring < ring_end; ++ring) {{
      const compute_t sq = point_coords_min_sq_distance(
          px, py, x, y, center_x, center_y, ring_offsets[ring], ring_offsets[ring + 1]);
      if (sq < best) best = sq;
      if (best <= (compute_t)0.0) return best;
    }}
  }}
  return best;
}}

// One family span from a shared NativeRelationFamilyPartition.  Point and
// multipoint rows are both coordinate ranges, so no host-side expansion is
// needed; each thread reduces one relation pair directly.
extern "C" __global__ __launch_bounds__(256, 4) void pointset_family_distance_from_owned(
    const unsigned char* __restrict__ query_validity,
    const signed char* __restrict__ query_tags,
    const int* __restrict__ query_family_row_offsets,
    const int* __restrict__ query_geometry_offsets,
    const unsigned char* __restrict__ query_empty_mask,
    const double* __restrict__ query_x,
    const double* __restrict__ query_y,
    int query_tag,
    const unsigned char* __restrict__ tree_validity,
    const signed char* __restrict__ tree_tags,
    const int* __restrict__ tree_family_row_offsets,
    const int* __restrict__ tree_geometry_offsets,
    const int* __restrict__ tree_part_offsets,
    const int* __restrict__ tree_ring_offsets,
    const unsigned char* __restrict__ tree_empty_mask,
    const double* __restrict__ tree_x,
    const double* __restrict__ tree_y,
    int tree_tag,
    int target_kind,
    const int* __restrict__ left_idx,
    const int* __restrict__ right_idx,
    const int* __restrict__ source_positions,
    const long long* __restrict__ source_offset,
    const long long* __restrict__ logical_count,
    double* __restrict__ out_distances,
    int exclusive,
    int launch_capacity,
    const double* __restrict__ center
) {{
  const double center_x = center[0], center_y = center[1];
  const long long offset = source_offset == 0 ? 0 : source_offset[0];
  const long long count = logical_count == 0 ? (long long)launch_capacity : logical_count[0];
  const long long stride = (long long)blockDim.x * gridDim.x;
  for (long long lane = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       lane < count; lane += stride) {{
    const long long pair = offset + lane;
    const int out_pos = source_positions == 0 ? (int)pair : source_positions[pair];
    const int li = left_idx[pair], ri = right_idx[pair];
    if ((exclusive && li == ri) || !query_validity[li] || !tree_validity[ri] ||
        query_tags[li] != query_tag || tree_tags[ri] != tree_tag) {{
      out_distances[out_pos] = INFINITY;
      continue;
    }}
    const int qrow = query_family_row_offsets[li];
    const int trow = tree_family_row_offsets[ri];
    if (qrow < 0 || trow < 0 || query_empty_mask[qrow] || tree_empty_mask[trow]) {{
      out_distances[out_pos] = INFINITY;
      continue;
    }}
    compute_t best = (compute_t)INFINITY;
    for (int coord = query_geometry_offsets[qrow];
         coord < query_geometry_offsets[qrow + 1]; ++coord) {{
      const compute_t px = CX(query_x[coord]), py = CY(query_y[coord]);
      const compute_t sq = point_family_sq_distance(
          px, py, tree_geometry_offsets, tree_part_offsets, tree_ring_offsets,
          tree_x, tree_y, trow, target_kind, center_x, center_y);
      if (sq < best) best = sq;
      if (best <= (compute_t)0.0) break;
    }}
    out_distances[out_pos] = (double)sqrt((double)best);
  }}
}}
"""
)

_POINT_DISTANCE_KERNEL_NAMES = (
    "point_linestring_distance_from_owned",
    "point_multilinestring_distance_from_owned",
    "point_polygon_distance_from_owned",
    "point_multipolygon_distance_from_owned",
    "pointset_family_distance_from_owned",
)


def format_distance_kernel_source(compute_type: str = "double") -> str:
    """Format the point-distance kernel source with the given compute type."""
    return _POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)


# Pre-formatted default source for warmup
POINT_DISTANCE_KERNEL_SOURCE_FP64 = format_distance_kernel_source("double")
POINT_DISTANCE_KERNEL_SOURCE_FP32 = format_distance_kernel_source("float")
