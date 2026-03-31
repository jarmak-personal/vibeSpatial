"""NVRTC kernel sources for point-to-geometry distance computation."""

from __future__ import annotations

from vibespatial.cuda.preamble import PRECISION_PREAMBLE

_POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE = PRECISION_PREAMBLE + """

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
      const compute_t scale = (bx > ax ? bx - ax : ax - bx) + (by > ay ? by - ay : ay - by) + (compute_t)1.0;
      if ((cross_val > (compute_t)0.0 ? cross_val : -cross_val) <= ((compute_t)1e-7 * scale)) {{
        const compute_t minx = ax < bx ? ax : bx;
        const compute_t maxx = ax > bx ? ax : bx;
        const compute_t miny = ay < by ? ay : by;
        const compute_t maxy = ay > by ? ay : by;
        if (px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7) &&
            py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7)) {{
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
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

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
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

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
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

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
    int                  pair_count,
    double               center_x,
    double               center_y
) {{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pair_count) return;

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
        const compute_t scale = (bx > ax ? bx - ax : ax - bx) + (by > ay ? by - ay : ay - by) + (compute_t)1.0;
        if ((cross_val > (compute_t)0.0 ? cross_val : -cross_val) <= ((compute_t)1e-7 * scale)) {{
          const compute_t minx = ax < bx ? ax : bx;
          const compute_t maxx = ax > bx ? ax : bx;
          const compute_t miny = ay < by ? ay : by;
          const compute_t maxy = ay > by ? ay : by;
          if (px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7) &&
              py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7)) {{
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
"""

_POINT_DISTANCE_KERNEL_NAMES = (
    "point_linestring_distance_from_owned",
    "point_multilinestring_distance_from_owned",
    "point_polygon_distance_from_owned",
    "point_multipolygon_distance_from_owned",
)


def format_distance_kernel_source(compute_type: str = "double") -> str:
    """Format the point-distance kernel source with the given compute type."""
    return _POINT_DISTANCE_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)


# Pre-formatted default source for warmup
POINT_DISTANCE_KERNEL_SOURCE_FP64 = format_distance_kernel_source("double")
