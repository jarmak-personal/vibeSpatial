from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

from vibespatial.cccl_precompile import request_warmup
from vibespatial.cccl_primitives import compact_indices

request_warmup(["select_i32", "select_i64"])
from vibespatial.cuda_runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry_buffers import GeometryFamily  # noqa: E402
from vibespatial.kernel_registry import register_kernel_variant  # noqa: E402
from vibespatial.kernels.core.geometry_analysis import _launch_family_bounds_kernel  # noqa: E402
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray  # noqa: E402
from vibespatial.precision import KernelClass, PrecisionMode  # noqa: E402
from vibespatial.predicate_support import (  # noqa: E402
    PointSequence,
    coerce_geometry_array,
    resolve_predicate_context,
)
from vibespatial.residency import Residency, TransferTrigger  # noqa: E402
from vibespatial.runtime import ExecutionMode  # noqa: E402

from .point_within_bounds import (  # noqa: E402
    NormalizedBoundsInput,
    _evaluate_point_within_bounds,
    _normalize_right_input,
)

_POINT_IN_POLYGON_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

extern "C" __device__ inline compute_t vibespatial_abs(compute_t value) {{
  return value < (compute_t)0.0 ? -value : value;
}}

extern "C" __device__ inline compute_t vibespatial_max(compute_t left, compute_t right) {{
  return left > right ? left : right;
}}

extern "C" __device__ inline bool point_on_segment(
    compute_t px,
    compute_t py,
    compute_t ax,
    compute_t ay,
    compute_t bx,
    compute_t by
) {{
  const compute_t dx = bx - ax;
  const compute_t dy = by - ay;
  const compute_t cross = ((px - ax) * dy) - ((py - ay) * dx);
  const compute_t scale = vibespatial_abs(dx) + vibespatial_abs(dy) + (compute_t)1.0;
  if (vibespatial_abs(cross) > ((compute_t)1e-7 * scale)) {{
    return false;
  }}
  const compute_t minx = ax < bx ? ax : bx;
  const compute_t maxx = ax > bx ? ax : bx;
  const compute_t miny = ay < by ? ay : by;
  const compute_t maxy = ay > by ? ay : by;
  return px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7) && py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7);
}}

extern "C" __device__ inline bool ring_contains_even_odd(
    compute_t px,
    compute_t py,
    const double* x,
    const double* y,
    double center_x,
    double center_y,
    int coord_start,
    int coord_end,
    bool* on_boundary
) {{
  bool inside = false;
  if ((coord_end - coord_start) < 2) {{
    return false;
  }}
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {{
    const compute_t ax = CX(x[coord - 1]);
    const compute_t ay = CY(y[coord - 1]);
    const compute_t bx = CX(x[coord]);
    const compute_t by = CY(y[coord]);
    if (point_on_segment(px, py, ax, ay, bx, by)) {{
      *on_boundary = true;
      return true;
    }}
    const bool intersects = ((ay > py) != (by > py)) &&
        (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax);
    if (intersects) {{
      inside = !inside;
    }}
  }}
  return inside;
}}

extern "C" __device__ inline bool polygon_contains_point(
    compute_t px,
    compute_t py,
    const double* x,
    const double* y,
    double center_x,
    double center_y,
    const int* geometry_offsets,
    const int* ring_offsets,
    int polygon_row
) {{
  const int ring_start = geometry_offsets[polygon_row];
  const int ring_end = geometry_offsets[polygon_row + 1];
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {{
    bool on_boundary = false;
    const int coord_start = ring_offsets[ring];
    const int coord_end = ring_offsets[ring + 1];
    const bool ring_inside = ring_contains_even_odd(px, py, x, y, center_x, center_y, coord_start, coord_end, &on_boundary);
    if (on_boundary) {{
      return true;
    }}
    if (ring_inside) {{
      inside = !inside;
    }}
  }}
  return inside;
}}

extern "C" __device__ inline bool multipolygon_contains_point(
    compute_t px,
    compute_t py,
    const double* x,
    const double* y,
    double center_x,
    double center_y,
    const int* geometry_offsets,
    const int* part_offsets,
    const int* ring_offsets,
    int multipolygon_row
) {{
  const int polygon_start = geometry_offsets[multipolygon_row];
  const int polygon_end = geometry_offsets[multipolygon_row + 1];
  for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {{
    bool inside = false;
    const int ring_start = part_offsets[polygon];
    const int ring_end = part_offsets[polygon + 1];
    for (int ring = ring_start; ring < ring_end; ++ring) {{
      bool on_boundary = false;
      const int coord_start = ring_offsets[ring];
      const int coord_end = ring_offsets[ring + 1];
      const bool ring_inside = ring_contains_even_odd(px, py, x, y, center_x, center_y, coord_start, coord_end, &on_boundary);
      if (on_boundary) {{
        return true;
      }}
      if (ring_inside) {{
        inside = !inside;
      }}
    }}
    if (inside) {{
      return true;
    }}
  }}
  return false;
}}

extern "C" __global__ void point_in_polygon_bounds_mask(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const double* polygon_bounds,
    const double* multipolygon_bounds,
    int polygon_tag,
    int multipolygon_tag,
    unsigned char* out,
    int row_count,
    double center_x,
    double center_y
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{
    return;
  }}
  out[row] = 0;
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {{
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const double px = point_x[point_coord];
  const double py = point_y[point_coord];

  const signed char tag = right_tags[row];
  const int family_row = right_family_row_offsets[row];
  if (family_row < 0) {{
    return;
  }}
  if (tag == polygon_tag && polygon_bounds != nullptr) {{
    const int base = family_row * 4;
    out[row] = (
      polygon_bounds[base + 0] <= px &&
      px <= polygon_bounds[base + 2] &&
      polygon_bounds[base + 1] <= py &&
      py <= polygon_bounds[base + 3]
    ) ? 1 : 0;
    return;
  }}
  if (tag == multipolygon_tag && multipolygon_bounds != nullptr) {{
    const int base = family_row * 4;
    out[row] = (
      multipolygon_bounds[base + 0] <= px &&
      px <= multipolygon_bounds[base + 2] &&
      multipolygon_bounds[base + 1] <= py &&
      py <= multipolygon_bounds[base + 3]
    ) ? 1 : 0;
  }}
}}

extern "C" __global__ void point_in_polygon_polygon_dense(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const unsigned char* candidate_mask,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    unsigned char* out,
    int row_count,
    double center_x,
    double center_y
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < row_count;
  const unsigned char is_candidate = valid ? candidate_mask[row] : 0;
  /* Warp-level ballot: if no thread in this warp is a candidate,
     skip all global memory reads for the expensive PIP computation. */
  if (__ballot_sync(0xFFFFFFFF, is_candidate) == 0) {{
    return;
  }}
  if (!valid || !is_candidate) {{
    return;
  }}
  const int point_row = point_row_offsets[row];
  const int polygon_row = polygon_row_offsets[row];
  if (point_row < 0 || polygon_row < 0 || polygon_empty_mask[polygon_row]) {{
    out[row] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[row] = polygon_contains_point(
      px,
      py,
      polygon_x,
      polygon_y,
      center_x,
      center_y,
      polygon_geometry_offsets,
      polygon_ring_offsets,
      polygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void point_in_polygon_polygon_compacted(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const int* polygon_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    unsigned char* out,
    int candidate_count,
    double center_x,
    double center_y
) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {{
    return;
  }}
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int polygon_row = polygon_row_offsets[row];
  if (point_row < 0 || polygon_row < 0 || polygon_empty_mask[polygon_row]) {{
    out[index] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[index] = polygon_contains_point(
      px,
      py,
      polygon_x,
      polygon_y,
      center_x,
      center_y,
      polygon_geometry_offsets,
      polygon_ring_offsets,
      polygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void point_in_polygon_multipolygon_dense(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const unsigned char* candidate_mask,
    const int* multipolygon_row_offsets,
    const unsigned char* multipolygon_empty_mask,
    const int* multipolygon_geometry_offsets,
    const int* multipolygon_part_offsets,
    const int* multipolygon_ring_offsets,
    const double* multipolygon_x,
    const double* multipolygon_y,
    unsigned char* out,
    int row_count,
    double center_x,
    double center_y
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < row_count;
  const unsigned char is_candidate = valid ? candidate_mask[row] : 0;
  /* Warp-level ballot: if no thread in this warp is a candidate,
     skip all global memory reads for the expensive PIP computation. */
  if (__ballot_sync(0xFFFFFFFF, is_candidate) == 0) {{
    return;
  }}
  if (!valid || !is_candidate) {{
    return;
  }}
  const int point_row = point_row_offsets[row];
  const int multipolygon_row = multipolygon_row_offsets[row];
  if (point_row < 0 || multipolygon_row < 0 || multipolygon_empty_mask[multipolygon_row]) {{
    out[row] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[row] = multipolygon_contains_point(
      px,
      py,
      multipolygon_x,
      multipolygon_y,
      center_x,
      center_y,
      multipolygon_geometry_offsets,
      multipolygon_part_offsets,
      multipolygon_ring_offsets,
      multipolygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void point_in_polygon_multipolygon_compacted(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const int* multipolygon_row_offsets,
    const unsigned char* multipolygon_empty_mask,
    const int* multipolygon_geometry_offsets,
    const int* multipolygon_part_offsets,
    const int* multipolygon_ring_offsets,
    const double* multipolygon_x,
    const double* multipolygon_y,
    unsigned char* out,
    int candidate_count,
    double center_x,
    double center_y
) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {{
    return;
  }}
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  const int multipolygon_row = multipolygon_row_offsets[row];
  if (point_row < 0 || multipolygon_row < 0 || multipolygon_empty_mask[multipolygon_row]) {{
    out[index] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[index] = multipolygon_contains_point(
      px,
      py,
      multipolygon_x,
      multipolygon_y,
      center_x,
      center_y,
      multipolygon_geometry_offsets,
      multipolygon_part_offsets,
      multipolygon_ring_offsets,
      multipolygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void point_in_polygon_polygon_compacted_tagged(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    int polygon_tag,
    unsigned char* out,
    int candidate_count,
    double center_x,
    double center_y
) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {{
    return;
  }}
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || right_tags[row] != polygon_tag) {{
    out[index] = 0;
    return;
  }}
  const int polygon_row = right_family_row_offsets[row];
  if (polygon_row < 0 || polygon_empty_mask[polygon_row]) {{
    out[index] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[index] = polygon_contains_point(
      px,
      py,
      polygon_x,
      polygon_y,
      center_x,
      center_y,
      polygon_geometry_offsets,
      polygon_ring_offsets,
      polygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void point_in_polygon_multipolygon_compacted_tagged(
    const int* candidate_rows,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const unsigned char* multipolygon_empty_mask,
    const int* multipolygon_geometry_offsets,
    const int* multipolygon_part_offsets,
    const int* multipolygon_ring_offsets,
    const double* multipolygon_x,
    const double* multipolygon_y,
    int multipolygon_tag,
    unsigned char* out,
    int candidate_count,
    double center_x,
    double center_y
) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count) {{
    return;
  }}
  const int row = candidate_rows[index];
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || right_tags[row] != multipolygon_tag) {{
    out[index] = 0;
    return;
  }}
  const int multipolygon_row = right_family_row_offsets[row];
  if (multipolygon_row < 0 || multipolygon_empty_mask[multipolygon_row]) {{
    out[index] = 0;
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const compute_t px = CX(point_x[point_coord]);
  const compute_t py = CY(point_y[point_coord]);
  out[index] = multipolygon_contains_point(
      px,
      py,
      multipolygon_x,
      multipolygon_y,
      center_x,
      center_y,
      multipolygon_geometry_offsets,
      multipolygon_part_offsets,
      multipolygon_ring_offsets,
      multipolygon_row
  ) ? 1 : 0;
}}

extern "C" __global__ void scatter_compacted_hits(
    const int* candidate_rows,
    const unsigned char* compacted_hits,
    unsigned char* dense_out,
    int candidate_count,
    double center_x,
    double center_y
) {{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidate_count || !compacted_hits[index]) {{
    return;
  }}
  dense_out[candidate_rows[index]] = 1;
}}

extern "C" __global__ void point_in_polygon_fused(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const double* polygon_bounds,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    const double* multipolygon_bounds,
    const unsigned char* multipolygon_empty_mask,
    const int* multipolygon_geometry_offsets,
    const int* multipolygon_part_offsets,
    const int* multipolygon_ring_offsets,
    const double* multipolygon_x,
    const double* multipolygon_y,
    int polygon_tag,
    int multipolygon_tag,
    unsigned char* out,
    int row_count,
    double center_x,
    double center_y
) {{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{
    return;
  }}
  out[row] = 0;
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {{
    return;
  }}
  const int point_coord = point_geometry_offsets[point_row];
  const double px = point_x[point_coord];
  const double py = point_y[point_coord];

  const signed char tag = right_tags[row];
  const int family_row = right_family_row_offsets[row];
  if (family_row < 0) {{
    return;
  }}
  if (tag == polygon_tag && polygon_bounds != nullptr) {{
    const int base = family_row * 4;
    if (!(polygon_bounds[base + 0] <= px && px <= polygon_bounds[base + 2] &&
          polygon_bounds[base + 1] <= py && py <= polygon_bounds[base + 3])) {{
      return;
    }}
    if (polygon_empty_mask[family_row]) {{
      return;
    }}
    const compute_t cpx = CX(px);
    const compute_t cpy = CY(py);
    out[row] = polygon_contains_point(
        cpx, cpy, polygon_x, polygon_y,
        center_x, center_y,
        polygon_geometry_offsets, polygon_ring_offsets, family_row
    ) ? 1 : 0;
    return;
  }}
  if (tag == multipolygon_tag && multipolygon_bounds != nullptr) {{
    const int base = family_row * 4;
    if (!(multipolygon_bounds[base + 0] <= px && px <= multipolygon_bounds[base + 2] &&
          multipolygon_bounds[base + 1] <= py && py <= multipolygon_bounds[base + 3])) {{
      return;
    }}
    if (multipolygon_empty_mask[family_row]) {{
      return;
    }}
    const compute_t cpx = CX(px);
    const compute_t cpy = CY(py);
    out[row] = multipolygon_contains_point(
        cpx, cpy, multipolygon_x, multipolygon_y,
        center_x, center_y,
        multipolygon_geometry_offsets, multipolygon_part_offsets,
        multipolygon_ring_offsets, family_row
    ) ? 1 : 0;
  }}
}}
"""

# ---------------------------------------------------------------------------
# Block-per-pair kernel: for complex polygons (>1024 vertices), one thread
# block cooperatively processes a single (point, polygon) pair.  Threads
# split ring iteration and use shared-memory XOR reduction for the even-odd
# containment result.
# ---------------------------------------------------------------------------
_PIP_BLOCK_PER_PAIR_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))

extern "C" __device__ inline compute_t vibespatial_abs_bp(compute_t value) {{
  return value < (compute_t)0.0 ? -value : value;
}}

extern "C" __device__ inline bool point_on_segment_bp(
    compute_t px, compute_t py,
    compute_t ax, compute_t ay,
    compute_t bx, compute_t by
) {{
  const compute_t dx = bx - ax;
  const compute_t dy = by - ay;
  const compute_t cross = ((px - ax) * dy) - ((py - ay) * dx);
  const compute_t scale = vibespatial_abs_bp(dx) + vibespatial_abs_bp(dy) + (compute_t)1.0;
  if (vibespatial_abs_bp(cross) > ((compute_t)1e-7 * scale)) {{
    return false;
  }}
  const compute_t minx = ax < bx ? ax : bx;
  const compute_t maxx = ax > bx ? ax : bx;
  const compute_t miny = ay < by ? ay : by;
  const compute_t maxy = ay > by ? ay : by;
  return px >= (minx - (compute_t)1e-7) && px <= (maxx + (compute_t)1e-7)
      && py >= (miny - (compute_t)1e-7) && py <= (maxy + (compute_t)1e-7);
}}

extern "C" __global__ void pip_block_per_pair_polygon(
    const int* candidate_left,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const unsigned char* polygon_empty_mask,
    const int* polygon_geometry_offsets,
    const int* polygon_ring_offsets,
    const double* polygon_x,
    const double* polygon_y,
    int polygon_tag,
    unsigned char* out,
    int pair_count,
    double center_x,
    double center_y
) {{
    const int pair = blockIdx.x;
    if (pair >= pair_count) return;

    const int row = candidate_left[pair];
    const int point_row = point_row_offsets[row];
    if (point_row < 0 || right_tags[row] != polygon_tag) {{
        if (threadIdx.x == 0) out[pair] = 0;
        return;
    }}
    const int polygon_row = right_family_row_offsets[row];
    if (polygon_row < 0 || polygon_empty_mask[polygon_row]) {{
        if (threadIdx.x == 0) out[pair] = 0;
        return;
    }}

    const int point_coord = point_geometry_offsets[point_row];
    const compute_t px = CX(point_x[point_coord]);
    const compute_t py = CY(point_y[point_coord]);

    const int ring_start = polygon_geometry_offsets[polygon_row];
    const int ring_end = polygon_geometry_offsets[polygon_row + 1];

    /* Block-level result accumulators */
    __shared__ int block_crossings;
    __shared__ int block_boundary;
    if (threadIdx.x == 0) {{
        block_crossings = 0;
        block_boundary = 0;
    }}
    __syncthreads();

    /* Shared memory for inter-warp reduction (up to 256 threads = 8 warps) */
    __shared__ int warp_crossings[8];
    __shared__ int warp_boundary[8];

    /* Process each ring cooperatively -- all threads split the ring's edges */
    for (int ring = ring_start; ring < ring_end; ++ring) {{
        const int cs = polygon_ring_offsets[ring];
        const int ce = polygon_ring_offsets[ring + 1];
        const int edge_count = ce - cs - 1;
        if (edge_count < 1) continue;

        /* Each thread processes a strided subset of edges */
        int my_crossings = 0;
        int my_boundary = 0;

        for (int e = (int)threadIdx.x; e < edge_count; e += (int)blockDim.x) {{
            const int c = cs + 1 + e;
            const compute_t ax = CX(polygon_x[c - 1]);
            const compute_t ay = CY(polygon_y[c - 1]);
            const compute_t bx = CX(polygon_x[c]);
            const compute_t by = CY(polygon_y[c]);

            if (point_on_segment_bp(px, py, ax, ay, bx, by)) {{
                my_boundary = 1;
            }}
            const bool intersects = ((ay > py) != (by > py)) &&
                (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax);
            if (intersects) {{
                my_crossings ^= 1;
            }}
        }}

        /* Warp-level reduction: XOR for crossings, OR for boundary */
        const unsigned int FULL_MASK = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset >>= 1) {{
            my_crossings ^= __shfl_xor_sync(FULL_MASK, my_crossings, offset);
            my_boundary  |= __shfl_xor_sync(FULL_MASK, my_boundary, offset);
        }}

        /* Lane 0 of each warp writes to shared memory for block-level reduction */
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;

        if (lane_id == 0) {{
            warp_crossings[warp_id] = my_crossings;
            warp_boundary[warp_id] = my_boundary;
        }}
        __syncthreads();

        /* Thread 0 reduces across warps for this ring */
        if (threadIdx.x == 0) {{
            int ring_crossings = 0;
            int ring_boundary = 0;
            const int num_warps = ((int)blockDim.x + 31) / 32;
            for (int w = 0; w < num_warps; ++w) {{
                ring_crossings ^= warp_crossings[w];
                ring_boundary  |= warp_boundary[w];
            }}
            block_boundary |= ring_boundary;
            if (ring_crossings) {{
                block_crossings ^= 1;
            }}
        }}
        __syncthreads();
    }}

    if (threadIdx.x == 0) {{
        out[pair] = (block_boundary || block_crossings) ? 1 : 0;
    }}
}}

extern "C" __global__ void pip_block_per_pair_multipolygon(
    const int* candidate_left,
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const signed char* right_tags,
    const int* right_family_row_offsets,
    const unsigned char* multipolygon_empty_mask,
    const int* multipolygon_geometry_offsets,
    const int* multipolygon_part_offsets,
    const int* multipolygon_ring_offsets,
    const double* multipolygon_x,
    const double* multipolygon_y,
    int multipolygon_tag,
    unsigned char* out,
    int pair_count,
    double center_x,
    double center_y
) {{
    const int pair = blockIdx.x;
    if (pair >= pair_count) return;

    const int row = candidate_left[pair];
    const int point_row = point_row_offsets[row];
    if (point_row < 0 || right_tags[row] != multipolygon_tag) {{
        if (threadIdx.x == 0) out[pair] = 0;
        return;
    }}
    const int mp_row = right_family_row_offsets[row];
    if (mp_row < 0 || multipolygon_empty_mask[mp_row]) {{
        if (threadIdx.x == 0) out[pair] = 0;
        return;
    }}

    const int point_coord = point_geometry_offsets[point_row];
    const compute_t px = CX(point_x[point_coord]);
    const compute_t py = CY(point_y[point_coord]);

    const int poly_start = multipolygon_geometry_offsets[mp_row];
    const int poly_end = multipolygon_geometry_offsets[mp_row + 1];
    const int ring_start_global = multipolygon_part_offsets[poly_start];
    const int ring_end_global = multipolygon_part_offsets[poly_end];

    /* Block-level result accumulators */
    __shared__ int block_crossings;
    __shared__ int block_boundary;
    if (threadIdx.x == 0) {{
        block_crossings = 0;
        block_boundary = 0;
    }}
    __syncthreads();

    /* Shared memory for inter-warp reduction (up to 256 threads = 8 warps) */
    __shared__ int warp_crossings[8];
    __shared__ int warp_boundary[8];

    /* Process each ring cooperatively -- all threads split the ring's edges */
    for (int ring = ring_start_global; ring < ring_end_global; ++ring) {{
        const int cs = multipolygon_ring_offsets[ring];
        const int ce = multipolygon_ring_offsets[ring + 1];
        const int edge_count = ce - cs - 1;
        if (edge_count < 1) continue;

        /* Each thread processes a strided subset of edges */
        int my_crossings = 0;
        int my_boundary = 0;

        for (int e = (int)threadIdx.x; e < edge_count; e += (int)blockDim.x) {{
            const int c = cs + 1 + e;
            const compute_t ax = CX(multipolygon_x[c - 1]);
            const compute_t ay = CY(multipolygon_y[c - 1]);
            const compute_t bx = CX(multipolygon_x[c]);
            const compute_t by = CY(multipolygon_y[c]);

            if (point_on_segment_bp(px, py, ax, ay, bx, by)) {{
                my_boundary = 1;
            }}
            const bool intersects = ((ay > py) != (by > py)) &&
                (px <= (((bx - ax) * (py - ay)) / ((by - ay) + (compute_t)0.0)) + ax);
            if (intersects) {{
                my_crossings ^= 1;
            }}
        }}

        /* Warp-level reduction: XOR for crossings, OR for boundary */
        const unsigned int FULL_MASK = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset >>= 1) {{
            my_crossings ^= __shfl_xor_sync(FULL_MASK, my_crossings, offset);
            my_boundary  |= __shfl_xor_sync(FULL_MASK, my_boundary, offset);
        }}

        /* Lane 0 of each warp writes to shared memory for block-level reduction */
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;

        if (lane_id == 0) {{
            warp_crossings[warp_id] = my_crossings;
            warp_boundary[warp_id] = my_boundary;
        }}
        __syncthreads();

        /* Thread 0 reduces across warps for this ring */
        if (threadIdx.x == 0) {{
            int ring_crossings = 0;
            int ring_boundary = 0;
            const int num_warps = ((int)blockDim.x + 31) / 32;
            for (int w = 0; w < num_warps; ++w) {{
                ring_crossings ^= warp_crossings[w];
                ring_boundary  |= warp_boundary[w];
            }}
            block_boundary |= ring_boundary;
            if (ring_crossings) {{
                block_crossings ^= 1;
            }}
        }}
        __syncthreads();
    }}

    if (threadIdx.x == 0) {{
        out[pair] = (block_boundary || block_crossings) ? 1 : 0;
    }}
}}
"""

_PIP_BLOCK_PER_PAIR_KERNEL_NAMES = (
    "pip_block_per_pair_polygon",
    "pip_block_per_pair_multipolygon",
)


def _format_block_per_pair_source(compute_type: str = "double") -> str:
    return _PIP_BLOCK_PER_PAIR_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)


def _pip_block_per_pair_kernels(compute_type: str = "double"):
    source = _format_block_per_pair_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(
        f"pip-block-per-pair-{compute_type}", source
    )
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_PIP_BLOCK_PER_PAIR_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Work-size binning infrastructure
# ---------------------------------------------------------------------------

# Complexity bins: simple (<64 verts), medium (64-1024), complex (>1024)
_PIP_WORK_BINS = [64, 1024]


def _estimate_pip_work_polygon(
    polygon_geometry_offsets: np.ndarray,
    polygon_ring_offsets: np.ndarray,
    candidate_right_family_rows: np.ndarray,
) -> np.ndarray:
    """Estimate work per candidate pair based on polygon vertex count."""
    ring_start = polygon_geometry_offsets[candidate_right_family_rows]
    ring_end = polygon_geometry_offsets[candidate_right_family_rows + 1]
    coord_start = polygon_ring_offsets[ring_start]
    coord_end = polygon_ring_offsets[ring_end]
    return (coord_end - coord_start).astype(np.int32)


def _estimate_pip_work_multipolygon(
    multipolygon_geometry_offsets: np.ndarray,
    multipolygon_part_offsets: np.ndarray,
    multipolygon_ring_offsets: np.ndarray,
    candidate_right_family_rows: np.ndarray,
) -> np.ndarray:
    """Estimate work per candidate pair based on multipolygon vertex count."""
    poly_start = multipolygon_geometry_offsets[candidate_right_family_rows]
    poly_end = multipolygon_geometry_offsets[candidate_right_family_rows + 1]
    ring_start = multipolygon_part_offsets[poly_start]
    ring_end = multipolygon_part_offsets[poly_end]
    coord_start = multipolygon_ring_offsets[ring_start]
    coord_end = multipolygon_ring_offsets[ring_end]
    return (coord_end - coord_start).astype(np.int32)


def _should_bin_dispatch(work_estimates: np.ndarray) -> bool:
    """Use binned dispatch when work variance is high.

    Returns True when the coefficient of variation exceeds 2.0 and there are
    enough candidates to amortise the overhead of multiple kernel launches.
    """
    if len(work_estimates) < 1024:
        return False
    mean_work = work_estimates.mean()
    if mean_work < 1:
        return False
    cv = work_estimates.std() / mean_work
    return cv > 2.0


def _scatter_bin_results(
    bin_indices_device,
    bin_results_device,
    full_output_device,
    bin_count: int,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    """Scatter bin results back into the full output buffer at original positions."""
    runtime = get_cuda_runtime()
    kernel = _point_in_polygon_kernels(compute_type)["scatter_compacted_hits"]
    params = (
        (
            runtime.pointer(bin_indices_device),
            runtime.pointer(bin_results_device),
            runtime.pointer(full_output_device),
            bin_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, bin_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _binned_polygon_dispatch(
    candidate_rows_device,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    work_estimates: np.ndarray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Dispatch polygon PIP kernel in work-balanced bins.

    For simple/medium bins uses the thread-per-pair kernel.  For the complex
    bin (>1024 vertices) uses the block-per-pair kernel where one thread block
    cooperatively processes a single candidate pair.
    """
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]

    candidate_rows_host = runtime.copy_device_to_host(candidate_rows_device)
    device_out = runtime.allocate((candidate_count,), np.uint8)

    bin_edges = [0] + _PIP_WORK_BINS + [int(work_estimates.max()) + 1]
    _log.debug(
        "binned_polygon_dispatch: %d candidates, bin_edges=%s, "
        "work min=%d max=%d mean=%.1f std=%.1f",
        candidate_count, bin_edges,
        work_estimates.min(), work_estimates.max(),
        work_estimates.mean(), work_estimates.std(),
    )

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (work_estimates >= lo) & (work_estimates < hi)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        bin_indices = np.flatnonzero(mask).astype(np.int32)
        bin_candidate_rows = candidate_rows_host[bin_indices].astype(np.int32)
        device_bin_candidates = runtime.from_host(bin_candidate_rows)
        device_bin_out = runtime.allocate((bin_count,), np.uint8)

        try:
            use_block_per_pair = lo >= _PIP_WORK_BINS[-1]

            if use_block_per_pair:
                kernel = _pip_block_per_pair_kernels(compute_type)["pip_block_per_pair_polygon"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(polygon_buffer.empty_mask),
                        runtime.pointer(polygon_buffer.geometry_offsets),
                        runtime.pointer(polygon_buffer.ring_offsets),
                        runtime.pointer(polygon_buffer.x),
                        runtime.pointer(polygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.POLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(kernel, grid=grid, block=block, params=params)
            else:
                kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_compacted_tagged"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(polygon_buffer.empty_mask),
                        runtime.pointer(polygon_buffer.geometry_offsets),
                        runtime.pointer(polygon_buffer.ring_offsets),
                        runtime.pointer(polygon_buffer.x),
                        runtime.pointer(polygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.POLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
                        KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(kernel, grid=grid, block=block, params=params)

            device_bin_indices = runtime.from_host(bin_indices)
            try:
                _scatter_bin_results(
                    device_bin_indices, device_bin_out, device_out, bin_count,
                    compute_type=compute_type, center_x=center_x, center_y=center_y,
                )
            finally:
                runtime.free(device_bin_indices)

        finally:
            runtime.free(device_bin_candidates)
            runtime.free(device_bin_out)

    return device_out


def _binned_multipolygon_dispatch(
    candidate_rows_device,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    work_estimates: np.ndarray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Dispatch multipolygon PIP kernel in work-balanced bins."""
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]

    candidate_rows_host = runtime.copy_device_to_host(candidate_rows_device)
    device_out = runtime.allocate((candidate_count,), np.uint8)

    bin_edges = [0] + _PIP_WORK_BINS + [int(work_estimates.max()) + 1]
    _log.debug(
        "binned_multipolygon_dispatch: %d candidates, work min=%d max=%d",
        candidate_count, work_estimates.min(), work_estimates.max(),
    )

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (work_estimates >= lo) & (work_estimates < hi)
        bin_count = int(mask.sum())
        if bin_count == 0:
            continue

        bin_indices = np.flatnonzero(mask).astype(np.int32)
        bin_candidate_rows = candidate_rows_host[bin_indices].astype(np.int32)
        device_bin_candidates = runtime.from_host(bin_candidate_rows)
        device_bin_out = runtime.allocate((bin_count,), np.uint8)

        try:
            use_block_per_pair = lo >= _PIP_WORK_BINS[-1]

            if use_block_per_pair:
                kernel = _pip_block_per_pair_kernels(compute_type)["pip_block_per_pair_multipolygon"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(multipolygon_buffer.empty_mask),
                        runtime.pointer(multipolygon_buffer.geometry_offsets),
                        runtime.pointer(multipolygon_buffer.part_offsets),
                        runtime.pointer(multipolygon_buffer.ring_offsets),
                        runtime.pointer(multipolygon_buffer.x),
                        runtime.pointer(multipolygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(kernel, grid=grid, block=block, params=params)
            else:
                kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_compacted_tagged"]
                params = (
                    (
                        runtime.pointer(device_bin_candidates),
                        runtime.pointer(left_state.family_row_offsets),
                        runtime.pointer(point_buffer.geometry_offsets),
                        runtime.pointer(point_buffer.x),
                        runtime.pointer(point_buffer.y),
                        runtime.pointer(right_state.tags),
                        runtime.pointer(right_state.family_row_offsets),
                        runtime.pointer(multipolygon_buffer.empty_mask),
                        runtime.pointer(multipolygon_buffer.geometry_offsets),
                        runtime.pointer(multipolygon_buffer.part_offsets),
                        runtime.pointer(multipolygon_buffer.ring_offsets),
                        runtime.pointer(multipolygon_buffer.x),
                        runtime.pointer(multipolygon_buffer.y),
                        FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                        runtime.pointer(device_bin_out),
                        bin_count, center_x, center_y,
                    ),
                    (
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR,
                        KERNEL_PARAM_I32, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                    ),
                )
                grid, block = runtime.launch_config(kernel, bin_count)
                runtime.launch(kernel, grid=grid, block=block, params=params)

            device_bin_indices = runtime.from_host(bin_indices)
            try:
                _scatter_bin_results(
                    device_bin_indices, device_bin_out, device_out, bin_count,
                    compute_type=compute_type, center_x=center_x, center_y=center_y,
                )
            finally:
                runtime.free(device_bin_indices)

        finally:
            runtime.free(device_bin_candidates)
            runtime.free(device_bin_out)

    return device_out


def _compute_work_estimates_for_candidates(
    candidate_rows_host: np.ndarray,
    right_array: OwnedGeometryArray,
) -> np.ndarray:
    """Compute work estimates for all candidates across families.

    Returns a single array with the vertex count per candidate.  Uses the
    host-side offset arrays which are small metadata and cheap to index.
    """
    tags = right_array.tags[candidate_rows_host]
    family_row_offsets = right_array.family_row_offsets[candidate_rows_host]
    work = np.zeros(len(candidate_rows_host), dtype=np.int32)

    if GeometryFamily.POLYGON in right_array.families:
        poly_buf = right_array.families[GeometryFamily.POLYGON]
        poly_mask = tags == FAMILY_TAGS[GeometryFamily.POLYGON]
        poly_family_rows = family_row_offsets[poly_mask]
        valid = poly_family_rows >= 0
        if valid.any():
            poly_work = _estimate_pip_work_polygon(
                poly_buf.geometry_offsets,
                poly_buf.ring_offsets,
                poly_family_rows[valid],
            )
            poly_indices = np.flatnonzero(poly_mask)
            work[poly_indices[valid]] = poly_work

    if GeometryFamily.MULTIPOLYGON in right_array.families:
        mp_buf = right_array.families[GeometryFamily.MULTIPOLYGON]
        mp_mask = tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
        mp_family_rows = family_row_offsets[mp_mask]
        valid = mp_family_rows >= 0
        if valid.any():
            mp_work = _estimate_pip_work_multipolygon(
                mp_buf.geometry_offsets,
                mp_buf.part_offsets,
                mp_buf.ring_offsets,
                mp_family_rows[valid],
            )
            mp_indices = np.flatnonzero(mp_mask)
            work[mp_indices[valid]] = mp_work

    return work


_POINT_IN_POLYGON_KERNEL_NAMES = (
    "point_in_polygon_bounds_mask",
    "point_in_polygon_polygon_dense",
    "point_in_polygon_polygon_compacted",
    "point_in_polygon_polygon_compacted_tagged",
    "point_in_polygon_multipolygon_dense",
    "point_in_polygon_multipolygon_compacted",
    "point_in_polygon_multipolygon_compacted_tagged",
    "scatter_compacted_hits",
    "point_in_polygon_fused",
)


def _format_pip_kernel_source(compute_type: str = "double") -> str:
    return _POINT_IN_POLYGON_KERNEL_SOURCE_TEMPLATE.format(compute_type=compute_type)


_POINT_IN_POLYGON_KERNEL_SOURCE = _format_pip_kernel_source("double")

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("point-in-polygon", _POINT_IN_POLYGON_KERNEL_SOURCE, _POINT_IN_POLYGON_KERNEL_NAMES),
])


def _point_in_polygon_kernels(compute_type: str = "double"):
    source = _format_pip_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"point-in-polygon-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_POINT_IN_POLYGON_KERNEL_NAMES,
    )


def _to_python_result(values: np.ndarray) -> list[bool | None]:
    """Convert object-dtype ndarray of {True, False, None} to list[bool | None].

    Uses numpy vectorized ops instead of per-element Python iteration.
    At 1M elements: ~2ms (numpy .tolist()) vs ~74ms (list comprehension).
    """
    null_mask = np.equal(values, None)
    result = np.empty(len(values), dtype=object)
    result[:] = np.where(null_mask, False, values).astype(bool)
    result[null_mask] = None
    return list(result)  # public API materialization boundary


def _candidate_rows_by_family(
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> dict[GeometryFamily, np.ndarray]:
    rows_by_family: dict[GeometryFamily, np.ndarray] = {}
    if candidate_rows.size == 0:
        return rows_by_family
    candidate_tags = right.tags[candidate_rows]
    for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
        rows = candidate_rows[candidate_tags == FAMILY_TAGS[family]]
        if rows.size:
            rows_by_family[family] = rows.astype(np.int32, copy=False)
    return rows_by_family


def _select_gpu_strategy(
    row_count: int,
    *,
    strategy: str,
    right_array: OwnedGeometryArray | None = None,
) -> str:
    del row_count
    if strategy != "auto":
        return strategy
    # Check if work-size binning would help.
    if right_array is not None:
        try:
            work_samples: list[np.ndarray] = []
            if GeometryFamily.POLYGON in right_array.families:
                buf = right_array.families[GeometryFamily.POLYGON]
                if buf.row_count > 0:
                    ring_s = buf.geometry_offsets[:-1]
                    ring_e = buf.geometry_offsets[1:]
                    coord_s = buf.ring_offsets[ring_s]
                    coord_e = buf.ring_offsets[ring_e]
                    work_samples.append((coord_e - coord_s).astype(np.int32))
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                buf = right_array.families[GeometryFamily.MULTIPOLYGON]
                if buf.row_count > 0:
                    poly_s = buf.geometry_offsets[:-1]
                    poly_e = buf.geometry_offsets[1:]
                    ring_s = buf.part_offsets[poly_s]
                    ring_e = buf.part_offsets[poly_e]
                    coord_s = buf.ring_offsets[ring_s]
                    coord_e = buf.ring_offsets[ring_e]
                    work_samples.append((coord_e - coord_s).astype(np.int32))
            if work_samples:
                all_work = np.concatenate(work_samples)
                if _should_bin_dispatch(all_work):
                    return "binned"
        except Exception:
            pass
    # Fused bounds+PIP kernel: single kernel launch replaces bounds mask +
    # compact_indices + per-family PIP + scatter (5-6 launches → 1).
    return "fused"


def _launch_polygon_dense(
    candidate_indices: np.ndarray,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    device_out,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    """Launch dense polygon PIP kernel writing into caller-owned *device_out*."""
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]
    n = left.row_count
    device_mask = runtime.allocate((n,), np.uint8, zero=True)
    device_indices = runtime.from_host(candidate_indices.astype(np.int32, copy=False))
    device_mask[device_indices] = np.uint8(1)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_dense"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(device_mask),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(polygon_buffer.empty_mask),
                runtime.pointer(polygon_buffer.geometry_offsets),
                runtime.pointer(polygon_buffer.ring_offsets),
                runtime.pointer(polygon_buffer.x),
                runtime.pointer(polygon_buffer.y),
                runtime.pointer(device_out),
                left.row_count,
                center_x,
                center_y,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, left.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        runtime.synchronize()
    finally:
        runtime.free(device_indices)
        runtime.free(device_mask)


def _launch_polygon_compacted(
    candidate_rows,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_buffer = right_state.families[GeometryFamily.POLYGON]
    device_out = runtime.allocate((candidate_count,), np.uint8)
    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_polygon_compacted_tagged"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            runtime.pointer(polygon_buffer.empty_mask),
            runtime.pointer(polygon_buffer.geometry_offsets),
            runtime.pointer(polygon_buffer.ring_offsets),
            runtime.pointer(polygon_buffer.x),
            runtime.pointer(polygon_buffer.y),
            FAMILY_TAGS[GeometryFamily.POLYGON],
            runtime.pointer(device_out),
            candidate_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    runtime.synchronize()
    return device_out


def _launch_multipolygon_dense(
    candidate_indices: np.ndarray,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    device_out,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    """Launch dense multipolygon PIP kernel writing into caller-owned *device_out*."""
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]
    n = left.row_count
    device_mask = runtime.allocate((n,), np.uint8, zero=True)
    device_indices = runtime.from_host(candidate_indices.astype(np.int32, copy=False))
    device_mask[device_indices] = np.uint8(1)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_dense"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(device_mask),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(multipolygon_buffer.empty_mask),
                runtime.pointer(multipolygon_buffer.geometry_offsets),
                runtime.pointer(multipolygon_buffer.part_offsets),
                runtime.pointer(multipolygon_buffer.ring_offsets),
                runtime.pointer(multipolygon_buffer.x),
                runtime.pointer(multipolygon_buffer.y),
                runtime.pointer(device_out),
                left.row_count,
                center_x,
                center_y,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, left.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        runtime.synchronize()
    finally:
        runtime.free(device_indices)
        runtime.free(device_mask)


def _launch_multipolygon_compacted(
    candidate_rows,
    candidate_count: int,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON]
    device_out = runtime.allocate((candidate_count,), np.uint8)
    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_multipolygon_compacted_tagged"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            runtime.pointer(multipolygon_buffer.empty_mask),
            runtime.pointer(multipolygon_buffer.geometry_offsets),
            runtime.pointer(multipolygon_buffer.part_offsets),
            runtime.pointer(multipolygon_buffer.ring_offsets),
            runtime.pointer(multipolygon_buffer.x),
            runtime.pointer(multipolygon_buffer.y),
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            runtime.pointer(device_out),
            candidate_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    runtime.synchronize()
    return device_out


def _scatter_compacted_hits(
    candidate_rows,
    compacted_hits,
    dense_out,
    candidate_count: int,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> None:
    runtime = get_cuda_runtime()
    kernel = _point_in_polygon_kernels(compute_type)["scatter_compacted_hits"]
    params = (
        (
            runtime.pointer(candidate_rows),
            runtime.pointer(compacted_hits),
            runtime.pointer(dense_out),
            candidate_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, candidate_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _launch_fused(
    points: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    """Single-kernel fused bounds check + PIP test for all rows."""
    runtime = get_cuda_runtime()
    left_state = points._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]

    has_polygon = GeometryFamily.POLYGON in right_state.families
    has_multipolygon = GeometryFamily.MULTIPOLYGON in right_state.families
    polygon_buffer = right_state.families[GeometryFamily.POLYGON] if has_polygon else None
    multipolygon_buffer = right_state.families[GeometryFamily.MULTIPOLYGON] if has_multipolygon else None

    # Ensure per-family bounds exist on device.  When the fused path skips
    # CPU compute_geometry_bounds, the device buffers have bounds=None.
    # Compute them directly on-device — this is much cheaper than the CPU
    # path that was the original bottleneck we're eliminating.
    for family, device_buffer in ((GeometryFamily.POLYGON, polygon_buffer), (GeometryFamily.MULTIPOLYGON, multipolygon_buffer)):
        if device_buffer is not None and device_buffer.bounds is None:
            row_count = right.families[family].row_count
            device_buffer.bounds = runtime.allocate((row_count, 4), np.float64)
            _launch_family_bounds_kernel(family, device_buffer, row_count=row_count)

    device_out = runtime.allocate((points.row_count,), np.uint8)

    kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_fused"]
    params = (
        (
            runtime.pointer(left_state.family_row_offsets),
            runtime.pointer(point_buffer.geometry_offsets),
            runtime.pointer(point_buffer.empty_mask),
            runtime.pointer(point_buffer.x),
            runtime.pointer(point_buffer.y),
            runtime.pointer(right_state.tags),
            runtime.pointer(right_state.family_row_offsets),
            # polygon family (null if absent)
            runtime.pointer(polygon_buffer.bounds if has_polygon else None),
            runtime.pointer(polygon_buffer.empty_mask if has_polygon else None),
            runtime.pointer(polygon_buffer.geometry_offsets if has_polygon else None),
            runtime.pointer(polygon_buffer.ring_offsets if has_polygon else None),
            runtime.pointer(polygon_buffer.x if has_polygon else None),
            runtime.pointer(polygon_buffer.y if has_polygon else None),
            # multipolygon family (null if absent)
            runtime.pointer(multipolygon_buffer.bounds if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.empty_mask if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.geometry_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.part_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.ring_offsets if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.x if has_multipolygon else None),
            runtime.pointer(multipolygon_buffer.y if has_multipolygon else None),
            # tags
            FAMILY_TAGS[GeometryFamily.POLYGON],
            FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            # output
            runtime.pointer(device_out),
            points.row_count,
            center_x,
            center_y,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
        ),
    )
    grid, block = runtime.launch_config(kernel, points.row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    runtime.synchronize()
    return device_out


def _launch_bounds_candidate_rows(
    points: OwnedGeometryArray,
    right: OwnedGeometryArray,
    compute_type: str = "double",
    center_x: float = 0.0,
    center_y: float = 0.0,
):
    runtime = get_cuda_runtime()
    left_state = points._ensure_device_state()
    right_state = right._ensure_device_state()
    point_buffer = left_state.families[GeometryFamily.POINT]
    polygon_bounds = (
        None
        if GeometryFamily.POLYGON not in right_state.families
        else right_state.families[GeometryFamily.POLYGON].bounds
    )
    multipolygon_bounds = (
        None
        if GeometryFamily.MULTIPOLYGON not in right_state.families
        else right_state.families[GeometryFamily.MULTIPOLYGON].bounds
    )
    device_mask = runtime.allocate((points.row_count,), np.uint8)
    try:
        kernel = _point_in_polygon_kernels(compute_type)["point_in_polygon_bounds_mask"]
        params = (
            (
                runtime.pointer(left_state.family_row_offsets),
                runtime.pointer(point_buffer.geometry_offsets),
                runtime.pointer(point_buffer.empty_mask),
                runtime.pointer(point_buffer.x),
                runtime.pointer(point_buffer.y),
                runtime.pointer(right_state.tags),
                runtime.pointer(right_state.family_row_offsets),
                runtime.pointer(polygon_bounds),
                runtime.pointer(multipolygon_bounds),
                FAMILY_TAGS[GeometryFamily.POLYGON],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
                runtime.pointer(device_mask),
                points.row_count,
                center_x,
                center_y,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
            ),
        )
        grid, block = runtime.launch_config(kernel, points.row_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        return compact_indices(device_mask)
    finally:
        runtime.free(device_mask)


def launch_point_region_candidate_rows(
    points: OwnedGeometryArray,
    regions: OwnedGeometryArray,
):
    """Return device-resident candidate rows for aligned point/region bounds hits."""
    return _launch_bounds_candidate_rows(points, regions)


def _evaluate_point_in_polygon_cpu(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
) -> np.ndarray:
    coarse = _evaluate_point_within_bounds(points, right)
    candidate_rows = np.flatnonzero(coarse == True)  # noqa: E712  # object-dtype array
    if candidate_rows.size == 0:
        return coarse

    point_values = points.to_shapely()
    polygon_values = right.geometry_array.to_shapely()
    refined = np.asarray(
        [
            bool(polygon_values[row_index].covers(point_values[row_index]))
            for row_index in candidate_rows
        ],
        dtype=bool,
    )
    coarse[candidate_rows] = refined
    return coarse


_log = logging.getLogger("vibespatial.kernels.point_in_polygon")

_last_gpu_substage_timings: dict[str, float] | None = None


def get_last_gpu_substage_timings() -> dict[str, float] | None:
    """Return sub-stage timing breakdown from the most recent GPU point-in-polygon call."""
    return _last_gpu_substage_timings


def _compute_pip_center(
    points: OwnedGeometryArray,
    right_array: OwnedGeometryArray,
) -> tuple[float, float]:
    """Compute the centroid of the combined coordinate extent for centering."""
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for owned in (points, right_array):
        for buffer in owned.families.values():
            if buffer.x.size > 0:
                all_x.append(buffer.x)
                all_y.append(buffer.y)
    if not all_x:
        return 0.0, 0.0
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    cx = (float(np.nanmin(combined_x)) + float(np.nanmax(combined_x))) * 0.5
    cy = (float(np.nanmin(combined_y)) + float(np.nanmax(combined_y))) * 0.5
    return cx, cy


def _evaluate_point_in_polygon_gpu(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
    *,
    strategy: str = "auto",
    return_device: bool = False,
) -> np.ndarray:
    global _last_gpu_substage_timings
    timings: dict[str, float] = {}

    right_array = right.geometry_array
    assert right_array is not None
    coarse = np.zeros(points.row_count, dtype=object)
    coarse[:] = False
    coarse[~points.validity | right.null_mask] = None

    # Determine compute precision from device profile.
    from vibespatial.adaptive_runtime import get_cached_snapshot
    snapshot = get_cached_snapshot()
    use_fp32 = not snapshot.device_profile.favors_native_fp64
    compute_type = "float" if use_fp32 else "double"

    # Compute center for coordinate centering.
    center_x, center_y = _compute_pip_center(points, right_array)

    t0 = perf_counter()
    points._ensure_device_state()
    timings["point_upload_s"] = perf_counter() - t0

    t0 = perf_counter()
    right_array._ensure_device_state()
    timings["polygon_upload_s"] = perf_counter() - t0

    selected_strategy = _select_gpu_strategy(
        points.row_count, strategy=strategy, right_array=right_array,
    )
    timings["strategy"] = selected_strategy

    if selected_strategy == "dense":
        t0 = perf_counter()
        coarse = _evaluate_point_within_bounds(points, right)
        timings["coarse_filter_s"] = perf_counter() - t0

        t0 = perf_counter()
        candidate_rows = np.flatnonzero(coarse == True)  # noqa: E712  # object-dtype array
        timings["candidate_mask_s"] = perf_counter() - t0
        timings["candidate_count"] = int(candidate_rows.size)
        timings["total_rows"] = int(points.row_count)

        if candidate_rows.size == 0:
            _last_gpu_substage_timings = timings
            return coarse

        t0 = perf_counter()
        rows_by_family = _candidate_rows_by_family(right_array, candidate_rows)
        timings["family_split_s"] = perf_counter() - t0
        if not rows_by_family:
            _last_gpu_substage_timings = timings
            return coarse

        t0 = perf_counter()
        runtime = get_cuda_runtime()
        device_dense_out = runtime.allocate((points.row_count,), np.uint8, zero=True)
        try:
            if GeometryFamily.POLYGON in rows_by_family:
                _launch_polygon_dense(rows_by_family[GeometryFamily.POLYGON], points, right_array, device_dense_out, compute_type=compute_type, center_x=center_x, center_y=center_y)
            if GeometryFamily.MULTIPOLYGON in rows_by_family:
                _launch_multipolygon_dense(rows_by_family[GeometryFamily.MULTIPOLYGON], points, right_array, device_dense_out, compute_type=compute_type, center_x=center_x, center_y=center_y)
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                _last_gpu_substage_timings = timings
                return device_dense_out
            dense_out = runtime.copy_device_to_host(device_dense_out)
        finally:
            if not return_device:
                runtime.free(device_dense_out)
        coarse[candidate_rows] = dense_out[candidate_rows].astype(bool, copy=False)
    elif selected_strategy == "compacted":
        # Ensure per-family bounds exist on device (same as fused/binned paths).
        runtime = get_cuda_runtime()
        right_state = right_array._ensure_device_state()
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if family in right_state.families:
                device_buffer = right_state.families[family]
                if device_buffer.bounds is None:
                    fam_row_count = right_array.families[family].row_count
                    device_buffer.bounds = runtime.allocate(
                        (fam_row_count, 4), np.float64,
                    )
                    _launch_family_bounds_kernel(
                        family, device_buffer, row_count=fam_row_count,
                    )
        t0 = perf_counter()
        candidate_rows = _launch_bounds_candidate_rows(points, right_array, compute_type=compute_type, center_x=center_x, center_y=center_y)
        timings["coarse_filter_s"] = perf_counter() - t0
        timings["candidate_mask_s"] = 0.0
        timings["candidate_count"] = int(candidate_rows.count)
        timings["total_rows"] = int(points.row_count)
        timings["family_split_s"] = 0.0

        if candidate_rows.count == 0:
            _last_gpu_substage_timings = timings
            return coarse

        runtime = get_cuda_runtime()
        device_dense_out = runtime.allocate((points.row_count,), np.uint8)
        device_dense_out[...] = 0
        try:
            t0 = perf_counter()
            if GeometryFamily.POLYGON in right_array.families:
                polygon_hits = _launch_polygon_compacted(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                try:
                    _scatter_compacted_hits(
                        candidate_rows.values,
                        polygon_hits,
                        device_dense_out,
                        candidate_rows.count,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                finally:
                    runtime.free(polygon_hits)
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                multipolygon_hits = _launch_multipolygon_compacted(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                try:
                    _scatter_compacted_hits(
                        candidate_rows.values,
                        multipolygon_hits,
                        device_dense_out,
                        candidate_rows.count,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                finally:
                    runtime.free(multipolygon_hits)
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                # Caller owns device_dense_out; free only candidate indices.
                runtime.free(candidate_rows.values)
                _last_gpu_substage_timings = timings
                return device_dense_out
            dense_out = runtime.copy_device_to_host(device_dense_out)
        finally:
            if not return_device:
                runtime.free(device_dense_out)
                runtime.free(candidate_rows.values)
        coarse[np.asarray(dense_out, dtype=bool)] = True
    elif selected_strategy == "binned":
        # Binned: bounds filter first, then dispatch PIP in work-balanced bins
        # with a block-per-pair kernel for the complex (>1024 vertex) bin.
        runtime = get_cuda_runtime()
        # Ensure per-family bounds exist on device (same as fused path).
        right_state = right_array._ensure_device_state()
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            if family in right_state.families:
                device_buffer = right_state.families[family]
                if device_buffer.bounds is None:
                    fam_row_count = right_array.families[family].row_count
                    device_buffer.bounds = runtime.allocate(
                        (fam_row_count, 4), np.float64,
                    )
                    _launch_family_bounds_kernel(
                        family, device_buffer, row_count=fam_row_count,
                    )
        t0 = perf_counter()
        candidate_rows = _launch_bounds_candidate_rows(
            points, right_array, compute_type=compute_type,
            center_x=center_x, center_y=center_y,
        )
        timings["coarse_filter_s"] = perf_counter() - t0
        timings["candidate_mask_s"] = 0.0
        timings["candidate_count"] = int(candidate_rows.count)
        timings["total_rows"] = int(points.row_count)
        timings["family_split_s"] = 0.0

        if candidate_rows.count == 0:
            _last_gpu_substage_timings = timings
            return coarse

        # Compute per-candidate work estimates on host
        t0 = perf_counter()
        candidate_rows_host = runtime.copy_device_to_host(candidate_rows.values)
        work_estimates = _compute_work_estimates_for_candidates(
            candidate_rows_host, right_array,
        )
        timings["work_estimation_s"] = perf_counter() - t0
        timings["work_cv"] = float(
            work_estimates.std() / max(work_estimates.mean(), 1e-9)
        )

        device_dense_out = runtime.allocate((points.row_count,), np.uint8)
        device_dense_out[...] = 0
        try:
            t0 = perf_counter()
            if GeometryFamily.POLYGON in right_array.families:
                polygon_hits = _binned_polygon_dispatch(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    work_estimates,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                try:
                    _scatter_compacted_hits(
                        candidate_rows.values,
                        polygon_hits,
                        device_dense_out,
                        candidate_rows.count,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                finally:
                    runtime.free(polygon_hits)
            if GeometryFamily.MULTIPOLYGON in right_array.families:
                multipolygon_hits = _binned_multipolygon_dispatch(
                    candidate_rows.values,
                    candidate_rows.count,
                    points,
                    right_array,
                    work_estimates,
                    compute_type=compute_type,
                    center_x=center_x,
                    center_y=center_y,
                )
                try:
                    _scatter_compacted_hits(
                        candidate_rows.values,
                        multipolygon_hits,
                        device_dense_out,
                        candidate_rows.count,
                        compute_type=compute_type,
                        center_x=center_x,
                        center_y=center_y,
                    )
                finally:
                    runtime.free(multipolygon_hits)
            timings["kernel_launch_and_sync_s"] = perf_counter() - t0
            if return_device:
                # Caller owns device_dense_out; free only candidate indices.
                runtime.free(candidate_rows.values)
                _last_gpu_substage_timings = timings
                return device_dense_out
            dense_out = runtime.copy_device_to_host(device_dense_out)
        finally:
            if not return_device:
                runtime.free(device_dense_out)
                runtime.free(candidate_rows.values)
        coarse[np.asarray(dense_out, dtype=bool)] = True
    else:
        # Fused: single kernel does bounds check + PIP in one launch.
        runtime = get_cuda_runtime()
        t0 = perf_counter()
        device_out = _launch_fused(points, right_array, compute_type=compute_type, center_x=center_x, center_y=center_y)
        timings["fused_kernel_s"] = perf_counter() - t0
        timings["coarse_filter_s"] = 0.0
        timings["candidate_mask_s"] = 0.0
        timings["family_split_s"] = 0.0
        timings["total_rows"] = int(points.row_count)
        if return_device:
            # Keep result on GPU — caller is responsible for freeing.
            _last_gpu_substage_timings = timings
            return device_out
        try:
            dense_out = runtime.copy_device_to_host(device_out)
        finally:
            runtime.free(device_out)
        null_mask = ~points.validity | right.null_mask
        # Normalize to object-dtype ndarray matching dense/compacted paths.
        coarse = np.empty(points.row_count, dtype=object)
        coarse[:] = False
        coarse[np.asarray(dense_out, dtype=bool)] = True
        coarse[null_mask] = None

    _last_gpu_substage_timings = timings
    _log.info(
        "point_in_polygon GPU substages: %s",
        " | ".join(
            f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in timings.items()
        ),
    )
    return coarse


@register_kernel_variant(
    "point_in_polygon",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.CPU,),
    supports_mixed=True,
    tags=("coarse-filter", "refine", "covers"),
)
def _point_in_polygon_cpu_variant(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
) -> np.ndarray:
    return _evaluate_point_in_polygon_cpu(points, right)


@register_kernel_variant(
    "point_in_polygon",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    geometry_families=("point", "polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    preferred_residency=Residency.DEVICE,
    tags=("coarse-filter", "refine", "cuda-python", "dense-or-compacted"),
)
def _point_in_polygon_gpu_variant(
    points: OwnedGeometryArray,
    right: NormalizedBoundsInput,
    *,
    strategy: str = "auto",
) -> np.ndarray:
    return _evaluate_point_in_polygon_gpu(points, right, strategy=strategy)


def point_in_polygon(
    points: PointSequence,
    polygons: PointSequence,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _return_device: bool = False,
) -> list[bool | None]:
    global _last_gpu_substage_timings

    t0 = perf_counter()
    left = coerce_geometry_array(
        points,
        arg_name="points",
        expected_families=(GeometryFamily.POINT,),
    )
    coerce_left_s = perf_counter() - t0

    # Coerce the right side to an OwnedGeometryArray cheaply (no bounds).
    # Full _normalize_right_input (including compute_geometry_bounds) is
    # deferred to the CPU path — the fused GPU kernel reads device-resident
    # bounds directly so the CPU bounds computation is redundant there.
    t0 = perf_counter()
    right_array = coerce_geometry_array(
        polygons,
        arg_name="polygons",
        expected_families=(GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON),
    )
    coerce_right_s = perf_counter() - t0

    if right_array.row_count != left.row_count:
        raise ValueError(
            f"point_in_polygon requires aligned inputs; "
            f"got {left.row_count} points and {right_array.row_count} polygon rows"
        )

    context = resolve_predicate_context(
        kernel_name="point_in_polygon",
        left=left,
        right=right_array,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    if context.runtime_selection.selected is ExecutionMode.GPU:
        # Build lightweight NormalizedBoundsInput without CPU bounds.
        right = NormalizedBoundsInput(
            bounds=np.empty(0),
            null_mask=~right_array.validity,
            empty_mask=np.zeros(right_array.row_count, dtype=bool),
            geometry_array=right_array,
        )
        t0 = perf_counter()
        left.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="point_in_polygon selected GPU execution",
        )
        right_array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="point_in_polygon selected GPU execution",
        )
        move_to_device_s = perf_counter() - t0

        gpu_out = _evaluate_point_in_polygon_gpu(
            left, right, return_device=_return_device,
        )

        # Merge outer timing into substage report
        if _last_gpu_substage_timings is not None:
            _last_gpu_substage_timings["coerce_left_s"] = coerce_left_s
            _last_gpu_substage_timings["coerce_right_s"] = coerce_right_s
            _last_gpu_substage_timings["move_to_device_s"] = move_to_device_s

        # _return_device: keep result on GPU as CuPy bool array for
        # zero-copy pipelines (feeds into device_take).
        if _return_device:
            import cupy as _cp
            return _cp.asarray(gpu_out, dtype=_cp.bool_)

        # All strategies now return a normalized object-dtype ndarray.
        return _to_python_result(gpu_out)

    # CPU path: full normalize with bounds (needed by CPU coarse filter).
    t0 = perf_counter()
    right = _normalize_right_input(polygons, expected_len=left.row_count)
    normalize_right_s = perf_counter() - t0
    if _last_gpu_substage_timings is not None:
        _last_gpu_substage_timings["normalize_right_s"] = normalize_right_s
    cpu_out = _point_in_polygon_cpu_variant(left, right)
    if _return_device:
        # CPU fallback for _return_device: return numpy bool array
        return np.array([bool(v) if v is not None else False for v in cpu_out], dtype=bool)
    return _to_python_result(cpu_out)


