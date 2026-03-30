"""CUDA kernel source for point-in-polygon predicate.

Contains NVRTC kernel source templates for:
- Point-in-polygon per-thread kernel (dense, compacted, fused variants)
- Block-per-pair cooperative kernel (for complex polygons >1024 verts)

Extracted from point_in_polygon.py -- dispatch logic remains there.
"""
from __future__ import annotations

from vibespatial.cuda.device_functions.point_on_segment import POINT_ON_SEGMENT_DEVICE

_POINT_IN_POLYGON_KERNEL_SOURCE_TEMPLATE = POINT_ON_SEGMENT_DEVICE + """
typedef {compute_type} compute_t;

#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))
#define PIP_BOUNDARY_TOLERANCE 1e-7

extern "C" __device__ inline compute_t vibespatial_abs(compute_t value) {{
  return value < (compute_t)0.0 ? -value : value;
}}

extern "C" __device__ inline compute_t vibespatial_max(compute_t left, compute_t right) {{
  return left > right ? left : right;
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
    if (vs_point_on_segment((double)px, (double)py, (double)ax, (double)ay, (double)bx, (double)by, PIP_BOUNDARY_TOLERANCE)) {{
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

_PIP_BLOCK_PER_PAIR_KERNEL_SOURCE_TEMPLATE = POINT_ON_SEGMENT_DEVICE + """
typedef {compute_type} compute_t;

#define CX(val) ((compute_t)((val) - center_x))
#define CY(val) ((compute_t)((val) - center_y))
#define PIP_BOUNDARY_TOLERANCE 1e-7

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

            if (vs_point_on_segment((double)px, (double)py, (double)ax, (double)ay, (double)bx, (double)by, PIP_BOUNDARY_TOLERANCE)) {{
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

            if (vs_point_on_segment((double)px, (double)py, (double)ax, (double)ay, (double)bx, (double)by, PIP_BOUNDARY_TOLERANCE)) {{
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
