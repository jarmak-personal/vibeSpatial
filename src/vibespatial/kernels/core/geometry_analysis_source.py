"""CUDA kernel source for geometry bounds computation.

Contains NVRTC kernel source templates for per-family bounds kernels
(thread-per-row) and cooperative bounds kernels (block-per-geometry
warp-level reduction).

Extracted from geometry_analysis.py -- dispatch logic remains there.
"""
from __future__ import annotations

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS

_BOUNDS_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

extern "C" __device__ inline double vibespatial_nan() {{{{
  return __longlong_as_double(0x7ff8000000000000ULL);
}}}}

extern "C" __device__ inline void write_nan_bounds(double* __restrict__ out, int row) {{{{
  const int base = row * 4;
  const double nan_value = vibespatial_nan();
  out[base + 0] = nan_value;
  out[base + 1] = nan_value;
  out[base + 2] = nan_value;
  out[base + 3] = nan_value;
}}}}

extern "C" __global__ void bounds_simple(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  const int coord_start = geometry_offsets[row];
  const int coord_end = geometry_offsets[row + 1];
  if (coord_end <= coord_start) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  compute_t minx = (compute_t)x[coord_start];
  compute_t miny = (compute_t)y[coord_start];
  compute_t maxx = minx;
  compute_t maxy = miny;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {{{{
    const compute_t xv = (compute_t)x[coord];
    const compute_t yv = (compute_t)y[coord];
    minx = xv < minx ? xv : minx;
    miny = yv < miny ? yv : miny;
    maxx = xv > maxx ? xv : maxx;
    maxy = yv > maxy ? yv : maxy;
  }}}}
  const int base = row * 4;
  out[base + 0] = (double)minx;
  out[base + 1] = (double)miny;
  out[base + 2] = (double)maxx;
  out[base + 3] = (double)maxy;
}}}}

extern "C" __global__ void bounds_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  const int ring_start = geometry_offsets[row];
  const int ring_end = geometry_offsets[row + 1];
  const int coord_start = ring_offsets[ring_start];
  const int coord_end = ring_offsets[ring_end];
  if (coord_end <= coord_start) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  compute_t minx = (compute_t)x[coord_start];
  compute_t miny = (compute_t)y[coord_start];
  compute_t maxx = minx;
  compute_t maxy = miny;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {{{{
    const compute_t xv = (compute_t)x[coord];
    const compute_t yv = (compute_t)y[coord];
    minx = xv < minx ? xv : minx;
    miny = yv < miny ? yv : miny;
    maxx = xv > maxx ? xv : maxx;
    maxy = yv > maxy ? yv : maxy;
  }}}}
  const int base = row * 4;
  out[base + 0] = (double)minx;
  out[base + 1] = (double)miny;
  out[base + 2] = (double)maxx;
  out[base + 3] = (double)maxy;
}}}}

extern "C" __global__ void bounds_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  const int part_start = geometry_offsets[row];
  const int part_end = geometry_offsets[row + 1];
  const int coord_start = part_offsets[part_start];
  const int coord_end = part_offsets[part_end];
  if (coord_end <= coord_start) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  compute_t minx = (compute_t)x[coord_start];
  compute_t miny = (compute_t)y[coord_start];
  compute_t maxx = minx;
  compute_t maxy = miny;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {{{{
    const compute_t xv = (compute_t)x[coord];
    const compute_t yv = (compute_t)y[coord];
    minx = xv < minx ? xv : minx;
    miny = yv < miny ? yv : miny;
    maxx = xv > maxx ? xv : maxx;
    maxy = yv > maxy ? yv : maxy;
  }}}}
  const int base = row * 4;
  out[base + 0] = (double)minx;
  out[base + 1] = (double)miny;
  out[base + 2] = (double)maxx;
  out[base + 3] = (double)maxy;
}}}}

extern "C" __global__ void bounds_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  const int polygon_start = geometry_offsets[row];
  const int polygon_end = geometry_offsets[row + 1];
  const int ring_start = part_offsets[polygon_start];
  const int ring_end = part_offsets[polygon_end];
  const int coord_start = ring_offsets[ring_start];
  const int coord_end = ring_offsets[ring_end];
  if (coord_end <= coord_start) {{{{
    write_nan_bounds(out, row);
    return;
  }}}}
  compute_t minx = (compute_t)x[coord_start];
  compute_t miny = (compute_t)y[coord_start];
  compute_t maxx = minx;
  compute_t maxy = miny;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {{{{
    const compute_t xv = (compute_t)x[coord];
    const compute_t yv = (compute_t)y[coord];
    minx = xv < minx ? xv : minx;
    miny = yv < miny ? yv : miny;
    maxx = xv > maxx ? xv : maxx;
    maxy = yv > maxy ? yv : maxy;
  }}}}
  const int base = row * 4;
  out[base + 0] = (double)minx;
  out[base + 1] = (double)miny;
  out[base + 2] = (double)maxx;
  out[base + 3] = (double)maxy;
}}}}

extern "C" __global__ void scatter_mixed_bounds(
    const unsigned char* __restrict__ validity,
    const signed char* __restrict__ tags,
    const int* __restrict__ family_row_offsets,
    const double* __restrict__ point_bounds,
    const double* __restrict__ linestring_bounds,
    const double* __restrict__ polygon_bounds,
    const double* __restrict__ multipoint_bounds,
    const double* __restrict__ multilinestring_bounds,
    const double* __restrict__ multipolygon_bounds,
    double* __restrict__ out_bounds,
    int row_count
) {{{{
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (!validity[row]) {{{{
    write_nan_bounds(out_bounds, row);
    return;
  }}}}
  const int family_row = family_row_offsets[row];
  const int base = row * 4;
  const int family_base = family_row * 4;
  const double* source = nullptr;
  switch (tags[row]) {{{{
    case {point_tag}:
      source = point_bounds;
      break;
    case {line_tag}:
      source = linestring_bounds;
      break;
    case {poly_tag}:
      source = polygon_bounds;
      break;
    case {mpoint_tag}:
      source = multipoint_bounds;
      break;
    case {mline_tag}:
      source = multilinestring_bounds;
      break;
    case {mpoly_tag}:
      source = multipolygon_bounds;
      break;
    default:
      write_nan_bounds(out_bounds, row);
      return;
  }}}}
  if (source == nullptr || family_row < 0) {{{{
    write_nan_bounds(out_bounds, row);
    return;
  }}}}
  out_bounds[base + 0] = source[family_base + 0];
  out_bounds[base + 1] = source[family_base + 1];
  out_bounds[base + 2] = source[family_base + 2];
  out_bounds[base + 3] = source[family_base + 3];
}}}}
"""


_BOUNDS_COOPERATIVE_KERNEL_SOURCE_TEMPLATE = """
/* Bounds kernels always use fp64: they are memory-bound (not compute-bound)
   and fp32 rounding can shrink bounds, causing false negatives in spatial
   filtering.  The compute_type parameter is accepted for cache-key
   consistency but bounds computation is always double. */
typedef {compute_type} compute_t;

extern "C" __device__ inline double vibespatial_nan_coop() {{{{
  return __longlong_as_double(0x7ff8000000000000ULL);
}}}}

extern "C" __device__ inline void write_nan_bounds_coop(double* __restrict__ out, int row) {{{{
  const int base = row * 4;
  const double nan_value = vibespatial_nan_coop();
  out[base + 0] = nan_value;
  out[base + 1] = nan_value;
  out[base + 2] = nan_value;
  out[base + 3] = nan_value;
}}}}

/* -----------------------------------------------------------------
 * Warp + block cooperative min/max reduction for bounds.
 * 1 block = 1 geometry.  Threads stride over coordinates, then
 * reduce via __shfl_down_sync + shared memory across warps.
 * ----------------------------------------------------------------- */

extern "C" __global__ __launch_bounds__(256, 4) void bounds_polygon_cooperative(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    if (threadIdx.x == 0) {{{{
      write_nan_bounds_coop(out, row);
    }}}}
    return;
  }}}}
  const int ring_start = geometry_offsets[row];
  const int ring_end = geometry_offsets[row + 1];
  const int coord_start = ring_offsets[ring_start];
  const int coord_end = ring_offsets[ring_end];
  const int n_coords = coord_end - coord_start;
  if (n_coords <= 0) {{{{
    if (threadIdx.x == 0) {{{{
      write_nan_bounds_coop(out, row);
    }}}}
    return;
  }}}}

  /* Each thread accumulates min/max over its stride of coordinates */
  double local_minx = 1e308;
  double local_miny = 1e308;
  double local_maxx = -1e308;
  double local_maxy = -1e308;
  for (int i = threadIdx.x; i < n_coords; i += blockDim.x) {{{{
    const int coord = coord_start + i;
    const double xv = x[coord];
    const double yv = y[coord];
    local_minx = xv < local_minx ? xv : local_minx;
    local_miny = yv < local_miny ? yv : local_miny;
    local_maxx = xv > local_maxx ? xv : local_maxx;
    local_maxy = yv > local_maxy ? yv : local_maxy;
  }}}}

  /* Warp-level reduction via shuffle */
  const unsigned int FULL_MASK = 0xFFFFFFFF;
  for (int offset = 16; offset > 0; offset >>= 1) {{{{
    double other_minx = __shfl_down_sync(FULL_MASK, local_minx, offset);
    double other_miny = __shfl_down_sync(FULL_MASK, local_miny, offset);
    double other_maxx = __shfl_down_sync(FULL_MASK, local_maxx, offset);
    double other_maxy = __shfl_down_sync(FULL_MASK, local_maxy, offset);
    local_minx = other_minx < local_minx ? other_minx : local_minx;
    local_miny = other_miny < local_miny ? other_miny : local_miny;
    local_maxx = other_maxx > local_maxx ? other_maxx : local_maxx;
    local_maxy = other_maxy > local_maxy ? other_maxy : local_maxy;
  }}}}

  /* Block-level reduction across warps via shared memory */
  __shared__ double warp_minx[8];
  __shared__ double warp_miny[8];
  __shared__ double warp_maxx[8];
  __shared__ double warp_maxy[8];
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  if (lane_id == 0) {{{{
    warp_minx[warp_id] = local_minx;
    warp_miny[warp_id] = local_miny;
    warp_maxx[warp_id] = local_maxx;
    warp_maxy[warp_id] = local_maxy;
  }}}}
  __syncthreads();

  /* Thread 0 reduces across warps and writes output */
  if (threadIdx.x == 0) {{{{
    double final_minx = warp_minx[0];
    double final_miny = warp_miny[0];
    double final_maxx = warp_maxx[0];
    double final_maxy = warp_maxy[0];
    const int num_warps = (blockDim.x + 31) >> 5;
    for (int w = 1; w < num_warps; ++w) {{{{
      final_minx = warp_minx[w] < final_minx ? warp_minx[w] : final_minx;
      final_miny = warp_miny[w] < final_miny ? warp_miny[w] : final_miny;
      final_maxx = warp_maxx[w] > final_maxx ? warp_maxx[w] : final_maxx;
      final_maxy = warp_maxy[w] > final_maxy ? warp_maxy[w] : final_maxy;
    }}}}
    const int base = row * 4;
    out[base + 0] = final_minx;
    out[base + 1] = final_miny;
    out[base + 2] = final_maxx;
    out[base + 3] = final_maxy;
  }}}}
}}}}

extern "C" __global__ __launch_bounds__(256, 4) void bounds_multipolygon_cooperative(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ empty_mask,
    double* __restrict__ out,
    int row_count
) {{{{
  const int row = blockIdx.x;
  if (row >= row_count) {{{{
    return;
  }}}}
  if (empty_mask[row]) {{{{
    if (threadIdx.x == 0) {{{{
      write_nan_bounds_coop(out, row);
    }}}}
    return;
  }}}}
  const int polygon_start = geometry_offsets[row];
  const int polygon_end = geometry_offsets[row + 1];
  const int ring_start = part_offsets[polygon_start];
  const int ring_end = part_offsets[polygon_end];
  const int coord_start = ring_offsets[ring_start];
  const int coord_end = ring_offsets[ring_end];
  const int n_coords = coord_end - coord_start;
  if (n_coords <= 0) {{{{
    if (threadIdx.x == 0) {{{{
      write_nan_bounds_coop(out, row);
    }}}}
    return;
  }}}}

  /* Each thread accumulates min/max over its stride of coordinates */
  double local_minx = 1e308;
  double local_miny = 1e308;
  double local_maxx = -1e308;
  double local_maxy = -1e308;
  for (int i = threadIdx.x; i < n_coords; i += blockDim.x) {{{{
    const int coord = coord_start + i;
    const double xv = x[coord];
    const double yv = y[coord];
    local_minx = xv < local_minx ? xv : local_minx;
    local_miny = yv < local_miny ? yv : local_miny;
    local_maxx = xv > local_maxx ? xv : local_maxx;
    local_maxy = yv > local_maxy ? yv : local_maxy;
  }}}}

  /* Warp-level reduction via shuffle */
  const unsigned int FULL_MASK = 0xFFFFFFFF;
  for (int offset = 16; offset > 0; offset >>= 1) {{{{
    double other_minx = __shfl_down_sync(FULL_MASK, local_minx, offset);
    double other_miny = __shfl_down_sync(FULL_MASK, local_miny, offset);
    double other_maxx = __shfl_down_sync(FULL_MASK, local_maxx, offset);
    double other_maxy = __shfl_down_sync(FULL_MASK, local_maxy, offset);
    local_minx = other_minx < local_minx ? other_minx : local_minx;
    local_miny = other_miny < local_miny ? other_miny : local_miny;
    local_maxx = other_maxx > local_maxx ? other_maxx : local_maxx;
    local_maxy = other_maxy > local_maxy ? other_maxy : local_maxy;
  }}}}

  /* Block-level reduction across warps via shared memory */
  __shared__ double warp_minx[8];
  __shared__ double warp_miny[8];
  __shared__ double warp_maxx[8];
  __shared__ double warp_maxy[8];
  const int warp_id = threadIdx.x >> 5;
  const int lane_id = threadIdx.x & 31;
  if (lane_id == 0) {{{{
    warp_minx[warp_id] = local_minx;
    warp_miny[warp_id] = local_miny;
    warp_maxx[warp_id] = local_maxx;
    warp_maxy[warp_id] = local_maxy;
  }}}}
  __syncthreads();

  /* Thread 0 reduces across warps and writes output */
  if (threadIdx.x == 0) {{{{
    double final_minx = warp_minx[0];
    double final_miny = warp_miny[0];
    double final_maxx = warp_maxx[0];
    double final_maxy = warp_maxy[0];
    const int num_warps = (blockDim.x + 31) >> 5;
    for (int w = 1; w < num_warps; ++w) {{{{
      final_minx = warp_minx[w] < final_minx ? warp_minx[w] : final_minx;
      final_miny = warp_miny[w] < final_miny ? warp_miny[w] : final_miny;
      final_maxx = warp_maxx[w] > final_maxx ? warp_maxx[w] : final_maxx;
      final_maxy = warp_maxy[w] > final_maxy ? warp_maxy[w] : final_maxy;
    }}}}
    const int base = row * 4;
    out[base + 0] = final_minx;
    out[base + 1] = final_miny;
    out[base + 2] = final_maxx;
    out[base + 3] = final_maxy;
  }}}}
}}}}
"""

_BOUNDS_KERNEL_NAMES = (
    "bounds_simple",
    "bounds_polygon",
    "bounds_multilinestring",
    "bounds_multipolygon",
    "scatter_mixed_bounds",
)

_BOUNDS_COOPERATIVE_KERNEL_NAMES = (
    "bounds_polygon_cooperative",
    "bounds_multipolygon_cooperative",
)

_COOPERATIVE_BOUNDS_THRESHOLD = 64  # avg coords per geometry to switch to cooperative


def _format_bounds_kernel_source(compute_type: str = "double") -> str:
    return _BOUNDS_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        point_tag=FAMILY_TAGS[GeometryFamily.POINT],
        line_tag=FAMILY_TAGS[GeometryFamily.LINESTRING],
        poly_tag=FAMILY_TAGS[GeometryFamily.POLYGON],
        mpoint_tag=FAMILY_TAGS[GeometryFamily.MULTIPOINT],
        mline_tag=FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
        mpoly_tag=FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
    )


def _format_cooperative_bounds_kernel_source(compute_type: str = "double") -> str:
    return _BOUNDS_COOPERATIVE_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
    )
