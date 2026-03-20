from __future__ import annotations

import math

import numpy as np

from vibespatial.adaptive_runtime import plan_kernel_dispatch
from vibespatial.cuda_runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    get_cuda_runtime,
    make_kernel_cache_key,
)
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.kernel_registry import register_kernel_variant
from vibespatial.owned_geometry import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DeviceFamilyGeometryBuffer,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.precision import KernelClass, PrecisionMode, normalize_precision_mode
from vibespatial.residency import Residency, TransferTrigger
from vibespatial.runtime import ExecutionMode, RuntimeSelection

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


_BOUNDS_KERNEL_SOURCE = _format_bounds_kernel_source("double")
_BOUNDS_COOPERATIVE_KERNEL_SOURCE = _format_cooperative_bounds_kernel_source("double")


def _family_bounds_scalar(buffer: FamilyGeometryBuffer, row_index: int) -> tuple[float, float, float, float]:
    if bool(buffer.empty_mask[row_index]):
        return (math.nan, math.nan, math.nan, math.nan)

    if buffer.family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        start = int(buffer.geometry_offsets[row_index])
        end = int(buffer.geometry_offsets[row_index + 1])
        xs = buffer.x[start:end]
        ys = buffer.y[start:end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.POLYGON:
        ring_start = int(buffer.geometry_offsets[row_index])
        ring_end = int(buffer.geometry_offsets[row_index + 1])
        coord_start = int(buffer.ring_offsets[ring_start])
        coord_end = int(buffer.ring_offsets[ring_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.MULTILINESTRING:
        part_start = int(buffer.geometry_offsets[row_index])
        part_end = int(buffer.geometry_offsets[row_index + 1])
        coord_start = int(buffer.part_offsets[part_start])
        coord_end = int(buffer.part_offsets[part_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_start = int(buffer.geometry_offsets[row_index])
        polygon_end = int(buffer.geometry_offsets[row_index + 1])
        ring_start = int(buffer.part_offsets[polygon_start])
        ring_end = int(buffer.part_offsets[polygon_end])
        coord_start = int(buffer.ring_offsets[ring_start])
        coord_end = int(buffer.ring_offsets[ring_end])
        xs = buffer.x[coord_start:coord_end]
        ys = buffer.y[coord_start:coord_end]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    raise NotImplementedError(f"unsupported family: {buffer.family.value}")


def _assemble_cached_bounds(geometry_array: OwnedGeometryArray) -> np.ndarray | None:
    if not geometry_array.families:
        return np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    if any(buffer.bounds is None for buffer in geometry_array.families.values()):
        return None
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for family, buffer in geometry_array.families.items():
        row_indexes = np.flatnonzero(geometry_array.tags == FAMILY_TAGS[family])
        if row_indexes.size == 0:
            continue
        bounds[row_indexes] = buffer.bounds
    return bounds


def _slice_bounds_vectorized(
    x: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    empty_mask: np.ndarray,
) -> np.ndarray:
    result = np.full((starts.size, 4), np.nan, dtype=np.float64)
    if starts.size == 0 or x.size == 0:
        return result
    non_empty = ~empty_mask
    if not np.any(non_empty):
        return result
    row_ids = np.repeat(np.flatnonzero(non_empty), (ends - starts)[non_empty])
    min_x = np.full(starts.size, math.inf, dtype=np.float64)
    min_y = np.full(starts.size, math.inf, dtype=np.float64)
    max_x = np.full(starts.size, -math.inf, dtype=np.float64)
    max_y = np.full(starts.size, -math.inf, dtype=np.float64)
    np.minimum.at(min_x, row_ids, x)
    np.minimum.at(min_y, row_ids, y)
    np.maximum.at(max_x, row_ids, x)
    np.maximum.at(max_y, row_ids, y)
    result[non_empty, 0] = min_x[non_empty]
    result[non_empty, 1] = min_y[non_empty]
    result[non_empty, 2] = max_x[non_empty]
    result[non_empty, 3] = max_y[non_empty]
    return result


def _family_bounds_vectorized(buffer: FamilyGeometryBuffer) -> np.ndarray:
    if buffer.row_count == 0:
        return np.empty((0, 4), dtype=np.float64)
    if buffer.family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        starts = buffer.geometry_offsets[:-1].astype(np.int64, copy=False)
        ends = buffer.geometry_offsets[1:].astype(np.int64, copy=False)
        return _slice_bounds_vectorized(
            buffer.x,
            buffer.y,
            starts,
            ends,
            buffer.empty_mask,
        )
    if buffer.family is GeometryFamily.POLYGON:
        starts = buffer.ring_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        ends = buffer.ring_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        return _slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    if buffer.family is GeometryFamily.MULTILINESTRING:
        starts = buffer.part_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        ends = buffer.part_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        return _slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    if buffer.family is GeometryFamily.MULTIPOLYGON:
        polygon_starts = buffer.part_offsets[buffer.geometry_offsets[:-1]].astype(np.int64, copy=False)
        polygon_ends = buffer.part_offsets[buffer.geometry_offsets[1:]].astype(np.int64, copy=False)
        starts = buffer.ring_offsets[polygon_starts].astype(np.int64, copy=False)
        ends = buffer.ring_offsets[polygon_ends].astype(np.int64, copy=False)
        return _slice_bounds_vectorized(buffer.x, buffer.y, starts, ends, buffer.empty_mask)
    raise NotImplementedError(f"unsupported family: {buffer.family.value}")


def _compute_geometry_bounds_cpu_scalar(geometry_array: OwnedGeometryArray) -> np.ndarray:
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for row_index in range(geometry_array.row_count):
        if not bool(geometry_array.validity[row_index]):
            continue
        family = TAG_FAMILIES[int(geometry_array.tags[row_index])]
        family_buffer = geometry_array.families[family]
        family_row = int(geometry_array.family_row_offsets[row_index])
        bounds[row_index] = np.asarray(_family_bounds_scalar(family_buffer, family_row), dtype=np.float64)
    return bounds


@register_kernel_variant(
    "compute_geometry_bounds",
    "cpu-vectorized",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
    tags=("vectorized",),
)
def _compute_geometry_bounds_cpu_vectorized(geometry_array: OwnedGeometryArray) -> np.ndarray:
    cached = _assemble_cached_bounds(geometry_array)
    if cached is not None:
        return cached
    family_bounds = {
        family: _family_bounds_vectorized(buffer)
        for family, buffer in geometry_array.families.items()
    }
    bounds = np.full((geometry_array.row_count, 4), np.nan, dtype=np.float64)
    for family, local_bounds in family_bounds.items():
        row_indexes = np.flatnonzero(geometry_array.tags == FAMILY_TAGS[family])
        if row_indexes.size == 0:
            continue
        bounds[row_indexes] = local_bounds[geometry_array.family_row_offsets[row_indexes]]
    geometry_array.cache_bounds(bounds)
    return bounds


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

from vibespatial.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("geometry-bounds", _BOUNDS_KERNEL_SOURCE, _BOUNDS_KERNEL_NAMES),
    ("geometry-bounds-cooperative", _BOUNDS_COOPERATIVE_KERNEL_SOURCE, _BOUNDS_COOPERATIVE_KERNEL_NAMES),
])


def _bounds_kernels(compute_type: str = "double"):
    source = _format_bounds_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"geometry-bounds-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_BOUNDS_KERNEL_NAMES,
    )


def _bounds_cooperative_kernels(compute_type: str = "double"):
    source = _format_cooperative_bounds_kernel_source(compute_type)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"geometry-bounds-cooperative-{compute_type}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_BOUNDS_COOPERATIVE_KERNEL_NAMES,
    )


def _family_kernel_name(family: GeometryFamily) -> str:
    if family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        return "bounds_simple"
    if family is GeometryFamily.POLYGON:
        return "bounds_polygon"
    if family is GeometryFamily.MULTILINESTRING:
        return "bounds_multilinestring"
    if family is GeometryFamily.MULTIPOLYGON:
        return "bounds_multipolygon"
    raise NotImplementedError(f"unsupported family: {family.value}")


def _launch_family_bounds_kernel(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    *,
    row_count: int,
    compute_type: str = "double",
) -> None:
    if row_count == 0:
        return
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernel = _bounds_kernels(compute_type)[_family_kernel_name(family)]
    if family in {GeometryFamily.POINT, GeometryFamily.LINESTRING, GeometryFamily.MULTIPOINT}:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
    elif family is GeometryFamily.POLYGON:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    elif family is GeometryFamily.MULTILINESTRING:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    else:
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    grid, block = runtime.launch_config(kernel, row_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _avg_coords_per_geometry(buffer: FamilyGeometryBuffer) -> float:
    """Return average coordinate count per geometry for cooperative dispatch heuristic."""
    if buffer.row_count == 0:
        return 0.0
    return float(len(buffer.x)) / float(buffer.row_count)


def _launch_family_bounds_cooperative(
    family: GeometryFamily,
    device_buffer: DeviceFamilyGeometryBuffer,
    *,
    row_count: int,
    compute_type: str = "double",
) -> None:
    """Launch cooperative (block-per-geometry) bounds kernel for polygon/multipolygon."""
    if row_count == 0:
        return
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    cooperative_kernels = _bounds_cooperative_kernels(compute_type)
    if family is GeometryFamily.POLYGON:
        kernel = cooperative_kernels["bounds_polygon_cooperative"]
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    elif family is GeometryFamily.MULTIPOLYGON:
        kernel = cooperative_kernels["bounds_multipolygon_cooperative"]
        params = (
            (
                ptr(device_buffer.x),
                ptr(device_buffer.y),
                ptr(device_buffer.geometry_offsets),
                ptr(device_buffer.part_offsets),
                ptr(device_buffer.ring_offsets),
                ptr(device_buffer.empty_mask),
                ptr(device_buffer.bounds),
                row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
    else:
        raise ValueError(f"cooperative bounds not supported for family {family.value}")
    # 1 block per geometry; fixed at 256 to match __launch_bounds__(256, 4)
    # and shared memory sized for 8 warps (256 / 32).
    grid = (row_count, 1, 1)
    block = (256, 1, 1)
    runtime.launch(kernel, grid=grid, block=block, params=params)


def _compute_geometry_bounds_gpu_impl(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
) -> np.ndarray:
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    state = geometry_array._ensure_device_state()
    temp_bounds: list[tuple[GeometryFamily, DeviceFamilyGeometryBuffer, object]] = []
    _cooperative_families = frozenset({GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON})
    try:
        for family, device_buffer in state.families.items():
            if device_buffer.bounds is None:
                host_buffer = geometry_array.families[family]
                device_buffer.bounds = runtime.allocate((host_buffer.row_count, 4), np.float64)
                temp_bounds.append((family, device_buffer, device_buffer.bounds))
                # Use cooperative (block-per-geometry) kernel when family supports it
                # and geometries are complex enough to benefit from warp-level reduction.
                use_cooperative = (
                    family in _cooperative_families
                    and _avg_coords_per_geometry(host_buffer) >= _COOPERATIVE_BOUNDS_THRESHOLD
                )
                if use_cooperative:
                    _launch_family_bounds_cooperative(
                        family, device_buffer,
                        row_count=host_buffer.row_count,
                        compute_type=compute_type,
                    )
                else:
                    _launch_family_bounds_kernel(
                        family, device_buffer,
                        row_count=host_buffer.row_count,
                        compute_type=compute_type,
                    )
        out_bounds = runtime.allocate((geometry_array.row_count, 4), np.float64)
        try:
            kernel = _bounds_kernels(compute_type)["scatter_mixed_bounds"]
            params = (
                (
                    ptr(state.validity),
                    ptr(state.tags),
                    ptr(state.family_row_offsets),
                    0 if GeometryFamily.POINT not in state.families else ptr(state.families[GeometryFamily.POINT].bounds),
                    0 if GeometryFamily.LINESTRING not in state.families else ptr(state.families[GeometryFamily.LINESTRING].bounds),
                    0 if GeometryFamily.POLYGON not in state.families else ptr(state.families[GeometryFamily.POLYGON].bounds),
                    0 if GeometryFamily.MULTIPOINT not in state.families else ptr(state.families[GeometryFamily.MULTIPOINT].bounds),
                    0 if GeometryFamily.MULTILINESTRING not in state.families else ptr(state.families[GeometryFamily.MULTILINESTRING].bounds),
                    0 if GeometryFamily.MULTIPOLYGON not in state.families else ptr(state.families[GeometryFamily.MULTIPOLYGON].bounds),
                    ptr(out_bounds),
                    geometry_array.row_count,
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
                    KERNEL_PARAM_I32,
                ),
            )
            grid, block = runtime.launch_config(kernel, geometry_array.row_count)
            runtime.launch(kernel, grid=grid, block=block, params=params)
            runtime.synchronize()
            bounds = np.empty((geometry_array.row_count, 4), dtype=np.float64)
            runtime.copy_device_to_host(out_bounds, bounds)
        except Exception:
            runtime.free(out_bounds)
            raise
        # Cache per-row device bounds instead of freeing — avoids
        # recomputation for subsequent device-side bbox queries (dwithin).
        state.row_bounds = out_bounds
        geometry_array.cache_bounds(bounds)
        for family, _, device_bounds in temp_bounds:
            geometry_array.cache_device_bounds(family, device_bounds)
        return bounds
    except Exception:
        for _, device_buffer, device_bounds in temp_bounds:
            device_buffer.bounds = None
            runtime.free(device_bounds)
        raise


@register_kernel_variant(
    "compute_geometry_bounds",
    "gpu-cuda-python",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python", "family-specialized"),
)
def _compute_geometry_bounds_gpu(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
) -> np.ndarray:
    return _compute_geometry_bounds_gpu_impl(geometry_array, compute_type=compute_type)


def _resolve_bounds_plan(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str,
    precision: PrecisionMode | str,
) -> tuple[RuntimeSelection, PrecisionMode]:
    requested_mode = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    geometry_families = tuple(sorted(family.value for family in geometry_array.families))
    plan = plan_kernel_dispatch(
        kernel_name="compute_geometry_bounds",
        kernel_class=KernelClass.COARSE,
        row_count=geometry_array.row_count,
        geometry_families=geometry_families,
        mixed_geometry=len(geometry_families) > 1,
        current_residency=geometry_array.residency,
        requested_mode=requested_mode,
        requested_precision=precision,
    )
    runtime_selection = plan.runtime_selection
    if plan.variant is None:
        if requested_mode is ExecutionMode.GPU:
            raise NotImplementedError("compute_geometry_bounds has no GPU variant registered yet")
        if runtime_selection.selected is ExecutionMode.GPU:
            runtime_selection = RuntimeSelection(
                requested=requested_mode,
                selected=ExecutionMode.CPU,
                reason="compute_geometry_bounds has no GPU variant registered; using explicit CPU fallback",
            )
    return runtime_selection, plan.precision_plan.compute_precision


def compute_geometry_bounds(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    normalize_precision_mode(precision)
    runtime_selection, compute_precision = _resolve_bounds_plan(
        geometry_array,
        dispatch_mode=dispatch_mode,
        precision=precision,
    )
    geometry_array.record_runtime_selection(runtime_selection)
    if runtime_selection.selected is ExecutionMode.GPU:
        geometry_array.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="compute_geometry_bounds selected GPU execution",
        )
        # Bounds always use fp64 compute: they are memory-bound (not compute-bound)
        # and fp32 rounding can shrink bounds, causing false negatives in spatial filtering.
        # The precision plan is still consulted for observability/diagnostics.
        return _compute_geometry_bounds_gpu(geometry_array, compute_type="double")
    if geometry_array.residency is Residency.DEVICE or any(
        not buffer.host_materialized for buffer in geometry_array.families.values()
    ):
        geometry_array.move_to(
            Residency.HOST,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="compute_geometry_bounds selected CPU execution",
        )
    return _compute_geometry_bounds_cpu_vectorized(geometry_array)


@register_kernel_variant(
    "compute_total_bounds",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_total_bounds(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> tuple[float, float, float, float]:
    normalize_precision_mode(precision)
    bounds = compute_geometry_bounds(geometry_array, dispatch_mode=dispatch_mode, precision=precision)
    if np.isnan(bounds).all():
        return (math.nan, math.nan, math.nan, math.nan)
    return (
        float(np.nanmin(bounds[:, 0])),
        float(np.nanmin(bounds[:, 1])),
        float(np.nanmax(bounds[:, 2])),
        float(np.nanmax(bounds[:, 3])),
    )


@register_kernel_variant(
    "compute_offset_spans",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_offset_spans(
    geometry_array: OwnedGeometryArray,
    *,
    level: str = "geometry",
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
) -> dict[GeometryFamily, np.ndarray]:
    del dispatch_mode
    result: dict[GeometryFamily, np.ndarray] = {}
    for family, buffer in geometry_array.families.items():
        if level == "geometry":
            offsets = buffer.geometry_offsets
        elif level == "part":
            offsets = buffer.part_offsets
        elif level == "ring":
            offsets = buffer.ring_offsets
        else:
            offsets = None
        if offsets is None:
            continue
        result[family] = np.diff(offsets)
    return result


def _spread_bits(value: int) -> int:
    value &= 0x00000000FFFFFFFF
    value = (value | (value << 16)) & 0x0000FFFF0000FFFF
    value = (value | (value << 8)) & 0x00FF00FF00FF00FF
    value = (value | (value << 4)) & 0x0F0F0F0F0F0F0F0F
    value = (value | (value << 2)) & 0x3333333333333333
    value = (value | (value << 1)) & 0x5555555555555555
    return value


def _morton_code(x: int, y: int) -> int:
    return _spread_bits(x) | (_spread_bits(y) << 1)


@register_kernel_variant(
    "compute_morton_keys",
    "cpu",
    kernel_class=KernelClass.COARSE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=tuple(family.value for family in GeometryFamily),
    supports_mixed=True,
)
def compute_morton_keys(
    geometry_array: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode = ExecutionMode.CPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    bits: int = 16,
) -> np.ndarray:
    del dispatch_mode
    normalize_precision_mode(precision)
    if bits != 16:
        raise ValueError("only 16-bit morton keys are currently supported")
    bounds = compute_geometry_bounds(geometry_array, precision=precision)
    total = compute_total_bounds(geometry_array, precision=precision)
    minx, miny, maxx, maxy = total
    keys = np.full(geometry_array.row_count, np.iinfo(np.uint64).max, dtype=np.uint64)
    if any(math.isnan(value) for value in total):
        return keys
    span_x = max(maxx - minx, 1e-12)
    span_y = max(maxy - miny, 1e-12)
    scale = (1 << bits) - 1
    for row_index, row_bounds in enumerate(bounds):
        if np.isnan(row_bounds).any():
            continue
        center_x = (float(row_bounds[0]) + float(row_bounds[2])) * 0.5
        center_y = (float(row_bounds[1]) + float(row_bounds[3])) * 0.5
        norm_x = int(round(((center_x - minx) / span_x) * scale))
        norm_y = int(round(((center_y - miny) / span_y) * scale))
        keys[row_index] = np.uint64(_morton_code(norm_x, norm_y))
    return keys
