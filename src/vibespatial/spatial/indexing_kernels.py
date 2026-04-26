"""NVRTC kernel sources for spatial indexing (Morton keys, MBR sweep, segment MBR)."""

from __future__ import annotations

from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

_INDEXING_KERNEL_SOURCE = SPATIAL_TOLERANCE_PREAMBLE + """
extern "C" __device__ unsigned long long spread_bits_32(unsigned int value) {
  unsigned long long x = (unsigned long long) value;
  x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FFULL;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0FULL;
  x = (x | (x << 2)) & 0x3333333333333333ULL;
  x = (x | (x << 1)) & 0x5555555555555555ULL;
  return x;
}

extern "C" __global__ void morton_keys_from_bounds(
    const double* bounds,
    double minx,
    double miny,
    double maxx,
    double maxy,
    unsigned long long* out_keys,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int base = row * 4;
  const double bx0 = bounds[base + 0];
  const double by0 = bounds[base + 1];
  const double bx1 = bounds[base + 2];
  const double by1 = bounds[base + 3];
  if (isnan(bx0) || isnan(by0) || isnan(bx1) || isnan(by1)) {
    out_keys[row] = 0xFFFFFFFFFFFFFFFFULL;
    return;
  }
  const double span_x = fmax(maxx - minx, VS_SPATIAL_EPSILON);
  const double span_y = fmax(maxy - miny, VS_SPATIAL_EPSILON);
  const double center_x = (bx0 + bx1) * 0.5;
  const double center_y = (by0 + by1) * 0.5;
  const unsigned int norm_x = (unsigned int) llround(((center_x - minx) / span_x) * 65535.0);
  const unsigned int norm_y = (unsigned int) llround(((center_y - miny) / span_y) * 65535.0);
  out_keys[row] = spread_bits_32(norm_x) | (spread_bits_32(norm_y) << 1);
}

extern "C" __global__ void __launch_bounds__(256, 4)
regular_rect_grid_certify(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geometry_offsets,
    const int*    __restrict__ ring_offsets,
    const bool*   __restrict__ empty_mask,
    const bool*   __restrict__ validity,
    const signed char* __restrict__ tags,
    int           row_count,
    int           coord_count,
    int           polygon_tag,
    double*       out_bounds,
    double*       out_summary
) {
  __shared__ double s_minx[256];
  __shared__ double s_miny[256];
  __shared__ double s_maxx[256];
  __shared__ double s_maxy[256];
  __shared__ double s_max_left_x[256];
  __shared__ double s_max_bottom_y[256];
  __shared__ int s_valid[256];
  __shared__ int s_grid_valid[256];
  __shared__ double origin_x;
  __shared__ double origin_y;
  __shared__ double cell_width;
  __shared__ double cell_height;
  __shared__ double tol;
  __shared__ long long grid_cols;
  __shared__ long long grid_rows;

  const int tid = threadIdx.x;
  const double inf = 1.0e300;

  if (tid == 0) {
    double fx0 = 0.0;
    double fy0 = 0.0;
    double fx1 = 0.0;
    double fy1 = 0.0;
    if (row_count > 0 && coord_count >= 5 && ring_offsets[0] >= 0 && ring_offsets[0] + 4 < coord_count) {
      const int c0 = ring_offsets[0];
      fx0 = fmin(fmin(fmin(fmin(x[c0 + 0], x[c0 + 1]), x[c0 + 2]), x[c0 + 3]), x[c0 + 4]);
      fy0 = fmin(fmin(fmin(fmin(y[c0 + 0], y[c0 + 1]), y[c0 + 2]), y[c0 + 3]), y[c0 + 4]);
      fx1 = fmax(fmax(fmax(fmax(x[c0 + 0], x[c0 + 1]), x[c0 + 2]), x[c0 + 3]), x[c0 + 4]);
      fy1 = fmax(fmax(fmax(fmax(y[c0 + 0], y[c0 + 1]), y[c0 + 2]), y[c0 + 3]), y[c0 + 4]);
    }
    origin_x = fx0;
    origin_y = fy0;
    cell_width = fx1 - fx0;
    cell_height = fy1 - fy0;
    tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
    grid_cols = 0;
    grid_rows = 0;
  }
  __syncthreads();

  double local_minx = inf;
  double local_miny = inf;
  double local_maxx = -inf;
  double local_maxy = -inf;
  double local_max_left_x = -inf;
  double local_max_bottom_y = -inf;
  int local_valid = 1;

  for (int row = tid; row < row_count; row += blockDim.x) {
    int row_valid = 1;
    if (!validity[row] || tags[row] != (signed char)polygon_tag || empty_mask[row]) {
      row_valid = 0;
    }
    if (geometry_offsets[row] != row || geometry_offsets[row + 1] != row + 1) {
      row_valid = 0;
    }

    const int c0 = ring_offsets[row];
    const int c1 = ring_offsets[row + 1];
    if (c0 < 0 || c1 != c0 + 5 || c1 > coord_count) {
      row_valid = 0;
    }

    double xs[5];
    double ys[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      int coord = c0 + k;
      if (coord < 0) {
        coord = 0;
      }
      const int max_coord = coord_count - 1;
      if (coord > max_coord) {
        coord = max_coord;
      }
      xs[k] = x[coord];
      ys[k] = y[coord];
    }

    double bx0 = xs[0];
    double by0 = ys[0];
    double bx1 = xs[0];
    double by1 = ys[0];
    #pragma unroll
    for (int k = 1; k < 5; k++) {
      bx0 = fmin(bx0, xs[k]);
      by0 = fmin(by0, ys[k]);
      bx1 = fmax(bx1, xs[k]);
      by1 = fmax(by1, ys[k]);
    }

    if (!isfinite(bx0) || !isfinite(by0) || !isfinite(bx1) || !isfinite(by1)) {
      row_valid = 0;
    }
    const double width = bx1 - bx0;
    const double height = by1 - by0;
    if (width <= 0.0 || height <= 0.0) {
      row_valid = 0;
    }
    if (fabs(width - cell_width) > tol || fabs(height - cell_height) > tol) {
      row_valid = 0;
    }
    if (fabs(xs[0] - xs[4]) > tol || fabs(ys[0] - ys[4]) > tol) {
      row_valid = 0;
    }

    int x_min_count = 0;
    int x_max_count = 0;
    int y_min_count = 0;
    int y_max_count = 0;
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      const bool x_is_min = fabs(xs[k] - bx0) <= tol;
      const bool x_is_max = fabs(xs[k] - bx1) <= tol;
      const bool y_is_min = fabs(ys[k] - by0) <= tol;
      const bool y_is_max = fabs(ys[k] - by1) <= tol;
      if (!(x_is_min || x_is_max) || !(y_is_min || y_is_max)) {
        row_valid = 0;
      }
      if (k < 4) {
        x_min_count += x_is_min ? 1 : 0;
        x_max_count += x_is_max ? 1 : 0;
        y_min_count += y_is_min ? 1 : 0;
        y_max_count += y_is_max ? 1 : 0;
      }
    }
    if (x_min_count != 2 || x_max_count != 2 || y_min_count != 2 || y_max_count != 2) {
      row_valid = 0;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      const bool same_x = fabs(xs[k + 1] - xs[k]) <= tol;
      const bool same_y = fabs(ys[k + 1] - ys[k]) <= tol;
      if (same_x == same_y) {
        row_valid = 0;
      }
    }

    const int base = row * 4;
    out_bounds[base + 0] = bx0;
    out_bounds[base + 1] = by0;
    out_bounds[base + 2] = bx1;
    out_bounds[base + 3] = by1;

    if (row_valid) {
      local_minx = fmin(local_minx, bx0);
      local_miny = fmin(local_miny, by0);
      local_maxx = fmax(local_maxx, bx1);
      local_maxy = fmax(local_maxy, by1);
      local_max_left_x = fmax(local_max_left_x, bx0);
      local_max_bottom_y = fmax(local_max_bottom_y, by0);
    } else {
      local_valid = 0;
    }
  }

  s_minx[tid] = local_minx;
  s_miny[tid] = local_miny;
  s_maxx[tid] = local_maxx;
  s_maxy[tid] = local_maxy;
  s_max_left_x[tid] = local_max_left_x;
  s_max_bottom_y[tid] = local_max_bottom_y;
  s_valid[tid] = local_valid;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_minx[tid] = fmin(s_minx[tid], s_minx[tid + stride]);
      s_miny[tid] = fmin(s_miny[tid], s_miny[tid + stride]);
      s_maxx[tid] = fmax(s_maxx[tid], s_maxx[tid + stride]);
      s_maxy[tid] = fmax(s_maxy[tid], s_maxy[tid + stride]);
      s_max_left_x[tid] = fmax(s_max_left_x[tid], s_max_left_x[tid + stride]);
      s_max_bottom_y[tid] = fmax(s_max_bottom_y[tid], s_max_bottom_y[tid + stride]);
      s_valid[tid] = s_valid[tid] && s_valid[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (s_valid[0] && cell_width > 0.0 && cell_height > 0.0) {
      grid_cols = llround((s_max_left_x[0] - origin_x) / cell_width) + 1LL;
      grid_rows = llround((s_max_bottom_y[0] - origin_y) / cell_height) + 1LL;
    }
    if (grid_cols <= 0 || grid_rows <= 0 || grid_cols * grid_rows < (long long)row_count) {
      s_valid[0] = 0;
    }
  }
  __syncthreads();

  int local_grid_valid = 1;
  for (int row = tid; row < row_count; row += blockDim.x) {
    if (!s_valid[0] || grid_cols <= 0 || grid_rows <= 0 || cell_width <= 0.0 || cell_height <= 0.0) {
      local_grid_valid = 0;
      continue;
    }
    const int base = row * 4;
    const double bx0 = out_bounds[base + 0];
    const double by0 = out_bounds[base + 1];
    const long long col = llround((bx0 - origin_x) / cell_width);
    const long long grid_row = llround((by0 - origin_y) / cell_height);
    if (
        col < 0 || col >= grid_cols ||
        grid_row < 0 || grid_row >= grid_rows ||
        grid_row * grid_cols + col != (long long)row ||
        fabs((origin_x + (double)col * cell_width) - bx0) > tol ||
        fabs((origin_y + (double)grid_row * cell_height) - by0) > tol
    ) {
      local_grid_valid = 0;
    }
  }

  s_grid_valid[tid] = local_grid_valid;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_grid_valid[tid] = s_grid_valid[tid] && s_grid_valid[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int valid = s_valid[0] && s_grid_valid[0];
    out_summary[0] = s_minx[0];
    out_summary[1] = s_miny[0];
    out_summary[2] = s_maxx[0];
    out_summary[3] = s_maxy[0];
    out_summary[4] = cell_width;
    out_summary[5] = cell_height;
    out_summary[6] = valid ? (double)grid_cols : 0.0;
    out_summary[7] = valid ? (double)grid_rows : 0.0;
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
regular_rect_grid_certify_blocks(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geometry_offsets,
    const int*    __restrict__ ring_offsets,
    const bool*   __restrict__ empty_mask,
    const bool*   __restrict__ validity,
    const signed char* __restrict__ tags,
    int           row_count,
    int           coord_count,
    int           polygon_tag,
    double*       out_bounds,
    double*       block_summaries
) {
  __shared__ double s_minx[256];
  __shared__ double s_miny[256];
  __shared__ double s_maxx[256];
  __shared__ double s_maxy[256];
  __shared__ double s_max_left_x[256];
  __shared__ double s_max_bottom_y[256];
  __shared__ int s_valid[256];
  __shared__ double origin_x;
  __shared__ double origin_y;
  __shared__ double cell_width;
  __shared__ double cell_height;
  __shared__ double tol;

  const int tid = threadIdx.x;
  const double inf = 1.0e300;

  if (tid == 0) {
    double fx0 = 0.0;
    double fy0 = 0.0;
    double fx1 = 0.0;
    double fy1 = 0.0;
    if (row_count > 0 && coord_count >= 5 && ring_offsets[0] >= 0 && ring_offsets[0] + 4 < coord_count) {
      const int c0 = ring_offsets[0];
      fx0 = fmin(fmin(fmin(fmin(x[c0 + 0], x[c0 + 1]), x[c0 + 2]), x[c0 + 3]), x[c0 + 4]);
      fy0 = fmin(fmin(fmin(fmin(y[c0 + 0], y[c0 + 1]), y[c0 + 2]), y[c0 + 3]), y[c0 + 4]);
      fx1 = fmax(fmax(fmax(fmax(x[c0 + 0], x[c0 + 1]), x[c0 + 2]), x[c0 + 3]), x[c0 + 4]);
      fy1 = fmax(fmax(fmax(fmax(y[c0 + 0], y[c0 + 1]), y[c0 + 2]), y[c0 + 3]), y[c0 + 4]);
    }
    origin_x = fx0;
    origin_y = fy0;
    cell_width = fx1 - fx0;
    cell_height = fy1 - fy0;
    tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);
  }
  __syncthreads();

  double local_minx = inf;
  double local_miny = inf;
  double local_maxx = -inf;
  double local_maxy = -inf;
  double local_max_left_x = -inf;
  double local_max_bottom_y = -inf;
  int local_valid = 1;

  for (int row = blockIdx.x * blockDim.x + tid; row < row_count; row += blockDim.x * gridDim.x) {
    int row_valid = 1;
    if (!validity[row] || tags[row] != (signed char)polygon_tag || empty_mask[row]) {
      row_valid = 0;
    }
    if (geometry_offsets[row] != row || geometry_offsets[row + 1] != row + 1) {
      row_valid = 0;
    }

    const int c0 = ring_offsets[row];
    const int c1 = ring_offsets[row + 1];
    if (c0 < 0 || c1 != c0 + 5 || c1 > coord_count) {
      row_valid = 0;
    }

    double xs[5];
    double ys[5];
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      int coord = c0 + k;
      if (coord < 0) {
        coord = 0;
      }
      const int max_coord = coord_count - 1;
      if (coord > max_coord) {
        coord = max_coord;
      }
      xs[k] = x[coord];
      ys[k] = y[coord];
    }

    double bx0 = xs[0];
    double by0 = ys[0];
    double bx1 = xs[0];
    double by1 = ys[0];
    #pragma unroll
    for (int k = 1; k < 5; k++) {
      bx0 = fmin(bx0, xs[k]);
      by0 = fmin(by0, ys[k]);
      bx1 = fmax(bx1, xs[k]);
      by1 = fmax(by1, ys[k]);
    }

    if (!isfinite(bx0) || !isfinite(by0) || !isfinite(bx1) || !isfinite(by1)) {
      row_valid = 0;
    }
    const double width = bx1 - bx0;
    const double height = by1 - by0;
    if (width <= 0.0 || height <= 0.0) {
      row_valid = 0;
    }
    if (fabs(width - cell_width) > tol || fabs(height - cell_height) > tol) {
      row_valid = 0;
    }
    if (fabs(xs[0] - xs[4]) > tol || fabs(ys[0] - ys[4]) > tol) {
      row_valid = 0;
    }

    int x_min_count = 0;
    int x_max_count = 0;
    int y_min_count = 0;
    int y_max_count = 0;
    #pragma unroll
    for (int k = 0; k < 5; k++) {
      const bool x_is_min = fabs(xs[k] - bx0) <= tol;
      const bool x_is_max = fabs(xs[k] - bx1) <= tol;
      const bool y_is_min = fabs(ys[k] - by0) <= tol;
      const bool y_is_max = fabs(ys[k] - by1) <= tol;
      if (!(x_is_min || x_is_max) || !(y_is_min || y_is_max)) {
        row_valid = 0;
      }
      if (k < 4) {
        x_min_count += x_is_min ? 1 : 0;
        x_max_count += x_is_max ? 1 : 0;
        y_min_count += y_is_min ? 1 : 0;
        y_max_count += y_is_max ? 1 : 0;
      }
    }
    if (x_min_count != 2 || x_max_count != 2 || y_min_count != 2 || y_max_count != 2) {
      row_valid = 0;
    }
    #pragma unroll
    for (int k = 0; k < 4; k++) {
      const bool same_x = fabs(xs[k + 1] - xs[k]) <= tol;
      const bool same_y = fabs(ys[k + 1] - ys[k]) <= tol;
      if (same_x == same_y) {
        row_valid = 0;
      }
    }

    const int base = row * 4;
    out_bounds[base + 0] = bx0;
    out_bounds[base + 1] = by0;
    out_bounds[base + 2] = bx1;
    out_bounds[base + 3] = by1;

    if (row_valid) {
      local_minx = fmin(local_minx, bx0);
      local_miny = fmin(local_miny, by0);
      local_maxx = fmax(local_maxx, bx1);
      local_maxy = fmax(local_maxy, by1);
      local_max_left_x = fmax(local_max_left_x, bx0);
      local_max_bottom_y = fmax(local_max_bottom_y, by0);
    } else {
      local_valid = 0;
    }
  }

  s_minx[tid] = local_minx;
  s_miny[tid] = local_miny;
  s_maxx[tid] = local_maxx;
  s_maxy[tid] = local_maxy;
  s_max_left_x[tid] = local_max_left_x;
  s_max_bottom_y[tid] = local_max_bottom_y;
  s_valid[tid] = local_valid;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_minx[tid] = fmin(s_minx[tid], s_minx[tid + stride]);
      s_miny[tid] = fmin(s_miny[tid], s_miny[tid + stride]);
      s_maxx[tid] = fmax(s_maxx[tid], s_maxx[tid + stride]);
      s_maxy[tid] = fmax(s_maxy[tid], s_maxy[tid + stride]);
      s_max_left_x[tid] = fmax(s_max_left_x[tid], s_max_left_x[tid + stride]);
      s_max_bottom_y[tid] = fmax(s_max_bottom_y[tid], s_max_bottom_y[tid + stride]);
      s_valid[tid] = s_valid[tid] && s_valid[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const int base = blockIdx.x * 8;
    block_summaries[base + 0] = s_minx[0];
    block_summaries[base + 1] = s_miny[0];
    block_summaries[base + 2] = s_maxx[0];
    block_summaries[base + 3] = s_maxy[0];
    block_summaries[base + 4] = s_max_left_x[0];
    block_summaries[base + 5] = s_max_bottom_y[0];
    block_summaries[base + 6] = (double)s_valid[0];
    block_summaries[base + 7] = 0.0;
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
regular_rect_grid_finalize(
    const double* __restrict__ out_bounds,
    const double* __restrict__ block_summaries,
    int block_count,
    int row_count,
    double* out_summary
) {
  __shared__ double s_minx[256];
  __shared__ double s_miny[256];
  __shared__ double s_maxx[256];
  __shared__ double s_maxy[256];
  __shared__ double s_max_left_x[256];
  __shared__ double s_max_bottom_y[256];
  __shared__ int s_valid[256];

  const int tid = threadIdx.x;
  const double inf = 1.0e300;
  double local_minx = inf;
  double local_miny = inf;
  double local_maxx = -inf;
  double local_maxy = -inf;
  double local_max_left_x = -inf;
  double local_max_bottom_y = -inf;
  int local_valid = row_count > 0 ? 1 : 0;

  for (int block = tid; block < block_count; block += blockDim.x) {
    const int base = block * 8;
    local_minx = fmin(local_minx, block_summaries[base + 0]);
    local_miny = fmin(local_miny, block_summaries[base + 1]);
    local_maxx = fmax(local_maxx, block_summaries[base + 2]);
    local_maxy = fmax(local_maxy, block_summaries[base + 3]);
    local_max_left_x = fmax(local_max_left_x, block_summaries[base + 4]);
    local_max_bottom_y = fmax(local_max_bottom_y, block_summaries[base + 5]);
    local_valid = local_valid && ((int)block_summaries[base + 6]);
  }

  s_minx[tid] = local_minx;
  s_miny[tid] = local_miny;
  s_maxx[tid] = local_maxx;
  s_maxy[tid] = local_maxy;
  s_max_left_x[tid] = local_max_left_x;
  s_max_bottom_y[tid] = local_max_bottom_y;
  s_valid[tid] = local_valid;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_minx[tid] = fmin(s_minx[tid], s_minx[tid + stride]);
      s_miny[tid] = fmin(s_miny[tid], s_miny[tid + stride]);
      s_maxx[tid] = fmax(s_maxx[tid], s_maxx[tid + stride]);
      s_maxy[tid] = fmax(s_maxy[tid], s_maxy[tid + stride]);
      s_max_left_x[tid] = fmax(s_max_left_x[tid], s_max_left_x[tid + stride]);
      s_max_bottom_y[tid] = fmax(s_max_bottom_y[tid], s_max_bottom_y[tid + stride]);
      s_valid[tid] = s_valid[tid] && s_valid[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const double origin_x = out_bounds[0];
    const double origin_y = out_bounds[1];
    const double cell_width = out_bounds[2] - out_bounds[0];
    const double cell_height = out_bounds[3] - out_bounds[1];
    long long grid_cols = 0;
    long long grid_rows = 0;
    int valid = s_valid[0] && cell_width > 0.0 && cell_height > 0.0;
    if (valid) {
      grid_cols = llround((s_max_left_x[0] - origin_x) / cell_width) + 1LL;
      grid_rows = llround((s_max_bottom_y[0] - origin_y) / cell_height) + 1LL;
      if (grid_cols <= 0 || grid_rows <= 0 || grid_cols * grid_rows < (long long)row_count) {
        valid = 0;
      }
    }
    out_summary[0] = s_minx[0];
    out_summary[1] = s_miny[0];
    out_summary[2] = s_maxx[0];
    out_summary[3] = s_maxy[0];
    out_summary[4] = cell_width;
    out_summary[5] = cell_height;
    out_summary[6] = valid ? (double)grid_cols : 0.0;
    out_summary[7] = valid ? (double)grid_rows : 0.0;
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
regular_rect_grid_validate_positions(
    const double* __restrict__ out_bounds,
    int row_count,
    double* out_summary,
    int* valid_flag
) {
  const double origin_x = out_summary[0];
  const double origin_y = out_summary[1];
  const double cell_width = out_summary[4];
  const double cell_height = out_summary[5];
  const long long grid_cols = llround(out_summary[6]);
  const long long grid_rows = llround(out_summary[7]);
  const double tol = 1e-9 * fmax(fmax(fabs(cell_width), fabs(cell_height)), 1.0);

  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < row_count; row += blockDim.x * gridDim.x) {
    if (grid_cols <= 0 || grid_rows <= 0 || cell_width <= 0.0 || cell_height <= 0.0) {
      atomicExch(valid_flag, 0);
      continue;
    }
    const int base = row * 4;
    const double bx0 = out_bounds[base + 0];
    const double by0 = out_bounds[base + 1];
    const long long col = llround((bx0 - origin_x) / cell_width);
    const long long grid_row = llround((by0 - origin_y) / cell_height);
    if (
        col < 0 || col >= grid_cols ||
        grid_row < 0 || grid_row >= grid_rows ||
        grid_row * grid_cols + col != (long long)row ||
        fabs((origin_x + (double)col * cell_width) - bx0) > tol ||
        fabs((origin_y + (double)grid_row * cell_height) - by0) > tol
    ) {
      atomicExch(valid_flag, 0);
    }
  }
}

extern "C" __global__ void regular_rect_grid_apply_valid(
    const int* valid_flag,
    double* out_summary
) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && valid_flag[0] == 0) {
    out_summary[6] = 0.0;
    out_summary[7] = 0.0;
  }
}

/* Sort-and-sweep MBR overlap test.
 *
 * Operates on a concatenated array of geometries, sorted by minx.
 * Each thread handles one element and sweeps forward through the sorted
 * array.  The sweep terminates when sorted_minx[j] > current_maxx,
 * pruning the search space from O(n) to O(k) where k is the average
 * x-overlap count.
 *
 * For same_input mode: thread i sweeps forward and emits upper-triangle
 * pairs (based on original row index) to avoid duplicates.
 *
 * For two-input mode: thread i sweeps forward and emits a pair whenever
 * the two elements come from different sides (left vs right).  Because
 * the sweep is directional (i < j in sorted order), each cross-side
 * pair is discovered exactly once -- by whichever element appears first.
 * The emitted pair is always (left_orig, right_orig) regardless of which
 * side the sweeping element belongs to.
 *
 * Two passes: pass 0 counts valid pairs per sorted element;
 *             pass 1 writes pairs using a prefix-sum offset array.
 */
extern "C" __global__ void sweep_sorted_mbr_overlap(
    const double* __restrict__ sorted_bounds,
    const int*    __restrict__ sorted_orig,
    const int*    __restrict__ sorted_side,
    int           total_count,
    int*          out_left,
    int*          out_right,
    int*          counts,
    int           pass_number,
    int           same_input,
    int           include_self
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total_count) return;

  const int ibase = i * 4;
  const double ix0 = sorted_bounds[ibase + 0];
  const double iy0 = sorted_bounds[ibase + 1];
  const double ix1 = sorted_bounds[ibase + 2];
  const double iy1 = sorted_bounds[ibase + 3];
  const int i_orig = sorted_orig[i];
  const int i_side = sorted_side[i];

  int pair_count = 0;
  int write_offset = 0;
  if (pass_number == 1) write_offset = counts[i];

  /* Sweep forward: elements are sorted by minx, so once sorted_minx[j] > ix1
   * no further element can overlap in x with element i. */
  for (int j = i + 1; j < total_count; j++) {
    const int jbase = j * 4;
    const double jx0 = sorted_bounds[jbase + 0];
    /* Early exit: sorted by minx, so jx0 > ix1 means no more x-overlap */
    if (jx0 > ix1) break;

    const double jy0 = sorted_bounds[jbase + 1];
    const double jx1 = sorted_bounds[jbase + 2];
    const double jy1 = sorted_bounds[jbase + 3];

    /* Full y-axis overlap test (x-overlap guaranteed by sweep condition) */
    if (iy0 > jy1 || iy1 < jy0) continue;

    const int j_orig = sorted_orig[j];
    const int j_side = sorted_side[j];

    if (same_input) {
      /* Same-input mode: each pair (i,j) where i < j in sorted order is
       * discovered exactly once by the thread at position i.  Emit as
       * (min_orig, max_orig) for upper-triangle.  Skip self-pairs
       * (same original row index) unless include_self is set. */
      if (i_orig == j_orig && !include_self) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Emit as (smaller_orig, larger_orig) */
        if (i_orig <= j_orig) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    } else {
      /* Two-input mode: emit only cross-side pairs */
      if (i_side == j_side) continue;
      if (pass_number == 0) {
        pair_count++;
      } else {
        /* Always emit as (left_orig, right_orig) */
        if (i_side == 0) {
          out_left[write_offset + pair_count] = i_orig;
          out_right[write_offset + pair_count] = j_orig;
        } else {
          out_left[write_offset + pair_count] = j_orig;
          out_right[write_offset + pair_count] = i_orig;
        }
        pair_count++;
      }
    }
  }

  if (pass_number == 0) counts[i] = pair_count;
}
"""

_INDEXING_KERNEL_NAMES = (
    "morton_keys_from_bounds",
    "regular_rect_grid_certify",
    "regular_rect_grid_certify_blocks",
    "regular_rect_grid_finalize",
    "regular_rect_grid_validate_positions",
    "regular_rect_grid_apply_valid",
    "sweep_sorted_mbr_overlap",
)

# ---------------------------------------------------------------------------
# Segment MBR extraction kernels (Tier 1: geometry-specific inner loops)
# ---------------------------------------------------------------------------
# One kernel per geometry family.  Two-pass count/scatter:
#   Pass 0 (pass_number==0): each thread counts segments for one geometry row.
#   Pass 1 (pass_number==1): each thread scatters (row, seg_idx, minx,miny,maxx,maxy).
#
# The kernels receive coordinate buffers in SoA layout (separate x[], y[])
# plus the offset hierarchy for the family.
# ---------------------------------------------------------------------------

_SEGMENT_MBR_KERNEL_SOURCE = """
/* ---- LineString segment MBR ---- */
extern "C" __global__ void segment_mbr_linestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int c0 = geom_offsets[g];
    const int c1 = geom_offsets[g + 1];
    const int n_seg = c1 - c0 - 1;
    if (n_seg <= 0) {
      if (pass_number == 0) counts[g] = 0;
      continue;
    }
    if (pass_number == 0) {
      counts[g] = n_seg;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      for (int s = 0; s < n_seg; s++) {
        const int ci = c0 + s;
        const double x0 = x[ci], y0 = y[ci];
        const double x1 = x[ci + 1], y1 = y[ci + 1];
        const int idx = base + s;
        row_out[idx] = grow;
        seg_out[idx] = s;
        bounds_out[idx * 4 + 0] = fmin(x0, x1);
        bounds_out[idx * 4 + 1] = fmin(y0, y1);
        bounds_out[idx * 4 + 2] = fmax(x0, x1);
        bounds_out[idx * 4 + 3] = fmax(y0, y1);
      }
    }
  }
}

/* ---- Polygon segment MBR ---- */
extern "C" __global__ void segment_mbr_polygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int r0 = geom_offsets[g];
    const int r1 = geom_offsets[g + 1];
    int total = 0;
    for (int r = r0; r < r1; r++) {
      const int rc0 = ring_offsets[r];
      const int rc1 = ring_offsets[r + 1];
      total += rc1 - rc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        const int n_seg = rc1 - rc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = rc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiLineString segment MBR ---- */
extern "C" __global__ void segment_mbr_multilinestring(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int pc0 = part_offsets[p];
      const int pc1 = part_offsets[p + 1];
      total += pc1 - pc0 - 1;
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int pc0 = part_offsets[p];
        const int pc1 = part_offsets[p + 1];
        const int n_seg = pc1 - pc0 - 1;
        for (int s = 0; s < n_seg; s++) {
          const int ci = pc0 + s;
          const double x0 = x[ci], y0 = y[ci];
          const double x1 = x[ci + 1], y1 = y[ci + 1];
          const int idx = base + seg;
          row_out[idx] = grow;
          seg_out[idx] = seg;
          bounds_out[idx * 4 + 0] = fmin(x0, x1);
          bounds_out[idx * 4 + 1] = fmin(y0, y1);
          bounds_out[idx * 4 + 2] = fmax(x0, x1);
          bounds_out[idx * 4 + 3] = fmax(y0, y1);
          seg++;
        }
      }
    }
  }
}

/* ---- MultiPolygon segment MBR ---- */
extern "C" __global__ void segment_mbr_multipolygon(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int*    __restrict__ geom_offsets,
    const int*    __restrict__ part_offsets,
    const int*    __restrict__ ring_offsets,
    const int*    __restrict__ global_row_indices,
    int           geom_count,
    int*          counts,
    int*          row_out,
    int*          seg_out,
    double*       bounds_out,
    int           pass_number
) {
  for (int g = blockIdx.x * blockDim.x + threadIdx.x;
       g < geom_count;
       g += blockDim.x * gridDim.x) {
    const int p0 = geom_offsets[g];
    const int p1 = geom_offsets[g + 1];
    int total = 0;
    for (int p = p0; p < p1; p++) {
      const int r0 = part_offsets[p];
      const int r1 = part_offsets[p + 1];
      for (int r = r0; r < r1; r++) {
        const int rc0 = ring_offsets[r];
        const int rc1 = ring_offsets[r + 1];
        total += rc1 - rc0 - 1;
      }
    }
    if (pass_number == 0) {
      counts[g] = total;
    } else {
      const int base = counts[g];
      const int grow = global_row_indices[g];
      int seg = 0;
      for (int p = p0; p < p1; p++) {
        const int r0 = part_offsets[p];
        const int r1 = part_offsets[p + 1];
        for (int r = r0; r < r1; r++) {
          const int rc0 = ring_offsets[r];
          const int rc1 = ring_offsets[r + 1];
          const int n_seg = rc1 - rc0 - 1;
          for (int s = 0; s < n_seg; s++) {
            const int ci = rc0 + s;
            const double x0 = x[ci], y0 = y[ci];
            const double x1 = x[ci + 1], y1 = y[ci + 1];
            const int idx = base + seg;
            row_out[idx] = grow;
            seg_out[idx] = seg;
            bounds_out[idx * 4 + 0] = fmin(x0, x1);
            bounds_out[idx * 4 + 1] = fmin(y0, y1);
            bounds_out[idx * 4 + 2] = fmax(x0, x1);
            bounds_out[idx * 4 + 3] = fmax(y0, y1);
            seg++;
          }
        }
      }
    }
  }
}
"""

_SEGMENT_MBR_KERNEL_NAMES = (
    "segment_mbr_linestring",
    "segment_mbr_polygon",
    "segment_mbr_multilinestring",
    "segment_mbr_multipolygon",
)
