"""NVRTC kernel sources for point."""

from __future__ import annotations

_POINT_CONSTRUCTIVE_KERNEL_SOURCE = """
extern "C" __global__ void point_rect_mask(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    double xmin,
    double ymin,
    double xmax,
    double ymax,
    int inclusive,
    unsigned char* out,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    out[row] = 0;
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  out[row] = inclusive
      ? ((px >= xmin && px <= xmax && py >= ymin && py <= ymax) ? 1 : 0)
      : ((px > xmin && px < xmax && py > ymin && py < ymax) ? 1 : 0);
}

extern "C" __global__ void point_buffer_quad1(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const double* radii,
    double* out_x,
    double* out_y,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  const double radius = radii[row];
  const int base = row * 5;
  out_x[base + 0] = px + radius; out_y[base + 0] = py;
  out_x[base + 1] = px;          out_y[base + 1] = py - radius;
  out_x[base + 2] = px - radius; out_y[base + 2] = py;
  out_x[base + 3] = px;          out_y[base + 3] = py + radius;
  out_x[base + 4] = px + radius; out_y[base + 4] = py;
}

extern "C" __global__ void point_buffer_round(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const unsigned char* point_empty_mask,
    const double* point_x,
    const double* point_y,
    const double* radii,
    double* out_x,
    double* out_y,
    int quad_segs,
    int verts_per_ring,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const int point_row = point_row_offsets[row];
  if (point_row < 0 || point_empty_mask[point_row]) {
    return;
  }
  const int coord = point_geometry_offsets[point_row];
  const double px = point_x[coord];
  const double py = point_y[coord];
  const double radius = radii[row];
  const int base = row * verts_per_ring;
  const int n_arc = 4 * quad_segs;
  const double step = -2.0 * 3.14159265358979323846 / (double)n_arc;
  for (int i = 0; i < n_arc; i++) {
    double angle = (double)i * step;
    out_x[base + i] = px + radius * cos(angle);
    out_y[base + i] = py + radius * sin(angle);
  }
  out_x[base + n_arc] = out_x[base];
  out_y[base + n_arc] = out_y[base];
}

extern "C" __global__ void point_subset_gather(
    const int* point_row_offsets,
    const int* point_geometry_offsets,
    const double* point_x,
    const double* point_y,
    const int* keep_rows,
    double* out_x,
    double* out_y,
    int out_row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= out_row_count) {
    return;
  }
  const int source_row = keep_rows[row];
  const int point_row = point_row_offsets[source_row];
  const int coord = point_geometry_offsets[point_row];
  out_x[row] = point_x[coord];
  out_y[row] = point_y[coord];
}
"""
_POINT_CONSTRUCTIVE_KERNEL_NAMES = ("point_rect_mask", "point_buffer_quad1", "point_buffer_round", "point_subset_gather")
