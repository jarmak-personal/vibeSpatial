"""NVRTC kernel sources for clip_rect."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

# ---------------------------------------------------------------------------
# GPU clip thresholds (ADR-0033 tier system)
# ---------------------------------------------------------------------------
_POLYGON_CLIP_GPU_THRESHOLD = 1_000
_LINE_CLIP_GPU_THRESHOLD = 1_000
# ---------------------------------------------------------------------------
# GPU Sutherland-Hodgman polygon clip kernel (NVRTC)
# ---------------------------------------------------------------------------
# Per-ring clip against a rectangle.  Each thread processes one ring through
# all four boundary edges sequentially.  The kernel writes clipped vertices
# into a pre-allocated output buffer using per-ring offsets computed via
# exclusive_scan on a vertex-count pass.

_SUTHERLAND_HODGMAN_KERNEL_SOURCE = (
    STRIP_CLOSURE_DEVICE
    + SPATIAL_TOLERANCE_PREAMBLE
    + r"""
#define EPSILON VS_SPATIAL_EPSILON

/* Clip a ring against one boundary edge.
   Returns the number of output vertices written to out_x/out_y. */
__device__ int clip_edge(
    const double* in_x, const double* in_y, int in_count,
    double* out_x, double* out_y, int max_out,
    int edge_type,  /* 0=left, 1=right, 2=bottom, 3=top */
    double edge_val
) {
  if (in_count == 0) return 0;
  int out_count = 0;

  double prev_x = in_x[in_count - 1];
  double prev_y = in_y[in_count - 1];

  int prev_inside;
  if (edge_type == 0)      prev_inside = (prev_x >= edge_val) ? 1 : 0;
  else if (edge_type == 1) prev_inside = (prev_x <= edge_val) ? 1 : 0;
  else if (edge_type == 2) prev_inside = (prev_y >= edge_val) ? 1 : 0;
  else                     prev_inside = (prev_y <= edge_val) ? 1 : 0;

  for (int i = 0; i < in_count; i++) {
    double cur_x = in_x[i];
    double cur_y = in_y[i];

    int cur_inside;
    if (edge_type == 0)      cur_inside = (cur_x >= edge_val) ? 1 : 0;
    else if (edge_type == 1) cur_inside = (cur_x <= edge_val) ? 1 : 0;
    else if (edge_type == 2) cur_inside = (cur_y >= edge_val) ? 1 : 0;
    else                     cur_inside = (cur_y <= edge_val) ? 1 : 0;

    if (cur_inside) {
      if (!prev_inside) {
        /* Compute intersection */
        double ix, iy;
        if (edge_type <= 1) {
          /* Vertical edge */
          double dx = cur_x - prev_x;
          if (fabs(dx) <= EPSILON) { ix = edge_val; iy = prev_y; }
          else { double t = (edge_val - prev_x) / dx; ix = edge_val; iy = prev_y + t * (cur_y - prev_y); }
        } else {
          /* Horizontal edge */
          double dy = cur_y - prev_y;
          if (fabs(dy) <= EPSILON) { ix = prev_x; iy = edge_val; }
          else { double t = (edge_val - prev_y) / dy; ix = prev_x + t * (cur_x - prev_x); iy = edge_val; }
        }
        if (out_count < max_out) { out_x[out_count] = ix; out_y[out_count] = iy; out_count++; }
      }
      if (out_count < max_out) { out_x[out_count] = cur_x; out_y[out_count] = cur_y; out_count++; }
    } else if (prev_inside) {
      double ix, iy;
      if (edge_type <= 1) {
        double dx = cur_x - prev_x;
        if (fabs(dx) <= EPSILON) { ix = edge_val; iy = prev_y; }
        else { double t = (edge_val - prev_x) / dx; ix = edge_val; iy = prev_y + t * (cur_y - prev_y); }
      } else {
        double dy = cur_y - prev_y;
        if (fabs(dy) <= EPSILON) { ix = prev_x; iy = edge_val; }
        else { double t = (edge_val - prev_y) / dy; ix = prev_x + t * (cur_x - prev_x); iy = edge_val; }
      }
      if (out_count < max_out) { out_x[out_count] = ix; out_y[out_count] = iy; out_count++; }
    }
    prev_x = cur_x;
    prev_y = cur_y;
    prev_inside = cur_inside;
  }
  return out_count;
}

/* Count output vertices for one ring after Sutherland-Hodgman clipping.
   Each thread handles one ring.  We use shared-memory scratch buffers. */
extern "C" __global__ void sh_count_vertices(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    int* out_vertex_counts,
    double xmin, double ymin, double xmax, double ymax,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) { out_vertex_counts[ring >= ring_count ? 0 : ring] = 0; return; }
  if (ring >= ring_count) return;

  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  int n = end - start;

  /* Strip closure vertex */
  n = vs_strip_closure(ring_x, ring_y, start, end, n, 1e-24);
  if (n < 3) { out_vertex_counts[ring] = 0; return; }

  /* Use local buffers (max reasonable ring size for GPU clip) */
  const int MAX_VERTS = 256;
  double buf_a_x[256], buf_a_y[256];
  double buf_b_x[256], buf_b_y[256];

  if (n > MAX_VERTS) { out_vertex_counts[ring] = 0; return; }

  for (int i = 0; i < n; i++) { buf_a_x[i] = ring_x[start + i]; buf_a_y[i] = ring_y[start + i]; }

  /* Clip against 4 edges: left, right, bottom, top */
  double edges[4] = {xmin, xmax, ymin, ymax};
  int count = n;
  double *src_x = buf_a_x, *src_y = buf_a_y;
  double *dst_x = buf_b_x, *dst_y = buf_b_y;

  for (int e = 0; e < 4; e++) {
    count = clip_edge(src_x, src_y, count, dst_x, dst_y, MAX_VERTS, e, edges[e]);
    if (count == 0) break;
    /* Swap buffers */
    double *tmp;
    tmp = src_x; src_x = dst_x; dst_x = tmp;
    tmp = src_y; src_y = dst_y; dst_y = tmp;
  }

  /* Add closure vertex if result is a valid ring */
  out_vertex_counts[ring] = (count >= 3) ? count + 1 : 0;
}

/* Write clipped vertices for one ring, using pre-computed offsets. */
extern "C" __global__ void sh_clip_rings(
    const double* ring_x,
    const double* ring_y,
    const int* ring_offsets,
    const int* out_offsets,
    double* out_x,
    double* out_y,
    double xmin, double ymin, double xmax, double ymax,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) return;

  const int out_start = out_offsets[ring];
  const int out_end = out_offsets[ring + 1];
  const int expected = out_end - out_start;
  if (expected <= 0) return;

  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  int n = end - start;

  n = vs_strip_closure(ring_x, ring_y, start, end, n, 1e-24);
  if (n < 3) return;

  const int MAX_VERTS = 256;
  double buf_a_x[256], buf_a_y[256];
  double buf_b_x[256], buf_b_y[256];

  if (n > MAX_VERTS) return;

  for (int i = 0; i < n; i++) { buf_a_x[i] = ring_x[start + i]; buf_a_y[i] = ring_y[start + i]; }

  double edges[4] = {xmin, xmax, ymin, ymax};
  int count = n;
  double *src_x = buf_a_x, *src_y = buf_a_y;
  double *dst_x = buf_b_x, *dst_y = buf_b_y;

  for (int e = 0; e < 4; e++) {
    count = clip_edge(src_x, src_y, count, dst_x, dst_y, MAX_VERTS, e, edges[e]);
    if (count == 0) break;
    double *tmp;
    tmp = src_x; src_x = dst_x; dst_x = tmp;
    tmp = src_y; src_y = dst_y; dst_y = tmp;
  }

  if (count < 3) return;

  /* Write output vertices + closure */
  for (int i = 0; i < count && i < expected - 1; i++) {
    out_x[out_start + i] = src_x[i];
    out_y[out_start + i] = src_y[i];
  }
  /* Closure vertex */
  out_x[out_start + count] = src_x[0];
  out_y[out_start + count] = src_y[0];
}
"""
)
_SH_KERNEL_NAMES = ("sh_count_vertices", "sh_clip_rings")
# ---------------------------------------------------------------------------
# Line-family rectangle clip assembly kernels
# ---------------------------------------------------------------------------

_LINE_ROW_KERNEL_SOURCE = (
    SPATIAL_TOLERANCE_PREAMBLE
    + r"""
#define EPSILON VS_SPATIAL_EPSILON

__device__ __forceinline__ int line_rect_clip_segment(
    const double x0,
    const double y0,
    const double x1,
    const double y1,
    const double xmin,
    const double ymin,
    const double xmax,
    const double ymax,
    double* __restrict__ out_x0,
    double* __restrict__ out_y0,
    double* __restrict__ out_x1,
    double* __restrict__ out_y1
) {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double p[4] = {-dx, dx, -dy, dy};
    const double q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};
    double u1 = 0.0;
    double u2 = 1.0;

    for (int edge = 0; edge < 4; ++edge) {
        if (fabs(p[edge]) <= EPSILON) {
            if (q[edge] < 0.0) return 0;
            continue;
        }
        const double t = q[edge] / p[edge];
        if (p[edge] < 0.0) {
            if (t > u1) u1 = t;
        } else if (t < u2) {
            u2 = t;
        }
        if (u1 > u2) return 0;
    }

    double cx0 = fmin(fmax(x0 + u1 * dx, xmin), xmax);
    double cy0 = fmin(fmax(y0 + u1 * dy, ymin), ymax);
    double cx1 = fmin(fmax(x0 + u2 * dx, xmin), xmax);
    double cy1 = fmin(fmax(y0 + u2 * dy, ymin), ymax);
    const double ddx = cx0 - cx1;
    const double ddy = cy0 - cy1;
    if (ddx * ddx + ddy * ddy < EPSILON * EPSILON) return 0;

    *out_x0 = cx0;
    *out_y0 = cy0;
    *out_x1 = cx1;
    *out_y1 = cy1;
    return 1;
}

extern "C" __global__ void __launch_bounds__(256, 4)
line_rect_capacity_count(
    const unsigned char* __restrict__ validity,
    const signed char* __restrict__ tags,
    const int* __restrict__ family_row_offsets,
    const double* __restrict__ line_x,
    const double* __restrict__ line_y,
    const int* __restrict__ line_geometry_offsets,
    const unsigned char* __restrict__ line_empty,
    const double* __restrict__ multi_x,
    const double* __restrict__ multi_y,
    const int* __restrict__ multi_geometry_offsets,
    const int* __restrict__ multi_part_offsets,
    const unsigned char* __restrict__ multi_empty,
    int* __restrict__ out_run_counts,
    int* __restrict__ out_coord_counts,
    unsigned char* __restrict__ out_has_output,
    const double xmin,
    const double ymin,
    const double xmax,
    const double ymax,
    const int row_count
) {
    const int stride = blockDim.x * gridDim.x;
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        int runs = 0;
        int coords = 0;
        int have_prev = 0;
        double prev_x1 = 0.0;
        double prev_y1 = 0.0;

        if (validity[row]) {
            const int family = (int)tags[row];
            const int family_row = family_row_offsets[row];
            int part_start = 0;
            int part_end = 0;
            if (family == 1 && !line_empty[family_row]) {
                part_start = family_row;
                part_end = family_row + 1;
            } else if (family == 4 && !multi_empty[family_row]) {
                part_start = multi_geometry_offsets[family_row];
                part_end = multi_geometry_offsets[family_row + 1];
            }

            for (int part = part_start; part < part_end; ++part) {
                const int coord_start = family == 1
                    ? line_geometry_offsets[family_row]
                    : multi_part_offsets[part];
                const int coord_end = family == 1
                    ? line_geometry_offsets[family_row + 1]
                    : multi_part_offsets[part + 1];
                int part_boundary = have_prev;
                for (int coord = coord_start; coord < coord_end - 1; ++coord) {
                    const double* x = family == 1 ? line_x : multi_x;
                    const double* y = family == 1 ? line_y : multi_y;
                    double sx, sy, ex, ey;
                    if (!line_rect_clip_segment(
                        x[coord], y[coord], x[coord + 1], y[coord + 1],
                        xmin, ymin, xmax, ymax, &sx, &sy, &ex, &ey
                    )) continue;

                    if (!have_prev) {
                        runs = 1;
                        coords = 2;
                        have_prev = 1;
                    } else if (
                        part_boundary
                        || fabs(prev_x1 - sx) > EPSILON
                        || fabs(prev_y1 - sy) > EPSILON
                    ) {
                        runs += 1;
                        coords += 2;
                    } else {
                        coords += 1;
                    }
                    prev_x1 = ex;
                    prev_y1 = ey;
                    part_boundary = 0;
                }
            }
        }

        out_run_counts[row] = runs;
        out_coord_counts[row] = coords;
        out_has_output[row] = have_prev ? 1 : 0;
    }
}

extern "C" __global__ void __launch_bounds__(256, 4)
line_rect_capacity_scatter(
    const unsigned char* __restrict__ validity,
    const signed char* __restrict__ tags,
    const int* __restrict__ family_row_offsets,
    const double* __restrict__ line_x,
    const double* __restrict__ line_y,
    const int* __restrict__ line_geometry_offsets,
    const unsigned char* __restrict__ line_empty,
    const double* __restrict__ multi_x,
    const double* __restrict__ multi_y,
    const int* __restrict__ multi_geometry_offsets,
    const int* __restrict__ multi_part_offsets,
    const unsigned char* __restrict__ multi_empty,
    const int* __restrict__ run_counts,
    const int* __restrict__ single_coord_offsets,
    const int* __restrict__ multi_part_output_offsets,
    const int* __restrict__ multi_coord_offsets,
    int* __restrict__ out_multi_part_offsets,
    double* __restrict__ out_single_x,
    double* __restrict__ out_single_y,
    double* __restrict__ out_multi_x,
    double* __restrict__ out_multi_y,
    const double xmin,
    const double ymin,
    const double xmax,
    const double ymax,
    const int row_count
) {
    const int stride = blockDim.x * gridDim.x;
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < row_count;
         row += stride) {
        const int output_runs = run_counts[row];
        if (output_runs <= 0 || !validity[row]) continue;

        const int family = (int)tags[row];
        const int family_row = family_row_offsets[row];
        int part_start = 0;
        int part_end = 0;
        if (family == 1 && !line_empty[family_row]) {
            part_start = family_row;
            part_end = family_row + 1;
        } else if (family == 4 && !multi_empty[family_row]) {
            part_start = multi_geometry_offsets[family_row];
            part_end = multi_geometry_offsets[family_row + 1];
        } else {
            continue;
        }

        int part_write = multi_part_output_offsets[row];
        int coord_write = output_runs == 1
            ? single_coord_offsets[row]
            : multi_coord_offsets[row];
        int started = 0;
        double prev_x1 = 0.0;
        double prev_y1 = 0.0;

        for (int part = part_start; part < part_end; ++part) {
            const int coord_start = family == 1
                ? line_geometry_offsets[family_row]
                : multi_part_offsets[part];
            const int coord_end = family == 1
                ? line_geometry_offsets[family_row + 1]
                : multi_part_offsets[part + 1];
            int part_boundary = started;
            for (int coord = coord_start; coord < coord_end - 1; ++coord) {
                const double* x = family == 1 ? line_x : multi_x;
                const double* y = family == 1 ? line_y : multi_y;
                double sx, sy, ex, ey;
                if (!line_rect_clip_segment(
                    x[coord], y[coord], x[coord + 1], y[coord + 1],
                    xmin, ymin, xmax, ymax, &sx, &sy, &ex, &ey
                )) continue;

                const int new_run = !started || part_boundary
                    || fabs(prev_x1 - sx) > EPSILON
                    || fabs(prev_y1 - sy) > EPSILON;
                double* out_x = output_runs == 1 ? out_single_x : out_multi_x;
                double* out_y = output_runs == 1 ? out_single_y : out_multi_y;
                if (new_run) {
                    if (output_runs > 1) {
                        if (started) part_write += 1;
                        out_multi_part_offsets[part_write] = coord_write;
                    }
                    out_x[coord_write] = sx;
                    out_y[coord_write] = sy;
                    coord_write += 1;
                    started = 1;
                }
                out_x[coord_write] = ex;
                out_y[coord_write] = ey;
                coord_write += 1;
                prev_x1 = ex;
                prev_y1 = ey;
                part_boundary = 0;
            }
        }
        if (output_runs > 1) {
            out_multi_part_offsets[multi_part_output_offsets[row] + output_runs] = coord_write;
        }
    }
}
"""
)
_LINE_ROW_KERNEL_NAMES = (
    "line_rect_capacity_count",
    "line_rect_capacity_scatter",
)
