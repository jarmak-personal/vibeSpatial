"""NVRTC kernel sources for clip_rect."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE

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

_SUTHERLAND_HODGMAN_KERNEL_SOURCE = STRIP_CLOSURE_DEVICE + r"""
#define EPSILON 1e-12

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
_SH_KERNEL_NAMES = ("sh_count_vertices", "sh_clip_rings")
# ---------------------------------------------------------------------------
# GPU Liang-Barsky line clip kernel (NVRTC)
# ---------------------------------------------------------------------------
# Per-segment clip against a rectangle.  Each thread processes one segment.

_LIANG_BARSKY_KERNEL_SOURCE = r"""
#define LB_EPSILON 1e-12

extern "C" __global__ void lb_clip_segments(
    const double* seg_x0,
    const double* seg_y0,
    const double* seg_x1,
    const double* seg_y1,
    double* out_x0,
    double* out_y0,
    double* out_x1,
    double* out_y1,
    unsigned char* out_valid,
    double xmin, double ymin, double xmax, double ymax,
    int segment_count
) {
  const int seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= segment_count) return;

  double x0 = seg_x0[seg], y0 = seg_y0[seg];
  double x1 = seg_x1[seg], y1 = seg_y1[seg];
  double dx = x1 - x0, dy = y1 - y0;

  double p[4] = {-dx, dx, -dy, dy};
  double q[4] = {x0 - xmin, xmax - x0, y0 - ymin, ymax - y0};

  double u1 = 0.0, u2 = 1.0;
  for (int k = 0; k < 4; k++) {
    if (fabs(p[k]) <= LB_EPSILON) {
      if (q[k] < 0.0) { out_valid[seg] = 0; return; }
      continue;
    }
    double t = q[k] / p[k];
    if (p[k] < 0.0) { if (t > u1) u1 = t; }
    else             { if (t < u2) u2 = t; }
    if (u1 > u2) { out_valid[seg] = 0; return; }
  }

  double cx0 = x0 + u1 * dx, cy0 = y0 + u1 * dy;
  double cx1 = x0 + u2 * dx, cy1 = y0 + u2 * dy;

  /* Reject degenerate segments */
  double ddx = cx0 - cx1, ddy = cy0 - cy1;
  if (ddx * ddx + ddy * ddy < LB_EPSILON * LB_EPSILON) {
    out_valid[seg] = 0;
    return;
  }

  out_x0[seg] = cx0;
  out_y0[seg] = cy0;
  out_x1[seg] = cx1;
  out_y1[seg] = cy1;
  out_valid[seg] = 1;
}
"""
_LB_KERNEL_NAMES = ("lb_clip_segments",)
# ---------------------------------------------------------------------------
# Segmented arange NVRTC kernel (Tier 1 — geometry-specific gather index)
# ---------------------------------------------------------------------------
# Given per-ring (start, length, write_offset), produce a flat array of
# gather indices: for ring i, write  start[i], start[i]+1, ..., start[i]+len[i]-1
# at output[write_offset[i] .. write_offset[i]+len[i]).
# One thread per surviving ring — each thread writes a short range.

_SEGMENTED_ARANGE_KERNEL_SOURCE = r"""
extern "C" __global__ void __launch_bounds__(256, 4)
segmented_arange(
    const long long* __restrict__ starts,
    const long long* __restrict__ lens,
    const long long* __restrict__ write_offsets,
    long long* __restrict__ output,
    const int n_segments
) {
    const int stride = blockDim.x * gridDim.x;
    for (int seg = blockIdx.x * blockDim.x + threadIdx.x;
         seg < n_segments;
         seg += stride) {
        long long base = starts[seg];
        long long off  = write_offsets[seg];
        long long len  = lens[seg];
        for (long long j = 0; j < len; j++) {
            output[off + j] = base + j;
        }
    }
}
"""
_SEG_ARANGE_KERNEL_NAMES = ("segmented_arange",)
