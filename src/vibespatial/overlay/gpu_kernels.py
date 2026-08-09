"""NVRTC kernel sources for overlay/gpu.py.

This module holds the CUDA C++ source strings and kernel name tuples for
the six NVRTC compilation groups used by the GPU overlay pipeline.  All
Python dispatch logic, CCCL primitive calls, warmup registration, and
compile_kernel_group wrappers remain in gpu.py.
"""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.point_in_ring import (
    POINT_IN_RING_DEVICE,
    POINT_IN_RING_KIND_DEVICE,
)
from vibespatial.cuda.device_functions.point_on_segment import POINT_ON_SEGMENT_DEVICE
from vibespatial.cuda.device_functions.segment_crossing import SEGMENT_CROSSING_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

# ---------------------------------------------------------------------------
# 1. Split event emission kernels
# ---------------------------------------------------------------------------
# Kernels: emit_endpoint_split_events, count_pair_split_events,
#          scatter_pair_split_events, emit_atomic_edges

_OVERLAY_SPLIT_KERNEL_SOURCE = """
extern "C" __device__ double abs_f64(double value) {
  return value < 0.0 ? -value : value;
}

extern "C" __device__ double clamp01(double value) {
  if (value < 0.0) {
    return 0.0;
  }
  if (value > 1.0) {
    return 1.0;
  }
  return value;
}

extern "C" __device__ double project_t(
    double px,
    double py,
    double x0,
    double y0,
    double x1,
    double y1
) {
  const double dx = x1 - x0;
  const double dy = y1 - y0;
  if (abs_f64(dx) >= abs_f64(dy)) {
    if (dx == 0.0) {
      return 0.0;
    }
    return clamp01((px - x0) / dx);
  }
  if (dy == 0.0) {
    return 0.0;
  }
  return clamp01((py - y0) / dy);
}

extern "C" __global__ void __launch_bounds__(256, 4)
emit_endpoint_split_events(
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    int left_count,
    int right_physical_count,
    int right_logical_count,
    int* __restrict__ out_source_segment_ids,
    double* __restrict__ out_t,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int event_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= event_count) {
    return;
  }

  const int segment_row = row >> 1;
  const int endpoint_kind = row & 1;
  const int is_right = segment_row >= left_count;
  const int source_segment_id = segment_row;
  const int right_logical_index = segment_row - left_count;
  if (is_right && right_logical_index >= right_logical_count) {
    return;
  }
  const int segment_index = is_right
      ? (right_logical_index % right_physical_count)
      : segment_row;

  const double* x0_values = is_right ? right_x0 : left_x0;
  const double* y0_values = is_right ? right_y0 : left_y0;
  const double* x1_values = is_right ? right_x1 : left_x1;
  const double* y1_values = is_right ? right_y1 : left_y1;

  const double t = endpoint_kind == 0 ? 0.0 : 1.0;
  const double x = endpoint_kind == 0 ? x0_values[segment_index] : x1_values[segment_index];
  const double y = endpoint_kind == 0 ? y0_values[segment_index] : y1_values[segment_index];

  out_source_segment_ids[row] = source_segment_id;
  out_t[row] = t;
  out_x[row] = x;
  out_y[row] = y;
}

extern "C" __global__ void __launch_bounds__(256, 4)
count_pair_split_events(
    const signed char* __restrict__ kinds,
    int* __restrict__ out_counts,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= row_count) {
    return;
  }
  const signed char kind = kinds[row];
  /* Predicated write: branchless count lookup. */
  out_counts[row] = (kind == 1 || kind == 2) ? 2 : (kind == 3) ? 4 : 0;
}

extern "C" __global__ void __launch_bounds__(256, 4)
scatter_pair_split_events(
    const int* __restrict__ left_lookup,
    const int* __restrict__ right_lookup,
    const int* __restrict__ pair_rows,
    const signed char* __restrict__ kinds,
    const double* __restrict__ point_x,
    const double* __restrict__ point_y,
    const double* __restrict__ overlap_x0,
    const double* __restrict__ overlap_y0,
    const double* __restrict__ overlap_x1,
    const double* __restrict__ overlap_y1,
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    const int* __restrict__ pair_offsets,
    int left_count,
    int right_physical_count,
    int broadcast_right,
    int* __restrict__ out_source_segment_ids,
    double* __restrict__ out_t,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int valid = row < row_count;

  /* Warp-cooperative skip: if no thread in this warp has a non-disjoint
     pair, the entire warp can return without reading any lookup/offset
     arrays.  After coarse filtering, ~85-95% of candidate pairs are
     disjoint, so most warps skip entirely. */
  const signed char kind = valid ? kinds[row] : 0;
  if (__ballot_sync(0xFFFFFFFF, kind != 0) == 0) {
    return;
  }

  if (!valid || kind == 0) {
    return;
  }

  const int left_index = left_lookup[row];
  const int right_index = right_lookup[row];
  const int right_source_id = left_count + right_index + (
      broadcast_right ? pair_rows[row] * right_physical_count : 0
  );
  const int base = pair_offsets[row];

  if (kind == 1 || kind == 2) {
    double x = point_x[row];
    double y = point_y[row];
    const double left_t = project_t(x, y, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double right_t = project_t(x, y, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);
    out_source_segment_ids[base + 0] = left_index;
    out_t[base + 0] = left_t;
    out_x[base + 0] = x;
    out_y[base + 0] = y;
    out_source_segment_ids[base + 1] = right_source_id;
    out_t[base + 1] = right_t;
    out_x[base + 1] = x;
    out_y[base + 1] = y;
    return;
  }

  if (kind == 3) {
    double x0 = overlap_x0[row];
    double y0 = overlap_y0[row];
    double x1 = overlap_x1[row];
    double y1 = overlap_y1[row];

    const double left_t0 = project_t(x0, y0, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double left_t1 = project_t(x1, y1, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double right_t0 = project_t(x0, y0, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);
    const double right_t1 = project_t(x1, y1, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);
    out_source_segment_ids[base + 0] = left_index;
    out_t[base + 0] = left_t0;
    out_x[base + 0] = x0;
    out_y[base + 0] = y0;
    out_source_segment_ids[base + 1] = left_index;
    out_t[base + 1] = left_t1;
    out_x[base + 1] = x1;
    out_y[base + 1] = y1;
    out_source_segment_ids[base + 2] = right_source_id;
    out_t[base + 2] = right_t0;
    out_x[base + 2] = x0;
    out_y[base + 2] = y0;
    out_source_segment_ids[base + 3] = right_source_id;
    out_t[base + 3] = right_t1;
    out_x[base + 3] = x1;
    out_y[base + 3] = y1;
  }
}

__device__ __forceinline__ bool split_key_less(
    int left_source,
    double left_t,
    int right_source,
    double right_t
) {
  return left_source < right_source ||
      (left_source == right_source && left_t < right_t);
}

__device__ __forceinline__ bool split_key_less_equal(
    int left_source,
    double left_t,
    int right_source,
    double right_t
) {
  return left_source < right_source ||
      (left_source == right_source && left_t <= right_t);
}

// Rank two exact lexicographic split-event runs into one stable merge. Left
// rows use lower_bound in the right run; right rows use upper_bound in the
// left run so constructive right payloads win only truly equal fp64 keys.
extern "C" __global__ void __launch_bounds__(256, 4)
rank_exact_split_event_merge(
    const int* __restrict__ left_source_ids,
    const double* __restrict__ left_t,
    int left_count,
    const int* __restrict__ right_source_ids,
    const double* __restrict__ right_t,
    int right_count,
    long long* __restrict__ out_left_positions,
    long long* __restrict__ out_right_positions,
    int total_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= total_count) return;
  if (row < left_count) {
    const int source = left_source_ids[row];
    const double t = left_t[row];
    int lo = 0;
    int hi = right_count;
    while (lo < hi) {
      const int mid = lo + ((hi - lo) >> 1);
      if (split_key_less(
              right_source_ids[mid], right_t[mid], source, t)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out_left_positions[row] = (long long)row + (long long)lo;
    return;
  }
  const int right_row = row - left_count;
  const int source = right_source_ids[right_row];
  const double t = right_t[right_row];
  int lo = 0;
  int hi = left_count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (split_key_less_equal(
            left_source_ids[mid], left_t[mid], source, t)) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  out_right_positions[right_row] =
      (long long)right_row + (long long)lo;
}

extern "C" __global__ void __launch_bounds__(256, 4)
emit_atomic_edges(
    const int* __restrict__ source_segment_ids,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const unsigned char* __restrict__ adjacency_mask,
    const int* __restrict__ adjacency_offsets,
    int* __restrict__ out_source_segment_ids,
    signed char* __restrict__ out_direction,
    double* __restrict__ out_src_x,
    double* __restrict__ out_src_y,
    double* __restrict__ out_dst_x,
    double* __restrict__ out_dst_y,
    int row_count
) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int valid = row < row_count;

  /* Warp-cooperative skip: if no thread in this warp has an adjacent
     pair (adjacency_mask set), skip all global memory writes. */
  const unsigned char mask_val = valid ? adjacency_mask[row] : 0;
  if (__ballot_sync(0xFFFFFFFF, mask_val) == 0) {
    return;
  }

  if (!valid || mask_val == 0) {
    return;
  }

  const int base = adjacency_offsets[row] * 2;
  const int segment_id = source_segment_ids[row];
  const double src_x = x[row];
  const double src_y = y[row];
  const double dst_x = x[row + 1];
  const double dst_y = y[row + 1];

  out_source_segment_ids[base + 0] = segment_id;
  out_direction[base + 0] = 0;
  out_src_x[base + 0] = src_x;
  out_src_y[base + 0] = src_y;
  out_dst_x[base + 0] = dst_x;
  out_dst_y[base + 0] = dst_y;

  out_source_segment_ids[base + 1] = segment_id;
  out_direction[base + 1] = 1;
  out_src_x[base + 1] = dst_x;
  out_src_y[base + 1] = dst_y;
  out_dst_x[base + 1] = src_x;
  out_dst_y[base + 1] = src_y;
}
"""

_OVERLAY_SPLIT_KERNEL_NAMES = (
    "emit_endpoint_split_events",
    "count_pair_split_events",
    "scatter_pair_split_events",
    "rank_exact_split_event_merge",
    "emit_atomic_edges",
)

# ---------------------------------------------------------------------------
# 2. Half-edge face traversal kernels
# ---------------------------------------------------------------------------
# Kernels: compute_face_metrics, compute_face_sample_points

_OVERLAY_FACE_WALK_KERNEL_SOURCE = (
    SPATIAL_TOLERANCE_PREAMBLE
    + r"""
// -------------------------------------------------------------------
// Phase 1: GPU Face Walk via Pointer Jumping
// -------------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(256, 4)
mark_endpoint_group_ends(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ source_rows,
    const int* __restrict__ point_order,
    int* __restrict__ out_group_end,
    int isolate_rows,
    int edge_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= edge_count) return;
  if (pos + 1 >= edge_count) {
    out_group_end[pos] = 0;
    return;
  }
  const int current = point_order[pos];
  const int next = point_order[pos + 1];
  const bool changed =
      src_x[current] != src_x[next]
      || src_y[current] != src_y[next]
      || (isolate_rows != 0 && source_rows[current] != source_rows[next]);
  out_group_end[pos] = changed ? 1 : 0;
}

extern "C" __global__ void __launch_bounds__(256, 4)
build_radial_successors(
    const int* __restrict__ src_node_ids,
    const int* __restrict__ sorted_edge_ids,
    int* __restrict__ out_next_edge_ids,
    int edge_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= edge_count) return;
  const int edge = sorted_edge_ids[pos];
  const int node = src_node_ids[edge];
  int predecessor_pos = pos - 1;
  if (pos == 0 || src_node_ids[sorted_edge_ids[pos - 1]] != node) {
    int lo = pos + 1;
    int hi = edge_count;
    while (lo < hi) {
      const int mid = lo + ((hi - lo) >> 1);
      if (src_node_ids[sorted_edge_ids[mid]] <= node) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    predecessor_pos = lo - 1;
  }
  out_next_edge_ids[edge ^ 1] = sorted_edge_ids[predecessor_pos];
}

static __device__ __forceinline__ bool
vs_face_contains_point(
    double px,
    double py,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ next_edge_ids,
    const int* __restrict__ face_edge_ids,
    int start,
    int end,
    int total_edge_count
) {
  bool inside = false;
  for (int k = start; k < end; ++k) {
    const int eid = face_edge_ids[k];
    if (eid < 0 || eid >= total_edge_count) continue;
    const int next_eid = (int)next_edge_ids[eid];
    if (next_eid < 0 || next_eid >= total_edge_count) continue;
    const double x0 = src_x[eid];
    const double y0 = src_y[eid];
    const double x1 = src_x[next_eid];
    const double y1 = src_y[next_eid];
    const bool crosses = (y1 > py) != (y0 > py);
    if (crosses) {
      const double denom = y0 - y1;
      if (denom != 0.0) {
        const double x_intersection = ((x0 - x1) * (py - y1) / denom) + x1;
        if (px < x_intersection) inside = !inside;
      }
    }
  }
  return inside;
}

// Compute per-face area and centroid in one pass over the face edge span.
// Each block handles one face; threads cooperatively walk the sorted edge ids
// for that face and reduce. Coordinates are translated to the first vertex so
// positive-area slivers retain their sign independently of world-coordinate
// magnitude:
//   cross = (x0 - ox) * (y1 - oy) - (x1 - ox) * (y0 - oy)
//   cx += (x0 + x1) * cross
//   cy += (y0 + y1) * cross
// When twice_area is zero, fall back to the mean of the source vertices.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_face_metrics(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ next_edge_ids,
    const int* __restrict__ sorted_edge_ids,
    const int* __restrict__ face_starts,
    const int* __restrict__ face_ends,
    double* __restrict__ out_signed_area,
    double* __restrict__ out_centroid_x,
    double* __restrict__ out_centroid_y,
    int face_count,
    int total_edge_count
) {
  const int f = blockIdx.x;
  const int tid = threadIdx.x;
  if (f >= face_count) return;

  const int start = face_starts[f];
  const int end = face_ends[f];
  const int origin_edge = sorted_edge_ids[start];
  const double origin_x = src_x[origin_edge];
  const double origin_y = src_y[origin_edge];

  __shared__ double sh_cross[256];
  __shared__ double sh_cx[256];
  __shared__ double sh_cy[256];
  __shared__ double sh_sx[256];
  __shared__ double sh_sy[256];
  __shared__ int sh_count[256];

  double local_cross = 0.0;
  double local_cx = 0.0;
  double local_cy = 0.0;
  double local_sx = 0.0;
  double local_sy = 0.0;
  int local_count = 0;

  for (int pos = start + tid; pos < end; pos += blockDim.x) {
    const int eid = sorted_edge_ids[pos];
    if (eid < 0 || eid >= total_edge_count) continue;
    const int next_i = (int)next_edge_ids[eid];
    if (next_i < 0 || next_i >= total_edge_count) continue;
    const double x0 = src_x[eid] - origin_x;
    const double y0 = src_y[eid] - origin_y;
    const double x1 = src_x[next_i] - origin_x;
    const double y1 = src_y[next_i] - origin_y;
    const double cross = x0 * y1 - x1 * y0;
    local_cross += cross;
    local_cx += (x0 + x1) * cross;
    local_cy += (y0 + y1) * cross;
    local_sx += x0;
    local_sy += y0;
    local_count += 1;
  }

  sh_cross[tid] = local_cross;
  sh_cx[tid] = local_cx;
  sh_cy[tid] = local_cy;
  sh_sx[tid] = local_sx;
  sh_sy[tid] = local_sy;
  sh_count[tid] = local_count;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_cross[tid] += sh_cross[tid + stride];
      sh_cx[tid] += sh_cx[tid + stride];
      sh_cy[tid] += sh_cy[tid + stride];
      sh_sx[tid] += sh_sx[tid + stride];
      sh_sy[tid] += sh_sy[tid + stride];
      sh_count[tid] += sh_count[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const double twice_area = sh_cross[0];
    out_signed_area[f] = twice_area * 0.5;
    if (twice_area == 0.0 || sh_count[0] == 0) {
      const double denom = sh_count[0] > 0 ? (double)sh_count[0] : 1.0;
      out_centroid_x[f] = origin_x + sh_sx[0] / denom;
      out_centroid_y[f] = origin_y + sh_sy[0] / denom;
    } else {
      out_centroid_x[f] = origin_x + sh_cx[0] / (3.0 * twice_area);
      out_centroid_y[f] = origin_y + sh_cy[0] / (3.0 * twice_area);
    }
  }
}

// One thread per face: compute a sample point by walking the face edges.
// face_starts[f] and face_ends[f] give the range into sorted_edge_ids
// (edges sorted by face_id). The sample point is the perpendicular-offset
// midpoint of the first non-degenerate edge.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_face_sample_points(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ next_edge_ids,
    const int* __restrict__ face_starts,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ signed_area,
    const double* __restrict__ centroid_x,
    const double* __restrict__ centroid_y,
    double* __restrict__ out_label_x,
    double* __restrict__ out_label_y,
    signed char* __restrict__ out_bounded,
    double area_epsilon,
    int face_count,
    int total_edge_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;

  const double area = signed_area[f];
  if (fabs(area) <= area_epsilon) {
    out_bounded[f] = 0;
    out_label_x[f] = 0.0;
    out_label_y[f] = 0.0;
    return;
  }
  out_bounded[f] = area > area_epsilon ? 1 : 0;

  const int start = face_starts[f];
  const int end = face_starts[f + 1];
  const int n_edges = end - start;

  // A cycle centroid is not a valid face label when the cycle contains a
  // nested hole contour. Walk directed edges and probe locally inward instead.
  double best_lx = src_x[face_edge_ids[start]];
  double best_ly = src_y[face_edge_ids[start]];
  double extent_min_x = best_lx, extent_max_x = best_lx;
  double extent_min_y = best_ly, extent_max_y = best_ly;

  for (int k = 0; k < n_edges; k++) {
    const int eid = face_edge_ids[start + k];
    double ex = src_x[eid];
    double ey = src_y[eid];
    if (ex < extent_min_x) extent_min_x = ex;
    if (ex > extent_max_x) extent_max_x = ex;
    if (ey < extent_min_y) extent_min_y = ey;
    if (ey > extent_max_y) extent_max_y = ey;
  }

  double extent = extent_max_x - extent_min_x;
  double ey_range = extent_max_y - extent_min_y;
  if (ey_range > extent) extent = ey_range;
  if (extent <= 0.0) {
    out_label_x[f] = best_lx;
    out_label_y[f] = best_ly;
    return;
  }
  // Bound the first probe by both face extent and area-per-edge. Unlike the
  // old one-unit floor, this remains inside the scale of narrow real faces.
  const double area_step = fabs(area) / (extent * (double)n_edges);
  const double epsilon = fmin(extent * 1e-6, area_step * 0.5);

  for (int k = 0; k < n_edges; k++) {
    const int eid = face_edge_ids[start + k];
    const int next_eid = (int)next_edge_ids[eid];
    // Bounds check: prevent ILLEGAL_ADDRESS from corrupted topology.
    if (next_eid < 0 || next_eid >= total_edge_count) continue;
    const double x0 = src_x[eid];
    const double y0 = src_y[eid];
    const double x1 = src_x[next_eid];
    const double y1 = src_y[next_eid];
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double length = sqrt(dx * dx + dy * dy);
    if (length <= 0.0) continue;
    double trial = epsilon;
    for (int attempt = 0; attempt < 16; ++attempt) {
      const double midpoint_x = x0 + (x1 - x0) * 0.5;
      const double midpoint_y = y0 + (y1 - y0) * 0.5;
      const double normal_x = -dy / length;
      const double normal_y = dx / length;
      double candidate_x = midpoint_x + normal_x * trial;
      double candidate_y = midpoint_y + normal_y * trial;
      if (normal_x != 0.0 && candidate_x == midpoint_x) {
        candidate_x = nextafter(
            midpoint_x,
            normal_x > 0.0 ? 1.7976931348623157e308 : -1.7976931348623157e308
        );
      }
      if (normal_y != 0.0 && candidate_y == midpoint_y) {
        candidate_y = nextafter(
            midpoint_y,
            normal_y > 0.0 ? 1.7976931348623157e308 : -1.7976931348623157e308
        );
      }
      best_lx = candidate_x;
      best_ly = candidate_y;
      const int cycle_contains = vs_face_contains_point(
          candidate_x, candidate_y, src_x, src_y, next_edge_ids,
          face_edge_ids, start, end, total_edge_count);
      if ((area > 0.0 && cycle_contains) || (area < 0.0 && !cycle_contains)) {
        out_label_x[f] = candidate_x;
        out_label_y[f] = candidate_y;
        return;
      }
      trial *= 0.25;
    }
  }

  out_label_x[f] = best_lx;
  out_label_y[f] = best_ly;
}

// Classify nesting in a disconnected boundary-contour relation. One block
// owns one face side. Threads first tag every bounded contour with its face id,
// then cooperatively test both face sides against other same-row bounded
// contours. A bounded side is selected at even ancestor depth; an unbounded
// side is selected at odd containment depth. Selecting both sides by parity is
// what exposes hole contours to the canonical selected-boundary walk.
extern "C" __global__ void __launch_bounds__(256, 4)
count_boundary_face_nesting_depth(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const signed char* __restrict__ bounded_mask,
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ next_edge_ids,
    const int* __restrict__ source_rows,
    int* __restrict__ source_ring_ids,
    int* __restrict__ out_depth,
    int isolate_rows,
    int face_count,
    int total_edge_count
) {
  const int face = blockIdx.x;
  const int tid = threadIdx.x;
  if (face >= face_count) return;

  const int start = face_offsets[face];
  const int end = face_offsets[face + 1];
  if (bounded_mask[face] != 0) {
    for (int pos = start + tid; pos < end; pos += blockDim.x) {
      source_ring_ids[face_edge_ids[pos]] = face + 1;
    }
  }

  __shared__ int sh_depth[256];
  int local_depth = 0;
  if (start < end) {
    const int target_row = source_rows[face_edge_ids[start]];
    const double px = label_x[face];
    const double py = label_y[face];
    for (int container = tid; container < face_count; container += blockDim.x) {
      if (container == face || bounded_mask[container] == 0) continue;
      const int container_start = face_offsets[container];
      const int container_end = face_offsets[container + 1];
      if (container_start >= container_end) continue;
      if (isolate_rows != 0
          && source_rows[face_edge_ids[container_start]] != target_row) {
        continue;
      }
      if (vs_face_contains_point(
          px, py, src_x, src_y, next_edge_ids, face_edge_ids,
          container_start, container_end, total_edge_count)) {
        local_depth += 1;
      }
    }
  }
  sh_depth[tid] = local_depth;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) sh_depth[tid] += sh_depth[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out_depth[face] = sh_depth[0];
}

"""
)

_OVERLAY_FACE_WALK_KERNEL_NAMES = (
    "mark_endpoint_group_ends",
    "build_radial_successors",
    "compute_face_metrics",
    "compute_face_sample_points",
    "count_boundary_face_nesting_depth",
)

# ---------------------------------------------------------------------------
# 3. Face coverage labeling kernels
# ---------------------------------------------------------------------------
# Kernels: label_face_coverage_polygon, label_face_coverage_multipolygon

_OVERLAY_FACE_LABEL_KERNEL_SOURCE = (
    POINT_ON_SEGMENT_DEVICE
    + ORIENT2D_DEVICE
    + POINT_IN_RING_KIND_DEVICE
    + r"""
// -------------------------------------------------------------------
// Phase 2: GPU Face Labeling via Batch Point-in-Polygon
// -------------------------------------------------------------------
#define OVERLAY_BOUNDARY_TOLERANCE 0.0

__device__ __forceinline__ bool vs_ring_contains_point_with_boundary(
    double px,
    double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int end,
    double tolerance,
    bool* on_boundary
) {
  (void)tolerance;
  const unsigned char kind = vs_ring_point_classify(
      px, py, x, y, start, end, 0.0);
  *on_boundary = kind == (unsigned char)1;
  return kind != (unsigned char)0;
}

__device__ __forceinline__ int overlay_lower_bound_i32(
    const int* values,
    int count,
    int target
) {
  int first = 0;
  while (first < count) {
    const int step = (count - first) >> 1;
    const int probe = first + step;
    if (values[probe] < target) first = probe + 1;
    else count = probe;
  }
  return first;
}

__device__ __forceinline__ int overlay_upper_bound_i32(
    const int* values,
    int count,
    int target
) {
  int first = 0;
  while (first < count) {
    const int step = (count - first) >> 1;
    const int probe = first + step;
    if (values[probe] <= target) first = probe + 1;
    else count = probe;
  }
  return first;
}

__device__ __forceinline__ bool overlay_block_polygon_contains_point(
    double px,
    double py,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_ring_bounds,
    int polygon,
    int* warp_crossings,
    int* warp_boundary
) {
  __shared__ int polygon_crossings;
  __shared__ int polygon_boundary;
  if (threadIdx.x == 0) {
    polygon_crossings = 0;
    polygon_boundary = 0;
  }
  __syncthreads();

  const int ring_start = polygon_geometry_offsets[polygon];
  const int ring_end = polygon_geometry_offsets[polygon + 1];
  for (int ring = ring_start; ring < ring_end; ++ring) {
    if (polygon_ring_bounds != nullptr) {
      const int base = ring * 4;
      if (
          px < polygon_ring_bounds[base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_ring_bounds[base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_ring_bounds[base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_ring_bounds[base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    const int coord_start = polygon_ring_offsets[ring];
    const int coord_end = polygon_ring_offsets[ring + 1];
    const int edge_count = coord_end - coord_start - 1;
    int local_crossings = 0;
    int local_boundary = 0;
    for (int edge = (int)threadIdx.x; edge < edge_count; edge += (int)blockDim.x) {
      const int coord = coord_start + edge;
      const double ax = polygon_x[coord];
      const double ay = polygon_y[coord];
      const double bx = polygon_x[coord + 1];
      const double by = polygon_y[coord + 1];
      const bool crosses_scanline = (ay > py) != (by > py);
      const bool boundary_candidate =
          py >= fmin(ay, by) && py <= fmax(ay, by) &&
          px >= fmin(ax, bx) && px <= fmax(ax, bx);
      if (boundary_candidate && vs_orient2d(ax, ay, bx, by, px, py) == 0) {
        local_boundary = 1;
      }
      if (crosses_scanline) {
        if (px < (bx - ax) * (py - ay) / (by - ay) + ax) {
          local_crossings ^= 1;
        }
      }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
      local_crossings ^= __shfl_xor_sync(0xFFFFFFFFu, local_crossings, offset);
      local_boundary |= __shfl_xor_sync(0xFFFFFFFFu, local_boundary, offset);
    }
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) {
      warp_crossings[warp] = local_crossings;
      warp_boundary[warp] = local_boundary;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int ring_crossings = 0;
      int ring_boundary = 0;
      const int warp_count = ((int)blockDim.x + 31) >> 5;
      for (int index = 0; index < warp_count; ++index) {
        ring_crossings ^= warp_crossings[index];
        ring_boundary |= warp_boundary[index];
      }
      polygon_crossings ^= ring_crossings;
      polygon_boundary |= ring_boundary;
    }
    __syncthreads();
  }
  return polygon_boundary != 0 || polygon_crossings != 0;
}

__device__ __forceinline__ int overlay_block_ring_point_relation(
    double px,
    double py,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_ring_offsets,
    int ring,
    int* warp_crossings,
    int* warp_boundary
) {
  const int coord_start = polygon_ring_offsets[ring];
  const int coord_end = polygon_ring_offsets[ring + 1];
  const int edge_count = coord_end - coord_start - 1;
  int local_crossings = 0;
  int local_boundary = 0;
  for (int edge = (int)threadIdx.x; edge < edge_count;
       edge += (int)blockDim.x) {
    const int coord = coord_start + edge;
    const double ax = polygon_x[coord];
    const double ay = polygon_y[coord];
    const double bx = polygon_x[coord + 1];
    const double by = polygon_y[coord + 1];
    const bool crosses_scanline = (ay > py) != (by > py);
    const bool boundary_candidate =
        py >= fmin(ay, by) && py <= fmax(ay, by) &&
        px >= fmin(ax, bx) && px <= fmax(ax, bx);
    if (boundary_candidate && vs_orient2d(ax, ay, bx, by, px, py) == 0) {
      local_boundary = 1;
    }
    if (crosses_scanline &&
        px < (bx - ax) * (py - ay) / (by - ay) + ax) {
      local_crossings ^= 1;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_crossings ^= __shfl_xor_sync(0xFFFFFFFFu, local_crossings, offset);
    local_boundary |= __shfl_xor_sync(0xFFFFFFFFu, local_boundary, offset);
  }
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  if (lane == 0) {
    warp_crossings[warp] = local_crossings;
    warp_boundary[warp] = local_boundary;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int crossings = 0;
    int boundary = 0;
    const int warp_count = ((int)blockDim.x + 31) >> 5;
    for (int i = 0; i < warp_count; ++i) {
      crossings ^= warp_crossings[i];
      boundary |= warp_boundary[i];
    }
    warp_crossings[0] = crossings;
    warp_boundary[0] = boundary;
  }
  __syncthreads();
  return (warp_boundary[0] != 0 ? 2 : 0) | (warp_crossings[0] != 0 ? 1 : 0);
}

__device__ __forceinline__ int overlay_block_polygon_holes_point_relation(
    double px,
    double py,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_ring_bounds,
    int polygon,
    int* warp_crossings,
    int* warp_boundary
) {
  const int ring_start = polygon_geometry_offsets[polygon] + 1;
  const int ring_end = polygon_geometry_offsets[polygon + 1];
  int local_crossings = 0;
  int local_boundary = 0;
  for (int ring = ring_start + (int)threadIdx.x; ring < ring_end;
       ring += (int)blockDim.x) {
    if (polygon_ring_bounds != nullptr) {
      const int base = ring * 4;
      if (
          px < polygon_ring_bounds[base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_ring_bounds[base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_ring_bounds[base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_ring_bounds[base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    const int coord_start = polygon_ring_offsets[ring];
    const int coord_end = polygon_ring_offsets[ring + 1];
    int ring_crossings = 0;
    for (int coord = coord_start; coord + 1 < coord_end; ++coord) {
      const double ax = polygon_x[coord];
      const double ay = polygon_y[coord];
      const double bx = polygon_x[coord + 1];
      const double by = polygon_y[coord + 1];
      const bool crosses_scanline = (ay > py) != (by > py);
      const bool boundary_candidate =
          py >= fmin(ay, by) && py <= fmax(ay, by) &&
          px >= fmin(ax, bx) && px <= fmax(ax, bx);
      if (boundary_candidate && vs_orient2d(ax, ay, bx, by, px, py) == 0) {
        local_boundary = 1;
      }
      if (crosses_scanline &&
          px < (bx - ax) * (py - ay) / (by - ay) + ax) {
        ring_crossings ^= 1;
      }
    }
    local_crossings ^= ring_crossings;
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_crossings ^= __shfl_xor_sync(0xFFFFFFFFu, local_crossings, offset);
    local_boundary |= __shfl_xor_sync(0xFFFFFFFFu, local_boundary, offset);
  }
  const int lane = (int)threadIdx.x & 31;
  const int warp = (int)threadIdx.x >> 5;
  if (lane == 0) {
    warp_crossings[warp] = local_crossings;
    warp_boundary[warp] = local_boundary;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int crossings = 0;
    int boundary = 0;
    const int warp_count = ((int)blockDim.x + 31) >> 5;
    for (int i = 0; i < warp_count; ++i) {
      crossings ^= warp_crossings[i];
      boundary |= warp_boundary[i];
    }
    warp_crossings[0] = crossings;
    warp_boundary[0] = boundary;
  }
  __syncthreads();
  return (warp_boundary[0] != 0 ? 2 : 0) | (warp_crossings[0] != 0 ? 1 : 0);
}

// Test face sample points against all polygons on one side.
// One thread per face.
// polygon_geometry_offsets: maps polygon row -> ring range
// polygon_ring_offsets: maps ring -> coordinate range
// polygon_x, polygon_y: flat coordinate arrays
// polygon_count: number of polygons
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const int* __restrict__ polygon_source_rows,
    int polygon_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;
  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && polygon_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;

  for (int poly = 0; poly < polygon_count; ++poly) {
    if (restrict_row && polygon_source_rows[poly] != face_row) continue;
    if (polygon_bounds != nullptr) {
      const int bounds_base = poly * 4;
      if (
          px < polygon_bounds[bounds_base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_bounds[bounds_base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_bounds[bounds_base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_bounds[bounds_base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    const int ring_start = polygon_geometry_offsets[poly];
    const int ring_end = polygon_geometry_offsets[poly + 1];
    bool inside = false;
    for (int ring = ring_start; ring < ring_end; ++ring) {
      bool on_boundary = false;
      const int coord_start = polygon_ring_offsets[ring];
      const int coord_end = polygon_ring_offsets[ring + 1];
      const bool ring_inside = vs_ring_contains_point_with_boundary(
          px, py, polygon_x, polygon_y, coord_start, coord_end,
          OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
      if (on_boundary) { inside = true; break; }
      if (ring_inside) inside = !inside;
    }
    if (inside) { out_covered[f] = 1; return; }
  }
}

// Warp-cooperative polygon coverage for grouped batches. One warp owns one
// face and lanes fan out over polygon rows, preserving the same optional
// source-row restriction and bounds pruning as label_face_coverage_polygon.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_warp(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const int* __restrict__ polygon_source_rows,
    int polygon_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = tid >> 5;
  const int lane = tid & 31;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;

  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && polygon_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;

  for (int base_poly = 0; base_poly < polygon_count; base_poly += 32) {
    const int poly = base_poly + lane;
    int hit = 0;
    if (poly < polygon_count) {
      hit = 1;
      if (restrict_row && polygon_source_rows[poly] != face_row) {
        hit = 0;
      }
      if (hit && polygon_bounds != nullptr) {
        const int bounds_base = poly * 4;
        if (
            px < polygon_bounds[bounds_base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
            px > polygon_bounds[bounds_base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
            py < polygon_bounds[bounds_base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
            py > polygon_bounds[bounds_base + 3] + OVERLAY_BOUNDARY_TOLERANCE
        ) {
          hit = 0;
        }
      }
      if (hit) {
        const int ring_start = polygon_geometry_offsets[poly];
        const int ring_end = polygon_geometry_offsets[poly + 1];
        bool inside = false;
        for (int ring = ring_start; ring < ring_end; ++ring) {
          bool on_boundary = false;
          const int coord_start = polygon_ring_offsets[ring];
          const int coord_end = polygon_ring_offsets[ring + 1];
          const bool ring_inside = vs_ring_contains_point_with_boundary(
              px, py, polygon_x, polygon_y, coord_start, coord_end,
              OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
          if (on_boundary) {
            inside = true;
            break;
          }
          if (ring_inside) inside = !inside;
        }
        hit = inside ? 1 : 0;
      }
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, hit != 0);
    if (mask) {
      if (lane == 0) out_covered[f] = 1;
      return;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_block(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const double* __restrict__ polygon_ring_bounds,
    const int* __restrict__ polygon_source_rows,
    int polygon_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int face = (int)blockIdx.x;
  if (face >= face_count || out_covered[face] == 1) return;
  const double px = label_x[face];
  const double py = label_y[face];
  const bool restrict_row = face_source_rows != nullptr && polygon_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[face] : -1;
  __shared__ int warp_crossings[8];
  __shared__ int warp_boundary[8];

  for (int polygon = 0; polygon < polygon_count; ++polygon) {
    if (restrict_row && polygon_source_rows[polygon] != face_row) continue;
    if (polygon_bounds != nullptr) {
      const int base = polygon * 4;
      if (
          px < polygon_bounds[base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_bounds[base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_bounds[base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_bounds[base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    const int ring_count =
        polygon_geometry_offsets[polygon + 1] - polygon_geometry_offsets[polygon];
    bool covered;
    if (ring_count > 32) {
      const int exterior_relation = overlay_block_ring_point_relation(
          px, py, polygon_x, polygon_y, polygon_ring_offsets,
          polygon_geometry_offsets[polygon], warp_crossings, warp_boundary);
      const int holes_relation = overlay_block_polygon_holes_point_relation(
          px, py, polygon_x, polygon_y, polygon_geometry_offsets,
          polygon_ring_offsets, polygon_ring_bounds, polygon,
          warp_crossings, warp_boundary);
      covered = ((exterior_relation | holes_relation) & 2) != 0 ||
          ((exterior_relation ^ holes_relation) & 1) != 0;
    } else {
      covered = overlay_block_polygon_contains_point(
            px, py, polygon_x, polygon_y, polygon_geometry_offsets,
            polygon_ring_offsets, polygon_ring_bounds, polygon,
            warp_crossings, warp_boundary);
    }
    if (covered) {
      if (threadIdx.x == 0) out_covered[face] = 1;
      return;
    }
  }
}

// Row-isolated fast path: each face only needs to test against the polygon
// whose row id matches the face source row.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_same_row(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    int polygon_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;
  const int poly = face_source_rows[f];
  if (poly < 0 || poly >= polygon_count) return;
  const double px = label_x[f];
  const double py = label_y[f];
  const int ring_start = polygon_geometry_offsets[poly];
  const int ring_end = polygon_geometry_offsets[poly + 1];
  bool inside = false;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    bool on_boundary = false;
    const int coord_start = polygon_ring_offsets[ring];
    const int coord_end = polygon_ring_offsets[ring + 1];
    const bool ring_inside = vs_ring_contains_point_with_boundary(
        px, py, polygon_x, polygon_y, coord_start, coord_end,
        OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
    if (on_boundary) { out_covered[f] = 1; return; }
    if (ring_inside) inside = !inside;
  }
  if (inside) out_covered[f] = 1;
}

// Row-indirected grouped path: logical rows may be indexed views over shared
// physical family rows.  Source-row matching therefore has to happen against
// logical rows, then dereference the physical family row for coordinate tests.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_logical_rows(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;
  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;

  for (int logical = logical_start; logical < logical_end; ++logical) {
    const int poly = logical_family_rows[logical];
    if (poly < 0) continue;
    if (polygon_bounds != nullptr) {
      const int bounds_base = poly * 4;
      if (
          px < polygon_bounds[bounds_base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_bounds[bounds_base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_bounds[bounds_base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_bounds[bounds_base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    const int ring_start = polygon_geometry_offsets[poly];
    const int ring_end = polygon_geometry_offsets[poly + 1];
    bool inside = false;
    for (int ring = ring_start; ring < ring_end; ++ring) {
      bool on_boundary = false;
      const int coord_start = polygon_ring_offsets[ring];
      const int coord_end = polygon_ring_offsets[ring + 1];
      const bool ring_inside = vs_ring_contains_point_with_boundary(
          px, py, polygon_x, polygon_y, coord_start, coord_end,
          OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
      if (on_boundary) { inside = true; break; }
      if (ring_inside) inside = !inside;
    }
    if (inside) { out_covered[f] = 1; return; }
  }
}

// Warp-cooperative row-indirected grouped path.  One warp owns one face and
// lanes fan out over logical rows before dereferencing the physical polygon
// row.  This keeps indexed grouped coverage device-shaped without a serial
// per-face logical-row scan.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_logical_rows_warp(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = tid >> 5;
  const int lane = tid & 31;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;

  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;

  for (int base_logical = logical_start; base_logical < logical_end; base_logical += 32) {
    const int logical = base_logical + lane;
    int hit = 0;
    if (logical < logical_end) {
      hit = 1;
      const int poly = hit ? logical_family_rows[logical] : -1;
      if (poly < 0) {
        hit = 0;
      }
      if (hit && polygon_bounds != nullptr) {
        const int bounds_base = poly * 4;
        if (
            px < polygon_bounds[bounds_base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
            px > polygon_bounds[bounds_base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
            py < polygon_bounds[bounds_base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
            py > polygon_bounds[bounds_base + 3] + OVERLAY_BOUNDARY_TOLERANCE
        ) {
          hit = 0;
        }
      }
      if (hit) {
        const int ring_start = polygon_geometry_offsets[poly];
        const int ring_end = polygon_geometry_offsets[poly + 1];
        bool inside = false;
        for (int ring = ring_start; ring < ring_end; ++ring) {
          bool on_boundary = false;
          const int coord_start = polygon_ring_offsets[ring];
          const int coord_end = polygon_ring_offsets[ring + 1];
          const bool ring_inside = vs_ring_contains_point_with_boundary(
              px, py, polygon_x, polygon_y, coord_start, coord_end,
              OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
          if (on_boundary) {
            inside = true;
            break;
          }
          if (ring_inside) inside = !inside;
        }
        hit = inside ? 1 : 0;
      }
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, hit != 0);
    if (mask) {
      if (lane == 0) out_covered[f] = 1;
      return;
    }
  }
}

// Coordinate-cooperative row-indirected path. One block owns one face and
// divides the matching physical polygon edges across threads. This is the
// saturation shape for indexed views over large shared polygons.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_polygon_logical_rows_block(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    const double* __restrict__ polygon_bounds,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int face = (int)blockIdx.x;
  if (face >= face_count || out_covered[face] == 1) return;
  const double px = label_x[face];
  const double py = label_y[face];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[face] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;
  __shared__ int warp_crossings[8];
  __shared__ int warp_boundary[8];

  for (int logical = logical_start; logical < logical_end; ++logical) {
    const int polygon = logical_family_rows[logical];
    if (polygon < 0) continue;
    if (polygon_bounds != nullptr) {
      const int base = polygon * 4;
      if (
          px < polygon_bounds[base + 0] - OVERLAY_BOUNDARY_TOLERANCE ||
          px > polygon_bounds[base + 2] + OVERLAY_BOUNDARY_TOLERANCE ||
          py < polygon_bounds[base + 1] - OVERLAY_BOUNDARY_TOLERANCE ||
          py > polygon_bounds[base + 3] + OVERLAY_BOUNDARY_TOLERANCE
      ) {
        continue;
      }
    }
    if (overlay_block_polygon_contains_point(
            px, py, polygon_x, polygon_y, polygon_geometry_offsets,
            polygon_ring_offsets, nullptr, polygon,
            warp_crossings, warp_boundary)) {
      if (threadIdx.x == 0) out_covered[face] = 1;
      return;
    }
  }
}

// Compute per-polygon bounds for a polygon family buffer.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_polygon_bounds(
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_geometry_offsets,
    const int* __restrict__ polygon_ring_offsets,
    int polygon_count,
    double* __restrict__ out_bounds
) {
  const int polygon = blockIdx.x * blockDim.x + threadIdx.x;
  if (polygon >= polygon_count) return;
  const int ring_start = polygon_geometry_offsets[polygon];
  const int ring_end = polygon_geometry_offsets[polygon + 1];
  if (ring_start >= ring_end) {
    const int base = polygon * 4;
    out_bounds[base + 0] = 0.0;
    out_bounds[base + 1] = 0.0;
    out_bounds[base + 2] = 0.0;
    out_bounds[base + 3] = 0.0;
    return;
  }
  const int first_coord = polygon_ring_offsets[ring_start];
  double min_x = polygon_x[first_coord];
  double min_y = polygon_y[first_coord];
  double max_x = min_x;
  double max_y = min_y;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int coord_start = polygon_ring_offsets[ring];
    const int coord_end = polygon_ring_offsets[ring + 1];
    for (int coord = coord_start; coord < coord_end; ++coord) {
      const double x = polygon_x[coord];
      const double y = polygon_y[coord];
      min_x = fmin(min_x, x);
      min_y = fmin(min_y, y);
      max_x = fmax(max_x, x);
      max_y = fmax(max_y, y);
    }
  }
  const int base = polygon * 4;
  out_bounds[base + 0] = min_x;
  out_bounds[base + 1] = min_y;
  out_bounds[base + 2] = max_x;
  out_bounds[base + 3] = max_y;
}

// Compute per-ring bounds for coordinate-heavy broadcast point-in-polygon.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_polygon_ring_bounds(
    const double* __restrict__ polygon_x,
    const double* __restrict__ polygon_y,
    const int* __restrict__ polygon_ring_offsets,
    int ring_count,
    double* __restrict__ out_bounds
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) return;
  const int coord_start = polygon_ring_offsets[ring];
  const int coord_end = polygon_ring_offsets[ring + 1];
  const int base = ring * 4;
  if (coord_start >= coord_end) {
    out_bounds[base + 0] = 0.0;
    out_bounds[base + 1] = 0.0;
    out_bounds[base + 2] = 0.0;
    out_bounds[base + 3] = 0.0;
    return;
  }
  double min_x = polygon_x[coord_start];
  double min_y = polygon_y[coord_start];
  double max_x = min_x;
  double max_y = min_y;
  for (int coord = coord_start + 1; coord < coord_end; ++coord) {
    const double x = polygon_x[coord];
    const double y = polygon_y[coord];
    min_x = fmin(min_x, x);
    min_y = fmin(min_y, y);
    max_x = fmax(max_x, x);
    max_y = fmax(max_y, y);
  }
  out_bounds[base + 0] = min_x;
  out_bounds[base + 1] = min_y;
  out_bounds[base + 2] = max_x;
  out_bounds[base + 3] = max_y;
}

// Compute per-polygon bounds for a multipolygon family buffer.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_multipolygon_polygon_bounds(
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    int polygon_count,
    double* __restrict__ out_bounds
) {
  const int polygon = blockIdx.x * blockDim.x + threadIdx.x;
  if (polygon >= polygon_count) return;
  const int ring_start = mp_part_offsets[polygon];
  const int ring_end = mp_part_offsets[polygon + 1];
  if (ring_start >= ring_end) {
    const int base = polygon * 4;
    out_bounds[base + 0] = 0.0;
    out_bounds[base + 1] = 0.0;
    out_bounds[base + 2] = 0.0;
    out_bounds[base + 3] = 0.0;
    return;
  }
  const int first_coord = mp_ring_offsets[ring_start];
  double min_x = mp_x[first_coord];
  double min_y = mp_y[first_coord];
  double max_x = min_x;
  double max_y = min_y;
  for (int ring = ring_start; ring < ring_end; ++ring) {
    const int coord_start = mp_ring_offsets[ring];
    const int coord_end = mp_ring_offsets[ring + 1];
    for (int coord = coord_start; coord < coord_end; ++coord) {
      const double x = mp_x[coord];
      const double y = mp_y[coord];
      min_x = fmin(min_x, x);
      min_y = fmin(min_y, y);
      max_x = fmax(max_x, x);
      max_y = fmax(max_y, y);
    }
  }
  const int base = polygon * 4;
  out_bounds[base + 0] = min_x;
  out_bounds[base + 1] = min_y;
  out_bounds[base + 2] = max_x;
  out_bounds[base + 3] = max_y;
}

// Test face sample points against all multipolygons on one side.
// One thread per face.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const int* __restrict__ mp_source_rows,
    int mp_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;  // already covered by polygon pass
  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && mp_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;

  for (int mp = 0; mp < mp_count; ++mp) {
    if (restrict_row && mp_source_rows[mp] != face_row) continue;
    const int polygon_start = mp_geometry_offsets[mp];
    const int polygon_end = mp_geometry_offsets[mp + 1];
    for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
      const int ring_start = mp_part_offsets[polygon];
      const int ring_end = mp_part_offsets[polygon + 1];
      bool inside = false;
      for (int ring = ring_start; ring < ring_end; ++ring) {
        bool on_boundary = false;
        const int coord_start = mp_ring_offsets[ring];
        const int coord_end = mp_ring_offsets[ring + 1];
        const bool ring_inside = vs_ring_contains_point_with_boundary(
            px, py, mp_x, mp_y, coord_start, coord_end,
            OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
        if (on_boundary) { inside = true; break; }
        if (ring_inside) inside = !inside;
      }
      if (inside) { out_covered[f] = 1; return; }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon_block(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const int* __restrict__ mp_source_rows,
    int mp_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int face = (int)blockIdx.x;
  if (face >= face_count || out_covered[face] == 1) return;
  const double px = label_x[face];
  const double py = label_y[face];
  const bool restrict_row = face_source_rows != nullptr && mp_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[face] : -1;
  __shared__ int warp_crossings[8];
  __shared__ int warp_boundary[8];

  for (int multipolygon = 0; multipolygon < mp_count; ++multipolygon) {
    if (restrict_row && mp_source_rows[multipolygon] != face_row) continue;
    const int polygon_start = mp_geometry_offsets[multipolygon];
    const int polygon_end = mp_geometry_offsets[multipolygon + 1];
    for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
      if (overlay_block_polygon_contains_point(
              px, py, mp_x, mp_y, mp_part_offsets, mp_ring_offsets,
              nullptr, polygon, warp_crossings, warp_boundary)) {
        if (threadIdx.x == 0) out_covered[face] = 1;
        return;
      }
    }
  }
}

// Row-isolated fast path: each face only needs to test against the
// multipolygon whose row id matches the face source row.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon_same_row(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const double* __restrict__ mp_polygon_bounds,
    int mp_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;
  const int mp = face_source_rows[f];
  if (mp < 0 || mp >= mp_count) return;
  const double px = label_x[f];
  const double py = label_y[f];
  const int polygon_start = mp_geometry_offsets[mp];
  const int polygon_end = mp_geometry_offsets[mp + 1];
  for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
    const int bounds_base = polygon * 4;
    if (
        px < mp_polygon_bounds[bounds_base + 0] ||
        px > mp_polygon_bounds[bounds_base + 2] ||
        py < mp_polygon_bounds[bounds_base + 1] ||
        py > mp_polygon_bounds[bounds_base + 3]
    ) {
      continue;
    }
    const int ring_start = mp_part_offsets[polygon];
    const int ring_end = mp_part_offsets[polygon + 1];
    bool inside = false;
    for (int ring = ring_start; ring < ring_end; ++ring) {
      bool on_boundary = false;
      const int coord_start = mp_ring_offsets[ring];
      const int coord_end = mp_ring_offsets[ring + 1];
      const bool ring_inside = vs_ring_contains_point_with_boundary(
          px, py, mp_x, mp_y, coord_start, coord_end,
          OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
      if (on_boundary) { out_covered[f] = 1; return; }
      if (ring_inside) inside = !inside;
    }
    if (inside) { out_covered[f] = 1; return; }
  }
}

// Row-indirected grouped path for indexed multipolygon logical rows.  The
// logical row supplies the grouping/source semantics while the physical
// family row supplies the coordinate spans.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon_logical_rows(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;
  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;

  for (int logical = logical_start; logical < logical_end; ++logical) {
    const int mp = logical_family_rows[logical];
    if (mp < 0) continue;
    const int polygon_start = mp_geometry_offsets[mp];
    const int polygon_end = mp_geometry_offsets[mp + 1];
    for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
      const int ring_start = mp_part_offsets[polygon];
      const int ring_end = mp_part_offsets[polygon + 1];
      bool inside = false;
      for (int ring = ring_start; ring < ring_end; ++ring) {
        bool on_boundary = false;
        const int coord_start = mp_ring_offsets[ring];
        const int coord_end = mp_ring_offsets[ring + 1];
        const bool ring_inside = vs_ring_contains_point_with_boundary(
            px, py, mp_x, mp_y, coord_start, coord_end,
            OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
        if (on_boundary) { inside = true; break; }
        if (ring_inside) inside = !inside;
      }
      if (inside) { out_covered[f] = 1; return; }
    }
  }
}

// Warp-cooperative row-indirected grouped path for multipolygon logical rows.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon_logical_rows_warp(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = tid >> 5;
  const int lane = tid & 31;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;

  const double px = label_x[f];
  const double py = label_y[f];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[f] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;

  for (int base_logical = logical_start; base_logical < logical_end; base_logical += 32) {
    const int logical = base_logical + lane;
    int hit = 0;
    if (logical < logical_end) {
      int active = 1;
      const int mp = active ? logical_family_rows[logical] : -1;
      if (mp < 0) {
        active = 0;
      }
      if (active) {
        const int polygon_start = mp_geometry_offsets[mp];
        const int polygon_end = mp_geometry_offsets[mp + 1];
        for (int polygon = polygon_start; polygon < polygon_end && hit == 0; ++polygon) {
          const int ring_start = mp_part_offsets[polygon];
          const int ring_end = mp_part_offsets[polygon + 1];
          bool inside = false;
          for (int ring = ring_start; ring < ring_end; ++ring) {
            bool on_boundary = false;
            const int coord_start = mp_ring_offsets[ring];
            const int coord_end = mp_ring_offsets[ring + 1];
            const bool ring_inside = vs_ring_contains_point_with_boundary(
                px, py, mp_x, mp_y, coord_start, coord_end,
                OVERLAY_BOUNDARY_TOLERANCE, &on_boundary);
            if (on_boundary) {
              inside = true;
              break;
            }
            if (ring_inside) inside = !inside;
          }
          if (inside) {
            hit = 1;
          }
        }
      }
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFu, hit != 0);
    if (mask) {
      if (lane == 0) out_covered[f] = 1;
      return;
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon_logical_rows_block(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const int* __restrict__ face_source_rows,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    const int* __restrict__ logical_family_rows,
    const int* __restrict__ logical_source_rows,
    int logical_row_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int face = (int)blockIdx.x;
  if (face >= face_count || out_covered[face] == 1) return;
  const double px = label_x[face];
  const double py = label_y[face];
  const bool restrict_row = face_source_rows != nullptr && logical_source_rows != nullptr;
  const int face_row = restrict_row ? face_source_rows[face] : -1;
  const int logical_start = restrict_row
      ? overlay_lower_bound_i32(logical_source_rows, logical_row_count, face_row)
      : 0;
  const int logical_end = restrict_row
      ? overlay_upper_bound_i32(logical_source_rows, logical_row_count, face_row)
      : logical_row_count;
  __shared__ int warp_crossings[8];
  __shared__ int warp_boundary[8];

  for (int logical = logical_start; logical < logical_end; ++logical) {
    const int multipolygon = logical_family_rows[logical];
    if (multipolygon < 0) continue;
    const int polygon_start = mp_geometry_offsets[multipolygon];
    const int polygon_end = mp_geometry_offsets[multipolygon + 1];
    for (int polygon = polygon_start; polygon < polygon_end; ++polygon) {
      if (overlay_block_polygon_contains_point(
              px, py, mp_x, mp_y, mp_part_offsets, mp_ring_offsets,
              nullptr, polygon, warp_crossings, warp_boundary)) {
        if (threadIdx.x == 0) out_covered[face] = 1;
        return;
      }
    }
  }
}
"""
)

_OVERLAY_FACE_LABEL_KERNEL_NAMES = (
    "label_face_coverage_polygon",
    "label_face_coverage_polygon_warp",
    "label_face_coverage_polygon_block",
    "label_face_coverage_polygon_same_row",
    "label_face_coverage_polygon_logical_rows",
    "label_face_coverage_polygon_logical_rows_warp",
    "label_face_coverage_polygon_logical_rows_block",
    "compute_polygon_bounds",
    "compute_polygon_ring_bounds",
    "compute_multipolygon_polygon_bounds",
    "label_face_coverage_multipolygon",
    "label_face_coverage_multipolygon_block",
    "label_face_coverage_multipolygon_same_row",
    "label_face_coverage_multipolygon_logical_rows",
    "label_face_coverage_multipolygon_logical_rows_warp",
    "label_face_coverage_multipolygon_logical_rows_block",
)

# ---------------------------------------------------------------------------
# 4. Face assembly (ring reconstruction) kernels
# ---------------------------------------------------------------------------
# Kernels: compute_boundary_edges, compute_boundary_next,
#          scatter_boundary_ring_coordinates, compute_ring_sample_points,
#          assign_holes_to_exteriors, count_sibling_hole_depth

_OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE = (
    POINT_IN_RING_DEVICE
    + r"""
// Scatter one selected-face bit to each edge in the face-membership carrier.
// One block owns one face so no slot-to-face inverse relation is needed.
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_edge_face_selection(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const signed char* __restrict__ face_selected,
    signed char* __restrict__ out_edge_selected,
    int face_count
) {
  const int face = blockIdx.x;
  if (face >= face_count) return;
  const signed char selected = face_selected[face];
  const int start = face_offsets[face];
  const int end = face_offsets[face + 1];
  for (int pos = start + threadIdx.x; pos < end; pos += blockDim.x) {
    out_edge_selected[face_edge_ids[pos]] = selected;
  }
}

// Identify boundary edges: edges whose face is selected while the twin face is
// not. The edge-selection bit is the only inverse relation assembly needs.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_boundary_edges(
    const signed char* __restrict__ edge_selected,
    signed char* __restrict__ out_is_boundary,
    int edge_count
) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edge_count) return;
  if (edge_selected[e] == 0) {
    out_is_boundary[e] = 0;
    return;
  }
  const int twin = e ^ 1;
  out_is_boundary[e] =
      (twin >= edge_count || edge_selected[twin] == 0) ? 1 : 0;
}

// Compute next boundary edge for each boundary edge.
// For boundary edge e: follow next_edge_ids from e, crossing through
// non-boundary edges via twin traversal, until finding next boundary edge.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_boundary_next(
    const int* __restrict__ boundary_edge_ids,
    const bool* __restrict__ boundary_active,
    const int* __restrict__ next_edge_ids,
    const signed char* __restrict__ is_boundary,
    int* __restrict__ out_boundary_next,
    int boundary_count,
    int edge_count,
    int max_steps
) {
  const int boundary_pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (boundary_pos >= boundary_count) return;
  const int e = boundary_edge_ids[boundary_pos];
  if (!boundary_active[boundary_pos]) {
    out_boundary_next[boundary_pos] = e;
    return;
  }

  int current = (int)next_edge_ids[e];
  for (int step = 0; step < max_steps; ++step) {
    if (current < 0 || current >= edge_count) break;
    if (is_boundary[current] != 0) {
      out_boundary_next[boundary_pos] = current;
      return;
    }
    // Cross through twin and follow next
    const int twin = current ^ 1;
    if (twin < 0 || twin >= edge_count) break;
    current = (int)next_edge_ids[twin];
  }
  out_boundary_next[boundary_pos] = e;  // self-loop as fallback
}

// Scatter one compact boundary cycle. Compact ids index boundary_edge_ids and
// boundary_next; full half-edge ids are dereferenced only for coordinates.
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_boundary_ring_coordinates(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ boundary_edge_ids,
    const int* __restrict__ ring_edge_starts,
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const bool* __restrict__ ring_active,
    const int* __restrict__ boundary_next,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int ring_count,
    int boundary_count,
    int src_x_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) return;
  if (!ring_active[ring]) return;
  const int start_compact = ring_edge_starts[ring];
  const int offset = ring_coord_offsets[ring];
  const int edge_total = ring_edge_counts[ring];
  if (start_compact < 0 || start_compact >= boundary_count) return;

  int current = start_compact;
  const int first_edge = boundary_edge_ids[start_compact];
  for (int k = 0; k < edge_total; ++k) {
    if (current < 0 || current >= boundary_count) {
      for (int j = k; j <= edge_total; ++j) {
        out_x[offset + j] = src_x[first_edge];
        out_y[offset + j] = src_y[first_edge];
      }
      return;
    }
    const int edge = boundary_edge_ids[current];
    if (edge < 0 || edge >= src_x_count) return;
    out_x[offset + k] = src_x[edge];
    out_y[offset + k] = src_y[edge];
    current = boundary_next[current];
  }
  out_x[offset + edge_total] = src_x[first_edge];
  out_y[offset + edge_total] = src_y[first_edge];
}

// One block owns one boundary cycle. The cycle membership is sorted but does
// not need traversal order because boundary_next supplies each directed edge's
// successor. Centering at the first edge prevents translated slivers from
// cancelling before exact positive-area admission.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_centered_boundary_ring_areas(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ boundary_edge_ids,
    const int* __restrict__ sorted_compact_ids,
    const int* __restrict__ cycle_starts,
    const int* __restrict__ cycle_ends,
    const bool* __restrict__ ring_active,
    const int* __restrict__ boundary_next,
    double* __restrict__ out_area,
    int ring_count,
    int boundary_count,
    int edge_count
) {
  const int ring = blockIdx.x;
  const int tid = threadIdx.x;
  if (ring >= ring_count) return;

  __shared__ double sh_cross[256];
  double local_cross = 0.0;
  if (ring_active[ring]) {
    const int start = cycle_starts[ring];
    const int end = cycle_ends[ring];
    if (start >= 0 && start < end && end <= boundary_count) {
      const int origin_compact = sorted_compact_ids[start];
      const int origin_edge = boundary_edge_ids[origin_compact];
      const double origin_x = src_x[origin_edge];
      const double origin_y = src_y[origin_edge];
      for (int pos = start + tid; pos < end; pos += blockDim.x) {
        const int compact = sorted_compact_ids[pos];
        if (compact < 0 || compact >= boundary_count) continue;
        const int edge = boundary_edge_ids[compact];
        const int next_edge = boundary_next[compact];
        if (edge < 0 || edge >= edge_count ||
            next_edge < 0 || next_edge >= edge_count) continue;
        const double x0 = src_x[edge] - origin_x;
        const double y0 = src_y[edge] - origin_y;
        const double x1 = src_x[next_edge] - origin_x;
        const double y1 = src_y[next_edge] - origin_y;
        local_cross += x0 * y1 - x1 * y0;
      }
    }
  }
  sh_cross[tid] = local_cross;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) sh_cross[tid] += sh_cross[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out_area[ring] = sh_cross[0] * 0.5;
}

// Compute one host-style sample point per ring. The sample point is taken
// from the first non-degenerate edge midpoint, offset by a tiny inward
// perpendicular scaled by the ring extent. This mirrors the host fallback's
// _face_sample_point helper and avoids using centroids that can lie outside
// concave rings.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_ring_sample_points(
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const bool* __restrict__ ring_active,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    double* __restrict__ out_sample_x,
    double* __restrict__ out_sample_y,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;
  if (!ring_active[r]) {
    out_sample_x[r] = 0.0;
    out_sample_y[r] = 0.0;
    return;
  }

  const int start = ring_coord_offsets[r];
  const int n_edges = ring_edge_counts[r];
  if (n_edges <= 0) {
    out_sample_x[r] = 0.0;
    out_sample_y[r] = 0.0;
    return;
  }

  double min_x = all_x[start];
  double max_x = all_x[start];
  double min_y = all_y[start];
  double max_y = all_y[start];
  for (int i = 1; i < n_edges; ++i) {
    const double x = all_x[start + i];
    const double y = all_y[start + i];
    if (x < min_x) min_x = x;
    if (x > max_x) max_x = x;
    if (y < min_y) min_y = y;
    if (y > max_y) max_y = y;
  }
  double extent = max_x - min_x;
  const double extent_y = max_y - min_y;
  if (extent_y > extent) extent = extent_y;
  if (extent < 1.0) extent = 1.0;
  const double epsilon = extent * 1.0e-6;

  for (int i = 0; i < n_edges; ++i) {
    const int j = (i + 1) % n_edges;
    const double x0 = all_x[start + i];
    const double y0 = all_y[start + i];
    const double x1 = all_x[start + j];
    const double y1 = all_y[start + j];
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double length = sqrt(dx * dx + dy * dy);
    if (length <= 0.0) continue;
    const double midpoint_x = 0.5 * (x0 + x1);
    const double midpoint_y = 0.5 * (y0 + y1);
    out_sample_x[r] = midpoint_x - (dy / length) * epsilon;
    out_sample_y[r] = midpoint_y + (dx / length) * epsilon;
    return;
  }

  out_sample_x[r] = all_x[start];
  out_sample_y[r] = all_y[start];
}

// Test each ring sample point against each exterior ring to determine
// hole-to-exterior assignment. One thread per candidate ring.
extern "C" __global__ void __launch_bounds__(256, 4)
assign_holes_to_exteriors(
    const double* __restrict__ ring_sample_x,
    const double* __restrict__ ring_sample_y,
    const double* __restrict__ ring_area,
    const signed char* __restrict__ is_true_exterior,
    const int* __restrict__ source_rows,
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    const int* __restrict__ exterior_indices,
    const long long* __restrict__ exterior_count_ptr,
    int* __restrict__ out_exterior_id,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;
  if (source_rows[r] < 0) {
    out_exterior_id[r] = -1;
    return;
  }
  const long long exterior_count = exterior_count_ptr[0];
  // Only true exteriors map to themselves. Nested positive-area boundary
  // rings still need containment assignment, matching the host assembler.
  if (is_true_exterior[r] != 0) {
    out_exterior_id[r] = r;
    return;
  }
  // Non-exterior ring: find smallest containing exterior of the same row
  // whose area exceeds |ring area|.
  const double px = ring_sample_x[r];
  const double py = ring_sample_y[r];
  const double abs_ring_area = ring_area[r] < 0.0 ? -ring_area[r] : ring_area[r];
  const int row_r = source_rows[r];
  double best_area = 1e308;
  int best_exterior = -1;
  for (long long ei = 0; ei < exterior_count; ++ei) {
    const int ext = exterior_indices[ei];
    if (source_rows[ext] != row_r) continue;
    const double ext_area = ring_area[ext];
    if (ext_area <= 0.0 || ext_area >= best_area) continue;
    // Exterior must be strictly larger than the candidate ring
    if (ext_area <= abs_ring_area) continue;
    // PIP test against exterior ring coordinates
    const int coord_start = ring_coord_offsets[ext];
    const int coord_end = coord_start + ring_edge_counts[ext] + 1;
    if (vs_ring_contains_point(px, py, all_x, all_y, coord_start, coord_end)) {
      best_area = ext_area;
      best_exterior = ext;
    }
  }
  out_exterior_id[r] = best_exterior;
}

// Assign each exterior ring from a grouped polygon-part stream to the
// smallest containing interior ring in the same group.  The grouped
// complement assembler uses that parent to preserve nested islands when a
// valid MultiPolygon component lies inside another component's hole.
extern "C" __global__ void __launch_bounds__(256, 4)
grouped_complement_hole_metrics(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ interior_ring_ids,
    double* __restrict__ out_abs_area,
    double* __restrict__ out_bounds,
    int interior_count
) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= interior_count) return;
  const int ring = interior_ring_ids[h];
  const int start = ring_offsets[ring];
  const int end = ring_offsets[ring + 1];
  if (end <= start) {
    out_abs_area[h] = 0.0;
    out_bounds[h * 4] = 0.0;
    out_bounds[h * 4 + 1] = 0.0;
    out_bounds[h * 4 + 2] = 0.0;
    out_bounds[h * 4 + 3] = 0.0;
    return;
  }

  const double center_x = x[start];
  const double center_y = y[start];
  double min_x = center_x;
  double min_y = center_y;
  double max_x = center_x;
  double max_y = center_y;
  double area2 = 0.0;
  double correction = 0.0;
  for (int i = start; i + 1 < end; ++i) {
    const double ax = x[i];
    const double ay = y[i];
    const double bx = x[i + 1];
    const double by = y[i + 1];
    if (ax < min_x) min_x = ax;
    if (ay < min_y) min_y = ay;
    if (ax > max_x) max_x = ax;
    if (ay > max_y) max_y = ay;
    const double term =
        (ax - center_x) * (by - center_y)
        - (bx - center_x) * (ay - center_y);
    const double corrected = term - correction;
    const double updated = area2 + corrected;
    correction = (updated - area2) - corrected;
    area2 = updated;
  }
  const double last_x = x[end - 1];
  const double last_y = y[end - 1];
  if (last_x < min_x) min_x = last_x;
  if (last_y < min_y) min_y = last_y;
  if (last_x > max_x) max_x = last_x;
  if (last_y > max_y) max_y = last_y;
  out_abs_area[h] = 0.5 * (area2 < 0.0 ? -area2 : area2);
  out_bounds[h * 4] = min_x;
  out_bounds[h * 4 + 1] = min_y;
  out_bounds[h * 4 + 2] = max_x;
  out_bounds[h * 4 + 3] = max_y;
}

extern "C" __global__ void __launch_bounds__(256, 4)
assign_grouped_complement_exterior_parents(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_exterior_ring_ids,
    const int* __restrict__ part_group_rows,
    const int* __restrict__ interior_ring_ids,
    const int* __restrict__ group_interior_offsets,
    const double* __restrict__ interior_abs_area,
    const double* __restrict__ interior_bounds,
    int* __restrict__ out_parent_interior,
    int part_count
) {
  const int part = blockIdx.x * blockDim.x + threadIdx.x;
  if (part >= part_count) return;
  const int exterior = part_exterior_ring_ids[part];
  const int start = ring_offsets[exterior];
  const int end = ring_offsets[exterior + 1];
  if (end <= start) {
    out_parent_interior[part] = -1;
    return;
  }

  double min_x = x[start];
  double min_y = y[start];
  double max_x = min_x;
  double max_y = min_y;
  for (int i = start + 1; i < end; ++i) {
    const double px = x[i];
    const double py = y[i];
    if (px < min_x) min_x = px;
    if (py < min_y) min_y = py;
    if (px > max_x) max_x = px;
    if (py > max_y) max_y = py;
  }

  const int group = part_group_rows[part];
  const int candidate_start = group_interior_offsets[group];
  const int candidate_end = group_interior_offsets[group + 1];
  const double sample_x = x[start];
  const double sample_y = y[start];
  double best_area = 1.7976931348623157e308;
  int best = -1;
  for (int h = candidate_start; h < candidate_end; ++h) {
    const double* bounds = interior_bounds + h * 4;
    if (bounds[0] > min_x || bounds[1] > min_y
        || bounds[2] < max_x || bounds[3] < max_y) {
      continue;
    }
    const double area = interior_abs_area[h];
    if (!(area > 0.0) || area > best_area) continue;
    const int hole_ring = interior_ring_ids[h];
    const int hole_start = ring_offsets[hole_ring];
    const int hole_end = ring_offsets[hole_ring + 1];
    if (!vs_ring_contains_point(
            sample_x, sample_y, x, y, hole_start, hole_end)) {
      continue;
    }
    if (area < best_area || best < 0 || h < best) {
      best_area = area;
      best = h;
    }
  }
  out_parent_interior[part] = best;
}

// Count nesting depth among sibling holes assigned to the same exterior.
// For each ring r that has been assigned to an exterior (exterior_id[r] >= 0
// and exterior_id[r] != r), count how many other rings sharing the same
// exterior with strictly larger |area| contain r's sample point.
// Even local depth -> direct hole; odd -> nested inside another hole (skip).
extern "C" __global__ void __launch_bounds__(256, 4)
count_sibling_hole_depth(
    const double* __restrict__ sample_x,
    const double* __restrict__ sample_y,
    const double* __restrict__ ring_area,
    const int* __restrict__ exterior_id,
    const signed char* __restrict__ can_be_hole,
    const int* __restrict__ coord_offsets,
    const int* __restrict__ edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    const int* __restrict__ hole_ring_ids,
    const long long* __restrict__ hole_count_ptr,
    int* __restrict__ out_depth,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;

  const int ext_r = exterior_id[r];
  const long long hole_count = hole_count_ptr[0];
  // Not a hole: either unassigned or self-assigned (exterior)
  if (ext_r < 0 || ext_r == r || can_be_hole[r] == 0 || hole_count <= 1) {
    out_depth[r] = 0;
    return;
  }

  const double px = sample_x[r];
  const double py = sample_y[r];
  const double abs_area_r = ring_area[r] < 0.0 ? -ring_area[r] : ring_area[r];
  int depth = 0;

  for (long long pos = 0; pos < hole_count; ++pos) {
    const int c = hole_ring_ids[pos];
    if (c == r) continue;
    if (can_be_hole[c] == 0) continue;
    if (exterior_id[c] != ext_r) continue;  // same exterior
    if (exterior_id[c] == c) continue;       // c is the exterior itself
    // c must have strictly larger |area|
    const double abs_area_c = ring_area[c] < 0.0 ? -ring_area[c] : ring_area[c];
    if (abs_area_c <= abs_area_r) continue;

    // PIP test: does ring c contain (px, py)?
    const int cs = coord_offsets[c];
    const int ce = cs + edge_counts[c] + 1;
    if (vs_ring_contains_point(px, py, all_x, all_y, cs, ce)) depth += 1;
  }
  out_depth[r] = depth;
}

// Ring membership is already known at structural capacity. Scatter holes into
// the scanned span owned by their exterior; no compact row or count is needed.
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_output_holes(
    const unsigned char* __restrict__ valid_hole,
    const int* __restrict__ hole_polygon_lanes,
    const long long* __restrict__ polygon_ring_offsets,
    int* __restrict__ polygon_hole_counters,
    int* __restrict__ output_ring_ids,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count || valid_hole[ring] == 0u) return;
  const int polygon = hole_polygon_lanes[ring];
  if (polygon < 0) return;
  const int local = atomicAdd(polygon_hole_counters + polygon, 1);
  output_ring_ids[polygon_ring_offsets[polygon] + 1LL + local] = ring;
}
"""
)

_OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES = (
    "scatter_edge_face_selection",
    "compute_boundary_edges",
    "compute_boundary_next",
    "scatter_boundary_ring_coordinates",
    "compute_centered_boundary_ring_areas",
    "compute_ring_sample_points",
    "assign_holes_to_exteriors",
    "grouped_complement_hole_metrics",
    "assign_grouped_complement_exterior_parents",
    "count_sibling_hole_depth",
    "scatter_output_holes",
)

# ---------------------------------------------------------------------------
# 5. Batch point-in-ring kernel
# ---------------------------------------------------------------------------
# Kernel: batch_point_in_ring

_BATCH_POINT_IN_RING_KERNEL_SOURCE = (
    POINT_IN_RING_DEVICE
    + r"""
extern "C" __global__ void __launch_bounds__(256, 4)
batch_point_in_ring(
    const double* __restrict__ sample_x,
    const double* __restrict__ sample_y,
    const double* __restrict__ ring_x,
    const double* __restrict__ ring_y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ pair_ring_idx,
    int* __restrict__ results,
    int pair_count
) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;

    const double px = sample_x[pair];
    const double py = sample_y[pair];
    const int ring = pair_ring_idx[pair];
    const int cs = ring_offsets[ring];
    const int ce = ring_offsets[ring + 1];

    results[pair] = vs_ring_contains_point(px, py, ring_x, ring_y, cs, ce) ? 1 : 0;
}
"""
)

_BATCH_POINT_IN_RING_KERNEL_NAMES = ("batch_point_in_ring",)

# ---------------------------------------------------------------------------
# 6. Containment bypass kernels
# ---------------------------------------------------------------------------
# Kernels: containment_poly_vs_poly, containment_poly_vs_mpoly,
#          containment_mpoly_vs_poly, containment_mpoly_vs_mpoly

_CONTAINMENT_BYPASS_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + SEGMENT_CROSSING_DEVICE
    + POINT_IN_RING_DEVICE
    + r"""
// Device helper: polygon containment via even-odd rule across all rings.
extern "C" __device__ inline bool _cb_polygon_contains(
    double px, double py,
    const double* __restrict__ rx, const double* __restrict__ ry,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int polygon_row
) {
    const int ring_start = corr_geom_offsets[polygon_row];
    const int ring_end = corr_geom_offsets[polygon_row + 1];
    bool inside = false;
    for (int ring = ring_start; ring < ring_end; ++ring) {
        const int cs = corr_ring_offsets[ring];
        const int ce = corr_ring_offsets[ring + 1];
        bool ring_inside = vs_ring_contains_point(px, py, rx, ry, cs, ce);
        if (ring_inside) {
            inside = !inside;
        }
    }
    return inside;
}

// Device helper: multipolygon containment (any part polygon contains).
extern "C" __device__ inline bool _cb_multipolygon_contains(
    double px, double py,
    const double* __restrict__ rx, const double* __restrict__ ry,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_part_offsets,
    const int* __restrict__ corr_ring_offsets,
    int multipolygon_row
) {
    const int poly_start = corr_geom_offsets[multipolygon_row];
    const int poly_end = corr_geom_offsets[multipolygon_row + 1];
    for (int polygon = poly_start; polygon < poly_end; ++polygon) {
        const int ring_start = corr_part_offsets[polygon];
        const int ring_end = corr_part_offsets[polygon + 1];
        bool inside = false;
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int cs = corr_ring_offsets[ring];
            const int ce = corr_ring_offsets[ring + 1];
            bool ring_inside = vs_ring_contains_point(px, py, rx, ry, cs, ce);
            if (ring_inside) {
                inside = !inside;
            }
        }
        if (inside) return true;
    }
    return false;
}

extern "C" __device__ inline bool _cb_ring_has_proper_crossing(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    int left_start,
    int left_end,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    int corr_start,
    int corr_end
) {
    for (int li = left_start; li + 1 < left_end; ++li) {
        const double p1x = left_x[li];
        const double p1y = left_y[li];
        const double p2x = left_x[li + 1];
        const double p2y = left_y[li + 1];
        const double left_min_x = p1x < p2x ? p1x : p2x;
        const double left_max_x = p1x > p2x ? p1x : p2x;
        const double left_min_y = p1y < p2y ? p1y : p2y;
        const double left_max_y = p1y > p2y ? p1y : p2y;
        for (int ri = corr_start; ri + 1 < corr_end; ++ri) {
            const double q1x = corr_x[ri];
            const double q1y = corr_y[ri];
            const double q2x = corr_x[ri + 1];
            const double q2y = corr_y[ri + 1];
            const double corr_min_x = q1x < q2x ? q1x : q2x;
            const double corr_max_x = q1x > q2x ? q1x : q2x;
            const double corr_min_y = q1y < q2y ? q1y : q2y;
            const double corr_max_y = q1y > q2y ? q1y : q2y;
            if (left_max_x < corr_min_x || corr_max_x < left_min_x
                    || left_max_y < corr_min_y || corr_max_y < left_min_y) {
                continue;
            }
            if (vs_segments_properly_cross(
                    p1x, p1y, p2x, p2y,
                    q1x, q1y, q2x, q2y)) {
                return true;
            }
        }
    }
    return false;
}

extern "C" __device__ inline bool _cb_polygon_crosses_polygon_boundary(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_ring_offsets,
    int left_row,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row
) {
    const int left_ring_start = left_geom_offsets[left_row];
    const int left_ring_end = left_geom_offsets[left_row + 1];
    const int corr_ring_start = corr_geom_offsets[corr_row];
    const int corr_ring_end = corr_geom_offsets[corr_row + 1];
    for (int lring = left_ring_start; lring < left_ring_end; ++lring) {
        const int left_start = left_ring_offsets[lring];
        const int left_end = left_ring_offsets[lring + 1];
        for (int cring = corr_ring_start; cring < corr_ring_end; ++cring) {
            const int corr_start = corr_ring_offsets[cring];
            const int corr_end = corr_ring_offsets[cring + 1];
            if (_cb_ring_has_proper_crossing(
                    left_x, left_y, left_start, left_end,
                    corr_x, corr_y, corr_start, corr_end)) {
                return true;
            }
        }
    }
    return false;
}

extern "C" __device__ inline bool _cb_polygon_crosses_multipolygon_boundary(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_part_offsets,
    const int* __restrict__ left_ring_offsets,
    int left_row,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_part_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    bool left_is_multi
) {
    const int left_part_start = left_geom_offsets[left_row];
    const int left_part_end = left_geom_offsets[left_row + 1];
    const int corr_poly_start = corr_geom_offsets[corr_row];
    const int corr_poly_end = corr_geom_offsets[corr_row + 1];
    for (int lpart = left_part_start; lpart < left_part_end; ++lpart) {
        const int left_ring_start = left_is_multi ? left_part_offsets[lpart] : lpart;
        const int left_ring_end = left_is_multi ? left_part_offsets[lpart + 1] : (lpart + 1);
        for (int lring = left_ring_start; lring < left_ring_end; ++lring) {
            const int left_start = left_ring_offsets[lring];
            const int left_end = left_ring_offsets[lring + 1];
            for (int cpoly = corr_poly_start; cpoly < corr_poly_end; ++cpoly) {
                const int corr_ring_start = corr_part_offsets[cpoly];
                const int corr_ring_end = corr_part_offsets[cpoly + 1];
                for (int cring = corr_ring_start; cring < corr_ring_end; ++cring) {
                    const int corr_start = corr_ring_offsets[cring];
                    const int corr_end = corr_ring_offsets[cring + 1];
                    if (_cb_ring_has_proper_crossing(
                            left_x, left_y, left_start, left_end,
                            corr_x, corr_y, corr_start, corr_end)) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

extern "C" __device__ inline bool _cb_multipolygon_crosses_polygon_boundary(
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_part_offsets,
    const int* __restrict__ left_ring_offsets,
    int left_row,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row
) {
    const int left_part_start = left_geom_offsets[left_row];
    const int left_part_end = left_geom_offsets[left_row + 1];
    const int corr_ring_start = corr_geom_offsets[corr_row];
    const int corr_ring_end = corr_geom_offsets[corr_row + 1];
    for (int lpart = left_part_start; lpart < left_part_end; ++lpart) {
        const int left_ring_start = left_part_offsets[lpart];
        const int left_ring_end = left_part_offsets[lpart + 1];
        for (int lring = left_ring_start; lring < left_ring_end; ++lring) {
            const int left_start = left_ring_offsets[lring];
            const int left_end = left_ring_offsets[lring + 1];
            for (int cring = corr_ring_start; cring < corr_ring_end; ++cring) {
                const int corr_start = corr_ring_offsets[cring];
                const int corr_end = corr_ring_offsets[cring + 1];
                if (_cb_ring_has_proper_crossing(
                        left_x, left_y, left_start, left_end,
                        corr_x, corr_y, corr_start, corr_end)) {
                    return true;
                }
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------
// Thread-per-polygon kernel: test one sample point per candidate ring
// against the corridor, then reject any proper boundary crossings.
// If a simple ring has one point inside the corridor and its boundary
// does not cross the corridor boundary, the whole ring is inside.
// Reads directly from source family coordinate buffers -- no scatter.
// -----------------------------------------------------------------

// Row bbox vs. polygon corridor boundary.  Rows whose bbox cannot touch the
// corridor boundary cannot have a segment crossing; containment kernels can
// skip the expensive left-edge x corridor-edge loop for those rows.
extern "C" __global__ void
containment_boundary_overlap_poly(
    const double* __restrict__ left_bounds,
    int n_rows,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    int* __restrict__ out
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const double lminx = left_bounds[row * 4 + 0];
    const double lminy = left_bounds[row * 4 + 1];
    const double lmaxx = left_bounds[row * 4 + 2];
    const double lmaxy = left_bounds[row * 4 + 3];
    const int ring_start = corr_geom_offsets[corr_row];
    const int ring_end = corr_geom_offsets[corr_row + 1];
    for (int ring = ring_start; ring < ring_end; ++ring) {
        const int cs = corr_ring_offsets[ring];
        const int ce = corr_ring_offsets[ring + 1];
        for (int ri = cs; ri + 1 < ce; ++ri) {
            const double q1x = corr_x[ri];
            const double q1y = corr_y[ri];
            const double q2x = corr_x[ri + 1];
            const double q2y = corr_y[ri + 1];
            const double sminx = q1x < q2x ? q1x : q2x;
            const double smaxx = q1x > q2x ? q1x : q2x;
            const double sminy = q1y < q2y ? q1y : q2y;
            const double smaxy = q1y > q2y ? q1y : q2y;
            if (!(smaxx < lminx || sminx > lmaxx || smaxy < lminy || sminy > lmaxy)) {
                out[row] = 1;
                return;
            }
        }
    }
    out[row] = 0;
}

// Row bbox vs. multipolygon corridor boundary.
extern "C" __global__ void
containment_boundary_overlap_mpoly(
    const double* __restrict__ left_bounds,
    int n_rows,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_part_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    int* __restrict__ out
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const double lminx = left_bounds[row * 4 + 0];
    const double lminy = left_bounds[row * 4 + 1];
    const double lmaxx = left_bounds[row * 4 + 2];
    const double lmaxy = left_bounds[row * 4 + 3];
    const int poly_start = corr_geom_offsets[corr_row];
    const int poly_end = corr_geom_offsets[corr_row + 1];
    for (int polygon = poly_start; polygon < poly_end; ++polygon) {
        const int ring_start = corr_part_offsets[polygon];
        const int ring_end = corr_part_offsets[polygon + 1];
        for (int ring = ring_start; ring < ring_end; ++ring) {
            const int cs = corr_ring_offsets[ring];
            const int ce = corr_ring_offsets[ring + 1];
            for (int ri = cs; ri + 1 < ce; ++ri) {
                const double q1x = corr_x[ri];
                const double q1y = corr_y[ri];
                const double q2x = corr_x[ri + 1];
                const double q2y = corr_y[ri + 1];
                const double sminx = q1x < q2x ? q1x : q2x;
                const double smaxx = q1x > q2x ? q1x : q2x;
                const double sminy = q1y < q2y ? q1y : q2y;
                const double smaxy = q1y > q2y ? q1y : q2y;
                if (!(smaxx < lminx || sminx > lmaxx || smaxy < lminy || sminy > lmaxy)) {
                    out[row] = 1;
                    return;
                }
            }
        }
    }
    out[row] = 0;
}

// Polygon candidates vs. polygon corridor.
extern "C" __global__ void
containment_poly_vs_poly(
    const int* __restrict__ cand_family_rows,
    int n_candidates,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_ring_offsets,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    const int* __restrict__ candidate_boundary_overlap,
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_ring = left_geom_offsets[frow];
    const int last_ring = left_geom_offsets[frow + 1];
    for (int lring = first_ring; lring < last_ring; ++lring) {
        const int sample = left_ring_offsets[lring];
        if (!_cb_polygon_contains(
                left_x[sample], left_y[sample],
                corr_x, corr_y,
                corr_geom_offsets, corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
    }
    if (candidate_boundary_overlap[cid] != 0 && _cb_polygon_crosses_polygon_boundary(
            left_x, left_y,
            left_geom_offsets, left_ring_offsets, frow,
            corr_x, corr_y,
            corr_geom_offsets, corr_ring_offsets, corr_row)) {
        out[cid] = 0;
        return;
    }
    out[cid] = 1;
}

// Polygon candidates vs. polygon corridor, cooperative sample PIP.
// One block processes one left polygon row.  This keeps exact containment
// semantics for single-ring left polygons while giving complex right-side
// corridors enough parallelism for the sample point-in-polygon pass.
// Multi-ring left polygons return 2 so the caller can route them through the
// conservative exact remainder path instead of admitting them as contained.
extern "C" __global__ void
containment_poly_vs_poly_single_ring_block(
    const int* __restrict__ cand_family_rows,
    int n_candidates,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_ring_offsets,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    const int* __restrict__ candidate_boundary_overlap,
    int* __restrict__ out
) {
    const int cid = blockIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_ring = left_geom_offsets[frow];
    const int last_ring = left_geom_offsets[frow + 1];
    if (last_ring - first_ring != 1) {
        if (threadIdx.x == 0) out[cid] = 2;
        return;
    }

    const int sample = left_ring_offsets[first_ring];
    const double px = left_x[sample];
    const double py = left_y[sample];
    const int corr_ring_start = corr_geom_offsets[corr_row];
    const int corr_ring_end = corr_geom_offsets[corr_row + 1];

    __shared__ int block_crossings;
    __shared__ int warp_crossings[8];
    if (threadIdx.x == 0) {
        block_crossings = 0;
    }
    __syncthreads();

    for (int ring = corr_ring_start; ring < corr_ring_end; ++ring) {
        const int cs = corr_ring_offsets[ring];
        const int ce = corr_ring_offsets[ring + 1];
        const int edge_count = ce - cs - 1;
        int my_crossings = 0;
        for (int e = (int)threadIdx.x; e < edge_count; e += (int)blockDim.x) {
            const int c = cs + 1 + e;
            const double ax = corr_x[c - 1];
            const double ay = corr_y[c - 1];
            const double bx = corr_x[c];
            const double by = corr_y[c];
            const bool intersects = ((ay > py) != (by > py)) &&
                (px <= (((bx - ax) * (py - ay)) / ((by - ay) + 0.0)) + ax);
            if (intersects) {
                my_crossings ^= 1;
            }
        }

        const unsigned int FULL_MASK = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            my_crossings ^= __shfl_xor_sync(FULL_MASK, my_crossings, offset);
        }

        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            warp_crossings[warp_id] = my_crossings;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int ring_crossings = 0;
            const int num_warps = ((int)blockDim.x + 31) / 32;
            for (int w = 0; w < num_warps; ++w) {
                ring_crossings ^= warp_crossings[w];
            }
            if (ring_crossings) {
                block_crossings ^= 1;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (!block_crossings) {
            out[cid] = 0;
            return;
        }
        if (candidate_boundary_overlap[cid] != 0 && _cb_polygon_crosses_polygon_boundary(
                left_x, left_y,
                left_geom_offsets, left_ring_offsets, frow,
                corr_x, corr_y,
                corr_geom_offsets, corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
        out[cid] = 1;
    }
}

// Polygon candidates vs. multipolygon corridor.
extern "C" __global__ void
containment_poly_vs_mpoly(
    const int* __restrict__ cand_family_rows,
    int n_candidates,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_ring_offsets,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_part_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    const int* __restrict__ candidate_boundary_overlap,
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_ring = left_geom_offsets[frow];
    const int last_ring = left_geom_offsets[frow + 1];
    for (int lring = first_ring; lring < last_ring; ++lring) {
        const int sample = left_ring_offsets[lring];
        if (!_cb_multipolygon_contains(
                left_x[sample], left_y[sample],
                corr_x, corr_y,
                corr_geom_offsets, corr_part_offsets,
                corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
    }
    if (candidate_boundary_overlap[cid] != 0 && _cb_polygon_crosses_multipolygon_boundary(
            left_x, left_y,
            left_geom_offsets, left_geom_offsets,
            left_ring_offsets, frow,
            corr_x, corr_y,
            corr_geom_offsets, corr_part_offsets, corr_ring_offsets, corr_row,
            false)) {
        out[cid] = 0;
        return;
    }
    out[cid] = 1;
}

// MultiPolygon candidates vs. polygon corridor.
extern "C" __global__ void
containment_mpoly_vs_poly(
    const int* __restrict__ cand_family_rows,
    int n_candidates,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_part_offsets,
    const int* __restrict__ left_ring_offsets,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    const int* __restrict__ candidate_boundary_overlap,
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_part = left_geom_offsets[frow];
    const int last_part = left_geom_offsets[frow + 1];
    for (int part = first_part; part < last_part; ++part) {
        const int first_ring = left_part_offsets[part];
        const int last_ring = left_part_offsets[part + 1];
        for (int lring = first_ring; lring < last_ring; ++lring) {
            const int sample = left_ring_offsets[lring];
            if (!_cb_polygon_contains(
                    left_x[sample], left_y[sample],
                    corr_x, corr_y,
                    corr_geom_offsets, corr_ring_offsets, corr_row)) {
                out[cid] = 0;
                return;
            }
        }
    }
    if (candidate_boundary_overlap[cid] != 0 && _cb_multipolygon_crosses_polygon_boundary(
            left_x, left_y,
            left_geom_offsets, left_part_offsets,
            left_ring_offsets, frow,
            corr_x, corr_y,
            corr_geom_offsets, corr_ring_offsets, corr_row)) {
        out[cid] = 0;
        return;
    }
    out[cid] = 1;
}

// MultiPolygon candidates vs. multipolygon corridor.
extern "C" __global__ void
containment_mpoly_vs_mpoly(
    const int* __restrict__ cand_family_rows,
    int n_candidates,
    const double* __restrict__ left_x,
    const double* __restrict__ left_y,
    const int* __restrict__ left_geom_offsets,
    const int* __restrict__ left_part_offsets,
    const int* __restrict__ left_ring_offsets,
    const double* __restrict__ corr_x,
    const double* __restrict__ corr_y,
    const int* __restrict__ corr_geom_offsets,
    const int* __restrict__ corr_part_offsets,
    const int* __restrict__ corr_ring_offsets,
    int corr_row,
    const int* __restrict__ candidate_boundary_overlap,
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_part = left_geom_offsets[frow];
    const int last_part = left_geom_offsets[frow + 1];
    for (int part = first_part; part < last_part; ++part) {
        const int first_ring = left_part_offsets[part];
        const int last_ring = left_part_offsets[part + 1];
        for (int lring = first_ring; lring < last_ring; ++lring) {
            const int sample = left_ring_offsets[lring];
            if (!_cb_multipolygon_contains(
                    left_x[sample], left_y[sample],
                    corr_x, corr_y,
                    corr_geom_offsets, corr_part_offsets,
                    corr_ring_offsets, corr_row)) {
                out[cid] = 0;
                return;
            }
        }
    }
    if (candidate_boundary_overlap[cid] != 0 && _cb_polygon_crosses_multipolygon_boundary(
            left_x, left_y,
            left_geom_offsets, left_part_offsets,
            left_ring_offsets, frow,
            corr_x, corr_y,
            corr_geom_offsets, corr_part_offsets, corr_ring_offsets, corr_row,
            true)) {
        out[cid] = 0;
        return;
    }
    out[cid] = 1;
}
"""
)

_CONTAINMENT_BYPASS_KERNEL_NAMES = (
    "containment_boundary_overlap_poly",
    "containment_boundary_overlap_mpoly",
    "containment_poly_vs_poly",
    "containment_poly_vs_poly_single_ring_block",
    "containment_poly_vs_mpoly",
    "containment_mpoly_vs_poly",
    "containment_mpoly_vs_mpoly",
)
