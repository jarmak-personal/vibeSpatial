"""NVRTC kernel sources for overlay/gpu.py.

This module holds the CUDA C++ source strings and kernel name tuples for
the five NVRTC compilation groups used by the GPU overlay pipeline.  All
Python dispatch logic, CCCL primitive calls, warmup registration, and
compile_kernel_group wrappers remain in gpu.py.
"""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.point_in_ring import POINT_IN_RING_DEVICE
from vibespatial.cuda.device_functions.segment_crossing import SEGMENT_CROSSING_DEVICE
from vibespatial.cuda.preamble import SPATIAL_TOLERANCE_PREAMBLE

# ---------------------------------------------------------------------------
# 1. Split event emission kernels
# ---------------------------------------------------------------------------
# Kernels: emit_endpoint_split_events, count_pair_split_events,
#          scatter_pair_split_events, emit_atomic_edges

_OVERLAY_SPLIT_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + r"""
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
  out_counts[row] = (kind == 1 || kind == 2 || kind == 4) ? 2 : (kind == 3) ? 4 : 0;
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
    signed char* __restrict__ out_priority,
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

  if (kind == 1 || kind == 2 || kind == 4) {
    double x = point_x[row];
    double y = point_y[row];
    const double left_t = project_t(x, y, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double right_t = project_t(x, y, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);
    out_source_segment_ids[base + 0] = left_index;
    out_t[base + 0] = left_t;
    out_x[base + 0] = x;
    out_y[base + 0] = y;
    out_priority[base + 0] = kind == 1 ? 3 : (kind == 2 ? 2 : 1);
    out_source_segment_ids[base + 1] = right_source_id;
    out_t[base + 1] = right_t;
    out_x[base + 1] = x;
    out_y[base + 1] = y;
    out_priority[base + 1] = kind == 1 ? 3 : (kind == 2 ? 2 : 1);
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
    out_priority[base + 0] = 3;
    out_source_segment_ids[base + 1] = left_index;
    out_t[base + 1] = left_t1;
    out_x[base + 1] = x1;
    out_y[base + 1] = y1;
    out_priority[base + 1] = 3;
    out_source_segment_ids[base + 2] = right_source_id;
    out_t[base + 2] = right_t0;
    out_x[base + 2] = x0;
    out_y[base + 2] = y0;
    out_priority[base + 2] = 3;
    out_source_segment_ids[base + 3] = right_source_id;
    out_t[base + 3] = right_t1;
    out_x[base + 3] = x1;
    out_y[base + 3] = y1;
    out_priority[base + 3] = 3;
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

extern "C" __global__ void __launch_bounds__(256, 4)
derive_source_ring_transition_signs(
    const double* __restrict__ source_x0,
    const double* __restrict__ source_y0,
    const double* __restrict__ source_x1,
    const double* __restrict__ source_y1,
    const int* __restrict__ ring_starts,
    const int* __restrict__ ring_ends,
    const int* __restrict__ ring_local_ids,
    int* __restrict__ out_transition,
    int ring_count
) {
  const int ring = blockIdx.x * blockDim.x + threadIdx.x;
  if (ring >= ring_count) return;
  const int start = ring_starts[ring];
  const int end = ring_ends[ring];
  if (end <= start || source_x0[start] != source_x1[end - 1] ||
      source_y0[start] != source_y1[end - 1]) {
    for (int segment = start; segment < end; ++segment) {
      out_transition[segment] = 0;
    }
    return;
  }
  /* The extreme-vertex turn determines the source ring winding.  Repeated
     minima and zero-length source segments are legal input artifacts, so a
     single adjacent triple is not sufficient.  Search distinct predecessor
     and successor coordinates and choose the lexicographically first exact
     non-collinear turn. */
  int representative = -1;
  int orientation = 0;
  const int segment_count = end - start;
  for (int candidate = start; candidate < end; ++candidate) {
    const double cx = source_x0[candidate];
    const double cy = source_y0[candidate];
    int immediate_prior = candidate - 1;
    if (immediate_prior < start) immediate_prior = end - 1;
    if (source_x0[immediate_prior] == cx &&
        source_y0[immediate_prior] == cy) {
      continue;
    }
    int previous = -1;
    int next = -1;
    for (int step = 1; step <= segment_count; ++step) {
      int prior = candidate - step;
      if (prior < start) prior += segment_count;
      if (source_x0[prior] != cx || source_y0[prior] != cy) {
        previous = prior;
        break;
      }
    }
    for (int step = 0; step < segment_count; ++step) {
      int following = candidate + step;
      if (following >= end) following -= segment_count;
      if (source_x1[following] != cx || source_y1[following] != cy) {
        next = following;
        break;
      }
    }
    if (previous < 0 || next < 0) continue;
    const int candidate_orientation = vs_orient2d(
        source_x0[previous], source_y0[previous],
        cx, cy,
        source_x1[next], source_y1[next]
    );
    if (candidate_orientation == 0) continue;
    if (representative < 0 || cx < source_x0[representative] ||
        (cx == source_x0[representative] &&
         (cy < source_y0[representative] ||
          (cy == source_y0[representative] && candidate < representative)))) {
      representative = candidate;
      orientation = candidate_orientation;
    }
  }
  const int role = ring_local_ids[start] == 0 ? 1 : -1;
  const int transition = orientation > 0 ? role :
      (orientation < 0 ? -role : 0);
  for (int segment = start; segment < end; ++segment) {
    out_transition[segment] = transition;
  }
}
"""
)

_OVERLAY_SPLIT_KERNEL_NAMES = (
    "emit_endpoint_split_events",
    "count_pair_split_events",
    "scatter_pair_split_events",
    "rank_exact_split_event_merge",
    "emit_atomic_edges",
    "derive_source_ring_transition_signs",
)

# ---------------------------------------------------------------------------
# 2. Half-edge face traversal kernels
# ---------------------------------------------------------------------------
# Kernels: face metrics, edge-to-face identity, dual propagation, containment

_OVERLAY_FACE_WALK_KERNEL_SOURCE = (
    SPATIAL_TOLERANCE_PREAMBLE
    + ORIENT2D_DEVICE
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

__device__ __forceinline__ int robust_polar_half(double dx, double dy) {
  return (dy > 0.0 || (dy == 0.0 && dx >= 0.0)) ? 0 : 1;
}

__device__ __forceinline__ bool robust_polar_less(
    int left_edge,
    int right_edge,
    const double* __restrict__ tangent_x,
    const double* __restrict__ tangent_y
) {
  const double left_dx = tangent_x[left_edge];
  const double left_dy = tangent_y[left_edge];
  const double right_dx = tangent_x[right_edge];
  const double right_dy = tangent_y[right_edge];
  const int left_half = robust_polar_half(left_dx, left_dy);
  const int right_half = robust_polar_half(right_dx, right_dy);
  if (left_half != right_half) return left_half < right_half;
  const int orientation = vs_orient2d(
      0.0, 0.0, left_dx, left_dy, right_dx, right_dy
  );
  if (orientation != 0) return orientation > 0;
  return left_edge < right_edge;
}

extern "C" __global__ void __launch_bounds__(256, 4)
scatter_node_offsets(
    const int* __restrict__ group_ends,
    const int* __restrict__ group_ids,
    int* __restrict__ node_offsets,
    int* __restrict__ out_node_count,
    int edge_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= edge_count) return;
  if (pos == 0) node_offsets[0] = 0;
  if (group_ends[pos] != 0 || pos + 1 == edge_count) {
    node_offsets[group_ids[pos] + 1] = pos + 1;
  }
  if (pos + 1 == edge_count) out_node_count[0] = group_ids[pos] + 1;
}

__device__ __forceinline__ void enqueue_degree_two_core_node(
    int node,
    int* __restrict__ queued,
    int* __restrict__ queue,
    int* __restrict__ queue_tail,
    int* __restrict__ queue_ready,
    int* __restrict__ pending
) {
  if (atomicCAS(queued + node, 0, 1) != 0) return;
  atomicAdd(pending, 1);
  const int slot = atomicAdd(queue_tail, 1);
  queue[slot] = node;
  __threadfence();
  atomicExch(queue_ready + slot, 1);
}

// Seed every exact endpoint whose active degree is below two. Degrees and the
// queue use endpoint CSR capacity; no host-visible node count is required.
extern "C" __global__ void __launch_bounds__(256, 4)
initialize_degree_two_core_frontier(
    const int* __restrict__ node_offsets,
    const int* __restrict__ node_count,
    int* __restrict__ degree,
    int* __restrict__ queued,
    int* __restrict__ queue,
    int* __restrict__ queue_tail,
    int* __restrict__ queue_ready,
    int* __restrict__ pending,
    int node_capacity
) {
  const int node = blockIdx.x * blockDim.x + threadIdx.x;
  if (node >= node_capacity || node >= node_count[0]) return;
  const int node_degree = node_offsets[node + 1] - node_offsets[node];
  degree[node] = node_degree;
  if (node_degree < 2) {
    enqueue_degree_two_core_node(
        node, queued, queue, queue_tail, queue_ready, pending);
  }
}

// Persistent CAS-protected leaf-chain peel. Each node enters the frontier at
// most once and each incidence is visited at most once, so total work is O(E).
// Segment ownership is claimed atomically when both leaf ends race.
extern "C" __global__ void __launch_bounds__(256, 4)
peel_degree_two_core_frontier(
    const int* __restrict__ incidence_edge_ids,
    const int* __restrict__ node_offsets,
    const int* __restrict__ src_node_ids,
    const int* __restrict__ node_count,
    int* __restrict__ active_segments,
    int* __restrict__ degree,
    int* __restrict__ queued,
    int* __restrict__ queue,
    int* __restrict__ queue_head,
    int* __restrict__ queue_tail,
    int* __restrict__ queue_ready,
    int* __restrict__ pending,
    int node_capacity,
    int segment_count
) {
  while (atomicAdd(pending, 0) != 0) {
    const int head = atomicAdd(queue_head, 0);
    const int tail = atomicAdd(queue_tail, 0);
    if (head >= tail || atomicCAS(queue_head, head, head + 1) != head) {
      __nanosleep(64);
      continue;
    }
    while (atomicAdd(queue_ready + head, 0) == 0) __nanosleep(32);
    __threadfence();
    const int node = queue[head];
    if (node >= 0 && node < node_count[0] && node < node_capacity) {
      const int start = node_offsets[node];
      const int end = node_offsets[node + 1];
      for (int pos = start; pos < end; ++pos) {
        const int edge = incidence_edge_ids[pos];
        const int segment = edge >> 1;
        if (segment < 0 || segment >= segment_count ||
            atomicCAS(active_segments + segment, 1, 0) != 1) {
          continue;
        }
        atomicSub(degree + node, 1);
        const int other = src_node_ids[edge ^ 1];
        const int previous_degree = atomicSub(degree + other, 1);
        if (previous_degree == 2) {
          enqueue_degree_two_core_node(
              other, queued, queue, queue_tail, queue_ready, pending);
        }
      }
    }
    atomicSub(pending, 1);
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
merge_node_edges_robust_pass(
    const double* __restrict__ tangent_x,
    const double* __restrict__ tangent_y,
    const int* __restrict__ input_edge_ids,
    int* __restrict__ output_edge_ids,
    const int* __restrict__ group_ids,
    const int* __restrict__ node_offsets,
    int merge_width,
    int edge_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= edge_count) return;
  const int group_id = group_ids[pos];
  const int node_start = node_offsets[group_id];
  const int node_end = node_offsets[group_id + 1];
  const int local = pos - node_start;
  const int pair_width = merge_width * 2;
  const int pair_start = node_start + (local / pair_width) * pair_width;
  const int left_start = pair_start;
  const int left_end = min(left_start + merge_width, node_end);
  const int right_start = left_end;
  const int right_end = min(right_start + merge_width, node_end);
  if (right_start >= right_end) {
    output_edge_ids[pos] = input_edge_ids[pos];
    return;
  }

  const int left_count = left_end - left_start;
  const int right_count = right_end - right_start;
  const int diagonal = pos - pair_start;
  int low = max(0, diagonal - right_count);
  int high = min(diagonal, left_count);
  while (low < high) {
    const int left_index = (low + high) >> 1;
    const int right_index = diagonal - left_index;
    if (
        left_index < left_count && right_index > 0 &&
        robust_polar_less(
            input_edge_ids[left_start + left_index],
            input_edge_ids[right_start + right_index - 1],
            tangent_x,
            tangent_y
        )
    ) {
      low = left_index + 1;
    } else {
      high = left_index;
    }
  }
  const int left_index = low;
  const int right_index = diagonal - left_index;
  const bool take_left =
      left_index < left_count &&
      (right_index >= right_count ||
       robust_polar_less(
           input_edge_ids[left_start + left_index],
           input_edge_ids[right_start + right_index],
           tangent_x,
           tangent_y
       ));
  output_edge_ids[pos] = take_left
      ? input_edge_ids[left_start + left_index]
      : input_edge_ids[right_start + right_index];
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

extern "C" __global__ void __launch_bounds__(256, 4)
scatter_edge_face_ids(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    int* __restrict__ out_edge_face_ids,
    int face_count
) {
  const int face = blockIdx.x;
  if (face >= face_count) return;
  for (int pos = face_offsets[face] + threadIdx.x;
       pos < face_offsets[face + 1]; pos += blockDim.x) {
    out_edge_face_ids[face_edge_ids[pos]] = face;
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
scatter_previous_edge_ids(
    const int* __restrict__ next_edge_ids,
    int* __restrict__ out_previous_edge_ids,
    int edge_count
) {
  const int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edge_count) return;
  const int next = next_edge_ids[edge];
  if (next >= 0 && next < edge_count) out_previous_edge_ids[next] = edge;
}

// Topological cycle orientation is an exact adaptive-predicate carrier, not a
// sign extracted from the fp64 shoelace metric.  Each block classifies the
// lexicographic extreme of one face cycle; a retraced extreme spike is the
// exact unbounded-face signature for weak cycles with attached linework.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_face_exact_orientation(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const int* __restrict__ previous_edge_ids,
    const int* __restrict__ next_edge_ids,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    signed char* __restrict__ out_orientation,
    int face_count,
    int edge_count
) {
  const int face = blockIdx.x;
  const int tid = threadIdx.x;
  if (face >= face_count) return;
  __shared__ double sh_x[256];
  __shared__ double sh_y[256];
  __shared__ int sh_edge[256];
  __shared__ int sh_orientation[256];
  __shared__ int sh_rank[256];
  __shared__ int sh_valid[256];

  int local_edge = -1;
  int local_orientation = 0;
  int local_rank = 3;
  double local_x = 0.0;
  double local_y = 0.0;
  for (int pos = face_offsets[face] + tid;
       pos < face_offsets[face + 1]; pos += blockDim.x) {
    const int edge = face_edge_ids[pos];
    if (edge < 0 || edge >= edge_count) continue;
    const int previous = previous_edge_ids[edge];
    const int next = next_edge_ids[edge];
    if (previous < 0 || previous >= edge_count ||
        next < 0 || next >= edge_count) continue;
    const int orientation = vs_orient2d(
        src_x[previous], src_y[previous],
        src_x[edge], src_y[edge],
        src_x[next], src_y[next]
    );
    // At the lexicographic extreme, a retraced collinear spike is itself an
    // exact unbounded-face signature.  Prefer a clockwise turn, then a
    // counter-clockwise turn, and use the spike only when no turn exists at
    // that same extreme coordinate.
    const int orientation_rank = orientation < 0 ? 0 : (orientation > 0 ? 1 : 2);
    const double x = src_x[edge];
    const double y = src_y[edge];
    if (local_edge < 0 || x < local_x ||
        (x == local_x && (y < local_y ||
         (y == local_y &&
          (orientation_rank < local_rank ||
           (orientation_rank == local_rank && edge < local_edge)))))) {
      local_edge = edge;
      local_orientation = orientation;
      local_rank = orientation_rank;
      local_x = x;
      local_y = y;
    }
  }
  sh_x[tid] = local_x;
  sh_y[tid] = local_y;
  sh_edge[tid] = local_edge;
  sh_orientation[tid] = local_orientation;
  sh_rank[tid] = local_rank;
  sh_valid[tid] = local_edge >= 0;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride && sh_valid[tid + stride]) {
      const bool take = !sh_valid[tid] || sh_x[tid + stride] < sh_x[tid] ||
          (sh_x[tid + stride] == sh_x[tid] &&
           (sh_y[tid + stride] < sh_y[tid] ||
            (sh_y[tid + stride] == sh_y[tid] &&
             (sh_rank[tid + stride] < sh_rank[tid] ||
              (sh_rank[tid + stride] == sh_rank[tid] &&
               sh_edge[tid + stride] < sh_edge[tid])))));
      if (take) {
        sh_x[tid] = sh_x[tid + stride];
        sh_y[tid] = sh_y[tid + stride];
        sh_edge[tid] = sh_edge[tid + stride];
        sh_orientation[tid] = sh_orientation[tid + stride];
        sh_rank[tid] = sh_rank[tid + stride];
        sh_valid[tid] = 1;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_orientation[face] = sh_valid[0]
        ? (signed char)(sh_orientation[0] > 0 ? 1 : -1)
        : (signed char)0;
  }
}

// Exact fp64 bounds for each cycle. One thread walks one CSR span, so total
// work is linear in the number of half-edges and bounds are computed once.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_face_bounds(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    double* __restrict__ out_bounds,
    int face_count
) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= face_count) return;
  const int start = face_offsets[face];
  const int end = face_offsets[face + 1];
  const int first_edge = face_edge_ids[start];
  double min_x = src_x[first_edge];
  double min_y = src_y[first_edge];
  double max_x = min_x;
  double max_y = min_y;
  for (int pos = start + 1; pos < end; ++pos) {
    const int edge = face_edge_ids[pos];
    min_x = fmin(min_x, src_x[edge]);
    min_y = fmin(min_y, src_y[edge]);
    max_x = fmax(max_x, src_x[edge]);
    max_y = fmax(max_y, src_y[edge]);
  }
  out_bounds[face * 4] = min_x;
  out_bounds[face * 4 + 1] = min_y;
  out_bounds[face * 4 + 2] = max_x;
  out_bounds[face * 4 + 3] = max_y;
}

extern "C" __global__ void __launch_bounds__(256, 4)
initialize_dual_face_queue(
    const signed char* __restrict__ cycle_orientation,
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    int* __restrict__ queue,
    int* __restrict__ queue_tail,
    int* __restrict__ queue_ready,
    int* __restrict__ pending,
    int* __restrict__ left_winding,
    int* __restrict__ right_winding,
    int* __restrict__ face_component,
    int face_count
) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= face_count) return;
  if (cycle_orientation[face] < 0) {
    left_winding[face] = 0;
    right_winding[face] = 0;
    face_component[face] = face;
    const int start = face_offsets[face];
    const int degree = face_offsets[face + 1] - start;
    if (degree > 0) {
      atomicAdd(pending, degree);
      const int queue_start = atomicAdd(queue_tail, degree);
      for (int local = 0; local < degree; ++local) {
        queue[queue_start + local] = face_edge_ids[start + local];
        queue_ready[queue_start + local] = 1;
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 4)
propagate_dual_face_queue(
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const int* __restrict__ edge_face_ids,
    const int* __restrict__ left_delta,
    const int* __restrict__ right_delta,
    int* __restrict__ queue,
    int* __restrict__ queue_head,
    int* __restrict__ queue_tail,
    int* __restrict__ queue_ready,
    int* __restrict__ pending,
    int* __restrict__ left_winding,
    int* __restrict__ right_winding,
    int* __restrict__ face_component,
    int face_count,
    int edge_count
) {
  // Every face is claimed at most once and contributes each incidence once,
  // so queue capacity is exactly edge_count.  ``pending`` counts published or
  // actively processed incidences;
  // it can reach zero only after the last producer has finished enqueueing.
  while (atomicAdd(pending, 0) != 0) {
    const int head = atomicAdd(queue_head, 0);
    const int tail = atomicAdd(queue_tail, 0);
    if (head >= tail || atomicCAS(queue_head, head, head + 1) != head) {
      __nanosleep(64);
      continue;
    }
    while (atomicAdd(queue_ready + head, 0) == 0) __nanosleep(32);
    __threadfence();
    const int edge = queue[head];
    if (edge < 0 || edge >= edge_count) {
      atomicSub(pending, 1);
      continue;
    }
    const int face = edge_face_ids[edge];
    if (face < 0 || face >= face_count) {
      atomicSub(pending, 1);
      continue;
    }
    const int current_left = left_winding[face];
    const int current_right = right_winding[face];
    const int component = face_component[face];
    const int neighbor = edge_face_ids[edge ^ 1];
    if (neighbor >= 0 && neighbor < face_count) {
      const int candidate_left = current_left - left_delta[edge];
      const int candidate_right = current_right - right_delta[edge];
      if (atomicCAS(left_winding + neighbor, (-2147483647 - 1), candidate_left)
              == (-2147483647 - 1)) {
        right_winding[neighbor] = candidate_right;
        face_component[neighbor] = component;
        const int neighbor_start = face_offsets[neighbor];
        const int degree = face_offsets[neighbor + 1] - neighbor_start;
        if (degree > 0) {
          atomicAdd(pending, degree);
          const int queue_start = atomicAdd(queue_tail, degree);
          for (int local = 0; local < degree; ++local) {
            const int slot = queue_start + local;
            if (slot < edge_count) {
              queue[slot] = face_edge_ids[neighbor_start + local];
            }
          }
          __threadfence();
          for (int local = 0; local < degree; ++local) {
            const int slot = queue_start + local;
            if (slot < edge_count) atomicExch(queue_ready + slot, 1);
          }
        }
      }
    }
    atomicSub(pending, 1);
  }
}

__device__ __forceinline__ bool exact_cycle_contains_vertex(
    double px, double py,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ face_edge_ids,
    int start, int end
) {
  bool inside = false;
  for (int pos = start; pos < end; ++pos) {
    const int edge = face_edge_ids[pos];
    const int twin = edge ^ 1;
    const double ax = src_x[edge];
    const double ay = src_y[edge];
    const double bx = src_x[twin];
    const double by = src_y[twin];
    if ((ay <= py && py < by && vs_orient2d(ax, ay, bx, by, px, py) > 0) ||
        (by <= py && py < ay && vs_orient2d(ax, ay, bx, by, px, py) < 0)) {
      inside = !inside;
    }
  }
  return inside;
}

__device__ __forceinline__ int face_bounds_prefix_upper_bound(
    double px,
    const int* __restrict__ candidate_faces,
    const double* __restrict__ face_bounds,
    int candidate_capacity
) {
  int lo = 0;
  int hi = candidate_capacity;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    const int candidate = candidate_faces[mid];
    if (candidate >= 0 && face_bounds[candidate * 4] <= px) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ bool indexed_containment_bounds_match(
    int root_face,
    int candidate_face,
    double px,
    double py,
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ face_bounds,
    const int* __restrict__ source_rows,
    const int* __restrict__ face_component,
    int isolate_rows
) {
  if (face_component[root_face] < 0 || face_component[candidate_face] < 0 ||
      face_component[root_face] == face_component[candidate_face]) return false;
  const int root_edge = face_edge_ids[face_offsets[root_face]];
  const int candidate_edge = face_edge_ids[face_offsets[candidate_face]];
  if (isolate_rows != 0 && source_rows[root_edge] != source_rows[candidate_edge]) {
    return false;
  }
  return px >= face_bounds[candidate_face * 4] &&
      px <= face_bounds[candidate_face * 4 + 2] &&
      py >= face_bounds[candidate_face * 4 + 1] &&
      py <= face_bounds[candidate_face * 4 + 3];
}

// Fixed-capacity indexed reduction for disconnected component containment.
// One block owns one face-capacity root lane.  Inactive lanes return uniformly;
// active negative roots split the max-X tree at depth eight so all 256 threads
// traverse disjoint subtrees.  Index nodes are therefore visited once per root,
// while exact orient2d PIP runs only for actual bbox candidates.
extern "C" __global__ void __launch_bounds__(256, 2)
reduce_indexed_component_containment(
    const int* __restrict__ root_faces,
    const int* __restrict__ candidate_faces,
    const double* __restrict__ interval_max_x,
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ face_bounds,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ source_rows,
    const int* __restrict__ face_component,
    const int* __restrict__ left_winding,
    const int* __restrict__ right_winding,
    int* __restrict__ out_left_baseline,
    int* __restrict__ out_right_baseline,
    int* __restrict__ out_depth,
    int face_capacity,
    int leaf_count,
    int isolate_rows,
    int reduce_winding
) {
  const int root_pos = blockIdx.x;
  if (root_pos >= face_capacity) return;
  const int root_face = root_faces[root_pos];
  if (root_face < 0) return;
  const int tid = threadIdx.x;
  __shared__ int sh_upper;
  __shared__ int sh_left[256];
  __shared__ int sh_right[256];
  __shared__ int sh_depth[256];
  const int root_edge = face_edge_ids[face_offsets[root_face]];
  const double px = src_x[root_edge];
  const double py = src_y[root_edge];
  if (tid == 0) {
    sh_upper = face_bounds_prefix_upper_bound(
        px, candidate_faces, face_bounds, face_capacity);
  }
  __syncthreads();

  const int tree_depth = 31 - __clz(leaf_count);
  const int split_depth = tree_depth < 8 ? tree_depth : 8;
  const int split_width = 1 << split_depth;
  int stack[24];
  int stack_size = 0;
  if (tid < split_width) stack[stack_size++] = split_width + tid;
  int local_left = 0;
  int local_right = 0;
  int local_depth = 0;
  while (stack_size > 0) {
    const int node = stack[--stack_size];
    const int depth = 31 - __clz(node);
    const int level_start = 1 << depth;
    const int span = leaf_count >> depth;
    const int left = (node - level_start) * span;
    if (left >= sh_upper || interval_max_x[node] < px) continue;
    if (node >= leaf_count) {
      const int candidate_pos = node - leaf_count;
      if (candidate_pos < sh_upper && candidate_pos < face_capacity) {
        const int candidate = candidate_faces[candidate_pos];
        if (candidate >= 0 && indexed_containment_bounds_match(
                root_face, candidate, px, py, face_offsets, face_edge_ids,
                face_bounds, source_rows, face_component, isolate_rows) &&
            exact_cycle_contains_vertex(
                px, py, src_x, src_y, face_edge_ids,
                face_offsets[candidate], face_offsets[candidate + 1])) {
          ++local_depth;
          if (reduce_winding != 0) {
            local_left += left_winding[candidate];
            local_right += right_winding[candidate];
          }
        }
      }
      continue;
    }
    stack[stack_size++] = node * 2 + 1;
    stack[stack_size++] = node * 2;
  }
  sh_left[tid] = local_left;
  sh_right[tid] = local_right;
  sh_depth[tid] = local_depth;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sh_left[tid] += sh_left[tid + stride];
      sh_right[tid] += sh_right[tid + stride];
      sh_depth[tid] += sh_depth[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_left_baseline[root_face] = sh_left[0];
    out_right_baseline[root_face] = sh_right[0];
    out_depth[root_face] = sh_depth[0];
  }
}

// A second indexed pass selects the structurally immediate containing face.
// Component depth, not fp64 area ordering, defines parenthood.
extern "C" __global__ void __launch_bounds__(256, 2)
select_indexed_component_containment_parent(
    const int* __restrict__ root_faces,
    const int* __restrict__ candidate_faces,
    const double* __restrict__ interval_max_x,
    const int* __restrict__ face_offsets,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ face_bounds,
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ source_rows,
    const int* __restrict__ face_component,
    const int* __restrict__ component_depth,
    int* __restrict__ out_parent,
    int face_capacity,
    int leaf_count,
    int isolate_rows
) {
  const int root_pos = blockIdx.x;
  if (root_pos >= face_capacity) return;
  const int root_face = root_faces[root_pos];
  if (root_face < 0 || component_depth[root_face] <= 0) return;
  const int tid = threadIdx.x;
  __shared__ int sh_upper;
  __shared__ int sh_parent[256];
  const int root_edge = face_edge_ids[face_offsets[root_face]];
  const double px = src_x[root_edge];
  const double py = src_y[root_edge];
  if (tid == 0) {
    sh_upper = face_bounds_prefix_upper_bound(
        px, candidate_faces, face_bounds, face_capacity);
  }
  __syncthreads();

  const int tree_depth = 31 - __clz(leaf_count);
  const int split_depth = tree_depth < 8 ? tree_depth : 8;
  const int split_width = 1 << split_depth;
  int stack[24];
  int stack_size = 0;
  if (tid < split_width) stack[stack_size++] = split_width + tid;
  int local_parent = -1;
  const int target_depth = component_depth[root_face] - 1;
  while (stack_size > 0) {
    const int node = stack[--stack_size];
    const int depth = 31 - __clz(node);
    const int level_start = 1 << depth;
    const int span = leaf_count >> depth;
    const int left = (node - level_start) * span;
    if (left >= sh_upper || interval_max_x[node] < px) continue;
    if (node >= leaf_count) {
      const int candidate_pos = node - leaf_count;
      if (candidate_pos < sh_upper && candidate_pos < face_capacity) {
        const int candidate = candidate_faces[candidate_pos];
        const int candidate_component = candidate >= 0
            ? face_component[candidate] : -1;
        if (candidate_component >= 0 &&
            component_depth[candidate_component] == target_depth &&
            indexed_containment_bounds_match(
                root_face, candidate, px, py, face_offsets, face_edge_ids,
                face_bounds, source_rows, face_component, isolate_rows) &&
            exact_cycle_contains_vertex(
                px, py, src_x, src_y, face_edge_ids,
                face_offsets[candidate], face_offsets[candidate + 1]) &&
            (local_parent < 0 || candidate < local_parent)) {
          local_parent = candidate;
        }
      }
      continue;
    }
    stack[stack_size++] = node * 2 + 1;
    stack[stack_size++] = node * 2;
  }
  sh_parent[tid] = local_parent;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const int other_parent = sh_parent[tid + stride];
      if (other_parent >= 0 &&
          (sh_parent[tid] < 0 || other_parent < sh_parent[tid])) {
        sh_parent[tid] = other_parent;
      }
    }
    __syncthreads();
  }
  if (tid == 0) out_parent[root_face] = sh_parent[0];
}

extern "C" __global__ void __launch_bounds__(256, 4)
finalize_face_coverage(
    const int* __restrict__ face_component,
    const int* __restrict__ left_baseline,
    const int* __restrict__ right_baseline,
    int* __restrict__ left_winding,
    int* __restrict__ right_winding,
    signed char* __restrict__ left_covered,
    signed char* __restrict__ right_covered,
    int face_count
) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= face_count) return;
  const int component = face_component[face];
  if (component < 0 || component >= face_count) {
    left_covered[face] = 0;
    right_covered[face] = 0;
    return;
  }
  left_winding[face] += left_baseline[component];
  right_winding[face] += right_baseline[component];
  left_covered[face] = left_winding[face] != 0 ? 1 : 0;
  right_covered[face] = right_winding[face] != 0 ? 1 : 0;
}

"""
)

_OVERLAY_FACE_WALK_KERNEL_NAMES = (
    "mark_endpoint_group_ends",
    "scatter_node_offsets",
    "initialize_degree_two_core_frontier",
    "peel_degree_two_core_frontier",
    "merge_node_edges_robust_pass",
    "build_radial_successors",
    "compute_face_metrics",
    "scatter_edge_face_ids",
    "scatter_previous_edge_ids",
    "compute_face_exact_orientation",
    "compute_face_bounds",
    "initialize_dual_face_queue",
    "propagate_dual_face_queue",
    "reduce_indexed_component_containment",
    "select_indexed_component_containment_parent",
    "finalize_face_coverage",
)

# ---------------------------------------------------------------------------
# 4. Face assembly (ring reconstruction) kernels
# ---------------------------------------------------------------------------
# Kernels: boundary extraction, exact ring references/bounds, and nesting

_OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE = (
    ORIENT2D_DEVICE
    + r"""
__device__ __forceinline__ bool exact_ring_contains_vertex(
    double px,
    double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    int start,
    int end
) {
  bool inside = false;
  for (int i = start; i + 1 < end; ++i) {
    const double ax = x[i];
    const double ay = y[i];
    const double bx = x[i + 1];
    const double by = y[i + 1];
    if ((ay <= py && py < by && vs_orient2d(ax, ay, bx, by, px, py) > 0) ||
        (by <= py && py < ay && vs_orient2d(ax, ay, bx, by, px, py) < 0)) {
      inside = !inside;
    }
  }
  return inside;
}

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

// Preserve one existing boundary vertex and exact fp64 bounds per ring. The
// historical entry-point name is retained as an internal ABI for the assembly
// launcher; no synthetic interior coordinate is constructed.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_ring_sample_points(
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const bool* __restrict__ ring_active,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    double* __restrict__ out_reference_x,
    double* __restrict__ out_reference_y,
    double* __restrict__ out_bounds,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;
  if (!ring_active[r]) {
    out_reference_x[r] = 0.0;
    out_reference_y[r] = 0.0;
    out_bounds[r * 4] = 0.0;
    out_bounds[r * 4 + 1] = 0.0;
    out_bounds[r * 4 + 2] = 0.0;
    out_bounds[r * 4 + 3] = 0.0;
    return;
  }

  const int start = ring_coord_offsets[r];
  const int n_edges = ring_edge_counts[r];
  if (n_edges <= 0) {
    out_reference_x[r] = 0.0;
    out_reference_y[r] = 0.0;
    out_bounds[r * 4] = 0.0;
    out_bounds[r * 4 + 1] = 0.0;
    out_bounds[r * 4 + 2] = 0.0;
    out_bounds[r * 4 + 3] = 0.0;
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
  out_bounds[r * 4] = min_x;
  out_bounds[r * 4 + 1] = min_y;
  out_bounds[r * 4 + 2] = max_x;
  out_bounds[r * 4 + 3] = max_y;

  out_reference_x[r] = all_x[start];
  out_reference_y[r] = all_y[start];
}

// Locate the equal-key source-row span for each sorted ring lane. Inactive
// capacity lanes are never consumed by containment kernels and receive a
// one-lane span to keep every output initialized.
extern "C" __global__ void __launch_bounds__(256, 4)
locate_boundary_ring_group_spans(
    const int* __restrict__ sorted_group_keys,
    const int* __restrict__ sorted_ring_ids,
    const signed char* __restrict__ ring_active,
    int* __restrict__ out_group_start,
    int* __restrict__ out_group_end,
    int ring_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= ring_count) return;
  const int ring = sorted_ring_ids[pos];
  if (ring_active[ring] == 0) {
    out_group_start[pos] = pos;
    out_group_end[pos] = pos + 1;
    return;
  }
  const int key = sorted_group_keys[pos];
  int lo = 0;
  int hi = pos + 1;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (sorted_group_keys[mid] < key) lo = mid + 1;
    else hi = mid;
  }
  out_group_start[pos] = lo;
  lo = pos;
  hi = ring_count;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (sorted_group_keys[mid] <= key) lo = mid + 1;
    else hi = mid;
  }
  out_group_end[pos] = lo;
}

// Classify all boundary cycles by containment parity.  sorted_ring_ids and
// group_start/end keep each thread's search within one source-row span, so a
// million independent one-ring rows remain O(R), not O(R^2).
extern "C" __global__ void __launch_bounds__(256, 4)
count_boundary_ring_containment_depth(
    const double* __restrict__ ring_reference_x,
    const double* __restrict__ ring_reference_y,
    const double* __restrict__ ring_area,
    const signed char* __restrict__ ring_active,
    const int* __restrict__ sorted_ring_ids,
    const int* __restrict__ group_start,
    const int* __restrict__ group_end,
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    const double* __restrict__ ring_bounds,
    int* __restrict__ out_depth,
    int ring_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= ring_count) return;
  const int r = sorted_ring_ids[pos];
  if (ring_active[r] == 0) {
    out_depth[r] = 0;
    return;
  }

  const double px = ring_reference_x[r];
  const double py = ring_reference_y[r];
  const double area_r = ring_area[r] < 0.0 ? -ring_area[r] : ring_area[r];
  int depth = 0;
  for (int candidate_pos = group_start[pos];
       candidate_pos < group_end[pos]; ++candidate_pos) {
    const int c = sorted_ring_ids[candidate_pos];
    if (c == r || ring_active[c] == 0) continue;
    const double area_c = ring_area[c] < 0.0 ? -ring_area[c] : ring_area[c];
    if (area_c <= area_r) continue;
    const double* bounds = ring_bounds + c * 4;
    if (px < bounds[0] || py < bounds[1]
        || px > bounds[2] || py > bounds[3]) continue;
    const int coord_start = ring_coord_offsets[c];
    const int coord_end = coord_start + ring_edge_counts[c] + 1;
    if (exact_ring_contains_vertex(px, py, all_x, all_y, coord_start, coord_end)) {
      depth += 1;
    }
  }
  out_depth[r] = depth;
}

// Assign each odd-depth boundary cycle to the smallest containing even-depth
// cycle in the same source-row span.  Exterior cycles map to themselves.
extern "C" __global__ void __launch_bounds__(256, 4)
assign_holes_to_exteriors(
    const double* __restrict__ ring_reference_x,
    const double* __restrict__ ring_reference_y,
    const double* __restrict__ ring_area,
    const signed char* __restrict__ is_true_exterior,
    const signed char* __restrict__ ring_active,
    const int* __restrict__ sorted_ring_ids,
    const int* __restrict__ group_start,
    const int* __restrict__ group_end,
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    const double* __restrict__ ring_bounds,
    int* __restrict__ out_exterior_id,
    int ring_count
) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= ring_count) return;
  const int r = sorted_ring_ids[pos];
  if (ring_active[r] == 0) {
    out_exterior_id[r] = -1;
    return;
  }
  if (is_true_exterior[r] != 0) {
    out_exterior_id[r] = r;
    return;
  }
  const double px = ring_reference_x[r];
  const double py = ring_reference_y[r];
  const double abs_ring_area = ring_area[r] < 0.0 ? -ring_area[r] : ring_area[r];
  double best_area = 1e308;
  int best_exterior = -1;
  for (int candidate_pos = group_start[pos];
       candidate_pos < group_end[pos]; ++candidate_pos) {
    const int ext = sorted_ring_ids[candidate_pos];
    if (is_true_exterior[ext] == 0) continue;
    const double ext_area = ring_area[ext] < 0.0 ? -ring_area[ext] : ring_area[ext];
    if (ext_area >= best_area) continue;
    if (ext_area <= abs_ring_area) continue;
    const double* bounds = ring_bounds + ext * 4;
    if (px < bounds[0] || py < bounds[1]
        || px > bounds[2] || py > bounds[3]) continue;
    const int coord_start = ring_coord_offsets[ext];
    const int coord_end = coord_start + ring_edge_counts[ext] + 1;
    if (exact_ring_contains_vertex(px, py, all_x, all_y, coord_start, coord_end)) {
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
  const double reference_x = x[start];
  const double reference_y = y[start];
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
    if (!exact_ring_contains_vertex(
            reference_x, reference_y, x, y, hole_start, hole_end)) {
      continue;
    }
    if (area < best_area || best < 0 || h < best) {
      best_area = area;
      best = h;
    }
  }
  out_parent_interior[part] = best;
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
    "locate_boundary_ring_group_spans",
    "count_boundary_ring_containment_depth",
    "assign_holes_to_exteriors",
    "grouped_complement_hole_metrics",
    "assign_grouped_complement_exterior_parents",
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
