"""NVRTC kernel sources for overlay/gpu.py.

This module holds the CUDA C++ source strings and kernel name tuples for
the six NVRTC compilation groups used by the GPU overlay pipeline.  All
Python dispatch logic, CCCL primitive calls, warmup registration, and
compile_kernel_group wrappers remain in gpu.py.
"""

from __future__ import annotations

from vibespatial.cuda.device_functions.point_in_ring import (
    POINT_IN_RING_BOUNDARY_DEVICE,
    POINT_IN_RING_DEVICE,
)
from vibespatial.cuda.device_functions.point_on_segment import POINT_ON_SEGMENT_DEVICE
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

extern "C" __device__ unsigned int quantize_t(double t) {
  const double scaled = clamp01(t) * 1000000000.0;
  return (unsigned int)(scaled + 0.5);
}

extern "C" __device__ unsigned long long pack_key(int source_segment_id, double t) {
  return (((unsigned long long)(unsigned int)source_segment_id) << 32) | (unsigned long long)quantize_t(t);
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
    int right_count,
    int* __restrict__ out_source_segment_ids,
    double* __restrict__ out_t,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned long long* __restrict__ out_key,
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
  const int segment_index = is_right ? (segment_row - left_count) : segment_row;

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
  out_key[row] = pack_key(source_segment_id, t);
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
    int* __restrict__ out_source_segment_ids,
    double* __restrict__ out_t,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    unsigned long long* __restrict__ out_key,
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
  const int base = pair_offsets[row];

  if (kind == 1 || kind == 2) {
    const double x = point_x[row];
    const double y = point_y[row];
    const double left_t = project_t(x, y, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double right_t = project_t(x, y, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);

    out_source_segment_ids[base + 0] = left_index;
    out_t[base + 0] = left_t;
    out_x[base + 0] = x;
    out_y[base + 0] = y;
    out_key[base + 0] = pack_key(left_index, left_t);

    out_source_segment_ids[base + 1] = left_count + right_index;
    out_t[base + 1] = right_t;
    out_x[base + 1] = x;
    out_y[base + 1] = y;
    out_key[base + 1] = pack_key(left_count + right_index, right_t);
    return;
  }

  if (kind == 3) {
    const double x0 = overlap_x0[row];
    const double y0 = overlap_y0[row];
    const double x1 = overlap_x1[row];
    const double y1 = overlap_y1[row];

    const double left_t0 = project_t(x0, y0, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double left_t1 = project_t(x1, y1, left_x0[left_index], left_y0[left_index], left_x1[left_index], left_y1[left_index]);
    const double right_t0 = project_t(x0, y0, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);
    const double right_t1 = project_t(x1, y1, right_x0[right_index], right_y0[right_index], right_x1[right_index], right_y1[right_index]);

    out_source_segment_ids[base + 0] = left_index;
    out_t[base + 0] = left_t0;
    out_x[base + 0] = x0;
    out_y[base + 0] = y0;
    out_key[base + 0] = pack_key(left_index, left_t0);

    out_source_segment_ids[base + 1] = left_index;
    out_t[base + 1] = left_t1;
    out_x[base + 1] = x1;
    out_y[base + 1] = y1;
    out_key[base + 1] = pack_key(left_index, left_t1);

    out_source_segment_ids[base + 2] = left_count + right_index;
    out_t[base + 2] = right_t0;
    out_x[base + 2] = x0;
    out_y[base + 2] = y0;
    out_key[base + 2] = pack_key(left_count + right_index, right_t0);

    out_source_segment_ids[base + 3] = left_count + right_index;
    out_t[base + 3] = right_t1;
    out_x[base + 3] = x1;
    out_y[base + 3] = y1;
    out_key[base + 3] = pack_key(left_count + right_index, right_t1);
  }
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
    "emit_atomic_edges",
)

# ---------------------------------------------------------------------------
# 2. Half-edge face traversal kernels
# ---------------------------------------------------------------------------
# Kernels: compute_shoelace_contributions, compute_face_sample_points,
#          list_rank_within_cycle

_OVERLAY_FACE_WALK_KERNEL_SOURCE = SPATIAL_TOLERANCE_PREAMBLE + r"""
// -------------------------------------------------------------------
// Phase 1: GPU Face Walk via Pointer Jumping
// -------------------------------------------------------------------

// Compute per-edge shoelace contributions for area and centroid.
// src_x[i], src_y[i] are the source coordinates of edge i.
// next_edge_ids[i] gives the next edge in the cycle.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_shoelace_contributions(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const long long* __restrict__ next_edge_ids,
    double* __restrict__ out_cross,
    double* __restrict__ out_cx_contrib,
    double* __restrict__ out_cy_contrib,
    int edge_count
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= edge_count) return;
  const int next_i = (int)next_edge_ids[i];
  // Bounds check: prevent ILLEGAL_ADDRESS from corrupted topology.
  if (next_i < 0 || next_i >= edge_count) {
    out_cross[i] = 0.0;
    out_cx_contrib[i] = 0.0;
    out_cy_contrib[i] = 0.0;
    return;
  }
  const double x0 = src_x[i];
  const double y0 = src_y[i];
  const double x1 = src_x[next_i];
  const double y1 = src_y[next_i];
  const double cross = x0 * y1 - x1 * y0;
  out_cross[i] = cross;
  out_cx_contrib[i] = (x0 + x1) * cross;
  out_cy_contrib[i] = (y0 + y1) * cross;
}

// One thread per face: compute a sample point by walking the face edges.
// face_starts[f] and face_ends[f] give the range into sorted_edge_ids
// (edges sorted by face_id). The sample point is the perpendicular-offset
// midpoint of the first non-degenerate edge.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_face_sample_points(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const long long* __restrict__ next_edge_ids,
    const int* __restrict__ face_starts,
    const int* __restrict__ face_edge_ids,
    const double* __restrict__ signed_area,
    double* __restrict__ out_label_x,
    double* __restrict__ out_label_y,
    signed char* __restrict__ out_bounded,
    int face_count,
    int total_edge_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;

  const double area = signed_area[f];
  if (area <= VS_SPATIAL_EPSILON) {
    out_bounded[f] = 0;
    out_label_x[f] = 0.0;
    out_label_y[f] = 0.0;
    return;
  }
  out_bounded[f] = 1;

  const int start = face_starts[f];
  const int end = face_starts[f + 1];
  const int n_edges = end - start;

  // Walk face edges to find first non-degenerate edge for sample point
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
  if (extent < 1.0) extent = 1.0;
  const double epsilon = extent * 1e-6;

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
    best_lx = (x0 + x1) * 0.5 - (dy / length) * epsilon;
    best_ly = (y0 + y1) * 0.5 + (dx / length) * epsilon;
    break;
  }

  out_label_x[f] = best_lx;
  out_label_y[f] = best_ly;
}

// One thread per edge: compute the rank (position) of each edge within its
// face cycle.  face_id[e] holds the canonical root of the cycle (the minimum
// edge id, computed by prior pointer jumping).  next_edge_ids[e] gives the
// next edge in the cycle.  Each thread walks from the root forward through
// next_edge_ids counting steps until reaching e.
//
// Bounds safety: every next_edge_ids dereference checks 0 <= cur < edge_count
// to prevent ILLEGAL_ADDRESS from corrupted half-edge topology.  If the walk
// reaches max_walk without finding e (degenerate/broken cycle), rank is set
// to the thread index within the cycle to produce a deterministic (if
// arbitrary) ordering rather than colliding ranks that corrupt the sort.
extern "C" __global__ void __launch_bounds__(256, 4)
list_rank_within_cycle(
    const int* __restrict__ face_id,
    const long long* __restrict__ next_edge_ids,
    int* __restrict__ out_rank,
    int edge_count,
    int max_walk
) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edge_count) return;

  const int root = face_id[e];
  if (e == root) {
    out_rank[e] = 0;
    return;
  }

  // Bounds-check root before first dereference.
  if (root < 0 || root >= edge_count) {
    out_rank[e] = e;  // fallback: use edge id as rank
    return;
  }

  // Walk next_edge_ids from root until we reach e, counting steps.
  int cur = (int)next_edge_ids[root];
  int steps = 1;
  while (cur != e && steps < max_walk) {
    // Bounds check: prevent ILLEGAL_ADDRESS on corrupted topology.
    if (cur < 0 || cur >= edge_count) {
      out_rank[e] = e;  // fallback: deterministic but arbitrary rank
      return;
    }
    cur = (int)next_edge_ids[cur];
    steps++;
  }
  out_rank[e] = steps;
}
"""

_OVERLAY_FACE_WALK_KERNEL_NAMES = (
    "compute_shoelace_contributions",
    "compute_face_sample_points",
    "list_rank_within_cycle",
)

# ---------------------------------------------------------------------------
# 3. Face coverage labeling kernels
# ---------------------------------------------------------------------------
# Kernels: label_face_coverage_polygon, label_face_coverage_multipolygon

_OVERLAY_FACE_LABEL_KERNEL_SOURCE = (
    POINT_ON_SEGMENT_DEVICE + POINT_IN_RING_BOUNDARY_DEVICE + SPATIAL_TOLERANCE_PREAMBLE + r"""
// -------------------------------------------------------------------
// Phase 2: GPU Face Labeling via Batch Point-in-Polygon
// -------------------------------------------------------------------
#define OVERLAY_BOUNDARY_TOLERANCE VS_SPATIAL_EPSILON

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
  const double px = label_x[f];
  const double py = label_y[f];

  for (int poly = 0; poly < polygon_count; ++poly) {
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

// Test face sample points against all multipolygons on one side.
// One thread per face.
extern "C" __global__ void __launch_bounds__(256, 4)
label_face_coverage_multipolygon(
    const double* __restrict__ label_x,
    const double* __restrict__ label_y,
    const double* __restrict__ mp_x,
    const double* __restrict__ mp_y,
    const int* __restrict__ mp_geometry_offsets,
    const int* __restrict__ mp_part_offsets,
    const int* __restrict__ mp_ring_offsets,
    int mp_count,
    signed char* __restrict__ out_covered,
    int face_count
) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= face_count) return;
  if (out_covered[f] == 1) return;  // already covered by polygon pass
  const double px = label_x[f];
  const double py = label_y[f];

  for (int mp = 0; mp < mp_count; ++mp) {
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
""")

_OVERLAY_FACE_LABEL_KERNEL_NAMES = (
    "label_face_coverage_polygon",
    "label_face_coverage_multipolygon",
)

# ---------------------------------------------------------------------------
# 4. Face assembly (ring reconstruction) kernels
# ---------------------------------------------------------------------------
# Kernels: compute_boundary_edges, compute_boundary_next,
#          scatter_ring_coordinates, compute_ring_sample_points,
#          assign_holes_to_exteriors,
#          count_boundary_nesting_depth, count_sibling_hole_depth

_OVERLAY_FACE_ASSEMBLY_KERNEL_SOURCE = POINT_IN_RING_DEVICE + r"""
// Identify boundary edges: edges where the face on this side is selected
// but the twin face is not (or has no face).
// boundary_next[e] = next boundary edge following e along the boundary.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_boundary_edges(
    const int* __restrict__ edge_face_ids,
    const signed char* __restrict__ face_selected,
    const long long* __restrict__ next_edge_ids,
    signed char* __restrict__ out_is_boundary,
    int edge_count
) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edge_count) return;
  const int face_id = edge_face_ids[e];
  if (face_id < 0 || face_selected[face_id] == 0) {
    out_is_boundary[e] = 0;
    return;
  }
  // Check twin face
  const int twin = e ^ 1;
  if (twin >= edge_count) {
    out_is_boundary[e] = 1;
    return;
  }
  const int twin_face = edge_face_ids[twin];
  if (twin_face < 0 || face_selected[twin_face] == 0) {
    out_is_boundary[e] = 1;
  } else {
    out_is_boundary[e] = 0;
  }
}

// Compute next boundary edge for each boundary edge.
// For boundary edge e: follow next_edge_ids from e, crossing through
// non-boundary edges via twin traversal, until finding next boundary edge.
extern "C" __global__ void __launch_bounds__(256, 4)
compute_boundary_next(
    const int* __restrict__ edge_face_ids,
    const signed char* __restrict__ face_selected,
    const long long* __restrict__ next_edge_ids,
    const signed char* __restrict__ is_boundary,
    int* __restrict__ out_boundary_next,
    int edge_count,
    int max_steps
) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= edge_count || is_boundary[e] == 0) return;

  int current = (int)next_edge_ids[e];
  for (int step = 0; step < max_steps; ++step) {
    if (current < 0 || current >= edge_count) break;
    if (is_boundary[current] != 0) {
      out_boundary_next[e] = current;
      return;
    }
    // Cross through twin and follow next
    const int twin = current ^ 1;
    if (twin < 0 || twin >= edge_count) break;
    current = (int)next_edge_ids[twin];
  }
  out_boundary_next[e] = e;  // self-loop as fallback
}

// Scatter ring coordinates from boundary edges into output buffer.
// Each ring is a boundary cycle. ring_edge_starts[r] is the starting
// boundary edge for ring r. We follow boundary_next to walk the cycle
// and write coordinates.
//
// Bounds safety: src_x_count is the total number of elements in src_x/src_y.
// Every edge index is bounds-checked before dereferencing to prevent
// ILLEGAL_ADDRESS from corrupted half-edge topology or boundary_next cycles.
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_ring_coordinates(
    const double* __restrict__ src_x,
    const double* __restrict__ src_y,
    const int* __restrict__ ring_edge_starts,
    const int* __restrict__ ring_coord_offsets,
    const int* __restrict__ ring_edge_counts,
    const int* __restrict__ boundary_next,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    int ring_count,
    int src_x_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;

  const int start_edge = ring_edge_starts[r];
  const int offset = ring_coord_offsets[r];
  const int n_edges = ring_edge_counts[r];

  // Bounds check start_edge
  if (start_edge < 0 || start_edge >= src_x_count) {
    // Write zeros for the ring + closure to avoid uninitialized output
    for (int k = 0; k <= n_edges; ++k) {
      out_x[offset + k] = 0.0;
      out_y[offset + k] = 0.0;
    }
    return;
  }

  int current = start_edge;
  for (int k = 0; k < n_edges; ++k) {
    if (current < 0 || current >= src_x_count) {
      // Fill remaining with start_edge coords (degenerate but safe)
      for (int j = k; j <= n_edges; ++j) {
        out_x[offset + j] = src_x[start_edge];
        out_y[offset + j] = src_y[start_edge];
      }
      return;
    }
    out_x[offset + k] = src_x[current];
    out_y[offset + k] = src_y[current];
    current = boundary_next[current];
  }
  // Close the ring
  out_x[offset + n_edges] = src_x[start_edge];
  out_y[offset + n_edges] = src_y[start_edge];
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
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    double* __restrict__ out_sample_x,
    double* __restrict__ out_sample_y,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;

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
    int exterior_count,
    int* __restrict__ out_exterior_id,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;
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
  for (int ei = 0; ei < exterior_count; ++ei) {
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

// Count containment depth for boundary rings (positive area).
// For each boundary ring r, count how many OTHER boundary rings from
// the SAME source row with strictly larger area contain r's sample point.
// Even depth -> true exterior, odd depth -> nested interior.
// out_depth[r] is written for ALL boundary_count rings; callers
// inspect it only for positive-area boundary rings.
extern "C" __global__ void __launch_bounds__(256, 4)
count_boundary_nesting_depth(
    const double* __restrict__ sample_x,
    const double* __restrict__ sample_y,
    const double* __restrict__ ring_area,
    const int* __restrict__ source_rows,
    const int* __restrict__ coord_offsets,
    const int* __restrict__ edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    int* __restrict__ out_depth,
    int boundary_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= boundary_count) return;

  const double area_r = ring_area[r];
  if (area_r <= 0.0) {
    out_depth[r] = 0;
    return;
  }
  const double px = sample_x[r];
  const double py = sample_y[r];
  const int row_r = source_rows[r];
  int depth = 0;

  for (int c = 0; c < boundary_count; ++c) {
    if (c == r) continue;
    const double area_c = ring_area[c];
    if (area_c <= area_r) continue;          // only larger rings
    if (source_rows[c] != row_r) continue;   // same source row

    // PIP test: does ring c contain (px, py)?
    const int cs = coord_offsets[c];
    const int ce = cs + edge_counts[c] + 1;
    if (vs_ring_contains_point(px, py, all_x, all_y, cs, ce)) depth += 1;
  }
  out_depth[r] = depth;
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
    const int* __restrict__ coord_offsets,
    const int* __restrict__ edge_counts,
    const double* __restrict__ all_x,
    const double* __restrict__ all_y,
    int* __restrict__ out_depth,
    int ring_count
) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= ring_count) return;

  const int ext_r = exterior_id[r];
  // Not a hole: either unassigned or self-assigned (exterior)
  if (ext_r < 0 || ext_r == r) {
    out_depth[r] = 0;
    return;
  }

  const double px = sample_x[r];
  const double py = sample_y[r];
  const double abs_area_r = ring_area[r] < 0.0 ? -ring_area[r] : ring_area[r];
  int depth = 0;

  for (int c = 0; c < ring_count; ++c) {
    if (c == r) continue;
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
"""

_OVERLAY_FACE_ASSEMBLY_KERNEL_NAMES = (
    "compute_boundary_edges",
    "compute_boundary_next",
    "scatter_ring_coordinates",
    "compute_ring_sample_points",
    "assign_holes_to_exteriors",
    "count_boundary_nesting_depth",
    "count_sibling_hole_depth",
)

# ---------------------------------------------------------------------------
# 5. Batch point-in-ring kernel
# ---------------------------------------------------------------------------
# Kernel: batch_point_in_ring

_BATCH_POINT_IN_RING_KERNEL_SOURCE = POINT_IN_RING_DEVICE + r"""
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

_BATCH_POINT_IN_RING_KERNEL_NAMES = ("batch_point_in_ring",)

# ---------------------------------------------------------------------------
# 6. Containment bypass kernels
# ---------------------------------------------------------------------------
# Kernels: containment_poly_vs_poly, containment_poly_vs_mpoly,
#          containment_mpoly_vs_poly, containment_mpoly_vs_mpoly

_CONTAINMENT_BYPASS_KERNEL_SOURCE = POINT_IN_RING_DEVICE + r"""
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

// -----------------------------------------------------------------
// Thread-per-polygon kernel: test all vertices of each candidate
// polygon against the corridor.  Output 1 if ALL vertices are
// inside, 0 otherwise.  Reads vertices directly from the source
// family coordinate buffers using offset indirection -- no vertex
// scatter required.
// -----------------------------------------------------------------

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
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_ring = left_geom_offsets[frow];
    const int last_ring = left_geom_offsets[frow + 1];
    const int coord_start = left_ring_offsets[first_ring];
    const int coord_end = left_ring_offsets[last_ring];
    for (int v = coord_start; v < coord_end; ++v) {
        if (!_cb_polygon_contains(
                left_x[v], left_y[v],
                corr_x, corr_y,
                corr_geom_offsets, corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
    }
    out[cid] = 1;
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
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_ring = left_geom_offsets[frow];
    const int last_ring = left_geom_offsets[frow + 1];
    const int coord_start = left_ring_offsets[first_ring];
    const int coord_end = left_ring_offsets[last_ring];
    for (int v = coord_start; v < coord_end; ++v) {
        if (!_cb_multipolygon_contains(
                left_x[v], left_y[v],
                corr_x, corr_y,
                corr_geom_offsets, corr_part_offsets,
                corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
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
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_part = left_geom_offsets[frow];
    const int last_part = left_geom_offsets[frow + 1];
    const int first_ring = left_part_offsets[first_part];
    const int last_ring = left_part_offsets[last_part];
    const int coord_start = left_ring_offsets[first_ring];
    const int coord_end = left_ring_offsets[last_ring];
    for (int v = coord_start; v < coord_end; ++v) {
        if (!_cb_polygon_contains(
                left_x[v], left_y[v],
                corr_x, corr_y,
                corr_geom_offsets, corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
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
    int* __restrict__ out
) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_candidates) return;
    const int frow = cand_family_rows[cid];
    const int first_part = left_geom_offsets[frow];
    const int last_part = left_geom_offsets[frow + 1];
    const int first_ring = left_part_offsets[first_part];
    const int last_ring = left_part_offsets[last_part];
    const int coord_start = left_ring_offsets[first_ring];
    const int coord_end = left_ring_offsets[last_ring];
    for (int v = coord_start; v < coord_end; ++v) {
        if (!_cb_multipolygon_contains(
                left_x[v], left_y[v],
                corr_x, corr_y,
                corr_geom_offsets, corr_part_offsets,
                corr_ring_offsets, corr_row)) {
            out[cid] = 0;
            return;
        }
    }
    out[cid] = 1;
}
"""

_CONTAINMENT_BYPASS_KERNEL_NAMES = (
    "containment_poly_vs_poly",
    "containment_poly_vs_mpoly",
    "containment_mpoly_vs_poly",
    "containment_mpoly_vs_mpoly",
)
