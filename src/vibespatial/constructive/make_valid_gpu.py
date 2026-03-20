"""GPU-resident make_valid repair pipeline (ADR-0019 + ADR-0033).

Replaces the Shapely CPU fallback in make_valid_pipeline.py with a fully
GPU-resident repair path.  All repaired geometry stays device-resident in
OwnedGeometryArray (zero host round-trip per ADR-0005).

Pipeline stages:
  Phase A: GPU self-intersection detection (extract ring segments, classify pairs)
  Phase B: Simple repair kernels (close rings, remove duplicate vertices, fix orientation)
  Phase C: Self-intersection splitting (count/scatter split events, sort, dedup, rebuild)
  Phase D: Re-polygonization (half-edges, face walk, face containment, assembly)
  Phase E: Output assembly (build DeviceFamilyGeometryBuffer)

All kernels use fp64 compute (CONSTRUCTIVE class per ADR-0002).
Tier 1: NVRTC for geometry-specific work.
Tier 3a: CCCL for scan/sort/compact.
Tier 2: CuPy for element-wise ops.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    segmented_reduce_sum,
    sort_pairs,
    unique_sorted_pairs,
)

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "unique_by_key_i32_i32", "unique_by_key_u64_i32",
    "segmented_reduce_sum_f64",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily, get_geometry_buffer_schema  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.overlay.gpu import (  # noqa: E402
    _build_polygon_output_from_faces_gpu,
    _gpu_face_walk,
    build_gpu_half_edge_graph,
)
from vibespatial.overlay.types import (  # noqa: E402
    AtomicEdgeDeviceState,
    AtomicEdgeTable,
    OverlayFaceDeviceState,
    OverlayFaceTable,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.residency import Residency  # noqa: E402

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None


# ---------------------------------------------------------------------------
# Phase B: Simple Repair NVRTC Kernels (Tier 1)
# ---------------------------------------------------------------------------
# close_rings: per-ring, if first != last vertex, append closing vertex
# flag_duplicate_vertices: per vertex, compare to previous within same ring
# compute_ring_signed_area: shoelace cross-product per ring for orientation
# reverse_ring_coords: reverse coordinates for wrong-orientation rings

_REPAIR_KERNEL_SOURCE = r"""
// Phase B: close_rings — one thread per ring.
// If first vertex != last vertex, write closing vertex into reserved slot.
// new_ring_offsets[ring+1] has space for the extra vertex if needed.
extern "C" __global__ void close_rings(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* ring_needs_close,
    const int* new_ring_offsets,
    double* out_x,
    double* out_y,
    int ring_count
) {
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;
    const int src_start = ring_offsets[ring];
    const int src_end = ring_offsets[ring + 1];
    const int src_len = src_end - src_start;
    const int dst_start = new_ring_offsets[ring];
    // Copy existing vertices
    for (int i = 0; i < src_len; i++) {
        out_x[dst_start + i] = x[src_start + i];
        out_y[dst_start + i] = y[src_start + i];
    }
    // Append closing vertex if needed
    if (ring_needs_close[ring]) {
        out_x[dst_start + src_len] = x[src_start];
        out_y[dst_start + src_len] = y[src_start];
    }
}

// Phase B: flag_duplicate_vertices — one thread per vertex.
// Flag vertex if it equals the previous vertex within the same ring.
extern "C" __global__ void flag_duplicate_vertices(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* vertex_ring_ids,
    unsigned char* out_keep,
    int vertex_count
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vertex_count) return;
    const int ring = vertex_ring_ids[v];
    const int ring_start = ring_offsets[ring];
    // Always keep the first vertex in a ring
    if (v == ring_start) {
        out_keep[v] = 1;
        return;
    }
    // Flag as duplicate if coords match previous vertex
    if (x[v] == x[v - 1] && y[v] == y[v - 1]) {
        out_keep[v] = 0;
    } else {
        out_keep[v] = 1;
    }
}

// Phase B: compute_ring_shoelace — one thread per segment within a ring.
// Outputs cross product xi*y(i+1) - x(i+1)*yi for shoelace formula.
extern "C" __global__ void compute_ring_shoelace(
    const double* x,
    const double* y,
    const int* ring_offsets,
    double* out_cross,
    int ring_count,
    int vertex_count
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vertex_count) return;
    // out_cross[v] = x[v]*y[v+1] - x[v+1]*y[v]
    // But we must not cross ring boundaries.
    // We handle this by setting cross=0 for the last vertex in each ring.
    // The segmented_reduce handles per-ring summation.
    out_cross[v] = x[v] * y[v + 1] - x[v + 1] * y[v];
}

// Phase B: reverse_ring_coords — one thread per vertex in rings that need reversal.
extern "C" __global__ void reverse_ring_coords(
    double* x,
    double* y,
    const int* ring_offsets,
    const unsigned char* ring_needs_reverse,
    int ring_count
) {
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;
    if (!ring_needs_reverse[ring]) return;
    const int start = ring_offsets[ring];
    const int end = ring_offsets[ring + 1];
    // Reverse all vertices except the last (closure) vertex
    // After reversal, update closure vertex
    const int n = end - start;
    if (n < 3) return;
    // Reverse interior vertices (skip closure point)
    const int interior_end = (n > 0 && x[end - 1] == x[start] && y[end - 1] == y[start])
                             ? end - 1 : end;
    const int count = interior_end - start;
    for (int i = 0; i < count / 2; i++) {
        const int a = start + i;
        const int b = start + count - 1 - i;
        double tmp_x = x[a]; x[a] = x[b]; x[b] = tmp_x;
        double tmp_y = y[a]; y[a] = y[b]; y[b] = tmp_y;
    }
    // Update closure vertex
    if (interior_end < end) {
        x[end - 1] = x[start];
        y[end - 1] = y[start];
    }
}

// Phase A: extract_ring_segments — one thread per segment.
// Each thread computes its ring via binary search in seg_offsets, then reads
// the two consecutive vertices from x/y to populate the flat segment table.
extern "C" __global__ void extract_ring_segments(
    const double* x,
    const double* y,
    const int* ring_offsets,
    const int* seg_offsets,
    double* seg_x0,
    double* seg_y0,
    double* seg_x1,
    double* seg_y1,
    int* seg_ring_ids,
    int ring_count,
    int total_segments
) {
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= total_segments) return;
    // Binary search for the ring that owns this segment
    int lo = 0, hi = ring_count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (seg_offsets[mid + 1] <= seg) lo = mid + 1;
        else hi = mid;
    }
    const int ring = lo;
    const int local_seg = seg - seg_offsets[ring];
    const int v = ring_offsets[ring] + local_seg;
    seg_x0[seg] = x[v];
    seg_y0[seg] = y[v];
    seg_x1[seg] = x[v + 1];
    seg_y1[seg] = y[v + 1];
    seg_ring_ids[seg] = ring;
}

// Phase A: generate_intra_ring_pairs — one thread per candidate pair.
// Decodes (i, j) within a ring using triangular number formula.
// pair_counts[ring] = k*(k-1)/2 - k for k segments (all non-adjacent pairs,
// minus the wrap-around adjacency).
extern "C" __global__ void generate_intra_ring_pairs(
    const int* pair_offsets,
    const int* seg_offsets,
    const int* ring_seg_counts,
    int* out_left,
    int* out_right,
    int ring_count,
    int total_pairs
) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= total_pairs) return;
    // Binary search for the ring that owns this pair
    int lo = 0, hi = ring_count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (pair_offsets[mid + 1] <= pid) lo = mid + 1;
        else hi = mid;
    }
    const int ring = lo;
    const int local_pid = pid - pair_offsets[ring];
    const int k = ring_seg_counts[ring];
    const int base = seg_offsets[ring];
    // Decode (i, j) from the linear pair index.
    // We enumerate all pairs (i, j) with 0 <= i < j < k, skipping
    // adjacent pairs (j == i+1) and the wrap-around pair (i==0, j==k-1).
    // Iterate through the enumeration to find the right (i, j).
    int idx = 0;
    for (int i = 0; i < k; i++) {
        for (int j = i + 2; j < k; j++) {
            if (i == 0 && j == k - 1) continue;  // wrap-around adjacency
            if (idx == local_pid) {
                out_left[pid] = base + i;
                out_right[pid] = base + j;
                return;
            }
            idx++;
        }
    }
}

// Phase C: count self-intersection split events per intra-ring segment pair.
// kind: 0=none, 1=proper crossing, 2=endpoint touch
extern "C" __global__ void count_self_split_events(
    const signed char* kinds,
    int* out_counts,
    int pair_count
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pair_count) return;
    const signed char kind = kinds[p];
    // kind==1 (proper crossing): one split point on each segment = 2 events
    // kind==2 (endpoint touch): one split point on each segment = 2 events
    out_counts[p] = (kind == 1 || kind == 2) ? 2 : 0;
}

// Phase C: scatter self-intersection split events.
// For a proper crossing, emit the intersection point as a split on both segments.
extern "C" __device__ double clamp01(double value) {
    return value < 0.0 ? 0.0 : (value > 1.0 ? 1.0 : value);
}

extern "C" __device__ double project_t_self(
    double px, double py,
    double x0, double y0, double x1, double y1
) {
    const double dx = x1 - x0;
    const double dy = y1 - y0;
    const double abs_dx = dx < 0.0 ? -dx : dx;
    const double abs_dy = dy < 0.0 ? -dy : dy;
    if (abs_dx >= abs_dy) {
        return dx == 0.0 ? 0.0 : clamp01((px - x0) / dx);
    }
    return dy == 0.0 ? 0.0 : clamp01((py - y0) / dy);
}

extern "C" __device__ unsigned long long pack_self_key(int segment_id, double t) {
    const double scaled = clamp01(t) * 1000000000.0;
    unsigned int qt = (unsigned int)(scaled + 0.5);
    return (((unsigned long long)(unsigned int)segment_id) << 32) | (unsigned long long)qt;
}

extern "C" __global__ void scatter_self_split_events(
    const int* seg_a_ids,
    const int* seg_b_ids,
    const signed char* kinds,
    const double* point_x,
    const double* point_y,
    const double* seg_x0,
    const double* seg_y0,
    const double* seg_x1,
    const double* seg_y1,
    const int* event_offsets,
    int* out_segment_ids,
    double* out_t,
    double* out_x,
    double* out_y,
    unsigned long long* out_key,
    int pair_count
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pair_count) return;
    const signed char kind = kinds[p];
    if (kind != 1 && kind != 2) return;
    const int base = event_offsets[p];
    const int a = seg_a_ids[p];
    const int b = seg_b_ids[p];
    const double ix = point_x[p];
    const double iy = point_y[p];
    const double t_a = project_t_self(ix, iy, seg_x0[a], seg_y0[a], seg_x1[a], seg_y1[a]);
    const double t_b = project_t_self(ix, iy, seg_x0[b], seg_y0[b], seg_x1[b], seg_y1[b]);
    out_segment_ids[base] = a;
    out_t[base] = t_a;
    out_x[base] = ix;
    out_y[base] = iy;
    out_key[base] = pack_self_key(a, t_a);
    out_segment_ids[base + 1] = b;
    out_t[base + 1] = t_b;
    out_x[base + 1] = ix;
    out_y[base + 1] = iy;
    out_key[base + 1] = pack_self_key(b, t_b);
}

// Phase C: rebuild ring coordinates from original vertices + split points.
// One thread per output vertex. Uses binary search in segment offsets.
extern "C" __global__ void rebuild_ring_coords(
    const double* orig_x,
    const double* orig_y,
    const int* orig_ring_offsets,
    const int* split_seg_ids,
    const double* split_x,
    const double* split_y,
    const double* split_t,
    const int* seg_split_offsets,
    const int* seg_split_counts,
    const int* new_ring_offsets,
    const int* ring_seg_offsets,
    double* out_x,
    double* out_y,
    int ring_count,
    int total_segments
) {
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;
    const int orig_start = orig_ring_offsets[ring];
    const int orig_end = orig_ring_offsets[ring + 1];
    const int n_verts = orig_end - orig_start;
    if (n_verts < 2) return;
    const int out_start = new_ring_offsets[ring];
    const int seg_base = ring_seg_offsets[ring];
    int out_pos = out_start;
    // For each segment in the ring
    const int n_segs = n_verts - 1;
    for (int s = 0; s < n_segs; s++) {
        const int seg_id = seg_base + s;
        // Emit original start vertex of segment
        out_x[out_pos] = orig_x[orig_start + s];
        out_y[out_pos] = orig_y[orig_start + s];
        out_pos++;
        // Emit any split points for this segment (already sorted by t)
        if (seg_id < total_segments) {
            const int sp_start = seg_split_offsets[seg_id];
            const int sp_count = seg_split_counts[seg_id];
            for (int k = 0; k < sp_count; k++) {
                out_x[out_pos] = split_x[sp_start + k];
                out_y[out_pos] = split_y[sp_start + k];
                out_pos++;
            }
        }
    }
    // Emit closing vertex (same as first)
    out_x[out_pos] = orig_x[orig_start];
    out_y[out_pos] = orig_y[orig_start];
}
"""

_REPAIR_KERNEL_NAMES = (
    "close_rings",
    "flag_duplicate_vertices",
    "compute_ring_shoelace",
    "reverse_ring_coords",
    "extract_ring_segments",
    "generate_intra_ring_pairs",
    "count_self_split_events",
    "scatter_self_split_events",
    "rebuild_ring_coords",
)

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES),
])


def _compile_repair_kernels():
    return compile_kernel_group("make-valid-repair", _REPAIR_KERNEL_SOURCE, _REPAIR_KERNEL_NAMES)


# ---------------------------------------------------------------------------
# Phase B: Simple repair operations
# ---------------------------------------------------------------------------

def _gpu_close_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Close unclosed rings by appending first vertex. Returns new x, y, ring_offsets.

    Phase 7 GPU-resident: closure check via CuPy vectorised gather (no host copy).
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    block = (256, 1, 1)

    # --- GPU-resident closure check (Tier 2: CuPy element-wise) ---
    d_starts = d_ring_offsets[:-1]            # first vertex index per ring
    d_ends = d_ring_offsets[1:]               # one-past-last index per ring
    d_sizes = d_ends - d_starts               # vertex count per ring

    # Gather first and last coordinates per ring (device-resident)
    d_first_x = d_x[d_starts]
    d_first_y = d_y[d_starts]
    d_last_x = d_x[d_ends - 1]
    d_last_y = d_y[d_ends - 1]

    # Ring needs closing when size >= 2 and first != last
    d_needs_close = (
        (d_sizes >= 2)
        & ((d_first_x != d_last_x) | (d_first_y != d_last_y))
    ).astype(cp.int32)

    if int(cp.sum(d_needs_close)) == 0:
        return d_x, d_y, d_ring_offsets

    # Compute new ring sizes and offsets on device
    d_new_sizes = d_sizes + d_needs_close
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)
    total_new = int(d_new_offsets[-1])

    d_out_x = cp.empty(total_new, dtype=cp.float64)
    d_out_y = cp.empty(total_new, dtype=cp.float64)

    grid = (max(1, (ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["close_rings"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_close),
             ptr(d_new_offsets), ptr(d_out_x), ptr(d_out_y), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    return d_out_x, d_out_y, d_new_offsets


def _gpu_remove_duplicate_vertices(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """Remove consecutive duplicate vertices within each ring.

    Phase 7 GPU-resident: vertex-to-ring mapping via cp.searchsorted,
    compaction via compact_indices (CCCL), ring offsets via CuPy cumsum.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    block = (256, 1, 1)
    vertex_count = int(d_x.size)
    if vertex_count == 0:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident vertex→ring mapping (Tier 2: CuPy searchsorted) ---
    d_vertex_ids = cp.arange(vertex_count, dtype=cp.int32)
    d_vertex_ring_ids = cp.searchsorted(
        d_ring_offsets[1:], d_vertex_ids, side="right"
    ).astype(cp.int32)

    d_keep = cp.empty(vertex_count, dtype=cp.uint8)

    grid = (max(1, (vertex_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["flag_duplicate_vertices"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_vertex_ring_ids),
             ptr(d_keep), vertex_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    runtime.synchronize()

    # Check if any duplicates found (device-side sum)
    dup_count = int(vertex_count - int(cp.sum(d_keep)))
    if dup_count == 0:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident compaction (Tier 3a: CCCL compact_indices) ---
    compact_result = compact_indices(d_keep)
    d_kept_indices = compact_result.values.astype(cp.int64)

    new_x = d_x[d_kept_indices]
    new_y = d_y[d_kept_indices]

    # --- GPU-resident ring offset rebuild (Tier 2: CuPy cumsum) ---
    # Count kept vertices per ring using the keep mask
    d_keep_i32 = d_keep.astype(cp.int32)
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    # Per-ring kept count via segmented reduce (CCCL)
    seg_result = segmented_reduce_sum(
        d_keep_i32, d_starts, d_ends, num_segments=ring_count,
    )
    d_new_sizes = seg_result.values.astype(cp.int32)
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)

    return new_x, new_y, d_new_offsets


def _gpu_fix_ring_orientation(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Fix ring orientation: exterior rings CCW (positive area), holes CW (negative area).

    Phase 7 GPU-resident: activates the compute_ring_shoelace NVRTC kernel,
    uses segmented_reduce_sum (CCCL) for per-ring signed area, then CuPy
    element-wise logic with d_geom_offsets to determine orientation.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    block = (256, 1, 1)

    vertex_count = int(d_x.size)
    if vertex_count < 3 or ring_count == 0:
        return d_x, d_y

    # --- Step 1: Compute per-vertex shoelace cross products on GPU ---
    # The compute_ring_shoelace kernel writes x[v]*y[v+1] - x[v+1]*y[v]
    # for every vertex v.  The last vertex in each ring would read beyond
    # its ring boundary, so we zero those contributions afterwards.
    d_cross = cp.zeros(vertex_count, dtype=cp.float64)
    # Launch for vertex_count - 1 to avoid reading x[vertex_count] / y[vertex_count]
    safe_count = vertex_count - 1
    if safe_count > 0:
        grid_safe = (max(1, (safe_count + 255) // 256), 1, 1)
        runtime.launch(
            kernels["compute_ring_shoelace"],
            grid=grid_safe, block=block,
            params=(
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_cross),
                 ring_count, safe_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
            ),
        )

    # Zero out cross products at the last vertex of each ring (ring boundary)
    # to prevent cross-ring contamination.
    d_last_verts = d_ring_offsets[1:] - 1  # index of last vertex per ring
    d_cross[d_last_verts] = 0.0

    # --- Step 2: Segmented reduce to get per-ring signed area (CCCL) ---
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_cross, d_starts, d_ends, num_segments=ring_count,
    )
    d_ring_areas = seg_result.values * 0.5  # fp64 multiply

    # --- Step 3: Determine which rings need reversal (CuPy element-wise) ---
    # Build per-ring "is exterior" mask using d_geom_offsets.
    # A ring is exterior if its index equals d_geom_offsets[p] for some polygon p.
    # Equivalently, use searchsorted: ring_idx → polygon → first ring of that poly.
    d_ring_ids = cp.arange(ring_count, dtype=cp.int32)
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_ring_ids, side="right"
    ).astype(cp.int32)
    d_first_ring_of_poly = d_geom_offsets[d_poly_of_ring]
    d_is_exterior = (d_ring_ids == d_first_ring_of_poly)

    # Exterior → should be CCW (positive area), reverse if negative
    # Hole → should be CW (negative area), reverse if positive
    d_needs_reverse = (
        (d_is_exterior & (d_ring_areas < 0))
        | (~d_is_exterior & (d_ring_areas > 0))
    ).astype(cp.uint8)

    if int(cp.sum(d_needs_reverse)) == 0:
        return d_x, d_y

    grid = (max(1, (ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["reverse_ring_coords"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_reverse), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    runtime.synchronize()
    return d_x, d_y


# ---------------------------------------------------------------------------
# Phase A + C: Self-intersection detection and splitting
# ---------------------------------------------------------------------------

def _extract_ring_segments_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Extract consecutive vertex pairs as flat segment table from ring coords.

    Phase 8 GPU-resident: seg_counts via CuPy, seg_offsets via exclusive_sum
    (CCCL), then one NVRTC kernel thread per segment.

    Returns (seg_x0, seg_y0, seg_x1, seg_y1, seg_ring_ids) on device.
    """
    # --- Compute seg_counts and seg_offsets on device ---
    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int32)
    _ = exclusive_sum(d_seg_counts)
    # Build (ring_count+1)-element offset array
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_seg_counts)
    total_segments = int(d_seg_offsets[-1])

    if total_segments == 0:
        empty = cp.empty(0, dtype=cp.float64)
        return empty, empty, empty, empty, cp.empty(0, dtype=cp.int32)

    d_seg_x0 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_y0 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_x1 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_y1 = cp.empty(total_segments, dtype=cp.float64)
    d_seg_ring_ids = cp.empty(total_segments, dtype=cp.int32)

    if kernels is not None:
        runtime = get_cuda_runtime()
        ptr = runtime.pointer
        block = (256, 1, 1)
        grid = (max(1, (total_segments + 255) // 256), 1, 1)
        runtime.launch(
            kernels["extract_ring_segments"],
            grid=grid, block=block,
            params=(
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_seg_offsets),
                 ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
                 ptr(d_seg_ring_ids), ring_count, total_segments),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
            ),
        )
        runtime.synchronize()
    else:
        # Fallback: CuPy vectorised segment extraction (no NVRTC needed)
        d_seg_ids = cp.arange(total_segments, dtype=cp.int32)
        d_ring_of_seg = cp.searchsorted(d_seg_offsets[1:], d_seg_ids, side="right").astype(cp.int32)
        d_local_seg = d_seg_ids - d_seg_offsets[d_ring_of_seg]
        d_v = d_ring_offsets[d_ring_of_seg] + d_local_seg
        d_seg_x0[:] = d_x[d_v]
        d_seg_y0[:] = d_y[d_v]
        d_seg_x1[:] = d_x[d_v + 1]
        d_seg_y1[:] = d_y[d_v + 1]
        d_seg_ring_ids[:] = d_ring_of_seg

    return d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids


def _detect_intra_ring_intersections(
    d_seg_x0: cp.ndarray,
    d_seg_y0: cp.ndarray,
    d_seg_x1: cp.ndarray,
    d_seg_y1: cp.ndarray,
    d_seg_ring_ids: cp.ndarray,
    total_segments: int,
    ring_count: int,
    d_ring_offsets: cp.ndarray,
    kernels: dict | None = None,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Detect self-intersections within each ring using segment pair classification.

    Phase 8 GPU-resident: pair generation via NVRTC kernel (or CuPy fallback).
    The classify_segment_pairs kernel call (already GPU) is unchanged.

    Returns (seg_a_ids, seg_b_ids, kinds, point_x, point_y) as device
    (CuPy) arrays.  Only returns pairs with kind != 0 (actual intersections).
    """
    if total_segments < 2:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    # --- GPU-resident segment offset table per ring (CuPy) ---
    # d_seg_ring_ids is already on device; compute per-ring segment counts
    d_ring_seg_counts = cp.bincount(d_seg_ring_ids, minlength=ring_count).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_ring_seg_counts)

    # --- Compute pair counts per ring on device ---
    # For ring with k segments: pairs = k*(k-1)/2 - k (all pairs minus adjacent minus wrap)
    # More precisely: total combos = k*(k-1)/2, adjacent = k (including wrap), non-adj = total - k
    d_k = d_ring_seg_counts.astype(cp.int64)
    d_pair_counts = cp.maximum(d_k * (d_k - 1) // 2 - d_k, 0).astype(cp.int32)

    d_pair_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_pair_offsets[1:] = cp.cumsum(d_pair_counts)
    total_pairs = int(d_pair_offsets[-1])

    if total_pairs == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    # --- Generate pairs on GPU ---
    d_left_lookup = cp.empty(total_pairs, dtype=cp.int32)
    d_right_lookup = cp.empty(total_pairs, dtype=cp.int32)

    if kernels is not None and "generate_intra_ring_pairs" in kernels:
        runtime_gen = get_cuda_runtime()
        ptr_gen = runtime_gen.pointer
        block_gen = (256, 1, 1)
        grid_gen = (max(1, (total_pairs + 255) // 256), 1, 1)
        runtime_gen.launch(
            kernels["generate_intra_ring_pairs"],
            grid=grid_gen, block=block_gen,
            params=(
                (ptr_gen(d_pair_offsets), ptr_gen(d_seg_offsets),
                 ptr_gen(d_ring_seg_counts), ptr_gen(d_left_lookup),
                 ptr_gen(d_right_lookup), ring_count, total_pairs),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32),
            ),
        )
        runtime_gen.synchronize()
    else:
        # Fallback: host-side pair generation (NVRTC unavailable).
        # cp.asnumpy and Python for-loops are intentional here — this
        # path only executes when NVRTC compilation is not available.
        h_ring_seg_counts = cp.asnumpy(d_ring_seg_counts)
        h_seg_offsets = cp.asnumpy(d_seg_offsets)
        left_ids = []
        right_ids = []
        for r in range(ring_count):
            base = int(h_seg_offsets[r])
            count = int(h_ring_seg_counts[r])
            if count < 3:
                continue
            for i in range(count):
                for j in range(i + 2, count):
                    if i == 0 and j == count - 1:
                        continue
                    left_ids.append(base + i)
                    right_ids.append(base + j)
        if not left_ids:
            empty_i32 = cp.empty(0, dtype=cp.int32)
            empty_f64 = cp.empty(0, dtype=cp.float64)
            empty_i8 = cp.empty(0, dtype=cp.int8)
            return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64
        d_left_lookup = cp.asarray(np.asarray(left_ids, dtype=np.int32))
        d_right_lookup = cp.asarray(np.asarray(right_ids, dtype=np.int32))
        total_pairs = len(left_ids)

    pair_count = total_pairs

    # Use the existing classify_segment_pairs kernel from segment_primitives
    from vibespatial.spatial.segment_primitives import _segment_intersection_kernels

    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    block = (256, 1, 1)

    # d_left_lookup and d_right_lookup are already on device from pair generation above
    d_out_kind = cp.zeros(pair_count, dtype=cp.int8)
    d_out_px = cp.zeros(pair_count, dtype=cp.float64)
    d_out_py = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox1 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy1 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ambiguous = cp.zeros(pair_count, dtype=cp.uint8)

    seg_kernels = _segment_intersection_kernels()
    grid = (max(1, (pair_count + 255) // 256), 1, 1)
    runtime.launch(
        seg_kernels["classify_segment_pairs"],
        grid=grid, block=block,
        params=(
            (ptr(d_left_lookup), ptr(d_right_lookup),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_out_kind), ptr(d_out_px), ptr(d_out_py),
             ptr(d_out_ox0), ptr(d_out_oy0), ptr(d_out_ox1), ptr(d_out_oy1),
             ptr(d_out_ambiguous), pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    runtime.synchronize()

    # Filter hits on device (Tier 2: CuPy boolean mask + fancy indexing)
    d_hit_mask = d_out_kind != 0
    d_hit_indices = cp.flatnonzero(d_hit_mask)
    if d_hit_indices.size == 0:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    return (
        d_left_lookup[d_hit_indices],
        d_right_lookup[d_hit_indices],
        d_out_kind[d_hit_indices],
        d_out_px[d_hit_indices],
        d_out_py[d_hit_indices],
    )


def _split_self_intersections_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, bool]:
    """Detect and split self-intersections. Returns new x, y, ring_offsets, had_splits.

    Phases 8-9 GPU-resident: segment extraction via NVRTC, pair generation via
    NVRTC, dedup via CuPy mask, split counts via cp.bincount, ring sizes via
    segmented_reduce_sum.
    """
    # Phase A: Extract segments and detect intersections (Phase 8: GPU-resident)
    d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids = \
        _extract_ring_segments_gpu(d_x, d_y, d_ring_offsets, ring_count, kernels=kernels)
    total_segments = int(d_seg_x0.size)

    d_seg_a, d_seg_b, d_kinds, d_px, d_py = _detect_intra_ring_intersections(
        d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids,
        total_segments, ring_count, d_ring_offsets, kernels=kernels,
    )

    if d_seg_a.size == 0:
        return d_x, d_y, d_ring_offsets, False

    # Phase C: Count and scatter split events
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    block = (256, 1, 1)
    pair_count = int(d_seg_a.size)
    d_event_counts = cp.zeros(pair_count, dtype=cp.int32)

    grid = (max(1, (pair_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["count_self_split_events"],
        grid=grid, block=block,
        params=(
            (ptr(d_kinds), ptr(d_event_counts), pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    d_event_offsets = exclusive_sum(d_event_counts)
    total_events = count_scatter_total(runtime, d_event_counts, d_event_offsets)
    if total_events == 0:
        return d_x, d_y, d_ring_offsets, False

    d_out_seg_ids = cp.empty(total_events, dtype=cp.int32)
    d_out_t = cp.empty(total_events, dtype=cp.float64)
    d_out_x = cp.empty(total_events, dtype=cp.float64)
    d_out_y = cp.empty(total_events, dtype=cp.float64)
    d_out_key = cp.empty(total_events, dtype=cp.uint64)

    runtime.launch(
        kernels["scatter_self_split_events"],
        grid=grid, block=block,
        params=(
            (ptr(d_seg_a), ptr(d_seg_b), ptr(d_kinds), ptr(d_px), ptr(d_py),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_event_offsets),
             ptr(d_out_seg_ids), ptr(d_out_t), ptr(d_out_x), ptr(d_out_y),
             ptr(d_out_key), pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # Sort events by (segment_id, t) via packed key
    sorted_result = sort_pairs(d_out_key, d_out_seg_ids)
    d_sorted_keys = sorted_result.keys

    # Gather sorted t, x, y using the sorted order
    d_indices = cp.arange(total_events, dtype=cp.int32)
    idx_sort = sort_pairs(d_out_key, d_indices)
    d_perm = idx_sort.values
    d_perm_i64 = d_perm.astype(cp.int64)

    d_sorted_seg_ids = d_out_seg_ids[d_perm_i64]
    d_sorted_t = d_out_t[d_perm_i64]
    d_sorted_x = d_out_x[d_perm_i64]
    d_sorted_y = d_out_y[d_perm_i64]

    # --- Phase 9: GPU-resident dedup (CuPy mask instead of host loop) ---
    if total_events > 1:
        unique_result = unique_sorted_pairs(d_sorted_keys, d_sorted_seg_ids)
        unique_count = unique_result.count
        if unique_count < total_events:
            # CuPy device-side mask: keep first occurrence of each key
            d_mask = cp.ones(total_events, dtype=cp.bool_)
            d_mask[1:] = d_sorted_keys[1:] != d_sorted_keys[:-1]
            keep_idx = cp.flatnonzero(d_mask).astype(cp.int64)
            d_sorted_seg_ids = d_sorted_seg_ids[keep_idx]
            d_sorted_t = d_sorted_t[keep_idx]
            d_sorted_x = d_sorted_x[keep_idx]
            d_sorted_y = d_sorted_y[keep_idx]
            total_events = int(keep_idx.size)

    if total_events == 0:
        return d_x, d_y, d_ring_offsets, False

    # --- Phase 9: GPU-resident per-segment split counts (cp.bincount) ---
    d_seg_split_counts = cp.bincount(
        d_sorted_seg_ids, minlength=total_segments
    ).astype(cp.int32)
    d_seg_split_offsets = exclusive_sum(d_seg_split_counts)

    # --- Phase 9: GPU-resident ring size computation ---
    # ring_seg_offsets and ring_seg_counts via device ops
    d_ring_seg_counts = cp.bincount(d_seg_ring_ids, minlength=ring_count).astype(cp.int32)
    d_ring_seg_offsets = exclusive_sum(d_ring_seg_counts)

    # Per-ring extra splits = sum of seg_split_counts for segments in that ring
    # Use segmented_reduce_sum on d_seg_split_counts grouped by ring
    d_seg_starts = d_ring_seg_offsets.copy()
    d_seg_ends = d_seg_starts + d_ring_seg_counts
    seg_extra_result = segmented_reduce_sum(
        d_seg_split_counts.astype(cp.float64),
        d_seg_starts, d_seg_ends,
        num_segments=ring_count,
    )
    d_extra_per_ring = seg_extra_result.values.astype(cp.int32)

    # New ring size: n_segs original start vertices + extra splits + 1 closing vertex
    d_new_ring_sizes = d_ring_seg_counts + d_extra_per_ring + 1
    d_new_ring_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_ring_offsets[1:] = cp.cumsum(d_new_ring_sizes)
    total_new = int(d_new_ring_offsets[-1])

    # Launch rebuild kernel (all arrays already device-resident)
    d_new_x = cp.empty(total_new, dtype=cp.float64)
    d_new_y = cp.empty(total_new, dtype=cp.float64)

    ring_grid = (max(1, (ring_count + 255) // 256), 1, 1)
    runtime.launch(
        kernels["rebuild_ring_coords"],
        grid=ring_grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets),
             ptr(d_sorted_seg_ids), ptr(d_sorted_x), ptr(d_sorted_y), ptr(d_sorted_t),
             ptr(d_seg_split_offsets), ptr(d_seg_split_counts),
             ptr(d_new_ring_offsets), ptr(d_ring_seg_offsets),
             ptr(d_new_x), ptr(d_new_y),
             ring_count, total_segments),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )
    runtime.synchronize()
    return d_new_x, d_new_y, d_new_ring_offsets, True


# ---------------------------------------------------------------------------
# Phase D + E: Re-polygonization and output assembly
# ---------------------------------------------------------------------------

def _atomic_edges_from_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
) -> AtomicEdgeTable:
    """Build an AtomicEdgeTable from split ring coordinates (Part A).

    Bridges make_valid's split-ring output to overlay's reconstruction input.
    Each ring segment (consecutive vertex pair) becomes a directed edge pair
    (forward + reverse), matching the overlay pipeline's atomic edge format.

    All computation uses CuPy vectorised operations (Tier 2, fp64).
    """
    # Compute segments per ring: for a ring with n vertices, n-1 segments
    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_seg_counts)
    total_segments = int(d_seg_offsets[-1])

    if total_segments == 0:
        empty_d_i32 = cp.empty(0, dtype=cp.int32)
        empty_d_i8 = cp.empty(0, dtype=cp.int8)
        empty_d_f64 = cp.empty(0, dtype=cp.float64)
        return AtomicEdgeTable(
            left_segment_count=0, right_segment_count=0,
            runtime_selection=runtime_selection,
            device_state=AtomicEdgeDeviceState(
                source_segment_ids=empty_d_i32, direction=empty_d_i8,
                src_x=empty_d_f64, src_y=empty_d_f64,
                dst_x=empty_d_f64, dst_y=empty_d_f64,
            ),
            _count=0,
        )

    # Expand segments: for each segment, find its ring and local index
    d_seg_ids = cp.arange(total_segments, dtype=cp.int32)
    d_ring_of_seg = cp.searchsorted(
        d_seg_offsets[1:], d_seg_ids, side="right"
    ).astype(cp.int32)
    d_local_seg = d_seg_ids - d_seg_offsets[d_ring_of_seg]
    d_v = d_ring_offsets[d_ring_of_seg] + d_local_seg

    # Forward edges: src = v[i], dst = v[i+1]
    d_fwd_src_x = d_x[d_v]
    d_fwd_src_y = d_y[d_v]
    d_fwd_dst_x = d_x[d_v + 1]
    d_fwd_dst_y = d_y[d_v + 1]

    # Reverse edges: src = v[i+1], dst = v[i]
    d_rev_src_x = d_fwd_dst_x
    d_rev_src_y = d_fwd_dst_y
    d_rev_dst_x = d_fwd_src_x
    d_rev_dst_y = d_fwd_src_y

    # Interleave forward and reverse: edge 2k = forward, edge 2k+1 = reverse
    total_atomic = total_segments * 2
    d_src_x = cp.empty(total_atomic, dtype=cp.float64)
    d_src_y = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_x = cp.empty(total_atomic, dtype=cp.float64)
    d_dst_y = cp.empty(total_atomic, dtype=cp.float64)
    d_src_x[0::2] = d_fwd_src_x
    d_src_x[1::2] = d_rev_src_x
    d_src_y[0::2] = d_fwd_src_y
    d_src_y[1::2] = d_rev_src_y
    d_dst_x[0::2] = d_fwd_dst_x
    d_dst_x[1::2] = d_rev_dst_x
    d_dst_y[0::2] = d_fwd_dst_y
    d_dst_y[1::2] = d_rev_dst_y

    # Source segment IDs: forward and reverse share the same segment ID
    d_source_seg_ids = cp.empty(total_atomic, dtype=cp.int32)
    d_source_seg_ids[0::2] = d_seg_ids
    d_source_seg_ids[1::2] = d_seg_ids

    # Direction: +1 for forward, -1 for reverse
    d_direction = cp.empty(total_atomic, dtype=cp.int8)
    d_direction[0::2] = 1
    d_direction[1::2] = -1

    # Map segments to polygon indices via geom_offsets
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_ring_of_seg, side="right"
    ).astype(cp.int32)
    d_row_indices = cp.empty(total_atomic, dtype=cp.int32)
    d_row_indices[0::2] = d_poly_of_ring
    d_row_indices[1::2] = d_poly_of_ring

    # Ring indices
    d_ring_indices = cp.empty(total_atomic, dtype=cp.int32)
    d_ring_indices[0::2] = d_ring_of_seg
    d_ring_indices[1::2] = d_ring_of_seg

    # Source side: all left (1) since we have one geometry
    d_source_side = cp.ones(total_atomic, dtype=cp.int8)

    # Part indices: same as row indices for simple polygons
    d_part_indices = d_row_indices.copy()

    # Build AtomicEdgeTable with lazy host materialization.
    # The GPU path (build_gpu_half_edge_graph) only reads device_state,
    # count, left_segment_count, right_segment_count, and runtime_selection
    # — so the 6 coordinate/id D2H copies are skipped entirely on the hot
    # path.  Eagerly provide row/part/ring/side indices needed by the
    # HalfEdgeGraph constructor.
    h_row_indices = cp.asnumpy(d_row_indices)
    h_part_indices = cp.asnumpy(d_part_indices)
    h_ring_indices = cp.asnumpy(d_ring_indices)
    h_source_side = cp.asnumpy(d_source_side)

    return AtomicEdgeTable(
        left_segment_count=total_segments,
        right_segment_count=0,
        runtime_selection=runtime_selection,
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=d_source_seg_ids,
            direction=d_direction,
            src_x=d_src_x, src_y=d_src_y,
            dst_x=d_dst_x, dst_y=d_dst_y,
        ),
        _count=total_atomic,
        _row_indices=h_row_indices,
        _part_indices=h_part_indices,
        _ring_indices=h_ring_indices,
        _source_side=h_source_side,
    )


def _repolygonize_from_split_rings(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection | None = None,
) -> OwnedGeometryArray | None:
    """Re-polygonize from split ring coordinates using GPU overlay pipeline (Part B).

    Replaces the prior Shapely CPU polygonize with the fully GPU-parallel
    half-edge graph / face walk / face assembly from overlay_gpu.py.

    Returns an OwnedGeometryArray (device-resident per ADR-0005), or None
    if the GPU pipeline produces no valid faces.
    """
    if runtime_selection is None:
        runtime_selection = RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="make_valid GPU repolygonize",
        )

    # Step 1: Build AtomicEdgeTable from split ring coordinates
    atomic_edges = _atomic_edges_from_rings(
        d_x, d_y, d_ring_offsets, d_geom_offsets,
        ring_count, polygon_count, runtime_selection,
    )
    if atomic_edges.count == 0:
        return None

    # Step 2: Build half-edge graph
    half_edge_graph = build_gpu_half_edge_graph(atomic_edges)
    if half_edge_graph.edge_count == 0:
        return None

    # Step 3: Face walk to extract faces
    (face_offsets, face_edge_ids, bounded_mask, signed_area,
     centroid_x, centroid_y, label_x, label_y, face_count) = _gpu_face_walk(half_edge_graph)

    if face_count == 0:
        return None

    # Step 4: Select bounded positive-area faces (no coverage labeling needed)
    # For make_valid, we want all bounded faces with positive area — these
    # are the valid polygon faces extracted from the self-intersection split.
    d_select_mask = (bounded_mask != 0) & (signed_area > 0)
    selected_face_indices = cp.asnumpy(
        cp.flatnonzero(d_select_mask).astype(cp.int32)
    )

    if selected_face_indices.size == 0:
        return None

    # Build OverlayFaceTable for the assembly function
    faces = OverlayFaceTable(
        runtime_selection=runtime_selection,
        device_state=OverlayFaceDeviceState(
            face_offsets=face_offsets,
            face_edge_ids=face_edge_ids,
            bounded_mask=bounded_mask,
            signed_area=signed_area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            left_covered=cp.ones(face_count, dtype=cp.int8),
            right_covered=cp.zeros(face_count, dtype=cp.int8),
        ),
        _face_count=face_count,
    )

    # Step 5: Assemble output polygons from selected faces
    result = _build_polygon_output_from_faces_gpu(
        half_edge_graph, faces, selected_face_indices,
    )
    return result


def _build_repaired_owned(
    polygon_results: list[list[list[np.ndarray]]],
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray:
    """Build an OwnedGeometryArray from the list of repaired polygon results.

    Each entry in polygon_results corresponds to one input invalid row.
    Each entry is a list of polygon ring-lists (potentially multipolygon).
    """
    row_count = len(polygon_results)
    if row_count == 0:
        return OwnedGeometryArray(
            validity=np.asarray([], dtype=bool),
            tags=np.asarray([], dtype=np.int8),
            family_row_offsets=np.asarray([], dtype=np.int32),
            families={},
            residency=Residency.HOST,
            runtime_history=[runtime_selection],
        )

    validity = np.ones(row_count, dtype=bool)
    tags = np.full(row_count, -1, dtype=np.int8)
    family_row_offsets = np.full(row_count, -1, dtype=np.int32)

    poly_x: list[float] = []
    poly_y: list[float] = []
    poly_geom_offsets: list[int] = []
    poly_ring_offsets: list[int] = []
    poly_bounds: list[tuple[float, float, float, float]] = []
    poly_count = 0

    mp_x: list[float] = []
    mp_y: list[float] = []
    mp_geom_offsets: list[int] = []
    mp_part_offsets: list[int] = []
    mp_ring_offsets: list[int] = []
    mp_bounds: list[tuple[float, float, float, float]] = []
    mp_count = 0

    for row_idx, polys in enumerate(polygon_results):
        if not polys:
            validity[row_idx] = False
            continue

        if len(polys) == 1:
            # Single polygon
            tags[row_idx] = FAMILY_TAGS[GeometryFamily.POLYGON]
            family_row_offsets[row_idx] = poly_count
            poly_geom_offsets.append(len(poly_ring_offsets))
            rings = polys[0]
            all_x = []
            all_y = []
            for ring in rings:
                coords = np.asarray(ring, dtype=np.float64)
                if coords.shape[0] > 0 and (
                    coords[0, 0] != coords[-1, 0] or coords[0, 1] != coords[-1, 1]
                ):
                    coords = np.vstack((coords, coords[:1]))
                poly_ring_offsets.append(len(poly_x))
                for c in coords:
                    poly_x.append(float(c[0]))
                    poly_y.append(float(c[1]))
                    all_x.append(float(c[0]))
                    all_y.append(float(c[1]))
            if all_x:
                poly_bounds.append((min(all_x), min(all_y), max(all_x), max(all_y)))
            else:
                poly_bounds.append((0.0, 0.0, 0.0, 0.0))
            poly_count += 1
        else:
            # Multipolygon
            tags[row_idx] = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
            family_row_offsets[row_idx] = mp_count
            mp_geom_offsets.append(len(mp_part_offsets))
            min_x = np.inf
            min_y = np.inf
            max_x = -np.inf
            max_y = -np.inf
            for rings in polys:
                mp_part_offsets.append(len(mp_ring_offsets))
                for ring in rings:
                    coords = np.asarray(ring, dtype=np.float64)
                    if coords.shape[0] > 0 and (
                        coords[0, 0] != coords[-1, 0] or coords[0, 1] != coords[-1, 1]
                    ):
                        coords = np.vstack((coords, coords[:1]))
                    mp_ring_offsets.append(len(mp_x))
                    for c in coords:
                        mp_x.append(float(c[0]))
                        mp_y.append(float(c[1]))
                        min_x = min(min_x, float(c[0]))
                        min_y = min(min_y, float(c[1]))
                        max_x = max(max_x, float(c[0]))
                        max_y = max(max_y, float(c[1]))
            mp_bounds.append((min_x, min_y, max_x, max_y))
            mp_count += 1

    families: dict[GeometryFamily, FamilyGeometryBuffer] = {}
    if poly_count > 0:
        families[GeometryFamily.POLYGON] = FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=poly_count,
            x=np.asarray(poly_x, dtype=np.float64) if poly_x else np.empty(0, dtype=np.float64),
            y=np.asarray(poly_y, dtype=np.float64) if poly_y else np.empty(0, dtype=np.float64),
            geometry_offsets=np.asarray(
                [*poly_geom_offsets, len(poly_ring_offsets)], dtype=np.int32
            ),
            empty_mask=np.zeros(poly_count, dtype=bool),
            ring_offsets=np.asarray(
                [*poly_ring_offsets, len(poly_x)], dtype=np.int32
            ),
            bounds=np.asarray(poly_bounds, dtype=np.float64) if poly_bounds else None,
        )
    if mp_count > 0:
        families[GeometryFamily.MULTIPOLYGON] = FamilyGeometryBuffer(
            family=GeometryFamily.MULTIPOLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.MULTIPOLYGON),
            row_count=mp_count,
            x=np.asarray(mp_x, dtype=np.float64) if mp_x else np.empty(0, dtype=np.float64),
            y=np.asarray(mp_y, dtype=np.float64) if mp_y else np.empty(0, dtype=np.float64),
            geometry_offsets=np.asarray(
                [*mp_geom_offsets, len(mp_part_offsets)], dtype=np.int32
            ),
            empty_mask=np.zeros(mp_count, dtype=bool),
            part_offsets=np.asarray(
                [*mp_part_offsets, len(mp_ring_offsets)], dtype=np.int32
            ),
            ring_offsets=np.asarray(
                [*mp_ring_offsets, len(mp_x)], dtype=np.int32
            ),
            bounds=np.asarray(mp_bounds, dtype=np.float64) if mp_bounds else None,
        )

    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
        runtime_history=[runtime_selection],
    )


def _build_single_polygon_owned(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray | None:
    """Build an OwnedGeometryArray for a single polygon from device coords.

    Transfers coordinates and ring offsets to host in bulk (no per-ring
    Python loop).  Returns None if the polygon has insufficient vertices.
    """
    h_x = cp.asnumpy(d_x).astype(np.float64, copy=False)
    h_y = cp.asnumpy(d_y).astype(np.float64, copy=False)
    h_ro = cp.asnumpy(d_ring_offsets).astype(np.int32, copy=False)

    # Vectorised ring-length filter: keep rings with >= 4 vertices
    ring_lens = np.diff(h_ro)
    valid_rings = ring_lens >= 4
    if not np.any(valid_rings):
        return None

    # If all rings are valid, skip reindex
    if np.all(valid_rings):
        ring_offsets = h_ro
        x = h_x
        y = h_y
    else:
        # Filter rings without a per-ring Python loop
        keep_idx = np.flatnonzero(valid_rings)
        new_ring_count = int(keep_idx.size)
        starts = h_ro[keep_idx]
        ends = h_ro[keep_idx + 1]
        new_lens = ends - starts
        new_ring_offsets = np.empty(new_ring_count + 1, dtype=np.int32)
        new_ring_offsets[0] = 0
        np.cumsum(new_lens, out=new_ring_offsets[1:])
        total_verts = int(new_ring_offsets[-1])
        x = np.empty(total_verts, dtype=np.float64)
        y = np.empty(total_verts, dtype=np.float64)
        for i, (s, e) in enumerate(zip(starts, ends)):
            dst_s = int(new_ring_offsets[i])
            dst_e = int(new_ring_offsets[i + 1])
            x[dst_s:dst_e] = h_x[s:e]
            y[dst_s:dst_e] = h_y[s:e]
        ring_offsets = new_ring_offsets
        ring_count = new_ring_count

    if x.size == 0:
        return None

    bounds = np.array([[x.min(), y.min(), x.max(), y.max()]], dtype=np.float64)
    families = {
        GeometryFamily.POLYGON: FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=1,
            x=x,
            y=y,
            geometry_offsets=np.array([0, ring_count], dtype=np.int32),
            empty_mask=np.zeros(1, dtype=bool),
            ring_offsets=ring_offsets,
            bounds=bounds,
        ),
    }
    return OwnedGeometryArray(
        validity=np.array([True]),
        tags=np.array([FAMILY_TAGS[GeometryFamily.POLYGON]], dtype=np.uint8),
        family_row_offsets=np.array([0], dtype=np.int32),
        families=families,
        residency=Residency.HOST,
        runtime_history=[runtime_selection],
    )


# ---------------------------------------------------------------------------
# Main GPU repair entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPURepairResult:
    """Result of GPU make_valid repair."""
    repaired_geometries: np.ndarray  # shapely geometry array
    repaired_count: int
    gpu_phases_used: tuple[str, ...]


def _owned_to_shapely_geom(owned_result: OwnedGeometryArray) -> object | None:
    """Convert an OwnedGeometryArray (from overlay pipeline) to a shapely geometry.

    Returns a single Polygon or MultiPolygon, or None if the result is empty.
    """
    import shapely as shp
    try:
        geoms = owned_result.to_shapely()
        if geoms is None or len(geoms) == 0:
            return None
        # Collect all non-None polygons
        polys = []
        for g in geoms:
            if g is None or g.is_empty:
                continue
            if isinstance(g, shp.MultiPolygon):
                polys.extend(g.geoms)
            elif isinstance(g, shp.Polygon):
                polys.append(g)
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        return shp.MultiPolygon(polys)
    except Exception:
        return None


def gpu_repair_invalid_polygons(
    owned: OwnedGeometryArray,
    invalid_rows: np.ndarray,
    geometries: np.ndarray,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
) -> GPURepairResult | None:
    """GPU-resident repair of invalid polygon geometries (Phase 10).

    Implements the full make_valid pipeline on GPU:
    Phase B: Close rings, remove duplicates, fix orientation (per-polygon, device-resident)
    Phase A+C: Detect and split self-intersections (per-polygon, device-resident)
    Phase D: Re-polygonize via overlay half-edge/face-walk pipeline (per-polygon)

    Returns None if GPU repair is not applicable (no GPU, no polygon families,
    or CuPy not available). Falls back to Shapely for non-polygon families.

    Parameters
    ----------
    owned : OwnedGeometryArray with device_state
    invalid_rows : indices of invalid rows to repair
    geometries : shapely geometry array for all rows
    method : repair method (only "linework" supported on GPU)
    keep_collapsed : whether to keep collapsed geometries
    """
    if cp is None:
        return None

    from vibespatial.runtime import has_gpu_runtime
    if not has_gpu_runtime():
        return None

    if invalid_rows.size == 0:
        return GPURepairResult(
            repaired_geometries=geometries.copy(),
            repaired_count=0,
            gpu_phases_used=(),
        )

    # Only repair polygon family rows on GPU; non-polygon invalids use Shapely
    polygon_families = set()
    for family_name in ("polygon", "multipolygon"):
        if family_name in owned.families:
            polygon_families.add(family_name)

    if not polygon_families:
        return None

    # Compile repair kernels
    try:
        kernels = _compile_repair_kernels()
    except Exception:
        return None

    runtime_sel = RuntimeSelection(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        reason="make_valid GPU repair pipeline (Phase 10)",
    )

    result = geometries.copy()
    phases_used = []
    gpu_repaired_count = 0

    for family_name in polygon_families:
        buffer = owned.families[family_name]
        if not hasattr(buffer, "ring_offsets") or buffer.ring_offsets is None:
            continue

        ring_offsets = np.asarray(buffer.ring_offsets, dtype=np.int32)
        ring_count = len(ring_offsets) - 1
        if ring_count <= 0:
            continue

        geom_offsets = np.asarray(buffer.geometry_offsets, dtype=np.int32)
        polygon_count = len(geom_offsets) - 1
        if polygon_count <= 0:
            continue

        # Map invalid rows to family rows
        family_tag = FAMILY_TAGS[family_name]
        family_invalid_mask = np.zeros(polygon_count, dtype=bool)
        for row_idx in invalid_rows:
            if owned.tags[row_idx] == family_tag:
                fro = int(owned.family_row_offsets[row_idx])
                if 0 <= fro < polygon_count:
                    family_invalid_mask[fro] = True

        invalid_family_rows = np.flatnonzero(family_invalid_mask)
        if invalid_family_rows.size == 0:
            continue

        # Upload coordinate data to GPU once for all invalid polygons
        d_x = cp.asarray(np.ascontiguousarray(buffer.x, dtype=np.float64))
        d_y = cp.asarray(np.ascontiguousarray(buffer.y, dtype=np.float64))
        d_ring_offsets = cp.asarray(ring_offsets)

        # --- Part C: Per-polygon processing, data stays on device ---
        for fam_row in invalid_family_rows:
            r_start = int(geom_offsets[fam_row])
            r_end = int(geom_offsets[fam_row + 1])
            local_ring_count = r_end - r_start
            if local_ring_count == 0:
                continue

            coord_start = int(ring_offsets[r_start])
            coord_end = int(ring_offsets[r_end])

            # Build local offsets on device (no host round-trip for offsets)
            d_local_ring_offsets = d_ring_offsets[r_start:r_end + 1] - coord_start
            d_local_geom_offsets = cp.asarray(
                np.asarray([0, local_ring_count], dtype=np.int32)
            )

            d_local_x = d_x[coord_start:coord_end].copy()
            d_local_y = d_y[coord_start:coord_end].copy()

            try:
                # Phase B: Close rings (device-resident)
                d_local_x, d_local_y, d_local_ring_offsets = _gpu_close_rings(
                    d_local_x, d_local_y, d_local_ring_offsets,
                    local_ring_count, kernels,
                )
                phases_used.append("close_rings")

                # Phase B: Remove duplicate vertices (device-resident)
                d_local_x, d_local_y, d_local_ring_offsets = _gpu_remove_duplicate_vertices(
                    d_local_x, d_local_y, d_local_ring_offsets,
                    local_ring_count, kernels,
                )
                phases_used.append("remove_duplicates")

                # Phase B: Fix ring orientation (device-resident)
                d_local_x, d_local_y = _gpu_fix_ring_orientation(
                    d_local_x, d_local_y, d_local_ring_offsets,
                    d_local_geom_offsets, local_ring_count, 1, kernels,
                )
                phases_used.append("fix_orientation")

                # Phase A+C: Detect and split self-intersections
                d_local_x, d_local_y, d_local_ring_offsets, had_splits = \
                    _split_self_intersections_gpu(
                        d_local_x, d_local_y, d_local_ring_offsets,
                        local_ring_count, kernels,
                    )
                if had_splits:
                    phases_used.append("split_intersections")

                # Phase D: Re-polygonize via overlay pipeline (Part B+D)
                if had_splits:
                    owned_result = _repolygonize_from_split_rings(
                        d_local_x, d_local_y, d_local_ring_offsets,
                        d_local_geom_offsets, local_ring_count, 1,
                        runtime_selection=runtime_sel,
                    )
                    phases_used.append("repolygonize")

                    if owned_result is not None:
                        geom = _owned_to_shapely_geom(owned_result)
                        if geom is not None and geom.is_valid:
                            global_row = _family_row_to_global(
                                owned, family_name, fam_row,
                            )
                            if global_row is not None and global_row in invalid_rows:
                                result[global_row] = geom
                                gpu_repaired_count += 1
                                continue
                else:
                    # No self-intersections — rebuild from repaired device
                    # coords.  Build an OwnedGeometryArray from the device
                    # ring data and materialise via _owned_to_shapely_geom
                    # (no per-ring Python loop on the hot path).
                    owned_result = _build_single_polygon_owned(
                        d_local_x, d_local_y, d_local_ring_offsets,
                        local_ring_count, runtime_sel,
                    )
                    if owned_result is not None:
                        geom = _owned_to_shapely_geom(owned_result)
                        if geom is not None and geom.is_valid:
                            global_row = _family_row_to_global(
                                owned, family_name, fam_row,
                            )
                            if global_row is not None and global_row in invalid_rows:
                                result[global_row] = geom
                                gpu_repaired_count += 1
                                continue

            except Exception:
                # GPU repair failed for this polygon; will fall through to Shapely
                pass

    # For any rows not yet repaired, fall back to Shapely
    import shapely as shp
    still_invalid = []
    for row_idx in invalid_rows:
        if result[row_idx] is geometries[row_idx]:
            still_invalid.append(row_idx)

    if still_invalid:
        still_invalid_arr = np.asarray(still_invalid)
        result[still_invalid_arr] = shp.make_valid(
            geometries[still_invalid_arr],
            method=method,
            keep_collapsed=keep_collapsed,
        )

    return GPURepairResult(
        repaired_geometries=result,
        repaired_count=gpu_repaired_count,
        gpu_phases_used=tuple(set(phases_used)),
    )


def _family_row_to_global(
    owned: OwnedGeometryArray,
    family_name: str,
    family_row: int,
) -> int | None:
    """Map a family row index back to a global row index."""
    family_tag = FAMILY_TAGS[family_name]
    matches = np.flatnonzero(
        (owned.tags == family_tag) & (owned.family_row_offsets == family_row) & owned.validity
    )
    if matches.size > 0:
        return int(matches[0])
    return None
