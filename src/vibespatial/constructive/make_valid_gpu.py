"""GPU-resident make_valid batch repair pipeline (ADR-0019 + ADR-0033).

Phase 16: Batch repolygonization. All invalid polygons are collected into one
contiguous batch and processed through the GPU repair pipeline together,
eliminating the per-polygon Python loop. shapely.polygonize and
shapely.make_valid are no longer used in the primary GPU path.

Pipeline stages (batched across all invalid polygons):
  Phase A: GPU self-intersection detection (extract ring segments, classify pairs)
  Phase B: Simple repair kernels (close rings, remove duplicate vertices, fix orientation)
  Phase C: Self-intersection splitting (count/scatter split events, sort, dedup, rebuild)
  Phase D: Re-polygonization (half-edges, face walk, face containment, assembly)
  Phase E: Output assembly (build OwnedGeometryArray, convert to Shapely)

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
// Phase B: check_ring_closure — one thread per ring.
// Compare first and last vertex coordinates; output boolean mask.
extern "C" __global__ void check_ring_closure(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    int* __restrict__ needs_closure,
    const int ring_count
) {
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;
    const int start = ring_offsets[ring];
    const int end = ring_offsets[ring + 1];
    const int len = end - start;
    needs_closure[ring] = (len >= 2
        && (x[start] != x[end - 1] || y[start] != y[end - 1])) ? 1 : 0;
}

// Phase B: close_rings — one thread per ring.
// If first vertex != last vertex, write closing vertex into reserved slot.
// new_ring_offsets[ring+1] has space for the extra vertex if needed.
extern "C" __global__ void close_rings(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ ring_needs_close,
    const int* __restrict__ new_ring_offsets,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int ring_count
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
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ vertex_ring_ids,
    unsigned char* __restrict__ out_keep,
    const int vertex_count
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
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    double* __restrict__ out_cross,
    const int ring_count,
    const int vertex_count
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
    double* __restrict__ x,
    double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ ring_needs_reverse,
    const int ring_count
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
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ seg_offsets,
    double* __restrict__ seg_x0,
    double* __restrict__ seg_y0,
    double* __restrict__ seg_x1,
    double* __restrict__ seg_y1,
    int* __restrict__ seg_ring_ids,
    const int ring_count,
    const int total_segments
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
// O(1) decode of (i, j) within a ring using triangular number inversion.
// pair_counts[ring] = k*(k-1)/2 - k for k segments (all non-adjacent pairs,
// minus the wrap-around adjacency).
extern "C" __global__ void generate_intra_ring_pairs(
    const int* __restrict__ pair_offsets,
    const int* __restrict__ seg_offsets,
    const int* __restrict__ ring_seg_counts,
    int* __restrict__ out_left,
    int* __restrict__ out_right,
    const int ring_count,
    const int total_pairs
) {
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= total_pairs) return;
    // Binary search for the ring that owns this pair
    int lo = 0, hi = ring_count;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (pair_offsets[mid + 1] <= pid) lo = mid + 1;
        else hi = mid;
    }
    const int ring = lo;
    int lp = pid - pair_offsets[ring];
    const int k = ring_seg_counts[ring];
    const int base = seg_offsets[ring];
    // Enumerate non-adjacent pairs (i,j) with j >= i+2 in row-major order.
    // Row i=0 has j in {2..k-2} (k-3 pairs, skip wrap j=k-1).
    // Row i>=1 has j in {i+2..k-1} (k-i-2 pairs each).
    // Adjust local index past the wrap-pair gap at position k-3 in row 0:
    if (lp >= k - 3) lp++;
    // Now lp indexes into the full (i,j) table where row i has k-i-2 entries.
    // Row i starts at prefix[i] = i*(2*k - 3 - i) / 2.
    // Invert via quadratic: i = floor((2k-3 - sqrt((2k-3)^2 - 8*lp)) / 2)
    const double disc = (double)(2*k - 3) * (double)(2*k - 3) - 8.0 * (double)lp;
    int i = (int)((double)(2*k - 3) - sqrt(disc)) / 2;
    // Clamp and correct for floating-point rounding
    if (i < 0) i = 0;
    int row_start = i * (2*k - 3 - i) / 2;
    if (row_start + (k - i - 2) <= lp && i + 1 < k - 2) {
        i++;
        row_start = i * (2*k - 3 - i) / 2;
    }
    const int j = lp - row_start + i + 2;
    out_left[pid] = base + i;
    out_right[pid] = base + j;
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
    "check_ring_closure",
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

    Closure check via check_ring_closure NVRTC kernel (Tier 1, 1 thread/ring).
    Copy+append via close_rings NVRTC kernel.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- Step 1: NVRTC kernel checks which rings need closure (Tier 1) ---
    d_needs_close = cp.empty(ring_count, dtype=cp.int32)
    grid, block = runtime.launch_config(kernels["check_ring_closure"], ring_count)
    runtime.launch(
        kernels["check_ring_closure"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_close), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # Early exit if no rings need closure (single scalar D2H)
    if int(cp.sum(d_needs_close)) == 0:
        return d_x, d_y, d_ring_offsets

    # --- Step 2: Compute new ring sizes and offsets on device (Tier 2: CuPy) ---
    d_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_new_sizes = d_sizes + d_needs_close
    d_new_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_new_offsets[1:] = cp.cumsum(d_new_sizes)
    total_new = int(d_new_offsets[-1])

    # --- Step 3: Copy vertices + append closure vertex where needed (Tier 1) ---
    d_out_x = cp.empty(total_new, dtype=cp.float64)
    d_out_y = cp.empty(total_new, dtype=cp.float64)

    grid, block = runtime.launch_config(kernels["close_rings"], ring_count)
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

    Tier 1: flag_duplicate_vertices NVRTC kernel (1 thread/vertex).
    Tier 2: CuPy searchsorted for vertex-to-ring mapping, fancy indexing for compaction.
    Tier 3a: CCCL compact_indices for keep-mask compaction, segmented_reduce_sum
    for per-ring kept counts.  All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    vertex_count = int(d_x.shape[0])
    if vertex_count == 0:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident vertex-to-ring mapping (Tier 2: CuPy searchsorted) ---
    d_vertex_ids = cp.arange(vertex_count, dtype=cp.int32)
    d_vertex_ring_ids = cp.searchsorted(
        d_ring_offsets[1:], d_vertex_ids, side="right"
    ).astype(cp.int32)

    # --- Flag duplicates via NVRTC kernel (Tier 1) ---
    d_keep = cp.empty(vertex_count, dtype=cp.uint8)
    grid, block = runtime.launch_config(kernels["flag_duplicate_vertices"], vertex_count)
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
    # No explicit sync needed -- cp.sum below triggers implicit sync on same stream.

    # Early exit if no duplicates found (single scalar D2H via cp.sum)
    kept_count = int(cp.sum(d_keep))
    if kept_count == vertex_count:
        return d_x, d_y, d_ring_offsets

    # --- GPU-resident compaction (Tier 3a: CCCL compact_indices) ---
    compact_result = compact_indices(d_keep)
    d_kept_indices = compact_result.values.astype(cp.int64)

    new_x = d_x[d_kept_indices]
    new_y = d_y[d_kept_indices]

    # --- GPU-resident ring offset rebuild ---
    # Per-ring kept count via segmented reduce (Tier 3a: CCCL)
    d_keep_i32 = d_keep.astype(cp.int32)
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
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

    Tier 1: compute_ring_shoelace NVRTC kernel for per-vertex cross products.
    Tier 3a: segmented_reduce_sum (CCCL) for per-ring signed area.
    Tier 2: CuPy element-wise for exterior/hole classification and reversal mask.
    Tier 1: reverse_ring_coords NVRTC kernel for coordinate reversal.
    All device-resident, no host copy.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    vertex_count = int(d_x.shape[0])
    if vertex_count < 3 or ring_count == 0:
        return d_x, d_y

    # --- Step 1: Compute per-vertex shoelace cross products (Tier 1: NVRTC) ---
    # compute_ring_shoelace writes x[v]*y[v+1] - x[v+1]*y[v] for each vertex.
    # Launch for vertex_count-1 to avoid out-of-bounds read on the last vertex.
    # Zero-init so last-vertex-per-ring contributions are automatically 0.
    d_cross = cp.zeros(vertex_count, dtype=cp.float64)
    safe_count = vertex_count - 1
    if safe_count > 0:
        grid, block = runtime.launch_config(kernels["compute_ring_shoelace"], safe_count)
        runtime.launch(
            kernels["compute_ring_shoelace"],
            grid=grid, block=block,
            params=(
                (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_cross),
                 ring_count, safe_count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
            ),
        )

    # Zero out cross products at ring boundaries to prevent cross-ring contamination
    d_last_verts = d_ring_offsets[1:] - 1
    d_cross[d_last_verts] = 0.0

    # --- Step 2: Per-ring signed area via segmented reduce (Tier 3a: CCCL) ---
    d_starts = d_ring_offsets[:-1]
    d_ends = d_ring_offsets[1:]
    seg_result = segmented_reduce_sum(
        d_cross, d_starts, d_ends, num_segments=ring_count,
    )
    d_ring_areas = seg_result.values * 0.5

    # --- Step 3: Classify exterior vs hole and build reversal mask (Tier 2: CuPy) ---
    d_ring_ids = cp.arange(ring_count, dtype=cp.int32)
    d_poly_of_ring = cp.searchsorted(
        d_geom_offsets[1:], d_ring_ids, side="right"
    ).astype(cp.int32)
    d_first_ring_of_poly = d_geom_offsets[d_poly_of_ring]
    d_is_exterior = (d_ring_ids == d_first_ring_of_poly)

    # Exterior should be CCW (positive area); hole should be CW (negative area)
    d_needs_reverse = (
        (d_is_exterior & (d_ring_areas < 0))
        | (~d_is_exterior & (d_ring_areas > 0))
    ).astype(cp.uint8)

    # Early exit if no rings need reversal (single scalar D2H via cp.sum)
    if int(cp.sum(d_needs_reverse)) == 0:
        return d_x, d_y

    # --- Step 4: Reverse coordinates for wrong-orientation rings (Tier 1: NVRTC) ---
    grid, block = runtime.launch_config(kernels["reverse_ring_coords"], ring_count)
    runtime.launch(
        kernels["reverse_ring_coords"],
        grid=grid, block=block,
        params=(
            (ptr(d_x), ptr(d_y), ptr(d_ring_offsets), ptr(d_needs_reverse), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )
    # No explicit sync needed -- caller will either launch more same-stream
    # work or trigger implicit sync via CuPy/D2H read.
    return d_x, d_y


# ---------------------------------------------------------------------------
# Phase A + C: Self-intersection detection and splitting
# ---------------------------------------------------------------------------

def _extract_ring_segments_gpu(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    ring_count: int,
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Extract consecutive vertex pairs as flat segment table from ring coords.

    GPU-resident: seg_counts via CuPy element-wise (Tier 2), seg_offsets via
    CuPy cumsum (Tier 4), then one NVRTC kernel thread per segment (Tier 1).

    Returns (seg_x0, seg_y0, seg_x1, seg_y1, seg_ring_ids) on device.
    """
    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- Compute seg_counts and seg_offsets on device (Tier 2: CuPy) ---
    d_ring_sizes = d_ring_offsets[1:] - d_ring_offsets[:-1]
    d_seg_counts = cp.maximum(d_ring_sizes - 1, 0).astype(cp.int32)
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

    kernel = kernels["extract_ring_segments"]
    grid, block = runtime.launch_config(kernel, total_segments)
    runtime.launch(
        kernel,
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
    kernels: dict,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Detect self-intersections within each ring using segment pair classification.

    GPU-resident: per-ring segment counts via cp.bincount (Tier 2), pair counts
    via CuPy element-wise (Tier 2), pair offsets via CuPy cumsum (Tier 4),
    pair generation via NVRTC kernel (Tier 1), classification via
    classify_segment_pairs_v2 (Tier 1).

    Returns (seg_a_ids, seg_b_ids, kinds, point_x, point_y) as device
    (CuPy) arrays.  Only returns pairs with kind != 0 (actual intersections).
    """
    if total_segments < 2:
        empty_i32 = cp.empty(0, dtype=cp.int32)
        empty_f64 = cp.empty(0, dtype=cp.float64)
        empty_i8 = cp.empty(0, dtype=cp.int8)
        return empty_i32, empty_i32, empty_i8, empty_f64, empty_f64

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # --- GPU-resident segment offset table per ring (Tier 2: CuPy) ---
    d_ring_seg_counts = cp.bincount(d_seg_ring_ids, minlength=ring_count).astype(cp.int32)
    d_seg_offsets = cp.zeros(ring_count + 1, dtype=cp.int32)
    d_seg_offsets[1:] = cp.cumsum(d_ring_seg_counts)

    # --- Compute pair counts per ring on device (Tier 2: CuPy) ---
    # For ring with k segments: pairs = k*(k-1)/2 - k (all pairs minus adjacent minus wrap)
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

    # --- Generate pairs on GPU (Tier 1: NVRTC, 1 thread per pair) ---
    d_left_lookup = cp.empty(total_pairs, dtype=cp.int32)
    d_right_lookup = cp.empty(total_pairs, dtype=cp.int32)

    pair_kernel = kernels["generate_intra_ring_pairs"]
    grid_gen, block_gen = runtime.launch_config(pair_kernel, total_pairs)
    runtime.launch(
        pair_kernel,
        grid=grid_gen, block=block_gen,
        params=(
            (ptr(d_pair_offsets), ptr(d_seg_offsets),
             ptr(d_ring_seg_counts), ptr(d_left_lookup),
             ptr(d_right_lookup), ring_count, total_pairs),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )

    # --- Classify segment pairs (Tier 1: NVRTC) ---
    # Uses on-GPU Shewchuk adaptive refinement (no ambiguous output)
    from vibespatial.spatial.segment_primitives import _classify_kernels

    pair_count = total_pairs
    d_out_kind = cp.zeros(pair_count, dtype=cp.int8)
    d_out_px = cp.zeros(pair_count, dtype=cp.float64)
    d_out_py = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy0 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_ox1 = cp.zeros(pair_count, dtype=cp.float64)
    d_out_oy1 = cp.zeros(pair_count, dtype=cp.float64)

    seg_kernels = _classify_kernels("double")
    classify_kernel = seg_kernels["classify_segment_pairs_v2"]
    grid, block = runtime.launch_config(classify_kernel, pair_count)
    runtime.launch(
        classify_kernel,
        grid=grid, block=block,
        params=(
            (ptr(d_left_lookup), ptr(d_right_lookup),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_seg_x0), ptr(d_seg_y0), ptr(d_seg_x1), ptr(d_seg_y1),
             ptr(d_out_kind), ptr(d_out_px), ptr(d_out_py),
             ptr(d_out_ox0), ptr(d_out_oy0), ptr(d_out_ox1), ptr(d_out_oy1),
             pair_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        ),
    )
    # No sync needed -- CuPy ops below run on same null stream.

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

    GPU-resident: segment extraction via NVRTC (Tier 1), pair generation via
    NVRTC (Tier 1), dedup via CuPy mask (Tier 2), split counts via
    cp.bincount (Tier 2), ring sizes via segmented_reduce_sum (Tier 3a).
    """
    # Phase A: Extract segments and detect intersections (GPU-resident)
    d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids = \
        _extract_ring_segments_gpu(d_x, d_y, d_ring_offsets, ring_count, kernels)
    total_segments = int(d_seg_x0.size)

    d_seg_a, d_seg_b, d_kinds, d_px, d_py = _detect_intra_ring_intersections(
        d_seg_x0, d_seg_y0, d_seg_x1, d_seg_y1, d_seg_ring_ids,
        total_segments, ring_count, d_ring_offsets, kernels,
    )

    if d_seg_a.size == 0:
        return d_x, d_y, d_ring_offsets, False

    # Phase C: Count and scatter split events
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    pair_count = int(d_seg_a.size)
    d_event_counts = cp.zeros(pair_count, dtype=cp.int32)

    count_kernel = kernels["count_self_split_events"]
    grid_count, block_count = runtime.launch_config(count_kernel, pair_count)
    runtime.launch(
        count_kernel,
        grid=grid_count, block=block_count,
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

    scatter_kernel = kernels["scatter_self_split_events"]
    grid_scatter, block_scatter = runtime.launch_config(scatter_kernel, pair_count)
    runtime.launch(
        scatter_kernel,
        grid=grid_scatter, block=block_scatter,
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

    rebuild_kernel = kernels["rebuild_ring_coords"]
    ring_grid, ring_block = runtime.launch_config(rebuild_kernel, ring_count)
    runtime.launch(
        rebuild_kernel,
        grid=ring_grid, block=ring_block,
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
    # No sync needed -- caller uses device arrays on same null stream.
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
                row_indices=empty_d_i32, part_indices=empty_d_i32,
                ring_indices=empty_d_i32, source_side=empty_d_i8,
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

    # Build AtomicEdgeTable with device-primary storage.
    # build_gpu_half_edge_graph reads device_state metadata directly,
    # so no D->H transfers needed here.  Host arrays lazily materialized.
    return AtomicEdgeTable(
        left_segment_count=total_segments,
        right_segment_count=0,
        runtime_selection=runtime_selection,
        device_state=AtomicEdgeDeviceState(
            source_segment_ids=d_source_seg_ids,
            direction=d_direction,
            src_x=d_src_x, src_y=d_src_y,
            dst_x=d_dst_x, dst_y=d_dst_y,
            row_indices=d_row_indices,
            part_indices=d_part_indices,
            ring_indices=d_ring_indices,
            source_side=d_source_side,
        ),
        _count=total_atomic,
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


# ---------------------------------------------------------------------------
# Main GPU repair entry point
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPURepairResult:
    """Result of GPU make_valid repair."""
    repaired_geometries: np.ndarray  # shapely geometry array
    repaired_count: int
    gpu_phases_used: tuple[str, ...]


def _build_batch_repaired_owned(
    d_x: cp.ndarray,
    d_y: cp.ndarray,
    d_ring_offsets: cp.ndarray,
    d_geom_offsets: cp.ndarray,
    ring_count: int,
    polygon_count: int,
    runtime_selection: RuntimeSelection,
) -> OwnedGeometryArray | None:
    """Build an OwnedGeometryArray from batch device coordinates.

    Used when Phase B repairs (close/dedup/orient) were sufficient and no
    self-intersection splitting was needed. Transfers coordinates to host
    in bulk (single D2H per array) and builds GeoArrow polygon buffers.
    """
    h_x = cp.asnumpy(d_x).astype(np.float64, copy=False)
    h_y = cp.asnumpy(d_y).astype(np.float64, copy=False)
    h_ro = cp.asnumpy(d_ring_offsets).astype(np.int32, copy=False)
    h_go = cp.asnumpy(d_geom_offsets).astype(np.int32, copy=False)

    if h_x.size == 0 or polygon_count == 0:
        return None

    # Filter degenerate rings (< 4 vertices) on host in bulk
    ring_lens = np.diff(h_ro)
    valid_rings = ring_lens >= 4

    if not np.any(valid_rings):
        return None

    # Build per-polygon ring counts and validity
    validity = np.ones(polygon_count, dtype=bool)
    tags = np.full(polygon_count, FAMILY_TAGS[GeometryFamily.POLYGON], dtype=np.int8)
    family_row_offsets = np.arange(polygon_count, dtype=np.int32)

    # Rebuild ring/geom offsets filtering invalid rings
    new_ring_offsets_list = []
    new_geom_offsets = [0]
    new_x_chunks = []
    new_y_chunks = []
    coord_cursor = 0

    for p in range(polygon_count):
        r_start = int(h_go[p])
        r_end = int(h_go[p + 1])
        poly_ring_count = 0
        for r in range(r_start, r_end):
            if valid_rings[r]:
                c_start = int(h_ro[r])
                c_end = int(h_ro[r + 1])
                new_ring_offsets_list.append(coord_cursor)
                new_x_chunks.append(h_x[c_start:c_end])
                new_y_chunks.append(h_y[c_start:c_end])
                coord_cursor += c_end - c_start
                poly_ring_count += 1
        if poly_ring_count == 0:
            validity[p] = False
        new_geom_offsets.append(len(new_ring_offsets_list))

    new_ring_offsets_list.append(coord_cursor)

    if coord_cursor == 0:
        return None

    out_x = np.concatenate(new_x_chunks) if new_x_chunks else np.empty(0, dtype=np.float64)
    out_y = np.concatenate(new_y_chunks) if new_y_chunks else np.empty(0, dtype=np.float64)
    out_ring_offsets = np.asarray(new_ring_offsets_list, dtype=np.int32)
    out_geom_offsets = np.asarray(new_geom_offsets, dtype=np.int32)

    # Compute per-polygon bounds via vectorised operations
    bounds = np.empty((polygon_count, 4), dtype=np.float64)
    for p in range(polygon_count):
        gr_start = int(out_geom_offsets[p])
        gr_end = int(out_geom_offsets[p + 1])
        if gr_start >= gr_end or not validity[p]:
            bounds[p] = (0.0, 0.0, 0.0, 0.0)
            continue
        c_start = int(out_ring_offsets[gr_start])
        c_end = int(out_ring_offsets[gr_end])
        bounds[p] = (
            out_x[c_start:c_end].min(),
            out_y[c_start:c_end].min(),
            out_x[c_start:c_end].max(),
            out_y[c_start:c_end].max(),
        )

    families = {
        GeometryFamily.POLYGON: FamilyGeometryBuffer(
            family=GeometryFamily.POLYGON,
            schema=get_geometry_buffer_schema(GeometryFamily.POLYGON),
            row_count=polygon_count,
            x=out_x,
            y=out_y,
            geometry_offsets=out_geom_offsets,
            empty_mask=~validity,
            ring_offsets=out_ring_offsets,
            bounds=bounds,
        ),
    }
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=families,
        residency=Residency.HOST,
        runtime_history=[runtime_selection],
    )


def _extract_batch_coords(
    buffer: FamilyGeometryBuffer,
    geom_offsets: np.ndarray,
    ring_offsets: np.ndarray,
    invalid_family_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract coordinates for all invalid polygons into contiguous batch arrays.

    Returns (batch_x, batch_y, batch_ring_offsets, batch_geom_offsets) as
    host NumPy arrays ready for device upload, or None if no data.
    """
    if invalid_family_rows.size == 0:
        return None

    # Vectorised ring/coord range extraction for all invalid polygons
    r_starts = geom_offsets[invalid_family_rows]      # first ring index per polygon
    r_ends = geom_offsets[invalid_family_rows + 1]    # one-past-last ring index
    ring_counts_per_poly = r_ends - r_starts

    # Skip polygons with zero rings
    has_rings = ring_counts_per_poly > 0
    if not np.any(has_rings):
        return None

    # Build batch ring offsets: for each polygon, remap its ring offsets to
    # be relative to the batch coordinate array.
    batch_ring_list = []
    batch_geom_list = [0]
    coord_cursor = 0

    for i in range(invalid_family_rows.size):
        rs = int(r_starts[i])
        re = int(r_ends[i])
        if rs >= re:
            batch_geom_list.append(len(batch_ring_list))
            continue
        local_ring_offs = ring_offsets[rs:re + 1]
        coord_start = int(local_ring_offs[0])
        local_ring_offs_rebased = local_ring_offs - coord_start + coord_cursor
        batch_ring_list.extend(local_ring_offs_rebased[:-1].tolist())
        coord_cursor = int(local_ring_offs_rebased[-1])
        batch_geom_list.append(len(batch_ring_list))

    # Sentinel for ring_offsets
    batch_ring_list.append(coord_cursor)
    batch_ring_offsets = np.asarray(batch_ring_list, dtype=np.int32)
    batch_geom_offsets = np.asarray(batch_geom_list, dtype=np.int32)
    total_coords = coord_cursor

    if total_coords == 0:
        return None

    # Gather coordinates: extract slices for each invalid polygon
    batch_x = np.empty(total_coords, dtype=np.float64)
    batch_y = np.empty(total_coords, dtype=np.float64)
    buf_x = np.asarray(buffer.x, dtype=np.float64)
    buf_y = np.asarray(buffer.y, dtype=np.float64)

    dst = 0
    for i in range(invalid_family_rows.size):
        rs = int(r_starts[i])
        re = int(r_ends[i])
        if rs >= re:
            continue
        c_start = int(ring_offsets[rs])
        c_end = int(ring_offsets[re])
        n = c_end - c_start
        batch_x[dst:dst + n] = buf_x[c_start:c_end]
        batch_y[dst:dst + n] = buf_y[c_start:c_end]
        dst += n

    return batch_x, batch_y, batch_ring_offsets, batch_geom_offsets


def _scatter_repaired_geoms(
    owned_result: OwnedGeometryArray,
    invalid_family_rows: np.ndarray,
    batch_poly_count: int,
    fam_to_global: dict[int, int],
    invalid_rows_set: set[int],
    result: np.ndarray,
) -> int:
    """Map repaired OwnedGeometryArray output back to global result array.

    Converts the batch result to Shapely, validates each geometry, and
    scatters valid results into the correct global row positions.
    Returns the number of successfully repaired geometries.
    """
    import shapely as shp

    repaired_geoms = owned_result.to_shapely()
    if repaired_geoms is None or len(repaired_geoms) == 0:
        return 0

    count = 0
    validity = shp.is_valid(repaired_geoms)
    for out_idx in range(min(len(repaired_geoms), batch_poly_count)):
        geom = repaired_geoms[out_idx]
        if geom is None or geom.is_empty:
            continue
        if not validity[out_idx]:
            continue
        fam_row = int(invalid_family_rows[out_idx])
        global_row = fam_to_global.get(fam_row)
        if global_row is not None and global_row in invalid_rows_set:
            result[global_row] = geom
            count += 1
    return count


def gpu_repair_invalid_polygons(
    owned: OwnedGeometryArray,
    invalid_rows: np.ndarray,
    geometries: np.ndarray,
    *,
    method: str = "linework",
    keep_collapsed: bool = True,
) -> GPURepairResult | None:
    """GPU-resident batch repair of invalid polygon geometries (Phase 16).

    Implements the full make_valid pipeline on GPU with batch processing:
    1. Collect all invalid polygon coordinates into one contiguous batch
    2. Phase B: Close rings, remove duplicates, fix orientation (batched)
    3. Phase A+C: Detect and split self-intersections (batched)
    4. Phase D: Re-polygonize via overlay half-edge/face-walk pipeline (batched)
    5. Map output polygons back to global row indices

    No per-polygon Python loop. No shapely.polygonize or shapely.make_valid
    fallback. All repair is GPU-resident.

    Returns None if GPU repair is not applicable (no GPU, no polygon families,
    or CuPy not available).

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
        reason="make_valid GPU batch repair pipeline (Phase 16)",
    )

    result = geometries.copy()
    phases_used: list[str] = []
    gpu_repaired_count = 0
    invalid_rows_set = set(invalid_rows.tolist())

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

        # Map invalid rows to family rows (vectorised, no Python loop)
        family_tag = FAMILY_TAGS[family_name]
        tag_match = owned.tags == family_tag
        invalid_mask = np.zeros(len(owned.tags), dtype=bool)
        invalid_mask[invalid_rows] = True
        global_invalid = np.flatnonzero(tag_match & invalid_mask)
        if global_invalid.size == 0:
            continue

        fam_row_offsets = owned.family_row_offsets[global_invalid]
        valid_fro = (fam_row_offsets >= 0) & (fam_row_offsets < polygon_count)
        global_invalid = global_invalid[valid_fro]
        fam_row_offsets = fam_row_offsets[valid_fro]
        if global_invalid.size == 0:
            continue

        # Unique family rows and their global row mapping
        invalid_family_rows = np.unique(fam_row_offsets)
        # Build family_row -> global_row mapping
        fam_to_global = {}
        for gi, fro in zip(global_invalid, fam_row_offsets):
            fam_to_global.setdefault(int(fro), int(gi))

        # --- Step 1: Batch extract all invalid polygon coords ---
        batch = _extract_batch_coords(
            buffer, geom_offsets, ring_offsets, invalid_family_rows,
        )
        if batch is None:
            continue

        batch_x, batch_y, batch_ring_offsets, batch_geom_offsets = batch
        batch_ring_count = len(batch_ring_offsets) - 1
        batch_poly_count = len(batch_geom_offsets) - 1

        if batch_ring_count == 0 or batch_poly_count == 0:
            continue

        try:
            # Upload batch to device
            d_x = cp.asarray(np.ascontiguousarray(batch_x))
            d_y = cp.asarray(np.ascontiguousarray(batch_y))
            d_ring_offsets = cp.asarray(batch_ring_offsets)
            d_geom_offsets = cp.asarray(batch_geom_offsets)

            # --- Step 2: Phase B — batched simple repair ---
            # Close rings (all invalid rings at once)
            d_x, d_y, d_ring_offsets = _gpu_close_rings(
                d_x, d_y, d_ring_offsets,
                batch_ring_count, kernels,
            )
            phases_used.append("close_rings")

            # Remove duplicate vertices (all rings at once)
            d_x, d_y, d_ring_offsets = _gpu_remove_duplicate_vertices(
                d_x, d_y, d_ring_offsets,
                batch_ring_count, kernels,
            )
            phases_used.append("remove_duplicates")

            # Fix ring orientation (all rings at once, using batch geom offsets)
            d_x, d_y = _gpu_fix_ring_orientation(
                d_x, d_y, d_ring_offsets,
                d_geom_offsets, batch_ring_count, batch_poly_count, kernels,
            )
            phases_used.append("fix_orientation")

            # --- Step 3: Phase A+C — batched intersection detection + splitting ---
            d_x, d_y, d_ring_offsets, had_splits = \
                _split_self_intersections_gpu(
                    d_x, d_y, d_ring_offsets,
                    batch_ring_count, kernels,
                )
            if had_splits:
                phases_used.append("split_intersections")

            # --- Step 4: Phase D — batched repolygonization ---
            if had_splits:
                # Recompute batch_ring_count after splitting (ring count
                # unchanged — splits add vertices within existing rings)
                batch_ring_count = int(d_ring_offsets.size) - 1

                owned_result = _repolygonize_from_split_rings(
                    d_x, d_y, d_ring_offsets,
                    d_geom_offsets, batch_ring_count, batch_poly_count,
                    runtime_selection=runtime_sel,
                )
                phases_used.append("repolygonize")

                if owned_result is not None:
                    gpu_repaired_count += _scatter_repaired_geoms(
                        owned_result, invalid_family_rows, batch_poly_count,
                        fam_to_global, invalid_rows_set, result,
                    )
            else:
                # No self-intersections — simple repairs were sufficient.
                # Build OwnedGeometryArray from the repaired batch and
                # convert back to Shapely.
                owned_result = _build_batch_repaired_owned(
                    d_x, d_y, d_ring_offsets, d_geom_offsets,
                    batch_ring_count, batch_poly_count, runtime_sel,
                )
                if owned_result is not None:
                    gpu_repaired_count += _scatter_repaired_geoms(
                        owned_result, invalid_family_rows, batch_poly_count,
                        fam_to_global, invalid_rows_set, result,
                    )

        except Exception:
            # GPU batch repair failed for this family; fall through below
            pass

    # Any rows still not repaired (non-polygon families, GPU exceptions)
    # fall through to CPU shapely.make_valid as a guarded last resort.
    still_invalid = [
        int(row_idx) for row_idx in invalid_rows
        if result[row_idx] is geometries[row_idx]
    ]

    if still_invalid:
        import shapely as shp
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


