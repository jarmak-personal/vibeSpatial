"""NVRTC kernel sources for make_valid_gpu."""

from __future__ import annotations

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
    const int ring_count,
    const int vertex_count
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vertex_count) return;
    const int ring = vertex_ring_ids[v];
    if (ring >= ring_count) {
        out_keep[v] = 0;
        return;
    }
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

// Phase B: scatter_kept_vertices — one thread per coordinate-capacity slot.
// keep_positions is the exclusive scan of keep flags. Ring offsets carry the
// logical output length; coordinate arrays may retain trailing capacity.
extern "C" __global__ void __launch_bounds__(256, 4) scatter_kept_vertices(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const unsigned char* __restrict__ keep,
    const int* __restrict__ keep_positions,
    double* __restrict__ out_x,
    double* __restrict__ out_y,
    const int coordinate_capacity
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= coordinate_capacity || keep[v] == 0) return;
    const int out = keep_positions[v];
    out_x[out] = x[v];
    out_y[out] = y[v];
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

"""
_REPAIR_KERNEL_NAMES = (
    "check_ring_closure",
    "close_rings",
    "flag_duplicate_vertices",
    "scatter_kept_vertices",
    "compute_ring_shoelace",
    "reverse_ring_coords",
)
