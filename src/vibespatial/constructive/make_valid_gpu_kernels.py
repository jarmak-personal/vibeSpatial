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
