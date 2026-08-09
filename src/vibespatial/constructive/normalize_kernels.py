"""NVRTC kernel sources for normalize."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE

# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------

_RING_KERNEL_SOURCE = (
    STRIP_CLOSURE_DEVICE
    + r"""
typedef {compute_type} compute_t;

extern "C" __global__ void normalize_ring_rotate(
    const double* __restrict__ x_in,
    const double* __restrict__ y_in,
    double* __restrict__ x_out,
    double* __restrict__ y_out,
    const int* __restrict__ ring_offsets,
    const unsigned char* __restrict__ is_exterior,
    double center_x,
    double center_y,
    int total_rings
) {{
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= total_rings) return;

    const int coord_start = ring_offsets[ring];
    const int coord_end = ring_offsets[ring + 1];
    const int total = coord_end - coord_start;
    if (total <= 0) return;

    // Determine unique vertex count (strip closing vertex)
    int n = total;
    n = vs_strip_closure(x_in, y_in, coord_start, coord_end, n, 1e-24);
    if (n <= 0) return;

    // Select and consume the canonical start in one ring-owned launch. This
    // avoids a cross-launch intermediate and its allocator-lifetime hazard.
    int best = coord_start;
    compute_t best_x = (compute_t)(x_in[coord_start] - center_x);
    compute_t best_y = (compute_t)(y_in[coord_start] - center_y);
    for (int i = 1; i < n; i++) {{
        const int idx = coord_start + i;
        const compute_t cx = (compute_t)(x_in[idx] - center_x);
        const compute_t cy = (compute_t)(y_in[idx] - center_y);
        if (cx < best_x || (cx == best_x && cy < best_y)) {{
            best = idx;
            best_x = cx;
            best_y = cy;
        }}
    }}
    const int offset_in_ring = best - coord_start;

    // Choose the lexicographically smaller cyclic direction. This makes the
    // normalized ring independent of input winding and matches GEOS ring
    // normalization rather than merely rotating the existing direction.
    bool reverse = false;
    for (int i = 1; i < n; i++) {{
        int forward_local = (offset_in_ring + i) % n;
        int reverse_local = (offset_in_ring - i) % n;
        if (reverse_local < 0) reverse_local += n;
        const int forward_idx = coord_start + forward_local;
        const int reverse_idx = coord_start + reverse_local;
        const double forward_x = x_in[forward_idx];
        const double reverse_x = x_in[reverse_idx];
        const double forward_y = y_in[forward_idx];
        const double reverse_y = y_in[reverse_idx];
        if (reverse_x < forward_x ||
            (reverse_x == forward_x && reverse_y < forward_y)) {{
            reverse = true;
            break;
        }}
        if (forward_x < reverse_x ||
            (forward_x == reverse_x && forward_y < reverse_y)) {{
            break;
        }}
    }}
    if (is_exterior[ring] == 0u) reverse = !reverse;

    // Cyclic copy: rotate so that best vertex is first and use the canonical
    // direction selected above.
    for (int i = 0; i < n; i++) {{
        int local = reverse ? (offset_in_ring - i) % n : (offset_in_ring + i) % n;
        if (local < 0) local += n;
        const int src = coord_start + local;
        const int dst = coord_start + i;
        x_out[dst] = x_in[src];
        y_out[dst] = y_in[src];
    }}

    // Restore closing vertex = new first vertex
    if (total > n) {{
        x_out[coord_start + n] = x_out[coord_start];
        y_out[coord_start + n] = y_out[coord_start];
    }}
}}
"""
)
_LINE_KERNEL_SOURCE = r"""
typedef {compute_type} compute_t;

extern "C" __global__ void normalize_linestring_reverse(
    double* x,
    double* y,
    const int* geometry_offsets,
    double center_x,
    double center_y,
    int row_count
) {{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int coord_start = geometry_offsets[row];
    const int coord_end = geometry_offsets[row + 1];
    const int n = coord_end - coord_start;
    if (n < 2) return;

    // Compare first vs last vertex lexicographically
    const compute_t first_x = (compute_t)(x[coord_start] - center_x);
    const compute_t first_y = (compute_t)(y[coord_start] - center_y);
    const compute_t last_x = (compute_t)(x[coord_end - 1] - center_x);
    const compute_t last_y = (compute_t)(y[coord_end - 1] - center_y);

    bool should_reverse = (last_x < first_x) || (last_x == first_x && last_y < first_y);
    if (!should_reverse) return;

    // Reverse in-place
    for (int i = 0; i < n / 2; i++) {{
        const int a = coord_start + i;
        const int b = coord_end - 1 - i;
        double tmp_x = x[a]; x[a] = x[b]; x[b] = tmp_x;
        double tmp_y = y[a]; y[a] = y[b]; y[b] = tmp_y;
    }}
}}
"""
_RING_KERNEL_NAMES = ("normalize_ring_rotate",)
_LINE_KERNEL_NAMES = ("normalize_linestring_reverse",)
