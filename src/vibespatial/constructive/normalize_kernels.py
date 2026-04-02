"""NVRTC kernel sources for normalize."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE

# ---------------------------------------------------------------------------
# NVRTC kernel sources (Tier 1)
# ---------------------------------------------------------------------------

_RING_KERNEL_SOURCE = STRIP_CLOSURE_DEVICE + r"""
typedef {compute_type} compute_t;

extern "C" __global__ void normalize_ring_find_min(
    const double* x,
    const double* y,
    const int* ring_offsets,
    int* min_index,
    double center_x,
    double center_y,
    int total_rings
) {{
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= total_rings) return;

    const int coord_start = ring_offsets[ring];
    const int coord_end = ring_offsets[ring + 1];
    int n = coord_end - coord_start;

    // Strip closing vertex if present (last == first)
    n = vs_strip_closure(x, y, coord_start, coord_end, n, 1e-24);

    if (n <= 0) {{
        min_index[ring] = coord_start;
        return;
    }}

    // Find lexicographically smallest vertex (min x, then min y on tie)
    int best = coord_start;
    compute_t best_x = (compute_t)(x[coord_start] - center_x);
    compute_t best_y = (compute_t)(y[coord_start] - center_y);

    for (int i = 1; i < n; i++) {{
        const int idx = coord_start + i;
        const compute_t cx = (compute_t)(x[idx] - center_x);
        const compute_t cy = (compute_t)(y[idx] - center_y);
        if (cx < best_x || (cx == best_x && cy < best_y)) {{
            best = idx;
            best_x = cx;
            best_y = cy;
        }}
    }}
    min_index[ring] = best;
}}

extern "C" __global__ void normalize_ring_rotate(
    const double* x_in,
    const double* y_in,
    double* x_out,
    double* y_out,
    const int* ring_offsets,
    const int* min_index,
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

    const int best = min_index[ring];
    const int offset_in_ring = best - coord_start;

    // Cyclic copy: rotate so that best vertex is first
    for (int i = 0; i < n; i++) {{
        const int src = coord_start + ((offset_in_ring + i) % n);
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
_RING_KERNEL_NAMES = ("normalize_ring_find_min", "normalize_ring_rotate")
_LINE_KERNEL_NAMES = ("normalize_linestring_reverse",)
