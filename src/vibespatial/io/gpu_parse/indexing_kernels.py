"""NVRTC kernel sources for fused ingest + spatial index."""

from __future__ import annotations

_BOUNDS_KERNEL_SOURCE = r"""
// Per-feature bounding box computation from flat coordinate arrays.
// Input: d_x (float64), d_y (float64), geometry_offsets (int32)
// Output: d_bounds (float64, N*4: min_x, min_y, max_x, max_y)
//
// ADR-0002: Bounds stay fp64 (COARSE class, memory-bound).

extern "C" __global__ void __launch_bounds__(256, 4)
compute_feature_bounds(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int*    __restrict__ geometry_offsets,
    double*       __restrict__ d_bounds,
    const int                  n_features
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    const int coord_start = geometry_offsets[idx];
    const int coord_end   = geometry_offsets[idx + 1];

    const int base = idx * 4;
    if (coord_end <= coord_start) {
        // Empty geometry — write NaN bounds
        const double nan_val = __longlong_as_double(0x7ff8000000000000ULL);
        d_bounds[base + 0] = nan_val;
        d_bounds[base + 1] = nan_val;
        d_bounds[base + 2] = nan_val;
        d_bounds[base + 3] = nan_val;
        return;
    }

    double min_x = d_x[coord_start];
    double min_y = d_y[coord_start];
    double max_x = min_x;
    double max_y = min_y;

    for (int c = coord_start + 1; c < coord_end; ++c) {
        const double xv = d_x[c];
        const double yv = d_y[c];
        min_x = xv < min_x ? xv : min_x;
        min_y = yv < min_y ? yv : min_y;
        max_x = xv > max_x ? xv : max_x;
        max_y = yv > max_y ? yv : max_y;
    }

    d_bounds[base + 0] = min_x;
    d_bounds[base + 1] = min_y;
    d_bounds[base + 2] = max_x;
    d_bounds[base + 3] = max_y;
}
"""

_BOUNDS_KERNEL_NAMES = ("compute_feature_bounds",)


_HILBERT_KERNEL_SOURCE = r"""
// 32-bit Hilbert curve encoding from (x, y) integer coordinates on [0, 2^16).
// Input: d_bounds (float64, N*4), extent (minx, miny, maxx, maxy)
// Output: d_hilbert_codes (uint32, N)

extern "C" __device__ unsigned int interleave_bits(unsigned int v) {
    v = (v | (v << 8)) & 0x00FF00FFu;
    v = (v | (v << 4)) & 0x0F0F0F0Fu;
    v = (v | (v << 2)) & 0x33333333u;
    v = (v | (v << 1)) & 0x55555555u;
    return v;
}

extern "C" __global__ void __launch_bounds__(256, 4)
compute_hilbert_codes(
    const double* __restrict__ d_bounds,
    const double               extent_minx,
    const double               extent_miny,
    const double               extent_maxx,
    const double               extent_maxy,
    unsigned int*  __restrict__ d_hilbert_codes,
    const int                  n_features
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_features) return;

    const int base = idx * 4;
    const double bx0 = d_bounds[base + 0];
    const double by0 = d_bounds[base + 1];
    const double bx1 = d_bounds[base + 2];
    const double by1 = d_bounds[base + 3];

    // Handle NaN bounds (empty geometries) — assign max code to sort last
    if (isnan(bx0) || isnan(by0) || isnan(bx1) || isnan(by1)) {
        d_hilbert_codes[idx] = 0xFFFFFFFFu;
        return;
    }

    // Compute centroid
    const double cx = (bx0 + bx1) * 0.5;
    const double cy = (by0 + by1) * 0.5;

    // Normalize to [0, 65535] integer grid
    const double span_x = fmax(extent_maxx - extent_minx, 1e-12);
    const double span_y = fmax(extent_maxy - extent_miny, 1e-12);
    unsigned int x = (unsigned int)llround(((cx - extent_minx) / span_x) * 65535.0);
    unsigned int y = (unsigned int)llround(((cy - extent_miny) / span_y) * 65535.0);

    // Clamp to valid range
    x = x > 65535u ? 65535u : x;
    y = y > 65535u ? 65535u : y;

    // Hilbert encoding — threadlocalmutex algorithm (level=16)
    // Shift to fill 16-bit register (level 16, so no shift needed: 16-16=0)
    // x = x << 0; y = y << 0;

    // Initial prefix scan round
    unsigned int a = x ^ y;
    unsigned int b = 0xFFFFu ^ a;
    unsigned int c = 0xFFFFu ^ (x | y);
    unsigned int d = x & (y ^ 0xFFFFu);

    unsigned int A = a | (b >> 1);
    unsigned int B = (a >> 1) ^ a;
    unsigned int C = ((c >> 1) ^ (b & (d >> 1))) ^ c;
    unsigned int D = ((a & (c >> 1)) ^ (d >> 1)) ^ d;

    a = A; b = B; c = C; d = D;

    A = (a & (a >> 2)) ^ (b & (b >> 2));
    B = (a & (b >> 2)) ^ (b & ((a ^ b) >> 2));
    C ^= (a & (c >> 2)) ^ (b & (d >> 2));
    D ^= (b & (c >> 2)) ^ ((a ^ b) & (d >> 2));

    a = A; b = B; c = C; d = D;

    A = (a & (a >> 4)) ^ (b & (b >> 4));
    B = (a & (b >> 4)) ^ (b & ((a ^ b) >> 4));
    C ^= (a & (c >> 4)) ^ (b & (d >> 4));
    D ^= (b & (c >> 4)) ^ ((a ^ b) & (d >> 4));

    // Final round
    a = A; b = B; c = C; d = D;

    C ^= (a & (c >> 8)) ^ (b & (d >> 8));
    D ^= (b & (c >> 8)) ^ ((a ^ b) & (d >> 8));

    // Undo transformation prefix scan
    a = C ^ (C >> 1);
    b = D ^ (D >> 1);

    // Recover index bits
    unsigned int i0 = x ^ y;
    unsigned int i1 = b | (0xFFFFu ^ (i0 | a));

    d_hilbert_codes[idx] = ((interleave_bits(i1) << 1) | interleave_bits(i0));
}
"""

_HILBERT_KERNEL_NAMES = ("compute_hilbert_codes",)
