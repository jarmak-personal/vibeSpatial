"""NVRTC kernel source for shared_paths (binary constructive).

Tier 1 NVRTC (ADR-0033) -- geometry-specific inner loops that iterate all
segment pairs across two linear geometries and detect collinear overlapping
segments, classifying them as forward (same direction) or backward (opposite
direction).

ADR-0002: CONSTRUCTIVE class -- stays fp64 on all devices per policy.
PrecisionPlan is wired through dispatch for observability.

The kernel uses a two-pass count-scatter pattern:
  Pass 0 (count): each thread counts shared segments for its geometry pair
  Prefix sum: exclusive_sum(counts) -> offsets
  Pass 1 (scatter): each thread writes shared segment coords + direction flags

Output arrays:
  - out_x1, out_y1, out_x2, out_y2: segment endpoint coordinates
  - out_dir: 0 = forward, 1 = backward
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Collinearity + overlap detection kernel
#
# Two segments (p1,p2) and (q1,q2) are collinear if:
#   cross(p2-p1, q1-p1) ~= 0  AND  cross(p2-p1, q2-p1) ~= 0
#
# If collinear, project onto the line direction:
#   t_q1 = dot(q1-p1, d) / dot(d, d)  where d = p2-p1
#   t_q2 = dot(q2-p1, d) / dot(d, d)
#
# Overlap exists if max(0, min(t_q1,t_q2)) < min(1, max(t_q1,t_q2))
# and the overlap interval has non-zero length.
#
# Direction: if t_q2 > t_q1 => forward, else => backward
# ---------------------------------------------------------------------------

_SHARED_PATHS_KERNEL_SOURCE = r"""
// Collinearity tolerance: segments must lie on the same infinite line
// within this cross-product tolerance (relative to segment length squared).
#define COL_EPS 1e-10

// Minimum parametric overlap length to consider segments as shared.
#define OVERLAP_EPS 1e-12

// ===================================================================
// Device helper: count or scatter shared segments between two coord ranges
// ===================================================================
//
// For LineString: geometry_offsets[row] .. geometry_offsets[row+1] are coords.
// Each consecutive pair of coords forms a segment.
//
// mode == 0: count only (writes count to *out_count)
// mode == 1: scatter (writes segments starting at scatter_offset)
//
// Returns the number of shared segments found.

extern "C" __device__ int shared_segments_between_ranges(
    const double* __restrict__ ax, const double* __restrict__ ay,
    const int a_start, const int a_end,
    const double* __restrict__ bx, const double* __restrict__ by,
    const int b_start, const int b_end,
    const int mode,
    double* __restrict__ out_x1, double* __restrict__ out_y1,
    double* __restrict__ out_x2, double* __restrict__ out_y2,
    int* __restrict__ out_dir,
    const int scatter_offset
) {
    int found = 0;
    const int a_seg_count = a_end - a_start - 1;
    const int b_seg_count = b_end - b_start - 1;

    for (int i = 0; i < a_seg_count; i++) {
        const int ai = a_start + i;
        const double p1x = ax[ai], p1y = ay[ai];
        const double p2x = ax[ai + 1], p2y = ay[ai + 1];

        const double dx = p2x - p1x;
        const double dy = p2y - p1y;
        const double d_dot_d = dx * dx + dy * dy;

        // Skip degenerate (zero-length) segments in A
        if (d_dot_d < 1e-30) continue;

        for (int j = 0; j < b_seg_count; j++) {
            const int bj = b_start + j;
            const double q1x = bx[bj], q1y = by[bj];
            const double q2x = bx[bj + 1], q2y = by[bj + 1];

            // Skip degenerate segments in B
            const double ex = q2x - q1x, ey = q2y - q1y;
            if (ex * ex + ey * ey < 1e-30) continue;

            // Collinearity check: cross(d, q1-p1) and cross(d, q2-p1)
            const double cross1 = dx * (q1y - p1y) - dy * (q1x - p1x);
            const double cross2 = dx * (q2y - p1y) - dy * (q2x - p1x);

            // Tolerance is relative to segment length squared
            if (cross1 * cross1 > COL_EPS * d_dot_d * d_dot_d) continue;
            if (cross2 * cross2 > COL_EPS * d_dot_d * d_dot_d) continue;

            // Segments are collinear -- compute parametric projections
            const double t_q1 = (dx * (q1x - p1x) + dy * (q1y - p1y)) / d_dot_d;
            const double t_q2 = (dx * (q2x - p1x) + dy * (q2y - p1y)) / d_dot_d;

            // Overlap interval on segment A's parameter space [0, 1]
            const double t_min_q = (t_q1 < t_q2) ? t_q1 : t_q2;
            const double t_max_q = (t_q1 < t_q2) ? t_q2 : t_q1;

            const double t_lo = (t_min_q > 0.0) ? t_min_q : 0.0;  // max(0, t_min_q)
            const double t_hi = (t_max_q < 1.0) ? t_max_q : 1.0;  // min(1, t_max_q)

            if (t_hi - t_lo < OVERLAP_EPS) continue;  // No meaningful overlap

            // Compute the shared segment endpoints in world coordinates
            const double sx1 = p1x + t_lo * dx;
            const double sy1 = p1y + t_lo * dy;
            const double sx2 = p1x + t_hi * dx;
            const double sy2 = p1y + t_hi * dy;

            // Direction classification:
            // Forward  (0): B segment traverses same direction as A segment
            //               i.e., t_q2 > t_q1 (B's q2 is further along A's direction)
            // Backward (1): B segment traverses opposite direction
            //               i.e., t_q2 < t_q1
            const int direction = (t_q2 >= t_q1) ? 0 : 1;

            if (mode == 1) {
                const int idx = scatter_offset + found;
                out_x1[idx] = sx1;
                out_y1[idx] = sy1;
                out_x2[idx] = sx2;
                out_y2[idx] = sy2;
                out_dir[idx] = direction;
            }

            found++;
        }
    }
    return found;
}


// ===================================================================
// LineString x LineString: count pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_ls_ls_count(
    // Left OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    int*         __restrict__ counts,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    // Validity checks
    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        counts[tid] = 0;
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) {
        counts[tid] = 0;
        return;
    }

    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) {
        counts[tid] = 0;
        return;
    }

    const int l_start = l_geom_off[l_frow];
    const int l_end   = l_geom_off[l_frow + 1];
    const int r_start = r_geom_off[r_frow];
    const int r_end   = r_geom_off[r_frow + 1];

    if (l_end - l_start < 2 || r_end - r_start < 2) {
        counts[tid] = 0;
        return;
    }

    counts[tid] = shared_segments_between_ranges(
        l_x, l_y, l_start, l_end,
        r_x, r_y, r_start, r_end,
        0,  // mode = count
        (double*)0, (double*)0, (double*)0, (double*)0, (int*)0, 0
    );
}


// ===================================================================
// LineString x LineString: scatter pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_ls_ls_scatter(
    // Left OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    const int*   __restrict__ offsets,
    double*      __restrict__ out_x1,
    double*      __restrict__ out_y1,
    double*      __restrict__ out_x2,
    double*      __restrict__ out_y2,
    int*         __restrict__ out_dir,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) return;
    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) return;

    const int l_start = l_geom_off[l_frow];
    const int l_end   = l_geom_off[l_frow + 1];
    const int r_start = r_geom_off[r_frow];
    const int r_end   = r_geom_off[r_frow + 1];

    if (l_end - l_start < 2 || r_end - r_start < 2) return;

    const int scatter_off = offsets[tid];
    shared_segments_between_ranges(
        l_x, l_y, l_start, l_end,
        r_x, r_y, r_start, r_end,
        1,  // mode = scatter
        out_x1, out_y1, out_x2, out_y2, out_dir, scatter_off
    );
}


// ===================================================================
// MultiLineString x LineString: count pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_mls_ls_count(
    // Left MLS OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const int*   __restrict__ l_part_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right LS OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    int*         __restrict__ counts,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        counts[tid] = 0;
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) {
        counts[tid] = 0;
        return;
    }

    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) {
        counts[tid] = 0;
        return;
    }

    // Right LS coords
    const int r_start = r_geom_off[r_frow];
    const int r_end   = r_geom_off[r_frow + 1];
    if (r_end - r_start < 2) {
        counts[tid] = 0;
        return;
    }

    // Left MLS: iterate over parts
    const int l_part_begin = l_geom_off[l_frow];
    const int l_part_end   = l_geom_off[l_frow + 1];
    int total = 0;
    for (int p = l_part_begin; p < l_part_end; p++) {
        const int l_start = l_part_off[p];
        const int l_end   = l_part_off[p + 1];
        if (l_end - l_start < 2) continue;
        total += shared_segments_between_ranges(
            l_x, l_y, l_start, l_end,
            r_x, r_y, r_start, r_end,
            0, (double*)0, (double*)0, (double*)0, (double*)0, (int*)0, 0
        );
    }
    counts[tid] = total;
}


// ===================================================================
// MultiLineString x LineString: scatter pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_mls_ls_scatter(
    // Left MLS OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const int*   __restrict__ l_part_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right LS OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    const int*   __restrict__ offsets,
    double*      __restrict__ out_x1,
    double*      __restrict__ out_y1,
    double*      __restrict__ out_x2,
    double*      __restrict__ out_y2,
    int*         __restrict__ out_dir,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) return;
    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) return;

    const int r_start = r_geom_off[r_frow];
    const int r_end   = r_geom_off[r_frow + 1];
    if (r_end - r_start < 2) return;

    const int l_part_begin = l_geom_off[l_frow];
    const int l_part_end   = l_geom_off[l_frow + 1];
    int written = 0;
    for (int p = l_part_begin; p < l_part_end; p++) {
        const int l_start = l_part_off[p];
        const int l_end   = l_part_off[p + 1];
        if (l_end - l_start < 2) continue;
        written += shared_segments_between_ranges(
            l_x, l_y, l_start, l_end,
            r_x, r_y, r_start, r_end,
            1, out_x1, out_y1, out_x2, out_y2, out_dir,
            offsets[tid] + written
        );
    }
}


// ===================================================================
// MultiLineString x MultiLineString: count pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_mls_mls_count(
    // Left MLS OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const int*   __restrict__ l_part_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right MLS OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const int*   __restrict__ r_part_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    int*         __restrict__ counts,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        counts[tid] = 0;
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) {
        counts[tid] = 0;
        return;
    }

    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) {
        counts[tid] = 0;
        return;
    }

    const int l_part_begin = l_geom_off[l_frow];
    const int l_part_end   = l_geom_off[l_frow + 1];
    const int r_part_begin = r_geom_off[r_frow];
    const int r_part_end   = r_geom_off[r_frow + 1];

    int total = 0;
    for (int lp = l_part_begin; lp < l_part_end; lp++) {
        const int l_start = l_part_off[lp];
        const int l_end   = l_part_off[lp + 1];
        if (l_end - l_start < 2) continue;

        for (int rp = r_part_begin; rp < r_part_end; rp++) {
            const int r_start = r_part_off[rp];
            const int r_end   = r_part_off[rp + 1];
            if (r_end - r_start < 2) continue;
            total += shared_segments_between_ranges(
                l_x, l_y, l_start, l_end,
                r_x, r_y, r_start, r_end,
                0, (double*)0, (double*)0, (double*)0, (double*)0, (int*)0, 0
            );
        }
    }
    counts[tid] = total;
}


// ===================================================================
// MultiLineString x MultiLineString: scatter pass
// ===================================================================
extern "C" __global__ void __launch_bounds__(256, 4)
shared_paths_mls_mls_scatter(
    // Left MLS OGA routing
    const bool*  __restrict__ l_validity,
    const signed char* __restrict__ l_tags,
    const int*   __restrict__ l_fam_row_off,
    const int*   __restrict__ l_geom_off,
    const int*   __restrict__ l_part_off,
    const bool*  __restrict__ l_empty_mask,
    const double* __restrict__ l_x,
    const double* __restrict__ l_y,
    const int l_tag,
    // Right MLS OGA routing
    const bool*  __restrict__ r_validity,
    const signed char* __restrict__ r_tags,
    const int*   __restrict__ r_fam_row_off,
    const int*   __restrict__ r_geom_off,
    const int*   __restrict__ r_part_off,
    const bool*  __restrict__ r_empty_mask,
    const double* __restrict__ r_x,
    const double* __restrict__ r_y,
    const int r_tag,
    // Pair indices and output
    const int*   __restrict__ left_idx,
    const int*   __restrict__ right_idx,
    const int*   __restrict__ offsets,
    double*      __restrict__ out_x1,
    double*      __restrict__ out_y1,
    double*      __restrict__ out_x2,
    double*      __restrict__ out_y2,
    int*         __restrict__ out_dir,
    const int    pair_count
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pair_count) return;

    const int li = left_idx[tid];
    const int ri = right_idx[tid];

    if (!l_validity[li] || !r_validity[ri] ||
        l_tags[li] != l_tag || r_tags[ri] != r_tag) {
        return;
    }

    const int l_frow = l_fam_row_off[li];
    const int r_frow = r_fam_row_off[ri];
    if (l_frow < 0 || r_frow < 0) return;
    if (l_empty_mask[l_frow] || r_empty_mask[r_frow]) return;

    const int l_part_begin = l_geom_off[l_frow];
    const int l_part_end   = l_geom_off[l_frow + 1];
    const int r_part_begin = r_geom_off[r_frow];
    const int r_part_end   = r_geom_off[r_frow + 1];

    int written = 0;
    for (int lp = l_part_begin; lp < l_part_end; lp++) {
        const int l_start = l_part_off[lp];
        const int l_end   = l_part_off[lp + 1];
        if (l_end - l_start < 2) continue;

        for (int rp = r_part_begin; rp < r_part_end; rp++) {
            const int r_start = r_part_off[rp];
            const int r_end   = r_part_off[rp + 1];
            if (r_end - r_start < 2) continue;
            written += shared_segments_between_ranges(
                l_x, l_y, l_start, l_end,
                r_x, r_y, r_start, r_end,
                1, out_x1, out_y1, out_x2, out_y2, out_dir,
                offsets[tid] + written
            );
        }
    }
}
"""

SHARED_PATHS_KERNEL_NAMES: tuple[str, ...] = (
    "shared_paths_ls_ls_count",
    "shared_paths_ls_ls_scatter",
    "shared_paths_mls_ls_count",
    "shared_paths_mls_ls_scatter",
    "shared_paths_mls_mls_count",
    "shared_paths_mls_mls_scatter",
)
