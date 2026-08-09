"""NVRTC kernel sources for segment extraction, classification, and candidate scatter."""

from __future__ import annotations

import numpy as np

from vibespatial.cuda.device_functions.intersection_point import (
    INTERSECTION_POINT_DEVICE,
)
from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.device_functions.segment_crossing import SEGMENT_CROSSING_DEVICE
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS

# ---------------------------------------------------------------------------
# Family type codes matching GeometryFamily enum order (0-based)
# ---------------------------------------------------------------------------
_FAMILY_LINESTRING = FAMILY_TAGS[GeometryFamily.LINESTRING]
_FAMILY_POLYGON = FAMILY_TAGS[GeometryFamily.POLYGON]
_FAMILY_MULTILINESTRING = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
_FAMILY_MULTIPOLYGON = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]

# ---------------------------------------------------------------------------
# Kernel 1: GPU Segment Extraction (count + scatter)
# ---------------------------------------------------------------------------
# Extracts segments from OwnedGeometryArray offset arrays entirely on device.
# Two-pass count-scatter: pass 0 counts segments per geometry row,
# pass 1 scatters segment endpoints into SoA output arrays.
#
# Family dispatch:
#   linestring:       coords in [geom_off[r], geom_off[r+1]) => segments = coords - 1
#   polygon:          rings in [geom_off[r], geom_off[r+1]),
#                     each ring's coords in [ring_off[ri], ring_off[ri+1]) => segs = coords - 1
#   multilinestring:  parts in [geom_off[r], geom_off[r+1]),
#                     each part's coords in [part_off[pi], part_off[pi+1]) => segs = coords - 1
#   multipolygon:     parts(polygons) in [geom_off[r], geom_off[r+1]),
#                     each poly's rings in [part_off[pi], part_off[pi+1]),
#                     each ring's coords in [ring_off[ri], ring_off[ri+1]) => segs = coords - 1
# ---------------------------------------------------------------------------

_SEGMENT_EXTRACT_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

/* Family type codes (must match Python constants) */
#define FAMILY_LINESTRING {family_linestring}
#define FAMILY_POLYGON {family_polygon}
#define FAMILY_MULTILINESTRING {family_multilinestring}
#define FAMILY_MULTIPOLYGON {family_multipolygon}

extern "C" __global__ void __launch_bounds__(256, 4)
count_segments(
    const int* __restrict__ valid_rows,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_offsets,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const int* __restrict__ ring_off,
    const unsigned char* __restrict__ empty_mask,
    int* __restrict__ seg_counts,
    const int n_valid
) {{{{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_valid) return;

    const int global_row = valid_rows[tid];
    const int family = family_codes[global_row];
    const int fam_row = family_row_offsets[global_row];

    if (empty_mask[tid]) {{{{
        seg_counts[tid] = 0;
        return;
    }}}}

    int count = 0;

    if (family == FAMILY_LINESTRING) {{{{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        count = (ce - cs > 1) ? (ce - cs - 1) : 0;
    }}}} else if (family == FAMILY_POLYGON) {{{{
        const int rs = geom_off[fam_row];
        const int re = geom_off[fam_row + 1];
        for (int ri = rs; ri < re; ++ri) {{{{
            const int cs = ring_off[ri];
            const int ce = ring_off[ri + 1];
            if (ce - cs > 1) count += ce - cs - 1;
        }}}}
    }}}} else if (family == FAMILY_MULTILINESTRING) {{{{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{{{
            const int cs = part_off[pi];
            const int ce = part_off[pi + 1];
            if (ce - cs > 1) count += ce - cs - 1;
        }}}}
    }}}} else if (family == FAMILY_MULTIPOLYGON) {{{{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{{{
            const int rs = part_off[pi];
            const int re = part_off[pi + 1];
            for (int ri = rs; ri < re; ++ri) {{{{
                const int cs = ring_off[ri];
                const int ce = ring_off[ri + 1];
                if (ce - cs > 1) count += ce - cs - 1;
            }}}}
        }}}}
    }}}}

    seg_counts[tid] = count;
}}}}


extern "C" __global__ void __launch_bounds__(256, 4)
scatter_segments(
    const int* __restrict__ valid_rows,
    const int* __restrict__ family_codes,
    const int* __restrict__ family_row_offsets,
    const int* __restrict__ geom_off,
    const int* __restrict__ part_off,
    const int* __restrict__ ring_off,
    const unsigned char* __restrict__ empty_mask,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ seg_offsets,
    int* __restrict__ out_row_idx,
    int* __restrict__ out_seg_idx,
    int* __restrict__ out_part_idx,
    int* __restrict__ out_ring_idx,
    double* __restrict__ out_x0,
    double* __restrict__ out_y0,
    double* __restrict__ out_x1,
    double* __restrict__ out_y1,
    const int n_valid,
    const int tier_start,
    const int tier_width
) {{{{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_valid) return;

    if (empty_mask[tid]) return;

    const int global_row = valid_rows[tid];
    const int family = family_codes[global_row];
    const int fam_row = family_row_offsets[global_row];
    int write_pos = seg_offsets[tid];
    int seg_idx = 0;

    if (family == FAMILY_LINESTRING) {{{{
        const int cs = geom_off[fam_row];
        const int ce = geom_off[fam_row + 1];
        for (int c = cs; c < ce - 1; ++c) {{{{
            if (seg_idx >= tier_start && seg_idx < tier_start + tier_width) {{{{
                out_row_idx[write_pos] = global_row;
                out_seg_idx[write_pos] = seg_idx;
                out_part_idx[write_pos] = 0;
                out_ring_idx[write_pos] = 0;
                out_x0[write_pos] = x[c];
                out_y0[write_pos] = y[c];
                out_x1[write_pos] = x[c + 1];
                out_y1[write_pos] = y[c + 1];
                ++write_pos;
            }}}}
            ++seg_idx;
        }}}}
    }}}} else if (family == FAMILY_POLYGON) {{{{
        const int rs = geom_off[fam_row];
        const int re = geom_off[fam_row + 1];
        for (int ri = rs; ri < re; ++ri) {{{{
            const int ring_local = ri - rs;
            const int cs = ring_off[ri];
            const int ce = ring_off[ri + 1];
            for (int c = cs; c < ce - 1; ++c) {{{{
                if (seg_idx >= tier_start && seg_idx < tier_start + tier_width) {{{{
                    out_row_idx[write_pos] = global_row;
                    out_seg_idx[write_pos] = seg_idx;
                    out_part_idx[write_pos] = 0;
                    out_ring_idx[write_pos] = ring_local;
                    out_x0[write_pos] = x[c];
                    out_y0[write_pos] = y[c];
                    out_x1[write_pos] = x[c + 1];
                    out_y1[write_pos] = y[c + 1];
                    ++write_pos;
                }}}}
                ++seg_idx;
            }}}}
        }}}}
    }}}} else if (family == FAMILY_MULTILINESTRING) {{{{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{{{
            const int part_local = pi - ps;
            const int cs = part_off[pi];
            const int ce = part_off[pi + 1];
            for (int c = cs; c < ce - 1; ++c) {{{{
                if (seg_idx >= tier_start && seg_idx < tier_start + tier_width) {{{{
                    out_row_idx[write_pos] = global_row;
                    out_seg_idx[write_pos] = seg_idx;
                    out_part_idx[write_pos] = part_local;
                    out_ring_idx[write_pos] = -1;
                    out_x0[write_pos] = x[c];
                    out_y0[write_pos] = y[c];
                    out_x1[write_pos] = x[c + 1];
                    out_y1[write_pos] = y[c + 1];
                    ++write_pos;
                }}}}
                ++seg_idx;
            }}}}
        }}}}
    }}}} else if (family == FAMILY_MULTIPOLYGON) {{{{
        const int ps = geom_off[fam_row];
        const int pe = geom_off[fam_row + 1];
        for (int pi = ps; pi < pe; ++pi) {{{{
            const int polygon_local = pi - ps;
            const int rs = part_off[pi];
            const int re = part_off[pi + 1];
            for (int ri = rs; ri < re; ++ri) {{{{
                const int ring_local = ri - rs;
                const int cs = ring_off[ri];
                const int ce = ring_off[ri + 1];
                for (int c = cs; c < ce - 1; ++c) {{{{
                    if (seg_idx >= tier_start && seg_idx < tier_start + tier_width) {{{{
                        out_row_idx[write_pos] = global_row;
                        out_seg_idx[write_pos] = seg_idx;
                        out_part_idx[write_pos] = polygon_local;
                        out_ring_idx[write_pos] = ring_local;
                        out_x0[write_pos] = x[c];
                        out_y0[write_pos] = y[c];
                        out_x1[write_pos] = x[c + 1];
                        out_y1[write_pos] = y[c + 1];
                        ++write_pos;
                    }}}}
                    ++seg_idx;
                }}}}
            }}}}
        }}}}
    }}}}
}}}}
"""

_SEGMENT_EXTRACT_KERNEL_NAMES = ("count_segments", "scatter_segments")

# ---------------------------------------------------------------------------
# Kernel 3: GPU Segment Pair Classification with Shewchuk adaptive refinement
# ---------------------------------------------------------------------------
# PREDICATE class: orientation tests on segment pairs.
# Ambiguous cases (|orient| <= error_bound) are refined on-GPU using
# Shewchuk-style two-product adaptive arithmetic -- no host round-trip.
# ---------------------------------------------------------------------------

_SEGMENT_CLASSIFY_KERNEL_SOURCE_TEMPLATE = """
typedef {compute_type} compute_t;

__device__ inline compute_t abs_ct(compute_t value) {{{{
    return value < (compute_t)0.0 ? -value : value;
}}}}

/* orient2d predicate provided by ORIENT2D_DEVICE (vs_orient2d) */
/* collinear containment provided by SEGMENT_CROSSING_DEVICE (vs_point_on_segment_collinear) */
/* intersection point provided by INTERSECTION_POINT_DEVICE */

/* Classification codes */
#define CLASS_DISJOINT 0
#define CLASS_PROPER   1
#define CLASS_TOUCH    2
#define CLASS_OVERLAP  3
#define CLASS_DEGENERATE 4

extern "C" __global__ void __launch_bounds__(256, 4)
classify_segment_pairs_v2(
    const int* __restrict__ left_lookup,
    const int* __restrict__ right_lookup,
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    signed char* __restrict__ out_kind,
    double* __restrict__ out_point_x,
    double* __restrict__ out_point_y,
    double* __restrict__ out_overlap_x0,
    double* __restrict__ out_overlap_y0,
    double* __restrict__ out_overlap_x1,
    double* __restrict__ out_overlap_y1,
    const int row_count
) {{{{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = row < row_count;

    /* Phase 1: MBR overlap check with warp-cooperative skip */
    int has_overlap = 0;
    int left_index = 0;
    int right_index = 0;
    double ax, ay, bx, by, cx, cy, dx, dy;

    if (valid) {{{{
        left_index = left_lookup[row];
        right_index = right_lookup[row];

        ax = left_x0[left_index];
        ay = left_y0[left_index];
        bx = left_x1[left_index];
        by = left_y1[left_index];
        cx = right_x0[right_index];
        cy = right_y0[right_index];
        dx = right_x1[right_index];
        dy = right_y1[right_index];

        const double left_minx = ax < bx ? ax : bx;
        const double left_maxx = ax > bx ? ax : bx;
        const double left_miny = ay < by ? ay : by;
        const double left_maxy = ay > by ? ay : by;
        const double right_minx = cx < dx ? cx : dx;
        const double right_maxx = cx > dx ? cx : dx;
        const double right_miny = cy < dy ? cy : dy;
        const double right_maxy = cy > dy ? cy : dy;

        has_overlap = (left_minx <= right_maxx) && (left_maxx >= right_minx) &&
                      (left_miny <= right_maxy) && (left_maxy >= right_miny);
    }}}}

    /* Warp-level early exit: if no thread in warp has overlap, skip all */
    if (__ballot_sync(0xFFFFFFFF, has_overlap) == 0) {{{{
        if (valid) {{{{
            out_kind[row] = CLASS_DISJOINT;
            const double nan_value = 0.0 / 0.0;
            out_point_x[row] = nan_value;
            out_point_y[row] = nan_value;
            out_overlap_x0[row] = nan_value;
            out_overlap_y0[row] = nan_value;
            out_overlap_x1[row] = nan_value;
            out_overlap_y1[row] = nan_value;
        }}}}
        return;
    }}}}

    if (!valid) return;

    const double nan_value = 0.0 / 0.0;
    out_kind[row] = CLASS_DISJOINT;
    out_point_x[row] = nan_value;
    out_point_y[row] = nan_value;
    out_overlap_x0[row] = nan_value;
    out_overlap_y0[row] = nan_value;
    out_overlap_x1[row] = nan_value;
    out_overlap_y1[row] = nan_value;

    if (!has_overlap) return;

    /* Phase 2: Fast orientation filter */
    const compute_t abx = (compute_t)(bx - ax);
    const compute_t aby = (compute_t)(by - ay);
    const compute_t acx = (compute_t)(cx - ax);
    const compute_t acy = (compute_t)(cy - ay);
    const compute_t adx = (compute_t)(dx - ax);
    const compute_t ady = (compute_t)(dy - ay);
    const compute_t cdx = (compute_t)(dx - cx);
    const compute_t cdy = (compute_t)(dy - cy);
    const compute_t cax = (compute_t)(ax - cx);
    const compute_t cay = (compute_t)(ay - cy);
    const compute_t cbx = (compute_t)(bx - cx);
    const compute_t cby = (compute_t)(by - cy);

    const compute_t o1_term1 = abx * acy;
    const compute_t o1_term2 = aby * acx;
    const compute_t o1 = o1_term1 - o1_term2;
    const compute_t o2_term1 = abx * ady;
    const compute_t o2_term2 = aby * adx;
    const compute_t o2 = o2_term1 - o2_term2;
    const compute_t o3_term1 = cdx * cay;
    const compute_t o3_term2 = cdy * cax;
    const compute_t o3 = o3_term1 - o3_term2;
    const compute_t o4_term1 = cdx * cby;
    const compute_t o4_term2 = cdy * cbx;
    const compute_t o4 = o4_term1 - o4_term2;

    const compute_t errbound = (compute_t){errbound_val};
    const compute_t err1 = errbound * (abs_ct(o1_term1) + abs_ct(o1_term2));
    const compute_t err2 = errbound * (abs_ct(o2_term1) + abs_ct(o2_term2));
    const compute_t err3 = errbound * (abs_ct(o3_term1) + abs_ct(o3_term2));
    const compute_t err4 = errbound * (abs_ct(o4_term1) + abs_ct(o4_term2));

    const int fast_ambiguous =
        (abs_ct(o1) <= err1) ||
        (abs_ct(o2) <= err2) ||
        (abs_ct(o3) <= err3) ||
        (abs_ct(o4) <= err4);

    /* Degenerate segments (zero-length) */
    const int a_is_point = (ax == bx) && (ay == by);
    const int c_is_point = (cx == dx) && (cy == dy);

    if (!fast_ambiguous && !a_is_point && !c_is_point) {{{{
        /* Fast path: orientations are non-ambiguous */
        const int sign1 = (o1 > (compute_t)0.0) - (o1 < (compute_t)0.0);
        const int sign2 = (o2 > (compute_t)0.0) - (o2 < (compute_t)0.0);
        const int sign3 = (o3 > (compute_t)0.0) - (o3 < (compute_t)0.0);
        const int sign4 = (o4 > (compute_t)0.0) - (o4 < (compute_t)0.0);

        if (sign1 == 0 || sign2 == 0 || sign3 == 0 || sign4 == 0) {{{{
            /* Fall through to adaptive refinement */
        }}}} else if ((sign1 * sign2 < 0) && (sign3 * sign4 < 0)) {{{{
            /* Proper intersection */
            double ix = 0.0;
            double iy = 0.0;
            if (vs_proper_intersection_point_dd(
                ax, ay, bx, by, cx, cy, dx, dy, &ix, &iy
            )) {{{{
                out_kind[row] = CLASS_PROPER;
                out_point_x[row] = ix;
                out_point_y[row] = iy;
            }}}}
            return;
        }}}} else {{{{
            /* Disjoint: opposite sides confirmed */
            return;
        }}}}
    }}}}

    /* Phase 3: Shewchuk adaptive refinement on GPU (no host round-trip) */
    int s1 = vs_orient2d(ax, ay, bx, by, cx, cy);
    int s2 = vs_orient2d(ax, ay, bx, by, dx, dy);
    int s3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
    int s4 = vs_orient2d(cx, cy, dx, dy, bx, by);

    /* Handle degenerate (zero-length) segments */
    if (a_is_point && c_is_point) {{{{
        if (ax == cx && ay == cy) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = ax;
            out_point_y[row] = ay;
        }}}}
        return;
    }}}}
    if (a_is_point) {{{{
        if (s3 == 0 && vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy)) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = ax;
            out_point_y[row] = ay;
        }}}}
        return;
    }}}}
    if (c_is_point) {{{{
        if (s1 == 0 && vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by)) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = cx;
            out_point_y[row] = cy;
        }}}}
        return;
    }}}}

    /* Proper intersection: segments cross */
    if (s1 * s2 < 0 && s3 * s4 < 0) {{{{
        double ix = 0.0;
        double iy = 0.0;
        if (vs_proper_intersection_point_dd(
            ax, ay, bx, by, cx, cy, dx, dy, &ix, &iy
        )) {{{{
            out_kind[row] = CLASS_PROPER;
            out_point_x[row] = ix;
            out_point_y[row] = iy;
        }}}}
        return;
    }}}}

    /* Collinear: all four orientations zero */
    if (s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0) {{{{
        /* Find shared interval along the collinear segments */
        int a_on_cd = vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy);
        int b_on_cd = vs_point_on_segment_collinear(bx, by, cx, cy, dx, dy);
        int c_on_ab = vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by);
        int d_on_ab = vs_point_on_segment_collinear(dx, dy, ax, ay, bx, by);

        /* Collect shared endpoints */
        double pts_x[4];
        double pts_y[4];
        int n_pts = 0;
        if (a_on_cd) {{{{ pts_x[n_pts] = ax; pts_y[n_pts] = ay; ++n_pts; }}}}
        if (b_on_cd) {{{{ pts_x[n_pts] = bx; pts_y[n_pts] = by; ++n_pts; }}}}
        if (c_on_ab) {{{{ pts_x[n_pts] = cx; pts_y[n_pts] = cy; ++n_pts; }}}}
        if (d_on_ab) {{{{ pts_x[n_pts] = dx; pts_y[n_pts] = dy; ++n_pts; }}}}

        if (n_pts == 0) return; /* disjoint collinear */

        /* De-duplicate shared points */
        double ux[4];
        double uy[4];
        int n_unique = 0;
        for (int i = 0; i < n_pts; ++i) {{{{
            int dup = 0;
            for (int j = 0; j < n_unique; ++j) {{{{
                if (pts_x[i] == ux[j] && pts_y[i] == uy[j]) {{{{ dup = 1; break; }}}}
            }}}}
            if (!dup) {{{{ ux[n_unique] = pts_x[i]; uy[n_unique] = pts_y[i]; ++n_unique; }}}}
        }}}}

        if (n_unique == 1) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = ux[0];
            out_point_y[row] = uy[0];
        }}}} else {{{{
            /* Sort by dominant axis to find extent */
            int use_x = (abs_ct((compute_t)(bx - ax)) >= abs_ct((compute_t)(by - ay))) ? 1 : 0;
            /* Simple insertion sort on up to 4 points */
            for (int i = 1; i < n_unique; ++i) {{{{
                double kx = ux[i], ky = uy[i];
                int j = i - 1;
                while (j >= 0) {{{{
                    int swap = use_x ? (ux[j] > kx || (ux[j] == kx && uy[j] > ky))
                                     : (uy[j] > ky || (uy[j] == ky && ux[j] > kx));
                    if (!swap) break;
                    ux[j + 1] = ux[j]; uy[j + 1] = uy[j];
                    --j;
                }}}}
                ux[j + 1] = kx; uy[j + 1] = ky;
            }}}}
            out_kind[row] = CLASS_OVERLAP;
            out_overlap_x0[row] = ux[0];
            out_overlap_y0[row] = uy[0];
            out_overlap_x1[row] = ux[n_unique - 1];
            out_overlap_y1[row] = uy[n_unique - 1];
        }}}}
        return;
    }}}}

    /* Touch cases: one endpoint lies on the other segment */
    if (s1 == 0 && vs_point_on_segment_collinear(cx, cy, ax, ay, bx, by)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = cx;
        out_point_y[row] = cy;
        return;
    }}}}
    if (s2 == 0 && vs_point_on_segment_collinear(dx, dy, ax, ay, bx, by)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = dx;
        out_point_y[row] = dy;
        return;
    }}}}
    if (s3 == 0 && vs_point_on_segment_collinear(ax, ay, cx, cy, dx, dy)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = ax;
        out_point_y[row] = ay;
        return;
    }}}}
    if (s4 == 0 && vs_point_on_segment_collinear(bx, by, cx, cy, dx, dy)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = bx;
        out_point_y[row] = by;
        return;
    }}}}
    /* Disjoint: no intersection */
}}}}
"""

_SEGMENT_CLASSIFY_KERNEL_NAMES = ("classify_segment_pairs_v2",)


# ---------------------------------------------------------------------------
# Candidate scatter kernel: expand (range_start, range_end) into pair arrays
# entirely on device, avoiding the D->H->D ping-pong.
# ---------------------------------------------------------------------------

_SAME_ROW_CANDIDATE_KERNEL_SOURCE = """
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_same_row_overlap_candidates_capacity(
    const int* __restrict__ left_rows,
    const double* __restrict__ left_x0,
    const double* __restrict__ left_y0,
    const double* __restrict__ left_x1,
    const double* __restrict__ left_y1,
    const int* __restrict__ right_row_starts,
    const int* __restrict__ right_row_ends,
    const double* __restrict__ right_x0,
    const double* __restrict__ right_y0,
    const double* __restrict__ right_x1,
    const double* __restrict__ right_y1,
    const int* __restrict__ upper_left_rows,
    const int* __restrict__ upper_right_rows,
    const int use_upper_rows,
    int* __restrict__ out_left,
    int* __restrict__ out_right,
    const int left_start,
    const int batch_size,
    const int right_capacity
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (warp >= batch_size) return;

    const int left_idx = left_start + warp;
    const int row = left_rows[left_idx];
    const int rs = right_row_starts[row];
    const int re = right_row_ends[row];
    if (rs < 0 || re <= rs) return;

    const double lx0 = left_x0[left_idx];
    const double ly0 = left_y0[left_idx];
    const double lx1 = left_x1[left_idx];
    const double ly1 = left_y1[left_idx];
    const double lminx = fmin(lx0, lx1);
    const double lmaxx = fmax(lx0, lx1);
    const double lminy = fmin(ly0, ly1);
    const double lmaxy = fmax(ly0, ly1);
    const int base_write = warp * right_capacity;
    int written = 0;

    for (int base = rs; base < re; base += 32) {
        const int ridx = base + lane;
        int hit = 0;
        if (ridx < re) {
            const double rx0 = right_x0[ridx];
            const double ry0 = right_y0[ridx];
            const double rx1 = right_x1[ridx];
            const double ry1 = right_y1[ridx];
            const double rminx = fmin(rx0, rx1);
            const double rmaxx = fmax(rx0, rx1);
            const double rminy = fmin(ry0, ry1);
            const double rmaxy = fmax(ry0, ry1);
            hit = (lminx <= rmaxx) && (lmaxx >= rminx) &&
                  (lminy <= rmaxy) && (lmaxy >= rminy);
            if (hit && use_upper_rows != 0 && upper_left_rows[left_idx] >= upper_right_rows[ridx]) {
                hit = 0;
            }
        }

        const unsigned mask = __ballot_sync(0xFFFFFFFFu, hit);
        if (!mask) continue;

        const int prefix = __popc(mask & ((1u << lane) - 1));
        const int total = __popc(mask);
        int chunk_base = 0;
        if (lane == 0) {
            chunk_base = base_write + written;
            written += total;
        }
        chunk_base = __shfl_sync(0xFFFFFFFFu, chunk_base, 0);
        if (hit) {
            out_left[chunk_base + prefix] = left_idx;
            out_right[chunk_base + prefix] = ridx;
        }
    }
}
"""

_SAME_ROW_CANDIDATE_KERNEL_NAMES = ("scatter_same_row_overlap_candidates_capacity",)


_SWEEP_CANDIDATE_KERNEL_SOURCE = r"""
__device__ __forceinline__ int sweep_candidate_overlaps(
    const int left_idx,
    const int right_idx,
    const double* left_minx,
    const double* left_maxx,
    const double* left_miny,
    const double* left_maxy,
    const double* right_minx,
    const double* right_maxx,
    const double* right_miny,
    const double* right_maxy,
    const int* left_rows,
    const int* right_rows,
    const int require_same_row,
    const unsigned char* outlier_mask,
    const int exclude_outliers,
    const int* upper_left_rows,
    const int* upper_right_rows,
    const int use_upper_rows
) {
    if (
        left_minx[left_idx] > right_maxx[right_idx]
        || left_maxx[left_idx] < right_minx[right_idx]
        || left_miny[left_idx] > right_maxy[right_idx]
        || left_maxy[left_idx] < right_miny[right_idx]
    ) return 0;
    if (require_same_row && left_rows[left_idx] != right_rows[right_idx]) return 0;
    if (exclude_outliers && outlier_mask[right_idx]) return 0;
    if (use_upper_rows && upper_left_rows[left_idx] >= upper_right_rows[right_idx]) return 0;
    return 1;
}

extern "C" __global__ void __launch_bounds__(128, 4)
count_sweep_overlap_candidates(
    const long long* range_start,
    const long long* range_end,
    const int* sorted_right_idx,
    const double* left_minx,
    const double* left_maxx,
    const double* left_miny,
    const double* left_maxy,
    const double* right_minx,
    const double* right_maxx,
    const double* right_miny,
    const double* right_maxy,
    const int* left_rows,
    const int* right_rows,
    const int require_same_row,
    const unsigned char* outlier_mask,
    const int exclude_outliers,
    const int* upper_left_rows,
    const int* upper_right_rows,
    const int use_upper_rows,
    const int left_start,
    const int batch_size,
    const int range_offset,
    const int range_capacity,
    int* out_counts
) {
    const int local = blockIdx.x;
    if (local >= batch_size) return;
    const int left_idx = left_start + local;
    const long long base = range_start[left_idx];
    const long long start = base + (long long)range_offset;
    const long long stop = min(
        range_end[left_idx],
        start + (long long)range_capacity
    );
    unsigned int local_count = 0;
    for (long long pos = start + threadIdx.x; pos < stop; pos += blockDim.x) {
        const int right_idx = sorted_right_idx[pos];
        local_count += (unsigned int)sweep_candidate_overlaps(
            left_idx, right_idx,
            left_minx, left_maxx, left_miny, left_maxy,
            right_minx, right_maxx, right_miny, right_maxy,
            left_rows, right_rows, require_same_row,
            outlier_mask, exclude_outliers,
            upper_left_rows, upper_right_rows, use_upper_rows
        );
    }
    if (local_count) atomicAdd((unsigned int*)&out_counts[local], local_count);
}

extern "C" __global__ void __launch_bounds__(128, 4)
scatter_sweep_overlap_candidates(
    const long long* range_start,
    const long long* range_end,
    const int* sorted_right_idx,
    const double* left_minx,
    const double* left_maxx,
    const double* left_miny,
    const double* left_maxy,
    const double* right_minx,
    const double* right_maxx,
    const double* right_miny,
    const double* right_maxy,
    const int* left_rows,
    const int* right_rows,
    const int require_same_row,
    const unsigned char* outlier_mask,
    const int exclude_outliers,
    const int* upper_left_rows,
    const int* upper_right_rows,
    const int use_upper_rows,
    const int* selected_left_indices,
    const int selected_count,
    const int range_offset,
    const int range_capacity,
    const int tier_start,
    const int tier_width,
    int* out_left,
    int* out_right
) {
    const int local = blockIdx.x;
    if (local >= selected_count) return;
    const int lane = threadIdx.x & 31;
    const int left_idx = selected_left_indices[local];
    const long long base = range_start[left_idx];
    const long long start = base + (long long)range_offset;
    const long long stop = min(
        range_end[left_idx],
        start + (long long)range_capacity
    );
    int written = 0;
    for (long long base_pos = start; base_pos < stop; base_pos += 32) {
        const long long pos = base_pos + lane;
        int hit = 0;
        int right_idx = -1;
        if (pos < stop) {
            right_idx = sorted_right_idx[pos];
            hit = sweep_candidate_overlaps(
                left_idx, right_idx,
                left_minx, left_maxx, left_miny, left_maxy,
                right_minx, right_maxx, right_miny, right_maxy,
                left_rows, right_rows, require_same_row,
                outlier_mask, exclude_outliers,
                upper_left_rows, upper_right_rows, use_upper_rows
            );
        }
        const unsigned int hit_mask = __ballot_sync(0xFFFFFFFFu, hit);
        const int prefix = __popc(hit_mask & ((1u << lane) - 1));
        const int ordinal = written + prefix;
        if (hit && ordinal >= tier_start && ordinal < tier_start + tier_width) {
            const int output = local * tier_width + ordinal - tier_start;
            out_left[output] = left_idx;
            out_right[output] = right_idx;
        }
        written += __popc(hit_mask);
    }
}
"""

_SWEEP_CANDIDATE_KERNEL_NAMES = (
    "count_sweep_overlap_candidates",
    "scatter_sweep_overlap_candidates",
)


# ---------------------------------------------------------------------------
# Format-string generators (produce CUDA C++ from templates)
# ---------------------------------------------------------------------------


def format_extract_source(compute_type: str = "double") -> str:
    """Format the segment extraction kernel source with the given compute type."""
    return _SEGMENT_EXTRACT_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        family_linestring=_FAMILY_LINESTRING,
        family_polygon=_FAMILY_POLYGON,
        family_multilinestring=_FAMILY_MULTILINESTRING,
        family_multipolygon=_FAMILY_MULTIPOLYGON,
    )


def format_classify_source(compute_type: str = "double") -> str:
    """Format the classify kernel source with the given compute type and errbound."""
    if compute_type == "float":
        eps = float(np.finfo(np.float32).eps)
    else:
        eps = float(np.finfo(np.float64).eps)
    errbound = (3.0 + 16.0 * eps) * eps
    formatted = _SEGMENT_CLASSIFY_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        errbound_val=f"{errbound:.20e}",
    )
    return ORIENT2D_DEVICE + SEGMENT_CROSSING_DEVICE + INTERSECTION_POINT_DEVICE + formatted


# Pre-formatted kernel sources for warmup
EXTRACT_SOURCE_FP64 = format_extract_source("double")
EXTRACT_SOURCE_FP32 = format_extract_source("float")
CLASSIFY_SOURCE_FP64 = format_classify_source("double")
CLASSIFY_SOURCE_FP32 = format_classify_source("float")
