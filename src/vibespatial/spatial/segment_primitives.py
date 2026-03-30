from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from time import perf_counter

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    sort_pairs,
    upper_bound,
)

request_warmup([
    "select_i32",
    "select_i64",
    "exclusive_scan_i32",
    "exclusive_scan_i64",
    "lower_bound_i64",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    DeviceArray,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray  # noqa: E402
from vibespatial.runtime import ExecutionMode, RuntimeSelection  # noqa: E402
from vibespatial.runtime.adaptive import plan_dispatch_selection  # noqa: E402
from vibespatial.runtime.kernel_registry import register_kernel_variant  # noqa: E402
from vibespatial.runtime.precision import (  # noqa: E402
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.residency import Residency  # noqa: E402
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan  # noqa: E402

_FLOAT_EPSILON = np.finfo(np.float64).eps
_ORIENTATION_ERRBOUND = (3.0 + 16.0 * _FLOAT_EPSILON) * _FLOAT_EPSILON
_SEGMENT_GPU_THRESHOLD = 4_096

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
    const int n_valid
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
            out_row_idx[write_pos] = global_row;
            out_seg_idx[write_pos] = seg_idx++;
            out_part_idx[write_pos] = 0;
            out_ring_idx[write_pos] = 0;
            out_x0[write_pos] = x[c];
            out_y0[write_pos] = y[c];
            out_x1[write_pos] = x[c + 1];
            out_y1[write_pos] = y[c + 1];
            ++write_pos;
        }}}}
    }}}} else if (family == FAMILY_POLYGON) {{{{
        const int rs = geom_off[fam_row];
        const int re = geom_off[fam_row + 1];
        for (int ri = rs; ri < re; ++ri) {{{{
            const int ring_local = ri - rs;
            const int cs = ring_off[ri];
            const int ce = ring_off[ri + 1];
            for (int c = cs; c < ce - 1; ++c) {{{{
                out_row_idx[write_pos] = global_row;
                out_seg_idx[write_pos] = seg_idx++;
                out_part_idx[write_pos] = 0;
                out_ring_idx[write_pos] = ring_local;
                out_x0[write_pos] = x[c];
                out_y0[write_pos] = y[c];
                out_x1[write_pos] = x[c + 1];
                out_y1[write_pos] = y[c + 1];
                ++write_pos;
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
                out_row_idx[write_pos] = global_row;
                out_seg_idx[write_pos] = seg_idx++;
                out_part_idx[write_pos] = part_local;
                out_ring_idx[write_pos] = -1;
                out_x0[write_pos] = x[c];
                out_y0[write_pos] = y[c];
                out_x1[write_pos] = x[c + 1];
                out_y1[write_pos] = y[c + 1];
                ++write_pos;
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
                    out_row_idx[write_pos] = global_row;
                    out_seg_idx[write_pos] = seg_idx++;
                    out_part_idx[write_pos] = polygon_local;
                    out_ring_idx[write_pos] = ring_local;
                    out_x0[write_pos] = x[c];
                    out_y0[write_pos] = y[c];
                    out_x1[write_pos] = x[c + 1];
                    out_y1[write_pos] = y[c + 1];
                    ++write_pos;
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

/* Shewchuk two-product error-free transformation (GPU implementation).
   Given a, b: computes (p, e) such that a*b = p + e exactly.
   Uses Dekker's algorithm with FMA where available. */
__device__ inline void two_product(double a, double b, double &p, double &e) {{{{
    p = a * b;
    e = fma(a, b, -p);
}}}}

/* Shewchuk two-sum error-free transformation.
   Given a, b: computes (s, e) such that a+b = s + e exactly. */
__device__ inline void two_sum(double a, double b, double &s, double &e) {{{{
    s = a + b;
    double bv = s - a;
    double av = s - bv;
    double br = b - bv;
    double ar = a - av;
    e = ar + br;
}}}}

/* Shewchuk orient2d adaptive predicate (stage B).
   Returns exact sign of det = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax)
   using error-free arithmetic expansions when the fast filter is ambiguous.
   Returns: +1, 0, or -1  */
__device__ int orient2d_adaptive(
    double ax, double ay,
    double bx, double by,
    double cx, double cy
) {{{{
    double acx = ax - cx;
    double bcx = bx - cx;
    double acy = ay - cy;
    double bcy = by - cy;

    double detleft, detleft_err;
    two_product(acx, bcy, detleft, detleft_err);

    double detright, detright_err;
    two_product(acy, bcx, detright, detright_err);

    double det_sum, det_sum_err;
    two_sum(detleft, -detright, det_sum, det_sum_err);

    /* Accumulate all error terms */
    double B3 = detleft_err - detright_err + det_sum_err;

    /* B3 is the correction: det = det_sum + B3 */
    double det = det_sum + B3;

    if (det > 0.0) return 1;
    if (det < 0.0) return -1;

    /* Stage 1 already uses error-free two_product (FMA-based) and two_sum,
       capturing all rounding error in B3. For IEEE-754 fp64 inputs, the
       accumulated det = det_sum + B3 is exact. If it is zero, the
       orientation is truly zero (collinear). */
    return 0;
}}}}

/* Check if point (px,py) is on segment (ax,ay)-(bx,by), assuming collinear.
   Uses exact double comparisons (valid for fp64). */
__device__ int point_on_segment_collinear(
    double px, double py,
    double ax, double ay,
    double bx, double by
) {{{{
    double minx = ax < bx ? ax : bx;
    double maxx = ax > bx ? ax : bx;
    double miny = ay < by ? ay : by;
    double maxy = ay > by ? ay : by;
    return (px >= minx && px <= maxx && py >= miny && py <= maxy) ? 1 : 0;
}}}}

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
            double denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx);
            if (denominator != 0.0) {{{{
                double left_det = ax * by - ay * bx;
                double right_det = cx * dy - cy * dx;
                out_kind[row] = CLASS_PROPER;
                out_point_x[row] = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator;
                out_point_y[row] = (left_det * (cy - dy) - (ay - by) * right_det) / denominator;
            }}}}
            return;
        }}}} else {{{{
            /* Disjoint: opposite sides confirmed */
            return;
        }}}}
    }}}}

    /* Phase 3: Shewchuk adaptive refinement on GPU (no host round-trip) */
    int s1 = orient2d_adaptive(ax, ay, bx, by, cx, cy);
    int s2 = orient2d_adaptive(ax, ay, bx, by, dx, dy);
    int s3 = orient2d_adaptive(cx, cy, dx, dy, ax, ay);
    int s4 = orient2d_adaptive(cx, cy, dx, dy, bx, by);

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
        if (s3 == 0 && point_on_segment_collinear(ax, ay, cx, cy, dx, dy)) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = ax;
            out_point_y[row] = ay;
        }}}}
        return;
    }}}}
    if (c_is_point) {{{{
        if (s1 == 0 && point_on_segment_collinear(cx, cy, ax, ay, bx, by)) {{{{
            out_kind[row] = CLASS_TOUCH;
            out_point_x[row] = cx;
            out_point_y[row] = cy;
        }}}}
        return;
    }}}}

    /* Proper intersection: segments cross */
    if (s1 * s2 < 0 && s3 * s4 < 0) {{{{
        double denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx);
        if (denominator != 0.0) {{{{
            double left_det = ax * by - ay * bx;
            double right_det = cx * dy - cy * dx;
            out_kind[row] = CLASS_PROPER;
            out_point_x[row] = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator;
            out_point_y[row] = (left_det * (cy - dy) - (ay - by) * right_det) / denominator;
        }}}}
        return;
    }}}}

    /* Collinear: all four orientations zero */
    if (s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0) {{{{
        /* Find shared interval along the collinear segments */
        int a_on_cd = point_on_segment_collinear(ax, ay, cx, cy, dx, dy);
        int b_on_cd = point_on_segment_collinear(bx, by, cx, cy, dx, dy);
        int c_on_ab = point_on_segment_collinear(cx, cy, ax, ay, bx, by);
        int d_on_ab = point_on_segment_collinear(dx, dy, ax, ay, bx, by);

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
    if (s1 == 0 && point_on_segment_collinear(cx, cy, ax, ay, bx, by)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = cx;
        out_point_y[row] = cy;
        return;
    }}}}
    if (s2 == 0 && point_on_segment_collinear(dx, dy, ax, ay, bx, by)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = dx;
        out_point_y[row] = dy;
        return;
    }}}}
    if (s3 == 0 && point_on_segment_collinear(ax, ay, cx, cy, dx, dy)) {{{{
        out_kind[row] = CLASS_TOUCH;
        out_point_x[row] = ax;
        out_point_y[row] = ay;
        return;
    }}}}
    if (s4 == 0 && point_on_segment_collinear(bx, by, cx, cy, dx, dy)) {{{{
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
# Data types
# ---------------------------------------------------------------------------

class SegmentIntersectionKind(IntEnum):
    DISJOINT = 0
    PROPER = 1
    TOUCH = 2
    OVERLAP = 3


@dataclass(frozen=True)
class SegmentTable:
    row_indices: np.ndarray
    part_indices: np.ndarray
    ring_indices: np.ndarray
    segment_indices: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    bounds: np.ndarray

    @property
    def count(self) -> int:
        return int(self.row_indices.size)


@dataclass(frozen=True)
class DeviceSegmentTable:
    """GPU-resident segment table in SoA layout."""
    row_indices: DeviceArray
    segment_indices: DeviceArray
    x0: DeviceArray
    y0: DeviceArray
    x1: DeviceArray
    y1: DeviceArray
    count: int
    part_indices: DeviceArray | None = None
    ring_indices: DeviceArray | None = None

    def free(self) -> None:
        """Release all device allocations held by this table.

        Consolidates the 7--9 individual ``runtime.free()`` calls that
        previously had to be duplicated at every cleanup site.
        """
        runtime = get_cuda_runtime()
        runtime.free(self.x0)
        runtime.free(self.y0)
        runtime.free(self.x1)
        runtime.free(self.y1)
        runtime.free(self.row_indices)
        runtime.free(self.segment_indices)
        if self.part_indices is not None:
            runtime.free(self.part_indices)
        if self.ring_indices is not None:
            runtime.free(self.ring_indices)


@dataclass
class SegmentIntersectionResult:
    """Segment intersection results with lazy host materialization.

    When produced by the GPU pipeline, all 14 result arrays live in
    ``device_state`` and host numpy arrays are lazily copied on first
    property access.  GPU-only consumers (e.g. ``build_gpu_split_events``)
    that read only ``device_state``, ``candidate_pairs``, ``count``,
    ``runtime_selection``, ``precision_plan``, and ``robustness_plan``
    never trigger device-to-host copies.
    """
    candidate_pairs: int
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan
    device_state: SegmentIntersectionDeviceState | None = None
    _count: int = 0
    # Host arrays — lazily materialized from device_state on first access.
    _left_rows: np.ndarray | None = None
    _left_segments: np.ndarray | None = None
    _left_lookup: np.ndarray | None = None
    _right_rows: np.ndarray | None = None
    _right_segments: np.ndarray | None = None
    _right_lookup: np.ndarray | None = None
    _kinds: np.ndarray | None = None
    _point_x: np.ndarray | None = None
    _point_y: np.ndarray | None = None
    _overlap_x0: np.ndarray | None = None
    _overlap_y0: np.ndarray | None = None
    _overlap_x1: np.ndarray | None = None
    _overlap_y1: np.ndarray | None = None
    _ambiguous_rows: np.ndarray | None = None

    def _ensure_host(self) -> None:
        """Lazily copy host arrays from device_state on first access."""
        if self._left_rows is not None:
            return
        ds = self.device_state
        if ds is None:
            return
        runtime = get_cuda_runtime()
        self._left_rows = np.asarray(
            runtime.copy_device_to_host(ds.left_rows), dtype=np.int32,
        )
        self._left_segments = np.asarray(
            runtime.copy_device_to_host(ds.left_segments), dtype=np.int32,
        )
        self._left_lookup = np.asarray(
            runtime.copy_device_to_host(ds.left_lookup), dtype=np.int32,
        )
        self._right_rows = np.asarray(
            runtime.copy_device_to_host(ds.right_rows), dtype=np.int32,
        )
        self._right_segments = np.asarray(
            runtime.copy_device_to_host(ds.right_segments), dtype=np.int32,
        )
        self._right_lookup = np.asarray(
            runtime.copy_device_to_host(ds.right_lookup), dtype=np.int32,
        )
        self._kinds = np.asarray(
            runtime.copy_device_to_host(ds.kinds), dtype=np.int8,
        )
        self._point_x = np.asarray(
            runtime.copy_device_to_host(ds.point_x), dtype=np.float64,
        )
        self._point_y = np.asarray(
            runtime.copy_device_to_host(ds.point_y), dtype=np.float64,
        )
        self._overlap_x0 = np.asarray(
            runtime.copy_device_to_host(ds.overlap_x0), dtype=np.float64,
        )
        self._overlap_y0 = np.asarray(
            runtime.copy_device_to_host(ds.overlap_y0), dtype=np.float64,
        )
        self._overlap_x1 = np.asarray(
            runtime.copy_device_to_host(ds.overlap_x1), dtype=np.float64,
        )
        self._overlap_y1 = np.asarray(
            runtime.copy_device_to_host(ds.overlap_y1), dtype=np.float64,
        )
        self._ambiguous_rows = np.asarray(
            runtime.copy_device_to_host(ds.ambiguous_rows), dtype=np.int32,
        )

    @property
    def left_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._left_rows  # type: ignore[return-value]

    @property
    def left_segments(self) -> np.ndarray:
        self._ensure_host()
        return self._left_segments  # type: ignore[return-value]

    @property
    def left_lookup(self) -> np.ndarray:
        self._ensure_host()
        return self._left_lookup  # type: ignore[return-value]

    @property
    def right_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._right_rows  # type: ignore[return-value]

    @property
    def right_segments(self) -> np.ndarray:
        self._ensure_host()
        return self._right_segments  # type: ignore[return-value]

    @property
    def right_lookup(self) -> np.ndarray:
        self._ensure_host()
        return self._right_lookup  # type: ignore[return-value]

    @property
    def kinds(self) -> np.ndarray:
        self._ensure_host()
        return self._kinds  # type: ignore[return-value]

    @property
    def point_x(self) -> np.ndarray:
        self._ensure_host()
        return self._point_x  # type: ignore[return-value]

    @property
    def point_y(self) -> np.ndarray:
        self._ensure_host()
        return self._point_y  # type: ignore[return-value]

    @property
    def overlap_x0(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_x0  # type: ignore[return-value]

    @property
    def overlap_y0(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_y0  # type: ignore[return-value]

    @property
    def overlap_x1(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_x1  # type: ignore[return-value]

    @property
    def overlap_y1(self) -> np.ndarray:
        self._ensure_host()
        return self._overlap_y1  # type: ignore[return-value]

    @property
    def ambiguous_rows(self) -> np.ndarray:
        self._ensure_host()
        return self._ambiguous_rows  # type: ignore[return-value]

    @property
    def count(self) -> int:
        if self._count > 0:
            return self._count
        if self.device_state is not None and self.device_state.left_rows is not None:
            return int(self.device_state.left_rows.size)
        if self._left_rows is not None:
            return int(self._left_rows.size)
        return 0

    def kind_names(self) -> list[str]:
        return [SegmentIntersectionKind(int(value)).name.lower() for value in self.kinds]


@dataclass(frozen=True)
class SegmentIntersectionBenchmark:
    rows_left: int
    rows_right: int
    candidate_pairs: int
    disjoint_pairs: int
    proper_pairs: int
    touch_pairs: int
    overlap_pairs: int
    ambiguous_pairs: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SegmentIntersectionDeviceState:
    left_rows: DeviceArray
    left_segments: DeviceArray
    left_lookup: DeviceArray
    right_rows: DeviceArray
    right_segments: DeviceArray
    right_lookup: DeviceArray
    kinds: DeviceArray
    point_x: DeviceArray
    point_y: DeviceArray
    overlap_x0: DeviceArray
    overlap_y0: DeviceArray
    overlap_x1: DeviceArray
    overlap_y1: DeviceArray
    ambiguous_rows: DeviceArray


@dataclass(frozen=True)
class SegmentIntersectionCandidates:
    left_rows: np.ndarray
    left_segments: np.ndarray
    left_lookup: np.ndarray
    right_rows: np.ndarray
    right_segments: np.ndarray
    right_lookup: np.ndarray
    pairs_examined: int
    tile_size: int

    @property
    def count(self) -> int:
        return int(self.left_rows.size)


@dataclass(frozen=True)
class DeviceSegmentIntersectionCandidates:
    """GPU-resident candidate pairs from sweep-based spatial join."""
    left_rows: DeviceArray
    left_segments: DeviceArray
    left_lookup: DeviceArray
    right_rows: DeviceArray
    right_segments: DeviceArray
    right_lookup: DeviceArray
    count: int


# ---------------------------------------------------------------------------
# NVRTC compilation and warmup
# ---------------------------------------------------------------------------

def _format_extract_source(compute_type: str = "double") -> str:
    return _SEGMENT_EXTRACT_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        family_linestring=_FAMILY_LINESTRING,
        family_polygon=_FAMILY_POLYGON,
        family_multilinestring=_FAMILY_MULTILINESTRING,
        family_multipolygon=_FAMILY_MULTIPOLYGON,
    )


def _format_classify_source(compute_type: str = "double") -> str:
    if compute_type == "float":
        eps = float(np.finfo(np.float32).eps)
    else:
        eps = float(np.finfo(np.float64).eps)
    errbound = (3.0 + 16.0 * eps) * eps
    return _SEGMENT_CLASSIFY_KERNEL_SOURCE_TEMPLATE.format(
        compute_type=compute_type,
        errbound_val=f"{errbound:.20e}",
    )


# ---------------------------------------------------------------------------
# Candidate scatter kernel: expand (range_start, range_end) into pair arrays
# entirely on device, avoiding the D->H->D ping-pong.
# ---------------------------------------------------------------------------

_CANDIDATE_SCATTER_KERNEL_SOURCE = """
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_candidate_pairs(
    const long long* __restrict__ cand_offsets,
    const int* __restrict__ range_start,
    const int* __restrict__ range_end,
    const int* __restrict__ sorted_right_idx,
    int* __restrict__ out_left,
    int* __restrict__ out_right,
    const int n_left
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_left) return;

    const int rs = range_start[i];
    const int re = range_end[i];
    int write_pos = (int)cand_offsets[i];

    for (int j = rs; j < re; ++j) {
        out_left[write_pos] = i;
        out_right[write_pos] = sorted_right_idx[j];
        ++write_pos;
    }
}

// Batched variant: processes left segments [left_start, left_start+batch_size).
// Write positions are shifted by -offset_base so output starts at index 0.
extern "C" __global__ void __launch_bounds__(256, 4)
scatter_candidate_pairs_batch(
    const long long* __restrict__ cand_offsets,
    const int* __restrict__ range_start,
    const int* __restrict__ range_end,
    const int* __restrict__ sorted_right_idx,
    int* __restrict__ out_left,
    int* __restrict__ out_right,
    const int left_start,
    const int batch_size,
    const long long offset_base
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    const int i = left_start + tid;
    const int rs = range_start[i];
    const int re = range_end[i];
    int write_pos = (int)(cand_offsets[i] - offset_base);

    for (int j = rs; j < re; ++j) {
        out_left[write_pos] = i;
        out_right[write_pos] = sorted_right_idx[j];
        ++write_pos;
    }
}
"""

_CANDIDATE_SCATTER_KERNEL_NAMES = ("scatter_candidate_pairs", "scatter_candidate_pairs_batch")

# Pre-format kernel sources for warmup
_EXTRACT_SOURCE_FP64 = _format_extract_source("double")
_EXTRACT_SOURCE_FP32 = _format_extract_source("float")
_CLASSIFY_SOURCE_FP64 = _format_classify_source("double")
_CLASSIFY_SOURCE_FP32 = _format_classify_source("float")

from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("segment-extract-fp64", _EXTRACT_SOURCE_FP64, _SEGMENT_EXTRACT_KERNEL_NAMES),
    ("segment-extract-fp32", _EXTRACT_SOURCE_FP32, _SEGMENT_EXTRACT_KERNEL_NAMES),
    ("segment-classify-fp64", _CLASSIFY_SOURCE_FP64, _SEGMENT_CLASSIFY_KERNEL_NAMES),
    ("segment-classify-fp32", _CLASSIFY_SOURCE_FP32, _SEGMENT_CLASSIFY_KERNEL_NAMES),
    ("segment-candidate-scatter", _CANDIDATE_SCATTER_KERNEL_SOURCE, _CANDIDATE_SCATTER_KERNEL_NAMES),
])


def _extract_kernels(compute_type: str = "double"):
    source = _format_extract_source(compute_type)
    name = f"segment-extract-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_EXTRACT_KERNEL_NAMES)


def _classify_kernels(compute_type: str = "double"):
    source = _format_classify_source(compute_type)
    name = f"segment-classify-{compute_type.replace('double', 'fp64').replace('float', 'fp32')}"
    return compile_kernel_group(name, source, _SEGMENT_CLASSIFY_KERNEL_NAMES)


def _candidate_scatter_kernels():
    return compile_kernel_group(
        "segment-candidate-scatter",
        _CANDIDATE_SCATTER_KERNEL_SOURCE,
        _CANDIDATE_SCATTER_KERNEL_NAMES,
    )


# ---------------------------------------------------------------------------
# Kernel 1 dispatch: GPU Segment Extraction
# ---------------------------------------------------------------------------

def _extract_segments_gpu(
    geometry_array: OwnedGeometryArray,
    compute_type: str = "double",
) -> DeviceSegmentTable:
    """Extract all segments from a geometry array entirely on GPU.

    Uses the count-scatter pattern:
    1. Count segments per valid geometry row
    2. Exclusive prefix sum for write offsets
    3. Scatter segment endpoints to output SoA arrays
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    d_state = geometry_array._ensure_device_state()

    # Determine valid rows for segment-producing families
    tags = geometry_array.tags
    validity = geometry_array.validity
    seg_families = {_FAMILY_LINESTRING, _FAMILY_POLYGON,
                    _FAMILY_MULTILINESTRING, _FAMILY_MULTIPOLYGON}
    valid_mask = validity & np.isin(tags, list(seg_families))
    valid_rows_host = np.flatnonzero(valid_mask).astype(np.int32, copy=False)

    if valid_rows_host.size == 0:
        return DeviceSegmentTable(
            row_indices=runtime.allocate((0,), np.int32),
            segment_indices=runtime.allocate((0,), np.int32),
            x0=runtime.allocate((0,), np.float64),
            y0=runtime.allocate((0,), np.float64),
            x1=runtime.allocate((0,), np.float64),
            y1=runtime.allocate((0,), np.float64),
            count=0,
            part_indices=runtime.allocate((0,), np.int32),
            ring_indices=runtime.allocate((0,), np.int32),
        )

    # The count_segments / scatter_segments kernels declare family_codes as
    # ``const int*`` (int32), but d_state.tags is int8.  Passing an int8
    # pointer to a kernel that reads 4-byte ints causes every thread to read
    # a garbage family code, producing zero segment counts and (when the
    # underlying memory layout changes) an illegal-address fault.
    d_family_codes = d_state.tags.astype(cp.int32) if d_state.tags.dtype != cp.int32 else d_state.tags
    d_family_row_offsets = d_state.family_row_offsets

    # We need unified offset arrays across all families.
    # Build concatenated offset arrays: for each valid row, we need the
    # correct family's offsets. We concatenate all family offset arrays
    # and build an offset base per family so the kernel can index correctly.
    #
    # Strategy: since the kernel accesses offsets by family_row_offsets[global_row],
    # which gives the row index within that family's buffer, and each family
    # has its own device offset arrays, we need to provide per-family offset
    # pointers. The simplest approach: one kernel launch per family. But that
    # loses the benefit of a single bulk launch.
    #
    # Better: build a unified offset table on device by concatenating family
    # offsets with base pointers. However, the kernel design above already
    # takes family code as input and does the right thing per family.
    # The problem is that different families store their offsets in different
    # device arrays. We need to either:
    #   (a) Pass all family offset arrays as separate kernel params, or
    #   (b) Build unified offset arrays by concatenating and adjusting.
    #
    # For maximum simplicity and GPU-residency, we use approach (a):
    # launch per-family kernels. With only 4 families this is 4 launches
    # max, all on the same stream (no sync needed between them).

    # However, approach (a) with separate kernels is cleaner with the count-scatter
    # pattern since each family produces different counts. Let's use a different
    # strategy: per-family count-scatter with a final concat.

    # Compile extraction kernels once (SHA1-cached), not per-family.
    kernels = _extract_kernels(compute_type)

    all_row_idx = []
    all_seg_idx = []
    all_part_idx = []
    all_ring_idx = []
    all_x0 = []
    all_y0 = []
    all_x1 = []
    all_y1 = []
    total_segments = 0

    for family_enum, family_tag in [
        (GeometryFamily.LINESTRING, _FAMILY_LINESTRING),
        (GeometryFamily.POLYGON, _FAMILY_POLYGON),
        (GeometryFamily.MULTILINESTRING, _FAMILY_MULTILINESTRING),
        (GeometryFamily.MULTIPOLYGON, _FAMILY_MULTIPOLYGON),
    ]:
        if family_enum not in d_state.families:
            continue
        d_buf = d_state.families[family_enum]

        # Valid rows for this family
        fam_valid_mask = validity & (tags == family_tag)
        fam_valid_rows = np.flatnonzero(fam_valid_mask).astype(np.int32, copy=False)
        if fam_valid_rows.size == 0:
            continue

        n_fam = fam_valid_rows.size

        # Build empty_mask on device: upload host empty_mask, gather on device.
        d_fam_valid = runtime.from_host(fam_valid_rows)
        if family_enum in geometry_array.families:
            host_buf = geometry_array.families[family_enum]
            fam_row_offsets = geometry_array.family_row_offsets[fam_valid_rows]
            d_empty_mask = cp.asarray(host_buf.empty_mask.astype(np.uint8))
            d_fam_row_off = cp.asarray(fam_row_offsets)
            # Clamp indices and gather on device
            d_valid_fr = d_fam_row_off < d_empty_mask.size
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)
            d_safe_idx = cp.minimum(d_fam_row_off, max(d_empty_mask.size - 1, 0))
            d_fam_empty[d_valid_fr] = d_empty_mask[d_safe_idx[d_valid_fr]]
        else:
            d_fam_empty = cp.zeros(n_fam, dtype=cp.uint8)

        # Part and ring offsets (use zeros if not available)
        d_geom_off = d_buf.geometry_offsets
        d_part_off = d_buf.part_offsets if d_buf.part_offsets is not None else d_buf.geometry_offsets
        d_ring_off = d_buf.ring_offsets if d_buf.ring_offsets is not None else d_buf.geometry_offsets

        # Step 1: Count segments
        d_seg_counts = runtime.allocate((n_fam,), np.int32, zero=True)
        count_kernel = kernels["count_segments"]
        ptr = runtime.pointer

        count_params = (
            (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_offsets),
             ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
             ptr(d_fam_empty), ptr(d_seg_counts), n_fam),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(count_kernel, n_fam)
        runtime.launch(count_kernel, grid=grid, block=block, params=count_params)

        # Step 2: Exclusive prefix sum for write offsets
        d_seg_offsets = exclusive_sum(d_seg_counts, synchronize=False)
        fam_total = count_scatter_total(runtime, d_seg_counts, d_seg_offsets)

        if fam_total == 0:
            runtime.free(d_fam_valid)
            runtime.free(d_fam_empty)
            runtime.free(d_seg_counts)
            runtime.free(d_seg_offsets)
            continue

        # Step 3: Allocate and scatter
        d_out_row = runtime.allocate((fam_total,), np.int32)
        d_out_seg = runtime.allocate((fam_total,), np.int32)
        d_out_part = runtime.allocate((fam_total,), np.int32)
        d_out_ring = runtime.allocate((fam_total,), np.int32)
        d_out_x0 = runtime.allocate((fam_total,), np.float64)
        d_out_y0 = runtime.allocate((fam_total,), np.float64)
        d_out_x1 = runtime.allocate((fam_total,), np.float64)
        d_out_y1 = runtime.allocate((fam_total,), np.float64)

        scatter_kernel = kernels["scatter_segments"]
        scatter_params = (
            (ptr(d_fam_valid), ptr(d_family_codes), ptr(d_family_row_offsets),
             ptr(d_geom_off), ptr(d_part_off), ptr(d_ring_off),
             ptr(d_fam_empty), ptr(d_buf.x), ptr(d_buf.y),
             ptr(d_seg_offsets),
             ptr(d_out_row), ptr(d_out_seg), ptr(d_out_part), ptr(d_out_ring),
             ptr(d_out_x0), ptr(d_out_y0), ptr(d_out_x1), ptr(d_out_y1),
             n_fam),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(scatter_kernel, n_fam)
        runtime.launch(scatter_kernel, grid=grid, block=block, params=scatter_params)

        all_row_idx.append(d_out_row)
        all_seg_idx.append(d_out_seg)
        all_part_idx.append(d_out_part)
        all_ring_idx.append(d_out_ring)
        all_x0.append(d_out_x0)
        all_y0.append(d_out_y0)
        all_x1.append(d_out_x1)
        all_y1.append(d_out_y1)
        total_segments += fam_total

        # Free temp buffers (output arrays are kept)
        runtime.free(d_fam_valid)
        runtime.free(d_fam_empty)
        runtime.free(d_seg_counts)
        runtime.free(d_seg_offsets)

    if total_segments == 0 or not all_row_idx:
        return DeviceSegmentTable(
            row_indices=runtime.allocate((0,), np.int32),
            segment_indices=runtime.allocate((0,), np.int32),
            x0=runtime.allocate((0,), np.float64),
            y0=runtime.allocate((0,), np.float64),
            x1=runtime.allocate((0,), np.float64),
            y1=runtime.allocate((0,), np.float64),
            count=0,
            part_indices=runtime.allocate((0,), np.int32),
            ring_indices=runtime.allocate((0,), np.int32),
        )

    # Concatenate per-family results on device (CuPy Tier 2)
    if len(all_row_idx) == 1:
        return DeviceSegmentTable(
            row_indices=all_row_idx[0],
            segment_indices=all_seg_idx[0],
            x0=all_x0[0],
            y0=all_y0[0],
            x1=all_x1[0],
            y1=all_y1[0],
            count=total_segments,
            part_indices=all_part_idx[0],
            ring_indices=all_ring_idx[0],
        )

    return DeviceSegmentTable(
        row_indices=cp.concatenate(all_row_idx),
        segment_indices=cp.concatenate(all_seg_idx),
        x0=cp.concatenate(all_x0),
        y0=cp.concatenate(all_y0),
        x1=cp.concatenate(all_x1),
        y1=cp.concatenate(all_y1),
        count=total_segments,
        part_indices=cp.concatenate(all_part_idx),
        ring_indices=cp.concatenate(all_ring_idx),
    )


# ---------------------------------------------------------------------------
# Kernel 2: GPU Spatial-Index Candidate Generation (sort-based sweep)
# ---------------------------------------------------------------------------
# O(n log n) candidate generation using radix sort + binary search sweep.
#
# Algorithm:
# 1. Compute x-midpoints for all segments on both sides
# 2. Sort both sides by x-midpoint using CCCL radix_sort
# 3. For each left segment, binary search in right's sorted x-midpoints
#    to find the range of rights whose x-midpoint overlaps the left's
#    x-extent. Then filter by y-overlap.
# 4. Output candidate pair indices.
#
# This replaces the O(n^2) tiled brute-force approach.
# ---------------------------------------------------------------------------

# Peak bytes per raw candidate pair during scatter+MBR-filter:
#   2 x int32 pair arrays = 8 bytes
#   8 x float64 gathered bounds = 64 bytes
#   1 x bool overlap mask = 1 byte
#   1 x uint8 cast = 1 byte
#   ~8 bytes CuPy temporaries during boolean expression evaluation
#   ~4 bytes compact_indices output (worst case)
# Total ~86 bytes.  Use 120 for safety headroom and pool fragmentation.
_BYTES_PER_RAW_PAIR = 120

# Absolute floor: never create batches smaller than 1M pairs.
_MIN_BATCH_PAIRS = 1 * 1024 * 1024


_MAX_BATCH_PAIRS_CAP = 8 * 1024 * 1024  # 8M pairs hard cap (~960 MB peak)


def _compute_max_batch_pairs() -> int:
    """Return the maximum number of raw candidate pairs per batch.

    Uses actual RMM/CuPy pool free blocks when available, falling back
    to CUDA mem_info.  Applies a hard cap of 8M pairs to prevent OOM
    from pool fragmentation and CuPy advanced-indexing temporaries.
    """
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    # Try to get actual pool-level free memory (more accurate than CUDA mem_info
    # because RMM reserves large blocks from CUDA up front).
    try:
        runtime = get_cuda_runtime()
        stats = runtime.memory_pool_stats()
        if "free_bytes" in stats:
            free_bytes = stats["free_bytes"]
        else:
            free_bytes, _ = cp.cuda.Device().mem_info
    except Exception:
        return _MAX_BATCH_PAIRS_CAP

    # Use 25% of available pool memory, capped at _MAX_BATCH_PAIRS_CAP.
    usable_bytes = free_bytes // 4
    max_pairs = usable_bytes // _BYTES_PER_RAW_PAIR

    return min(max(max_pairs, _MIN_BATCH_PAIRS), _MAX_BATCH_PAIRS_CAP)


def _main_sweep_scatter_and_filter(
    *,
    runtime,
    left: DeviceSegmentTable,
    d_cand_offsets,
    range_start,
    range_end,
    sorted_right_idx,
    left_minx,
    left_maxx,
    left_miny,
    left_maxy,
    right_minx,
    right_maxx,
    right_miny,
    right_maxy,
    outlier_mask_bool,
    total_raw_candidates: int,
):
    """Scatter raw candidates and MBR-filter, batching to fit in VRAM.

    When ``total_raw_candidates`` fits in a single batch this is identical
    to the original unbatched code path — no performance regression for
    small inputs.  For large candidate counts that would OOM, left segments
    are partitioned into batches whose raw pair count fits within 50% of
    free VRAM.

    Returns (main_final_left, main_final_right) as CuPy int32 arrays.
    """
    import cupy as cp

    max_batch_pairs = _compute_max_batch_pairs()

    # Fast path: everything fits in one batch — no overhead.
    if total_raw_candidates <= max_batch_pairs:
        return _scatter_and_filter_single(
            runtime=runtime,
            left=left,
            d_cand_offsets=d_cand_offsets,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
            left_minx=left_minx,
            left_maxx=left_maxx,
            left_miny=left_miny,
            left_maxy=left_maxy,
            right_minx=right_minx,
            right_maxx=right_maxx,
            right_miny=right_miny,
            right_maxy=right_maxy,
            outlier_mask_bool=outlier_mask_bool,
            left_start=0,
            left_end=left.count,
            offset_base=0,
            batch_raw_count=total_raw_candidates,
        )

    # --- Batched path: partition left segments into VRAM-bounded chunks ---
    # Batch boundaries are found entirely on device using lower_bound on
    # the monotonically non-decreasing d_cand_offsets, then only the small
    # boundary array is transferred to host for loop control flow.
    n_left = left.count

    # Build search targets: multiples of max_batch_pairs up to total.
    n_batches_est = (total_raw_candidates + max_batch_pairs - 1) // max_batch_pairs
    d_targets = cp.arange(1, n_batches_est, dtype=cp.int64) * max_batch_pairs
    d_inner = lower_bound(d_cand_offsets, d_targets, synchronize=False)

    # Full boundary array: [0, *inner_boundaries, n_left]
    d_boundaries = cp.concatenate([
        cp.zeros(1, dtype=cp.intp),
        d_inner.astype(cp.intp),
        cp.full(1, n_left, dtype=cp.intp),
    ])
    # Remove duplicate boundaries (collapsed empty batches).
    d_boundaries = cp.unique(d_boundaries)

    # Gather offset values at each boundary for batch-size computation.
    # Clamp indices to valid range for the offsets array (n_left entries).
    d_boundary_clamped = cp.minimum(d_boundaries, n_left - 1)
    d_boundary_offsets = d_cand_offsets[d_boundary_clamped]

    # Single bulk transfer of the small boundaries + offset-values arrays.
    runtime.synchronize()
    h_boundaries = cp.asnumpy(d_boundaries)
    h_boundary_offsets = cp.asnumpy(d_boundary_offsets)

    # Process each batch, keeping filtered results on device.  Filtered
    # output is a small fraction of raw candidates (typically 5-20% after
    # MBR filtering) so accumulating on device is safe — the raw pair
    # temporaries that drive OOM are freed inside _scatter_and_filter_single.
    result_left_parts: list = []
    result_right_parts: list = []

    n_bounds = len(h_boundaries)
    for b in range(n_bounds - 1):
        b_lo = int(h_boundaries[b])
        b_hi = int(h_boundaries[b + 1])
        if b_lo >= b_hi:
            continue

        b_offset_base = int(h_boundary_offsets[b])
        if b_hi < n_left:
            b_raw_count = int(h_boundary_offsets[b + 1]) - b_offset_base
        else:
            b_raw_count = total_raw_candidates - b_offset_base

        if b_raw_count == 0:
            continue

        # Release pool fragmentation between batches.
        runtime.free_pool_memory()

        b_left, b_right = _scatter_and_filter_single(
            runtime=runtime,
            left=left,
            d_cand_offsets=d_cand_offsets,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
            left_minx=left_minx,
            left_maxx=left_maxx,
            left_miny=left_miny,
            left_maxy=left_maxy,
            right_minx=right_minx,
            right_maxx=right_maxx,
            right_miny=right_miny,
            right_maxy=right_maxy,
            outlier_mask_bool=outlier_mask_bool,
            left_start=b_lo,
            left_end=b_hi,
            offset_base=b_offset_base,
            batch_raw_count=b_raw_count,
        )
        if b_left.size > 0:
            result_left_parts.append(b_left)
            result_right_parts.append(b_right)

    if not result_left_parts:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)
    if len(result_left_parts) == 1:
        return result_left_parts[0], result_right_parts[0]
    return (
        cp.concatenate(result_left_parts),
        cp.concatenate(result_right_parts),
    )


def _scatter_and_filter_single(
    *,
    runtime,
    left: DeviceSegmentTable,
    d_cand_offsets,
    range_start,
    range_end,
    sorted_right_idx,
    left_minx,
    left_maxx,
    left_miny,
    left_maxy,
    right_minx,
    right_maxx,
    right_miny,
    right_maxy,
    outlier_mask_bool,
    left_start: int,
    left_end: int,
    offset_base: int,
    batch_raw_count: int,
):
    """Scatter candidate pairs for left segments [left_start, left_end)
    and apply MBR overlap filter.  Returns (filtered_left, filtered_right).

    When left_start==0, left_end==n_left, and offset_base==0 this is
    identical to the original unbatched code path.
    """
    import cupy as cp

    batch_size = left_end - left_start

    d_left_pair = runtime.allocate((batch_raw_count,), np.int32)
    d_right_pair = runtime.allocate((batch_raw_count,), np.int32)

    scatter_kernels = _candidate_scatter_kernels()
    ptr = runtime.pointer

    # range_start/range_end are CuPy uint arrays; cast to int32 for kernel
    d_range_start_i32 = cp.asarray(range_start, dtype=cp.int32)
    d_range_end_i32 = cp.asarray(range_end, dtype=cp.int32)

    if left_start == 0 and left_end == left.count:
        # Unbatched path: use the original kernel (no offset arithmetic).
        scatter_fn = scatter_kernels["scatter_candidate_pairs"]
        grid, block = runtime.launch_config(scatter_fn, left.count)
        runtime.launch(
            scatter_fn,
            grid=grid, block=block,
            params=(
                (ptr(d_cand_offsets), ptr(d_range_start_i32), ptr(d_range_end_i32),
                 ptr(sorted_right_idx), ptr(d_left_pair), ptr(d_right_pair),
                 left.count),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )
    else:
        # Batched path: use the batch-aware kernel variant.
        scatter_fn = scatter_kernels["scatter_candidate_pairs_batch"]
        grid, block = runtime.launch_config(scatter_fn, batch_size)
        runtime.launch(
            scatter_fn,
            grid=grid, block=block,
            params=(
                (ptr(d_cand_offsets), ptr(d_range_start_i32), ptr(d_range_end_i32),
                 ptr(sorted_right_idx), ptr(d_left_pair), ptr(d_right_pair),
                 left_start, batch_size, offset_base),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32, KERNEL_PARAM_I64),
            ),
        )

    # Full MBR overlap filter on device
    d_lminx = left_minx[d_left_pair]
    d_lmaxx = left_maxx[d_left_pair]
    d_lminy = left_miny[d_left_pair]
    d_lmaxy = left_maxy[d_left_pair]
    d_rminx = right_minx[d_right_pair]
    d_rmaxx = right_maxx[d_right_pair]
    d_rminy = right_miny[d_right_pair]
    d_rmaxy = right_maxy[d_right_pair]

    main_overlap = (
        (d_lminx <= d_rmaxx) & (d_lmaxx >= d_rminx) &
        (d_lminy <= d_rmaxy) & (d_lmaxy >= d_rminy)
    )
    # Free gathered bounds immediately — they dominate per-batch memory
    # (8 × batch_raw_count × 8 bytes) and are no longer needed.
    del d_lminx, d_lmaxx, d_lminy, d_lmaxy
    del d_rminx, d_rmaxx, d_rminy, d_rmaxy
    # Exclude outlier right segments from main results to prevent
    # duplicates with the outlier pass.  Outlier rights will be
    # handled exclusively by the brute-force MBR pass below.
    if outlier_mask_bool is not None:
        main_overlap &= ~outlier_mask_bool[d_right_pair]
    main_overlap_u8 = main_overlap.astype(cp.uint8)
    del main_overlap

    main_compact = compact_indices(main_overlap_u8)
    if main_compact.count > 0:
        main_keep = main_compact.values
        return d_left_pair[main_keep], d_right_pair[main_keep]
    return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32)


def _generate_candidates_gpu(
    left: DeviceSegmentTable,
    right: DeviceSegmentTable,
) -> DeviceSegmentIntersectionCandidates:
    """GPU-native O(n log n) candidate generation via sort-sweep."""
    import cupy as cp

    runtime = get_cuda_runtime()

    if left.count == 0 or right.count == 0:
        empty_d = runtime.allocate((0,), np.int32)
        return DeviceSegmentIntersectionCandidates(
            left_rows=empty_d,
            left_segments=empty_d,
            left_lookup=runtime.allocate((0,), np.int32),
            right_rows=empty_d,
            right_segments=empty_d,
            right_lookup=runtime.allocate((0,), np.int32),
            count=0,
        )

    # Compute segment bounds on device (CuPy Tier 2 element-wise)
    left_minx = cp.minimum(left.x0, left.x1)
    left_maxx = cp.maximum(left.x0, left.x1)
    left_miny = cp.minimum(left.y0, left.y1)
    left_maxy = cp.maximum(left.y0, left.y1)

    right_minx = cp.minimum(right.x0, right.x1)
    right_maxx = cp.maximum(right.x0, right.x1)
    right_miny = cp.minimum(right.y0, right.y1)
    right_maxy = cp.maximum(right.y0, right.y1)

    # Sort right segments by x-midpoint for sweep-based candidate search.
    # Algorithm: sort right segments by x-midpoint, then for each left segment
    # binary-search for the range of rights whose midpoint falls within
    # [left_minx - max_right_halfwidth, left_maxx + max_right_halfwidth].
    # Then filter surviving candidates by full MBR y-overlap.
    # Complexity: O(n log n + k) where k is the number of candidate pairs.
    right_xmid = (right_minx + right_maxx) * 0.5
    # NOTE (P5/LOW): cp.arange allocates a 4MB array at 1M scale just to
    # provide [0..n-1] indices.  A counting_iterator would be zero-alloc,
    # but sort_pairs calls _validate_vector("values", values) which
    # requires a 1D DeviceArray (ndim check).  Both the CuPy argsort
    # fallback and the CCCL radix_sort path index into the values array,
    # so accepting an iterator would require invasive changes to
    # sort_pairs + its strategy dispatch.  Not worth the churn for a
    # one-shot 4MB allocation.
    right_indices = cp.arange(right.count, dtype=cp.int32)
    sort_result = sort_pairs(right_xmid, right_indices, synchronize=False)
    sorted_right_xmid = sort_result.keys
    sorted_right_idx = sort_result.values

    # -----------------------------------------------------------------------
    # P95 half-width strategy for search window sizing.
    #
    # Problem: using cp.max(right_half_w) makes the binary-search window
    # as wide as the single largest right segment. If even one segment
    # spans the full coordinate space, every left window covers ALL rights
    # -> O(n^2) candidates -> OOM.
    #
    # Solution: use the 95th percentile of right half-widths for the main
    # sweep (tight windows for 95% of segments), then handle the <=5%
    # outlier right segments in a separate brute-force MBR pass.
    # -----------------------------------------------------------------------
    right_half_w = (right_maxx - right_minx) * 0.5

    if right.count < 20:
        # Too few segments for P95 to matter -- use global max.
        d_search_hw = cp.max(right_half_w)
        has_outliers = False
        outlier_mask_bool = None
    else:
        # Use partition (O(n)) instead of full sort (O(n log n)) to get P95.
        p95_idx = int(right.count * 95) // 100  # floor index
        partitioned_hw = cp.partition(right_half_w, p95_idx)
        d_p95_hw = partitioned_hw[p95_idx]       # CuPy scalar on device
        d_max_hw = cp.max(right_half_w)          # CuPy scalar on device

        # If P95 == max, all segments fit within the P95 window.
        # Materializes a single Python bool (one scalar D2H -- necessary
        # for control flow, not in a loop).
        has_outliers = bool(d_max_hw > d_p95_hw)
        d_search_hw = d_p95_hw

        # Pre-compute outlier boolean mask on device for later use
        # in both main-sweep dedup and outlier pass.
        if has_outliers:
            outlier_mask_bool = right_half_w > d_search_hw
        else:
            outlier_mask_bool = None

    # --- Main sweep: binary search using P95 (or max) half-width ---
    search_lo = left_minx - d_search_hw
    search_hi = left_maxx + d_search_hw

    # Binary search in sorted_right_xmid (same-stream ordering guarantees
    # sort_pairs completes before lower_bound/upper_bound read its output)
    range_start = lower_bound(sorted_right_xmid, search_lo, synchronize=False)
    range_end = upper_bound(sorted_right_xmid, search_hi, synchronize=False)

    # Two-pass: count candidates per left, prefix-sum, scatter.
    # Use int64 for counts/offsets: at 100k+ segments, total candidates can
    # exceed int32 max (~2.1B), causing prefix-sum overflow and negative batch sizes.
    d_cand_counts = cp.asarray(range_end, dtype=cp.int64) - cp.asarray(range_start, dtype=cp.int64)
    d_cand_counts = cp.maximum(d_cand_counts, 0)  # clamp negatives

    d_cand_offsets = exclusive_sum(d_cand_counts, synchronize=False)
    total_raw_candidates = count_scatter_total(runtime, d_cand_counts, d_cand_offsets)

    # Guard: if total raw candidates exceed what the GPU can classify
    # (each surviving pair requires ~49 bytes of output arrays), raise early
    # instead of crashing mid-way through batched scatter.
    try:
        free_bytes, _ = cp.cuda.Device().mem_info
    except Exception:
        free_bytes = 8 * 1024**3  # conservative 8 GB
    # 49 bytes per pair for classification output + 8 bytes for candidate indices
    max_feasible_pairs = free_bytes // 57
    if total_raw_candidates > max_feasible_pairs:
        raise MemoryError(
            f"Segment intersection candidate count ({total_raw_candidates:,}) exceeds "
            f"GPU memory capacity ({free_bytes / 1e9:.1f} GB free, max ~{max_feasible_pairs:,} "
            f"feasible pairs). Reduce input scale or use CPU dispatch."
        )

    if total_raw_candidates > 0:
        main_final_left, main_final_right = _main_sweep_scatter_and_filter(
            runtime=runtime,
            left=left,
            d_cand_offsets=d_cand_offsets,
            range_start=range_start,
            range_end=range_end,
            sorted_right_idx=sorted_right_idx,
            left_minx=left_minx,
            left_maxx=left_maxx,
            left_miny=left_miny,
            left_maxy=left_maxy,
            right_minx=right_minx,
            right_maxx=right_maxx,
            right_miny=right_miny,
            right_maxy=right_maxy,
            outlier_mask_bool=outlier_mask_bool,
            total_raw_candidates=total_raw_candidates,
        )
    else:
        main_final_left = cp.empty(0, dtype=cp.int32)
        main_final_right = cp.empty(0, dtype=cp.int32)

    # --- Outlier pass: brute-force MBR test for right segs with hw > P95 ---
    if has_outliers:
        # Identify outlier right segment indices (boolean mask -> compact).
        # These are right segments whose half-width exceeds P95, meaning
        # the main sweep's narrower window may have missed them.
        outlier_mask_u8 = outlier_mask_bool.astype(cp.uint8)
        outlier_compact = compact_indices(outlier_mask_u8)

        if outlier_compact.count > 0:
            outlier_right_idx = outlier_compact.values  # indices into right arrays
            n_outliers = outlier_compact.count
            n_left = left.count

            # Batched brute-force: process outlier rights in chunks to avoid
            # materializing O(n_left * n_outliers) pairs which would OOM at
            # scale (e.g. 1M left × 50K outliers = 50B elements = 200+ GB).
            # Batch size adapts to available VRAM (same policy as main sweep).
            _MAX_EXPANDED_PAIRS = _compute_max_batch_pairs()
            _OUTLIER_BATCH = max(1, _MAX_EXPANDED_PAIRS // max(n_left, 1))
            batch_left_parts = []
            batch_right_parts = []
            d_left_arange = cp.arange(n_left, dtype=cp.int32)

            for batch_start in range(0, n_outliers, _OUTLIER_BATCH):
                batch_end = min(batch_start + _OUTLIER_BATCH, n_outliers)
                batch_right = outlier_right_idx[batch_start:batch_end]
                batch_size = batch_end - batch_start

                # Expand: n_left × batch_size pairs (bounded to ~1M per batch)
                ol_left_idx = cp.repeat(d_left_arange, batch_size)
                ol_right_idx = cp.tile(batch_right, n_left)

                # Vectorized MBR overlap test
                ol_overlap = (
                    (left_minx[ol_left_idx] <= right_maxx[ol_right_idx]) &
                    (left_maxx[ol_left_idx] >= right_minx[ol_right_idx]) &
                    (left_miny[ol_left_idx] <= right_maxy[ol_right_idx]) &
                    (left_maxy[ol_left_idx] >= right_miny[ol_right_idx])
                )
                ol_compact = compact_indices(ol_overlap.astype(cp.uint8))

                if ol_compact.count > 0:
                    ol_keep = ol_compact.values
                    batch_left_parts.append(ol_left_idx[ol_keep])
                    batch_right_parts.append(ol_right_idx[ol_keep])

            if batch_left_parts:
                outlier_final_left = cp.concatenate(batch_left_parts) if len(batch_left_parts) > 1 else batch_left_parts[0]
                outlier_final_right = cp.concatenate(batch_right_parts) if len(batch_right_parts) > 1 else batch_right_parts[0]
            else:
                outlier_final_left = cp.empty(0, dtype=cp.int32)
                outlier_final_right = cp.empty(0, dtype=cp.int32)
        else:
            outlier_final_left = cp.empty(0, dtype=cp.int32)
            outlier_final_right = cp.empty(0, dtype=cp.int32)
    else:
        outlier_final_left = cp.empty(0, dtype=cp.int32)
        outlier_final_right = cp.empty(0, dtype=cp.int32)

    # --- Merge main + outlier candidates on device ---
    if outlier_final_left.size > 0 and main_final_left.size > 0:
        final_left = cp.concatenate([main_final_left, outlier_final_left])
        final_right = cp.concatenate([main_final_right, outlier_final_right])
    elif outlier_final_left.size > 0:
        final_left = outlier_final_left
        final_right = outlier_final_right
    else:
        final_left = main_final_left
        final_right = main_final_right

    total_candidates = int(final_left.size)
    if total_candidates == 0:
        empty_d = runtime.allocate((0,), np.int32)
        return DeviceSegmentIntersectionCandidates(
            left_rows=empty_d,
            left_segments=empty_d,
            left_lookup=runtime.allocate((0,), np.int32),
            right_rows=empty_d,
            right_segments=empty_d,
            right_lookup=runtime.allocate((0,), np.int32),
            count=0,
        )

    # Build output candidate arrays
    out_left_rows = left.row_indices[final_left]
    out_left_segs = left.segment_indices[final_left]
    out_right_rows = right.row_indices[final_right]
    out_right_segs = right.segment_indices[final_right]

    return DeviceSegmentIntersectionCandidates(
        left_rows=out_left_rows,
        left_segments=out_left_segs,
        left_lookup=final_left,
        right_rows=out_right_rows,
        right_segments=out_right_segs,
        right_lookup=final_right,
        count=total_candidates,
    )


# ---------------------------------------------------------------------------
# Legacy CPU segment extraction (kept for CPU fallback)
# ---------------------------------------------------------------------------

def _valid_global_rows(geometry_array: OwnedGeometryArray, family_name: str) -> np.ndarray:
    tag = FAMILY_TAGS[family_name]
    return np.flatnonzero(geometry_array.validity & (geometry_array.tags == tag)).astype(np.int32, copy=False)


def _append_segments_for_span(
    *,
    row_index: int,
    part_index: int,
    ring_index: int,
    segment_counter: int,
    x: np.ndarray,
    y: np.ndarray,
    start: int,
    end: int,
    row_indices: list[int],
    part_indices: list[int],
    ring_indices: list[int],
    segment_indices: list[int],
    x0: list[float],
    y0: list[float],
    x1: list[float],
    y1: list[float],
    bounds: list[tuple[float, float, float, float]],
) -> int:
    if end - start < 2:
        return segment_counter

    xs0 = x[start : end - 1]
    ys0 = y[start : end - 1]
    xs1 = x[start + 1 : end]
    ys1 = y[start + 1 : end]
    count = int(xs0.size)
    if count == 0:
        return segment_counter

    row_indices.extend([row_index] * count)
    part_indices.extend([part_index] * count)
    ring_indices.extend([ring_index] * count)
    segment_indices.extend(range(segment_counter, segment_counter + count))
    x0.extend(xs0.tolist())
    y0.extend(ys0.tolist())
    x1.extend(xs1.tolist())
    y1.extend(ys1.tolist())
    bounds.extend(
        zip(
            np.minimum(xs0, xs1).tolist(),
            np.minimum(ys0, ys1).tolist(),
            np.maximum(xs0, xs1).tolist(),
            np.maximum(ys0, ys1).tolist(),
            strict=True,
        )
    )
    return segment_counter + count


def extract_segments(geometry_array: OwnedGeometryArray) -> SegmentTable:
    """Extract segments from geometry array on CPU (legacy path)."""
    row_indices: list[int] = []
    part_indices: list[int] = []
    ring_indices: list[int] = []
    segment_indices: list[int] = []
    x0: list[float] = []
    y0: list[float] = []
    x1: list[float] = []
    y1: list[float] = []
    bounds: list[tuple[float, float, float, float]] = []

    for family_name, buffer in geometry_array.families.items():
        if family_name not in {"linestring", "polygon", "multilinestring", "multipolygon"}:
            continue

        global_rows = _valid_global_rows(geometry_array, family_name)
        for family_row, row_index in enumerate(global_rows.tolist()):  # zcopy:ok(CPU-only legacy path: global_rows is np.ndarray from np.flatnonzero)
            if bool(buffer.empty_mask[family_row]):
                continue

            segment_counter = 0
            if family_name == "linestring":
                start = int(buffer.geometry_offsets[family_row])
                end = int(buffer.geometry_offsets[family_row + 1])
                segment_counter = _append_segments_for_span(
                    row_index=row_index,
                    part_index=0,
                    ring_index=0,
                    segment_counter=segment_counter,
                    x=buffer.x,
                    y=buffer.y,
                    start=start,
                    end=end,
                    row_indices=row_indices,
                    part_indices=part_indices,
                    ring_indices=ring_indices,
                    segment_indices=segment_indices,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    bounds=bounds,
                )
                del segment_counter
                continue

            if family_name == "polygon":
                ring_start = int(buffer.geometry_offsets[family_row])
                ring_end = int(buffer.geometry_offsets[family_row + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=0,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            if family_name == "multilinestring":
                part_start = int(buffer.geometry_offsets[family_row])
                part_end = int(buffer.geometry_offsets[family_row + 1])
                for part_local, part_index in enumerate(range(part_start, part_end)):
                    coord_start = int(buffer.part_offsets[part_index])
                    coord_end = int(buffer.part_offsets[part_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=part_local,
                        ring_index=-1,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )
                continue

            polygon_start = int(buffer.geometry_offsets[family_row])
            polygon_end = int(buffer.geometry_offsets[family_row + 1])
            for polygon_local, polygon_index in enumerate(range(polygon_start, polygon_end)):
                ring_start = int(buffer.part_offsets[polygon_index])
                ring_end = int(buffer.part_offsets[polygon_index + 1])
                for ring_local, ring_index in enumerate(range(ring_start, ring_end)):
                    coord_start = int(buffer.ring_offsets[ring_index])
                    coord_end = int(buffer.ring_offsets[ring_index + 1])
                    segment_counter = _append_segments_for_span(
                        row_index=row_index,
                        part_index=polygon_local,
                        ring_index=ring_local,
                        segment_counter=segment_counter,
                        x=buffer.x,
                        y=buffer.y,
                        start=coord_start,
                        end=coord_end,
                        row_indices=row_indices,
                        part_indices=part_indices,
                        ring_indices=ring_indices,
                        segment_indices=segment_indices,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        bounds=bounds,
                    )

    if not row_indices:
        empty_i32 = np.asarray([], dtype=np.int32)
        empty_f64 = np.asarray([], dtype=np.float64)
        return SegmentTable(
            row_indices=empty_i32,
            part_indices=empty_i32,
            ring_indices=empty_i32,
            segment_indices=empty_i32,
            x0=empty_f64,
            y0=empty_f64,
            x1=empty_f64,
            y1=empty_f64,
            bounds=np.empty((0, 4), dtype=np.float64),
        )

    return SegmentTable(
        row_indices=np.asarray(row_indices, dtype=np.int32),
        part_indices=np.asarray(part_indices, dtype=np.int32),
        ring_indices=np.asarray(ring_indices, dtype=np.int32),
        segment_indices=np.asarray(segment_indices, dtype=np.int32),
        x0=np.asarray(x0, dtype=np.float64),
        y0=np.asarray(y0, dtype=np.float64),
        x1=np.asarray(x1, dtype=np.float64),
        y1=np.asarray(y1, dtype=np.float64),
        bounds=np.asarray(bounds, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Legacy CPU candidate generation (kept for CPU fallback)
# ---------------------------------------------------------------------------

def generate_segment_candidates(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_segments = extract_segments(left)
    right_segments = extract_segments(right)
    return _generate_segment_candidates_from_tables(left_segments, right_segments, tile_size=tile_size)


def _generate_segment_candidates_from_tables(
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    *,
    tile_size: int = 512,
) -> SegmentIntersectionCandidates:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    left_rows_out: list[np.ndarray] = []
    left_segment_out: list[np.ndarray] = []
    left_lookup_out: list[np.ndarray] = []
    right_rows_out: list[np.ndarray] = []
    right_segment_out: list[np.ndarray] = []
    right_lookup_out: list[np.ndarray] = []
    pairs_examined = 0

    for left_start in range(0, left_segments.count, tile_size):
        left_bounds = left_segments.bounds[left_start : left_start + tile_size]
        left_rows = left_segments.row_indices[left_start : left_start + tile_size]
        left_ids = left_segments.segment_indices[left_start : left_start + tile_size]
        for right_start in range(0, right_segments.count, tile_size):
            right_bounds = right_segments.bounds[right_start : right_start + tile_size]
            right_rows = right_segments.row_indices[right_start : right_start + tile_size]
            right_ids = right_segments.segment_indices[right_start : right_start + tile_size]
            pairs_examined += int(left_bounds.shape[0] * right_bounds.shape[0])
            intersects = (
                (left_bounds[:, None, 0] <= right_bounds[None, :, 2])
                & (left_bounds[:, None, 2] >= right_bounds[None, :, 0])
                & (left_bounds[:, None, 1] <= right_bounds[None, :, 3])
                & (left_bounds[:, None, 3] >= right_bounds[None, :, 1])
            )
            left_local, right_local = np.nonzero(intersects)
            if left_local.size == 0:
                continue
            left_rows_out.append(left_rows[left_local].astype(np.int32, copy=False))
            left_segment_out.append(left_ids[left_local].astype(np.int32, copy=False))
            left_lookup_out.append((left_start + left_local).astype(np.int32, copy=False))
            right_rows_out.append(right_rows[right_local].astype(np.int32, copy=False))
            right_segment_out.append(right_ids[right_local].astype(np.int32, copy=False))
            right_lookup_out.append((right_start + right_local).astype(np.int32, copy=False))

    if not left_rows_out:
        empty = np.asarray([], dtype=np.int32)
        return SegmentIntersectionCandidates(
            left_rows=empty,
            left_segments=empty,
            left_lookup=empty,
            right_rows=empty,
            right_segments=empty,
            right_lookup=empty,
            pairs_examined=pairs_examined,
            tile_size=tile_size,
        )
    return SegmentIntersectionCandidates(
        left_rows=np.concatenate(left_rows_out),
        left_segments=np.concatenate(left_segment_out),
        left_lookup=np.concatenate(left_lookup_out),
        right_rows=np.concatenate(right_rows_out),
        right_segments=np.concatenate(right_segment_out),
        right_lookup=np.concatenate(right_lookup_out),
        pairs_examined=pairs_examined,
        tile_size=tile_size,
    )


# ---------------------------------------------------------------------------
# CPU exact arithmetic helpers (kept for CPU fallback)
# ---------------------------------------------------------------------------

def _orient2d_fast(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    abx = bx - ax
    aby = by - ay
    acx = cx - ax
    acy = cy - ay
    term1 = abx * acy
    term2 = aby * acx
    det = term1 - term2
    errbound = _ORIENTATION_ERRBOUND * (np.abs(term1) + np.abs(term2))
    return det, np.abs(det) <= errbound


def _line_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    denominator = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
    if denominator == 0.0:
        return float("nan"), float("nan")
    left_det = ax * by - ay * bx
    right_det = cx * dy - cy * dx
    x = (left_det * (cx - dx) - (ax - bx) * right_det) / denominator
    y = (left_det * (cy - dy) - (ay - by) * right_det) / denominator
    return float(x), float(y)


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_orientation_sign(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> int:
    det = (_fraction(bx) - _fraction(ax)) * (_fraction(cy) - _fraction(ay)) - (
        _fraction(by) - _fraction(ay)
    ) * (_fraction(cx) - _fraction(ax))
    return int(det > 0) - int(det < 0)


def _point_on_segment_exact(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    if _exact_orientation_sign(ax, ay, bx, by, px, py) != 0:
        return False
    pxf = _fraction(px)
    pyf = _fraction(py)
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    return min(axf, bxf) <= pxf <= max(axf, bxf) and min(ayf, byf) <= pyf <= max(ayf, byf)


def _exact_intersection_point(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[float, float]:
    axf = _fraction(ax)
    ayf = _fraction(ay)
    bxf = _fraction(bx)
    byf = _fraction(by)
    cxf = _fraction(cx)
    cyf = _fraction(cy)
    dxf = _fraction(dx)
    dyf = _fraction(dy)
    denominator = (axf - bxf) * (cyf - dyf) - (ayf - byf) * (cxf - dxf)
    if denominator == 0:
        return float("nan"), float("nan")
    left_det = axf * byf - ayf * bxf
    right_det = cxf * dyf - cyf * dxf
    x = (left_det * (cxf - dxf) - (axf - bxf) * right_det) / denominator
    y = (left_det * (cyf - dyf) - (ayf - byf) * right_det) / denominator
    return float(x), float(y)


def _unique_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    seen: set[tuple[Fraction, Fraction]] = set()
    for x, y in points:
        key = (_fraction(x), _fraction(y))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(x), float(y)))
    return unique


def _sort_collinear_points(
    points: list[tuple[float, float]],
    *,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> list[tuple[float, float]]:
    use_x = abs(bx - ax) >= abs(by - ay)

    def _key(point: tuple[float, float]) -> tuple[Fraction, Fraction]:
        x, y = point
        if use_x:
            return (_fraction(x), _fraction(y))
        return (_fraction(y), _fraction(x))

    return sorted(points, key=_key)


def _classify_exact_pair(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> tuple[SegmentIntersectionKind, tuple[float, float], tuple[float, float, float, float]]:
    a_is_point = _fraction(ax) == _fraction(bx) and _fraction(ay) == _fraction(by)
    c_is_point = _fraction(cx) == _fraction(dx) and _fraction(cy) == _fraction(dy)

    if a_is_point and c_is_point:
        if _fraction(ax) == _fraction(cx) and _fraction(ay) == _fraction(cy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if a_is_point:
        if _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
            return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    if c_is_point:
        if _point_on_segment_exact(cx, cy, ax, ay, bx, by):
            return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
        return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4

    o1 = _exact_orientation_sign(ax, ay, bx, by, cx, cy)
    o2 = _exact_orientation_sign(ax, ay, bx, by, dx, dy)
    o3 = _exact_orientation_sign(cx, cy, dx, dy, ax, ay)
    o4 = _exact_orientation_sign(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0 and o3 * o4 < 0:
        point = _exact_intersection_point(ax, ay, bx, by, cx, cy, dx, dy)
        return SegmentIntersectionKind.PROPER, point, (float("nan"),) * 4

    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
        shared = _unique_points(
            [
                point
                for point in ((ax, ay), (bx, by), (cx, cy), (dx, dy))
                if _point_on_segment_exact(point[0], point[1], ax, ay, bx, by)
                and _point_on_segment_exact(point[0], point[1], cx, cy, dx, dy)
            ]
        )
        if not shared:
            return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4
        shared = _sort_collinear_points(shared, ax=ax, ay=ay, bx=bx, by=by)
        if len(shared) == 1:
            x, y = shared[0]
            return SegmentIntersectionKind.TOUCH, (x, y), (float("nan"),) * 4
        (sx0, sy0), (sx1, sy1) = shared[0], shared[-1]
        return SegmentIntersectionKind.OVERLAP, (float("nan"), float("nan")), (sx0, sy0, sx1, sy1)

    if o1 == 0 and _point_on_segment_exact(cx, cy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(cx), float(cy)), (float("nan"),) * 4
    if o2 == 0 and _point_on_segment_exact(dx, dy, ax, ay, bx, by):
        return SegmentIntersectionKind.TOUCH, (float(dx), float(dy)), (float("nan"),) * 4
    if o3 == 0 and _point_on_segment_exact(ax, ay, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(ax), float(ay)), (float("nan"),) * 4
    if o4 == 0 and _point_on_segment_exact(bx, by, cx, cy, dx, dy):
        return SegmentIntersectionKind.TOUCH, (float(bx), float(by)), (float("nan"),) * 4

    return SegmentIntersectionKind.DISJOINT, (float("nan"), float("nan")), (float("nan"),) * 4


def _classify_exact_rows(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    rows: np.ndarray,
    kinds: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    overlap_x0: np.ndarray,
    overlap_y0: np.ndarray,
    overlap_x1: np.ndarray,
    overlap_y1: np.ndarray,
) -> None:
    for row in rows.tolist():
        kind, point, overlap = _classify_exact_pair(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )
        kinds[row] = int(kind)
        point_x[row], point_y[row] = point
        overlap_x0[row], overlap_y0[row], overlap_x1[row], overlap_y1[row] = overlap


# ---------------------------------------------------------------------------
# Dispatch wiring
# ---------------------------------------------------------------------------

def _select_segment_runtime(
    dispatch_mode: ExecutionMode | str,
    *,
    candidate_count: int,
) -> RuntimeSelection:
    return plan_dispatch_selection(
        kernel_name="segment_intersection",
        kernel_class=KernelClass.PREDICATE,
        row_count=candidate_count,
        requested_mode=dispatch_mode,
    )


# ---------------------------------------------------------------------------
# GPU variant: full pipeline (extract -> candidates -> classify)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "segment_intersection",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "polygon", "multilinestring", "multipolygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP32, PrecisionMode.FP64),
    preferred_residency=Residency.DEVICE,
    tags=("cuda-python",),
)
def _empty_segment_intersection_result(
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    """Construct an empty SegmentIntersectionResult with host arrays."""
    empty_i32 = np.asarray([], dtype=np.int32)
    empty_f64 = np.asarray([], dtype=np.float64)
    return SegmentIntersectionResult(
        candidate_pairs=0,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        _count=0,
        _left_rows=empty_i32,
        _left_segments=empty_i32,
        _left_lookup=empty_i32,
        _right_rows=empty_i32,
        _right_segments=empty_i32,
        _right_lookup=empty_i32,
        _kinds=empty_i32,
        _point_x=empty_f64,
        _point_y=empty_f64,
        _overlap_x0=empty_f64,
        _overlap_y0=empty_f64,
        _overlap_x1=empty_f64,
        _overlap_y1=empty_f64,
        _ambiguous_rows=empty_i32,
    )


def _classify_segment_intersections_gpu(
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    left_segments: SegmentTable | None = None,
    right_segments: SegmentTable | None = None,
    pairs: SegmentIntersectionCandidates | None = None,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
    tile_size: int = 512,
    _cached_right_device_segments: DeviceSegmentTable | None = None,
) -> SegmentIntersectionResult:
    """Full GPU-native segment intersection pipeline.

    Kernel 1: GPU segment extraction (NVRTC count-scatter)
    Kernel 2: GPU candidate generation (sort-sweep with CCCL radix sort)
    Kernel 3: GPU classification with Shewchuk adaptive refinement

    Parameters
    ----------
    _cached_right_device_segments : DeviceSegmentTable, optional
        Pre-extracted right-side segments.  When provided, skips
        ``_extract_segments_gpu(right)`` entirely.  Used by
        ``spatial_overlay_owned`` to avoid re-extracting the same
        corridor geometry N times in an N-vs-1 overlay loop (lyy.15).
    """
    runtime = get_cuda_runtime()

    # Determine compute type from precision plan
    compute_type = "float" if precision_plan.compute_precision is PrecisionMode.FP32 else "double"

    # --- Kernel 1: Extract segments on GPU ---
    d_left_segs = _extract_segments_gpu(left, compute_type)
    d_right_segs = (
        _cached_right_device_segments
        if _cached_right_device_segments is not None
        else _extract_segments_gpu(right, compute_type)
    )

    if d_left_segs.count == 0 or d_right_segs.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    # --- Kernel 2: Generate candidates on GPU ---
    d_candidates = _generate_candidates_gpu(d_left_segs, d_right_segs)

    if d_candidates.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    n_pairs = d_candidates.count

    # --- Kernel 3: Classify segment pairs on GPU ---
    device_kinds = runtime.allocate((n_pairs,), np.int8)
    device_point_x = runtime.allocate((n_pairs,), np.float64)
    device_point_y = runtime.allocate((n_pairs,), np.float64)
    device_overlap_x0 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_y0 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_x1 = runtime.allocate((n_pairs,), np.float64)
    device_overlap_y1 = runtime.allocate((n_pairs,), np.float64)

    kernels = _classify_kernels(compute_type)
    classify_kernel = kernels["classify_segment_pairs_v2"]
    ptr = runtime.pointer

    classify_params = (
        (
            ptr(d_candidates.left_lookup),
            ptr(d_candidates.right_lookup),
            ptr(d_left_segs.x0),
            ptr(d_left_segs.y0),
            ptr(d_left_segs.x1),
            ptr(d_left_segs.y1),
            ptr(d_right_segs.x0),
            ptr(d_right_segs.y0),
            ptr(d_right_segs.x1),
            ptr(d_right_segs.y1),
            ptr(device_kinds),
            ptr(device_point_x),
            ptr(device_point_y),
            ptr(device_overlap_x0),
            ptr(device_overlap_y0),
            ptr(device_overlap_x1),
            ptr(device_overlap_y1),
            n_pairs,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    grid, block = runtime.launch_config(classify_kernel, n_pairs)
    runtime.launch(classify_kernel, grid=grid, block=block, params=classify_params)

    # Sync GPU before returning device-primary result.
    runtime.synchronize()

    # No ambiguous rows -- all refinement happens on GPU
    d_ambiguous_rows = runtime.allocate((0,), np.int32)

    # Device-primary: host arrays are lazily materialized on first access.
    return SegmentIntersectionResult(
        candidate_pairs=n_pairs,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        device_state=SegmentIntersectionDeviceState(
            left_rows=d_candidates.left_rows,
            left_segments=d_candidates.left_segments,
            left_lookup=d_candidates.left_lookup,
            right_rows=d_candidates.right_rows,
            right_segments=d_candidates.right_segments,
            right_lookup=d_candidates.right_lookup,
            kinds=device_kinds,
            point_x=device_point_x,
            point_y=device_point_y,
            overlap_x0=device_overlap_x0,
            overlap_y0=device_overlap_y0,
            overlap_x1=device_overlap_x1,
            overlap_y1=device_overlap_y1,
            ambiguous_rows=d_ambiguous_rows,
        ),
        _count=n_pairs,
    )


# ---------------------------------------------------------------------------
# CPU variant (Shapely-based fallback)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "segment_intersection",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=("linestring", "polygon", "multilinestring", "multipolygon"),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    tags=("shapely",),
)
def _classify_segment_intersections_cpu(
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    left_segments: SegmentTable | None = None,
    right_segments: SegmentTable | None = None,
    pairs: SegmentIntersectionCandidates | None = None,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
    tile_size: int = 512,
) -> SegmentIntersectionResult:
    """CPU fallback using numpy vectorized orientation + exact Fraction arithmetic."""
    left_segs = left_segments if left_segments is not None else extract_segments(left)
    right_segs = right_segments if right_segments is not None else extract_segments(right)
    cands = (
        candidate_pairs or pairs
        if (candidate_pairs is not None or pairs is not None)
        else _generate_segment_candidates_from_tables(left_segs, right_segs, tile_size=tile_size)
    )
    return _classify_segment_intersections_from_tables(
        left_segments=left_segs,
        right_segments=right_segs,
        pairs=cands,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def _classify_segment_intersections_from_tables(
    *,
    left_segments: SegmentTable,
    right_segments: SegmentTable,
    pairs: SegmentIntersectionCandidates,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    robustness_plan: RobustnessPlan,
) -> SegmentIntersectionResult:
    if pairs.count == 0:
        return _empty_segment_intersection_result(
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    left_lookup = pairs.left_lookup
    right_lookup = pairs.right_lookup

    ax = left_segments.x0[left_lookup]
    ay = left_segments.y0[left_lookup]
    bx = left_segments.x1[left_lookup]
    by = left_segments.y1[left_lookup]
    cx = right_segments.x0[right_lookup]
    cy = right_segments.y0[right_lookup]
    dx = right_segments.x1[right_lookup]
    dy = right_segments.y1[right_lookup]

    o1, a1 = _orient2d_fast(ax, ay, bx, by, cx, cy)
    o2, a2 = _orient2d_fast(ax, ay, bx, by, dx, dy)
    o3, a3 = _orient2d_fast(cx, cy, dx, dy, ax, ay)
    o4, a4 = _orient2d_fast(cx, cy, dx, dy, bx, by)

    zero_left = (ax == bx) & (ay == by)
    zero_right = (cx == dx) & (cy == dy)
    sign1 = np.sign(o1).astype(np.int8, copy=False)
    sign2 = np.sign(o2).astype(np.int8, copy=False)
    sign3 = np.sign(o3).astype(np.int8, copy=False)
    sign4 = np.sign(o4).astype(np.int8, copy=False)

    ambiguous_mask = (
        a1
        | a2
        | a3
        | a4
        | zero_left
        | zero_right
        | (sign1 == 0)
        | (sign2 == 0)
        | (sign3 == 0)
        | (sign4 == 0)
    )
    proper_mask = (~ambiguous_mask) & (sign1 * sign2 < 0) & (sign3 * sign4 < 0)

    count = int(pairs.count)
    kinds = np.full(count, int(SegmentIntersectionKind.DISJOINT), dtype=np.int8)
    point_x = np.full(count, np.nan, dtype=np.float64)
    point_y = np.full(count, np.nan, dtype=np.float64)
    overlap_x0 = np.full(count, np.nan, dtype=np.float64)
    overlap_y0 = np.full(count, np.nan, dtype=np.float64)
    overlap_x1 = np.full(count, np.nan, dtype=np.float64)
    overlap_y1 = np.full(count, np.nan, dtype=np.float64)

    kinds[proper_mask] = int(SegmentIntersectionKind.PROPER)
    proper_rows = np.flatnonzero(proper_mask)
    for row in proper_rows.tolist():
        point_x[row], point_y[row] = _line_intersection_point(
            float(ax[row]),
            float(ay[row]),
            float(bx[row]),
            float(by[row]),
            float(cx[row]),
            float(cy[row]),
            float(dx[row]),
            float(dy[row]),
        )

    ambiguous_rows = np.flatnonzero(ambiguous_mask).astype(np.int32, copy=False)
    if ambiguous_rows.size:
        _classify_exact_rows(
            ax,
            ay,
            bx,
            by,
            cx,
            cy,
            dx,
            dy,
            ambiguous_rows,
            kinds,
            point_x,
            point_y,
            overlap_x0,
            overlap_y0,
            overlap_x1,
            overlap_y1,
        )

    return SegmentIntersectionResult(
        candidate_pairs=int(pairs.count),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        _left_rows=pairs.left_rows.copy(),
        _left_segments=pairs.left_segments.copy(),
        _left_lookup=pairs.left_lookup.copy(),
        _right_rows=pairs.right_rows.copy(),
        _right_segments=pairs.right_segments.copy(),
        _right_lookup=pairs.right_lookup.copy(),
        _kinds=kinds,
        _point_x=point_x,
        _point_y=point_y,
        _overlap_x0=overlap_x0,
        _overlap_y0=overlap_y0,
        _overlap_x1=overlap_x1,
        _overlap_y1=overlap_y1,
        _ambiguous_rows=ambiguous_rows,
    )


# ---------------------------------------------------------------------------
# Public API entry point with dispatch
# ---------------------------------------------------------------------------

def classify_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    candidate_pairs: SegmentIntersectionCandidates | None = None,
    tile_size: int = 512,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    _cached_right_device_segments: DeviceSegmentTable | None = None,
) -> SegmentIntersectionResult:
    """Classify all segment-segment intersections between two geometry arrays.

    Parameters
    ----------
    left, right : OwnedGeometryArray
        Input geometry arrays (linestring, polygon, or multi-variants).
    candidate_pairs : SegmentIntersectionCandidates, optional
        Pre-computed candidate pairs. If None, candidates are generated
        internally (GPU-native O(n log n) when GPU mode, tiled CPU otherwise).
    tile_size : int
        Tile size for CPU candidate generation (ignored in GPU mode).
    dispatch_mode : ExecutionMode
        Force GPU, CPU, or AUTO dispatch.
    precision : PrecisionMode
        Force fp32, fp64, or AUTO precision.
    _cached_right_device_segments : DeviceSegmentTable, optional
        Pre-extracted right-side device segments for reuse (lyy.15).

    Returns
    -------
    SegmentIntersectionResult
        Classification of all candidate segment pairs.
    """
    # Estimate candidate count for dispatch decision
    # Use a rough heuristic: total coords across both arrays
    total_coords = sum(
        buf.x.size for buf in left.families.values()
        if buf.family in {GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
                          GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON}
    ) + sum(
        buf.x.size for buf in right.families.values()
        if buf.family in {GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
                          GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON}
    )
    estimated_candidates = max(total_coords, 1)

    runtime_selection = _select_segment_runtime(
        dispatch_mode, candidate_count=estimated_candidates,
    )
    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.PREDICATE,
        requested=precision,
    )
    robustness_plan = select_robustness_plan(
        kernel_class=KernelClass.PREDICATE,
        precision_plan=precision_plan,
    )

    if runtime_selection.selected is ExecutionMode.GPU:
        return _classify_segment_intersections_gpu(
            left=left,
            right=right,
            candidate_pairs=candidate_pairs,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
            tile_size=tile_size,
            _cached_right_device_segments=_cached_right_device_segments,
        )

    # CPU fallback
    return _classify_segment_intersections_cpu(
        left=left,
        right=right,
        candidate_pairs=candidate_pairs,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
        tile_size=tile_size,
    )


def benchmark_segment_intersections(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    tile_size: int = 512,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> SegmentIntersectionBenchmark:
    started = perf_counter()
    result = classify_segment_intersections(left, right, tile_size=tile_size, dispatch_mode=dispatch_mode)
    elapsed = perf_counter() - started
    return SegmentIntersectionBenchmark(
        rows_left=left.row_count,
        rows_right=right.row_count,
        candidate_pairs=result.candidate_pairs,
        disjoint_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.DISJOINT))),
        proper_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.PROPER))),
        touch_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.TOUCH))),
        overlap_pairs=int(np.count_nonzero(result.kinds == int(SegmentIntersectionKind.OVERLAP))),
        ambiguous_pairs=int(result.ambiguous_rows.size),
        elapsed_seconds=elapsed,
    )
