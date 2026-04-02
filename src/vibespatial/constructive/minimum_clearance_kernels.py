"""NVRTC kernel sources for minimum_clearance."""

from __future__ import annotations

from vibespatial.cuda.device_functions.strip_closure import STRIP_CLOSURE_DEVICE

# ---------------------------------------------------------------------------
# NVRTC kernel: segment-to-segment distance device helper
# ---------------------------------------------------------------------------

_SEGMENT_DISTANCE_HELPER = r"""
/* Compute the minimum distance between two line segments (p1,p2) and (q1,q2).
   Uses parametric closest-approach:
   - Project each endpoint onto the other segment, clamp parameter to [0,1]
   - Also check segment-segment closest approach (perpendicular case)
   - Return the minimum of all candidate distances.

   This correctly handles:
   - Parallel segments
   - Degenerate (zero-length) segments
   - Perpendicular closest approach when projections overlap
*/
__device__ double seg_seg_distance(
    double p1x, double p1y, double p2x, double p2y,
    double q1x, double q1y, double q2x, double q2y
) {
    /* Direction vectors */
    const double dx_p = p2x - p1x;
    const double dy_p = p2y - p1y;
    const double dx_q = q2x - q1x;
    const double dy_q = q2y - q1y;

    const double len_p_sq = dx_p * dx_p + dy_p * dy_p;
    const double len_q_sq = dx_q * dx_q + dy_q * dy_q;

    double min_dist_sq = 1e300;  /* effectively infinity */

    /* --- Point-to-segment projections (4 cases) --- */

    /* p1 onto segment q */
    if (len_q_sq > 0.0) {
        double t = ((p1x - q1x) * dx_q + (p1y - q1y) * dy_q) / len_q_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double cx = q1x + t * dx_q - p1x;
        double cy = q1y + t * dy_q - p1y;
        double d2 = cx * cx + cy * cy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
    }

    /* p2 onto segment q */
    if (len_q_sq > 0.0) {
        double t = ((p2x - q1x) * dx_q + (p2y - q1y) * dy_q) / len_q_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double cx = q1x + t * dx_q - p2x;
        double cy = q1y + t * dy_q - p2y;
        double d2 = cx * cx + cy * cy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
    }

    /* q1 onto segment p */
    if (len_p_sq > 0.0) {
        double t = ((q1x - p1x) * dx_p + (q1y - p1y) * dy_p) / len_p_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double cx = p1x + t * dx_p - q1x;
        double cy = p1y + t * dy_p - q1y;
        double d2 = cx * cx + cy * cy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
    }

    /* q2 onto segment p */
    if (len_p_sq > 0.0) {
        double t = ((q2x - p1x) * dx_p + (q2y - p1y) * dy_p) / len_p_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double cx = p1x + t * dx_p - q2x;
        double cy = p1y + t * dy_p - q2y;
        double d2 = cx * cx + cy * cy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
    }

    /* --- Endpoint-to-endpoint distances (4 cases, for degenerate segs) --- */
    {
        double dx, dy, d2;
        dx = p1x - q1x; dy = p1y - q1y; d2 = dx*dx + dy*dy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
        dx = p1x - q2x; dy = p1y - q2y; d2 = dx*dx + dy*dy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
        dx = p2x - q1x; dy = p2y - q1y; d2 = dx*dx + dy*dy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
        dx = p2x - q2x; dy = p2y - q2y; d2 = dx*dx + dy*dy;
        if (d2 < min_dist_sq) min_dist_sq = d2;
    }

    return sqrt(min_dist_sq);
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel: segment-to-segment closest points device helper
# ---------------------------------------------------------------------------
# Same algorithm as seg_seg_distance but tracks the two closest points
# (one on each segment) alongside the squared distance.  Returns the squared
# distance; the caller's min_dist_sq / best_ax / best_ay / best_bx / best_by
# variables are updated when a new minimum is found.
# ---------------------------------------------------------------------------

_SEGMENT_CLOSEST_POINTS_HELPER = r"""
/* Update running minimum with segment-pair closest point tracking.
   Computes the closest point pair between segments (p1,p2) and (q1,q2).
   When a new minimum is found, updates *best_dist_sq, *best_ax, *best_ay,
   *best_bx, *best_by to the closest point on segment P (a) and on segment Q (b). */
__device__ void seg_seg_closest_points(
    double p1x, double p1y, double p2x, double p2y,
    double q1x, double q1y, double q2x, double q2y,
    double* best_dist_sq,
    double* best_ax, double* best_ay,
    double* best_bx, double* best_by
) {
    const double dx_p = p2x - p1x;
    const double dy_p = p2y - p1y;
    const double dx_q = q2x - q1x;
    const double dy_q = q2y - q1y;

    const double len_p_sq = dx_p * dx_p + dy_p * dy_p;
    const double len_q_sq = dx_q * dx_q + dy_q * dy_q;

    /* Helper macro: update best if d2 < current best */
    #define UPDATE_BEST(d2_val, ax_val, ay_val, bx_val, by_val) \
        if ((d2_val) < *best_dist_sq) {                         \
            *best_dist_sq = (d2_val);                           \
            *best_ax = (ax_val);                                \
            *best_ay = (ay_val);                                \
            *best_bx = (bx_val);                                \
            *best_by = (by_val);                                \
        }

    /* --- Point-to-segment projections (4 cases) --- */

    /* p1 onto segment q: closest on Q is (q1 + t*dq), point on P is p1 */
    if (len_q_sq > 0.0) {
        double t = ((p1x - q1x) * dx_q + (p1y - q1y) * dy_q) / len_q_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double qx = q1x + t * dx_q;
        double qy = q1y + t * dy_q;
        double dx = qx - p1x;
        double dy = qy - p1y;
        double d2 = dx * dx + dy * dy;
        UPDATE_BEST(d2, p1x, p1y, qx, qy);
    }

    /* p2 onto segment q */
    if (len_q_sq > 0.0) {
        double t = ((p2x - q1x) * dx_q + (p2y - q1y) * dy_q) / len_q_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double qx = q1x + t * dx_q;
        double qy = q1y + t * dy_q;
        double dx = qx - p2x;
        double dy = qy - p2y;
        double d2 = dx * dx + dy * dy;
        UPDATE_BEST(d2, p2x, p2y, qx, qy);
    }

    /* q1 onto segment p: closest on P is (p1 + t*dp), point on Q is q1 */
    if (len_p_sq > 0.0) {
        double t = ((q1x - p1x) * dx_p + (q1y - p1y) * dy_p) / len_p_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double px = p1x + t * dx_p;
        double py = p1y + t * dy_p;
        double dx = px - q1x;
        double dy = py - q1y;
        double d2 = dx * dx + dy * dy;
        UPDATE_BEST(d2, px, py, q1x, q1y);
    }

    /* q2 onto segment p */
    if (len_p_sq > 0.0) {
        double t = ((q2x - p1x) * dx_p + (q2y - p1y) * dy_p) / len_p_sq;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        double px = p1x + t * dx_p;
        double py = p1y + t * dy_p;
        double dx = px - q2x;
        double dy = py - q2y;
        double d2 = dx * dx + dy * dy;
        UPDATE_BEST(d2, px, py, q2x, q2y);
    }

    /* --- Endpoint-to-endpoint distances (4 cases) --- */
    {
        double dx, dy, d2;
        dx = p1x - q1x; dy = p1y - q1y; d2 = dx*dx + dy*dy;
        UPDATE_BEST(d2, p1x, p1y, q1x, q1y);
        dx = p1x - q2x; dy = p1y - q2y; d2 = dx*dx + dy*dy;
        UPDATE_BEST(d2, p1x, p1y, q2x, q2y);
        dx = p2x - q1x; dy = p2y - q1y; d2 = dx*dx + dy*dy;
        UPDATE_BEST(d2, p2x, p2y, q1x, q1y);
        dx = p2x - q2x; dy = p2y - q2y; d2 = dx*dx + dy*dy;
        UPDATE_BEST(d2, p2x, p2y, q2x, q2y);
    }

    #undef UPDATE_BEST
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel: LineString minimum clearance (1 thread per geometry)
# ---------------------------------------------------------------------------
# One thread per linestring.  Extracts all segments from the coordinate
# span [geom_offsets[row], geom_offsets[row+1]).  Each segment i goes
# from coord[i] to coord[i+1].  Two segments are adjacent if they share
# a coordinate index (consecutive in the linestring).  We check all
# non-adjacent pairs.
# ---------------------------------------------------------------------------

_LINESTRING_CLEARANCE_KERNEL_SOURCE = _SEGMENT_DISTANCE_HELPER + r"""
extern "C" __global__ void linestring_minimum_clearance(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_clearance,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int ncoords = ce - cs;
    const int nsegs = ncoords - 1;

    /* Need at least 2 segments to have any non-adjacent pair.
       With 2 segments (3 coords), seg 0 and seg 1 are adjacent -> infinity.
       Non-adjacent pairs only exist when nsegs >= 3 (open) or nsegs >= 4 (closed). */
    if (nsegs < 2) {
        out_clearance[row] = 1.0 / 0.0;  /* infinity */
        return;
    }

    /* Detect closed linestring: first coord == last coord.
       When closed, segment 0 and segment nsegs-1 share the closure
       vertex and are adjacent -- must be excluded. */
    int is_closed = 0;
    if (nsegs >= 3) {
        double dx = x[cs] - x[ce - 1];
        double dy = y[cs] - y[ce - 1];
        if (dx * dx + dy * dy < 1e-30) is_closed = 1;
    }

    double min_clear = 1.0 / 0.0;

    for (int i = 0; i < nsegs; i++) {
        const double p1x = x[cs + i];
        const double p1y = y[cs + i];
        const double p2x = x[cs + i + 1];
        const double p2y = y[cs + i + 1];

        for (int j = i + 2; j < nsegs; j++) {
            /* Skip first-last adjacency for closed linestrings */
            if (is_closed && i == 0 && j == nsegs - 1) continue;

            const double q1x = x[cs + j];
            const double q1y = y[cs + j];
            const double q2x = x[cs + j + 1];
            const double q2y = y[cs + j + 1];

            double d = seg_seg_distance(p1x, p1y, p2x, p2y,
                                        q1x, q1y, q2x, q2y);
            if (d < min_clear) min_clear = d;
        }
    }

    out_clearance[row] = min_clear;
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel: Polygon minimum clearance (1 thread per geometry)
# ---------------------------------------------------------------------------
# Per polygon, we collect ALL segments from ALL rings (exterior + holes).
# Each segment is identified by its start coordinate index in the flat
# coordinate array.  Two segments are adjacent if they share a coordinate
# index within the same ring.  Cross-ring segment pairs are never adjacent.
# ---------------------------------------------------------------------------

_POLYGON_CLEARANCE_KERNEL_SOURCE = STRIP_CLOSURE_DEVICE + _SEGMENT_DISTANCE_HELPER + r"""
extern "C" __global__ void polygon_minimum_clearance(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_clearance,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];

    double min_clear = 1.0 / 0.0;

    /* For each pair of rings (including same-ring pairs) */
    for (int ri = first_ring; ri < last_ring; ri++) {
        const int ci_start = ring_offsets[ri];
        const int ci_end = ring_offsets[ri + 1];
        int ni = ci_end - ci_start;

        /* Strip closure vertex if present */
        ni = vs_strip_closure(x, y, ci_start, ci_end, ni, 1e-30);
        if (ni < 2) continue;  /* need at least 1 segment */

        for (int rj = ri; rj < last_ring; rj++) {
            const int cj_start = ring_offsets[rj];
            const int cj_end = ring_offsets[rj + 1];
            int nj = cj_end - cj_start;

            /* Strip closure vertex */
            nj = vs_strip_closure(x, y, cj_start, cj_end, nj, 1e-30);
            if (nj < 2) continue;

            const int nsegs_i = ni - 1;
            const int nsegs_j = nj - 1;
            const int same_ring = (ri == rj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                /* For same ring, start from si+2 to skip adjacent.
                   For different rings, start from 0 (cross-ring never adjacent). */
                const int sj_start = same_ring ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    /* Same ring: skip first-last adjacency for closed rings.
                       Ring is closed (we stripped closure vertex), so segment 0
                       and segment nsegs-1 share the closure vertex. */
                    if (same_ring && si == 0 && sj == nsegs_i - 1) continue;

                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    double d = seg_seg_distance(p1x, p1y, p2x, p2y,
                                                q1x, q1y, q2x, q2y);
                    if (d < min_clear) min_clear = d;
                }
            }
        }
    }

    out_clearance[row] = min_clear;
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel: MultiPolygon minimum clearance (1 thread per geometry)
# ---------------------------------------------------------------------------

_MULTIPOLYGON_CLEARANCE_KERNEL_SOURCE = STRIP_CLOSURE_DEVICE + _SEGMENT_DISTANCE_HELPER + r"""
extern "C" __global__ void multipolygon_minimum_clearance(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_clearance,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    /* Collect all ring spans: we need to iterate all rings across all parts.
       We'll iterate ring pairs with a flat "all rings of this multipolygon" view. */

    /* First, determine the total ring range for this multipolygon */
    const int first_ring = part_offsets[first_part];
    const int last_ring = part_offsets[last_part];

    double min_clear = 1.0 / 0.0;

    for (int ri = first_ring; ri < last_ring; ri++) {
        const int ci_start = ring_offsets[ri];
        const int ci_end = ring_offsets[ri + 1];
        int ni = ci_end - ci_start;

        /* Strip closure vertex */
        ni = vs_strip_closure(x, y, ci_start, ci_end, ni, 1e-30);
        if (ni < 2) continue;

        for (int rj = ri; rj < last_ring; rj++) {
            const int cj_start = ring_offsets[rj];
            const int cj_end = ring_offsets[rj + 1];
            int nj = cj_end - cj_start;

            /* Strip closure vertex */
            nj = vs_strip_closure(x, y, cj_start, cj_end, nj, 1e-30);
            if (nj < 2) continue;

            const int nsegs_i = ni - 1;
            const int nsegs_j = nj - 1;
            const int same_ring = (ri == rj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                const int sj_start = same_ring ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    if (same_ring && si == 0 && sj == nsegs_i - 1) continue;

                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    double d = seg_seg_distance(p1x, p1y, p2x, p2y,
                                                q1x, q1y, q2x, q2y);
                    if (d < min_clear) min_clear = d;
                }
            }
        }
    }

    out_clearance[row] = min_clear;
}
"""
# ---------------------------------------------------------------------------
# NVRTC kernel: MultiLineString minimum clearance (1 thread per geometry)
# ---------------------------------------------------------------------------

_MULTILINESTRING_CLEARANCE_KERNEL_SOURCE = _SEGMENT_DISTANCE_HELPER + r"""
extern "C" __global__ void multilinestring_minimum_clearance(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_clearance,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    double min_clear = 1.0 / 0.0;

    /* Iterate all pairs of parts (including same-part pairs) */
    for (int pi = first_part; pi < last_part; pi++) {
        const int ci_start = part_offsets[pi];
        const int ci_end = part_offsets[pi + 1];
        const int ni = ci_end - ci_start;
        if (ni < 2) continue;
        const int nsegs_i = ni - 1;

        for (int pj = pi; pj < last_part; pj++) {
            const int cj_start = part_offsets[pj];
            const int cj_end = part_offsets[pj + 1];
            const int nj = cj_end - cj_start;
            if (nj < 2) continue;
            const int nsegs_j = nj - 1;
            const int same_part = (pi == pj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                const int sj_start = same_part ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    double d = seg_seg_distance(p1x, p1y, p2x, p2y,
                                                q1x, q1y, q2x, q2y);
                    if (d < min_clear) min_clear = d;
                }
            }
        }
    }

    out_clearance[row] = min_clear;
}
"""
# ===========================================================================
# CLEARANCE LINE kernels: same iteration but track closest point pair
# ===========================================================================
# These kernels output 4 doubles per row: (ax, ay, bx, by) where (ax,ay) and
# (bx,by) are the closest non-adjacent point pair.  For degenerate cases
# (< 2 non-adjacent segments), output NaN to signal empty LineString.
# ===========================================================================

_LINESTRING_CLEARANCE_LINE_KERNEL_SOURCE = _SEGMENT_CLOSEST_POINTS_HELPER + r"""
extern "C" __global__ void linestring_clearance_line(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_ax,
    double* __restrict__ out_ay,
    double* __restrict__ out_bx,
    double* __restrict__ out_by,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int cs = geometry_offsets[row];
    const int ce = geometry_offsets[row + 1];
    const int ncoords = ce - cs;
    const int nsegs = ncoords - 1;

    if (nsegs < 2) {
        out_ax[row] = 0.0 / 0.0;  /* NaN = empty */
        out_ay[row] = 0.0 / 0.0;
        out_bx[row] = 0.0 / 0.0;
        out_by[row] = 0.0 / 0.0;
        return;
    }

    int is_closed = 0;
    if (nsegs >= 3) {
        double dx = x[cs] - x[ce - 1];
        double dy = y[cs] - y[ce - 1];
        if (dx * dx + dy * dy < 1e-30) is_closed = 1;
    }

    double best_d2 = 1e300;
    double best_ax = 0.0 / 0.0, best_ay = 0.0 / 0.0;
    double best_bx = 0.0 / 0.0, best_by = 0.0 / 0.0;

    for (int i = 0; i < nsegs; i++) {
        const double p1x = x[cs + i];
        const double p1y = y[cs + i];
        const double p2x = x[cs + i + 1];
        const double p2y = y[cs + i + 1];

        for (int j = i + 2; j < nsegs; j++) {
            if (is_closed && i == 0 && j == nsegs - 1) continue;

            const double q1x = x[cs + j];
            const double q1y = y[cs + j];
            const double q2x = x[cs + j + 1];
            const double q2y = y[cs + j + 1];

            seg_seg_closest_points(p1x, p1y, p2x, p2y,
                                   q1x, q1y, q2x, q2y,
                                   &best_d2, &best_ax, &best_ay,
                                   &best_bx, &best_by);
        }
    }

    out_ax[row] = best_ax;
    out_ay[row] = best_ay;
    out_bx[row] = best_bx;
    out_by[row] = best_by;
}
"""
_POLYGON_CLEARANCE_LINE_KERNEL_SOURCE = _SEGMENT_CLOSEST_POINTS_HELPER + r"""
extern "C" __global__ void polygon_clearance_line(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_ax,
    double* __restrict__ out_ay,
    double* __restrict__ out_bx,
    double* __restrict__ out_by,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_ring = geometry_offsets[row];
    const int last_ring = geometry_offsets[row + 1];

    double best_d2 = 1e300;
    double best_ax = 0.0 / 0.0, best_ay = 0.0 / 0.0;
    double best_bx = 0.0 / 0.0, best_by = 0.0 / 0.0;

    for (int ri = first_ring; ri < last_ring; ri++) {
        const int ci_start = ring_offsets[ri];
        const int ci_end = ring_offsets[ri + 1];
        int ni = ci_end - ci_start;

        if (ni >= 2) {
            double dx = x[ci_start] - x[ci_end - 1];
            double dy = y[ci_start] - y[ci_end - 1];
            if (dx * dx + dy * dy < 1e-30) ni--;
        }
        if (ni < 2) continue;

        for (int rj = ri; rj < last_ring; rj++) {
            const int cj_start = ring_offsets[rj];
            const int cj_end = ring_offsets[rj + 1];
            int nj = cj_end - cj_start;

            if (nj >= 2) {
                double dx = x[cj_start] - x[cj_end - 1];
                double dy = y[cj_start] - y[cj_end - 1];
                if (dx * dx + dy * dy < 1e-30) nj--;
            }
            if (nj < 2) continue;

            const int nsegs_i = ni - 1;
            const int nsegs_j = nj - 1;
            const int same_ring = (ri == rj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                const int sj_start = same_ring ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    if (same_ring && si == 0 && sj == nsegs_i - 1) continue;

                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    seg_seg_closest_points(p1x, p1y, p2x, p2y,
                                           q1x, q1y, q2x, q2y,
                                           &best_d2, &best_ax, &best_ay,
                                           &best_bx, &best_by);
                }
            }
        }
    }

    out_ax[row] = best_ax;
    out_ay[row] = best_ay;
    out_bx[row] = best_bx;
    out_by[row] = best_by;
}
"""
_MULTIPOLYGON_CLEARANCE_LINE_KERNEL_SOURCE = _SEGMENT_CLOSEST_POINTS_HELPER + r"""
extern "C" __global__ void multipolygon_clearance_line(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_ax,
    double* __restrict__ out_ay,
    double* __restrict__ out_bx,
    double* __restrict__ out_by,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];
    const int first_ring = part_offsets[first_part];
    const int last_ring = part_offsets[last_part];

    double best_d2 = 1e300;
    double best_ax = 0.0 / 0.0, best_ay = 0.0 / 0.0;
    double best_bx = 0.0 / 0.0, best_by = 0.0 / 0.0;

    for (int ri = first_ring; ri < last_ring; ri++) {
        const int ci_start = ring_offsets[ri];
        const int ci_end = ring_offsets[ri + 1];
        int ni = ci_end - ci_start;

        if (ni >= 2) {
            double dx = x[ci_start] - x[ci_end - 1];
            double dy = y[ci_start] - y[ci_end - 1];
            if (dx * dx + dy * dy < 1e-30) ni--;
        }
        if (ni < 2) continue;

        for (int rj = ri; rj < last_ring; rj++) {
            const int cj_start = ring_offsets[rj];
            const int cj_end = ring_offsets[rj + 1];
            int nj = cj_end - cj_start;

            if (nj >= 2) {
                double dx = x[cj_start] - x[cj_end - 1];
                double dy = y[cj_start] - y[cj_end - 1];
                if (dx * dx + dy * dy < 1e-30) nj--;
            }
            if (nj < 2) continue;

            const int nsegs_i = ni - 1;
            const int nsegs_j = nj - 1;
            const int same_ring = (ri == rj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                const int sj_start = same_ring ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    if (same_ring && si == 0 && sj == nsegs_i - 1) continue;

                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    seg_seg_closest_points(p1x, p1y, p2x, p2y,
                                           q1x, q1y, q2x, q2y,
                                           &best_d2, &best_ax, &best_ay,
                                           &best_bx, &best_by);
                }
            }
        }
    }

    out_ax[row] = best_ax;
    out_ay[row] = best_ay;
    out_bx[row] = best_bx;
    out_by[row] = best_by;
}
"""
_MULTILINESTRING_CLEARANCE_LINE_KERNEL_SOURCE = _SEGMENT_CLOSEST_POINTS_HELPER + r"""
extern "C" __global__ void multilinestring_clearance_line(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ part_offsets,
    const int* __restrict__ geometry_offsets,
    double* __restrict__ out_ax,
    double* __restrict__ out_ay,
    double* __restrict__ out_bx,
    double* __restrict__ out_by,
    int row_count
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;

    const int first_part = geometry_offsets[row];
    const int last_part = geometry_offsets[row + 1];

    double best_d2 = 1e300;
    double best_ax = 0.0 / 0.0, best_ay = 0.0 / 0.0;
    double best_bx = 0.0 / 0.0, best_by = 0.0 / 0.0;

    for (int pi = first_part; pi < last_part; pi++) {
        const int ci_start = part_offsets[pi];
        const int ci_end = part_offsets[pi + 1];
        const int ni = ci_end - ci_start;
        if (ni < 2) continue;
        const int nsegs_i = ni - 1;

        for (int pj = pi; pj < last_part; pj++) {
            const int cj_start = part_offsets[pj];
            const int cj_end = part_offsets[pj + 1];
            const int nj = cj_end - cj_start;
            if (nj < 2) continue;
            const int nsegs_j = nj - 1;
            const int same_part = (pi == pj);

            for (int si = 0; si < nsegs_i; si++) {
                const double p1x = x[ci_start + si];
                const double p1y = y[ci_start + si];
                const double p2x = x[ci_start + si + 1];
                const double p2y = y[ci_start + si + 1];

                const int sj_start = same_part ? si + 2 : 0;

                for (int sj = sj_start; sj < nsegs_j; sj++) {
                    const double q1x = x[cj_start + sj];
                    const double q1y = y[cj_start + sj];
                    const double q2x = x[cj_start + sj + 1];
                    const double q2y = y[cj_start + sj + 1];

                    seg_seg_closest_points(p1x, p1y, p2x, p2y,
                                           q1x, q1y, q2x, q2y,
                                           &best_d2, &best_ax, &best_ay,
                                           &best_bx, &best_by);
                }
            }
        }
    }

    out_ax[row] = best_ax;
    out_ay[row] = best_ay;
    out_bx[row] = best_bx;
    out_by[row] = best_by;
}
"""
# ADR-0002: METRIC class -- fp64 required for distance computation.
# No fp32 variants needed; we wire precision for observability only.
_LINESTRING_CLEARANCE_FP64 = _LINESTRING_CLEARANCE_KERNEL_SOURCE
_POLYGON_CLEARANCE_FP64 = _POLYGON_CLEARANCE_KERNEL_SOURCE
_MULTIPOLYGON_CLEARANCE_FP64 = _MULTIPOLYGON_CLEARANCE_KERNEL_SOURCE
_MULTILINESTRING_CLEARANCE_FP64 = _MULTILINESTRING_CLEARANCE_KERNEL_SOURCE
# ADR-0002: CONSTRUCTIVE class -- fp64 required (output is geometry).
_LINESTRING_CLEARANCE_LINE_FP64 = _LINESTRING_CLEARANCE_LINE_KERNEL_SOURCE
_POLYGON_CLEARANCE_LINE_FP64 = _POLYGON_CLEARANCE_LINE_KERNEL_SOURCE
_MULTIPOLYGON_CLEARANCE_LINE_FP64 = _MULTIPOLYGON_CLEARANCE_LINE_KERNEL_SOURCE
_MULTILINESTRING_CLEARANCE_LINE_FP64 = _MULTILINESTRING_CLEARANCE_LINE_KERNEL_SOURCE

# Kernel name tuples
_LINESTRING_CLEARANCE_NAMES = ("linestring_minimum_clearance",)
_POLYGON_CLEARANCE_NAMES = ("polygon_minimum_clearance",)
_MULTIPOLYGON_CLEARANCE_NAMES = ("multipolygon_minimum_clearance",)
_MULTILINESTRING_CLEARANCE_NAMES = ("multilinestring_minimum_clearance",)

_LINESTRING_CLEARANCE_LINE_NAMES = ("linestring_clearance_line",)
_POLYGON_CLEARANCE_LINE_NAMES = ("polygon_clearance_line",)
_MULTIPOLYGON_CLEARANCE_LINE_NAMES = ("multipolygon_clearance_line",)
_MULTILINESTRING_CLEARANCE_LINE_NAMES = ("multilinestring_clearance_line",)
