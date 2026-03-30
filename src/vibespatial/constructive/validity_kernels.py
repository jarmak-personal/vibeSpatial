"""NVRTC kernel sources for validity."""

from __future__ import annotations

from vibespatial.cuda.device_functions.orient2d import ORIENT2D_DEVICE
from vibespatial.cuda.preamble import PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: is_valid_rings — 1 thread per ring (Tier 1, ADR-0033)
#
# Checks structural validity of polygon rings:
#   1. Minimum 4 coordinates
#   2. Ring closure (first == last)
# Note: orientation is NOT checked (GEOS does not enforce winding in is_valid).
# ---------------------------------------------------------------------------

_IS_VALID_RINGS_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ void is_valid_rings(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ is_exterior,
    int* __restrict__ ring_valid,
    int ring_count
) {{
    const int ring = blockIdx.x * blockDim.x + threadIdx.x;
    if (ring >= ring_count) return;

    const int start = ring_offsets[ring];
    const int end = ring_offsets[ring + 1];
    const int length = end - start;

    /* Check 1: minimum 4 coordinates for a valid ring */
    if (length < 4) {{
        ring_valid[ring] = 0;
        return;
    }}

    /* Check 2: ring closure (first == last) */
    if (x[start] != x[end - 1] || y[start] != y[end - 1]) {{
        ring_valid[ring] = 0;
        return;
    }}

    /* Note: orientation (CCW exterior / CW hole) is NOT checked here.
       GEOS/Shapely is_valid does not enforce winding order — it is a
       convention, not a validity requirement per OGC SFS.  Orientation
       can be checked separately via orient.py if needed. */

    ring_valid[ring] = 1;
}}
"""
_IS_VALID_RINGS_KERNEL_NAMES = ("is_valid_rings",)
_IS_VALID_RINGS_FP64 = _IS_VALID_RINGS_KERNEL_SOURCE.format(compute_type="double")
# ---------------------------------------------------------------------------
# NVRTC kernel: is_simple_segments — 1 block per span, O(n^2) segment check
#
# All threads in a block cooperate to test segment pairs for a single
# span (geometry or ring).  Uses atomicExch for early-exit on first
# crossing found.
# ---------------------------------------------------------------------------

_IS_SIMPLE_SEGMENTS_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __global__ __launch_bounds__(256, 4)
void is_simple_segments(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ span_offsets,
    int* __restrict__ result,
    int is_ring,
    int span_count
) {{
    const int span = blockIdx.x;
    if (span >= span_count) return;

    const int start = span_offsets[span];
    const int end = span_offsets[span + 1];
    const int n_coords = end - start;

    /* < 4 coords means at most 2 segments which are adjacent — always simple */
    if (n_coords < 4) {{
        if (threadIdx.x == 0) result[span] = 1;
        return;
    }}

    const int n_segs = n_coords - 1;

    __shared__ int found_crossing;
    if (threadIdx.x == 0) found_crossing = 0;
    __syncthreads();

    /* Thread-striped double loop over non-adjacent segment pairs */
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    for (int i = 0; i < n_segs && !found_crossing; i++) {{
        const double ax0 = x[start + i];
        const double ay0 = y[start + i];
        const double ax1 = x[start + i + 1];
        const double ay1 = y[start + i + 1];

        const double dx_a = ax1 - ax0;
        const double dy_a = ay1 - ay0;

        for (int j = i + 2 + tid; j < n_segs && !found_crossing; j += stride) {{
            /* Skip first-last adjacency for closed rings */
            if (is_ring && i == 0 && j == n_segs - 1) continue;

            const double bx0 = x[start + j];
            const double by0 = y[start + j];
            const double bx1 = x[start + j + 1];
            const double by1 = y[start + j + 1];

            /* Check 1: endpoint coincidence (figure-8 patterns where a
               vertex is visited twice at non-adjacent ring positions).
               Only flag if intermediate path has non-zero length (excludes
               degenerate duplicate vertices). */
            if ((ax0 == bx0 && ay0 == by0) || (ax0 == bx1 && ay0 == by1) ||
                (ax1 == bx0 && ay1 == by0) || (ax1 == bx1 && ay1 == by1)) {{
                int has_travel = 0;
                for (int k = i + 1; k < j && !has_travel; k++) {{
                    if (x[start + k] != x[start + k + 1] ||
                        y[start + k] != y[start + k + 1]) {{
                        has_travel = 1;
                    }}
                }}
                if (has_travel) {{
                    atomicExch(&found_crossing, 1);
                    continue;
                }}
            }}

            /* Check 2: proper interior crossing */
            const double dx_b = bx1 - bx0;
            const double dy_b = by1 - by0;
            const double denom = dx_a * dy_b - dy_a * dx_b;

            if (fabs(denom) > 1e-15) {{
                const double dx_ab = bx0 - ax0;
                const double dy_ab = by0 - ay0;
                const double t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
                const double u = (dx_ab * dy_a - dy_ab * dx_a) / denom;
                if (t > 1e-12 && t < (1.0 - 1e-12) &&
                    u > 1e-12 && u < (1.0 - 1e-12)) {{
                    atomicExch(&found_crossing, 1);
                }}
            }}
        }}
    }}

    __syncthreads();
    if (threadIdx.x == 0) {{
        result[span] = found_crossing ? 0 : 1;
    }}
}}
"""
_IS_SIMPLE_SEGMENTS_KERNEL_NAMES = ("is_simple_segments",)
_IS_SIMPLE_SEGMENTS_FP64 = _IS_SIMPLE_SEGMENTS_KERNEL_SOURCE.format(
    compute_type="double",
)
# ---------------------------------------------------------------------------
# NVRTC kernel: holes_in_shell — 1 thread per hole ring (Tier 1, ADR-0033)
#
# Checks whether each hole ring's first vertex is contained within its
# polygon's exterior ring using even-odd ray-casting.  A point on the
# boundary counts as "inside" (valid per OGC: hole may touch shell at a
# point).
#
# PREDICATE class, fp64 only — no centering needed for validity checks.
# ---------------------------------------------------------------------------

_HOLES_IN_SHELL_KERNEL_SOURCE = PRECISION_PREAMBLE + r"""
extern "C" __device__ inline bool point_on_segment_validity(
    double px,
    double py,
    double ax,
    double ay,
    double bx,
    double by
) {{
    const double dx = bx - ax;
    const double dy = by - ay;
    const double cross = ((px - ax) * dy) - ((py - ay) * dx);
    const double scale = fabs(dx) + fabs(dy) + 1.0;
    if (fabs(cross) > (1e-12 * scale)) {{
        return false;
    }}
    const double minx = ax < bx ? ax : bx;
    const double maxx = ax > bx ? ax : bx;
    const double miny = ay < by ? ay : by;
    const double maxy = ay > by ? ay : by;
    return px >= (minx - 1e-12) && px <= (maxx + 1e-12)
        && py >= (miny - 1e-12) && py <= (maxy + 1e-12);
}}

extern "C" __device__ inline bool ring_contains_point_validity(
    double px,
    double py,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    int ring_idx
) {{
    const int coord_start = ring_offsets[ring_idx];
    const int coord_end = ring_offsets[ring_idx + 1];

    if ((coord_end - coord_start) < 4) {{
        return false;
    }}

    bool inside = false;
    for (int coord = coord_start + 1; coord < coord_end; ++coord) {{
        const double ax = x[coord - 1];
        const double ay = y[coord - 1];
        const double bx = x[coord];
        const double by = y[coord];

        /* Boundary test: point on edge counts as inside (OGC validity) */
        if (point_on_segment_validity(px, py, ax, ay, bx, by)) {{
            return true;
        }}

        /* Even-odd crossing test */
        const bool intersects = ((ay > py) != (by > py)) &&
            (px < (((bx - ax) * (py - ay)) / (by - ay)) + ax);
        if (intersects) {{
            inside = !inside;
        }}
    }}
    return inside;
}}

extern "C" __global__ void holes_in_shell(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ hole_ring_indices,
    const int* __restrict__ exterior_ring_indices,
    int* __restrict__ hole_valid,
    int hole_count
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hole_count) return;

    const int hole_ring = hole_ring_indices[idx];
    const int ext_ring = exterior_ring_indices[idx];

    /* Read the first coordinate of the hole ring */
    const int hole_start = ring_offsets[hole_ring];
    const double px = x[hole_start];
    const double py = y[hole_start];

    /* Ray-cast against the exterior ring */
    hole_valid[idx] = ring_contains_point_validity(
        px, py, x, y, ring_offsets, ext_ring
    ) ? 1 : 0;
}}
"""
_HOLES_IN_SHELL_KERNEL_NAMES = ("holes_in_shell",)
_HOLES_IN_SHELL_FP64 = _HOLES_IN_SHELL_KERNEL_SOURCE.format(compute_type="double")
# ---------------------------------------------------------------------------
# NVRTC kernel: ring_pair_interaction — 1 block per polygon part (Tier 1)
#
# Detects inter-ring violations within each polygon:
#   - Proper crossing (segments from distinct rings cross) -> INVALID
#   - Collinear overlap (rings share a segment) -> INVALID
#   - Multi-touch (2+ distinct contact points between a ring pair) -> INVALID
#   - Single touch (1 contact point between a ring pair) -> VALID per OGC
#
# PREDICATE class, fp64 only.  Uses shared vs_orient2d for exact
# orientation predicates (via cuda.device_functions.orient2d).
# ---------------------------------------------------------------------------

_RING_PAIR_INTERACTION_KERNEL_SOURCE = PRECISION_PREAMBLE + ORIENT2D_DEVICE + r"""
/* orient2d predicate provided by ORIENT2D_DEVICE (vs_orient2d) */

/* Check if point (px,py) is on segment (ax,ay)-(bx,by), assuming collinear.
   Uses exact double comparisons (valid for fp64). */
__device__ int point_on_seg_collinear_rpi(
    double px, double py,
    double ax, double ay,
    double bx, double by
) {{
    double minx = ax < bx ? ax : bx;
    double maxx = ax > bx ? ax : bx;
    double miny = ay < by ? ay : by;
    double maxy = ay > by ? ay : by;
    return (px >= minx && px <= maxx && py >= miny && py <= maxy) ? 1 : 0;
}}

/* Record a touch point in shared memory, returning 1 if recorded. */
__device__ int record_touch(
    double tx, double ty, int ri, int rj,
    volatile int* touch_count,
    double* touch_x, double* touch_y,
    int* touch_ring_i, int* touch_ring_j
) {{
    int slot = atomicAdd((int*)touch_count, 1);
    if (slot < 64) {{
        touch_x[slot] = tx;
        touch_y[slot] = ty;
        touch_ring_i[slot] = ri;
        touch_ring_j[slot] = rj;
        return 1;
    }}
    return 0;  /* overflow -- will flag invalid conservatively */
}}

/* ------------------------------------------------------------------ */
/* Main kernel: 1 block per polygon part                              */
/* ------------------------------------------------------------------ */

extern "C" __global__ void __launch_bounds__(256, 4)
ring_pair_interaction(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const int* __restrict__ ring_offsets,
    const int* __restrict__ poly_ring_starts,
    const int* __restrict__ poly_ring_ends,
    int* __restrict__ poly_valid,
    const int poly_count
) {{
    const int poly = blockIdx.x;
    if (poly >= poly_count) return;

    const int ring_start = poly_ring_starts[poly];
    const int ring_end = poly_ring_ends[poly];
    const int n_rings = ring_end - ring_start;

    /* Single-ring polygons have no inter-ring interactions */
    if (n_rings < 2) {{
        if (threadIdx.x == 0) poly_valid[poly] = 1;
        return;
    }}

    /* Shared memory for early-exit flag + touch point recording */
    __shared__ int found_invalid;
    __shared__ int touch_count;
    __shared__ double touch_x[64];
    __shared__ double touch_y[64];
    __shared__ int touch_ring_i[64];
    __shared__ int touch_ring_j[64];

    if (threadIdx.x == 0) {{
        found_invalid = 0;
        touch_count = 0;
    }}
    __syncthreads();

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    /* Iterate all pairs of distinct rings (i, j) with i < j */
    for (int ri = 0; ri < n_rings && !found_invalid; ++ri) {{
        const int ring_i = ring_start + ri;
        const int ci_start = ring_offsets[ring_i];
        const int ci_end = ring_offsets[ring_i + 1];
        const int n_segs_i = ci_end - ci_start - 1;  /* closed ring: last = first */
        if (n_segs_i < 1) continue;

        for (int rj = ri + 1; rj < n_rings && !found_invalid; ++rj) {{
            const int ring_j = ring_start + rj;
            const int cj_start = ring_offsets[ring_j];
            const int cj_end = ring_offsets[ring_j + 1];
            const int n_segs_j = cj_end - cj_start - 1;
            if (n_segs_j < 1) continue;

            /* Total segment pairs for this ring pair */
            const int total_pairs = n_segs_i * n_segs_j;

            /* Threads stripe across segment pairs */
            for (int pair_idx = tid; pair_idx < total_pairs && !found_invalid; pair_idx += stride) {{
                const int si = pair_idx / n_segs_j;
                const int sj = pair_idx - si * n_segs_j;

                const double ax = x[ci_start + si];
                const double ay = y[ci_start + si];
                const double bx = x[ci_start + si + 1];
                const double by = y[ci_start + si + 1];

                const double cx = x[cj_start + sj];
                const double cy = y[cj_start + sj];
                const double dx = x[cj_start + sj + 1];
                const double dy = y[cj_start + sj + 1];

                /* MBR early reject */
                double ab_minx = ax < bx ? ax : bx;
                double ab_maxx = ax > bx ? ax : bx;
                double ab_miny = ay < by ? ay : by;
                double ab_maxy = ay > by ? ay : by;
                double cd_minx = cx < dx ? cx : dx;
                double cd_maxx = cx > dx ? cx : dx;
                double cd_miny = cy < dy ? cy : dy;
                double cd_maxy = cy > dy ? cy : dy;

                if (ab_maxx < cd_minx || cd_maxx < ab_minx ||
                    ab_maxy < cd_miny || cd_maxy < ab_miny) {{
                    continue;
                }}

                /* Shewchuk exact orientations */
                int o1 = vs_orient2d(ax, ay, bx, by, cx, cy);
                int o2 = vs_orient2d(ax, ay, bx, by, dx, dy);
                int o3 = vs_orient2d(cx, cy, dx, dy, ax, ay);
                int o4 = vs_orient2d(cx, cy, dx, dy, bx, by);

                /* Case 1: Proper crossing */
                if (o1 != 0 && o2 != 0 && o1 != o2 && o3 != 0 && o4 != 0 && o3 != o4) {{
                    atomicExch(&found_invalid, 1);
                    continue;
                }}

                /* Case 2: Collinear -- all four orientations zero */
                if (o1 == 0 && o2 == 0 && o3 == 0 && o4 == 0) {{
                    int a_on_cd = point_on_seg_collinear_rpi(ax, ay, cx, cy, dx, dy);
                    int b_on_cd = point_on_seg_collinear_rpi(bx, by, cx, cy, dx, dy);
                    int c_on_ab = point_on_seg_collinear_rpi(cx, cy, ax, ay, bx, by);
                    int d_on_ab = point_on_seg_collinear_rpi(dx, dy, ax, ay, bx, by);

                    int containments = a_on_cd + b_on_cd + c_on_ab + d_on_ab;
                    if (containments >= 3) {{
                        /* True overlap: segments share an interval */
                        atomicExch(&found_invalid, 1);
                        continue;
                    }}

                    if (containments == 2) {{
                        int shared_endpoints = 0;
                        if (ax == cx && ay == cy) shared_endpoints++;
                        if (ax == dx && ay == dy) shared_endpoints++;
                        if (bx == cx && by == cy) shared_endpoints++;
                        if (bx == dx && by == dy) shared_endpoints++;

                        if (shared_endpoints >= 1) {{
                            double tx, ty;
                            if (ax == cx && ay == cy) {{ tx = ax; ty = ay; }}
                            else if (ax == dx && ay == dy) {{ tx = ax; ty = ay; }}
                            else if (bx == cx && by == cy) {{ tx = bx; ty = by; }}
                            else {{ tx = bx; ty = by; }}
                            record_touch(tx, ty, ri, rj,
                                         &touch_count, touch_x, touch_y,
                                         touch_ring_i, touch_ring_j);
                        }} else {{
                            atomicExch(&found_invalid, 1);
                        }}
                        continue;
                    }}
                    if (containments == 1) {{
                        double tx, ty;
                        if (a_on_cd) {{ tx = ax; ty = ay; }}
                        else if (b_on_cd) {{ tx = bx; ty = by; }}
                        else if (c_on_ab) {{ tx = cx; ty = cy; }}
                        else {{ tx = dx; ty = dy; }}
                        record_touch(tx, ty, ri, rj,
                                     &touch_count, touch_x, touch_y,
                                     touch_ring_i, touch_ring_j);
                    }}
                    continue;
                }}

                /* Case 3: T-intersection -- one endpoint on the other segment */
                if (o1 == 0 && point_on_seg_collinear_rpi(cx, cy, ax, ay, bx, by)) {{
                    record_touch(cx, cy, ri, rj,
                                 &touch_count, touch_x, touch_y,
                                 touch_ring_i, touch_ring_j);
                }}
                else if (o2 == 0 && point_on_seg_collinear_rpi(dx, dy, ax, ay, bx, by)) {{
                    record_touch(dx, dy, ri, rj,
                                 &touch_count, touch_x, touch_y,
                                 touch_ring_i, touch_ring_j);
                }}
                else if (o3 == 0 && point_on_seg_collinear_rpi(ax, ay, cx, cy, dx, dy)) {{
                    record_touch(ax, ay, ri, rj,
                                 &touch_count, touch_x, touch_y,
                                 touch_ring_i, touch_ring_j);
                }}
                else if (o4 == 0 && point_on_seg_collinear_rpi(bx, by, cx, cy, dx, dy)) {{
                    record_touch(bx, by, ri, rj,
                                 &touch_count, touch_x, touch_y,
                                 touch_ring_i, touch_ring_j);
                }}
            }}
        }}
    }}

    __syncthreads();

    /* Thread 0: check for multi-touch (2+ distinct points per ring pair) */
    if (threadIdx.x == 0) {{
        if (found_invalid) {{
            poly_valid[poly] = 0;
            return;
        }}

        int tc = touch_count;
        if (tc > 64) {{
            /* Overflow: conservatively flag invalid */
            poly_valid[poly] = 0;
            return;
        }}

        /* Deduplicate touch points per ring pair and check for multi-touch.
           For each ring pair (ri, rj), count distinct touch points.
           If any pair has 2+, the interior is disconnected. */
        int invalid = 0;
        for (int a = 0; a < tc && !invalid; ++a) {{
            int pair_ri = touch_ring_i[a];
            int pair_rj = touch_ring_j[a];
            int is_first = 1;
            for (int b = 0; b < a; ++b) {{
                if (touch_ring_i[b] == pair_ri && touch_ring_j[b] == pair_rj) {{
                    is_first = 0;
                    break;
                }}
            }}
            if (!is_first) continue;
            int distinct = 1;
            for (int b = a + 1; b < tc; ++b) {{
                if (touch_ring_i[b] != pair_ri || touch_ring_j[b] != pair_rj) continue;
                int is_dup = 0;
                for (int c = a; c < b; ++c) {{
                    if (touch_ring_i[c] != pair_ri || touch_ring_j[c] != pair_rj) continue;
                    if (touch_x[c] == touch_x[b] && touch_y[c] == touch_y[b]) {{
                        is_dup = 1;
                        break;
                    }}
                }}
                if (!is_dup) {{
                    distinct++;
                    if (distinct >= 2) {{
                        invalid = 1;
                        break;
                    }}
                }}
            }}
        }}

        poly_valid[poly] = invalid ? 0 : 1;
    }}
}}
"""
_RING_PAIR_INTERACTION_KERNEL_NAMES = ("ring_pair_interaction",)
_RING_PAIR_INTERACTION_FP64 = _RING_PAIR_INTERACTION_KERNEL_SOURCE.format(
    compute_type="double",
)
