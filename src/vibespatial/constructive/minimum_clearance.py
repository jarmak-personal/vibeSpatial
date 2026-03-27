"""GPU-accelerated minimum_clearance computation.

Minimum clearance is the smallest distance between any two non-adjacent
vertices/edges of a geometry.  Returns a scalar float per geometry.

Architecture (ADR-0033 tier classification):
- Point/MultiPoint: infinity (no segments, no clearance)
- LineString: O(n^2) pairwise non-adjacent segment distance (Tier 1 NVRTC)
- Polygon: O(n^2) all-segments including cross-ring pairs (Tier 1 NVRTC)
- MultiLineString: O(n^2) all segments across all parts (Tier 1 NVRTC)
- MultiPolygon: O(n^2) all segments across all parts and rings (Tier 1 NVRTC)

Precision (ADR-0002): METRIC class -- fp64 required for distance computation.
The kernel is templated on compute_t for observability but stays fp64 per
the bead spec requirement that distance computation uses fp64.

Two segments are "adjacent" if they share a vertex index in the flattened
coordinate array.  Adjacent segments always have distance 0 at the shared
vertex and are excluded from the minimum.

The kernel computes full segment-to-segment distance (not just
vertex-to-vertex) using the parametric closest-approach method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

if TYPE_CHECKING:
    from vibespatial.runtime.precision import PrecisionMode, PrecisionPlan

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

_POLYGON_CLEARANCE_KERNEL_SOURCE = _SEGMENT_DISTANCE_HELPER + r"""
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
        if (ni >= 2) {
            double dx = x[ci_start] - x[ci_end - 1];
            double dy = y[ci_start] - y[ci_end - 1];
            if (dx * dx + dy * dy < 1e-30) ni--;
        }
        if (ni < 2) continue;  /* need at least 1 segment */

        for (int rj = ri; rj < last_ring; rj++) {
            const int cj_start = ring_offsets[rj];
            const int cj_end = ring_offsets[rj + 1];
            int nj = cj_end - cj_start;

            /* Strip closure vertex */
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

_MULTIPOLYGON_CLEARANCE_KERNEL_SOURCE = _SEGMENT_DISTANCE_HELPER + r"""
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

            /* Strip closure vertex */
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

# ---------------------------------------------------------------------------
# Kernel names and precompiled source
# ---------------------------------------------------------------------------

_LINESTRING_CLEARANCE_NAMES = ("linestring_minimum_clearance",)
_POLYGON_CLEARANCE_NAMES = ("polygon_minimum_clearance",)
_MULTIPOLYGON_CLEARANCE_NAMES = ("multipolygon_minimum_clearance",)
_MULTILINESTRING_CLEARANCE_NAMES = ("multilinestring_minimum_clearance",)

# ADR-0002: METRIC class -- fp64 required for distance computation.
# No fp32 variants needed; we wire precision for observability only.
_LINESTRING_CLEARANCE_FP64 = _LINESTRING_CLEARANCE_KERNEL_SOURCE
_POLYGON_CLEARANCE_FP64 = _POLYGON_CLEARANCE_KERNEL_SOURCE
_MULTIPOLYGON_CLEARANCE_FP64 = _MULTIPOLYGON_CLEARANCE_KERNEL_SOURCE
_MULTILINESTRING_CLEARANCE_FP64 = _MULTILINESTRING_CLEARANCE_KERNEL_SOURCE

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("linestring-clearance-fp64", _LINESTRING_CLEARANCE_FP64, _LINESTRING_CLEARANCE_NAMES),
    ("polygon-clearance-fp64", _POLYGON_CLEARANCE_FP64, _POLYGON_CLEARANCE_NAMES),
    ("multipolygon-clearance-fp64", _MULTIPOLYGON_CLEARANCE_FP64, _MULTIPOLYGON_CLEARANCE_NAMES),
    ("multilinestring-clearance-fp64", _MULTILINESTRING_CLEARANCE_FP64, _MULTILINESTRING_CLEARANCE_NAMES),
])


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "minimum_clearance",
    "gpu-cuda-python",
    kernel_class=KernelClass.METRIC,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=("linestring", "polygon", "multipolygon", "multilinestring"),
    supports_mixed=True,
    tags=("cuda-python", "metric", "clearance", "segment-distance"),
)
def _minimum_clearance_gpu(
    owned: OwnedGeometryArray,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray:
    """GPU-accelerated minimum clearance.  Returns float64 array of shape (row_count,)."""
    runtime = get_cuda_runtime()
    row_count = owned.row_count
    result = np.full(row_count, np.inf, dtype=np.float64)

    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    device_state = owned.device_state

    # --- LineString family ---
    _launch_linestring_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- MultiLineString family ---
    _launch_multilinestring_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- Polygon family ---
    _launch_polygon_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # --- MultiPolygon family ---
    _launch_multipolygon_clearance(
        owned, result, tags, family_row_offsets, device_state, runtime,
    )

    # Point, MultiPoint: clearance = infinity (already initialized)
    return result


def _launch_linestring_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch linestring minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.LINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.LINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.LINESTRING]
    if buf.row_count == 0 or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "linestring-clearance-fp64",
        _LINESTRING_CLEARANCE_FP64,
        _LINESTRING_CLEARANCE_NAMES,
    )
    kernel = kernels["linestring_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.LINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.LINESTRING]
        d_x, d_y = ds.x, ds.y
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_multilinestring_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch multilinestring minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTILINESTRING]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTILINESTRING not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTILINESTRING]
    if buf.row_count == 0 or buf.part_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "multilinestring-clearance-fp64",
        _MULTILINESTRING_CLEARANCE_FP64,
        _MULTILINESTRING_CLEARANCE_NAMES,
    )
    kernel = kernels["multilinestring_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTILINESTRING not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTILINESTRING]
        d_x, d_y = ds.x, ds.y
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_part, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_part), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_polygon_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch polygon minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.POLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.POLYGON]
    if buf.row_count == 0 or buf.ring_offsets is None or len(buf.geometry_offsets) < 2:
        return

    kernels = compile_kernel_group(
        "polygon-clearance-fp64",
        _POLYGON_CLEARANCE_FP64,
        _POLYGON_CLEARANCE_NAMES,
    )
    kernel = kernels["polygon_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.POLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.POLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_geom), ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


def _launch_multipolygon_clearance(
    owned, result, tags, family_row_offsets, device_state, runtime,
):
    """Launch multipolygon minimum clearance kernel."""
    tag = FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]
    mask = tags == tag
    if not np.any(mask) or GeometryFamily.MULTIPOLYGON not in owned.families:
        return
    buf = owned.families[GeometryFamily.MULTIPOLYGON]
    if (
        buf.row_count == 0
        or buf.ring_offsets is None
        or buf.part_offsets is None
        or len(buf.geometry_offsets) < 2
    ):
        return

    kernels = compile_kernel_group(
        "multipolygon-clearance-fp64",
        _MULTIPOLYGON_CLEARANCE_FP64,
        _MULTIPOLYGON_CLEARANCE_NAMES,
    )
    kernel = kernels["multipolygon_minimum_clearance"]
    global_rows = np.flatnonzero(mask)
    family_rows = family_row_offsets[global_rows]
    n = buf.row_count

    needs_free = (
        device_state is None
        or GeometryFamily.MULTIPOLYGON not in (device_state.families if device_state else {})
    )
    allocated = []
    if not needs_free:
        ds = device_state.families[GeometryFamily.MULTIPOLYGON]
        d_x, d_y = ds.x, ds.y
        d_ring = ds.ring_offsets
        d_part = ds.part_offsets
        d_geom = ds.geometry_offsets
    else:
        d_x = runtime.from_host(buf.x)
        d_y = runtime.from_host(buf.y)
        d_ring = runtime.from_host(buf.ring_offsets.astype(np.int32))
        d_part = runtime.from_host(buf.part_offsets.astype(np.int32))
        d_geom = runtime.from_host(buf.geometry_offsets.astype(np.int32))
        allocated.extend([d_x, d_y, d_ring, d_part, d_geom])

    d_out = runtime.allocate((n,), np.float64)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_x), ptr(d_y), ptr(d_ring), ptr(d_part), ptr(d_geom),
             ptr(d_out), n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, n)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        family_result = runtime.copy_device_to_host(d_out)
        result[global_rows] = family_result[family_rows]
    finally:
        runtime.free(d_out)
        for d in allocated:
            runtime.free(d)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def _minimum_clearance_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU fallback via Shapely."""
    import shapely as _shapely

    geoms = np.asarray(owned.to_shapely(), dtype=object)
    return _shapely.minimum_clearance(geoms)


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def minimum_clearance_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = "auto",
) -> np.ndarray:
    """Compute minimum clearance from OwnedGeometryArray coordinate buffers.

    GPU path uses ADR-0002 METRIC-class precision dispatch (fp64 required
    for distance computation).  Returns float64 array of shape (row_count,).

    Zero host/device transfers mid-process.  When owned.device_state
    is populated (vibeFrame path), GPU kernels read directly from
    device pointers with no copy.
    """
    from vibespatial.runtime import RuntimeSelection
    from vibespatial.runtime.precision import select_precision_plan

    row_count = owned.row_count
    if row_count == 0:
        return np.empty(0, dtype=np.float64)

    selection = plan_dispatch_selection(
        kernel_name="minimum_clearance",
        kernel_class=KernelClass.METRIC,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    if selection.selected is ExecutionMode.GPU:
        precision_plan = select_precision_plan(
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="minimum_clearance GPU dispatch",
            ),
            kernel_class=KernelClass.METRIC,
            requested=precision,
        )
        try:
            result = _minimum_clearance_gpu(owned, precision_plan=precision_plan)
            result[~owned.validity] = np.nan
        except Exception:
            pass  # fall through to CPU
        else:
            record_dispatch_event(
                surface="geopandas.array.minimum_clearance",
                operation="minimum_clearance",
                implementation="gpu_nvrtc_segment_distance",
                reason="GPU NVRTC minimum clearance kernel",
                detail=f"rows={row_count}",
                selected=ExecutionMode.GPU,
            )
            return result

    record_dispatch_event(
        surface="geopandas.array.minimum_clearance",
        operation="minimum_clearance",
        implementation="shapely",
        reason="CPU fallback",
        detail=f"rows={row_count}",
        selected=ExecutionMode.CPU,
    )
    result = _minimum_clearance_cpu(owned)
    result[~owned.validity] = np.nan
    return result
