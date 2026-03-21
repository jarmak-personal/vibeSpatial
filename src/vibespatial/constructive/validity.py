"""GPU-accelerated geometry validity and simplicity checks.

Structural validity checks (is_valid) and geometric simplicity checks
(is_simple) computed directly from OwnedGeometryArray coordinate buffers
without Shapely materialization.

is_valid checks:
- Ring closure (first coord == last coord for polygon rings)
- Minimum coordinate counts (LineString >= 2, Polygon ring >= 4)
- Orientation (exterior CCW, holes CW via shoelace signed area)
- Null/empty geometries return True (matching Shapely behavior)

is_simple checks:
- Points/MultiPoints: always True
- LineStrings: no self-intersection (brute-force O(n^2) segment test)
- Null/empty geometries return True

ADR-0033: Tier 1 NVRTC for is_valid_rings and is_simple_segments kernels.
          Tier 2 CuPy for LineString validity (offset arithmetic) and
          per-ring-to-per-geometry reduction.
ADR-0002: PREDICATE class.  Constructive-style ring checks stay fp64 on
          all devices (orientation/closure are exact comparisons, not
          distance metrics).  Precision preamble wired for observability.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass

from .measurement import _PRECISION_PREAMBLE

# ---------------------------------------------------------------------------
# NVRTC kernel: is_valid_rings — 1 thread per ring (Tier 1, ADR-0033)
#
# Checks structural validity of polygon rings:
#   1. Minimum 4 coordinates
#   2. Ring closure (first == last)
#   3. Correct orientation (exterior CCW / positive area, holes CW / negative)
# ---------------------------------------------------------------------------

_IS_VALID_RINGS_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
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

    /* Check 3: orientation via shoelace signed area (stays fp64 for exactness) */
    double area2 = 0.0;
    for (int j = start; j < end - 1; j++) {{
        area2 += x[j] * y[j + 1] - x[j + 1] * y[j];
    }}

    const int ext = is_exterior[ring];
    if (ext && area2 < 0.0) {{
        ring_valid[ring] = 0;  /* exterior should be CCW (positive area) */
        return;
    }}
    if (!ext && area2 > 0.0) {{
        ring_valid[ring] = 0;  /* hole should be CW (negative area) */
        return;
    }}

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

_IS_SIMPLE_SEGMENTS_KERNEL_SOURCE = _PRECISION_PREAMBLE + r"""
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

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup  # noqa: E402

request_nvrtc_warmup([
    ("is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES),
    ("is-simple-segments-fp64", _IS_SIMPLE_SEGMENTS_FP64, _IS_SIMPLE_SEGMENTS_KERNEL_NAMES),
])


# ---------------------------------------------------------------------------
# Helpers: shoelace signed area and segment intersection (CPU)
# ---------------------------------------------------------------------------

def _shoelace_area_2x(x: np.ndarray, y: np.ndarray, start: int, end: int) -> float:
    """Compute 2x signed area of a ring slice. Positive = CCW."""
    xs = x[start:end]
    ys = y[start:end]
    return float(np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]))


def _segments_cross(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> bool:
    """Return True if segments (a0-a1) and (b0-b1) properly cross.

    Uses the cross-product orientation test.  Endpoint-touching (t=0 or t=1)
    is *not* counted as a crossing -- only transversal intersection.
    """
    dx_a = ax1 - ax0
    dy_a = ay1 - ay0
    dx_b = bx1 - bx0
    dy_b = by1 - by0

    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-15:
        # Parallel / collinear -- skip (degenerate overlap not detected here)
        return False

    dx_ab = bx0 - ax0
    dy_ab = by0 - ay0

    t = (dx_ab * dy_b - dy_ab * dx_b) / denom
    u = (dx_ab * dy_a - dy_ab * dx_a) / denom

    # Strict interior crossing: both parameters in open interval (0, 1)
    eps = 1e-12
    return eps < t < (1.0 - eps) and eps < u < (1.0 - eps)


# ---------------------------------------------------------------------------
# Per-family validity helpers (CPU)
# ---------------------------------------------------------------------------

def _is_valid_points(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Points are always structurally valid."""
    return np.ones(buf.row_count, dtype=bool)


def _is_valid_linestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """LineString is valid if it has >= 2 coordinates (or is empty)."""
    result = np.ones(buf.row_count, dtype=bool)
    offsets = buf.geometry_offsets
    counts = offsets[1:buf.row_count + 1] - offsets[:buf.row_count]
    # Empty linestrings (0 coords) are valid; 1-coord linestrings are invalid
    invalid = (counts >= 1) & (counts < 2)
    result[invalid] = False
    return result


def _is_valid_polygon_ring(
    x: np.ndarray, y: np.ndarray, coord_start: int, coord_end: int,
    is_exterior: bool,
) -> bool:
    """Check structural validity of a single polygon ring.

    Returns False if:
    - Ring has fewer than 4 coordinates
    - Ring is not closed (first != last)
    - Ring has wrong orientation (exterior should be CCW, holes CW)
    """
    length = coord_end - coord_start
    if length < 4:
        return False
    # Ring closure check
    if x[coord_start] != x[coord_end - 1] or y[coord_start] != y[coord_end - 1]:
        return False
    # Orientation check via shoelace
    area2 = _shoelace_area_2x(x, y, coord_start, coord_end)
    if is_exterior and area2 < 0:
        return False  # exterior should be CCW (positive area)
    if not is_exterior and area2 > 0:
        return False  # holes should be CW (negative area)
    return True


def _is_valid_polygons(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Polygon validity: ring closure, minimum coords, orientation."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        ring_start_idx = int(geom_offsets[g])
        ring_end_idx = int(geom_offsets[g + 1])

        if ring_start_idx == ring_end_idx:
            # Empty polygon -- valid
            continue

        for ring_idx, r in enumerate(range(ring_start_idx, ring_end_idx)):
            coord_start = int(ring_offsets[r])
            coord_end = int(ring_offsets[r + 1])
            is_exterior = (ring_idx == 0)

            if not _is_valid_polygon_ring(x, y, coord_start, coord_end, is_exterior):
                result[g] = False
                break

    return result


def _is_valid_multipoints(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiPoints are always structurally valid."""
    return np.ones(buf.row_count, dtype=bool)


def _is_valid_multilinestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiLineString validity: each part must have >= 2 coordinates."""
    result = np.ones(buf.row_count, dtype=bool)
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for g in range(buf.row_count):
        part_start_idx = int(geom_offsets[g])
        part_end_idx = int(geom_offsets[g + 1])

        for p in range(part_start_idx, part_end_idx):
            coord_start = int(part_offsets[p])
            coord_end = int(part_offsets[p + 1])
            count = coord_end - coord_start
            if 1 <= count < 2:
                result[g] = False
                break

    return result


def _is_valid_multipolygons(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiPolygon validity: each polygon part checked for ring validity."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        poly_start = int(geom_offsets[g])
        poly_end = int(geom_offsets[g + 1])

        for p in range(poly_start, poly_end):
            ring_start_idx = int(part_offsets[p])
            ring_end_idx = int(part_offsets[p + 1])

            for ring_idx, r in enumerate(range(ring_start_idx, ring_end_idx)):
                coord_start = int(ring_offsets[r])
                coord_end = int(ring_offsets[r + 1])
                is_exterior = (ring_idx == 0)

                if not _is_valid_polygon_ring(x, y, coord_start, coord_end, is_exterior):
                    result[g] = False
                    break
            if not result[g]:
                break

    return result


_VALIDITY_DISPATCH = {
    GeometryFamily.POINT: _is_valid_points,
    GeometryFamily.LINESTRING: _is_valid_linestrings,
    GeometryFamily.POLYGON: _is_valid_polygons,
    GeometryFamily.MULTIPOINT: _is_valid_multipoints,
    GeometryFamily.MULTILINESTRING: _is_valid_multilinestrings,
    GeometryFamily.MULTIPOLYGON: _is_valid_multipolygons,
}


# ---------------------------------------------------------------------------
# Per-family simplicity helpers (CPU)
# ---------------------------------------------------------------------------

def _is_simple_points(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Points are always simple."""
    return np.ones(buf.row_count, dtype=bool)


def _is_simple_multipoints(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiPoints are always simple (duplicate detection not required here)."""
    return np.ones(buf.row_count, dtype=bool)


def _linestring_self_intersects(
    x: np.ndarray, y: np.ndarray, start: int, end: int,
    *, is_ring: bool = False,
) -> bool:
    """Brute-force O(n^2) self-intersection test for a linestring.

    Checks whether any two non-adjacent segments properly cross.

    Parameters
    ----------
    is_ring : bool
        If True, the first and last segments share an endpoint (closed ring)
        and are treated as adjacent.  For open linestrings this must be False
        so that the first-last segment pair is tested normally.
    """
    n_coords = end - start
    if n_coords < 4:
        # Fewer than 4 coords means at most 2 segments which are adjacent
        return False

    n_segs = n_coords - 1
    for i in range(n_segs):
        ax0 = float(x[start + i])
        ay0 = float(y[start + i])
        ax1 = float(x[start + i + 1])
        ay1 = float(y[start + i + 1])

        # Only check non-adjacent segments (skip i+1 which shares an endpoint)
        for j in range(i + 2, n_segs):
            # Skip the wrap-around adjacency for closed rings only
            if is_ring and i == 0 and j == n_segs - 1:
                continue
            bx0 = float(x[start + j])
            by0 = float(y[start + j])
            bx1 = float(x[start + j + 1])
            by1 = float(y[start + j + 1])

            if _segments_cross(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
                return True

    return False


def _is_simple_linestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """LineString is simple if no non-adjacent segments cross."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    offsets = buf.geometry_offsets

    for g in range(buf.row_count):
        start = int(offsets[g])
        end = int(offsets[g + 1])
        if start == end:
            # Empty -- simple
            continue
        if _linestring_self_intersects(x, y, start, end):
            result[g] = False

    return result


def _is_simple_polygons(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Polygon simplicity: check each ring for self-intersection."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        ring_start_idx = int(geom_offsets[g])
        ring_end_idx = int(geom_offsets[g + 1])

        for r in range(ring_start_idx, ring_end_idx):
            coord_start = int(ring_offsets[r])
            coord_end = int(ring_offsets[r + 1])
            if _linestring_self_intersects(x, y, coord_start, coord_end, is_ring=True):
                result[g] = False
                break

    return result


def _is_simple_multilinestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiLineString simplicity: each part checked individually."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for g in range(buf.row_count):
        part_start_idx = int(geom_offsets[g])
        part_end_idx = int(geom_offsets[g + 1])

        for p in range(part_start_idx, part_end_idx):
            coord_start = int(part_offsets[p])
            coord_end = int(part_offsets[p + 1])
            if _linestring_self_intersects(x, y, coord_start, coord_end):
                result[g] = False
                break

    return result


def _is_simple_multipolygons(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiPolygon simplicity: each ring in each polygon checked."""
    result = np.ones(buf.row_count, dtype=bool)
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        poly_start = int(geom_offsets[g])
        poly_end = int(geom_offsets[g + 1])

        for p in range(poly_start, poly_end):
            ring_start_idx = int(part_offsets[p])
            ring_end_idx = int(part_offsets[p + 1])

            for r in range(ring_start_idx, ring_end_idx):
                coord_start = int(ring_offsets[r])
                coord_end = int(ring_offsets[r + 1])
                if _linestring_self_intersects(x, y, coord_start, coord_end, is_ring=True):
                    result[g] = False
                    break
            if not result[g]:
                break

    return result


_SIMPLICITY_DISPATCH = {
    GeometryFamily.POINT: _is_simple_points,
    GeometryFamily.LINESTRING: _is_simple_linestrings,
    GeometryFamily.POLYGON: _is_simple_polygons,
    GeometryFamily.MULTIPOINT: _is_simple_multipoints,
    GeometryFamily.MULTILINESTRING: _is_simple_multilinestrings,
    GeometryFamily.MULTIPOLYGON: _is_simple_multipolygons,
}


# ---------------------------------------------------------------------------
# GPU helpers: build is_exterior mask and reduce per-ring to per-geometry
# ---------------------------------------------------------------------------

def _build_is_exterior_for_validity(d_buf, family):
    """Build a device int32 array marking exterior rings (1) vs interior (0).

    For Polygon:      geometry_offsets indexes into rings directly.
                      Ring 0 of each geometry is exterior.
    For MultiPolygon: part_offsets indexes into rings.
                      Ring 0 of each part is exterior.
    """
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        return cp.empty(0, dtype=cp.int32)

    if family is GeometryFamily.POLYGON:
        d_offsets = cp.asarray(d_buf.geometry_offsets)
    elif family is GeometryFamily.MULTIPOLYGON:
        d_offsets = cp.asarray(d_buf.part_offsets)
    else:
        raise ValueError(f"is_exterior only meaningful for polygon families, got {family}")

    d_is_exterior = cp.zeros(ring_count, dtype=cp.int32)
    d_exterior_indices = d_offsets[:-1]
    d_valid = d_exterior_indices < ring_count
    d_is_exterior[d_exterior_indices[d_valid]] = 1
    return d_is_exterior


def _reduce_ring_valid_to_geom(d_ring_valid, d_geometry_offsets, geom_count):
    """Reduce per-ring validity to per-geometry using vectorized cumsum.

    A geometry is valid iff ALL its rings are valid.  Uses the cumsum trick:
    cumsum(1 - ring_valid) gives running count of invalid rings, then
    per-geometry invalid count is the difference at geometry boundaries.

    Returns a host np.ndarray of bool (length geom_count).
    """
    ring_count = int(d_ring_valid.shape[0])
    if ring_count == 0:
        return np.ones(geom_count, dtype=bool)

    d_starts = d_geometry_offsets[:-1]
    d_ends = d_geometry_offsets[1:]

    # Cumsum of invalid flags
    d_invalid = 1 - d_ring_valid  # 1 where invalid
    d_cumsum = cp.cumsum(d_invalid)

    d_empty = d_starts == d_ends
    d_result = cp.ones(geom_count, dtype=cp.bool_)

    d_ne_indices = cp.flatnonzero(~d_empty)
    if d_ne_indices.size > 0:
        d_end_indices = d_ends[d_ne_indices] - 1
        d_end_sums = d_cumsum[d_end_indices]

        d_starts_ne = d_starts[d_ne_indices]
        d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
        d_has_prev = d_starts_ne > 0
        d_hp_indices = cp.flatnonzero(d_has_prev)
        if d_hp_indices.size > 0:
            d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

        d_invalid_counts = d_end_sums - d_start_sums
        d_result[d_ne_indices] = (d_invalid_counts == 0)

    return cp.asnumpy(d_result)


def _reduce_span_simple_to_geom(d_span_simple, d_geometry_offsets, geom_count):
    """Reduce per-span simplicity to per-geometry using vectorized cumsum.

    A geometry is simple iff ALL its spans (rings/parts) are simple.
    Same cumsum trick as _reduce_ring_valid_to_geom.

    Returns a host np.ndarray of bool (length geom_count).
    """
    return _reduce_ring_valid_to_geom(d_span_simple, d_geometry_offsets, geom_count)


# ---------------------------------------------------------------------------
# GPU dispatch: is_valid per-family
# ---------------------------------------------------------------------------

def _is_valid_gpu_points(d_buf, result, global_rows):
    """Points are always valid on GPU -- write True directly."""
    result[global_rows] = True


def _is_valid_gpu_multipoints(d_buf, result, global_rows):
    """MultiPoints are always valid on GPU -- write True directly."""
    result[global_rows] = True


def _is_valid_gpu_linestrings(d_buf, family_rows, result, global_rows):
    """LineString validity via CuPy offset arithmetic (Tier 2)."""
    d_family_rows = cp.asarray(family_rows)
    d_counts = d_buf.geometry_offsets[d_family_rows + 1] - d_buf.geometry_offsets[d_family_rows]
    # Valid if empty (0 coords) or >= 2 coords
    d_valid = (d_counts == 0) | (d_counts >= 2)
    result[global_rows] = cp.asnumpy(d_valid)


def _is_valid_gpu_multilinestrings(d_buf, family_rows, result, global_rows):
    """MultiLineString validity: each part must have >= 2 coords.

    Uses CuPy vectorized per-part coord count, then reduces per-geometry
    with the cumsum trick.
    """
    geom_count = int(family_rows.shape[0])
    d_family_rows = cp.asarray(family_rows)

    # Per-part coord counts for ALL parts in the buffer
    d_part_counts = d_buf.part_offsets[1:] - d_buf.part_offsets[:-1]

    # A part is invalid if it has exactly 1 coordinate (>= 1 and < 2)
    d_part_valid = (d_part_counts == 0) | (d_part_counts >= 2)
    d_part_valid_int = d_part_valid.astype(cp.int32)

    # Get the geometry offset slice for the requested family rows
    d_geom_offsets_for_rows = d_buf.geometry_offsets[d_family_rows]
    d_geom_offsets_for_rows_end = d_buf.geometry_offsets[d_family_rows + 1]

    # Reduce per-part to per-geometry: all parts must be valid
    d_starts = d_geom_offsets_for_rows
    d_ends = d_geom_offsets_for_rows_end

    total_parts = int(d_buf.part_offsets.shape[0]) - 1
    if total_parts == 0:
        result[global_rows] = True
        return

    d_invalid = (1 - d_part_valid_int)
    d_cumsum = cp.cumsum(d_invalid)

    d_empty = d_starts == d_ends
    d_result = cp.ones(geom_count, dtype=cp.bool_)

    d_ne_indices = cp.flatnonzero(~d_empty)
    if d_ne_indices.size > 0:
        d_end_indices = d_ends[d_ne_indices] - 1
        d_end_sums = d_cumsum[d_end_indices]

        d_starts_ne = d_starts[d_ne_indices]
        d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
        d_has_prev = d_starts_ne > 0
        d_hp_indices = cp.flatnonzero(d_has_prev)
        if d_hp_indices.size > 0:
            d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

        d_invalid_counts = d_end_sums - d_start_sums
        d_result[d_ne_indices] = (d_invalid_counts == 0)

    result[global_rows] = cp.asnumpy(d_result)


def _is_valid_gpu_polygons(d_buf, family_rows, result, global_rows):
    """Polygon validity via NVRTC is_valid_rings kernel + per-ring reduction."""
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    kernels = compile_kernel_group(
        "is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES,
    )
    kernel = kernels["is_valid_rings"]

    d_is_exterior = _build_is_exterior_for_validity(d_buf, GeometryFamily.POLYGON)
    d_ring_valid = runtime.allocate((ring_count,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_buf.x), ptr(d_buf.y), ptr(d_buf.ring_offsets),
             ptr(d_is_exterior), ptr(d_ring_valid), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, ring_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)

        # Reduce per-ring results to per-geometry (all rings must be valid)
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_starts = d_buf.geometry_offsets[d_family_rows]
        d_ends = d_buf.geometry_offsets[d_family_rows + 1]

        # Convert d_ring_valid to CuPy for cumsum
        d_ring_valid_cp = cp.asarray(d_ring_valid)
        d_invalid = 1 - d_ring_valid_cp
        d_cumsum = cp.cumsum(d_invalid)

        d_empty = d_starts == d_ends
        d_result = cp.ones(geom_count, dtype=cp.bool_)

        d_ne_indices = cp.flatnonzero(~d_empty)
        if d_ne_indices.size > 0:
            d_end_indices = d_ends[d_ne_indices] - 1
            d_end_sums = d_cumsum[d_end_indices]

            d_starts_ne = d_starts[d_ne_indices]
            d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
            d_has_prev = d_starts_ne > 0
            d_hp_indices = cp.flatnonzero(d_has_prev)
            if d_hp_indices.size > 0:
                d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

            d_invalid_counts = d_end_sums - d_start_sums
            d_result[d_ne_indices] = (d_invalid_counts == 0)

        result[global_rows] = cp.asnumpy(d_result)
    finally:
        runtime.free(d_ring_valid)


def _is_valid_gpu_multipolygons(d_buf, family_rows, result, global_rows):
    """MultiPolygon validity via NVRTC is_valid_rings kernel + reduction.

    MultiPolygon uses part_offsets to reach rings.  The exterior ring of
    each part is ring 0 within that part (indexed via part_offsets).
    """
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    kernels = compile_kernel_group(
        "is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES,
    )
    kernel = kernels["is_valid_rings"]

    d_is_exterior = _build_is_exterior_for_validity(d_buf, GeometryFamily.MULTIPOLYGON)
    d_ring_valid = runtime.allocate((ring_count,), np.int32, zero=True)
    try:
        ptr = runtime.pointer
        params = (
            (ptr(d_buf.x), ptr(d_buf.y), ptr(d_buf.ring_offsets),
             ptr(d_is_exterior), ptr(d_ring_valid), ring_count),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
             KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        )
        grid, block = runtime.launch_config(kernel, ring_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)

        # For MultiPolygon: geometry_offsets -> parts, part_offsets -> rings.
        # We need to reduce all rings across all parts of each geometry.
        # Build ring-level spans per geometry by chaining:
        #   ring_start = part_offsets[geometry_offsets[g]]
        #   ring_end = part_offsets[geometry_offsets[g+1]]
        # (because part_offsets are contiguous ring indices for all parts)
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_geom_offsets = d_buf.geometry_offsets[d_family_rows]
        d_geom_offsets_end = d_buf.geometry_offsets[d_family_rows + 1]

        # Map geometry part range to ring range via part_offsets
        d_starts = d_buf.part_offsets[d_geom_offsets]
        d_ends = d_buf.part_offsets[d_geom_offsets_end]

        # Convert d_ring_valid to CuPy for cumsum
        d_ring_valid_cp = cp.asarray(d_ring_valid)
        d_invalid = 1 - d_ring_valid_cp
        d_cumsum = cp.cumsum(d_invalid)

        d_empty = d_starts == d_ends
        d_result = cp.ones(geom_count, dtype=cp.bool_)

        d_ne_indices = cp.flatnonzero(~d_empty)
        if d_ne_indices.size > 0:
            d_end_indices = d_ends[d_ne_indices] - 1
            d_end_sums = d_cumsum[d_end_indices]

            d_starts_ne = d_starts[d_ne_indices]
            d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
            d_has_prev = d_starts_ne > 0
            d_hp_indices = cp.flatnonzero(d_has_prev)
            if d_hp_indices.size > 0:
                d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

            d_invalid_counts = d_end_sums - d_start_sums
            d_result[d_ne_indices] = (d_invalid_counts == 0)

        result[global_rows] = cp.asnumpy(d_result)
    finally:
        runtime.free(d_ring_valid)


# ---------------------------------------------------------------------------
# GPU dispatch: is_simple per-family
# ---------------------------------------------------------------------------

def _is_simple_gpu_points(d_buf, result, global_rows):
    """Points are always simple on GPU."""
    result[global_rows] = True


def _is_simple_gpu_multipoints(d_buf, result, global_rows):
    """MultiPoints are always simple on GPU."""
    result[global_rows] = True


def _launch_is_simple_kernel(runtime, d_buf_x, d_buf_y, d_span_offsets, span_count, is_ring):
    """Launch the is_simple_segments kernel and return the device result array."""
    kernels = compile_kernel_group(
        "is-simple-segments-fp64",
        _IS_SIMPLE_SEGMENTS_FP64,
        _IS_SIMPLE_SEGMENTS_KERNEL_NAMES,
    )
    kernel = kernels["is_simple_segments"]

    d_result = runtime.allocate((span_count,), np.int32, zero=True)
    ptr = runtime.pointer
    params = (
        (ptr(d_buf_x), ptr(d_buf_y), ptr(d_span_offsets),
         ptr(d_result), int(is_ring), span_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
    )
    # 1 block per span: grid = (span_count,), block size from __launch_bounds__ = 256
    block_size = 256
    grid = (span_count,)
    block = (block_size,)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_result


def _is_simple_gpu_linestrings(d_buf, family_rows, result, global_rows):
    """LineString simplicity: launch is_simple_segments with geometry_offsets."""
    runtime = get_cuda_runtime()

    # For linestrings, span_offsets = geometry_offsets, is_ring = 0
    # We launch over ALL spans in the buffer, then extract results for the
    # requested family rows.
    total_spans = int(d_buf.geometry_offsets.shape[0]) - 1
    if total_spans == 0:
        result[global_rows] = True
        return

    d_span_result = _launch_is_simple_kernel(
        runtime, d_buf.x, d_buf.y, d_buf.geometry_offsets, total_spans, is_ring=0,
    )
    try:
        d_family_rows = cp.asarray(family_rows)
        d_span_result_cp = cp.asarray(d_span_result)
        h_result = cp.asnumpy(d_span_result_cp[d_family_rows]).astype(bool)
        result[global_rows] = h_result
    finally:
        runtime.free(d_span_result)


def _is_simple_gpu_polygons(d_buf, family_rows, result, global_rows):
    """Polygon simplicity: launch is_simple_segments on ring_offsets.

    Each ring is a span.  Reduce per-ring to per-geometry (all rings simple).
    """
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    # Launch over ALL rings with is_ring=1 (closed ring adjacency)
    d_span_result = _launch_is_simple_kernel(
        runtime, d_buf.x, d_buf.y, d_buf.ring_offsets, ring_count, is_ring=1,
    )
    try:
        # Reduce per-ring to per-geometry
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_starts = d_buf.geometry_offsets[d_family_rows]
        d_ends = d_buf.geometry_offsets[d_family_rows + 1]

        d_span_cp = cp.asarray(d_span_result)
        d_invalid = 1 - d_span_cp
        d_cumsum = cp.cumsum(d_invalid)

        d_empty = d_starts == d_ends
        d_result = cp.ones(geom_count, dtype=cp.bool_)

        d_ne_indices = cp.flatnonzero(~d_empty)
        if d_ne_indices.size > 0:
            d_end_indices = d_ends[d_ne_indices] - 1
            d_end_sums = d_cumsum[d_end_indices]

            d_starts_ne = d_starts[d_ne_indices]
            d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
            d_has_prev = d_starts_ne > 0
            d_hp_indices = cp.flatnonzero(d_has_prev)
            if d_hp_indices.size > 0:
                d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

            d_invalid_counts = d_end_sums - d_start_sums
            d_result[d_ne_indices] = (d_invalid_counts == 0)

        result[global_rows] = cp.asnumpy(d_result)
    finally:
        runtime.free(d_span_result)


def _is_simple_gpu_multilinestrings(d_buf, family_rows, result, global_rows):
    """MultiLineString simplicity: each part checked (open linestring, is_ring=0).

    Span offsets = part_offsets.  Reduce per-part to per-geometry.
    """
    runtime = get_cuda_runtime()
    total_parts = int(d_buf.part_offsets.shape[0]) - 1
    if total_parts == 0:
        result[global_rows] = True
        return

    d_span_result = _launch_is_simple_kernel(
        runtime, d_buf.x, d_buf.y, d_buf.part_offsets, total_parts, is_ring=0,
    )
    try:
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_starts = d_buf.geometry_offsets[d_family_rows]
        d_ends = d_buf.geometry_offsets[d_family_rows + 1]

        d_span_cp = cp.asarray(d_span_result)
        d_invalid = 1 - d_span_cp
        d_cumsum = cp.cumsum(d_invalid)

        d_empty = d_starts == d_ends
        d_result = cp.ones(geom_count, dtype=cp.bool_)

        d_ne_indices = cp.flatnonzero(~d_empty)
        if d_ne_indices.size > 0:
            d_end_indices = d_ends[d_ne_indices] - 1
            d_end_sums = d_cumsum[d_end_indices]

            d_starts_ne = d_starts[d_ne_indices]
            d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
            d_has_prev = d_starts_ne > 0
            d_hp_indices = cp.flatnonzero(d_has_prev)
            if d_hp_indices.size > 0:
                d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

            d_invalid_counts = d_end_sums - d_start_sums
            d_result[d_ne_indices] = (d_invalid_counts == 0)

        result[global_rows] = cp.asnumpy(d_result)
    finally:
        runtime.free(d_span_result)


def _is_simple_gpu_multipolygons(d_buf, family_rows, result, global_rows):
    """MultiPolygon simplicity: each ring in each polygon checked (is_ring=1).

    Span offsets = ring_offsets.  Reduce per-ring to per-geometry via
    geometry_offsets -> part_offsets -> ring range.
    """
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    d_span_result = _launch_is_simple_kernel(
        runtime, d_buf.x, d_buf.y, d_buf.ring_offsets, ring_count, is_ring=1,
    )
    try:
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_geom_offsets = d_buf.geometry_offsets[d_family_rows]
        d_geom_offsets_end = d_buf.geometry_offsets[d_family_rows + 1]

        # Map geometry part range to ring range
        d_starts = d_buf.part_offsets[d_geom_offsets]
        d_ends = d_buf.part_offsets[d_geom_offsets_end]

        d_span_cp = cp.asarray(d_span_result)
        d_invalid = 1 - d_span_cp
        d_cumsum = cp.cumsum(d_invalid)

        d_empty = d_starts == d_ends
        d_result = cp.ones(geom_count, dtype=cp.bool_)

        d_ne_indices = cp.flatnonzero(~d_empty)
        if d_ne_indices.size > 0:
            d_end_indices = d_ends[d_ne_indices] - 1
            d_end_sums = d_cumsum[d_end_indices]

            d_starts_ne = d_starts[d_ne_indices]
            d_start_sums = cp.zeros(d_ne_indices.size, dtype=cp.int32)
            d_has_prev = d_starts_ne > 0
            d_hp_indices = cp.flatnonzero(d_has_prev)
            if d_hp_indices.size > 0:
                d_start_sums[d_hp_indices] = d_cumsum[d_starts_ne[d_hp_indices] - 1]

            d_invalid_counts = d_end_sums - d_start_sums
            d_result[d_ne_indices] = (d_invalid_counts == 0)

        result[global_rows] = cp.asnumpy(d_result)
    finally:
        runtime.free(d_span_result)


# ---------------------------------------------------------------------------
# Kernel registry (ADR-0033 observability)
# ---------------------------------------------------------------------------

@register_kernel_variant(
    "is_valid",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "predicate", "validity"),
)
def _is_valid_gpu(owned):
    """Registry marker — actual dispatch is inline in is_valid_owned."""
    raise NotImplementedError


@register_kernel_variant(
    "is_simple",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "linestring", "multilinestring", "polygon", "multipolygon",
    ),
    supports_mixed=True,
    tags=("cuda-python", "predicate", "simplicity"),
)
def _is_simple_gpu(owned):
    """Registry marker — actual dispatch is inline in is_simple_owned."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Check structural validity of each geometry in an OwnedGeometryArray.

    Checks performed per family:
    - Ring closure (first coord == last coord for polygon rings)
    - Minimum coordinate counts (LineString >= 2, Polygon ring >= 4)
    - Orientation (exterior CCW, holes CW via shoelace signed area)

    Null and empty geometries return True, matching Shapely semantics.

    Parameters
    ----------
    owned : OwnedGeometryArray
        The geometry array to validate.
    dispatch_mode : ExecutionMode or str
        GPU/CPU/AUTO execution mode.

    Returns
    -------
    np.ndarray of bool
        Per-geometry validity flags.
    """
    row_count = owned.row_count
    if row_count == 0:
        return np.array([], dtype=bool)

    selection = plan_dispatch_selection(
        kernel_name="is_valid",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    use_gpu = (
        selection.selected is ExecutionMode.GPU
        and cp is not None
        and owned.device_state is not None
    )

    result = np.ones(row_count, dtype=bool)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path ---
        if use_gpu and family in owned.device_state.families:
            d_buf = owned.device_state.families[family]
            try:
                if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
                    result[global_rows] = True

                elif family is GeometryFamily.LINESTRING:
                    _is_valid_gpu_linestrings(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.MULTILINESTRING:
                    _is_valid_gpu_multilinestrings(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.POLYGON:
                    _is_valid_gpu_polygons(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.MULTIPOLYGON:
                    _is_valid_gpu_multipolygons(d_buf, family_rows, result, global_rows)

                continue  # skip CPU fallback
            except Exception:
                pass  # fall through to CPU

        # --- CPU fallback ---
        family_result = _VALIDITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    # Null geometries are valid (Shapely convention)
    # Already True by default; no action needed for nulls.
    return result


def is_simple_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> np.ndarray:
    """Check geometric simplicity of each geometry in an OwnedGeometryArray.

    A geometry is simple if it has no self-intersections (no two
    non-adjacent segments properly cross).

    - Points / MultiPoints: always simple.
    - LineStrings: brute-force O(n^2) segment intersection test.
    - Polygons: each ring checked for self-intersection.

    Null and empty geometries return True, matching Shapely semantics.

    Parameters
    ----------
    owned : OwnedGeometryArray
        The geometry array to check.
    dispatch_mode : ExecutionMode or str
        GPU/CPU/AUTO execution mode.

    Returns
    -------
    np.ndarray of bool
        Per-geometry simplicity flags.
    """
    row_count = owned.row_count
    if row_count == 0:
        return np.array([], dtype=bool)

    selection = plan_dispatch_selection(
        kernel_name="is_simple",
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=dispatch_mode,
    )

    use_gpu = (
        selection.selected is ExecutionMode.GPU
        and cp is not None
        and owned.device_state is not None
    )

    result = np.ones(row_count, dtype=bool)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets

    for family, buf in owned.families.items():
        if buf.row_count == 0:
            continue
        tag = FAMILY_TAGS[family]
        mask = tags == tag
        if not np.any(mask):
            continue

        global_rows = np.flatnonzero(mask)
        family_rows = family_row_offsets[global_rows]

        # --- Device path ---
        if use_gpu and family in owned.device_state.families:
            d_buf = owned.device_state.families[family]
            try:
                if family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT):
                    result[global_rows] = True

                elif family is GeometryFamily.LINESTRING:
                    _is_simple_gpu_linestrings(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.POLYGON:
                    _is_simple_gpu_polygons(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.MULTILINESTRING:
                    _is_simple_gpu_multilinestrings(d_buf, family_rows, result, global_rows)

                elif family is GeometryFamily.MULTIPOLYGON:
                    _is_simple_gpu_multipolygons(d_buf, family_rows, result, global_rows)

                continue  # skip CPU fallback
            except Exception:
                pass  # fall through to CPU

        # --- CPU fallback ---
        family_result = _SIMPLICITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    # Null geometries are simple (Shapely convention)
    # Already True by default; no action needed for nulls.
    return result
