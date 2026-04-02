"""GPU-accelerated geometry validity and simplicity checks.

OGC validity checks (is_valid) and geometric simplicity checks
(is_simple) computed directly from OwnedGeometryArray coordinate buffers
without Shapely materialization.

is_valid checks (matching GEOS/Shapely semantics):
- Ring closure (first coord == last coord for polygon rings)
- Minimum coordinate counts (LineString >= 2, Polygon ring >= 4)
- Ring self-intersection (no non-adjacent segments cross within a ring)
- Hole-in-shell containment (each hole's first vertex inside exterior ring)
- Ring-pair interaction: no proper crossing, collinear overlap, or
  multi-touch (2+ distinct contact points) between distinct rings
- Null/empty geometries return True (matching Shapely behavior)
Note: orientation (CCW/CW) is NOT checked by is_valid — GEOS does not
enforce winding order as a validity rule.  Use orient.py for that.

is_simple checks:
- Points/MultiPoints: always True
- LineStrings: no self-intersection (brute-force O(n^2) segment test)
- Null/empty geometries return True

ADR-0033: Tier 1 NVRTC for is_valid_rings, is_simple_segments,
          holes_in_shell, and ring_pair_interaction kernels.
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

from vibespatial.constructive.validity_kernels import (
    _HOLES_IN_SHELL_FP64,
    _HOLES_IN_SHELL_KERNEL_NAMES,
    _IS_SIMPLE_SEGMENTS_FP64,
    _IS_SIMPLE_SEGMENTS_KERNEL_NAMES,
    _IS_VALID_RINGS_FP64,
    _IS_VALID_RINGS_KERNEL_NAMES,
    _RING_PAIR_INTERACTION_FP64,
    _RING_PAIR_INTERACTION_KERNEL_NAMES,
)
from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)

# Background precompilation (ADR-0034)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import KernelClass, PrecisionMode

request_nvrtc_warmup([
    ("is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES),
    ("is-simple-segments-fp64", _IS_SIMPLE_SEGMENTS_FP64, _IS_SIMPLE_SEGMENTS_KERNEL_NAMES),
    ("holes-in-shell-fp64", _HOLES_IN_SHELL_FP64, _HOLES_IN_SHELL_KERNEL_NAMES),
    ("ring-pair-interaction-fp64", _RING_PAIR_INTERACTION_FP64, _RING_PAIR_INTERACTION_KERNEL_NAMES),
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


def _point_on_segment_cpu(
    px: float, py: float,
    ax: float, ay: float, bx: float, by: float,
) -> bool:
    """Return True if point (px,py) lies on segment (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    cross = (px - ax) * dy - (py - ay) * dx
    scale = abs(dx) + abs(dy) + 1.0
    if abs(cross) > 1e-12 * scale:
        return False
    minx = min(ax, bx)
    maxx = max(ax, bx)
    miny = min(ay, by)
    maxy = max(ay, by)
    return (
        px >= minx - 1e-12 and px <= maxx + 1e-12
        and py >= miny - 1e-12 and py <= maxy + 1e-12
    )


def _point_in_ring_even_odd(
    px: float, py: float,
    x: np.ndarray, y: np.ndarray,
    coord_start: int, coord_end: int,
) -> bool:
    """Even-odd ray-cast: is (px,py) inside the ring [coord_start:coord_end]?

    Returns True if the point is inside the ring OR on its boundary.
    Used for OGC validity hole-in-shell containment check.
    """
    if (coord_end - coord_start) < 4:
        return False
    inside = False
    for c in range(coord_start + 1, coord_end):
        ax = float(x[c - 1])
        ay = float(y[c - 1])
        bx = float(x[c])
        by = float(y[c])

        # Boundary: point on edge counts as inside (OGC validity)
        if _point_on_segment_cpu(px, py, ax, ay, bx, by):
            return True

        # Even-odd crossing test
        if (ay > py) != (by > py):
            if px < ((bx - ax) * (py - ay) / (by - ay)) + ax:
                inside = not inside
    return inside


# ---------------------------------------------------------------------------
# Per-family validity helpers (CPU)
# ---------------------------------------------------------------------------

def _is_valid_points(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Points are always structurally valid."""
    return np.ones(buf.row_count, dtype=bool)


def _is_valid_linestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """LineString is valid if it has >= 2 coordinates (or is empty)."""
    result = np.ones(buf.row_count, dtype=bool)
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
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
    """Check OGC validity of a single polygon ring.

    Returns False if:
    - Ring has fewer than 4 coordinates
    - Ring is not closed (first != last)
    - Ring has self-intersection (non-adjacent segments cross)

    Note: orientation (CCW exterior / CW hole) is NOT checked.
    GEOS/Shapely is_valid does not enforce winding order.
    """
    length = coord_end - coord_start
    if length < 4:
        return False
    # Guard against corrupt offsets exceeding coordinate array bounds
    if coord_end > len(x):
        return False
    # Ring closure check
    if x[coord_start] != x[coord_end - 1] or y[coord_start] != y[coord_end - 1]:
        return False
    # Self-intersection check: ring must be simple (no non-adjacent crossings)
    if _linestring_self_intersects(x, y, coord_start, coord_end, is_ring=True):
        return False
    return True


def _orient2d_exact(ax, ay, bx, by, cx, cy):
    """Exact orient2d using fractions.Fraction for arbitrary-precision arithmetic (CPU).

    Returns +1, 0, or -1 for the sign of det = (bx-ax)*(cy-ay) - (by-ay)*(cx-ax).
    Uses Fraction to avoid FMA dependency (math.fma requires Python 3.13+).
    """
    from fractions import Fraction

    det = (Fraction(bx) - Fraction(ax)) * (Fraction(cy) - Fraction(ay)) - (
        Fraction(by) - Fraction(ay)
    ) * (Fraction(cx) - Fraction(ax))
    return int(det > 0) - int(det < 0)


def _point_on_seg_collinear_cpu(px, py, ax, ay, bx, by):
    """Check if point (px,py) is on segment (ax,ay)-(bx,by), assuming collinear."""
    minx = min(ax, bx)
    maxx = max(ax, bx)
    miny = min(ay, by)
    maxy = max(ay, by)
    return px >= minx and px <= maxx and py >= miny and py <= maxy


def _check_ring_pair_interaction_cpu(
    x: np.ndarray, y: np.ndarray,
    ring_offsets: np.ndarray,
    ring_start_idx: int, ring_end_idx: int,
) -> bool:
    """Check inter-ring interactions for a single polygon's rings (CPU).

    Returns True if the polygon passes all inter-ring OGC checks:
      - No proper crossings between segments of distinct rings
      - No collinear overlap (shared segments)
      - No multi-touch (2+ distinct contact points per ring pair)

    Returns False if any violation is found.
    """
    n_rings = ring_end_idx - ring_start_idx
    if n_rings < 2:
        return True

    # For each pair of rings
    for ri_idx in range(n_rings):
        ring_i = ring_start_idx + ri_idx
        ci_start = int(ring_offsets[ring_i])
        ci_end = int(ring_offsets[ring_i + 1])
        n_segs_i = ci_end - ci_start - 1
        if n_segs_i < 1:
            continue

        for rj_idx in range(ri_idx + 1, n_rings):
            ring_j = ring_start_idx + rj_idx
            cj_start = int(ring_offsets[ring_j])
            cj_end = int(ring_offsets[ring_j + 1])
            n_segs_j = cj_end - cj_start - 1
            if n_segs_j < 1:
                continue

            # Track touch points for this ring pair
            touch_points: list[tuple[float, float]] = []

            for si in range(n_segs_i):
                ax = float(x[ci_start + si])
                ay = float(y[ci_start + si])
                bx = float(x[ci_start + si + 1])
                by = float(y[ci_start + si + 1])

                for sj in range(n_segs_j):
                    cx = float(x[cj_start + sj])
                    cy = float(y[cj_start + sj])
                    dx = float(x[cj_start + sj + 1])
                    dy = float(y[cj_start + sj + 1])

                    # MBR early reject
                    if (max(ax, bx) < min(cx, dx) or max(cx, dx) < min(ax, bx) or
                            max(ay, by) < min(cy, dy) or max(cy, dy) < min(ay, by)):
                        continue

                    o1 = _orient2d_exact(ax, ay, bx, by, cx, cy)
                    o2 = _orient2d_exact(ax, ay, bx, by, dx, dy)
                    o3 = _orient2d_exact(cx, cy, dx, dy, ax, ay)
                    o4 = _orient2d_exact(cx, cy, dx, dy, bx, by)

                    # Case 1: Proper crossing
                    if o1 != 0 and o2 != 0 and o1 != o2 and o3 != 0 and o4 != 0 and o3 != o4:
                        return False

                    # Case 2: Collinear
                    if o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0:
                        a_on = _point_on_seg_collinear_cpu(ax, ay, cx, cy, dx, dy)
                        b_on = _point_on_seg_collinear_cpu(bx, by, cx, cy, dx, dy)
                        c_on = _point_on_seg_collinear_cpu(cx, cy, ax, ay, bx, by)
                        d_on = _point_on_seg_collinear_cpu(dx, dy, ax, ay, bx, by)
                        containments = int(a_on) + int(b_on) + int(c_on) + int(d_on)

                        if containments >= 3:
                            return False  # overlap

                        if containments == 2:
                            shared = 0
                            if ax == cx and ay == cy:
                                shared += 1
                            if ax == dx and ay == dy:
                                shared += 1
                            if bx == cx and by == cy:
                                shared += 1
                            if bx == dx and by == dy:
                                shared += 1
                            if shared >= 1:
                                # Shared endpoint touch
                                if ax == cx and ay == cy:
                                    tp = (ax, ay)
                                elif ax == dx and ay == dy:
                                    tp = (ax, ay)
                                elif bx == cx and by == cy:
                                    tp = (bx, by)
                                else:
                                    tp = (bx, by)
                                if tp not in touch_points:
                                    touch_points.append(tp)
                            else:
                                return False  # overlap
                            continue

                        if containments == 1:
                            if a_on:
                                tp = (ax, ay)
                            elif b_on:
                                tp = (bx, by)
                            elif c_on:
                                tp = (cx, cy)
                            else:
                                tp = (dx, dy)
                            if tp not in touch_points:
                                touch_points.append(tp)
                        continue

                    # Case 3: T-intersection
                    if o1 == 0 and _point_on_seg_collinear_cpu(cx, cy, ax, ay, bx, by):
                        tp = (cx, cy)
                        if tp not in touch_points:
                            touch_points.append(tp)
                    elif o2 == 0 and _point_on_seg_collinear_cpu(dx, dy, ax, ay, bx, by):
                        tp = (dx, dy)
                        if tp not in touch_points:
                            touch_points.append(tp)
                    elif o3 == 0 and _point_on_seg_collinear_cpu(ax, ay, cx, cy, dx, dy):
                        tp = (ax, ay)
                        if tp not in touch_points:
                            touch_points.append(tp)
                    elif o4 == 0 and _point_on_seg_collinear_cpu(bx, by, cx, cy, dx, dy):
                        tp = (bx, by)
                        if tp not in touch_points:
                            touch_points.append(tp)

            # Multi-touch check: 2+ distinct points for this ring pair
            if len(touch_points) >= 2:
                return False

    return True


def _is_valid_polygons(buf: FamilyGeometryBuffer) -> np.ndarray:
    """Polygon validity: ring closure, min coords, simplicity, hole containment, ring interaction."""
    result = np.ones(buf.row_count, dtype=bool)
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            # Empty polygon -- valid
            continue

        ring_start_idx = int(geom_offsets[g])
        ring_end_idx = int(geom_offsets[g + 1])

        if ring_start_idx == ring_end_idx:
            # No rings -- valid
            continue

        # Step 1: Check all rings for structural validity + simplicity
        rings_ok = True
        for ring_idx, r in enumerate(range(ring_start_idx, ring_end_idx)):
            coord_start = int(ring_offsets[r])
            coord_end = int(ring_offsets[r + 1])
            is_exterior = (ring_idx == 0)

            if not _is_valid_polygon_ring(x, y, coord_start, coord_end, is_exterior):
                result[g] = False
                rings_ok = False
                break

        if not rings_ok:
            continue

        # Step 2: Hole-in-shell containment
        # Exterior ring is ring_start_idx; holes are ring_start_idx+1 .. ring_end_idx-1
        holes_ok = True
        ext_coord_start = int(ring_offsets[ring_start_idx])
        ext_coord_end = int(ring_offsets[ring_start_idx + 1])
        for r in range(ring_start_idx + 1, ring_end_idx):
            hole_coord_start = int(ring_offsets[r])
            hx = float(x[hole_coord_start])
            hy = float(y[hole_coord_start])
            if not _point_in_ring_even_odd(hx, hy, x, y, ext_coord_start, ext_coord_end):
                result[g] = False
                holes_ok = False
                break

        if not holes_ok:
            continue

        # Step 3: Ring-pair interaction (crossing, overlap, multi-touch)
        if not _check_ring_pair_interaction_cpu(
            x, y, ring_offsets, ring_start_idx, ring_end_idx,
        ):
            result[g] = False

    return result


def _is_valid_multipoints(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiPoints are always structurally valid."""
    return np.ones(buf.row_count, dtype=bool)


def _is_valid_multilinestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """MultiLineString validity: each part must have >= 2 coordinates."""
    result = np.ones(buf.row_count, dtype=bool)
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            continue

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
    """MultiPolygon validity: ring validity, simplicity, hole containment, ring interaction."""
    result = np.ones(buf.row_count, dtype=bool)
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            continue

        poly_start = int(geom_offsets[g])
        poly_end = int(geom_offsets[g + 1])

        for p in range(poly_start, poly_end):
            ring_start_idx = int(part_offsets[p])
            ring_end_idx = int(part_offsets[p + 1])

            # Step 1: Check all rings for structural validity + simplicity
            rings_ok = True
            for ring_idx, r in enumerate(range(ring_start_idx, ring_end_idx)):
                coord_start = int(ring_offsets[r])
                coord_end = int(ring_offsets[r + 1])
                is_exterior = (ring_idx == 0)

                if not _is_valid_polygon_ring(x, y, coord_start, coord_end, is_exterior):
                    result[g] = False
                    rings_ok = False
                    break

            if not rings_ok:
                break

            # Step 2: Hole-in-shell containment for this polygon part
            holes_ok = True
            ext_coord_start = int(ring_offsets[ring_start_idx])
            ext_coord_end = int(ring_offsets[ring_start_idx + 1])
            for r in range(ring_start_idx + 1, ring_end_idx):
                hole_coord_start = int(ring_offsets[r])
                hx = float(x[hole_coord_start])
                hy = float(y[hole_coord_start])
                if not _point_in_ring_even_odd(
                    hx, hy, x, y, ext_coord_start, ext_coord_end,
                ):
                    result[g] = False
                    holes_ok = False
                    break

            if not holes_ok:
                break

            # Step 3: Ring-pair interaction (crossing, overlap, multi-touch)
            if not _check_ring_pair_interaction_cpu(
                x, y, ring_offsets, ring_start_idx, ring_end_idx,
            ):
                result[g] = False
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


def _endpoints_coincide(
    ax0: float, ay0: float, ax1: float, ay1: float,
    bx0: float, by0: float, bx1: float, by1: float,
) -> bool:
    """Return True if any endpoint of segment A equals any endpoint of segment B."""
    return (
        (ax0 == bx0 and ay0 == by0) or (ax0 == bx1 and ay0 == by1)
        or (ax1 == bx0 and ay1 == by0) or (ax1 == bx1 and ay1 == by1)
    )


def _linestring_self_intersects(
    x: np.ndarray, y: np.ndarray, start: int, end: int,
    *, is_ring: bool = False,
) -> bool:
    """Brute-force O(n^2) self-intersection test for a linestring.

    Checks whether any two non-adjacent segments properly cross OR share
    an endpoint (vertex coincidence at non-adjacent positions, e.g. figure-8).

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

            # Check proper interior crossing
            if _segments_cross(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
                return True

            # Check endpoint coincidence (e.g. figure-8 patterns where a
            # vertex is visited twice at non-adjacent ring positions).
            # Only flag if the ring actually travels between positions i and j
            # (not just degenerate zero-length segments from duplicate vertices).
            if _endpoints_coincide(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1):
                has_travel = False
                for k in range(i + 1, j):
                    kx0 = float(x[start + k])
                    ky0 = float(y[start + k])
                    kx1 = float(x[start + k + 1])
                    ky1 = float(y[start + k + 1])
                    if kx0 != kx1 or ky0 != ky1:
                        has_travel = True
                        break
                if has_travel:
                    return True

    return False


def _is_simple_linestrings(buf: FamilyGeometryBuffer) -> np.ndarray:
    """LineString is simple if no non-adjacent segments cross."""
    result = np.ones(buf.row_count, dtype=bool)
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
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
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            continue

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
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            continue

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
    if buf.x.size == 0 or buf.empty_mask.size == 0:
        return result
    x = buf.x
    y = buf.y
    geom_offsets = buf.geometry_offsets
    part_offsets = buf.part_offsets
    ring_offsets = buf.ring_offsets

    for g in range(buf.row_count):
        if buf.empty_mask[g]:
            continue

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


def _build_hole_and_exterior_indices(d_buf, family):
    """Build hole_ring_indices and exterior_ring_indices for holes_in_shell kernel.

    For Polygon:      geometry_offsets[g] is the exterior ring; rings
                      geometry_offsets[g]+1 .. geometry_offsets[g+1]-1 are holes.
                      Each hole's exterior ring is geometry_offsets[g].
    For MultiPolygon: part_offsets[p] is the exterior ring of polygon part p;
                      rings part_offsets[p]+1 .. part_offsets[p+1]-1 are holes.
                      Each hole's exterior ring is part_offsets[p].

    Returns (d_hole_ring_indices, d_exterior_ring_indices, hole_count).
    All arrays are CuPy int32 on device.
    """
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        return cp.empty(0, dtype=cp.int32), cp.empty(0, dtype=cp.int32), 0

    # Build is_exterior mask
    d_is_exterior = _build_is_exterior_for_validity(d_buf, family)

    # Hole ring indices: all rings where is_exterior == 0
    d_hole_ring_indices = cp.flatnonzero(d_is_exterior == 0).astype(cp.int32)
    hole_count = int(d_hole_ring_indices.shape[0])
    if hole_count == 0:
        return d_hole_ring_indices, cp.empty(0, dtype=cp.int32), 0

    # For each hole, find its polygon's exterior ring index.
    # The exterior ring of a hole is the largest value in d_offsets[:-1] that
    # is <= the hole's ring index.  We use searchsorted for this.
    if family is GeometryFamily.POLYGON:
        d_offsets = cp.asarray(d_buf.geometry_offsets)
    else:
        d_offsets = cp.asarray(d_buf.part_offsets)

    # d_offsets[:-1] are the exterior ring indices (one per polygon part).
    # For each hole ring index h, the exterior ring is d_offsets[k] where
    # k = searchsorted(d_offsets, h, side='right') - 1
    d_ext_starts = d_offsets[:-1]
    d_pos = cp.searchsorted(d_ext_starts, d_hole_ring_indices, side="right") - 1
    d_exterior_ring_indices = d_ext_starts[d_pos].astype(cp.int32)

    return d_hole_ring_indices, d_exterior_ring_indices, hole_count


def _launch_holes_in_shell_kernel(runtime, d_buf, d_hole_ring_indices,
                                  d_exterior_ring_indices, hole_count):
    """Launch the holes_in_shell kernel and return device result array (int32).

    Returns a device array of shape (hole_count,) where 1 = hole inside shell,
    0 = hole outside shell.
    """
    kernels = compile_kernel_group(
        "holes-in-shell-fp64",
        _HOLES_IN_SHELL_FP64,
        _HOLES_IN_SHELL_KERNEL_NAMES,
    )
    kernel = kernels["holes_in_shell"]

    d_hole_valid = runtime.allocate((hole_count,), np.int32, zero=True)
    ptr = runtime.pointer
    params = (
        (ptr(d_buf.x), ptr(d_buf.y), ptr(d_buf.ring_offsets),
         ptr(d_hole_ring_indices), ptr(d_exterior_ring_indices),
         ptr(d_hole_valid), hole_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    grid, block = runtime.launch_config(kernel, hole_count)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_hole_valid


def _reduce_hole_valid_to_geom(d_hole_valid, d_hole_ring_indices,
                               d_buf, family, family_rows, geom_count):
    """Reduce per-hole validity to per-geometry using cumsum.

    A geometry is valid (for hole containment) iff ALL its holes are inside
    their respective exterior rings.

    Maps hole indices back to geometries via the offset structure, then
    uses the same cumsum trick as _reduce_ring_valid_to_geom.

    Returns a CuPy array of bool (length geom_count) on device.
    """
    hole_count = int(d_hole_valid.shape[0])
    if hole_count == 0:
        return cp.ones(geom_count, dtype=cp.bool_)

    # Build per-geometry ring ranges (same as the ring reduction uses)
    d_family_rows = cp.asarray(family_rows)
    if family is GeometryFamily.POLYGON:
        d_starts = d_buf.geometry_offsets[d_family_rows]
        d_ends = d_buf.geometry_offsets[d_family_rows + 1]
    else:
        # MultiPolygon: geometry -> part -> ring range
        d_geom_offsets = d_buf.geometry_offsets[d_family_rows]
        d_geom_offsets_end = d_buf.geometry_offsets[d_family_rows + 1]
        d_starts = d_buf.part_offsets[d_geom_offsets]
        d_ends = d_buf.part_offsets[d_geom_offsets_end]

    # For each hole, find which geometry it belongs to, using the ring index.
    # We map hole_ring_index -> geometry by searchsorted on d_ends.
    # Hole ring h belongs to geometry g where d_starts[g] <= h < d_ends[g].
    # Equivalently, g = searchsorted(d_ends, h+1, side='left') but that would
    # require consecutive geometry ring ranges (which they are for the buffer).
    #
    # Simpler: cumsum of invalid holes, then diff at geometry boundaries.
    # The hole ring indices are a subset of all ring indices, and the geometry
    # ring ranges partition all ring indices.
    #
    # We use the full ring_count approach: create a full-ring invalid array
    # (all zeros), set the invalid holes, then reduce as usual.
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    d_hole_invalid_full = cp.zeros(ring_count, dtype=cp.int32)

    # Mark rings where hole is not inside shell
    d_invalid_holes = d_hole_valid == 0
    d_invalid_hole_indices = d_hole_ring_indices[d_invalid_holes]
    if d_invalid_hole_indices.size > 0:
        d_hole_invalid_full[d_invalid_hole_indices] = 1

    # Now reduce using the standard cumsum trick
    d_cumsum = cp.cumsum(d_hole_invalid_full)
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
        d_result[d_ne_indices] = d_result[d_ne_indices] & (d_invalid_counts == 0)

    return d_result


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


def _launch_ring_pair_interaction_kernel(runtime, d_buf, d_poly_ring_starts,
                                         d_poly_ring_ends, poly_count):
    """Launch the ring_pair_interaction kernel and return device result (int32).

    Returns a device array of shape (poly_count,) where 1 = valid (no
    inter-ring violation), 0 = invalid (crossing, overlap, or multi-touch).
    """
    kernels = compile_kernel_group(
        "ring-pair-interaction-fp64",
        _RING_PAIR_INTERACTION_FP64,
        _RING_PAIR_INTERACTION_KERNEL_NAMES,
    )
    kernel = kernels["ring_pair_interaction"]

    d_poly_valid = runtime.allocate((poly_count,), np.int32, zero=True)
    ptr = runtime.pointer
    params = (
        (ptr(d_buf.x), ptr(d_buf.y), ptr(d_buf.ring_offsets),
         ptr(d_poly_ring_starts), ptr(d_poly_ring_ends),
         ptr(d_poly_valid), poly_count),
        (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
         KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
    )
    # 1 block per polygon part — use occupancy API for optimal block size
    block_size = runtime.optimal_block_size(kernel)
    grid = (poly_count,)
    block = (block_size,)
    runtime.launch(kernel, grid=grid, block=block, params=params)
    return d_poly_valid


def _is_valid_gpu_polygons(d_buf, family_rows, result, global_rows):
    """Polygon validity via NVRTC kernels + CuPy reduction.

    OGC validity for polygon rings requires:
    1. Structural checks (closure, min coords) via is_valid_rings
    2. Ring simplicity (no self-intersection) via is_simple_segments
    3. Hole containment (each hole's first vertex inside exterior ring)
       via holes_in_shell
    4. Ring-pair interaction (no crossing, overlap, or multi-touch)
       via ring_pair_interaction
    """
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    # Step 1: structural ring validity (closure, min coords)
    kernels = compile_kernel_group(
        "is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES,
    )
    kernel = kernels["is_valid_rings"]

    d_is_exterior = _build_is_exterior_for_validity(d_buf, GeometryFamily.POLYGON)
    d_ring_valid = runtime.allocate((ring_count,), np.int32, zero=True)
    d_ring_simple = None
    d_hole_valid = None
    d_rpi_valid = None
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

        # Step 2: ring simplicity (no self-intersection) via is_simple_segments
        d_ring_simple = _launch_is_simple_kernel(
            runtime, d_buf.x, d_buf.y, d_buf.ring_offsets, ring_count, is_ring=1,
        )

        # Combine: ring is valid only if structurally valid AND simple
        d_ring_valid_cp = cp.asarray(d_ring_valid)
        d_ring_simple_cp = cp.asarray(d_ring_simple)
        d_ring_ok = (d_ring_valid_cp & d_ring_simple_cp).astype(cp.int32)

        # Reduce per-ring results to per-geometry (all rings must be valid)
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_starts = d_buf.geometry_offsets[d_family_rows]
        d_ends = d_buf.geometry_offsets[d_family_rows + 1]

        d_invalid = 1 - d_ring_ok
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

        # Stay on device: d_result is the running per-geometry validity mask

        # Step 3: hole-in-shell containment
        d_hole_indices, d_ext_indices, hole_count = \
            _build_hole_and_exterior_indices(d_buf, GeometryFamily.POLYGON)
        if hole_count > 0:
            d_hole_valid = _launch_holes_in_shell_kernel(
                runtime, d_buf, d_hole_indices, d_ext_indices, hole_count,
            )
            d_hole_geom = _reduce_hole_valid_to_geom(
                cp.asarray(d_hole_valid), d_hole_indices,
                d_buf, GeometryFamily.POLYGON, family_rows, geom_count,
            )
            # AND on device: geometry valid only if rings ok AND holes ok
            d_result &= d_hole_geom

        # Step 4: Ring-pair interaction (crossing, overlap, multi-touch)
        # For Polygon: poly_ring_starts[g] = geometry_offsets[g],
        #              poly_ring_ends[g]   = geometry_offsets[g+1]
        # One polygon part per geometry row.
        d_poly_ring_starts = d_buf.geometry_offsets[:-1]
        d_poly_ring_ends = d_buf.geometry_offsets[1:]
        total_polys = int(d_poly_ring_starts.shape[0])
        d_rpi_valid = None
        if total_polys > 0:
            d_rpi_valid = _launch_ring_pair_interaction_kernel(
                runtime, d_buf, d_poly_ring_starts, d_poly_ring_ends, total_polys,
            )
            # AND on device
            d_rpi_cp = cp.asarray(d_rpi_valid)
            d_result &= d_rpi_cp[d_family_rows].astype(cp.bool_)

        # Single D->H transfer at the end
        result[global_rows] = cp.asnumpy(d_result)
    finally:
        # LIFO deallocation order for pool coalescing
        if d_rpi_valid is not None:
            runtime.free(d_rpi_valid)
        if d_hole_valid is not None:
            runtime.free(d_hole_valid)
        if d_ring_simple is not None:
            runtime.free(d_ring_simple)
        runtime.free(d_ring_valid)


def _is_valid_gpu_multipolygons(d_buf, family_rows, result, global_rows):
    """MultiPolygon validity via NVRTC kernels + CuPy reduction.

    MultiPolygon uses part_offsets to reach rings. OGC validity requires:
    1. Structural checks (closure, min coords) via is_valid_rings
    2. Ring simplicity (no self-intersection) via is_simple_segments
    3. Hole containment via holes_in_shell
    4. Ring-pair interaction (no crossing, overlap, or multi-touch)
       via ring_pair_interaction
    """
    runtime = get_cuda_runtime()
    ring_count = int(d_buf.ring_offsets.shape[0]) - 1
    if ring_count == 0:
        result[global_rows] = True
        return

    # Step 1: structural ring validity (closure, min coords)
    kernels = compile_kernel_group(
        "is-valid-rings-fp64", _IS_VALID_RINGS_FP64, _IS_VALID_RINGS_KERNEL_NAMES,
    )
    kernel = kernels["is_valid_rings"]

    d_is_exterior = _build_is_exterior_for_validity(d_buf, GeometryFamily.MULTIPOLYGON)
    d_ring_valid = runtime.allocate((ring_count,), np.int32, zero=True)
    d_ring_simple = None
    d_hole_valid = None
    d_rpi_valid = None
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

        # Step 2: ring simplicity (no self-intersection) via is_simple_segments
        d_ring_simple = _launch_is_simple_kernel(
            runtime, d_buf.x, d_buf.y, d_buf.ring_offsets, ring_count, is_ring=1,
        )

        # Combine: ring is valid only if structurally valid AND simple
        d_ring_valid_cp = cp.asarray(d_ring_valid)
        d_ring_simple_cp = cp.asarray(d_ring_simple)
        d_ring_ok = (d_ring_valid_cp & d_ring_simple_cp).astype(cp.int32)

        # For MultiPolygon: geometry_offsets -> parts, part_offsets -> rings.
        # Build ring-level spans per geometry by chaining:
        #   ring_start = part_offsets[geometry_offsets[g]]
        #   ring_end = part_offsets[geometry_offsets[g+1]]
        geom_count = int(family_rows.shape[0])
        d_family_rows = cp.asarray(family_rows)
        d_geom_offsets = d_buf.geometry_offsets[d_family_rows]
        d_geom_offsets_end = d_buf.geometry_offsets[d_family_rows + 1]

        # Map geometry part range to ring range via part_offsets
        d_starts = d_buf.part_offsets[d_geom_offsets]
        d_ends = d_buf.part_offsets[d_geom_offsets_end]

        d_invalid = 1 - d_ring_ok
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

        # Stay on device: d_result is the running per-geometry validity mask

        # Step 3: hole-in-shell containment
        d_hole_indices, d_ext_indices, hole_count = \
            _build_hole_and_exterior_indices(d_buf, GeometryFamily.MULTIPOLYGON)
        if hole_count > 0:
            d_hole_valid = _launch_holes_in_shell_kernel(
                runtime, d_buf, d_hole_indices, d_ext_indices, hole_count,
            )
            d_hole_geom = _reduce_hole_valid_to_geom(
                cp.asarray(d_hole_valid), d_hole_indices,
                d_buf, GeometryFamily.MULTIPOLYGON, family_rows, geom_count,
            )
            # AND on device
            d_result &= d_hole_geom

        # Step 4: Ring-pair interaction (crossing, overlap, multi-touch)
        # For MultiPolygon: each polygon part p has rings part_offsets[p]..part_offsets[p+1]
        d_poly_ring_starts = d_buf.part_offsets[:-1]
        d_poly_ring_ends = d_buf.part_offsets[1:]
        total_polys = int(d_poly_ring_starts.shape[0])
        if total_polys > 0:
            d_rpi_valid = _launch_ring_pair_interaction_kernel(
                runtime, d_buf, d_poly_ring_starts, d_poly_ring_ends, total_polys,
            )
            # Reduce per-part RPI to per-geometry: a geometry is invalid if ANY
            # of its parts is invalid. Use cumsum trick on (1 - rpi_valid).
            d_rpi_cp = cp.asarray(d_rpi_valid)
            d_rpi_invalid = (1 - d_rpi_cp).astype(cp.int32)
            d_rpi_cumsum = cp.cumsum(d_rpi_invalid)

            d_family_rows_mp = cp.asarray(family_rows)
            d_gstart = d_buf.geometry_offsets[d_family_rows_mp]
            d_gend = d_buf.geometry_offsets[d_family_rows_mp + 1]

            d_ge = d_gstart == d_gend
            d_rpi_result = cp.ones(geom_count, dtype=cp.bool_)
            d_ne = cp.flatnonzero(~d_ge)
            if d_ne.size > 0:
                d_e_idx = d_gend[d_ne] - 1
                d_e_sums = d_rpi_cumsum[d_e_idx]
                d_s_ne = d_gstart[d_ne]
                d_s_sums = cp.zeros(d_ne.size, dtype=cp.int32)
                d_hp = d_s_ne > 0
                d_hp_idx = cp.flatnonzero(d_hp)
                if d_hp_idx.size > 0:
                    d_s_sums[d_hp_idx] = d_rpi_cumsum[d_s_ne[d_hp_idx] - 1]
                d_ic = d_e_sums - d_s_sums
                d_rpi_result[d_ne] = (d_ic == 0)

            # AND on device
            d_result &= d_rpi_result

        # Single D->H transfer at the end
        result[global_rows] = cp.asnumpy(d_result)
    finally:
        # LIFO deallocation order for pool coalescing
        if d_rpi_valid is not None:
            runtime.free(d_rpi_valid)
        if d_hole_valid is not None:
            runtime.free(d_hole_valid)
        if d_ring_simple is not None:
            runtime.free(d_ring_simple)
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
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    precision_modes=(PrecisionMode.FP64,),
    tags=("cuda-python", "predicate", "validity"),
)
def _is_valid_gpu_dispatch(owned):
    """Registry marker — actual dispatch is inline in is_valid_owned."""
    raise NotImplementedError


@register_kernel_variant(
    "is_valid",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    precision_modes=(PrecisionMode.FP64,),
    tags=("shapely", "predicate", "validity"),
)
def _is_valid_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU fallback: per-family structural validity check."""
    row_count = owned.row_count
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
        family_result = _VALIDITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    return result


@register_kernel_variant(
    "is_simple",
    "gpu-cuda-python",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.GPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    precision_modes=(PrecisionMode.FP64,),
    tags=("cuda-python", "predicate", "simplicity"),
)
def _is_simple_gpu_dispatch(owned):
    """Registry marker — actual dispatch is inline in is_simple_owned."""
    raise NotImplementedError


@register_kernel_variant(
    "is_simple",
    "cpu",
    kernel_class=KernelClass.PREDICATE,
    execution_modes=(ExecutionMode.CPU,),
    geometry_families=(
        "point", "multipoint", "linestring", "multilinestring",
        "polygon", "multipolygon",
    ),
    supports_mixed=True,
    precision_modes=(PrecisionMode.FP64,),
    tags=("shapely", "predicate", "simplicity"),
)
def _is_simple_cpu(owned: OwnedGeometryArray) -> np.ndarray:
    """CPU fallback: per-family structural simplicity check."""
    row_count = owned.row_count
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
        family_result = _SIMPLICITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_valid_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> np.ndarray:
    """Check OGC validity of each geometry in an OwnedGeometryArray.

    Checks performed per family:
    - Ring closure (first coord == last coord for polygon rings)
    - Minimum coordinate counts (LineString >= 2, Polygon ring >= 4)
    - Ring self-intersection (no non-adjacent segments cross within a ring)
    - Hole-in-shell containment (each hole's first vertex inside exterior)
    - Ring-pair interaction: no proper crossing, collinear overlap, or
      multi-touch (2+ distinct contact points) between distinct rings

    Null and empty geometries return True, matching Shapely semantics.

    Parameters
    ----------
    owned : OwnedGeometryArray
        The geometry array to validate.
    dispatch_mode : ExecutionMode or str
        GPU/CPU/AUTO execution mode.
    precision : PrecisionMode or str
        Precision dispatch mode (ADR-0002). PREDICATE class defaults to fp64.

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
        requested_precision=precision,
    )

    # ADR-0002: PREDICATE class kernels always use fp64 for exact comparisons
    # (ring closure, self-intersection, hole containment, ring-pair interaction).  select_precision_plan is called for
    # observability only -- the kernel source is always _FP64.
    precision_plan = selection.precision_plan

    use_gpu = (
        selection.selected is ExecutionMode.GPU
        and cp is not None
        and owned.device_state is not None
    )

    result = np.ones(row_count, dtype=bool)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    actually_used_gpu = False

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

                actually_used_gpu = True
                continue  # skip CPU fallback
            except Exception:
                pass  # fall through to CPU

        # --- CPU fallback ---
        family_result = _VALIDITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    selected_mode = ExecutionMode.GPU if actually_used_gpu else ExecutionMode.CPU
    impl = "is_valid_gpu_nvrtc" if actually_used_gpu else "is_valid_cpu"
    record_dispatch_event(
        surface="geopandas.array.is_valid",
        operation="is_valid",
        requested=dispatch_mode,
        selected=selected_mode,
        implementation=impl,
        reason=selection.reason,
        detail=f"precision=fp64 (PREDICATE class, plan={precision_plan.compute_precision.value})",
    )

    # Null geometries are valid (Shapely convention)
    # Already True by default; no action needed for nulls.
    return result


def is_simple_owned(
    owned: OwnedGeometryArray,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
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
    precision : PrecisionMode or str
        Precision dispatch mode (ADR-0002). PREDICATE class defaults to fp64.

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
        requested_precision=precision,
    )

    # ADR-0002: PREDICATE class kernels always use fp64 for exact comparisons.
    # select_precision_plan is called for observability only.
    precision_plan = selection.precision_plan

    use_gpu = (
        selection.selected is ExecutionMode.GPU
        and cp is not None
        and owned.device_state is not None
    )

    result = np.ones(row_count, dtype=bool)
    tags = owned.tags
    family_row_offsets = owned.family_row_offsets
    actually_used_gpu = False

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

                actually_used_gpu = True
                continue  # skip CPU fallback
            except Exception:
                pass  # fall through to CPU

        # --- CPU fallback ---
        family_result = _SIMPLICITY_DISPATCH[family](buf)
        result[global_rows] = family_result[family_rows]

    selected_mode = ExecutionMode.GPU if actually_used_gpu else ExecutionMode.CPU
    impl = "is_simple_gpu_nvrtc" if actually_used_gpu else "is_simple_cpu"
    record_dispatch_event(
        surface="geopandas.array.is_simple",
        operation="is_simple",
        requested=dispatch_mode,
        selected=selected_mode,
        implementation=impl,
        reason=selection.reason,
        detail=f"precision=fp64 (PREDICATE class, plan={precision_plan.compute_precision.value})",
    )

    # Null geometries are simple (Shapely convention)
    # Already True by default; no action needed for nulls.
    return result
