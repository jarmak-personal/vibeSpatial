"""Tests for GPU polygon intersection kernel.

Validates the Sutherland-Hodgman GPU kernel against Shapely oracle
for correctness, edge cases, and precision compliance (ADR-0002).
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from vibespatial.runtime import ExecutionMode
from vibespatial.testing import build_owned as _make_owned_polygons

try:
    from vibespatial.cuda._runtime import has_cuda_device

    _has_gpu = has_cuda_device()
except (ImportError, ModuleNotFoundError):
    _has_gpu = False

requires_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")
def _shapely_intersection(left_geoms, right_geoms):
    """Shapely oracle: element-wise intersection."""
    left_arr = np.empty(len(left_geoms), dtype=object)
    left_arr[:] = left_geoms
    right_arr = np.empty(len(right_geoms), dtype=object)
    right_arr[:] = right_geoms
    return shapely.intersection(left_arr, right_arr)


def _assert_geom_equal(gpu_geom, ref_geom, *, rtol=1e-6, msg=""):
    """Assert two geometries are equal within tolerance.

    Handles empty geometries and None values. Uses shapely.equals_exact
    for geometric comparison.
    """
    if ref_geom is None or (hasattr(ref_geom, "is_empty") and ref_geom.is_empty):
        # GPU result should also be None or empty
        if gpu_geom is not None and hasattr(gpu_geom, "is_empty"):
            assert gpu_geom.is_empty, f"Expected empty/None but got {gpu_geom}. {msg}"
        return

    if gpu_geom is None:
        pytest.fail(f"GPU returned None but expected {ref_geom}. {msg}")

    if hasattr(gpu_geom, "is_empty") and gpu_geom.is_empty:
        pytest.fail(f"GPU returned empty but expected {ref_geom}. {msg}")

    # For polygon-polygon intersection, the result should be a polygon.
    # Check area-based similarity since vertex ordering may differ.
    gpu_area = shapely.area(gpu_geom)
    ref_area = shapely.area(ref_geom)

    if ref_area < 1e-12:
        # Degenerate result (point or line intersection)
        return

    area_ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
    assert area_ratio < rtol, (
        f"Area mismatch: GPU={gpu_area}, ref={ref_area}, ratio={area_ratio}. {msg}"
    )

    # Also check that the symmetric difference is small
    sym_diff = shapely.area(shapely.symmetric_difference(gpu_geom, ref_geom))
    sym_ratio = sym_diff / max(abs(ref_area), 1e-15)
    assert sym_ratio < rtol, (
        f"Symmetric difference too large: {sym_diff} (ratio={sym_ratio}). {msg}"
    )


# ---------------------------------------------------------------------------
# Test: basic overlapping rectangles
# ---------------------------------------------------------------------------

@requires_gpu
def test_basic_rectangle_overlap(make_owned):
    """Two overlapping axis-aligned rectangles."""
    left_geoms = [box(0, 0, 4, 4)]
    right_geoms = [box(2, 2, 6, 6)]

    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="basic rectangle overlap")


# ---------------------------------------------------------------------------
# Test: fully contained polygon
# ---------------------------------------------------------------------------

@requires_gpu
def test_fully_contained(make_owned):
    """Left polygon fully inside right polygon."""
    left_geoms = [box(1, 1, 3, 3)]
    right_geoms = [box(0, 0, 10, 10)]

    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="fully contained")


# ---------------------------------------------------------------------------
# Test: no overlap (empty result)
# ---------------------------------------------------------------------------

@requires_gpu
def test_no_overlap():
    """Disjoint polygons -> empty result."""
    left_geoms = [box(0, 0, 1, 1)]
    right_geoms = [box(5, 5, 6, 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 1
    # Should be empty or None
    geom = result_geoms[0]
    assert geom is None or geom.is_empty, f"Expected empty but got {geom}"


# ---------------------------------------------------------------------------
# Test: touching edges
# ---------------------------------------------------------------------------

@requires_gpu
def test_touching_edges():
    """Polygons that share an edge but have zero-area intersection."""
    left_geoms = [box(0, 0, 2, 2)]
    right_geoms = [box(2, 0, 4, 2)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    # Touching edge -> degenerate (line) intersection -> treated as empty polygon
    assert len(result_geoms) == 1
    geom = result_geoms[0]
    if geom is not None and not geom.is_empty:
        # If it produces a polygon, its area should be negligible
        assert shapely.area(geom) < 1e-10


# ---------------------------------------------------------------------------
# Test: multiple pairs (batched)
# ---------------------------------------------------------------------------

@requires_gpu
def test_multiple_pairs():
    """Batched intersection of multiple polygon pairs."""
    left_geoms = [
        box(0, 0, 4, 4),
        box(0, 0, 2, 2),
        box(10, 10, 20, 20),
    ]
    right_geoms = [
        box(2, 2, 6, 6),
        box(5, 5, 8, 8),
        box(15, 15, 25, 25),
    ]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 3

    # Pair 0: overlapping
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="pair 0")
    # Pair 1: disjoint
    _assert_geom_equal(result_geoms[1], ref_geoms[1], msg="pair 1")
    # Pair 2: overlapping
    _assert_geom_equal(result_geoms[2], ref_geoms[2], msg="pair 2")


# ---------------------------------------------------------------------------
# Test: null input propagation
# ---------------------------------------------------------------------------

@requires_gpu
def test_null_input_propagation():
    """Null inputs should produce null outputs."""
    left_geoms = [box(0, 0, 4, 4), None, box(1, 1, 3, 3)]
    right_geoms = [None, box(2, 2, 6, 6), box(0, 0, 5, 5)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 3
    # Pair 0: right is None -> null
    assert result_geoms[0] is None or (hasattr(result_geoms[0], "is_empty") and result_geoms[0].is_empty)
    # Pair 1: left is None -> null
    assert result_geoms[1] is None or (hasattr(result_geoms[1], "is_empty") and result_geoms[1].is_empty)
    # Pair 2: both valid -> should have area
    ref = shapely.intersection(
        shapely.from_wkt("POLYGON ((1 1, 3 1, 3 3, 1 3, 1 1))"),
        shapely.from_wkt("POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))"),
    )
    _assert_geom_equal(result_geoms[2], ref, msg="pair 2 with valid inputs")


# ---------------------------------------------------------------------------
# Test: non-axis-aligned polygons (triangles)
# ---------------------------------------------------------------------------

@requires_gpu
def test_triangle_intersection():
    """Intersection of two overlapping triangles."""
    left_geoms = [Polygon([(0, 0), (4, 0), (2, 4), (0, 0)])]
    right_geoms = [Polygon([(1, 0), (5, 0), (3, 4), (1, 0)])]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="triangle intersection")


# ---------------------------------------------------------------------------
# Test: identical polygons
# ---------------------------------------------------------------------------

@requires_gpu
def test_identical_polygons():
    """Intersection of identical polygons should return the same polygon."""
    geom = box(0, 0, 5, 5)
    left_geoms = [geom]
    right_geoms = [geom]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], geom, msg="identical polygons")


# ---------------------------------------------------------------------------
# Test: large coordinate values (precision stress)
# ---------------------------------------------------------------------------

@requires_gpu
def test_large_coordinates():
    """Polygons with large absolute coordinate values."""
    offset = 1_000_000.0
    left_geoms = [box(offset, offset, offset + 4, offset + 4)]
    right_geoms = [box(offset + 2, offset + 2, offset + 6, offset + 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(
        result_geoms[0], ref_geoms[0],
        rtol=1e-5,
        msg="large coordinates",
    )


# ---------------------------------------------------------------------------
# Test: CPU fallback
# ---------------------------------------------------------------------------

def test_cpu_fallback():
    """CPU fallback produces correct results via Shapely."""
    left_geoms = [box(0, 0, 4, 4)]
    right_geoms = [box(2, 2, 6, 6)]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.CPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="CPU fallback")


# ---------------------------------------------------------------------------
# Test: row count mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_row_count_mismatch():
    """Mismatched row counts should raise ValueError."""
    left = _make_owned_polygons([box(0, 0, 1, 1)])
    right = _make_owned_polygons([box(0, 0, 1, 1), box(2, 2, 3, 3)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    with pytest.raises(ValueError, match="row count mismatch"):
        polygon_intersection(left, right)


# ---------------------------------------------------------------------------
# Test: empty input array
# ---------------------------------------------------------------------------

def test_empty_input():
    """Empty input arrays should return empty result."""
    left = _make_owned_polygons([])
    right = _make_owned_polygons([])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right)
    assert result.row_count == 0


# ---------------------------------------------------------------------------
# Test: device-resident result (no D->H in hot path)
# ---------------------------------------------------------------------------

@requires_gpu
def test_result_is_device_resident(strict_device_guard):
    """GPU result should stay device-resident with lazy host metadata."""
    left = _make_owned_polygons([box(0, 0, 4, 4)])
    right = _make_owned_polygons([box(2, 2, 6, 6)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    assert result.residency == Residency.DEVICE
    assert result.device_state is not None
    assert result._validity is None
    assert result._tags is None
    assert result._family_row_offsets is None


# ---------------------------------------------------------------------------
# Test: partial overlap (L-shaped result)
# ---------------------------------------------------------------------------

@requires_gpu
def test_partial_overlap_pentagon():
    """Intersection that produces a non-rectangular polygon."""
    left_geoms = [Polygon([(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)])]
    right_geoms = [Polygon([(2, -1), (8, -1), (8, 4), (2, 4), (2, -1)])]

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == 1
    _assert_geom_equal(result_geoms[0], ref_geoms[0], msg="partial overlap pentagon")


# ---------------------------------------------------------------------------
# Test: many pairs (stress test)
# ---------------------------------------------------------------------------

@requires_gpu
def test_many_pairs():
    """Stress test with many polygon pairs."""
    rng = np.random.default_rng(42)
    n = 500
    left_geoms = []
    right_geoms = []
    for _ in range(n):
        x, y = rng.uniform(0, 100, 2)
        w, h = rng.uniform(1, 10, 2)
        left_geoms.append(box(x, y, x + w, y + h))
        x2, y2 = rng.uniform(0, 100, 2)
        w2, h2 = rng.uniform(1, 10, 2)
        right_geoms.append(box(x2, y2, x2 + w2, y2 + h2))

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    assert len(result_geoms) == n

    # Check each pair
    mismatches = 0
    for i in range(n):
        ref = ref_geoms[i]
        gpu = result_geoms[i]

        ref_area = shapely.area(ref) if ref is not None else 0.0
        gpu_area = shapely.area(gpu) if gpu is not None else 0.0

        if ref_area < 1e-10 and gpu_area < 1e-10:
            continue  # Both empty/degenerate

        if ref_area < 1e-10:
            if gpu_area > 1e-6:
                mismatches += 1
            continue

        ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
        if ratio > 1e-4:
            mismatches += 1

    # Allow a small fraction of mismatches (Sutherland-Hodgman is not exact
    # for concave polygons, but boxes are convex so this should be zero)
    assert mismatches == 0, f"{mismatches}/{n} pairs had area mismatches"


# ---------------------------------------------------------------------------
# Test: ADR-0002 precision plan is wired through
# ---------------------------------------------------------------------------

@requires_gpu
def test_precision_plan_wired():
    """Verify that the precision plan is computed and stays fp64 for CONSTRUCTIVE."""
    from vibespatial.runtime.dispatch import get_dispatch_events

    left = _make_owned_polygons([box(0, 0, 4, 4)])
    right = _make_owned_polygons([box(2, 2, 6, 6)])

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)

    # Check dispatch event was recorded
    events = get_dispatch_events()
    pi_events = [
        e for e in events
        if e.operation == "polygon_intersection"
    ]
    assert len(pi_events) >= 1
    last_event = pi_events[-1]
    assert "precision=fp64" in last_event.detail


# ---------------------------------------------------------------------------
# Test: CW-wound clip polygon (winding direction fix)
# ---------------------------------------------------------------------------

@requires_gpu
def test_cw_clip_polygon_validity():
    """Clip polygons with CW winding must produce correct intersections.

    Regression test for the validity bitmap bug where Sutherland-Hodgman
    assumed CCW winding on the clip (right) polygon.  CW-wound polygons
    inverted the inside/outside test, producing degenerate (< 3 vertex)
    results that were marked invalid.  The fix computes the signed area
    of the clip polygon to detect winding and flips the test accordingly.
    """
    n = 200
    rng = np.random.default_rng(42)
    left_geoms = []
    right_geoms = []
    for _ in range(n):
        # Random triangles -- Polygon() from random points produces
        # CW or CCW winding depending on vertex order.
        pts_l = rng.uniform(0, 100, (3, 2))
        pts_r = pts_l + rng.uniform(-2, 2, (3, 2))
        left_geoms.append(Polygon(pts_l))
        right_geoms.append(Polygon(pts_r))

    left = _make_owned_polygons(left_geoms)
    right = _make_owned_polygons(right_geoms)

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
    result_geoms = result.to_shapely()
    ref_geoms = _shapely_intersection(left_geoms, right_geoms)

    # GPU valid count should match Shapely (some pairs may have degenerate
    # intersections that the kernel correctly marks invalid)
    false_invalids = 0
    for i in range(n):
        ref = ref_geoms[i]
        ref_area = shapely.area(ref) if ref is not None else 0.0
        if ref_area > 1e-10 and not result.validity[i]:
            false_invalids += 1

    assert false_invalids == 0, (
        f"{false_invalids} false invalids (GPU marked invalid but Shapely "
        f"produced non-degenerate polygon)"
    )

    # Geometric correctness for valid pairs
    for i in range(n):
        if not result.validity[i]:
            continue
        ref_area = shapely.area(ref_geoms[i])
        gpu_area = shapely.area(result_geoms[i]) if result_geoms[i] is not None else 0.0
        if ref_area < 1e-10:
            continue
        ratio = abs(gpu_area - ref_area) / max(abs(ref_area), 1e-15)
        assert ratio < 1e-4, (
            f"Pair {i}: GPU area={gpu_area}, ref area={ref_area}, ratio={ratio}"
        )


@requires_gpu
def test_mixed_winding_boxes():
    """Intersection works regardless of left/right winding direction.

    Tests all 4 combinations: CCW/CCW, CCW/CW, CW/CCW, CW/CW.
    """
    n = 10
    ccw_left = [box(i, i, i + 2, i + 2) for i in range(n)]
    ccw_right = [box(i + 1, i + 1, i + 3, i + 3) for i in range(n)]
    # Reverse coordinate order to get CW winding
    cw_left = [Polygon(list(g.exterior.coords)[::-1]) for g in ccw_left]
    cw_right = [Polygon(list(g.exterior.coords)[::-1]) for g in ccw_right]

    from vibespatial.kernels.constructive.polygon_intersection import polygon_intersection

    for label, left_geoms, right_geoms in [
        ("CCW/CCW", ccw_left, ccw_right),
        ("CCW/CW", ccw_left, cw_right),
        ("CW/CCW", cw_left, ccw_right),
        ("CW/CW", cw_left, cw_right),
    ]:
        left = _make_owned_polygons(left_geoms)
        right = _make_owned_polygons(right_geoms)
        result = polygon_intersection(left, right, dispatch_mode=ExecutionMode.GPU)
        assert result.validity.all(), (
            f"{label}: expected all {n} valid, got {result.validity.sum()}"
        )

        # Check areas match expected (1x1 intersection box = area 1.0)
        result_geoms = result.to_shapely()
        for i in range(n):
            gpu_area = shapely.area(result_geoms[i])
            assert abs(gpu_area - 1.0) < 1e-6, (
                f"{label} pair {i}: expected area 1.0, got {gpu_area}"
            )


# ---------------------------------------------------------------------------
# Import guard for Residency
# ---------------------------------------------------------------------------

from vibespatial.runtime.residency import Residency  # noqa: E402
