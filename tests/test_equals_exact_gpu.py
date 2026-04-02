"""Tests for GPU NVRTC equals_exact kernel.

Verifies correctness against Shapely oracle for all geometry families,
null handling, tolerance semantics, and mixed-family arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.testing import build_owned as _make_owned

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def _shapely_equals_exact(left_geoms, right_geoms, tolerance):
    """Shapely oracle for equals_exact."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray(right_geoms, dtype=object)
    return shapely.equals_exact(left_arr, right_arr, tolerance=tolerance)


# ---------------------------------------------------------------------------
# Point family
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_points_identical(make_owned):
    """Identical points should be equal with zero tolerance."""
    n = 1200  # above GPU threshold
    geoms = [Point(i * 0.1, i * 0.2) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_points_with_tolerance(make_owned):
    """Points within tolerance should match."""
    n = 1200
    base = [Point(i, i * 2) for i in range(n)]
    shifted = [Point(i + 0.005, i * 2 - 0.003) for i in range(n)]
    left = make_owned(base)
    right = make_owned(shifted)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    # Tolerance large enough to match
    result = geom_equals_exact_owned(left, right, tolerance=0.01, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(base, shifted, 0.01)
    np.testing.assert_array_equal(result, expected)
    assert result.all()

    # Tolerance too small
    result2 = geom_equals_exact_owned(left, right, tolerance=0.001, dispatch_mode=ExecutionMode.GPU)
    expected2 = _shapely_equals_exact(base, shifted, 0.001)
    np.testing.assert_array_equal(result2, expected2)
    assert not result2.any()


@requires_gpu
def test_equals_exact_points_different(make_owned):
    """Different points should not be equal."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [Point(i + 1, i + 1) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(left_geoms, right_geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# LineString family
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_linestrings(make_owned):
    """Identical linestrings should be equal."""
    n = 1200
    geoms = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_linestrings_different_length(make_owned):
    """LineStrings with different vertex counts should not match."""
    n = 1200
    left_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    right_geoms = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(left_geoms, right_geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Polygon family
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_polygons(make_owned):
    """Identical polygons should be equal."""
    n = 1200
    geoms = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)])
        for i in range(n)
    ]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_polygons_with_holes(make_owned):
    """Polygons with holes should check ring structure."""
    n = 1200
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    geoms_with_hole = [Polygon(outer, [hole]) for _ in range(n)]
    geoms_no_hole = [Polygon(outer) for _ in range(n)]
    left = make_owned(geoms_with_hole)
    right = make_owned(geoms_no_hole)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms_with_hole, geoms_no_hole, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Multi* families
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_multipoints(make_owned):
    """Identical MultiPoints should be equal."""
    n = 1200
    geoms = [MultiPoint([(i, 0), (i + 1, 1)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_multilinestrings():
    """Identical MultiLineStrings should be equal."""
    n = 1200
    geoms = [
        MultiLineString([[(i, 0), (i + 1, 1)], [(i + 2, 0), (i + 3, 1)]])
        for i in range(n)
    ]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_multipolygons():
    """Identical MultiPolygons should be equal."""
    n = 1200
    geoms = [
        MultiPolygon([
            ([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)], []),
            ([(i + 2, 0), (i + 3, 0), (i + 3, 1), (i + 2, 1), (i + 2, 0)], []),
        ])
        for i in range(n)
    ]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_null_geometries():
    """Null geometries should always return False (Shapely convention)."""
    n = 1200
    geoms = [Point(i, i) if i % 3 != 0 else None for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)

    # Null positions should be False
    for i in range(n):
        if geoms[i] is None:
            assert not result[i], f"Null at index {i} should be False"


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_mixed_families():
    """Array with mixed geometry families should handle tag mismatches."""
    n = 1200
    geoms = []
    for i in range(n):
        if i % 4 == 0:
            geoms.append(Point(i, i))
        elif i % 4 == 1:
            geoms.append(LineString([(i, 0), (i + 1, 1)]))
        elif i % 4 == 2:
            geoms.append(Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]))
        else:
            geoms.append(MultiPoint([(i, 0), (i + 1, 1)]))

    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_equals_exact_type_mismatch():
    """Different geometry types should always return False."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(left_geoms, right_geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_equals_exact_cpu_fallback():
    """CPU path should match Shapely output."""
    n = 100  # below GPU threshold
    geoms = [Point(i, i) for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0)
    expected = _shapely_equals_exact(geoms, geoms, 0.0)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@requires_gpu
def test_equals_exact_empty_array():
    """Empty arrays should return empty result."""
    left = _make_owned([])
    right = _make_owned([])

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.0)
    assert result.shape == (0,)


@requires_gpu
def test_equals_exact_partial_tolerance_match():
    """Some pairs match within tolerance, others do not."""
    n = 1200
    base = [Point(i, i) for i in range(n)]
    # Shift half of them beyond tolerance
    shifted = [
        Point(i + 0.005, i + 0.005) if i % 2 == 0
        else Point(i + 0.05, i + 0.05)
        for i in range(n)
    ]
    left = _make_owned(base)
    right = _make_owned(shifted)

    from vibespatial.geometry.equality import geom_equals_exact_owned
    result = geom_equals_exact_owned(left, right, tolerance=0.01, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact(base, shifted, 0.01)
    np.testing.assert_array_equal(result, expected)
