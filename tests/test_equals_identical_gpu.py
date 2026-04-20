"""Tests for GPU equals_identical (bitwise coordinate equality).

Verifies correctness against Shapely oracle for all geometry families,
null handling, mixed-family arrays, type mismatches, and the GeoPandas
surface API dispatch.

equals_identical delegates to the equals_exact NVRTC kernel with
tolerance=0, so this test suite validates that the wiring is correct
and the tolerance=0 semantics produce bitwise-exact comparison.
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

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency
from vibespatial.testing import build_owned as _make_owned

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def _shapely_equals_exact_tol0(left_geoms, right_geoms):
    """Shapely oracle: equals_exact with tolerance=0."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray(right_geoms, dtype=object)
    return shapely.equals_exact(left_arr, right_arr, tolerance=0.0)


# ---------------------------------------------------------------------------
# Point family
# ---------------------------------------------------------------------------

@requires_gpu
def test_identical_points_match(make_owned):
    """Identical points should return True."""
    n = 1200  # above GPU threshold
    geoms = [Point(i * 0.1, i * 0.2) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_different_points_do_not_match(make_owned):
    """Points with different coordinates should return False."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [Point(i + 0.001, i) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# LineString family
# ---------------------------------------------------------------------------

@requires_gpu
def test_identical_linestrings_match(make_owned):
    """Identical linestrings should return True."""
    n = 1200
    geoms = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_linestrings_different_vertex_count(make_owned):
    """LineStrings with different vertex counts should return False."""
    n = 1200
    left_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    right_geoms = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Polygon family
# ---------------------------------------------------------------------------

@requires_gpu
def test_identical_polygons_match(make_owned):
    """Identical polygons should return True."""
    n = 1200
    geoms = [
        Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)])
        for i in range(n)
    ]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_polygons_different_ring_structure(make_owned):
    """Polygons with vs without holes should return False."""
    n = 1200
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    geoms_with_hole = [Polygon(outer, [hole]) for _ in range(n)]
    geoms_no_hole = [Polygon(outer) for _ in range(n)]
    left = make_owned(geoms_with_hole)
    right = make_owned(geoms_no_hole)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms_with_hole, geoms_no_hole)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Multi* families
# ---------------------------------------------------------------------------

@requires_gpu
def test_identical_multipoints_match(make_owned):
    """Identical MultiPoints should return True."""
    n = 1200
    geoms = [MultiPoint([(i, 0), (i + 1, 1)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_identical_multilinestrings_match():
    """Identical MultiLineStrings should return True."""
    n = 1200
    geoms = [
        MultiLineString([[(i, 0), (i + 1, 1)], [(i + 2, 0), (i + 3, 1)]])
        for i in range(n)
    ]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_identical_multipolygons_match():
    """Identical MultiPolygons should return True."""
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

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
def test_null_geometries_return_false():
    """Null geometries should always return False (Shapely convention)."""
    n = 1200
    geoms = [Point(i, i) if i % 3 != 0 else None for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)

    # Null positions must be False
    for i in range(n):
        if geoms[i] is None:
            assert not result[i], f"Null at index {i} should be False"


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------

@requires_gpu
def test_mixed_families_identical():
    """Array with mixed geometry families should compare correctly."""
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

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_type_mismatch_returns_false():
    """Different geometry types should always return False."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals_exact_tol0(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------

def test_cpu_fallback():
    """CPU path should match Shapely output."""
    n = 100  # below GPU threshold
    geoms = [Point(i, i) for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_device_resident_small_equals_identical_sticks_to_gpu():
    """Small AUTO equality stays on GPU after inputs are device-resident."""
    from vibespatial.geometry.equality import geom_equals_identical_owned
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    geoms = [Point(0, 0), Point(1, 1), Point(2, 2)]
    left = from_shapely_geometries(geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(geoms, residency=Residency.DEVICE)

    clear_dispatch_events()
    result = geom_equals_identical_owned(left, right)
    events = get_dispatch_events(clear=True)

    np.testing.assert_array_equal(result, np.asarray([True, True, True], dtype=bool))
    assert any(
        event.surface == "geom_equals_exact" and event.selected is ExecutionMode.GPU
        for event in events
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@requires_gpu
def test_empty_array():
    """Empty arrays should return empty result."""
    left = _make_owned([])
    right = _make_owned([])

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right)
    assert result.shape == (0,)


@requires_gpu
def test_small_float_difference_not_identical():
    """Coordinate differences visible at fp32 precision should fail identity check.

    The GPU kernel runs in fp32 on consumer GPUs (ADR-0002), so the
    difference must be larger than fp32 epsilon (~1.19e-7) to be detected.
    We use 1e-5 which is safely above fp32 epsilon.
    """
    n = 1200
    left_geoms = [Point(1.0, 2.0) for _ in range(n)]
    right_geoms = [Point(1.0 + 1e-5, 2.0) for _ in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_identical_owned
    result = geom_equals_identical_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    # Both fp32 and fp64 should detect this difference
    assert not result.any()


# ---------------------------------------------------------------------------
# GeoPandas surface API dispatch
# ---------------------------------------------------------------------------

@requires_gpu
def test_geopandas_geoseries_dispatch():
    """GeoSeries.geom_equals_identical should route through GPU path."""
    import geopandas

    n = 1200
    geoms = [Point(i, i) for i in range(n)]
    gs = geopandas.GeoSeries(geoms)

    result = gs.geom_equals_identical(gs, align=False)
    expected = _shapely_equals_exact_tol0(geoms, geoms)
    np.testing.assert_array_equal(result.values, expected)
    assert result.all()


@requires_gpu
def test_geopandas_scalar_broadcast():
    """GeoSeries.geom_equals_identical with scalar should work."""
    import geopandas

    n = 1200
    pt = Point(5, 5)
    geoms = [Point(5, 5) if i % 2 == 0 else Point(i, i) for i in range(n)]
    gs = geopandas.GeoSeries(geoms)

    result = gs.geom_equals_identical(pt, align=False)
    # Even-indexed are identical, odd-indexed are not (except possibly i=5)
    assert result.iloc[0] is np.bool_(True)
    assert result.iloc[1] is np.bool_(False)


# ---------------------------------------------------------------------------
# Predicate dispatch wiring
# ---------------------------------------------------------------------------

@requires_gpu
def test_binary_predicate_dispatch_wiring():
    """equals_identical should be registered in supports_binary_predicate."""
    from vibespatial.predicates.binary import supports_binary_predicate
    assert supports_binary_predicate("equals_identical")
