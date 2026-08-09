"""Tests for GPU topological equality (geom_equals).

Verifies correctness against Shapely oracle for all geometry families,
null handling, mixed-family arrays, rotated rings, reversed linestrings,
and the GeoPandas surface API dispatch.

geom_equals evaluates mutual topological coverage with explicit empty-set
handling. Two geometries are topologically equal if their boundaries,
interiors, and exteriors coincide, regardless of coordinate ordering,
redundant vertices, or multi-part representation.
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
    box,
)

import vibespatial
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.residency import Residency
from vibespatial.testing import (
    build_owned as _make_owned,
)
from vibespatial.testing import (
    strict_native_environment,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")


def _shapely_equals(left_geoms, right_geoms):
    """Shapely oracle: topological equality."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray(right_geoms, dtype=object)
    return shapely.equals(left_arr, right_arr)


# ---------------------------------------------------------------------------
# Point family
# ---------------------------------------------------------------------------


@requires_gpu
def test_identical_points(make_owned):
    """Identical points should return True."""
    n = 1200
    geoms = [Point(i * 0.1, i * 0.2) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_different_points(make_owned):
    """Points with different coordinates should return False."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [Point(i + 0.001, i) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# LineString family
# ---------------------------------------------------------------------------


@requires_gpu
def test_identical_linestrings(make_owned):
    """Identical linestrings should return True."""
    n = 1200
    geoms = [LineString([(i, 0), (i + 1, 1), (i + 2, 0)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_reversed_linestrings(make_owned):
    """Reversed linestrings are topologically equal.

    shapely.equals returns True for reversed linestrings because they
    represent the same point set.  After normalization, both directions
    should produce the same canonical form.
    """
    n = 1200
    left_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    right_geoms = [LineString([(i + 1, 1), (i, 0)]) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)


@requires_gpu
def test_different_linestrings(make_owned):
    """LineStrings with different coordinates should return False."""
    n = 1200
    left_geoms = [LineString([(i, 0), (i + 1, 1)]) for i in range(n)]
    right_geoms = [LineString([(i, 0), (i + 2, 2)]) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Polygon family
# ---------------------------------------------------------------------------


@requires_gpu
def test_identical_polygons(make_owned):
    """Identical polygons should return True."""
    n = 1200
    geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(n)]
    left = make_owned(geoms)
    right = make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_rotated_ring_polygons(make_owned):
    """Polygons with rotated ring start vertex are topologically equal.

    The same polygon with different ring start points should be equal
    after normalization (ring rotation to lex-min vertex).
    """
    n = 1200
    left_geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(n)]
    # Same polygon, but ring starts at a different vertex
    right_geoms = [Polygon([(i + 1, 0), (i + 1, 1), (i, 1), (i, 0), (i + 1, 0)]) for i in range(n)]
    left = make_owned(left_geoms)
    right = make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)


@requires_gpu
def test_polygons_with_holes(make_owned):
    """Polygons with holes should compare correctly."""
    n = 1200
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    geoms_with_hole = [Polygon(outer, [hole]) for _ in range(n)]
    left = make_owned(geoms_with_hole)
    right = make_owned(geoms_with_hole)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms_with_hole, geoms_with_hole)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_polygons_with_vs_without_holes():
    """Polygons with vs without holes should return False."""
    n = 1200
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    left_geoms = [Polygon(outer, [hole]) for _ in range(n)]
    right_geoms = [Polygon(outer) for _ in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Multi* families
# ---------------------------------------------------------------------------


@requires_gpu
def test_identical_multipoints():
    """Identical MultiPoints should return True."""
    n = 1200
    geoms = [MultiPoint([(i, 0), (i + 1, 1)]) for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_identical_multilinestrings():
    """Identical MultiLineStrings should return True."""
    n = 1200
    geoms = [MultiLineString([[(i, 0), (i + 1, 1)], [(i + 2, 0), (i + 3, 1)]]) for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_identical_multipolygons():
    """Identical MultiPolygons should return True."""
    n = 1200
    geoms = [
        MultiPolygon(
            [
                ([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)], []),
                ([(i + 2, 0), (i + 3, 0), (i + 3, 1), (i + 2, 1), (i + 2, 0)], []),
            ]
        )
        for i in range(n)
    ]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


@requires_gpu
def test_topological_equality_ignores_redundant_vertices_and_part_order():
    """Native equality compares point sets, not normalized buffer structure."""
    left_geoms = [
        LineString([(0, 0), (2, 0)]),
        LineString([(3, 0), (4, 1), (5, 0)]),
        MultiPoint([(0, 0), (1, 1)]),
        MultiPolygon([box(0, 0, 1, 1), box(2, 2, 3, 3)]),
    ]
    right_geoms = [
        LineString([(0, 0), (1, 0), (2, 0)]),
        MultiLineString([[(3, 0), (4, 1), (5, 0)]]),
        MultiPoint([(1, 1), (0, 0)]),
        MultiPolygon([box(2, 2, 3, 3), box(0, 0, 1, 1)]),
    ]

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(
        _make_owned(left_geoms),
        _make_owned(right_geoms),
        dispatch_mode=ExecutionMode.GPU,
    )
    np.testing.assert_array_equal(result, _shapely_equals(left_geoms, right_geoms))
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

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)

    # Null positions must be False
    for i in range(n):
        if geoms[i] is None:
            assert not result[i], f"Null at index {i} should be False"


@requires_gpu
def test_null_vs_non_null():
    """Null vs non-null should return False."""
    n = 1200
    left_geoms = [Point(i, i) for i in range(n)]
    right_geoms = [None if i % 2 == 0 else Point(i, i) for i in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)

    # Even indices (null right) should be False
    for i in range(0, n, 2):
        assert not result[i], f"Null-vs-non-null at index {i} should be False"


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------


@requires_gpu
def test_mixed_families_equal():
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

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(geoms, geoms)
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

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right, dispatch_mode=ExecutionMode.GPU)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)
    assert not result.any()


# ---------------------------------------------------------------------------
# Row count validation
# ---------------------------------------------------------------------------


def test_mismatched_row_count_raises():
    """Different row counts should raise ValueError."""
    left = _make_owned([Point(0, 0), Point(1, 1)])
    right = _make_owned([Point(0, 0)])

    from vibespatial.geometry.equality import geom_equals_owned

    with pytest.raises(ValueError, match="same row count"):
        geom_equals_owned(left, right)


# ---------------------------------------------------------------------------
# CPU fallback
# ---------------------------------------------------------------------------


def test_cpu_fallback():
    """CPU path should match Shapely output."""
    n = 100  # below GPU threshold
    geoms = [Point(i, i) for i in range(n)]
    left = _make_owned(geoms)
    right = _make_owned(geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result, expected)
    assert result.all()


def test_cpu_fallback_rotated_rings():
    """CPU fallback should handle rotated polygon rings correctly."""
    n = 100
    left_geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(n)]
    right_geoms = [Polygon([(i + 1, 0), (i + 1, 1), (i, 1), (i, 0), (i + 1, 0)]) for i in range(n)]
    left = _make_owned(left_geoms)
    right = _make_owned(right_geoms)

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@requires_gpu
def test_empty_array():
    """Empty arrays should return empty result."""
    left = _make_owned([])
    right = _make_owned([])

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(left, right)
    assert result.shape == (0,)
    assert result.dtype == bool


@requires_gpu
def test_empty_geometries_are_equal_across_supported_families():
    """All valid empty geometries represent the same empty point set."""
    left_geoms = [Point(), LineString(), Polygon()]
    right_geoms = [Polygon(), Point(), MultiPoint([])]

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(
        _make_owned(left_geoms),
        _make_owned(right_geoms),
        dispatch_mode=ExecutionMode.GPU,
    )
    np.testing.assert_array_equal(result, np.ones(3, dtype=bool))


# ---------------------------------------------------------------------------
# GeoPandas surface API dispatch
# ---------------------------------------------------------------------------


@requires_gpu
def test_geopandas_geoseries_dispatch():
    """GeoSeries.geom_equals should route through native mutual coverage."""
    import geopandas

    n = 1200
    geoms = [Point(i, i) for i in range(n)]
    gs = geopandas.GeoSeries(geoms)

    result = gs.geom_equals(gs, align=False)
    expected = _shapely_equals(geoms, geoms)
    np.testing.assert_array_equal(result.values, expected)
    assert result.all()


@requires_gpu
def test_geopandas_scalar_broadcast():
    """GeoSeries.geom_equals with scalar should work."""
    import geopandas

    n = 1200
    pt = Point(5, 5)
    geoms = [Point(5, 5) if i % 2 == 0 else Point(i, i) for i in range(n)]
    gs = geopandas.GeoSeries(geoms)

    result = gs.geom_equals(pt, align=False)
    # Even-indexed are equal, odd-indexed are not (except possibly i=5)
    assert result.iloc[0] is np.bool_(True)
    assert result.iloc[1] is np.bool_(False)


@requires_gpu
def test_geopandas_scalar_broadcast_strict_native_stays_on_owned_path():
    """Strict scalar broadcast should not escape to the Shapely fallback path."""
    import geopandas

    n = 1200
    pt = Point(5, 5)
    geoms = [Point(5, 5) if i % 2 == 0 else Point(i, i) for i in range(n)]
    gs = geopandas.GeoSeries(geoms)

    with strict_native_environment():
        vibespatial.clear_dispatch_events()
        result = gs.geom_equals(pt, align=False)
        events = vibespatial.get_dispatch_events(clear=True)

    assert result.iloc[0] is np.bool_(True)
    assert result.iloc[1] is np.bool_(False)
    assert any(
        event.surface == "geopandas.array.equals"
        and event.implementation == "geom_equals_owned_broadcast"
        for event in events
    )
    assert not any(
        event.surface == "geopandas.array.equals"
        and event.implementation == "shapely_scalar_broadcast"
        for event in events
    )


@requires_gpu
def test_geopandas_rotated_rings():
    """GeoSeries.geom_equals with rotated ring start should be True."""
    import geopandas

    n = 1200
    left_geoms = [Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1), (i, 0)]) for i in range(n)]
    right_geoms = [Polygon([(i + 1, 0), (i + 1, 1), (i, 1), (i, 0), (i + 1, 0)]) for i in range(n)]
    gs_left = geopandas.GeoSeries(left_geoms)
    gs_right = geopandas.GeoSeries(right_geoms)

    result = gs_left.geom_equals(gs_right, align=False)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result.values, expected)


@requires_gpu
def test_gpu_normalize_fuses_ring_start_selection_with_rotation():
    """Canonical ring starts must not depend on an intermediate launch."""
    from vibespatial.constructive.normalize import normalize_owned
    from vibespatial.constructive.normalize_kernels import _RING_KERNEL_NAMES

    base = [
        (-1.0, 0.0),
        (-0.70710678118655, 0.707106781186545),
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-0.707106781186546, -0.707106781186549),
    ]
    geometries = []
    for row in range(256):
        shift = row % len(base)
        coordinates = base[shift:] + base[:shift]
        if row % 2:
            coordinates = list(reversed(coordinates))
        geometries.append(Polygon([*coordinates, coordinates[0]]))

    owned = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    normalized = normalize_owned(owned, dispatch_mode=ExecutionMode.GPU)
    actual = np.asarray(normalized.to_shapely(), dtype=object)
    expected = shapely.normalize(np.asarray(geometries, dtype=object))

    assert _RING_KERNEL_NAMES == ("normalize_ring_rotate",)
    assert shapely.equals_exact(actual, expected, tolerance=0.0).all()


@requires_gpu
def test_indexed_multipolygon_equality_survives_terminal_host_materialization():
    """Indexed device rows retain correct normalization after host export."""
    cp = pytest.importorskip("cupy")
    first = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])
    second = Polygon([(4, 0), (6, 0), (6, 2), (4, 2), (4, 0)])
    rotated_first = Polygon([(2, 0), (2, 2), (0, 2), (0, 0), (2, 0)])
    rotated_second = Polygon([(6, 0), (6, 2), (4, 2), (4, 0), (6, 0)])

    root = from_shapely_geometries(
        [MultiPolygon([first, second]), MultiPolygon([second, first])],
        residency=Residency.DEVICE,
    )
    indexed = root._device_indexed_take(
        cp.asarray([1, 0], dtype=cp.int64),
        assume_unique_indices=True,
    )
    expected = from_shapely_geometries(
        [
            MultiPolygon([rotated_second, rotated_first]),
            MultiPolygon([rotated_first, rotated_second]),
        ],
        residency=Residency.DEVICE,
    )

    # A terminal export may populate compact host family buffers while the
    # native indexed carrier remains rooted in the original device buffers.
    indexed.to_shapely()

    from vibespatial.geometry.equality import geom_equals_owned

    result = geom_equals_owned(
        indexed,
        expected,
        dispatch_mode=ExecutionMode.GPU,
    )
    np.testing.assert_array_equal(result, np.ones(2, dtype=bool))


@requires_gpu
def test_geopandas_polygon_and_singlepart_multipolygon_are_topologically_equal():
    """GeoSeries.geom_equals should treat single-part multi families as equal."""
    import geopandas

    left_geoms = [
        MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]),
    ]
    right_geoms = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        MultiPolygon([Polygon([(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)])]),
    ]
    gs_left = geopandas.GeoSeries(left_geoms)
    gs_right = geopandas.GeoSeries(right_geoms)

    result = gs_left.geom_equals(gs_right, align=False)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result.values, expected)
    assert result.all()


def test_geopandas_multilinestring_z_geom_equals_ignores_extra_ordinates():
    """GeoSeries.geom_equals should not crash on 3D multiline inputs."""
    import geopandas

    left_geoms = [
        MultiLineString([[(30, 10, 40), (10, 30, 40), (40, 40, 80)]]),
        MultiLineString([[(0, 0, 1), (1, 1, 2)]]),
    ]
    right_geoms = [
        MultiLineString([[(30, 10, 5), (10, 30, 6), (40, 40, 7)]]),
        MultiLineString([[(0, 0, 3), (1, 1, 4)]]),
    ]
    gs_left = geopandas.GeoSeries(left_geoms)
    gs_right = geopandas.GeoSeries(right_geoms)

    result = gs_left.geom_equals(gs_right, align=False)
    expected = _shapely_equals(left_geoms, right_geoms)
    np.testing.assert_array_equal(result.values, expected)
    assert result.all()


# ---------------------------------------------------------------------------
# Predicate dispatch wiring
# ---------------------------------------------------------------------------


@requires_gpu
def test_binary_predicate_dispatch_wiring():
    """equals should be registered in supports_binary_predicate."""
    from vibespatial.predicates.binary import supports_binary_predicate

    assert supports_binary_predicate("equals")


@requires_gpu
def test_special_predicate_registration():
    """equals should be in _SPECIAL_PREDICATES."""
    from vibespatial.predicates.binary import _SPECIAL_PREDICATES

    assert "equals" in _SPECIAL_PREDICATES
