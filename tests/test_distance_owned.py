"""Tests for on-device element-wise distance and dwithin.

Validates distance_owned and dwithin_owned against a Shapely oracle for all
geometry family combinations, null/empty handling, and the DGA/GeoSeries
integration surface.
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
from vibespatial.spatial.distance_owned import distance_owned, dwithin_owned

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shapely_distances(left_geoms, right_geoms):
    """Oracle: element-wise Shapely distance."""
    left_arr = np.asarray(left_geoms, dtype=object)
    right_arr = np.asarray(right_geoms, dtype=object)
    return np.asarray(shapely.distance(left_arr, right_arr), dtype=np.float64)


def _assert_distances_match(left_geoms, right_geoms, rtol=1e-10):
    """Assert distance_owned matches Shapely oracle."""
    left_owned = from_shapely_geometries(list(left_geoms))
    right_owned = from_shapely_geometries(list(right_geoms))
    got = distance_owned(left_owned, right_owned)
    expected = _shapely_distances(left_geoms, right_geoms)
    # NaN for null geometries
    nan_mask = np.isnan(expected)
    assert np.array_equal(np.isnan(got), nan_mask), (
        f"NaN mismatch: got NaN at {np.flatnonzero(np.isnan(got))}, "
        f"expected NaN at {np.flatnonzero(nan_mask)}"
    )
    valid = ~nan_mask
    if valid.any():
        np.testing.assert_allclose(got[valid], expected[valid], rtol=rtol)


# ---------------------------------------------------------------------------
# Point x Point
# ---------------------------------------------------------------------------

class TestPointPoint:
    def test_basic(self):
        left = [Point(0, 0), Point(1, 1), Point(3, 4)]
        right = [Point(1, 0), Point(1, 1), Point(0, 0)]
        _assert_distances_match(left, right)

    def test_coincident(self):
        left = [Point(5, 5), Point(-1, -1)]
        right = [Point(5, 5), Point(-1, -1)]
        _assert_distances_match(left, right)


# ---------------------------------------------------------------------------
# Point x LineString / Polygon
# ---------------------------------------------------------------------------

class TestPointGeometry:
    def test_point_linestring(self):
        left = [Point(0, 0), Point(0, 5), Point(2, 2)]
        right = [
            LineString([(1, 0), (1, 5)]),
            LineString([(0, 0), (0, 10)]),
            LineString([(0, 0), (4, 4)]),
        ]
        _assert_distances_match(left, right)

    def test_point_polygon(self):
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        left = [Point(1, 1), Point(3, 0), Point(-1, -1)]
        right = [square, square, square]
        _assert_distances_match(left, right)

    def test_point_multilinestring(self):
        mls = MultiLineString([[(0, 0), (1, 0)], [(2, 0), (3, 0)]])
        left = [Point(0.5, 1), Point(1.5, 1), Point(2.5, 1)]
        right = [mls, mls, mls]
        _assert_distances_match(left, right)

    def test_point_multipolygon(self):
        mpg = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
        ])
        left = [Point(0.5, 0.5), Point(2, 2), Point(5, 5)]
        right = [mpg, mpg, mpg]
        _assert_distances_match(left, right)


# ---------------------------------------------------------------------------
# Geometry x Point (reversed)
# ---------------------------------------------------------------------------

class TestGeometryPoint:
    def test_linestring_point(self):
        left = [
            LineString([(0, 0), (2, 0)]),
            LineString([(0, 0), (0, 5)]),
        ]
        right = [Point(1, 1), Point(3, 3)]
        _assert_distances_match(left, right)

    def test_polygon_point(self):
        square = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        left = [square, square]
        right = [Point(1, 1), Point(5, 5)]
        _assert_distances_match(left, right)


# ---------------------------------------------------------------------------
# Segment x Segment (non-point)
# ---------------------------------------------------------------------------

class TestSegmentSegment:
    def test_linestring_linestring(self):
        left = [
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 0), (0, 5)]),
        ]
        right = [
            LineString([(0, 1), (1, 1)]),
            LineString([(3, 0), (3, 5)]),
        ]
        _assert_distances_match(left, right)

    def test_polygon_polygon(self):
        sq1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        sq2 = Polygon([(3, 3), (4, 3), (4, 4), (3, 4)])
        sq3 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
        left = [sq1, sq1, sq2]
        right = [sq2, sq3, sq3]
        _assert_distances_match(left, right)

    def test_linestring_polygon(self):
        ls = LineString([(0, 0), (1, 0)])
        pg = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        _assert_distances_match([ls], [pg])

    def test_multipolygon_multipolygon(self):
        mpg1 = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ])
        mpg2 = MultiPolygon([
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            Polygon([(7, 7), (8, 7), (8, 8), (7, 8)]),
        ])
        _assert_distances_match([mpg1], [mpg2])


# ---------------------------------------------------------------------------
# MultiPoint
# ---------------------------------------------------------------------------

class TestMultiPoint:
    def test_multipoint_point(self):
        mp = MultiPoint([(0, 0), (3, 0), (6, 0)])
        left = [mp, mp]
        right = [Point(1, 0), Point(4, 0)]
        _assert_distances_match(left, right)

    def test_multipoint_polygon(self):
        mp = MultiPoint([(0, 0), (5, 5)])
        pg = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        _assert_distances_match([mp], [pg])

    def test_multipoint_multipoint(self):
        mp1 = MultiPoint([(0, 0), (1, 1)])
        mp2 = MultiPoint([(3, 3), (4, 4)])
        _assert_distances_match([mp1], [mp2])


# ---------------------------------------------------------------------------
# Mixed-family arrays
# ---------------------------------------------------------------------------

class TestMixedFamily:
    def test_mixed_left_right(self):
        left = [
            Point(0, 0),
            LineString([(0, 0), (1, 0)]),
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
        ]
        right = [
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            Point(0.5, 1),
            LineString([(5, 5), (6, 6)]),
        ]
        _assert_distances_match(left, right)


# ---------------------------------------------------------------------------
# Null and empty geometry
# ---------------------------------------------------------------------------

class TestNullEmpty:
    def test_null_returns_nan(self):
        left = [Point(0, 0), None, Point(1, 1)]
        right = [Point(1, 0), Point(2, 2), None]
        left_owned = from_shapely_geometries(left)
        right_owned = from_shapely_geometries(right)
        result = distance_owned(left_owned, right_owned)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        np.testing.assert_allclose(result[0], 1.0, rtol=1e-10)

    def test_both_null(self):
        left = [None, None]
        right = [None, None]
        left_owned = from_shapely_geometries(left)
        right_owned = from_shapely_geometries(right)
        result = distance_owned(left_owned, right_owned)
        assert np.all(np.isnan(result))

    def test_empty_array(self):
        left_owned = from_shapely_geometries([])
        right_owned = from_shapely_geometries([])
        result = distance_owned(left_owned, right_owned)
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_length_mismatch_raises(self):
        left_owned = from_shapely_geometries([Point(0, 0)])
        right_owned = from_shapely_geometries([Point(1, 1), Point(2, 2)])
        with pytest.raises(ValueError, match="Incompatible lengths"):
            distance_owned(left_owned, right_owned)


# ---------------------------------------------------------------------------
# dwithin
# ---------------------------------------------------------------------------

class TestDwithin:
    def test_basic(self):
        left = [Point(0, 0), Point(0, 0), Point(0, 0)]
        right = [Point(1, 0), Point(3, 0), Point(0.5, 0)]
        left_owned = from_shapely_geometries(left)
        right_owned = from_shapely_geometries(right)
        result = dwithin_owned(left_owned, right_owned, 1.5)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_null_returns_false(self):
        left = [Point(0, 0), None]
        right = [Point(0, 0), Point(0, 0)]
        left_owned = from_shapely_geometries(left)
        right_owned = from_shapely_geometries(right)
        result = dwithin_owned(left_owned, right_owned, 100.0)
        np.testing.assert_array_equal(result, [True, False])

    def test_per_row_threshold(self):
        left = [Point(0, 0), Point(0, 0), Point(0, 0)]
        right = [Point(1, 0), Point(2, 0), Point(3, 0)]
        left_owned = from_shapely_geometries(left)
        right_owned = from_shapely_geometries(right)
        thresholds = np.array([0.5, 2.5, 2.5])
        result = dwithin_owned(left_owned, right_owned, thresholds)
        np.testing.assert_array_equal(result, [False, True, False])


# ---------------------------------------------------------------------------
# Integration: DeviceGeometryArray
# ---------------------------------------------------------------------------

class TestDGAIntegration:
    def test_dga_distance(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        left_geoms = [Point(0, 0), Point(1, 1), Point(3, 4)]
        right_geoms = [Point(1, 0), Point(1, 1), Point(0, 0)]
        dga_left = DeviceGeometryArray._from_sequence(left_geoms)
        dga_right = DeviceGeometryArray._from_sequence(right_geoms)
        result = dga_left.distance(dga_right)
        expected = _shapely_distances(left_geoms, right_geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dga_distance_single_geometry(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        geoms = [Point(0, 0), Point(3, 4)]
        dga = DeviceGeometryArray._from_sequence(geoms)
        target = Point(0, 0)
        result = dga.distance(target)
        expected = np.array([0.0, 5.0])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dga_dwithin(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        geoms = [Point(0, 0), Point(0, 0)]
        others = [Point(1, 0), Point(5, 0)]
        dga_left = DeviceGeometryArray._from_sequence(geoms)
        dga_right = DeviceGeometryArray._from_sequence(others)
        result = dga_left.dwithin(dga_right, 2.0)
        np.testing.assert_array_equal(result, [True, False])

    def test_dga_distance_numpy_array(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        left_geoms = [Point(0, 0), Point(1, 1)]
        right_geoms = np.array([Point(1, 0), Point(4, 5)], dtype=object)
        dga = DeviceGeometryArray._from_sequence(left_geoms)
        result = dga.distance(right_geoms)
        expected = _shapely_distances(left_geoms, right_geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_dga_distance_mixed_families(self):
        from vibespatial.geometry.device_array import DeviceGeometryArray
        left_geoms = [Point(0, 0), LineString([(0, 0), (1, 0)])]
        right_geoms = [
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
            Point(0.5, 1),
        ]
        dga_left = DeviceGeometryArray._from_sequence(left_geoms)
        dga_right = DeviceGeometryArray._from_sequence(right_geoms)
        result = dga_left.distance(dga_right)
        expected = _shapely_distances(left_geoms, right_geoms)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Integration: GeoSeries
# ---------------------------------------------------------------------------

class TestGeoSeriesIntegration:
    def test_geoseries_distance(self):
        import pandas as pd

        from vibespatial.api.geoseries import GeoSeries
        left = GeoSeries([Point(0, 0), Point(3, 4)])
        right = GeoSeries([Point(1, 0), Point(0, 0)])
        result = left.distance(right, align=False)
        expected = pd.Series([1.0, 5.0])
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)

    def test_geoseries_dwithin(self):
        import pandas as pd

        from vibespatial.api.geoseries import GeoSeries
        left = GeoSeries([Point(0, 0), Point(0, 0)])
        right = GeoSeries([Point(1, 0), Point(5, 0)])
        result = left.dwithin(right, distance=2.0, align=False)
        expected = pd.Series([True, False])
        pd.testing.assert_series_equal(result, expected)

    def test_geoseries_distance_single_geom(self):
        import pandas as pd

        from vibespatial.api.geoseries import GeoSeries
        gs = GeoSeries([Point(0, 0), Point(3, 4)])
        result = gs.distance(Point(0, 0), align=False)
        expected = pd.Series([0.0, 5.0])
        pd.testing.assert_series_equal(result, expected, rtol=1e-10)
