"""Tests for device-native GPU paths in properties.py.

Verifies that CuPy (Tier 2) and NVRTC (Tier 1) device paths produce
the same results as Shapely oracles and host-path fallbacks for all
geometry property functions.
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

from vibespatial.constructive.properties import (
    get_x_owned,
    get_y_owned,
    is_ccw_owned,
    is_closed_owned,
    num_coordinates_owned,
    num_geometries_owned,
    num_interior_rings_owned,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.runtime import has_gpu_runtime

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_owned_with_device_state(geoms):
    """Create an OwnedGeometryArray and upload to device if GPU available."""
    owned = from_shapely_geometries(geoms)
    if has_gpu_runtime():
        owned._ensure_device_state()
    return owned


def _shapely_num_coordinates(geoms):
    return np.array([shapely.get_num_coordinates(g) for g in geoms], dtype=np.int32)


def _shapely_num_geometries(geoms):
    return np.array([shapely.get_num_geometries(g) for g in geoms], dtype=np.int32)


def _shapely_num_interior_rings(geoms):
    return np.array([shapely.get_num_interior_rings(g) for g in geoms], dtype=np.int32)


def _shapely_is_closed(geoms):
    return np.array([bool(shapely.is_closed(g)) for g in geoms], dtype=bool)


def _expected_is_ccw(geoms):
    """Compute expected is_ccw by checking exterior ring winding.

    Unlike shapely.is_ccw(polygon) which always returns False for Polygon
    types, our implementation checks the actual exterior ring winding.
    """
    result = []
    for g in geoms:
        if isinstance(g, Polygon):
            result.append(bool(shapely.is_ccw(g.exterior)))
        elif isinstance(g, MultiPolygon):
            # Check first polygon's exterior ring
            result.append(bool(shapely.is_ccw(g.geoms[0].exterior)))
        else:
            result.append(False)
    return np.array(result, dtype=bool)


# ---------------------------------------------------------------------------
# num_coordinates tests
# ---------------------------------------------------------------------------


class TestNumCoordinates:
    def test_points(self):
        geoms = [Point(1, 2), Point(3, 4), Point(5, 6)]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_linestrings(self):
        geoms = [
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (5, 5)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_polygons(self):
        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                [[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
            ),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multilinestrings(self):
        geoms = [
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3), (4, 4)]]),
            MultiLineString([[(0, 0), (1, 0), (1, 1)]]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multipolygons(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        geoms = [MultiPolygon([p1, p2])]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_points_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Point(1, 2), Point(3, 4), Point(5, 6)]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_linestrings_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (5, 5)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_polygons_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                [[(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]],
            ),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multilinestrings_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3), (4, 4)]]),
            MultiLineString([[(0, 0), (1, 0), (1, 1)]]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multipolygons_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        geoms = [MultiPolygon([p1, p2])]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# num_geometries tests
# ---------------------------------------------------------------------------


class TestNumGeometries:
    def test_simple_types(self):
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_geometries_owned(owned)
        expected = _shapely_num_geometries(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multi_types(self):
        geoms = [
            MultiPoint([(0, 0), (1, 1), (2, 2)]),
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_geometries_owned(owned)
        expected = _shapely_num_geometries(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multi_types_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            MultiPoint([(0, 0), (1, 1), (2, 2)]),
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_geometries_owned(owned)
        expected = _shapely_num_geometries(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_simple_types_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_geometries_owned(owned)
        expected = _shapely_num_geometries(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# num_interior_rings tests
# ---------------------------------------------------------------------------


class TestNumInteriorRings:
    def test_polygon_no_holes(self):
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = num_interior_rings_owned(owned)
        expected = _shapely_num_interior_rings(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_polygon_with_holes(self):
        geoms = [
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                [
                    [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
                    [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
                ],
            ),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_interior_rings_owned(owned)
        expected = _shapely_num_interior_rings(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_non_polygon_types(self):
        geoms = [Point(0, 0), LineString([(0, 0), (1, 1)])]
        owned = from_shapely_geometries(geoms)
        result = num_interior_rings_owned(owned)
        expected = _shapely_num_interior_rings(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_polygon_with_holes_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Polygon(
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                [
                    [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)],
                    [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
                ],
            ),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_interior_rings_owned(owned)
        expected = _shapely_num_interior_rings(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# is_closed tests
# ---------------------------------------------------------------------------


class TestIsClosed:
    def test_closed_linestring(self):
        geoms = [LineString([(0, 0), (1, 1), (2, 0), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_open_linestring(self):
        geoms = [LineString([(0, 0), (1, 1), (2, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_linestrings(self):
        geoms = [
            LineString([(0, 0), (1, 1), (2, 0), (0, 0)]),  # closed
            LineString([(0, 0), (1, 1), (2, 0)]),  # open
            LineString([(5, 5), (6, 6)]),  # open
        ]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multilinestring_all_closed(self):
        geoms = [
            MultiLineString([
                [(0, 0), (1, 1), (2, 0), (0, 0)],
                [(3, 3), (4, 4), (5, 3), (3, 3)],
            ]),
        ]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multilinestring_one_open(self):
        geoms = [
            MultiLineString([
                [(0, 0), (1, 1), (2, 0), (0, 0)],  # closed
                [(3, 3), (4, 4), (5, 3)],  # open
            ]),
        ]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_points_always_closed(self):
        geoms = [Point(0, 0), Point(1, 1)]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        # Points are treated as closed by the owned implementation
        assert all(result)

    def test_polygons_always_closed(self):
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        # Polygons are treated as closed by the owned implementation
        # (Shapely returns False for non-linear types, but rings are closed)
        assert all(result)

    @pytest.mark.gpu
    def test_linestrings_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            LineString([(0, 0), (1, 1), (2, 0), (0, 0)]),  # closed
            LineString([(0, 0), (1, 1), (2, 0)]),  # open
            LineString([(5, 5), (6, 6)]),  # open
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multilinestrings_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            MultiLineString([
                [(0, 0), (1, 1), (2, 0), (0, 0)],
                [(3, 3), (4, 4), (5, 3), (3, 3)],
            ]),
            MultiLineString([
                [(0, 0), (1, 1), (2, 0), (0, 0)],
                [(3, 3), (4, 4), (5, 3)],  # open
            ]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_closed_owned(owned)
        expected = _shapely_is_closed(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# is_ccw tests
# ---------------------------------------------------------------------------


class TestIsCCW:
    def test_ccw_polygon(self):
        # CCW exterior ring
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_cw_polygon(self):
        # CW exterior ring
        geoms = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_mixed_polygons(self):
        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # CCW
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),  # CW
        ]
        owned = from_shapely_geometries(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multipolygon_ccw(self):
        # CCW exterior rings
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        geoms = [MultiPolygon([p1, p2])]
        owned = from_shapely_geometries(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_non_polygon_types_false(self):
        geoms = [Point(0, 0), LineString([(0, 0), (1, 1)])]
        owned = from_shapely_geometries(geoms)
        result = is_ccw_owned(owned)
        # Non-polygon types return False
        assert not any(result)

    @pytest.mark.gpu
    def test_polygons_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # CCW
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]),  # CW
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multipolygons_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        geoms = [MultiPolygon([p1, p2])]
        owned = _make_owned_with_device_state(geoms)
        result = is_ccw_owned(owned)
        expected = _expected_is_ccw(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Mixed type tests
# ---------------------------------------------------------------------------


class TestMixedTypes:
    """Test properties across mixed geometry arrays."""

    def test_num_coordinates_mixed(self):
        geoms = [
            Point(1, 2),
            LineString([(0, 0), (1, 1), (2, 2)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            MultiPoint([(0, 0), (1, 1)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_num_geometries_mixed(self):
        geoms = [
            Point(1, 2),
            MultiPoint([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (1, 1)]),
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        ]
        owned = from_shapely_geometries(geoms)
        result = num_geometries_owned(owned)
        expected = _shapely_num_geometries(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_is_closed_mixed(self):
        geoms = [
            Point(1, 2),
            LineString([(0, 0), (1, 1), (0, 0)]),  # closed
            LineString([(0, 0), (1, 1)]),  # open
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # closed
        ]
        owned = from_shapely_geometries(geoms)
        result = is_closed_owned(owned)
        # Point=True, closed LS=True, open LS=False, Polygon=True
        # (differs from shapely which returns False for Point/Polygon)
        expected = np.array([True, True, False, True], dtype=bool)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_num_coordinates_mixed_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Point(1, 2),
            LineString([(0, 0), (1, 1), (2, 2)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            MultiPoint([(0, 0), (1, 1)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = num_coordinates_owned(owned)
        expected = _shapely_num_coordinates(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_is_closed_mixed_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Point(1, 2),
            LineString([(0, 0), (1, 1), (0, 0)]),  # closed
            LineString([(0, 0), (1, 1)]),  # open
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),  # closed
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_closed_owned(owned)
        # Point=True, closed LS=True, open LS=False, Polygon=True
        expected = np.array([True, True, False, True], dtype=bool)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# get_x / get_y tests
# ---------------------------------------------------------------------------


class TestGetX:
    def test_get_x_points(self):
        geoms = [Point(1.5, 2.5), Point(-3.0, 4.0), Point(0.0, 0.0)]
        owned = from_shapely_geometries(geoms)
        result = get_x_owned(owned)
        expected = shapely.get_x(np.array(geoms))
        np.testing.assert_array_equal(result, expected)

    def test_get_x_non_point_returns_nan(self):
        geoms = [
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = get_x_owned(owned)
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    @pytest.mark.gpu
    def test_get_x_points_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Point(1.5, 2.5), Point(-3.0, 4.0), Point(0.0, 0.0)]
        owned = _make_owned_with_device_state(geoms)
        result = get_x_owned(owned)
        expected = shapely.get_x(np.array(geoms))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_get_x_50k_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        rng = np.random.default_rng(42)
        coords = rng.uniform(-180, 180, size=(50_000, 2))
        geoms = [Point(x, y) for x, y in coords]
        owned = _make_owned_with_device_state(geoms)
        result = get_x_owned(owned)
        expected = shapely.get_coordinates(np.array(geoms))[:, 0]
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)


class TestGetY:
    def test_get_y_points(self):
        geoms = [Point(1.5, 2.5), Point(-3.0, 4.0), Point(0.0, 0.0)]
        owned = from_shapely_geometries(geoms)
        result = get_y_owned(owned)
        expected = shapely.get_y(np.array(geoms))
        np.testing.assert_array_equal(result, expected)

    def test_get_y_non_point_returns_nan(self):
        geoms = [
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        ]
        owned = from_shapely_geometries(geoms)
        result = get_y_owned(owned)
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    @pytest.mark.gpu
    def test_get_y_points_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Point(1.5, 2.5), Point(-3.0, 4.0), Point(0.0, 0.0)]
        owned = _make_owned_with_device_state(geoms)
        result = get_y_owned(owned)
        expected = shapely.get_y(np.array(geoms))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_get_y_50k_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        rng = np.random.default_rng(42)
        coords = rng.uniform(-180, 180, size=(50_000, 2))
        geoms = [Point(x, y) for x, y in coords]
        owned = _make_owned_with_device_state(geoms)
        result = get_y_owned(owned)
        expected = shapely.get_coordinates(np.array(geoms))[:, 1]
        np.testing.assert_allclose(result, expected, rtol=0, atol=0)


class TestGetXYNoHostMaterialization:
    """Verify get_x/get_y on device-resident data do not trigger host materialization."""

    @pytest.mark.gpu
    def test_get_xy_no_host_materialization(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Point(1.0, 2.0), Point(3.0, 4.0), Point(5.0, 6.0)]
        owned = _make_owned_with_device_state(geoms)
        # device_take with all indices produces host_materialized=False stubs
        import cupy as cp

        device_owned = owned.device_take(cp.arange(len(geoms), dtype=cp.int64))
        assert GeometryFamily.POINT in device_owned.families
        assert not device_owned.families[GeometryFamily.POINT].host_materialized

        result_x = get_x_owned(device_owned)
        result_y = get_y_owned(device_owned)

        # Verify correctness
        np.testing.assert_array_equal(result_x, np.array([1.0, 3.0, 5.0]))
        np.testing.assert_array_equal(result_y, np.array([2.0, 4.0, 6.0]))

        # Verify host buffers were NOT materialized
        assert not device_owned.families[GeometryFamily.POINT].host_materialized
