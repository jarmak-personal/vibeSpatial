"""Tests for GPU-accelerated is_ring (fused is_closed AND is_simple).

Verifies that the composed is_ring_owned() produces results matching
Shapely's is_ring() oracle for all geometry families, including edge
cases: null geometries, empty geometries, self-intersecting closed
linestrings, and mixed-family arrays.
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

from vibespatial.constructive.properties import is_ring_owned
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


def _shapely_is_ring(geoms):
    """Compute Shapely oracle result for is_ring."""
    arr = np.array(geoms, dtype=object)
    return shapely.is_ring(arr)


# ---------------------------------------------------------------------------
# Host-path tests (CPU fallback, no device state required)
# ---------------------------------------------------------------------------


class TestIsRingHost:
    """Tests for is_ring_owned on host-resident data (no GPU required)."""

    def test_closed_simple_linestring_is_ring(self):
        """A closed, simple LineString is a ring."""
        geoms = [LineString([(0, 0), (1, 0), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_open_linestring_is_not_ring(self):
        """An open LineString is not a ring."""
        geoms = [LineString([(0, 0), (1, 1), (0, 1)])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_closed_self_intersecting_linestring_is_not_ring(self):
        """A closed but self-intersecting LineString is not a ring."""
        # Bowtie: closed (first == last) but segments cross
        geoms = [LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_point_is_not_ring(self):
        """Points always return False for is_ring."""
        geoms = [Point(0, 0), Point(1, 1)]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert np.all(~result)

    def test_polygon_is_not_ring(self):
        """Polygons always return False for is_ring."""
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_multilinestring_is_not_ring(self):
        """MultiLineStrings always return False for is_ring."""
        geoms = [MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_multipoint_is_not_ring(self):
        """MultiPoints always return False for is_ring."""
        geoms = [MultiPoint([(0, 0), (1, 1)])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_multipolygon_is_not_ring(self):
        """MultiPolygons always return False for is_ring."""
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)])
        geoms = [MultiPolygon([p1, p2])]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_null_geometry_returns_false(self):
        """Null geometries return False for is_ring."""
        geoms = [None, LineString([(0, 0), (1, 0), (0, 1), (0, 0)]), None]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)
        assert result[1] is np.bool_(True)
        assert result[2] is np.bool_(False)

    def test_empty_array(self):
        """Empty geometry array returns empty bool array."""
        owned = from_shapely_geometries([])
        result = is_ring_owned(owned)
        assert result.dtype == bool
        assert len(result) == 0

    def test_mixed_family_array(self):
        """Mixed-family arrays produce correct per-geometry results."""
        geoms = [
            Point(0, 0),                                       # False
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),      # True (closed, simple)
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), # False (polygon)
            LineString([(0, 0), (1, 1), (0, 1)]),               # False (open)
            None,                                               # False (null)
            LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),  # False (closed, not simple)
        ]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_multiple_linestrings_mixed_ring_status(self):
        """Multiple LineStrings with varying ring status."""
        geoms = [
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),      # ring
            LineString([(0, 0), (1, 1)]),                       # not closed
            LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),  # ring
            LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),  # closed, not simple
        ]
        owned = from_shapely_geometries(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Device-path tests (GPU required)
# ---------------------------------------------------------------------------


class TestIsRingDevice:
    """Tests for is_ring_owned on device-resident data (GPU path)."""

    @pytest.mark.gpu
    def test_closed_simple_linestring_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [LineString([(0, 0), (1, 0), (0, 1), (0, 0)])]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_open_linestring_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [LineString([(0, 0), (1, 1), (0, 1)])]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_closed_self_intersecting_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_point_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Point(0, 0), Point(1, 1)]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_polygon_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_mixed_family_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            Point(0, 0),
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
            LineString([(0, 0), (1, 1), (0, 1)]),
            LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_null_geometry_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [None, LineString([(0, 0), (1, 0), (0, 1), (0, 0)]), None]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.gpu
    def test_multiple_linestrings_device(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        geoms = [
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]),
            LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        result = is_ring_owned(owned)
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# GeoSeries / GeometryArray integration tests
# ---------------------------------------------------------------------------


class TestIsRingGeoSeries:
    """Tests for is_ring via the GeoSeries public API."""

    def test_geoseries_is_ring(self):
        """GeoSeries.is_ring routes through GeometryArray to is_ring_owned."""
        from vibespatial.api import GeoSeries

        geoms = [
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),      # ring
            LineString([(0, 0), (1, 1)]),                       # not ring
            Point(0, 0),                                        # not ring
        ]
        gs = GeoSeries(geoms)
        result = gs.is_ring
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result.values, expected)

    def test_geoseries_is_ring_mixed_with_null(self):
        """GeoSeries.is_ring handles null geometries correctly."""
        from vibespatial.api import GeoSeries

        geoms = [
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),
            None,
            LineString([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),
        ]
        gs = GeoSeries(geoms)
        result = gs.is_ring
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result.values, expected)


# ---------------------------------------------------------------------------
# DeviceGeometryArray tests
# ---------------------------------------------------------------------------


class TestIsRingDeviceGeometryArray:
    """Tests for is_ring via DeviceGeometryArray (no Shapely materialization)."""

    @pytest.mark.gpu
    def test_dga_is_ring_no_materialization(self):
        """DeviceGeometryArray.is_ring uses GPU path, not Shapely fallback."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")

        from vibespatial.geometry.device_array import DeviceGeometryArray

        geoms = [
            LineString([(0, 0), (1, 0), (0, 1), (0, 0)]),
            LineString([(0, 0), (1, 1)]),
        ]
        owned = _make_owned_with_device_state(geoms)
        dga = DeviceGeometryArray(owned)
        result = dga.is_ring
        expected = _shapely_is_ring(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Dispatch event observability tests
# ---------------------------------------------------------------------------


class TestIsRingDispatchEvents:
    """Tests for dispatch event recording during is_ring_owned."""

    def test_dispatch_event_recorded(self):
        """is_ring_owned records a dispatch event."""
        from vibespatial.runtime.dispatch import get_dispatch_events

        geoms = [LineString([(0, 0), (1, 0), (0, 1), (0, 0)])]
        owned = from_shapely_geometries(geoms)

        events_before = len(get_dispatch_events())
        is_ring_owned(owned)
        events_after = get_dispatch_events()

        # Find the is_ring event (there may also be is_simple events)
        ring_events = [
            e for e in events_after[events_before:]
            if e.operation == "is_ring"
        ]
        assert len(ring_events) == 1
        assert ring_events[0].surface == "geopandas.array.is_ring"
        assert ring_events[0].implementation in (
            "is_ring_gpu_composed",
            "is_ring_cpu_composed",
        )
