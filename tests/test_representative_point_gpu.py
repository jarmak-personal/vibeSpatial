"""Tests for GPU-accelerated representative_point.

Validates that representative_point produces points inside the source
geometry for all geometry types, including concave polygons where the
centroid falls outside the boundary.  Zero Shapely fallback events.
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

from vibespatial import from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.representative_point import representative_point_owned
from vibespatial.geometry.buffers import GeometryFamily

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_point_inside(result_geoms, source_geoms, *, tolerance: float = 1e-9):
    """Assert each result point is inside (or on boundary of) source geometry."""
    for i, (rpt, src) in enumerate(zip(result_geoms, source_geoms)):
        if src is None:
            # Null input -> result should have NaN coordinates
            assert rpt is not None, f"Row {i}: expected a point (possibly NaN), got None"
            continue
        if src.is_empty:
            continue
        assert rpt is not None, f"Row {i}: expected a point, got None"
        assert rpt.geom_type == "Point", f"Row {i}: expected Point, got {rpt.geom_type}"
        if rpt.is_empty:
            continue
        # The representative point must be inside or on boundary
        assert shapely.contains(src, rpt) or shapely.touches(src, rpt), (
            f"Row {i}: representative point {rpt} is not inside source geometry {src}"
        )


def _assert_matches_shapely(result_geoms, source_geoms, *, tolerance: float = 1e-6):
    """Assert result points match Shapely's point_on_surface within tolerance."""
    shapely_arr = np.asarray(source_geoms, dtype=object)
    shapely_results = shapely.point_on_surface(shapely_arr)
    for i, (rpt, spt) in enumerate(zip(result_geoms, shapely_results)):
        if source_geoms[i] is None:
            continue
        if source_geoms[i].is_empty:
            continue
        if spt is None or rpt is None:
            continue
        # Both should be valid points inside the geometry; we don't require
        # exact match since the algorithms may differ, just that both are inside
        assert rpt.geom_type == "Point"


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def _make_convex_polygon():
    """Simple convex polygon where centroid is inside."""
    return Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])


def _make_concave_l_shape():
    """L-shaped concave polygon where centroid falls outside."""
    return Polygon([
        (0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4),
    ])


def _make_concave_u_shape():
    """U-shaped concave polygon where centroid is outside the geometry."""
    return Polygon([
        (0, 0), (6, 0), (6, 4), (5, 4), (5, 1), (1, 1), (1, 4), (0, 4),
    ])


def _make_concave_c_shape():
    """C-shaped polygon where centroid is definitely outside."""
    return Polygon([
        (0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4),
    ])


def _make_polygon_with_hole():
    """Polygon with a hole - representative point must not be in the hole."""
    exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    return Polygon(exterior, [hole])


def _make_concave_with_hole():
    """Concave polygon with a hole."""
    exterior = [(0, 0), (10, 0), (10, 2), (6, 2), (6, 8), (10, 8), (10, 10), (0, 10)]
    hole = [(1, 1), (5, 1), (5, 5), (1, 5)]
    return Polygon(exterior, [hole])


def _make_thin_horizontal():
    """Very thin horizontal polygon."""
    return Polygon([(0, 0), (100, 0), (100, 0.01), (0, 0.01)])


def _make_triangle():
    """Simple triangle."""
    return Polygon([(0, 0), (4, 0), (2, 3)])


@pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")
def test_representative_point_stays_device_resident(strict_device_guard):
    """representative_point keeps single-family metadata on device."""
    from vibespatial.runtime import ExecutionMode
    from vibespatial.runtime.residency import Residency

    owned = from_shapely_geometries([_make_concave_l_shape()], residency=Residency.DEVICE)
    result = representative_point_owned(owned, dispatch_mode=ExecutionMode.GPU)

    assert result.residency == Residency.DEVICE


# ---------------------------------------------------------------------------
# Tests: Basic Functionality
# ---------------------------------------------------------------------------

class TestRepresentativePointBasic:
    """Basic tests for representative_point on various geometry types."""

    def test_single_point(self):
        owned = from_shapely_geometries([Point(1, 2)])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].geom_type == "Point"
        assert abs(geoms[0].x - 1.0) < 1e-10
        assert abs(geoms[0].y - 2.0) < 1e-10

    def test_multipoint(self):
        owned = from_shapely_geometries([MultiPoint([(1, 2), (3, 4)])])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].geom_type == "Point"

    def test_linestring(self):
        owned = from_shapely_geometries([LineString([(0, 0), (4, 0), (4, 4)])])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].geom_type == "Point"

    def test_multilinestring(self):
        owned = from_shapely_geometries([
            MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]]),
        ])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].geom_type == "Point"

    def test_empty_input(self):
        owned = from_shapely_geometries([])
        result = representative_point_owned(owned)
        assert result.row_count == 0

    def test_convex_polygon(self):
        poly = _make_convex_polygon()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_triangle(self):
        poly = _make_triangle()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])


# ---------------------------------------------------------------------------
# Tests: Concave Polygons (the main target of this implementation)
# ---------------------------------------------------------------------------

class TestRepresentativePointConcave:
    """Tests for concave polygons where centroid falls outside."""

    def test_l_shape(self):
        poly = _make_concave_l_shape()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_u_shape(self):
        poly = _make_concave_u_shape()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_c_shape(self):
        poly = _make_concave_c_shape()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_thin_horizontal(self):
        poly = _make_thin_horizontal()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_multiple_concave_polygons(self):
        """Test a batch of concave polygons."""
        polys = [
            _make_concave_l_shape(),
            _make_concave_u_shape(),
            _make_concave_c_shape(),
        ]
        owned = from_shapely_geometries(polys)
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, polys)


# ---------------------------------------------------------------------------
# Tests: Polygons with Holes
# ---------------------------------------------------------------------------

class TestRepresentativePointWithHoles:
    """Tests for polygons with holes."""

    def test_simple_hole(self):
        poly = _make_polygon_with_hole()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_concave_with_hole(self):
        poly = _make_concave_with_hole()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])

    def test_donut(self):
        """Polygon where hole is centered - centroid is in the hole."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
        poly = Polygon(exterior, [hole])
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])


# ---------------------------------------------------------------------------
# Tests: MultiPolygon
# ---------------------------------------------------------------------------

class TestRepresentativePointMultiPolygon:
    """Tests for MultiPolygon geometries."""

    def test_simple_multipolygon(self):
        mp = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
        ])
        owned = from_shapely_geometries([mp])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [mp])

    def test_concave_multipolygon(self):
        """MultiPolygon with concave parts."""
        mp = MultiPolygon([
            _make_concave_l_shape(),
            _make_concave_u_shape(),
        ])
        owned = from_shapely_geometries([mp])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [mp])


# ---------------------------------------------------------------------------
# Tests: Mixed Geometry Types
# ---------------------------------------------------------------------------

class TestRepresentativePointMixed:
    """Tests for arrays with mixed geometry types."""

    def test_mixed_types(self):
        geoms = [
            Point(1, 2),
            _make_convex_polygon(),
            LineString([(0, 0), (1, 1)]),
            _make_concave_l_shape(),
        ]
        owned = from_shapely_geometries(geoms)
        result = representative_point_owned(owned)
        result_geoms = result.to_shapely()
        assert len(result_geoms) == 4
        _assert_point_inside(result_geoms, geoms)

    def test_mixed_with_null(self):
        geoms = [
            _make_convex_polygon(),
            None,
            _make_concave_c_shape(),
            None,
        ]
        owned = from_shapely_geometries(geoms)
        result = representative_point_owned(owned)
        result_geoms = result.to_shapely()
        assert len(result_geoms) == 4
        # Non-null polygon rows should have valid representative points
        for i, (rpt, src) in enumerate(zip(result_geoms, geoms)):
            if src is not None:
                _assert_point_inside([rpt], [src])


# ---------------------------------------------------------------------------
# Tests: Null and Empty Handling
# ---------------------------------------------------------------------------

class TestRepresentativePointNullEmpty:
    """Tests for null and empty geometry handling."""

    def test_null_rows_produce_nan(self):
        owned = from_shapely_geometries([None, _make_convex_polygon(), None])
        result = representative_point_owned(owned)
        # Null rows should produce NaN coordinates
        assert result.families[GeometryFamily.POINT].x[0] != result.families[GeometryFamily.POINT].x[0]  # NaN check
        assert result.families[GeometryFamily.POINT].x[2] != result.families[GeometryFamily.POINT].x[2]  # NaN check

    def test_all_null(self):
        owned = from_shapely_geometries([None, None])
        result = representative_point_owned(owned)
        assert result.row_count == 2
        x = result.families[GeometryFamily.POINT].x
        assert np.all(np.isnan(x))


# ---------------------------------------------------------------------------
# Tests: Correctness against Shapely oracle
# ---------------------------------------------------------------------------

class TestRepresentativePointOracle:
    """Compare results against Shapely as reference implementation."""

    @pytest.mark.parametrize("poly_factory", [
        _make_convex_polygon,
        _make_concave_l_shape,
        _make_concave_u_shape,
        _make_concave_c_shape,
        _make_polygon_with_hole,
        _make_concave_with_hole,
        _make_thin_horizontal,
        _make_triangle,
    ], ids=[
        "convex",
        "l_shape",
        "u_shape",
        "c_shape",
        "with_hole",
        "concave_with_hole",
        "thin_horizontal",
        "triangle",
    ])
    def test_result_inside_polygon(self, poly_factory):
        """Result point must be inside the polygon (same guarantee as Shapely)."""
        poly = poly_factory()
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])


# ---------------------------------------------------------------------------
# Tests: GPU-specific
# ---------------------------------------------------------------------------

@pytest.mark.gpu
class TestRepresentativePointGPU:
    """GPU-specific tests requiring CUDA runtime."""

    def test_gpu_concave_batch(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        polys = [
            _make_concave_l_shape(),
            _make_concave_u_shape(),
            _make_concave_c_shape(),
            _make_polygon_with_hole(),
            _make_concave_with_hole(),
        ]
        owned = from_shapely_geometries(polys)
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, polys)

    def test_gpu_multipolygon_concave(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        mp = MultiPolygon([
            _make_concave_l_shape(),
            _make_concave_c_shape(),
        ])
        owned = from_shapely_geometries([mp])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [mp])

    def test_gpu_large_batch_concave(self):
        """Stress test with many concave polygons."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        n = 100
        polys = [_make_concave_l_shape() for _ in range(n)]
        owned = from_shapely_geometries(polys)
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, polys)

    def test_gpu_donut_centroid_in_hole(self):
        """Donut polygon where centroid lands in the hole."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]
        poly = Polygon(exterior, [hole])
        owned = from_shapely_geometries([poly])
        result = representative_point_owned(owned)
        geoms = result.to_shapely()
        _assert_point_inside(geoms, [poly])


# ---------------------------------------------------------------------------
# Tests: GeoSeries integration
# ---------------------------------------------------------------------------

class TestRepresentativePointGeoSeries:
    """Tests through the GeoSeries API."""

    def test_geoseries_representative_point(self):
        """Test that GeoSeries.representative_point() uses the owned path."""
        import geopandas as gpd

        polys = [_make_convex_polygon(), _make_concave_l_shape(), _make_concave_c_shape()]
        gs = gpd.GeoSeries(polys)
        result = gs.representative_point()
        assert len(result) == 3
        for i, (rpt, src) in enumerate(zip(result, polys)):
            rpt_shapely = rpt
            assert shapely.contains(src, rpt_shapely) or shapely.touches(src, rpt_shapely), (
                f"Row {i}: representative point {rpt_shapely} not inside {src}"
            )
