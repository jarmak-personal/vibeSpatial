"""Tests for GPU-accelerated DE-9IM relate computation.

Verifies correctness against Shapely oracle for:
    - Point-Point (equal, disjoint)
    - Point-LineString (interior, boundary, exterior)
    - Point-Polygon (interior, boundary, exterior)
    - Polygon-Point (transposed)
    - Line-Point (transposed)
    - Non-point fallback (Polygon-Polygon, Line-Line)
    - Null geometry handling
    - Mixed-family arrays
    - GeometryArray.relate() wiring
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial import ExecutionMode, has_gpu_runtime
from vibespatial.api.geometry_array import GeometryArray
from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.predicates.relate import relate_de9im
from vibespatial.runtime.fallbacks import (
    STRICT_NATIVE_ENV_VAR,
    StrictNativeFallbackError,
    clear_fallback_events,
    get_fallback_events,
)

requires_gpu = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU not available")


def _assert_relate_matches_shapely(
    left_geoms,
    right_geoms,
    *,
    label="",
    dispatch_mode=ExecutionMode.AUTO,
):
    """Assert GPU relate produces the same DE-9IM strings as Shapely."""
    left_np = np.array(left_geoms, dtype=object)
    right_np = np.array(right_geoms, dtype=object)
    expected = shapely.relate(left_np, right_np)

    left_owned = from_shapely_geometries(list(left_geoms))
    right_owned = from_shapely_geometries(list(right_geoms))
    result = relate_de9im(left_owned, right_owned, dispatch_mode=dispatch_mode)

    assert len(result) == len(expected), f"{label}: length mismatch"
    for i in range(len(result)):
        if expected[i] is None or (isinstance(expected[i], float) and np.isnan(expected[i])):
            assert result[i] is None, (
                f"{label}[{i}]: expected None, got {result[i]!r}"
            )
        else:
            assert result[i] == expected[i], (
                f"{label}[{i}]: expected {expected[i]!r}, got {result[i]!r} "
                f"(left={left_geoms[i]}, right={right_geoms[i]})"
            )


# ---------------------------------------------------------------------------
# Point-Point
# ---------------------------------------------------------------------------

@requires_gpu
class TestPointPointRelate:
    def test_equal_points(self):
        """Two identical points -> '0FFFFFFF2'."""
        _assert_relate_matches_shapely(
            [Point(1, 2), Point(3, 4)],
            [Point(1, 2), Point(3, 4)],
            label="equal_points",
        )

    def test_disjoint_points(self):
        """Two different points -> 'FF0FFF0F2'."""
        _assert_relate_matches_shapely(
            [Point(1, 2), Point(5, 6)],
            [Point(3, 4), Point(7, 8)],
            label="disjoint_points",
        )

    def test_mixed_equal_disjoint_points(self):
        """Mix of equal and disjoint point pairs."""
        _assert_relate_matches_shapely(
            [Point(0, 0), Point(1, 1), Point(2, 2)],
            [Point(0, 0), Point(9, 9), Point(2, 2)],
            label="mixed_points",
        )


# ---------------------------------------------------------------------------
# Point-LineString
# ---------------------------------------------------------------------------

@requires_gpu
class TestPointLineRelate:
    def test_point_interior_of_line(self):
        """Point on interior of linestring."""
        _assert_relate_matches_shapely(
            [Point(1, 0)],
            [LineString([(0, 0), (2, 0)])],
            label="point_interior_line",
        )

    def test_point_on_line_endpoint(self):
        """Point at endpoint of linestring -> boundary."""
        _assert_relate_matches_shapely(
            [Point(0, 0)],
            [LineString([(0, 0), (2, 0)])],
            label="point_boundary_line",
        )

    def test_point_exterior_of_line(self):
        """Point not on linestring."""
        _assert_relate_matches_shapely(
            [Point(0, 5)],
            [LineString([(0, 0), (2, 0)])],
            label="point_exterior_line",
        )

    def test_point_line_mixed(self):
        """Mix of interior/boundary/exterior point-line pairs."""
        _assert_relate_matches_shapely(
            [Point(1, 0), Point(0, 0), Point(0, 5)],
            [
                LineString([(0, 0), (2, 0)]),
                LineString([(0, 0), (2, 0)]),
                LineString([(0, 0), (2, 0)]),
            ],
            label="point_line_mixed",
        )


# ---------------------------------------------------------------------------
# Line-Point (transposed)
# ---------------------------------------------------------------------------

@requires_gpu
class TestLinePointRelate:
    def test_line_point_interior(self):
        """LineString with point on its interior (transposed matrix)."""
        _assert_relate_matches_shapely(
            [LineString([(0, 0), (2, 0)])],
            [Point(1, 0)],
            label="line_point_interior",
        )

    def test_line_point_boundary(self):
        """LineString with point at its endpoint (transposed matrix)."""
        _assert_relate_matches_shapely(
            [LineString([(0, 0), (2, 0)])],
            [Point(0, 0)],
            label="line_point_boundary",
        )

    def test_line_point_exterior(self):
        """LineString with disjoint point (transposed matrix)."""
        _assert_relate_matches_shapely(
            [LineString([(0, 0), (2, 0)])],
            [Point(0, 5)],
            label="line_point_exterior",
        )


# ---------------------------------------------------------------------------
# Point-Polygon
# ---------------------------------------------------------------------------

@requires_gpu
class TestPointPolygonRelate:
    def test_point_interior_of_polygon(self):
        """Point inside polygon."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [Point(2, 2)],
            [poly],
            label="point_interior_polygon",
        )

    def test_point_on_polygon_boundary(self):
        """Point on polygon edge."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [Point(2, 0)],
            [poly],
            label="point_boundary_polygon",
        )

    def test_point_exterior_of_polygon(self):
        """Point outside polygon."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [Point(5, 5)],
            [poly],
            label="point_exterior_polygon",
        )

    def test_point_polygon_mixed(self):
        """Mix of interior/boundary/exterior point-polygon pairs."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [Point(2, 2), Point(2, 0), Point(5, 5)],
            [poly, poly, poly],
            label="point_polygon_mixed",
        )


# ---------------------------------------------------------------------------
# Polygon-Point (transposed)
# ---------------------------------------------------------------------------

@requires_gpu
class TestPolygonPointRelate:
    def test_polygon_point_interior(self):
        """Polygon containing a point (transposed matrix)."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [poly],
            [Point(2, 2)],
            label="polygon_point_interior",
        )

    def test_polygon_point_boundary(self):
        """Polygon with point on its boundary (transposed matrix)."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [poly],
            [Point(2, 0)],
            label="polygon_point_boundary",
        )

    def test_polygon_point_exterior(self):
        """Polygon with disjoint point (transposed matrix)."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        _assert_relate_matches_shapely(
            [poly],
            [Point(5, 5)],
            label="polygon_point_exterior",
        )


# ---------------------------------------------------------------------------
# Point-MultiLineString
# ---------------------------------------------------------------------------

@requires_gpu
class TestPointMultiLineRelate:
    def test_point_interior_of_multiline(self):
        """Point on interior of MultiLineString."""
        mls = MultiLineString([[(0, 0), (2, 0)], [(3, 0), (5, 0)]])
        _assert_relate_matches_shapely(
            [Point(1, 0)],
            [mls],
            label="point_interior_mls",
        )

    def test_point_exterior_of_multiline(self):
        """Point not on MultiLineString."""
        mls = MultiLineString([[(0, 0), (2, 0)], [(3, 0), (5, 0)]])
        _assert_relate_matches_shapely(
            [Point(0, 5)],
            [mls],
            label="point_exterior_mls",
        )


# ---------------------------------------------------------------------------
# Point-MultiPolygon
# ---------------------------------------------------------------------------

@requires_gpu
class TestPointMultiPolygonRelate:
    def test_point_interior_of_multipolygon(self):
        """Point inside one of the MultiPolygon components."""
        mp = MultiPolygon([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        ])
        _assert_relate_matches_shapely(
            [Point(1, 1)],
            [mp],
            label="point_interior_mpoly",
        )

    def test_point_exterior_of_multipolygon(self):
        """Point outside all MultiPolygon components."""
        mp = MultiPolygon([
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(5, 5), (7, 5), (7, 7), (5, 7)]),
        ])
        _assert_relate_matches_shapely(
            [Point(3, 3)],
            [mp],
            label="point_exterior_mpoly",
        )


# ---------------------------------------------------------------------------
# Non-point families (Shapely fallback)
# ---------------------------------------------------------------------------

@requires_gpu
class TestNonPointFallback:
    def test_polygon_polygon(self):
        """Polygon-Polygon falls back to Shapely."""
        poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        poly_b = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
        _assert_relate_matches_shapely(
            [poly_a],
            [poly_b],
            label="polygon_polygon",
        )

    def test_line_line(self):
        """LineString-LineString falls back to Shapely."""
        line_a = LineString([(0, 0), (2, 2)])
        line_b = LineString([(0, 2), (2, 0)])
        _assert_relate_matches_shapely(
            [line_a],
            [line_b],
            label="line_line",
        )

    def test_polygon_line(self):
        """Polygon-LineString falls back to Shapely."""
        poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        line = LineString([(1, 1), (5, 5)])
        _assert_relate_matches_shapely(
            [poly],
            [line],
            label="polygon_line",
        )


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------

@requires_gpu
class TestNullHandling:
    def test_left_null(self):
        """Null left geometry produces None."""
        left_owned = from_shapely_geometries([None])
        right_owned = from_shapely_geometries([Point(1, 2)])
        result = relate_de9im(left_owned, right_owned)
        assert result[0] is None

    def test_right_null(self):
        """Null right geometry produces None."""
        left_owned = from_shapely_geometries([Point(1, 2)])
        right_owned = from_shapely_geometries([None])
        result = relate_de9im(left_owned, right_owned)
        assert result[0] is None

    def test_both_null(self):
        """Both null produces None."""
        left_owned = from_shapely_geometries([None])
        right_owned = from_shapely_geometries([None])
        result = relate_de9im(left_owned, right_owned)
        assert result[0] is None

    def test_mixed_null_and_valid(self):
        """Mix of null and valid geometries."""
        left_owned = from_shapely_geometries([Point(1, 1), None, Point(3, 3)])
        right_owned = from_shapely_geometries([Point(1, 1), Point(2, 2), None])
        result = relate_de9im(left_owned, right_owned)
        assert result[0] is not None  # valid pair
        assert result[1] is None  # left null
        assert result[2] is None  # right null


# ---------------------------------------------------------------------------
# Mixed-family arrays
# ---------------------------------------------------------------------------

@requires_gpu
class TestMixedFamilyRelate:
    def test_mixed_point_and_polygon(self):
        """Array with both Point-Point and Polygon-Polygon pairs."""
        left = [
            Point(1, 1),  # point-point
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),  # polygon-polygon
            Point(2, 2),  # point-polygon
        ]
        right = [
            Point(1, 1),  # point-point
            Polygon([(2, 2), (6, 2), (6, 6), (2, 6)]),  # polygon-polygon
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),  # point-polygon
        ]
        _assert_relate_matches_shapely(left, right, label="mixed_family")


# ---------------------------------------------------------------------------
# Empty arrays
# ---------------------------------------------------------------------------

@requires_gpu
class TestEdgeCases:
    def test_empty_arrays(self):
        """Empty input arrays produce empty output."""
        left_owned = from_shapely_geometries([])
        right_owned = from_shapely_geometries([])
        result = relate_de9im(left_owned, right_owned)
        assert len(result) == 0

    def test_row_count_mismatch(self):
        """Mismatched row counts raise ValueError."""
        left_owned = from_shapely_geometries([Point(1, 1)])
        right_owned = from_shapely_geometries([Point(1, 1), Point(2, 2)])
        with pytest.raises(ValueError, match="same row count"):
            relate_de9im(left_owned, right_owned)


# ---------------------------------------------------------------------------
# GeometryArray.relate() wiring
# ---------------------------------------------------------------------------

@requires_gpu
class TestGeometryArrayWiring:
    def test_relate_via_geometry_array(self):
        """GeometryArray.relate() dispatches to GPU relate."""
        left_geoms = np.array([Point(1, 1), Point(2, 2)], dtype=object)
        right_geoms = np.array([Point(1, 1), Point(3, 3)], dtype=object)
        expected = shapely.relate(left_geoms, right_geoms)

        ga_left = GeometryArray(left_geoms)
        # Force owned creation to enable GPU path.
        ga_left.to_owned()
        ga_right = GeometryArray(right_geoms)

        result = ga_left.relate(ga_right)
        for i in range(len(result)):
            assert result[i] == expected[i], (
                f"[{i}]: expected {expected[i]!r}, got {result[i]!r}"
            )

    def test_relate_strict_native_non_point_decline_is_not_swallowed(self, monkeypatch):
        """Strict native must not be hidden by GeometryArray's Shapely fallback."""
        left_geoms = np.array([Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])], dtype=object)
        right_geoms = np.array([Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])], dtype=object)

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        ga_right = GeometryArray(right_geoms)
        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        with pytest.raises(StrictNativeFallbackError, match="non-point family"):
            ga_left.relate(ga_right)

    def test_relate_scalar_broadcast_stays_native_in_strict(self, monkeypatch):
        """Scalar right-hand relate uses an owned broadcast view before Shapely."""
        left_geoms = np.array([Point(2, 2), Point(5, 5)], dtype=object)
        polygon = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        expected = shapely.relate(left_geoms, polygon)

        ga_left = GeometryArray(left_geoms)
        ga_left.to_owned()
        clear_fallback_events()
        monkeypatch.setenv(STRICT_NATIVE_ENV_VAR, "1")

        result = ga_left.relate(polygon)

        np.testing.assert_array_equal(result, expected)
        assert get_fallback_events(clear=True) == []


# ---------------------------------------------------------------------------
# Larger batch to exercise actual GPU parallelism
# ---------------------------------------------------------------------------

@requires_gpu
class TestBatchRelate:
    def test_batch_point_point(self):
        """Batch of 1000 point-point pairs."""
        rng = np.random.default_rng(42)
        n = 1000
        left_pts = [Point(rng.random(), rng.random()) for _ in range(n)]
        right_pts = [Point(rng.random(), rng.random()) for _ in range(n)]
        # Make every 10th pair identical.
        for i in range(0, n, 10):
            right_pts[i] = left_pts[i]
        _assert_relate_matches_shapely(left_pts, right_pts, label="batch_pp")

    def test_batch_point_polygon(self):
        """Batch of point-polygon pairs."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        rng = np.random.default_rng(99)
        n = 100
        points = [Point(rng.uniform(-2, 12), rng.uniform(-2, 12)) for _ in range(n)]
        _assert_relate_matches_shapely(
            points,
            [poly] * n,
            label="batch_point_polygon",
        )

    def test_batch_mixed(self):
        """Batch with mixed Point-Point, Point-Line, Point-Polygon."""
        line = LineString([(0, 0), (10, 0)])
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        left = [
            Point(5, 0),   # point on line interior
            Point(0, 0),   # point on line boundary
            Point(5, 5),   # point in polygon interior
            Point(5, 0),   # point on polygon boundary
            Point(15, 15), # point exterior to polygon
        ]
        right = [
            line,
            line,
            poly,
            poly,
            poly,
        ]
        _assert_relate_matches_shapely(left, right, label="batch_mixed")


# ---------------------------------------------------------------------------
# Explicitly GPU-forced tests (bypass adaptive crossover threshold)
# ---------------------------------------------------------------------------

@requires_gpu
class TestExplicitGPUDispatch:
    """Force GPU dispatch to verify the GPU path is actually exercised."""

    def test_gpu_forced_point_point(self):
        """Point-Point with explicit GPU dispatch."""
        _assert_relate_matches_shapely(
            [Point(1, 2), Point(3, 4), Point(0, 0)],
            [Point(1, 2), Point(5, 6), Point(0, 0)],
            label="gpu_forced_pp",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_point_line(self):
        """Point-Line with explicit GPU dispatch."""
        line = LineString([(0, 0), (10, 0)])
        _assert_relate_matches_shapely(
            [Point(5, 0), Point(0, 0), Point(0, 5)],
            [line, line, line],
            label="gpu_forced_pl",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_point_polygon(self):
        """Point-Polygon with explicit GPU dispatch."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        _assert_relate_matches_shapely(
            [Point(5, 5), Point(5, 0), Point(15, 15)],
            [poly, poly, poly],
            label="gpu_forced_ppoly",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_polygon_point(self):
        """Polygon-Point (transposed) with explicit GPU dispatch."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        _assert_relate_matches_shapely(
            [poly, poly],
            [Point(5, 5), Point(15, 15)],
            label="gpu_forced_polyp",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_line_point(self):
        """Line-Point (transposed) with explicit GPU dispatch."""
        line = LineString([(0, 0), (10, 0)])
        _assert_relate_matches_shapely(
            [line, line],
            [Point(5, 0), Point(0, 5)],
            label="gpu_forced_lp",
            dispatch_mode=ExecutionMode.GPU,
        )

    def test_gpu_forced_mixed_with_fallback(self):
        """Mixed Point-* and Polygon-Polygon with GPU dispatch.

        Point-* pairs go to GPU, Polygon-Polygon falls back to Shapely.
        """
        poly_a = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
        poly_b = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
        _assert_relate_matches_shapely(
            [Point(1, 1), poly_a, Point(2, 2)],
            [Point(1, 1), poly_b, poly_a],
            label="gpu_forced_mixed",
            dispatch_mode=ExecutionMode.GPU,
        )
