from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box

from vibespatial import has_gpu_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.testing import build_owned as _make_owned

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def _compute_distances(query_owned, tree_owned, left_idx, right_idx, tree_family, exclusive=False):
    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.spatial.point_distance import compute_point_distance_gpu

    runtime = get_cuda_runtime()
    pair_count = len(left_idx)
    d_left = runtime.from_host(np.asarray(left_idx, dtype=np.int32))
    d_right = runtime.from_host(np.asarray(right_idx, dtype=np.int32))
    d_distances = runtime.allocate((pair_count,), np.float64)
    try:
        ok = compute_point_distance_gpu(
            query_owned, tree_owned, d_left, d_right, d_distances,
            pair_count, tree_family=tree_family, exclusive=exclusive,
        )
        assert ok, f"kernel dispatch failed for {tree_family}"
        out = np.empty(pair_count, dtype=np.float64)
        runtime.copy_device_to_host(d_distances, out)
        return out
    finally:
        runtime.free(d_left)
        runtime.free(d_right)
        runtime.free(d_distances)


class TestPointLinestringDistance:
    def test_basic_distances(self, make_owned):
        points = [Point(0, 0), Point(1, 1), Point(3, 0)]
        lines = [
            LineString([(0, 1), (2, 1)]),  # horizontal line at y=1
            LineString([(0, 0), (2, 0)]),  # horizontal line at y=0
            LineString([(0, 1), (2, 1)]),  # same as first
        ]
        query_owned = make_owned(points)
        tree_owned = make_owned(lines)

        left_idx = np.array([0, 1, 2], dtype=np.int32)
        right_idx = np.array([0, 1, 2], dtype=np.int32)

        gpu_dist = _compute_distances(
            query_owned, tree_owned, left_idx, right_idx,
            GeometryFamily.LINESTRING,
        )

        # Reference from Shapely.
        expected = np.array([
            shapely.distance(points[i], lines[j])
            for i, j in zip(left_idx, right_idx)
        ])
        np.testing.assert_allclose(gpu_dist, expected, atol=1e-10)

    def test_point_on_segment(self, make_owned):
        """Point exactly on a segment → distance 0."""
        points = [Point(1, 1)]
        lines = [LineString([(0, 0), (2, 2)])]
        query_owned = make_owned(points)
        tree_owned = make_owned(lines)

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING,
        )
        assert gpu_dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_point_nearest_segment_endpoint(self, make_owned):
        """Point closest to a segment endpoint (projection clamps to t=0 or t=1)."""
        points = [Point(5, 0)]
        lines = [LineString([(0, 0), (2, 0)])]
        query_owned = make_owned(points)
        tree_owned = make_owned(lines)

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING,
        )
        # Closest point is (2,0), distance = 3.0
        assert gpu_dist[0] == pytest.approx(3.0, abs=1e-10)

    def test_multi_segment_linestring(self, make_owned):
        """Min distance across multiple segments."""
        points = [Point(1, 2)]
        lines = [LineString([(0, 0), (2, 0), (2, 3)])]
        query_owned = make_owned(points)
        tree_owned = make_owned(lines)

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING,
        )
        expected = shapely.distance(points[0], lines[0])
        assert gpu_dist[0] == pytest.approx(expected, abs=1e-10)


class TestPointMultiLinestringDistance:
    def test_basic(self, make_owned):
        points = [Point(0, 0), Point(5, 5)]
        multilines = [
            MultiLineString([[(1, 0), (1, 2)], [(3, 0), (3, 2)]]),
            MultiLineString([[(4, 4), (6, 6)]]),
        ]
        query_owned = make_owned(points)
        tree_owned = make_owned(multilines)

        left_idx = np.array([0, 1], dtype=np.int32)
        right_idx = np.array([0, 1], dtype=np.int32)

        gpu_dist = _compute_distances(
            query_owned, tree_owned, left_idx, right_idx,
            GeometryFamily.MULTILINESTRING,
        )
        expected = np.array([
            shapely.distance(points[i], multilines[j])
            for i, j in zip(left_idx, right_idx)
        ])
        np.testing.assert_allclose(gpu_dist, expected, atol=1e-10)


class TestPointPolygonDistance:
    def test_point_inside_polygon_distance_zero(self, make_owned):
        points = [Point(1, 1)]
        polygons = [box(0, 0, 2, 2)]
        query_owned = make_owned(points)
        tree_owned = make_owned(polygons)

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON,
        )
        assert gpu_dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_point_on_boundary_distance_zero(self, make_owned):
        points = [Point(0, 1)]
        polygons = [box(0, 0, 2, 2)]
        query_owned = make_owned(points)
        tree_owned = make_owned(polygons)

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON,
        )
        assert gpu_dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_point_outside_polygon(self, make_owned):
        points = [Point(3, 1), Point(0, 5)]
        polygons = [box(0, 0, 2, 2), box(0, 0, 2, 2)]
        query_owned = make_owned(points)
        tree_owned = make_owned(polygons)

        left_idx = np.array([0, 1], dtype=np.int32)
        right_idx = np.array([0, 1], dtype=np.int32)

        gpu_dist = _compute_distances(
            query_owned, tree_owned, left_idx, right_idx,
            GeometryFamily.POLYGON,
        )
        expected = np.array([
            shapely.distance(points[i], polygons[j])
            for i, j in zip(left_idx, right_idx)
        ])
        np.testing.assert_allclose(gpu_dist, expected, atol=1e-10)

    def test_polygon_with_hole(self, make_owned):
        """Point inside hole → positive distance to nearest ring edge."""
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        polygon = Polygon(outer, [hole])
        point = Point(5, 5)  # center of the hole

        query_owned = make_owned([point])
        tree_owned = make_owned([polygon])

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON,
        )
        expected = shapely.distance(point, polygon)
        assert gpu_dist[0] == pytest.approx(expected, abs=1e-10)


class TestPointMultiPolygonDistance:
    def test_basic(self, make_owned):
        points = [Point(3, 3), Point(11, 11)]
        multipolys = [
            MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)]),
            MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)]),
        ]
        query_owned = make_owned(points)
        tree_owned = make_owned(multipolys)

        left_idx = np.array([0, 1], dtype=np.int32)
        right_idx = np.array([0, 1], dtype=np.int32)

        gpu_dist = _compute_distances(
            query_owned, tree_owned, left_idx, right_idx,
            GeometryFamily.MULTIPOLYGON,
        )
        expected = np.array([
            shapely.distance(points[i], multipolys[j])
            for i, j in zip(left_idx, right_idx)
        ])
        np.testing.assert_allclose(gpu_dist, expected, atol=1e-10)

    def test_point_inside_one_part(self):
        """Point inside one polygon of a multipolygon → distance 0."""
        mp = MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])
        point = Point(1, 1)

        query_owned = _make_owned([point])
        tree_owned = _make_owned([mp])

        gpu_dist = _compute_distances(
            query_owned, tree_owned,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTIPOLYGON,
        )
        assert gpu_dist[0] == pytest.approx(0.0, abs=1e-10)


class TestNearestPipelineIntegration:
    """Test that the nearest pipeline uses GPU point-distance kernels."""

    def test_point_nearest_linestrings(self):
        """Point nearest against linestrings should use GPU when available."""
        import geopandas

        points_geom = [Point(0, 0), Point(5, 5), Point(10, 0)]
        lines_geom = [
            LineString([(1, 0), (1, 2)]),
            LineString([(4, 4), (6, 6)]),
            LineString([(9, 0), (11, 0)]),
        ]

        tree_series = geopandas.GeoSeries(lines_geom)
        query_series = geopandas.GeoSeries(points_geom)

        result = tree_series.sindex.nearest(query_series, return_all=True)

        # Verify correctness: each point should find its nearest line.
        assert result.shape[1] >= 3

        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            gpu_nearest_dist = shapely.distance(points_geom[qi], lines_geom[ti])
            all_dists = [shapely.distance(points_geom[qi], ln) for ln in lines_geom]
            assert gpu_nearest_dist == pytest.approx(min(all_dists), abs=1e-10)

    def test_point_nearest_polygons(self):
        """Point nearest against polygons should use GPU when available."""
        import geopandas

        points_geom = [Point(3, 3), Point(15, 15)]
        poly_geom = [box(0, 0, 2, 2), box(10, 10, 12, 12), box(20, 20, 22, 22)]

        tree_series = geopandas.GeoSeries(poly_geom)
        query_series = geopandas.GeoSeries(points_geom)

        result = tree_series.sindex.nearest(query_series, return_all=True)

        assert result.shape[1] >= 2

        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            gpu_nearest_dist = shapely.distance(points_geom[qi], poly_geom[ti])
            all_dists = [shapely.distance(points_geom[qi], p) for p in poly_geom]
            assert gpu_nearest_dist == pytest.approx(min(all_dists), abs=1e-10)
