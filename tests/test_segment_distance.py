from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
    box,
)

from vibespatial import has_gpu_runtime
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.owned_geometry import from_shapely_geometries

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def _make_owned(geoms):
    return from_shapely_geometries(geoms)


def _compute_distances(query_owned, tree_owned, left_idx, right_idx, query_family, tree_family, exclusive=False):
    from vibespatial.cuda_runtime import get_cuda_runtime
    from vibespatial.segment_distance import compute_segment_distance_gpu

    runtime = get_cuda_runtime()
    pair_count = len(left_idx)
    d_left = runtime.from_host(np.asarray(left_idx, dtype=np.int32))
    d_right = runtime.from_host(np.asarray(right_idx, dtype=np.int32))
    d_distances = runtime.allocate((pair_count,), np.float64)
    try:
        ok = compute_segment_distance_gpu(
            query_owned, tree_owned, d_left, d_right, d_distances,
            pair_count, query_family=query_family, tree_family=tree_family,
            exclusive=exclusive,
        )
        assert ok, f"kernel dispatch failed for {query_family} × {tree_family}"
        out = np.empty(pair_count, dtype=np.float64)
        runtime.copy_device_to_host(d_distances, out)
        return out
    finally:
        runtime.free(d_left)
        runtime.free(d_right)
        runtime.free(d_distances)


def _shapely_distances(query_geoms, tree_geoms, left_idx, right_idx):
    return np.array([
        shapely.distance(query_geoms[li], tree_geoms[ri])
        for li, ri in zip(left_idx, right_idx)
    ])


# ---- LS × LS ----

class TestLsLsDistance:
    def test_parallel_lines(self):
        """Parallel lines — distance is perpendicular gap."""
        q = [LineString([(0, 0), (10, 0)])]
        t = [LineString([(2, 3), (8, 3)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert gpu[0] == pytest.approx(3.0, abs=1e-10)

    def test_crossing_lines(self):
        """Crossing lines → distance 0."""
        q = [LineString([(0, 0), (2, 2)])]
        t = [LineString([(0, 2), (2, 0)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)

    def test_endpoint_nearest(self):
        q = [LineString([(0, 0), (1, 0)])]
        t = [LineString([(3, 0), (4, 0)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert gpu[0] == pytest.approx(2.0, abs=1e-10)

    def test_multi_pair(self):
        q = [LineString([(0, 0), (1, 0)]), LineString([(5, 5), (6, 5)])]
        t = [LineString([(0, 1), (1, 1)]), LineString([(5, 0), (6, 0)])]
        left = np.array([0, 1, 0, 1], dtype=np.int32)
        right = np.array([0, 1, 1, 0], dtype=np.int32)
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t), left, right,
            GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        expected = _shapely_distances(q, t, left, right)
        np.testing.assert_allclose(gpu, expected, atol=1e-10)


# ---- LS × PG ----

class TestLsPgDistance:
    def test_line_outside_polygon(self):
        q = [LineString([(3, 0), (4, 0)])]
        t = [box(0, 0, 2, 2)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_line_inside_polygon(self):
        q = [LineString([(0.5, 0.5), (1.5, 1.5)])]
        t = [box(0, 0, 2, 2)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)

    def test_line_crossing_polygon_boundary(self):
        q = [LineString([(1, 1), (3, 1)])]
        t = [box(0, 0, 2, 2)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_dispatch(self):
        """PG × LS uses canonical swap and gives same result."""
        lines = [LineString([(3, 0), (4, 0)])]
        polys = [box(0, 0, 2, 2)]
        # Query=PG, Tree=LS (reversed).
        gpu = _compute_distances(
            _make_owned(polys), _make_owned(lines),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.LINESTRING,
        )
        expected = shapely.distance(polys[0], lines[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)


# ---- PG × PG ----

class TestPgPgDistance:
    def test_disjoint_boxes(self):
        q = [box(0, 0, 1, 1)]
        t = [box(3, 0, 4, 1)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert gpu[0] == pytest.approx(2.0, abs=1e-10)

    def test_overlapping_polygons(self):
        q = [box(0, 0, 2, 2)]
        t = [box(1, 1, 3, 3)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)

    def test_contained_polygon(self):
        q = [box(0, 0, 10, 10)]
        t = [box(2, 2, 3, 3)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)

    def test_polygon_with_hole(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        q = [Polygon(outer, [hole])]
        t = [box(4, 4, 6, 6)]  # inside the hole
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)


# ---- Multi-geometry variants ----

class TestMlsDistance:
    def test_mls_mls(self):
        q = [MultiLineString([[(0, 0), (1, 0)], [(5, 5), (6, 5)]])]
        t = [MultiLineString([[(0, 3), (1, 3)], [(5, 0), (6, 0)]])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTILINESTRING, GeometryFamily.MULTILINESTRING,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_ls_mls(self):
        q = [LineString([(0, 0), (1, 0)])]
        t = [MultiLineString([[(0, 5), (1, 5)], [(0, 2), (1, 2)]])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)


class TestMpgDistance:
    def test_pg_mpg(self):
        q = [box(0, 0, 1, 1)]
        t = [MultiPolygon([box(5, 5, 6, 6), box(2, 0, 3, 1)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_mpg_mpg(self):
        q = [MultiPolygon([box(0, 0, 1, 1), box(10, 10, 11, 11)])]
        t = [MultiPolygon([box(2, 0, 3, 1), box(8, 8, 9, 9)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_mpg_mpg_contained(self):
        q = [MultiPolygon([box(0, 0, 10, 10)])]
        t = [MultiPolygon([box(2, 2, 3, 3)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON,
        )
        assert gpu[0] == pytest.approx(0.0, abs=1e-10)


class TestCrossTypeDistance:
    def test_mls_pg(self):
        q = [MultiLineString([[(3, 0), (4, 0)], [(0, 5), (1, 5)]])]
        t = [box(0, 0, 2, 2)]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_ls_mpg(self):
        q = [LineString([(3, 0), (4, 0)])]
        t = [MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.LINESTRING, GeometryFamily.MULTIPOLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)

    def test_mls_mpg(self):
        q = [MultiLineString([[(3, 0), (4, 0)]])]
        t = [MultiPolygon([box(0, 0, 2, 2)])]
        gpu = _compute_distances(
            _make_owned(q), _make_owned(t),
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            GeometryFamily.MULTILINESTRING, GeometryFamily.MULTIPOLYGON,
        )
        expected = shapely.distance(q[0], t[0])
        assert gpu[0] == pytest.approx(expected, abs=1e-10)


# ---- Nearest pipeline integration ----

class TestNearestPipelineSegmentDistance:
    def test_polygon_nearest_polygon(self):
        import geopandas

        polys = [box(0, 0, 1, 1), box(5, 5, 6, 6), box(10, 0, 11, 1)]
        query_polys = [box(2, 0, 3, 1), box(8, 4, 9, 5)]

        tree_series = geopandas.GeoSeries(polys)
        query_series = geopandas.GeoSeries(query_polys)

        result = tree_series.sindex.nearest(query_series, return_all=True)

        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            actual_dist = shapely.distance(query_polys[qi], polys[ti])
            all_dists = [shapely.distance(query_polys[qi], p) for p in polys]
            assert actual_dist == pytest.approx(min(all_dists), abs=1e-10)

    def test_line_nearest_line(self):
        import geopandas

        tree_lines = [
            LineString([(0, 0), (1, 0)]),
            LineString([(5, 5), (6, 5)]),
            LineString([(10, 0), (11, 0)]),
        ]
        query_lines = [
            LineString([(2, 0), (3, 0)]),
            LineString([(7, 5), (8, 5)]),
        ]

        tree_series = geopandas.GeoSeries(tree_lines)
        query_series = geopandas.GeoSeries(query_lines)

        result = tree_series.sindex.nearest(query_series, return_all=True)

        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            actual_dist = shapely.distance(query_lines[qi], tree_lines[ti])
            all_dists = [shapely.distance(query_lines[qi], ln) for ln in tree_lines]
            assert actual_dist == pytest.approx(min(all_dists), abs=1e-10)

    def test_line_nearest_polygon(self):
        """Cross-type nearest: lines querying against polygons."""
        import geopandas

        polys = [box(0, 0, 2, 2), box(10, 10, 12, 12)]
        query_lines = [LineString([(3, 1), (4, 1)]), LineString([(11, 8), (11, 9)])]

        tree_series = geopandas.GeoSeries(polys)
        query_series = geopandas.GeoSeries(query_lines)

        result = tree_series.sindex.nearest(query_series, return_all=True)

        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            actual_dist = shapely.distance(query_lines[qi], polys[ti])
            all_dists = [shapely.distance(query_lines[qi], p) for p in polys]
            assert actual_dist == pytest.approx(min(all_dists), abs=1e-10)
