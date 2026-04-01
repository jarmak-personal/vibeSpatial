from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from vibespatial import has_gpu_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import from_shapely_geometries

pytestmark = pytest.mark.skipif(not has_gpu_runtime(), reason="GPU required")


def _make_owned(geoms):
    return from_shapely_geometries(geoms)


def _compute_de9im(query_owned, tree_owned, left_idx, right_idx, query_family, tree_family):
    from vibespatial.predicates.polygon import compute_polygon_de9im_gpu
    return compute_polygon_de9im_gpu(
        query_owned, tree_owned,
        np.asarray(left_idx, dtype=np.int32),
        np.asarray(right_idx, dtype=np.int32),
        query_family=query_family,
        tree_family=tree_family,
    )


def _eval_predicate(masks, predicate):
    from vibespatial.predicates.polygon import evaluate_predicate_from_de9im
    return evaluate_predicate_from_de9im(masks, predicate)


class TestDE9IMBitmask:
    def test_disjoint_boxes(self):
        q = [box(0, 0, 1, 1)]
        t = [box(3, 0, 4, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "intersects")[0]

    def test_overlapping_boxes(self):
        q = [box(0, 0, 2, 2)]
        t = [box(1, 1, 3, 3)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "overlaps")[0]
        assert not _eval_predicate(masks, "contains")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_contained_box(self):
        q = [box(0, 0, 10, 10)]
        t = [box(2, 2, 3, 3)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "contains_properly")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert _eval_predicate(masks, "covers")[0]

    def test_within(self):
        q = [box(2, 2, 3, 3)]
        t = [box(0, 0, 10, 10)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "covered_by")[0]

    def test_touching_boxes(self):
        q = [box(0, 0, 1, 1)]
        t = [box(1, 0, 2, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "touches")[0]
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "overlaps")[0]

    def test_identical_boxes(self):
        q = [box(0, 0, 1, 1)]
        t = [box(0, 0, 1, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "contains")[0]
        assert not _eval_predicate(masks, "contains_properly")[0]
        assert _eval_predicate(masks, "within")[0]
        assert _eval_predicate(masks, "covers")[0]
        assert _eval_predicate(masks, "covered_by")[0]
        assert not _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "touches")[0]

    def test_polygon_with_hole(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
        q = [Polygon(outer, [hole])]
        t = [box(4, 4, 6, 6)]  # inside the hole
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "disjoint")[0]
        assert not _eval_predicate(masks, "intersects")[0]


class TestMultiPolygonDE9IM:
    def test_mpg_mpg_overlapping(self):
        q = [MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)])]
        t = [MultiPolygon([box(1, 1, 3, 3)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.MULTIPOLYGON, GeometryFamily.MULTIPOLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]

    def test_pg_mpg_contained(self):
        q = [box(0, 0, 10, 10)]
        t = [MultiPolygon([box(1, 1, 2, 2), box(3, 3, 4, 4)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON,
        )
        assert _eval_predicate(masks, "contains")[0]

    def test_mpg_pg_swap(self):
        """MPG × PG swap: within should work correctly after transpose."""
        q = [MultiPolygon([box(1, 1, 2, 2)])]
        t = [box(0, 0, 10, 10)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.MULTIPOLYGON, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "contains")[0]


class TestBatchPredicates:
    """Test multiple candidate pairs in a single kernel launch."""

    def test_batch_intersects(self):
        polys = [box(0, 0, 1, 1), box(2, 2, 3, 3), box(0, 0, 5, 5), box(10, 10, 11, 11)]
        owned = _make_owned(polys)

        left = np.array([0, 0, 2, 3], dtype=np.int32)
        right = np.array([1, 2, 3, 0], dtype=np.int32)

        masks = _compute_de9im(
            owned, owned, left, right,
            GeometryFamily.POLYGON, GeometryFamily.POLYGON,
        )
        gpu_result = _eval_predicate(masks, "intersects")

        expected = np.array([
            shapely.intersects(polys[li], polys[ri])
            for li, ri in zip(left, right)
        ])
        np.testing.assert_array_equal(gpu_result, expected)


class TestQueryPipelineIntegration:
    """Test that the query pipeline uses GPU DE-9IM for polygon predicates."""

    def test_polygon_query_intersects(self):
        import geopandas

        polys = [box(0, 0, 2, 2), box(1, 1, 3, 3), box(5, 5, 7, 7), box(10, 10, 12, 12)]
        series = geopandas.GeoSeries(polys)

        query_geom = [box(0, 0, 1.5, 1.5)]
        query_series = geopandas.GeoSeries(query_geom)

        result = series.sindex.query(query_series, predicate="intersects")

        # Query box overlaps polys[0] and polys[1] only.
        result_right = set(result[1]) if result.ndim == 2 else set(result)
        expected = {i for i, p in enumerate(polys) if shapely.intersects(query_geom[0], p)}
        assert result_right == expected

    def test_polygon_self_query_intersects(self):
        """Self-join: all polygon pairs — the 100K benchmark workload shape."""
        import geopandas

        polys = [box(i, 0, i + 1.5, 1.5) for i in range(10)]
        series = geopandas.GeoSeries(polys)

        result = series.sindex.query(series, predicate="intersects")

        # Verify all result pairs actually intersect.
        for col in range(result.shape[1]):
            qi, ti = result[0, col], result[1, col]
            assert shapely.intersects(polys[qi], polys[ti]), f"pair ({qi}, {ti}) should intersect"

        # Verify no intersecting pairs are missing (self-join returns upper triangle only).
        result_set = set(zip(result[0], result[1]))
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if shapely.intersects(polys[i], polys[j]):
                    has_pair = (i, j) in result_set or (j, i) in result_set
                    assert has_pair, f"missing intersecting pair ({i}, {j})"

    def test_polygon_query_contains(self):
        import geopandas

        polys = [box(0, 0, 10, 10), box(1, 1, 2, 2), box(20, 20, 21, 21)]
        series = geopandas.GeoSeries(polys)

        query_geom = [box(0, 0, 10, 10)]
        query_series = geopandas.GeoSeries(query_geom)

        result = series.sindex.query(query_series, predicate="contains")

        result_right = set(result[1]) if result.ndim == 2 else set(result)
        expected = {i for i, p in enumerate(polys) if shapely.contains(query_geom[0], p)}
        assert result_right == expected


class TestLineDE9IM:
    """Test DE-9IM computation for line × polygon and line × line pairs."""

    def test_line_crossing_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0.5), (2, 0.5)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_inside_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(0.6, 0.5), (1.0, 0.5)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "within")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_disjoint_from_polygon(self):
        from shapely.geometry import LineString
        q = [LineString([(3, 0), (4, 0)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert not _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "disjoint")[0]

    def test_line_touching_polygon_boundary(self):
        q = [LineString([(0.5, 1), (0.5, 2)])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "touches")[0]
        assert not _eval_predicate(masks, "within")[0]

    def test_line_intersects_tiny_buffer_polygon_at_large_coordinates(self):
        q = [LineString([(970227.216003418, 145641.63360595703), (970273.9365844727, 145641.63360595703)])]
        t = [Point(970264.7347596437, 145641.63360595703).buffer(1e-8)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_crossing_lines(self):
        q = [LineString([(0, 0), (2, 2)])]
        t = [LineString([(0, 2), (2, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_parallel_lines(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (1, 0)])]
        t = [LineString([(0, 1), (1, 1)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert not _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "disjoint")[0]

    def test_collinear_overlapping_lines(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (2, 0)])]
        t = [LineString([(1, 0), (3, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert not _eval_predicate(masks, "disjoint")[0]

    def test_line_endpoint_touching_line(self):
        from shapely.geometry import LineString
        q = [LineString([(0, 0), (1, 0)])]
        t = [LineString([(1, 0), (2, 0)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.LINESTRING, GeometryFamily.LINESTRING,
        )
        assert _eval_predicate(masks, "intersects")[0]
        assert _eval_predicate(masks, "touches")[0]

    def test_multilinestring_polygon(self):
        from shapely.geometry import MultiLineString
        q = [MultiLineString([[(0.6, 0.5), (1.0, 0.5)], [(3, 3), (4, 4)]])]
        t = [box(0.5, 0, 1.5, 1)]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.MULTILINESTRING, GeometryFamily.POLYGON,
        )
        assert _eval_predicate(masks, "intersects")[0]

    def test_polygon_line_swap(self):
        """PG × LS swap: within should work correctly after transpose."""
        from shapely.geometry import LineString
        q = [box(0, 0, 10, 10)]
        t = [LineString([(1, 1), (2, 2)])]
        masks = _compute_de9im(
            _make_owned(q), _make_owned(t),
            [0], [0], GeometryFamily.POLYGON, GeometryFamily.LINESTRING,
        )
        # Polygon contains the line.
        assert _eval_predicate(masks, "contains")[0]
        assert _eval_predicate(masks, "intersects")[0]
