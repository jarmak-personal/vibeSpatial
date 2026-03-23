"""OGC validity tests for is_valid_owned.

Tests that is_valid_owned matches Shapely's is_valid for:
- Ring self-intersection (bowtie, Phase A)
- Hole containment (Phase B)
- Ring crossing / overlap / multi-touch (Phase C)
- Full degeneracy corpus
- Mixed geometry types

Each test is parametrized across CPU and GPU dispatch modes.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from vibespatial.constructive.validity import is_valid_owned
from vibespatial.geometry.owned import from_shapely_geometries


def _build_owned(*geoms) -> tuple:
    """Build an OwnedGeometryArray from Shapely geometries.

    Returns (owned, shapely_array) for oracle comparison.
    """
    geom_array = np.array(list(geoms), dtype=object)
    owned = from_shapely_geometries(list(geoms))
    return owned, geom_array


# ---------------------------------------------------------------------------
# Phase A: Ring self-intersection
# ---------------------------------------------------------------------------

class TestRingSelfIntersection:
    """is_valid must detect self-intersecting rings (bowties)."""

    def test_bowtie_polygon_is_invalid(self, dispatch_mode):
        """Bowtie polygon has a self-crossing exterior ring."""
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        owned, geoms = _build_owned(bowtie)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(bowtie)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_simple_polygon_is_valid(self, dispatch_mode):
        """A simple square polygon is valid."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        owned, geoms = _build_owned(square)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_donut_polygon_is_valid(self, dispatch_mode):
        """A polygon with a properly-contained hole is valid."""
        donut = Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
        )
        owned, geoms = _build_owned(donut)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_duplicate_vertex_polygon_is_valid(self, dispatch_mode):
        """Repeated adjacent vertices do not make a polygon invalid."""
        dup = Polygon([(0, 0), (4, 0), (4, 4), (4, 4), (0, 4), (0, 0)])
        owned, geoms = _build_owned(dup)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(dup)])
        np.testing.assert_array_equal(result, expected)

    def test_multipolygon_with_bowtie_part(self, dispatch_mode):
        """A MultiPolygon is invalid if any part has a self-crossing ring."""
        valid_poly = Polygon([(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)])
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        mp = MultiPolygon([valid_poly, bowtie])
        owned, geoms = _build_owned(mp)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(mp)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_figure_eight_self_intersection(self, dispatch_mode):
        """A figure-8 polygon (crosses at center point)."""
        fig8 = Polygon([(0, 0), (2, 1), (4, 0), (4, 2), (2, 1), (0, 2), (0, 0)])
        owned, geoms = _build_owned(fig8)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(fig8)])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_valid_invalid_batch(self, dispatch_mode):
        """Batch with valid and invalid polygons."""
        valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        valid2 = Polygon([(5, 5), (7, 5), (7, 7), (5, 7), (5, 5)])
        owned, geoms = _build_owned(valid, bowtie, valid2)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = shapely.is_valid(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Mixed geometry types and edge cases
# ---------------------------------------------------------------------------

class TestMixedTypes:
    """is_valid works correctly across geometry types."""

    def test_points_always_valid(self, dispatch_mode):
        owned, geoms = _build_owned(Point(0, 0), Point(1, 1))
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        np.testing.assert_array_equal(result, [True, True])

    def test_linestring_min_coords(self, dispatch_mode):
        valid_line = LineString([(0, 0), (1, 1)])
        owned, geoms = _build_owned(valid_line)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_mixed_geometry_types(self, dispatch_mode):
        """Array with points, lines, and polygons."""
        pt = Point(0, 0)
        line = LineString([(0, 0), (1, 1)])
        valid_poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        owned, geoms = _build_owned(pt, line, valid_poly, bowtie)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = shapely.is_valid(geoms)
        np.testing.assert_array_equal(result, expected)

    def test_empty_polygon_is_valid(self, dispatch_mode):
        owned, geoms = _build_owned(Polygon())
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_null_geometry_is_valid(self, dispatch_mode):
        """Null geometries return True (matching Shapely convention)."""
        owned, geoms = _build_owned(None)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)


# ---------------------------------------------------------------------------
# Phase B: Hole-in-shell containment
# ---------------------------------------------------------------------------

class TestHoleInShellContainment:
    """is_valid must detect holes outside their polygon's exterior ring."""

    def test_hole_entirely_outside_shell_is_invalid(self, dispatch_mode):
        """A hole ring entirely outside the shell makes the polygon invalid."""
        # Shell is a small square, hole is far outside
        shell = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole = [(10, 10), (12, 10), (12, 12), (10, 12), (10, 10)]
        poly = Polygon(shell, [hole])
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_hole_properly_inside_shell_is_valid(self, dispatch_mode):
        """A properly contained hole (donut polygon) is valid."""
        donut = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]],
        )
        owned, geoms = _build_owned(donut)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(donut)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_multi_hole_one_outside_is_invalid(self, dispatch_mode):
        """A polygon with two holes, one inside and one outside, is invalid."""
        shell = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        hole_inside = [(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]
        hole_outside = [(30, 30), (32, 30), (32, 32), (30, 32), (30, 30)]
        poly = Polygon(shell, [hole_inside, hole_outside])
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_multi_hole_all_inside_is_valid(self, dispatch_mode):
        """A polygon with multiple properly-contained holes is valid."""
        shell = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
        hole1 = [(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]
        hole2 = [(5, 5), (9, 5), (9, 9), (5, 9), (5, 5)]
        poly = Polygon(shell, [hole1, hole2])
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_multipolygon_hole_outside_shell(self, dispatch_mode):
        """MultiPolygon is invalid if any part has a hole outside its shell."""
        valid_poly = Polygon(
            [(10, 10), (14, 10), (14, 14), (10, 14), (10, 10)],
        )
        shell = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]
        hole_outside = [(20, 20), (22, 20), (22, 22), (20, 22), (20, 20)]
        invalid_poly = Polygon(shell, [hole_outside])
        mp = MultiPolygon([valid_poly, invalid_poly])
        owned, geoms = _build_owned(mp)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(mp)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_batch_mixed_hole_validity(self, dispatch_mode):
        """Batch with valid donut and invalid hole-outside polygon."""
        valid = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
        )
        shell = [(20, 20), (24, 20), (24, 24), (20, 24), (20, 20)]
        hole_outside = [(50, 50), (52, 50), (52, 52), (50, 52), (50, 50)]
        invalid = Polygon(shell, [hole_outside])
        owned, geoms = _build_owned(valid, invalid)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = shapely.is_valid(geoms)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Degeneracy corpus
# ---------------------------------------------------------------------------

class TestDegeneracyCorpus:
    """is_valid matches Shapely for the degeneracy corpus geometries."""

    def test_bowtie_invalid_polygon(self, dispatch_mode):
        bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
        owned, geoms = _build_owned(bowtie)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(False)

    def test_donut_window_polygon(self, dispatch_mode):
        donut = Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(2, 2), (4, 2), (4, 4), (2, 4), (2, 2)]],
        )
        owned, geoms = _build_owned(donut)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_duplicate_vertex_polygon(self, dispatch_mode):
        dup = Polygon([(0, 0), (4, 0), (4, 4), (4, 4), (0, 4), (0, 0)])
        owned, geoms = _build_owned(dup)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        assert result[0] is np.bool_(True)

    def test_touching_hole_invalid_polygon(self, dispatch_mode):
        """Hole shares edge with shell (from degeneracy corpus) -> INVALID."""
        touching_hole = Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(0, 2), (2, 2), (2, 4), (0, 4), (0, 2)]],
        )
        owned, geoms = _build_owned(touching_hole)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(touching_hole)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)


# ---------------------------------------------------------------------------
# Phase C: Ring-pair interaction (crossing, overlap, multi-touch)
# ---------------------------------------------------------------------------

class TestRingPairInteraction:
    """is_valid must detect inter-ring crossings, overlaps, and multi-touch."""

    def test_hole_touches_shell_at_one_point_is_valid(self, dispatch_mode):
        """A hole touching the shell at exactly 1 vertex is VALID per OGC."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(0, 5), (3, 3), (3, 7), (0, 5)]],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_hole_touches_shell_at_two_points_is_invalid(self, dispatch_mode):
        """A hole touching the shell at 2 vertices disconnects the interior."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(0, 3), (3, 5), (0, 7), (0, 3)]],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_two_holes_touch_at_one_point_is_valid(self, dispatch_mode):
        """Two holes touching at 1 point is VALID per OGC."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[
                [(1, 1), (5, 1), (5, 5), (1, 5), (1, 1)],
                [(5, 5), (9, 5), (9, 9), (5, 9), (5, 5)],
            ],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_two_holes_share_an_edge_is_invalid(self, dispatch_mode):
        """Two holes sharing an edge is INVALID (self-intersection)."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[
                [(1, 1), (5, 1), (5, 5), (1, 5), (1, 1)],
                [(5, 1), (9, 1), (9, 5), (5, 5), (5, 1)],
            ],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_proper_ring_crossing_is_invalid(self, dispatch_mode):
        """Hole ring extends outside the shell (proper crossing) -> INVALID."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(-1, 3), (5, 3), (5, 7), (-1, 7), (-1, 3)]],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_hole_shares_edge_with_shell_is_invalid(self, dispatch_mode):
        """Hole shares a collinear edge with the shell -> INVALID."""
        poly = Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(0, 2), (2, 2), (2, 4), (0, 4), (0, 2)]],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_multipolygon_with_invalid_ring_interaction(self, dispatch_mode):
        """MultiPolygon with a part that has a touching-hole violation."""
        valid_poly = Polygon([(10, 10), (14, 10), (14, 14), (10, 14), (10, 10)])
        invalid_poly = Polygon(
            shell=[(0, 0), (6, 0), (6, 6), (0, 6), (0, 0)],
            holes=[[(0, 2), (2, 2), (2, 4), (0, 4), (0, 2)]],
        )
        mp = MultiPolygon([valid_poly, invalid_poly])
        owned, geoms = _build_owned(mp)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(mp)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)

    def test_valid_multi_hole_polygon(self, dispatch_mode):
        """Polygon with multiple properly-separated holes is valid."""
        poly = Polygon(
            shell=[(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
            holes=[
                [(1, 1), (4, 1), (4, 4), (1, 4), (1, 1)],
                [(6, 6), (9, 6), (9, 9), (6, 9), (6, 6)],
                [(11, 11), (14, 11), (14, 14), (11, 14), (11, 11)],
            ],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(True)

    def test_vertex_on_edge_two_touches_is_invalid(self, dispatch_mode):
        """Hole touches shell at 2 points (vertex-on-edge + vertex) -> INVALID."""
        poly = Polygon(
            shell=[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            holes=[[(5, 0), (8, 5), (5, 10), (2, 5), (5, 0)]],
        )
        owned, geoms = _build_owned(poly)
        result = is_valid_owned(owned, dispatch_mode=dispatch_mode)
        expected = np.array([shapely.is_valid(poly)])
        np.testing.assert_array_equal(result, expected)
        assert result[0] is np.bool_(False)
