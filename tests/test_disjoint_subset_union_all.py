"""Tests for disjoint_subset_union_all GPU geometry collection assembly.

Verifies that the GPU path (CuPy buffer concatenation + offset chaining)
produces the same Shapely geometry as the CPU reference implementation
for homogeneous (same-family) and mixed-family inputs.

Covers:
  - Empty input (0 rows)
  - Single row (identity)
  - Null rows (skipped)
  - Empty geometries (skipped or passed through)
  - Homogeneous Point -> MultiPoint
  - Homogeneous LineString -> MultiLineString
  - Homogeneous Polygon -> MultiPolygon
  - Homogeneous MultiPoint -> MultiPoint (merged)
  - Homogeneous MultiLineString -> MultiLineString (merged)
  - Homogeneous MultiPolygon -> MultiPolygon (merged)
  - Mixed families -> fallback to CPU
  - GeoSeries.union_all(method="disjoint_subset") integration
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from vibespatial import GeoSeries, from_shapely_geometries, has_gpu_runtime
from vibespatial.constructive.union_all import disjoint_subset_union_all_owned
from vibespatial.runtime import ExecutionMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shapely_ref(geoms: list) -> object:
    """Reference: Shapely union_all on non-null geometries."""
    valid = [g for g in geoms if g is not None and not g.is_empty]
    if not valid:
        return GeometryCollection()
    arr = np.array(valid, dtype=object)
    return shapely.union_all(arr)


def _assert_geom_equal(actual, expected, *, tolerance: float = 1e-9):
    """Assert two Shapely geometries are structurally equal."""
    if expected.is_empty and actual.is_empty:
        return
    assert actual.geom_type == expected.geom_type, (
        f"geom_type mismatch: {actual.geom_type} != {expected.geom_type}"
    )
    assert actual.equals_exact(expected, tolerance), (
        f"geometry mismatch:\n  actual:   {actual}\n  expected: {expected}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Empty input, single row, all nulls."""

    def test_empty_input(self):
        """Zero rows -> empty geometry (empty Polygon since OGA doesn't support GC)."""
        owned = from_shapely_geometries([])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].is_empty

    def test_single_row_point(self):
        """Single Point -> result wraps that point as MultiPoint."""
        pt = Point(1, 2)
        owned = from_shapely_geometries([pt])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert shapely.equals(geoms[0], MultiPoint([pt]))

    def test_single_row_polygon(self):
        """Single Polygon -> result wraps that polygon as MultiPolygon."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        owned = from_shapely_geometries([poly])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert shapely.equals(geoms[0], MultiPolygon([poly]))

    def test_all_null_rows(self):
        """All null rows -> empty geometry."""
        owned = from_shapely_geometries([None, None, None])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        assert geoms[0].is_empty

    def test_null_rows_skipped(self):
        """Null rows are skipped; only valid geometries are assembled."""
        pts = [Point(1, 2), None, Point(3, 4)]
        owned = from_shapely_geometries(pts)
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPoint([Point(1, 2), Point(3, 4)])
        assert shapely.equals(geoms[0], expected)


# ---------------------------------------------------------------------------
# Homogeneous family assembly
# ---------------------------------------------------------------------------


class TestHomogeneousPointAssembly:
    """Point rows -> MultiPoint."""

    def test_multiple_points(self):
        pts = [Point(0, 0), Point(1, 1), Point(2, 3)]
        owned = from_shapely_geometries(pts)
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPoint(pts)
        assert shapely.equals(geoms[0], expected)

    def test_multipoint_inputs(self):
        """MultiPoint rows -> merged MultiPoint."""
        mp1 = MultiPoint([Point(0, 0), Point(1, 1)])
        mp2 = MultiPoint([Point(2, 2), Point(3, 3)])
        owned = from_shapely_geometries([mp1, mp2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)])
        assert shapely.equals(geoms[0], expected)


class TestHomogeneousLineStringAssembly:
    """LineString rows -> MultiLineString."""

    def test_multiple_linestrings(self):
        ls1 = LineString([(0, 0), (1, 1)])
        ls2 = LineString([(2, 2), (3, 3), (4, 4)])
        owned = from_shapely_geometries([ls1, ls2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiLineString([ls1, ls2])
        assert shapely.equals(geoms[0], expected)

    def test_multilinestring_inputs(self):
        """MultiLineString rows -> merged MultiLineString."""
        mls1 = MultiLineString([[(0, 0), (1, 1)], [(2, 2), (3, 3)]])
        mls2 = MultiLineString([[(4, 4), (5, 5)]])
        owned = from_shapely_geometries([mls1, mls2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiLineString([
            [(0, 0), (1, 1)],
            [(2, 2), (3, 3)],
            [(4, 4), (5, 5)],
        ])
        assert shapely.equals(geoms[0], expected)


class TestHomogeneousPolygonAssembly:
    """Polygon rows -> MultiPolygon."""

    def test_multiple_polygons(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        owned = from_shapely_geometries([p1, p2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon([p1, p2])
        assert shapely.equals(geoms[0], expected)

    def test_polygon_with_hole(self):
        """Polygon with holes assembled correctly."""
        shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(2, 2), (2, 4), (4, 4), (4, 2)]
        p1 = Polygon(shell, [hole])
        p2 = Polygon([(20, 20), (30, 20), (30, 30)])
        owned = from_shapely_geometries([p1, p2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon([p1, p2])
        assert shapely.equals(geoms[0], expected)

    def test_multipolygon_inputs(self):
        """MultiPolygon rows -> merged MultiPolygon."""
        mp1 = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ])
        mp2 = MultiPolygon([
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
        ])
        owned = from_shapely_geometries([mp1, mp2])
        result = disjoint_subset_union_all_owned(owned)
        assert result is not None
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
        ])
        assert shapely.equals(geoms[0], expected)


# ---------------------------------------------------------------------------
# Mixed families
# ---------------------------------------------------------------------------


class TestMixedFamilies:
    """Mixed family input returns None, signalling caller to use Shapely."""

    def test_point_and_polygon_mixed_returns_none(self):
        """Point + Polygon -> returns None (mixed families)."""
        pt = Point(5, 5)
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        owned = from_shapely_geometries([pt, poly])
        result = disjoint_subset_union_all_owned(owned)
        assert result is None

    def test_linestring_and_polygon_mixed_returns_none(self):
        """LineString + Polygon -> returns None (mixed families)."""
        ls = LineString([(0, 0), (1, 1)])
        poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 10)])
        owned = from_shapely_geometries([ls, poly])
        result = disjoint_subset_union_all_owned(owned)
        assert result is None

    def test_point_and_linestring_mixed(self):
        """Point + LineString merges to different targets -> returns None."""
        pt = Point(0, 0)
        ls = LineString([(1, 1), (2, 2)])
        owned = from_shapely_geometries([pt, ls])
        result = disjoint_subset_union_all_owned(owned)
        assert result is None

    def test_mixed_families_via_geoseries_still_works(self):
        """GeoSeries.union_all with mixed families falls through to Shapely."""
        pt = Point(5, 5)
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
        gs = GeoSeries([pt, poly])
        result = gs.union_all(method="disjoint_subset")
        # Shapely handles mixed families via GeometryCollection.
        ref = _shapely_ref([pt, poly])
        assert shapely.equals(result, ref)


# ---------------------------------------------------------------------------
# GeoSeries.union_all integration
# ---------------------------------------------------------------------------


class TestGeoSeriesIntegration:
    """Verify dispatch through GeoSeries.union_all(method='disjoint_subset')."""

    def test_geoseries_disjoint_subset_polygons(self):
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        gs = GeoSeries([p1, p2])
        result = gs.union_all(method="disjoint_subset")
        expected = MultiPolygon([p1, p2])
        assert shapely.equals(result, expected)

    def test_geoseries_disjoint_subset_points(self):
        gs = GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2)])
        result = gs.union_all(method="disjoint_subset")
        expected = MultiPoint([Point(0, 0), Point(1, 1), Point(2, 2)])
        assert shapely.equals(result, expected)

    def test_geoseries_disjoint_subset_with_nulls(self):
        gs = GeoSeries([Point(0, 0), None, Point(2, 2)])
        result = gs.union_all(method="disjoint_subset")
        expected = MultiPoint([Point(0, 0), Point(2, 2)])
        assert shapely.equals(result, expected)


# ---------------------------------------------------------------------------
# GPU-specific tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGPUExecution:
    """Verify GPU path specifically (requires CUDA runtime)."""

    def test_gpu_polygon_assembly(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        owned = from_shapely_geometries([p1, p2])
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon([p1, p2])
        assert shapely.equals(geoms[0], expected)

    def test_gpu_point_assembly(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        pts = [Point(i, i * 2) for i in range(10)]
        owned = from_shapely_geometries(pts)
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPoint(pts)
        assert shapely.equals(geoms[0], expected)

    def test_gpu_linestring_assembly(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        lines = [
            LineString([(0, 0), (1, 1)]),
            LineString([(2, 2), (3, 3), (4, 4)]),
            LineString([(5, 5), (6, 6)]),
        ]
        owned = from_shapely_geometries(lines)
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiLineString(lines)
        assert shapely.equals(geoms[0], expected)

    def test_gpu_multipolygon_merge(self):
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        mp1 = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ])
        mp2 = MultiPolygon([
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
        ])
        owned = from_shapely_geometries([mp1, mp2])
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon([
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
        ])
        assert shapely.equals(geoms[0], expected)

    def test_gpu_result_is_single_row_oga(self):
        """Output is a 1-row OwnedGeometryArray."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        pts = [Point(0, 0), Point(1, 1)]
        owned = from_shapely_geometries(pts)
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        assert result.row_count == 1

    def test_gpu_large_polygon_set(self):
        """Larger set of disjoint polygons to exercise batched concatenation."""
        if not has_gpu_runtime():
            pytest.skip("CUDA runtime not available")
        polys = [
            Polygon([
                (i * 10, 0), (i * 10 + 5, 0),
                (i * 10 + 5, 5), (i * 10, 5),
            ])
            for i in range(100)
        ]
        owned = from_shapely_geometries(polys)
        result = disjoint_subset_union_all_owned(
            owned, dispatch_mode=ExecutionMode.GPU,
        )
        geoms = result.to_shapely()
        assert len(geoms) == 1
        expected = MultiPolygon(polys)
        assert shapely.equals(geoms[0], expected)
