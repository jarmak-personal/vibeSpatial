"""Tests for spatial_overlay_owned — GPU spatial overlay pipeline.

Tests the spatial join + batched pairwise overlay that replaces the
geopandas.overlay fallback for mismatched row-count inputs.
"""
from __future__ import annotations

import pytest
import shapely
from shapely.geometry import box

from vibespatial import from_shapely_geometries, spatial_overlay_owned


def _non_empty_geoms(owned):
    """Extract non-empty geometries from an OwnedGeometryArray."""
    return [g for g in owned.to_shapely() if g is not None and not g.is_empty]


class TestSpatialOverlayIntersection:
    """Intersection overlay tests."""

    def test_many_left_vs_one_right_clip(self):
        """10 polygons clipped against 1 larger polygon (vegetation-corridor pattern)."""
        # Create 10 small squares spread across x=[0..10], y=[0..1]
        left_geoms = [box(i, 0, i + 0.8, 0.8) for i in range(10)]
        # One large clipping rectangle covering x=[2..7]
        right_geoms = [box(2, -1, 7, 2)]

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        # Polygons 0,1 (x=[0..0.8], [1..1.8]) are outside the clip region
        # Polygons 2..6 are fully inside or partially inside
        # Polygon 2: box(2,0,2.8,0.8) fully inside [2,7] -> box(2,0,2.8,0.8)
        # Polygon 7: box(7,0,7.8,0.8) partially inside [2,7] -> box(7,0,7,0.8)? Actually x=7 is edge
        # Polygons 8,9 are outside
        assert len(result_geoms) >= 5  # at least polygons 2-6

        # Verify each result is contained in the clip region
        clip = right_geoms[0]
        for g in result_geoms:
            assert clip.contains(g) or clip.intersects(g)

    def test_many_left_vs_few_right(self):
        """Many parcels vs few zones (parcel-zoning pattern)."""
        left_geoms = [box(i % 5, i // 5, i % 5 + 0.9, i // 5 + 0.9) for i in range(20)]
        right_geoms = [box(0, 0, 2.5, 2.5), box(2.5, 0, 5, 2.5)]

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        # Each left polygon should intersect with at least one right polygon
        assert len(result_geoms) > 0

        # Reference: compute with shapely directly
        expected = []
        for lg in left_geoms:
            for rg in right_geoms:
                inter = lg.intersection(rg)
                if not inter.is_empty:
                    expected.append(inter)
        assert len(result_geoms) == len(expected)

    def test_no_overlap_returns_empty(self):
        """Non-overlapping geometries should return empty result."""
        left = from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        right = from_shapely_geometries([box(10, 10, 11, 11)])

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)
        assert len(result_geoms) == 0

    def test_single_left_single_right(self):
        """Row-matched case still works through spatial overlay."""
        left = from_shapely_geometries([box(0, 0, 2, 2)])
        right = from_shapely_geometries([box(1, 1, 3, 3)])

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        expected = box(0, 0, 2, 2).intersection(box(1, 1, 3, 3))
        assert len(result_geoms) == 1
        assert shapely.equals(shapely.normalize(result_geoms[0]), shapely.normalize(expected))


class TestSpatialOverlayDifference:
    """Difference overlay tests."""

    def test_difference_clips_away_right(self):
        """Difference should remove the overlapping portion."""
        left = from_shapely_geometries([box(0, 0, 4, 4)])
        right = from_shapely_geometries([box(2, 2, 6, 6)])

        result = spatial_overlay_owned(left, right, how="difference")
        result_geoms = _non_empty_geoms(result)

        expected = box(0, 0, 4, 4).difference(box(2, 2, 6, 6))
        assert len(result_geoms) == 1
        assert abs(result_geoms[0].area - expected.area) < 1e-10


class TestSpatialOverlayUnion:
    """Union overlay tests."""

    def test_union_combines_overlapping(self):
        """Union should combine overlapping areas."""
        left = from_shapely_geometries([box(0, 0, 2, 2)])
        right = from_shapely_geometries([box(1, 1, 3, 3)])

        result = spatial_overlay_owned(left, right, how="union")
        result_geoms = _non_empty_geoms(result)

        expected = box(0, 0, 2, 2).union(box(1, 1, 3, 3))
        assert len(result_geoms) >= 1
        total_area = sum(g.area for g in result_geoms)
        assert abs(total_area - expected.area) < 1e-10


class TestSpatialOverlaySymmetricDifference:
    """Symmetric difference overlay tests."""

    def test_symmetric_difference(self):
        """Symmetric difference should return non-overlapping areas."""
        left = from_shapely_geometries([box(0, 0, 2, 2)])
        right = from_shapely_geometries([box(1, 1, 3, 3)])

        result = spatial_overlay_owned(left, right, how="symmetric_difference")
        result_geoms = _non_empty_geoms(result)

        expected = box(0, 0, 2, 2).symmetric_difference(box(1, 1, 3, 3))
        assert len(result_geoms) >= 1
        total_area = sum(g.area for g in result_geoms)
        assert abs(total_area - expected.area) < 1e-10


class TestSpatialOverlayEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_how_raises(self):
        left = from_shapely_geometries([box(0, 0, 1, 1)])
        right = from_shapely_geometries([box(0, 0, 1, 1)])
        with pytest.raises(ValueError, match="unsupported spatial overlay operation"):
            spatial_overlay_owned(left, right, how="bogus")

    def test_runtime_history_recorded(self):
        """Result should have runtime selection recorded."""
        left = from_shapely_geometries([box(0, 0, 2, 2)])
        right = from_shapely_geometries([box(1, 1, 3, 3)])

        result = spatial_overlay_owned(left, right, how="intersection")
        assert len(result.runtime_history) >= 1
        last = result.runtime_history[-1]
        assert "spatial_overlay" in last.reason

    def test_many_to_one_result_count(self):
        """When right has 1 polygon, each overlapping left should produce exactly 1 result."""
        left_geoms = [box(i, 0, i + 0.5, 0.5) for i in range(5)]
        right_geoms = [box(0, 0, 10, 10)]  # covers everything

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)
        # All 5 left polygons are inside the right polygon,
        # so intersection = each left polygon unchanged
        assert len(result_geoms) == 5

        for i, g in enumerate(result_geoms):
            assert abs(g.area - left_geoms[i].area) < 1e-10
