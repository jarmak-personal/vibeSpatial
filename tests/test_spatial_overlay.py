"""Tests for spatial_overlay_owned — GPU spatial overlay pipeline.

Tests the spatial join + batched pairwise overlay that replaces the
geopandas.overlay fallback for mismatched row-count inputs.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import shapely
from shapely.geometry import box

from vibespatial import from_shapely_geometries, spatial_overlay_owned
from vibespatial.overlay.assemble import _overlay_intersection_rectangles_gpu
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events
from vibespatial.runtime.residency import Residency


def _non_empty_geoms(owned):
    """Extract non-empty geometries from an OwnedGeometryArray."""
    return [g for g in owned.to_shapely() if g is not None and not g.is_empty]


class TestSpatialOverlayIntersection:
    """Intersection overlay tests."""

    def test_rectangle_fast_path_stays_device_resident(self, strict_device_guard):
        """Rectangle fast path should build device-resident output with lazy host metadata."""
        left = from_shapely_geometries([box(0, 0, 2, 2)], residency=Residency.DEVICE)
        right = from_shapely_geometries([box(1, 1, 3, 3)], residency=Residency.DEVICE)

        result = _overlay_intersection_rectangles_gpu(
            left,
            right,
            requested=ExecutionMode.GPU,
        )

        assert result is not None
        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

    def test_rectangle_fast_path_empty_stays_device_resident(self, strict_device_guard):
        """Disjoint rectangle fast path should return an empty device-resident result."""
        left = from_shapely_geometries([box(0, 0, 1, 1)], residency=Residency.DEVICE)
        right = from_shapely_geometries([box(2, 2, 3, 3)], residency=Residency.DEVICE)

        result = _overlay_intersection_rectangles_gpu(
            left,
            right,
            requested=ExecutionMode.GPU,
        )

        assert result is not None
        assert result.row_count == 0
        assert result.residency is Residency.DEVICE
        assert result._validity is None
        assert result._tags is None
        assert result._family_row_offsets is None

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


class TestBatchedSHClip:
    """Tests for lyy.18: batched SH clip for boundary-crossing simple polygons."""

    def test_boundary_crossing_simple_clip(self):
        """Boundary-crossing polygons clipped by a simple rectangle should
        produce correct results via the batched SH path."""
        # 6 polygons: 2 fully inside, 2 crossing boundary, 2 outside.
        left_geoms = [
            box(2, 1, 3, 2),    # fully inside [1, 0, 5, 4]
            box(3, 2, 4, 3),    # fully inside
            box(0, 1, 2, 3),    # crosses left boundary at x=1
            box(4, 1, 6, 3),    # crosses right boundary at x=5
            box(-2, 0, -1, 1),  # fully outside (left)
            box(6, 0, 7, 1),    # fully outside (right)
        ]
        right_geoms = [box(1, 0, 5, 4)]  # simple rectangle clip

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        # Expected: 4 results (2 fully inside + 2 boundary-crossing clipped).
        # The 2 outside polygons should not appear.
        assert len(result_geoms) == 4

        # Verify each result is a valid intersection.
        clip = right_geoms[0]
        expected_areas = []
        for lg in left_geoms:
            inter = lg.intersection(clip)
            if not inter.is_empty:
                expected_areas.append(inter.area)
        result_areas = sorted(g.area for g in result_geoms)
        expected_areas.sort()
        assert len(result_areas) == len(expected_areas)
        for ra, ea in zip(result_areas, expected_areas):
            assert abs(ra - ea) < 1e-10, f"Area mismatch: {ra} vs {ea}"

    def test_all_boundary_crossing_simple_clip(self):
        """When all polygons cross the boundary, all go through SH clip."""
        left_geoms = [
            box(0, 0, 2, 2),   # crosses at x=1
            box(4, 0, 6, 2),   # crosses at x=5
            box(1, -1, 3, 1),  # crosses at y=0
            box(2, 3, 4, 5),   # crosses at y=4
        ]
        right_geoms = [box(1, 0, 5, 4)]  # simple rectangle

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        # All 4 should produce non-empty intersections.
        assert len(result_geoms) == 4

        clip = right_geoms[0]
        for g in result_geoms:
            assert clip.contains(g) or g.within(clip)

    def test_complex_clip_skips_sh(self):
        """When the clip polygon has holes, SH tier should be skipped
        and results should still be correct via overlay pipeline."""
        # Clip polygon with a hole.
        outer = box(0, 0, 10, 10)
        inner = box(3, 3, 7, 7)
        clip_with_hole = outer.difference(inner)

        left_geoms = [box(1, 1, 5, 5), box(5, 5, 9, 9)]
        right_geoms = [clip_with_hole]

        left = from_shapely_geometries(left_geoms)
        right = from_shapely_geometries(right_geoms)

        result = spatial_overlay_owned(left, right, how="intersection")
        result_geoms = _non_empty_geoms(result)

        # Verify correctness against Shapely.
        expected = []
        for lg in left_geoms:
            inter = lg.intersection(clip_with_hole)
            if not inter.is_empty:
                expected.append(inter.area)
        result_areas = sorted(g.area for g in result_geoms)
        expected.sort()
        assert len(result_areas) == len(expected)
        for ra, ea in zip(result_areas, expected):
            assert abs(ra - ea) < 1e-6, f"Area mismatch: {ra} vs {ea}"


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

    def test_centroid_filter_failure_propagates(self, monkeypatch):
        left = from_shapely_geometries([box(i, 0, i + 0.8, 0.8) for i in range(120)])
        right = from_shapely_geometries([shapely.Polygon([(0, 0), (120, 0), (60, 20)])])

        from vibespatial.overlay import gpu as overlay_gpu

        def _force_owned_dispatch_fallback(*args, **kwargs):
            raise NotImplementedError("force shapely overlay fallback")

        monkeypatch.setattr(
            overlay_gpu,
            "_combine_bypass_results",
            _force_owned_dispatch_fallback,
        )

        real_plan_dispatch_selection = overlay_gpu.plan_dispatch_selection

        def _force_gpu_centroid_plan(**kwargs):
            if kwargs.get("kernel_name") == "polygon_centroid":
                return SimpleNamespace(selected=overlay_gpu.ExecutionMode.GPU)
            return real_plan_dispatch_selection(**kwargs)

        monkeypatch.setattr(overlay_gpu, "plan_dispatch_selection", _force_gpu_centroid_plan)

        def _boom(*args, **kwargs):
            raise RuntimeError("centroid-filter-boom")

        monkeypatch.setattr(
            "vibespatial.constructive.polygon.polygon_centroids_owned",
            _boom,
        )

        with pytest.raises(RuntimeError, match="centroid-filter-boom"):
            spatial_overlay_owned(left, right, how="intersection")


class TestOverlayDispatchEventWorkloadShape:
    """Verify dispatch events for overlay include workload_shape in detail (nsf.5)."""

    def test_broadcast_right_dispatch_event_has_workload_shape(self):
        """N-vs-1 pattern should record workload_shape=broadcast_right."""
        left = from_shapely_geometries([box(i, 0, i + 0.5, 0.5) for i in range(5)])
        right = from_shapely_geometries([box(0, 0, 10, 10)])

        clear_dispatch_events()
        spatial_overlay_owned(left, right, how="intersection")
        events = get_dispatch_events(clear=True)

        overlay_events = [
            e for e in events if e.surface == "geopandas.spatial_overlay"
        ]
        assert overlay_events, "Expected at least one spatial_overlay dispatch event"
        ev = overlay_events[0]
        assert "workload_shape=broadcast_right" in ev.detail

    def test_pairwise_dispatch_event_has_workload_shape(self):
        """Row-matched case should record workload_shape=pairwise."""
        left = from_shapely_geometries([box(0, 0, 2, 2)])
        right = from_shapely_geometries([box(1, 1, 3, 3)])

        clear_dispatch_events()
        spatial_overlay_owned(left, right, how="intersection")
        events = get_dispatch_events(clear=True)

        overlay_events = [
            e for e in events if e.surface == "geopandas.spatial_overlay"
        ]
        assert overlay_events, "Expected at least one spatial_overlay dispatch event"
        ev = overlay_events[0]
        assert "workload_shape=pairwise" in ev.detail

    def test_n_vs_m_dispatch_event_has_workload_shape(self):
        """N-vs-M case should record workload_shape=per_group (strategy name fallback)."""
        left = from_shapely_geometries([box(i, 0, i + 0.9, 0.9) for i in range(4)])
        right = from_shapely_geometries([box(0, 0, 2, 2), box(2, 0, 4, 2)])

        clear_dispatch_events()
        spatial_overlay_owned(left, right, how="intersection")
        events = get_dispatch_events(clear=True)

        overlay_events = [
            e for e in events if e.surface == "geopandas.spatial_overlay"
        ]
        assert overlay_events, "Expected at least one spatial_overlay dispatch event"
        ev = overlay_events[0]
        assert "workload_shape=per_group" in ev.detail


class TestSelectOverlayStrategyWorkloadShape:
    """Unit tests for select_overlay_strategy WorkloadShape integration (nsf.5)."""

    def test_broadcast_right_uses_shared_enum(self):
        from vibespatial.overlay.strategies import select_overlay_strategy
        from vibespatial.runtime.crossover import WorkloadShape

        left = from_shapely_geometries([box(i, 0, i + 1, 1) for i in range(5)])
        right = from_shapely_geometries([box(0, 0, 10, 10)])

        strategy = select_overlay_strategy(left, right, "intersection")
        assert strategy.name == "broadcast_right"
        assert strategy.workload_shape is WorkloadShape.BROADCAST_RIGHT

    def test_broadcast_left_has_no_shared_enum(self):
        from vibespatial.overlay.strategies import select_overlay_strategy

        left = from_shapely_geometries([box(0, 0, 10, 10)])
        right = from_shapely_geometries([box(i, 0, i + 1, 1) for i in range(5)])

        strategy = select_overlay_strategy(left, right, "intersection")
        assert strategy.name == "broadcast_left"
        assert strategy.workload_shape is None

    def test_pairwise_uses_shared_enum(self):
        from vibespatial.overlay.strategies import select_overlay_strategy
        from vibespatial.runtime.crossover import WorkloadShape

        left = from_shapely_geometries([box(0, 0, 1, 1), box(2, 0, 3, 1)])
        right = from_shapely_geometries([box(0, 0, 2, 2), box(2, 0, 4, 2)])

        strategy = select_overlay_strategy(left, right, "intersection")
        assert strategy.name == "per_group"
        assert strategy.workload_shape is WorkloadShape.PAIRWISE

    def test_n_vs_m_has_no_shared_enum(self):
        from vibespatial.overlay.strategies import select_overlay_strategy

        left = from_shapely_geometries([box(i, 0, i + 1, 1) for i in range(3)])
        right = from_shapely_geometries([box(0, 0, 2, 2), box(2, 0, 4, 2)])

        strategy = select_overlay_strategy(left, right, "intersection")
        assert strategy.name == "per_group"
        assert strategy.workload_shape is None
