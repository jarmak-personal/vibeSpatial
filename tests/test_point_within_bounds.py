from __future__ import annotations

import pytest
from shapely.geometry import Point, Polygon, box

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    DeviceSnapshot,
    ExecutionMode,
    MonitoringBackend,
    from_shapely_geometries,
)
from vibespatial.kernels.predicates.point_within_bounds import point_within_bounds
from vibespatial.testing import compare_with_shapely


def point_within_bounds_reference(points, polygons_or_bounds):
    results = []
    for point, other in zip(points, polygons_or_bounds, strict=True):
        if point is None or other is None:
            results.append(None)
            continue
        if point.is_empty:
            results.append(False)
            continue
        if hasattr(other, "is_empty"):
            if other.is_empty:
                results.append(False)
                continue
            minx, miny, maxx, maxy = other.bounds
        else:
            minx, miny, maxx, maxy = other
        results.append(minx <= point.x <= maxx and miny <= point.y <= maxy)
    return results


@compare_with_shapely(reference=point_within_bounds_reference, handle_empty=True)
def test_point_within_bounds_matches_polygon_bounds(oracle_runner) -> None:
    points = [None, Point(), Point(1, 1), Point(8, 2), Point(9, 4)]
    polygons = [box(0, 0, 2, 2), Polygon(), None, box(5, 1, 9, 3), box(0, 0, 4, 4)]

    comparison = oracle_runner(
        point_within_bounds,
        points,
        polygons,
        dispatch_mode=ExecutionMode.CPU,
    )

    assert comparison.mismatches == ()


@compare_with_shapely(reference=point_within_bounds_reference, handle_empty=True)
def test_point_within_bounds_accepts_explicit_bounds(oracle_runner) -> None:
    points = [Point(0, 0), Point(3, 3), None, Point()]
    bounds = [(-1.0, -1.0, 1.0, 1.0), (4.0, 4.0, 5.0, 5.0), None, (0.0, 0.0, 2.0, 2.0)]

    comparison = oracle_runner(
        point_within_bounds,
        points,
        bounds,
        dispatch_mode=ExecutionMode.CPU,
    )

    assert comparison.mismatches == ()


@pytest.mark.cpu_fallback
def test_point_within_bounds_auto_fallback_is_visible(monkeypatch) -> None:
    import vibespatial.adaptive_runtime as adaptive_runtime

    def fake_snapshot(**_kwargs):
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="test snapshot",
        )

    monkeypatch.setattr(adaptive_runtime, "capture_device_snapshot", fake_snapshot)
    adaptive_runtime.invalidate_snapshot_cache()
    points = from_shapely_geometries([Point(1, 1)] * 10_001)
    polygons = from_shapely_geometries([box(0, 0, 2, 2)] * 10_001)

    result = point_within_bounds(points, polygons, dispatch_mode=ExecutionMode.AUTO)

    assert result[0] is True
    report = points.diagnostics_report()
    assert any("no GPU variant registered" in reason for reason in report["runtime_history"])


def test_point_within_bounds_explicit_gpu_request_fails_without_variant(monkeypatch) -> None:
    import vibespatial.adaptive_runtime as adaptive_runtime

    def fake_snapshot(**_kwargs):
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="test snapshot",
        )

    monkeypatch.setattr(adaptive_runtime, "capture_device_snapshot", fake_snapshot)
    adaptive_runtime.invalidate_snapshot_cache()

    with pytest.raises(NotImplementedError, match="point_within_bounds has no GPU variant"):
        point_within_bounds(
            [Point(0, 0)],
            [box(-1, -1, 1, 1)],
            dispatch_mode=ExecutionMode.GPU,
        )
