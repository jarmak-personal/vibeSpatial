from __future__ import annotations

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    DeviceSnapshot,
    ExecutionMode,
    MonitoringBackend,
    has_gpu_runtime,
    from_shapely_geometries,
)
from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon
from vibespatial.testing import compare_with_shapely, point_in_polygon_reference


@compare_with_shapely(reference=point_in_polygon_reference, handle_empty=True)
def test_point_in_polygon_matches_reference(oracle_runner) -> None:
    polygons = [
        box(0, 0, 2, 2),
        Polygon(),
        None,
        box(5, 1, 9, 3),
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
    ]
    points = [None, Point(), Point(1, 1), Point(8, 2), Point(4, 2)]

    comparison = oracle_runner(
        point_in_polygon,
        points,
        polygons,
        dispatch_mode=ExecutionMode.CPU,
    )

    assert comparison.mismatches == ()


@compare_with_shapely(reference=point_in_polygon_reference, handle_empty=True)
def test_point_in_polygon_supports_multipolygon_inputs(oracle_runner) -> None:
    polygons = [
        MultiPolygon([box(0, 0, 2, 2), box(10, 10, 12, 12)]),
        MultiPolygon([box(0, 0, 2, 2)]),
        MultiPolygon([]),
    ]
    points = [Point(1, 1), Point(3, 3), Point(0, 0)]

    comparison = oracle_runner(
        point_in_polygon,
        points,
        polygons,
        dispatch_mode=ExecutionMode.CPU,
    )

    assert comparison.mismatches == ()


@pytest.mark.cpu_fallback
def test_point_in_polygon_auto_fallback_is_visible(monkeypatch) -> None:
    import vibespatial.adaptive_runtime as adaptive_runtime

    def fake_snapshot(**_kwargs):
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=False,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="test snapshot",
        )

    monkeypatch.setattr(adaptive_runtime, "capture_device_snapshot", fake_snapshot)
    adaptive_runtime.invalidate_snapshot_cache()
    points = from_shapely_geometries([Point(1, 1)] * 10_001)
    polygons = from_shapely_geometries([box(0, 0, 2, 2)] * 10_001)

    result = point_in_polygon(points, polygons, dispatch_mode=ExecutionMode.AUTO)

    assert result[0] is True
    report = points.diagnostics_report()
    assert any("CPU fallback" in reason for reason in report["runtime_history"])


@pytest.mark.gpu
def test_point_in_polygon_explicit_gpu_matches_cpu_result() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    polygons = [
        box(0, 0, 2, 2),
        MultiPolygon([box(5, 5, 8, 8), box(10, 10, 12, 12)]),
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)], holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]]),
        MultiPolygon([]),
    ]
    points = [Point(1, 1), Point(11, 11), Point(2, 2), Point(0, 0)]

    cpu_result = point_in_polygon(points, polygons, dispatch_mode=ExecutionMode.CPU)
    gpu_result = point_in_polygon(points, polygons, dispatch_mode=ExecutionMode.GPU)

    assert gpu_result == cpu_result


@pytest.mark.gpu
def test_point_in_polygon_auto_uses_gpu_variant_for_large_batches() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    points = from_shapely_geometries([Point(1, 1)] * 10_001)
    polygons = from_shapely_geometries([box(0, 0, 2, 2)] * 10_001)

    result = point_in_polygon(points, polygons, dispatch_mode=ExecutionMode.AUTO)

    assert result[0] is True
    report = points.diagnostics_report()
    assert not any("fallback" in reason.lower() for reason in report["runtime_history"])
