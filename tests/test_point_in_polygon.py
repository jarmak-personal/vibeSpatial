from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box

import vibespatial.api as geopandas
from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    DeviceSnapshot,
    ExecutionMode,
    MonitoringBackend,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.kernels.predicates.point_in_polygon import (
    point_in_polygon,
    point_in_polygon_expression,
)
from vibespatial.runtime.residency import Residency
from vibespatial.testing import compare_with_shapely, point_in_polygon_reference


def test_point_predicate_d2h_exports_are_runtime_accounted() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = (
        repo_root / "src" / "vibespatial" / "kernels" / "core" / "geometry_analysis.py",
        repo_root / "src" / "vibespatial" / "kernels" / "predicates" / "point_in_polygon.py",
    )
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            if func.attr == "asnumpy":
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            if func.attr == "copy_device_to_host" and not any(
                keyword.arg == "reason" for keyword in node.keywords
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert offenders == []


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
    import vibespatial.runtime.adaptive as adaptive_runtime

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


@pytest.mark.gpu
def test_gpu_strategies_produce_equivalent_results() -> None:
    """All GPU strategies (auto, dense, compacted, fused) must return equivalent arrays."""
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.kernels.predicates.point_in_polygon import (
        _evaluate_point_in_polygon_gpu,
    )
    from vibespatial.kernels.predicates.point_within_bounds import (
        _normalize_right_input,
    )

    polygons_list = [
        box(0, 0, 2, 2),
        MultiPolygon([box(5, 5, 8, 8), box(10, 10, 12, 12)]),
        Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
        None,
        box(1, 1, 3, 3),
    ]
    points_list = [Point(1, 1), Point(11, 11), Point(2, 2), Point(), Point(2, 2)]

    points = from_shapely_geometries(points_list)
    polygons = from_shapely_geometries(polygons_list)

    right = _normalize_right_input(polygons, expected_len=points.row_count)

    points.move_to(
        "device",
        trigger="explicit-runtime-request",
        reason="cross-strategy regression test",
    )
    polygons.move_to(
        "device",
        trigger="explicit-runtime-request",
        reason="cross-strategy regression test",
    )

    results = {}
    for strategy in ("auto", "dense", "compacted", "fused"):
        result = _evaluate_point_in_polygon_gpu(points, right, strategy=strategy)
        results[strategy] = np.asarray(result, dtype=object)

    # All strategies must produce identical object-dtype arrays.
    for strategy in ("dense", "compacted", "fused"):
        assert np.array_equal(results["auto"], results[strategy]), (
            f"GPU strategy '{strategy}' diverged from 'auto':\n"
            f"  auto: {results['auto']}\n"
            f"  {strategy}: {results[strategy]}"
        )


@pytest.mark.gpu
def test_binned_strategy_return_device_stays_device_resident(strict_device_guard) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.kernels.predicates.point_in_polygon import (
        _evaluate_point_in_polygon_gpu,
    )
    from vibespatial.kernels.predicates.point_within_bounds import (
        NormalizedBoundsInput,
    )

    def dense_ngon(center_x: float, center_y: float, radius: float, n: int) -> Polygon:
        angles = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False)
        coords = [
            (
                center_x + radius * float(np.cos(angle)),
                center_y + radius * float(np.sin(angle)),
            )
            for angle in angles
        ]
        coords.append(coords[0])
        return Polygon(coords)

    polygons = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            dense_ngon(10.0, 10.0, 2.0, 2048),
            MultiPolygon([box(20, 20, 22, 22), box(30, 30, 32, 32)]),
            MultiPolygon([dense_ngon(40.0, 40.0, 2.0, 1536)]),
        ],
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        [Point(1, 1), Point(10, 10), Point(21, 21), Point(100, 100)],
        residency=Residency.DEVICE,
    )
    right = NormalizedBoundsInput(
        bounds=np.zeros((points.row_count, 4), dtype=np.float64),
        null_mask=np.zeros(points.row_count, dtype=bool),
        empty_mask=np.zeros(points.row_count, dtype=bool),
        geometry_array=polygons,
    )
    runtime = get_cuda_runtime()
    original_sync = runtime.synchronize

    def _fail_sync() -> None:
        raise AssertionError("binned return_device path should not helper-synchronize")

    stream = cp.cuda.Stream(non_blocking=True)
    runtime.synchronize = _fail_sync
    try:
        with stream:
            result = _evaluate_point_in_polygon_gpu(
                points,
                right,
                strategy="binned",
                return_device=True,
            )
            assert hasattr(result, "__cuda_array_interface__")
            assert result.shape == (4,)
            device_sum = cp.asarray(result, dtype=cp.int32).sum()
            device_last = cp.asarray(result)[3]
        stream.synchronize()
        assert int(device_sum.item()) == 3
        assert bool(device_last.item()) is False
    finally:
        runtime.synchronize = original_sync


@pytest.mark.gpu
@pytest.mark.parametrize("strategy", ["dense", "compacted", "fused"])
def test_return_device_strategies_stay_device_resident(
    strategy: str,
    strict_device_guard,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.kernels.predicates.point_in_polygon import (
        _evaluate_point_in_polygon_gpu,
    )
    from vibespatial.kernels.predicates.point_within_bounds import (
        NormalizedBoundsInput,
    )

    polygons = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            MultiPolygon([box(5, 5, 8, 8), box(10, 10, 12, 12)]),
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            MultiPolygon([]),
        ],
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        [Point(1, 1), Point(11, 11), Point(2, 2), Point(0, 0)],
        residency=Residency.DEVICE,
    )
    right = NormalizedBoundsInput(
        bounds=np.zeros((points.row_count, 4), dtype=np.float64),
        null_mask=np.zeros(points.row_count, dtype=bool),
        empty_mask=np.zeros(points.row_count, dtype=bool),
        geometry_array=polygons,
    )
    runtime = get_cuda_runtime()
    original_sync = runtime.synchronize

    def _fail_sync() -> None:
        raise AssertionError(f"{strategy} return_device path should not helper-synchronize")

    stream = cp.cuda.Stream(non_blocking=True)
    runtime.synchronize = _fail_sync
    try:
        with stream:
            result = _evaluate_point_in_polygon_gpu(
                points,
                right,
                strategy=strategy,
                return_device=True,
            )
            assert hasattr(result, "__cuda_array_interface__")
            assert result.shape == (4,)
            device_sum = cp.asarray(result, dtype=cp.int32).sum()
            device_last = cp.asarray(result)[3]
        stream.synchronize()
        assert int(device_sum.item()) == 3
        assert bool(device_last.item()) is False
    finally:
        runtime.synchronize = original_sync


@pytest.mark.gpu
def test_public_return_device_path_does_not_materialize_lazy_device_metadata() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    polygons = from_shapely_geometries(
        [box(0, 0, 2, 2), box(10, 10, 12, 12), None],
        residency=Residency.DEVICE,
    )
    points = from_shapely_geometries(
        [Point(1, 1), Point(11, 11), Point(5, 5)],
        residency=Residency.DEVICE,
    )

    # Simulate device-native IO/import surfaces: routing metadata exists on
    # device, but host metadata has not crossed the compatibility boundary.
    for owned in (points, polygons):
        assert owned.device_state is not None
        owned._validity = None
        owned._tags = None
        owned._family_row_offsets = None

    with assert_zero_d2h_transfers():
        result = point_in_polygon(
            points,
            polygons,
            dispatch_mode=ExecutionMode.GPU,
            _return_device=True,
        )
        assert hasattr(result, "__cuda_array_interface__")

    assert cp.asnumpy(result).tolist() == [True, True, False]


@pytest.mark.gpu
def test_point_in_polygon_expression_feeds_native_rowset_without_public_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import cupy as cp

    from vibespatial.cuda._runtime import assert_zero_d2h_transfers

    points = from_shapely_geometries(
        [Point(0.5, 0.5), Point(2.0, 2.0), Point(0.0, 0.0)],
        residency=Residency.DEVICE,
    )
    polygons = from_shapely_geometries(
        [box(0, 0, 1, 1), box(0, 0, 1, 1), box(0, 0, 1, 1)],
        residency=Residency.DEVICE,
    )

    geopandas.clear_dispatch_events()
    with assert_zero_d2h_transfers():
        expression = point_in_polygon_expression(
            points,
            polygons,
            dispatch_mode=ExecutionMode.GPU,
            source_token="points",
        )
        rowset = expression.equal_to(True)
    events = geopandas.get_dispatch_events(clear=True)

    assert expression.is_device
    assert expression.source_token == "points"
    assert expression.source_row_count == 3
    assert any(
        event.implementation == "native_point_in_polygon_expression_gpu"
        and "carrier=NativeExpression" in event.detail
        for event in events
    )
    assert cp.asnumpy(expression.values).tolist() == [True, False, True]
    assert cp.asnumpy(rowset.positions).tolist() == [0, 2]
