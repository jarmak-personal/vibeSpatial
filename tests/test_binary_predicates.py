from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    DeviceSnapshot,
    ExecutionMode,
    MonitoringBackend,
    NullBehavior,
    evaluate_binary_predicate,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.kernel_registry import get_kernel_variants


@pytest.mark.parametrize(
    ("predicate", "left", "right"),
    [
        ("intersects", [box(0, 0, 2, 2), box(10, 10, 11, 11), None], box(1, 1, 3, 3)),
        ("within", [Point(1, 1), Point(5, 5), None], box(0, 0, 2, 2)),
        ("contains", [box(0, 0, 5, 5), box(0, 0, 1, 1), None], [Point(1, 1), Point(2, 2), Point(0, 0)]),
        ("covers", [box(0, 0, 1, 1), box(0, 0, 1, 1), None], [Point(0, 0), Point(2, 2), Point(0, 0)]),
        ("covered_by", [Point(0, 0), Point(2, 2), None], [box(0, 0, 1, 1), box(0, 0, 3, 3), box(0, 0, 1, 1)]),
        ("touches", [box(0, 0, 1, 1), box(0, 0, 1, 1), None], [box(1, 1, 2, 2), box(2, 2, 3, 3), box(0, 0, 1, 1)]),
        ("crosses", [LineString([(0, 0), (2, 2)]), LineString([(0, 0), (1, 0)]), None], [LineString([(0, 2), (2, 0)]), LineString([(2, 2), (3, 3)]), LineString([(0, 0), (1, 1)])]),
        ("contains_properly", [box(0, 0, 2, 2), box(0, 0, 2, 2), None], [Point(1, 1), Point(0, 0), Point(1, 1)]),
        ("overlaps", [box(0, 0, 2, 2), box(0, 0, 1, 1), None], [box(1, 1, 3, 3), box(2, 2, 3, 3), box(0, 0, 1, 1)]),
        ("disjoint", [box(0, 0, 1, 1), box(0, 0, 1, 1), None], [box(2, 2, 3, 3), box(0.5, 0.5, 1.5, 1.5), box(0, 0, 1, 1)]),
    ],
)
def test_binary_predicate_matches_shapely(predicate, left, right) -> None:
    result = evaluate_binary_predicate(predicate, left, right, null_behavior=NullBehavior.PROPAGATE)
    expected = getattr(shapely, predicate)(np.asarray(left, dtype=object), right)
    expected_values = []
    for index, (left_value, exact) in enumerate(zip(left, list(expected), strict=True)):
        if left_value is None or (isinstance(right, list) and right[index] is None):
            expected_values.append(None)
        else:
            expected_values.append(bool(exact))
    assert result.values.tolist() == expected_values


def test_binary_predicate_uses_coarse_filter_before_exact_refine() -> None:
    left = [box(index * 10.0, 0.0, index * 10.0 + 1.0, 1.0) for index in range(32)]
    right = [box(index * 10.0, 0.0, index * 10.0 + 1.0, 1.0) for index in range(12)] + [
        box(index * 10.0 + 1_000.0, 0.0, index * 10.0 + 1_001.0, 1.0) for index in range(12, 32)
    ]

    result = evaluate_binary_predicate("intersects", left, right, null_behavior=NullBehavior.FALSE)

    assert result.candidate_rows.size < result.row_count
    assert result.candidate_rows.size == 12
    assert np.count_nonzero(result.values) == 12


@pytest.mark.cpu_fallback
def test_binary_predicate_auto_fallback_is_visible(monkeypatch) -> None:
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
    # Use `crosses` predicate — it is not in _DE9IM_PREDICATES, so
    # line-line pairs still trigger CPU fallback even after the DE-9IM
    # kernel landed for other predicates.
    left = from_shapely_geometries([LineString([(0, 0), (4, 4)])] * 10_001)
    right = from_shapely_geometries([LineString([(0, 4), (4, 0)])] * 10_001)

    result = evaluate_binary_predicate("crosses", left, right, null_behavior=NullBehavior.FALSE)

    assert bool(result.values[0]) is True
    report = left.diagnostics_report()
    assert any("point-centric candidate rows" in reason for reason in report["runtime_history"])


@pytest.mark.gpu
def test_binary_predicate_explicit_gpu_matches_cpu_for_supported_point_region_case() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 2, 2), box(0, 0, 2, 2), box(0, 0, 2, 2)])
    right = from_shapely_geometries([Point(1, 1), Point(0, 0), Point(3, 3)])

    cpu = evaluate_binary_predicate("contains", left, right, dispatch_mode=ExecutionMode.CPU)
    gpu = evaluate_binary_predicate("contains", left, right, dispatch_mode=ExecutionMode.GPU)

    assert gpu.values.tolist() == cpu.values.tolist()


def test_binary_predicate_explicit_gpu_request_fails_for_unsupported_candidate_pairs(monkeypatch) -> None:
    import vibespatial.adaptive_runtime as adaptive_runtime
    from vibespatial.kernels.predicates.binary_refine import crosses_exact

    def fake_snapshot(**_kwargs):
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="test snapshot",
        )

    monkeypatch.setattr(adaptive_runtime, "capture_device_snapshot", fake_snapshot)
    adaptive_runtime.invalidate_snapshot_cache()

    # Use `crosses` — not in _DE9IM_PREDICATES, so line-line GPU is still unsupported.
    with pytest.raises(NotImplementedError, match="point-centric candidate rows"):
        crosses_exact(
            [LineString([(0, 0), (4, 4)])],
            [LineString([(0, 4), (4, 0)])],
            dispatch_mode=ExecutionMode.GPU,
        )


def test_all_binary_predicates_register_gpu_variants() -> None:
    for predicate in (
        "intersects",
        "within",
        "contains",
        "covers",
        "covered_by",
        "touches",
        "crosses",
        "contains_properly",
        "overlaps",
        "disjoint",
    ):
        variants = get_kernel_variants(predicate)
        assert any(ExecutionMode.GPU in variant.execution_modes for variant in variants), predicate
