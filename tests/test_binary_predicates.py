from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, Polygon, box

import vibespatial as geopandas
from tests.upstream.geopandas.tests.util import (
    _NATURALEARTH_CITIES,
    _NATURALEARTH_LOWRES,
)
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
from vibespatial.predicates.binary import _gpu_candidate_pairs_supported
from vibespatial.runtime.kernel_registry import get_kernel_variants
from vibespatial.runtime.residency import Residency


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


@pytest.mark.gpu
@pytest.mark.cpu_fallback
def test_binary_predicate_auto_fallback_is_visible(monkeypatch) -> None:
    import vibespatial.runtime.adaptive as adaptive_runtime

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
    assert any(
        "GPU refine currently supports only point-centric and DE-9IM" in reason
        for reason in report["runtime_history"]
    )


@pytest.mark.gpu
def test_binary_predicate_explicit_gpu_matches_cpu_for_supported_point_region_case() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    left = from_shapely_geometries([box(0, 0, 2, 2), box(0, 0, 2, 2), box(0, 0, 2, 2)])
    right = from_shapely_geometries([Point(1, 1), Point(0, 0), Point(3, 3)])

    cpu = evaluate_binary_predicate("contains", left, right, dispatch_mode=ExecutionMode.CPU)
    gpu = evaluate_binary_predicate("contains", left, right, dispatch_mode=ExecutionMode.GPU)

    assert gpu.values.tolist() == cpu.values.tolist()


@pytest.mark.gpu
def test_binary_predicate_gpu_intersects_matches_host_for_scalar_polygon_regression() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cities = geopandas.read_file(_NATURALEARTH_CITIES)
    world = geopandas.read_file(_NATURALEARTH_LOWRES)
    south_america = world.loc[world["continent"] == "South America", "geometry"].union_all()
    owned = from_shapely_geometries(
        np.asarray(cities.geometry, dtype=object),
        residency=Residency.DEVICE,
    )

    gpu = evaluate_binary_predicate(
        "intersects",
        owned,
        south_america,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    expected = np.asarray(cities.geometry.intersects(south_america), dtype=bool)

    np.testing.assert_array_equal(np.asarray(gpu.values, dtype=bool), expected)
    assert bool(gpu.values[62]) is True


@pytest.mark.gpu
def test_single_mask_covered_by_gpu_probe_matches_shapely_for_no_hole_mask() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.predicates.binary import _evaluate_covered_by_single_polygonal_mask_gpu

    left_geoms = [
        box(1, 1, 2, 2),
        box(8, 8, 12, 12),
        Polygon([(0, 5), (5, 5), (5, 8), (0, 8), (0, 5)]),
        None,
    ]
    mask_geom = box(0, 0, 10, 10)
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    mask = from_shapely_geometries([mask_geom], residency=Residency.DEVICE)

    result = _evaluate_covered_by_single_polygonal_mask_gpu(left, mask)

    assert result is not None
    expected = [
        False if geom is None else bool(geom.covered_by(mask_geom))
        for geom in left_geoms
    ]
    assert result.tolist() == expected


@pytest.mark.gpu
def test_single_mask_covered_by_gpu_probe_matches_shapely_for_hole_mask() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.predicates.binary import _evaluate_covered_by_single_polygonal_mask_gpu

    mask_geom = Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
        holes=[[(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]],
    )
    left_geoms = [box(1, 1, 2, 2), box(4.25, 4.25, 5.75, 5.75)]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    mask = from_shapely_geometries([mask_geom], residency=Residency.DEVICE)

    result = _evaluate_covered_by_single_polygonal_mask_gpu(left, mask)

    assert result is not None
    expected = [bool(geom.covered_by(mask_geom)) for geom in left_geoms]
    assert result.tolist() == expected


@pytest.mark.gpu
def test_single_mask_covered_by_gpu_probe_matches_shapely_for_concave_mask() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.predicates.binary import _evaluate_covered_by_single_polygonal_mask_gpu

    mask_geom = Polygon(
        [
            (0, 0),
            (5, 0),
            (5, 5),
            (3, 5),
            (3, 2),
            (2, 2),
            (2, 5),
            (0, 5),
            (0, 0),
        ],
    )
    left_geoms = [box(0.5, 0.5, 1.5, 1.5), box(2.25, 3.0, 2.75, 4.0)]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    mask = from_shapely_geometries([mask_geom], residency=Residency.DEVICE)

    result = _evaluate_covered_by_single_polygonal_mask_gpu(left, mask)

    assert result is not None
    expected = [bool(geom.covered_by(mask_geom)) for geom in left_geoms]
    assert result.tolist() == expected


def test_binary_predicate_explicit_gpu_request_fails_for_unsupported_candidate_pairs(monkeypatch) -> None:
    import vibespatial.runtime.adaptive as adaptive_runtime
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
    with pytest.raises(
        NotImplementedError,
        match="GPU refine currently supports only point-centric",
    ):
        crosses_exact(
            [LineString([(0, 0), (4, 4)])],
            [LineString([(0, 4), (4, 0)])],
            dispatch_mode=ExecutionMode.GPU,
        )


def test_gpu_candidate_pairs_supported_accepts_mixed_point_and_de9im_rows() -> None:
    left = from_shapely_geometries(
        [
            Point(1, 1),
            box(0, 0, 2, 2),
            LineString([(0, 0), (2, 2)]),
        ]
    )
    right = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            Point(1, 1),
            box(0, 0, 2, 2),
        ]
    )
    candidate_rows = np.arange(3, dtype=np.int32)

    assert _gpu_candidate_pairs_supported(left, right, candidate_rows, "intersects") is True


def test_gpu_candidate_pairs_supported_rejects_mixed_batches_with_unsupported_nonpoint_rows() -> None:
    left = from_shapely_geometries(
        [
            Point(1, 1),
            LineString([(0, 0), (2, 2)]),
        ]
    )
    right = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 2, 2),
        ]
    )
    candidate_rows = np.arange(2, dtype=np.int32)

    assert _gpu_candidate_pairs_supported(left, right, candidate_rows, "crosses") is False


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
