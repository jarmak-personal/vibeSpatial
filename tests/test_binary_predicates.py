from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box

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


def test_binary_predicate_d2h_exports_are_runtime_accounted() -> None:
    def _contains_raw_cupy_scalar_sync(node: ast.AST) -> bool:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cp"
            and node.func.attr in {"any", "all", "sum", "count_nonzero", "max", "min"}
        ):
            return True
        return any(_contains_raw_cupy_scalar_sync(child) for child in ast.iter_child_nodes(node))

    repo_root = Path(__file__).resolve().parents[1]
    paths = (
        repo_root / "src" / "vibespatial" / "predicates" / "binary.py",
        repo_root / "src" / "vibespatial" / "predicates" / "point_relations.py",
        repo_root / "src" / "vibespatial" / "predicates" / "polygon.py",
    )
    offenders: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "asnumpy":
                    offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
                if func.attr == "item" and path.name == "point_relations.py":
                    offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
                if func.attr == "copy_device_to_host" and not any(
                    keyword.arg == "reason" for keyword in node.keywords
                ):
                    offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
            elif (
                path.name in {"binary.py", "point_relations.py"}
                and isinstance(func, ast.Name)
                and func.id in {"bool", "int", "float"}
                and node.args
                and _contains_raw_cupy_scalar_sync(node.args[0])
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert offenders == []


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
def test_binary_predicate_gpu_point_candidates_export_once_at_public_boundary() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [Point(0, 0), Point(1, 1), Point(2, 2)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [Point(0, 0), Point(5, 5), Point(2, 2)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    result = evaluate_binary_predicate(
        "intersects",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.values.tolist() == [True, False, True]
    assert "binary predicate point-point intersects result host export" in reasons
    assert not any("point relation point_equals_compacted result host export" in reason for reason in reasons)


@pytest.mark.gpu
def test_binary_predicate_gpu_point_region_fast_path_avoids_candidate_row_export() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [box(0, 0, 2, 2), box(0, 0, 2, 2), box(0, 0, 2, 2)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [Point(1, 1), Point(0, 1), Point(5, 5)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    result = evaluate_binary_predicate(
        "contains",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.values.tolist() == [True, False, False]
    assert "binary predicate point-region contains result host export" in reasons
    assert not any("candidate-row contains host export" in reason for reason in reasons)
    assert not any("point relation point_in_polygon" in reason for reason in reasons)


@pytest.mark.gpu
def test_binary_predicate_gpu_de9im_candidates_export_once_at_public_boundary() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    left = from_shapely_geometries(
        [Point(0, 0), box(0, 0, 2, 2), box(5, 5, 6, 6)],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [Point(0, 0), box(1, 1, 3, 3), box(7, 7, 8, 8)],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    result = evaluate_binary_predicate(
        "intersects",
        left,
        right,
        dispatch_mode=ExecutionMode.GPU,
        null_behavior=NullBehavior.FALSE,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result.values.tolist() == [True, True, False]
    assert "binary predicate de9im-candidate intersects result host export" in reasons
    assert not any("de9im-mask host export" in reason for reason in reasons)


@pytest.mark.gpu
def test_fused_multi_predicate_public_export_does_not_export_candidate_rows() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )
    from vibespatial.predicates.binary import _evaluate_binary_predicates_fused_gpu

    left = from_shapely_geometries(
        [
            box(0, 0, 2, 2),
            box(0, 0, 1, 1),
            box(5, 5, 6, 6),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            box(1, 1, 3, 3),
            box(-1, -1, 2, 2),
            box(7, 7, 8, 8),
            box(0, 0, 1, 1),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    result = _evaluate_binary_predicates_fused_gpu(
        ("intersects", "covered_by", "disjoint"),
        left,
        right,
    )
    reasons = [event.reason for event in get_d2h_transfer_events(clear=True)]

    assert result is not None
    assert result["intersects"].tolist() == [True, True, False, True]
    assert result["covered_by"].tolist() == [False, True, False, True]
    assert result["disjoint"].tolist() == [False, False, True, False]
    assert "binary predicate fused predicate-results host export" in reasons
    assert not any("binary predicate fused candidate-row host export" in reason for reason in reasons)


@pytest.mark.gpu
def test_indexed_point_relation_device_dispatch_avoids_branch_scalar_syncs() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers, reset_d2h_transfer_count
    from vibespatial.predicates.point_relations import classify_point_predicates_indexed_device

    left = from_shapely_geometries(
        [
            Point(0, 0),
            Point(1, 1),
            box(0, 0, 2, 2),
            LineString([(0, 0), (0, 2)]),
        ],
        residency=Residency.DEVICE,
    )
    right = from_shapely_geometries(
        [
            Point(0, 0),
            LineString([(0, 0), (2, 2)]),
            Point(0.5, 0.5),
            Point(0, 1),
        ],
        residency=Residency.DEVICE,
    )

    reset_d2h_transfer_count()
    with assert_zero_d2h_transfers():
        result = classify_point_predicates_indexed_device(
            "intersects",
            left,
            right,
            cp.arange(4, dtype=cp.int32),
            cp.arange(4, dtype=cp.int32),
        )

    assert cp.asnumpy(result).tolist() == [True, True, True, True]


@pytest.mark.gpu
def test_indexed_point_relation_device_dispatch_handles_multipoint_rows_without_d2h() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.cuda._runtime import assert_zero_d2h_transfers, reset_d2h_transfer_count
    from vibespatial.predicates.point_relations import classify_point_predicates_indexed_device

    left_geoms = np.asarray(
        [
            MultiPoint([(0, 0), (2, 2)]),
            Point(1, 1),
            MultiPoint([(0.5, 0.5), (3, 3)]),
            box(0, 0, 2, 2),
            MultiPoint([(0, 0), (1, 1)]),
            LineString([(0, 0), (2, 0)]),
            MultiPoint([(0, 0), (1, 0)]),
        ],
        dtype=object,
    )
    right_geoms = np.asarray(
        [
            Point(0, 0),
            MultiPoint([(0, 0), (1, 1)]),
            box(0, 0, 1, 1),
            MultiPoint([(1, 1), (3, 3)]),
            MultiPoint([(0, 0)]),
            MultiPoint([(1, 0), (5, 5)]),
            LineString([(0.5, 0), (0.5, 1)]),
        ],
        dtype=object,
    )
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    right = from_shapely_geometries(right_geoms, residency=Residency.DEVICE)
    left_indices = cp.arange(left_geoms.size, dtype=cp.int32)
    right_indices = cp.arange(right_geoms.size, dtype=cp.int32)

    for predicate in ("intersects", "contains", "within", "covers", "covered_by", "disjoint"):
        reset_d2h_transfer_count()
        with assert_zero_d2h_transfers():
            result = classify_point_predicates_indexed_device(
                predicate,
                left,
                right,
                left_indices,
                right_indices,
            )
        expected = getattr(shapely, predicate)(left_geoms, right_geoms)
        np.testing.assert_array_equal(cp.asnumpy(result), np.asarray(expected, dtype=bool))


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
