from __future__ import annotations

import ast
import inspect
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
    RuntimeSelection,
    evaluate_binary_predicate,
    from_shapely_geometries,
    has_gpu_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily
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
def test_indexed_point_region_adaptive_fp64_handles_fp32_collapsed_coordinates() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.predicates.point_relations import (
        classify_point_predicates_indexed_device,
    )

    base = 100_000_000.0
    polygon = Polygon(
        [
            (base, base),
            (base + 4.0, base),
            (base + 4.0, base + 4.0),
            (base, base + 4.0),
            (base, base),
        ]
    )
    point_values = np.asarray(
        [
            Point(base + 2.0, base + 2.0),
            Point(base, base + 2.0),
            Point(base - 1.0, base + 2.0),
            Point(base + 4.0, base + 4.0),
        ],
        dtype=object,
    )
    polygon_values = np.asarray([polygon] * len(point_values), dtype=object)
    points = from_shapely_geometries(point_values, residency=Residency.DEVICE)
    polygons = from_shapely_geometries(polygon_values, residency=Residency.DEVICE)

    result = classify_point_predicates_indexed_device(
        "covered_by",
        points,
        polygons,
        cp.arange(len(point_values), dtype=cp.int32),
        cp.arange(len(point_values), dtype=cp.int32),
    )

    expected = shapely.covered_by(point_values, polygon_values)
    np.testing.assert_array_equal(cp.asnumpy(result), expected)


def test_indexed_point_family_launchers_require_explicit_precision_plan() -> None:
    from vibespatial.predicates import point_relations

    launchers = (
        point_relations._classify_indexed_point_equals,
        point_relations._classify_indexed_point_line,
        point_relations._classify_indexed_point_region,
        point_relations._classify_indexed_mp_point,
        point_relations._classify_indexed_mp_line,
        point_relations._classify_indexed_mp_region,
        point_relations._classify_indexed_mp_mp,
    )

    for launcher in launchers:
        parameter = inspect.signature(launcher).parameters["precision_plan"]
        assert parameter.default is inspect.Parameter.empty
        assert "precision_plan=precision_plan" in inspect.getsource(launcher)


def test_indexed_point_precision_planner_selects_adaptive_fp64_on_consumer(
    monkeypatch,
) -> None:
    from vibespatial.predicates import point_relations
    from vibespatial.runtime.precision import PrecisionMode

    consumer_runtime = type("ConsumerRuntime", (), {"fp64_to_fp32_ratio": 1.0 / 64.0})()
    monkeypatch.setattr(point_relations, "get_cuda_runtime", lambda: consumer_runtime)

    auto_plan = point_relations._plan_indexed_point_precision(PrecisionMode.AUTO)
    forced_fp64_plan = point_relations._plan_indexed_point_precision(PrecisionMode.FP64)
    assert auto_plan.compute_precision is PrecisionMode.FP64
    assert "exact point-in-polygon refinement kernel is implemented in fp64" in auto_plan.reason
    assert forced_fp64_plan.compute_precision is PrecisionMode.FP64
    assert "explicit fp64 precision requested" in forced_fp64_plan.reason
    with pytest.raises(NotImplementedError, match="measured interval-fp32"):
        point_relations._plan_indexed_point_precision(PrecisionMode.FP32)


def test_indexed_point_precision_planner_keeps_measured_fp64_on_datacenter(
    monkeypatch,
) -> None:
    from vibespatial.predicates import point_relations
    from vibespatial.runtime.precision import PrecisionMode

    datacenter_runtime = type("DatacenterRuntime", (), {"fp64_to_fp32_ratio": 0.5})()
    monkeypatch.setattr(point_relations, "get_cuda_runtime", lambda: datacenter_runtime)

    plan = point_relations._plan_indexed_point_precision(PrecisionMode.AUTO)

    assert plan.compute_precision is PrecisionMode.FP64
    assert "exact point-in-polygon refinement kernel is implemented in fp64" in plan.reason


@pytest.mark.gpu
def test_homogeneous_indexed_point_predicates_propagate_fp64_plan(monkeypatch) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    cp = pytest.importorskip("cupy")
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.predicates import point_relations
    from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan

    precision_plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="test indexed point predicate precision propagation",
        ),
        kernel_class=KernelClass.PREDICATE,
        requested=PrecisionMode.FP64,
    )
    observed = []

    def classify_equals(*args, precision_plan, **kwargs):
        observed.append(precision_plan)
        return cp.asarray([2], dtype=cp.uint8)

    monkeypatch.setattr(
        point_relations,
        "_classify_indexed_point_equals",
        classify_equals,
    )
    result = point_relations.classify_homogeneous_point_predicates_indexed_device(
        "intersects",
        object(),
        object(),
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([0], dtype=cp.int32),
        left_family=GeometryFamily.POINT,
        right_family=GeometryFamily.POINT,
        precision_plan=precision_plan,
    )

    assert observed == [precision_plan]
    assert cp.asnumpy(result).tolist() == [True]


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


@pytest.mark.gpu
def test_single_mask_convex_certificate_rejects_self_intersecting_star() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.predicates.binary import _evaluate_covered_by_single_polygonal_mask_gpu

    mask_geom = Polygon(
        [(0, 3), (1, -1), (-3, 1), (3, 1), (-1, -1), (0, 3)],
    )
    # Every local turn has one sign, but the mask is self-intersecting.  A
    # turn-only convex certificate would incorrectly admit the vertex theorem.
    left_geoms = [box(-0.25, 0.0, 0.25, 0.5), box(2.0, 0.9, 2.5, 1.1)]
    left = from_shapely_geometries(left_geoms, residency=Residency.DEVICE)
    mask = from_shapely_geometries([mask_geom], residency=Residency.DEVICE)

    result = _evaluate_covered_by_single_polygonal_mask_gpu(left, mask)

    assert result is not None
    expected = [bool(geom.covered_by(mask_geom)) for geom in left_geoms]
    assert result.tolist() == expected


@pytest.mark.gpu
def test_public_within_selects_exact_convex_grouped_path_at_measured_scale() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.cuda._runtime import (
        get_d2h_transfer_events,
        reset_d2h_transfer_count,
    )

    angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    target = Polygon(np.column_stack((np.cos(angles), np.sin(angles))))
    edge_start, edge_end = np.asarray(target.exterior.coords[:2], dtype=np.float64)
    edge = edge_end - edge_start
    outward = np.asarray((edge[1], -edge[0]), dtype=np.float64)
    outward /= np.linalg.norm(outward)
    edge_midpoint = (edge_start + edge_end) * 0.5
    outside = edge_midpoint + outward * 5e-8
    crossing = Polygon(
        [
            tuple(edge_midpoint - outward * 1e-4),
            tuple(edge_midpoint + np.asarray((-edge[1], edge[0])) * 1e-4),
            tuple(outside),
        ]
    )
    geometries = [box(-0.01, -0.01, 0.01, 0.01)] * 9_999 + [crossing]
    source = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    mask = from_shapely_geometries([target], residency=Residency.DEVICE)

    geopandas.clear_dispatch_events()
    reset_d2h_transfer_count()
    first = evaluate_binary_predicate(
        "within",
        source,
        mask,
        dispatch_mode=ExecutionMode.GPU,
    )
    second = evaluate_binary_predicate(
        "within",
        source,
        mask,
        dispatch_mode=ExecutionMode.GPU,
    )
    events = geopandas.get_dispatch_events(clear=True)
    certification_events = [
        event
        for event in get_d2h_transfer_events()
        if event.reason == "polygon predicate convex-mask certification planning packet"
    ]
    source_certification_events = [
        event
        for event in get_d2h_transfer_events()
        if event.reason == "polygon predicate simple-source certification planning packet"
    ]

    expected = shapely.within(np.asarray(geometries, dtype=object), target)
    assert np.array_equal(first.values, expected)
    assert np.array_equal(second.values, expected)
    assert bool(first.values[-1]) is False
    assert len(certification_events) == 1
    assert len(source_certification_events) == 1
    mask_state = mask._ensure_device_state(preserve_indexed_view=True)
    source_state = source._ensure_device_state(preserve_indexed_view=True)
    mask_certificate = mask_state.polygon_certificates[("convex-mask", GeometryFamily.POLYGON, 0)]
    source_certificate = source_state.polygon_certificates[
        ("simple-source-no-holes", GeometryFamily.POLYGON, -1)
    ]
    assert mask_certificate.source_token == f"owned-device-state:{id(mask_state)}"
    assert source_certificate.source_token == f"owned-device-state:{id(source_state)}"
    assert source_certificate.residency is Residency.DEVICE
    assert source_certificate.values.__cuda_array_interface__
    assert any(
        event.implementation == "gpu_convex_grouped_vertex_containment"
        for event in events
    )


@pytest.mark.gpu
def test_convex_grouped_selector_declines_invalid_zero_area_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    monkeypatch.setenv("VIBESPATIAL_STRICT_NATIVE", "1")
    target = box(0.0, 0.0, 10.0, 10.0)
    degenerate = Polygon([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 1.0)])
    geometries = [degenerate] * 10_000
    source = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    mask = from_shapely_geometries([target], residency=Residency.DEVICE)

    geopandas.clear_dispatch_events()
    for predicate in ("covered_by", "within"):
        result = evaluate_binary_predicate(
            predicate,
            source,
            mask,
            dispatch_mode=ExecutionMode.GPU,
        )
        expected = getattr(shapely, predicate)(
            np.asarray(geometries, dtype=object),
            target,
        )
        assert np.array_equal(result.values, expected), predicate
        assert not np.any(result.values), predicate
    assert all(
        event.implementation != "gpu_convex_grouped_vertex_containment"
        for event in geopandas.get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
def test_convex_grouped_selector_declines_single_skewed_quadratic_ring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    import vibespatial.predicates.polygon as polygon_predicates

    angles = np.linspace(0.0, 2.0 * np.pi, 20_000, endpoint=False)
    skewed = Polygon(np.column_stack((0.5 * np.cos(angles), 0.5 * np.sin(angles))))
    geometries = [box(-0.01, -0.01, 0.01, 0.01)] * 9_999 + [skewed]
    source = from_shapely_geometries(geometries, residency=Residency.DEVICE)
    mask = from_shapely_geometries([box(-1.0, -1.0, 1.0, 1.0)], residency=Residency.DEVICE)

    def fail_if_certification_launches(*_args, **_kwargs):
        raise AssertionError("quadratic source certification must not launch for skewed rings")

    monkeypatch.setattr(
        polygon_predicates,
        "certify_polygonal_sources_simple_no_holes_gpu",
        fail_if_certification_launches,
    )
    result = polygon_predicates.compute_polygonal_covered_by_single_convex_grouped_gpu(
        source,
        mask,
        query_family=GeometryFamily.POLYGON,
        mask_family=GeometryFamily.POLYGON,
    )

    source_buffer = source._ensure_device_state(
        preserve_indexed_view=True
    ).families[GeometryFamily.POLYGON]
    assert source_buffer.fixed_size is not None
    assert source_buffer.fixed_size.max_coord_count_per_row == 20_001
    assert int(source_buffer.x.size) / 10_000 < 8.0
    assert result is None


@pytest.mark.gpu
def test_convex_grouped_selector_declines_indexed_nonzero_mask_row() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")

    sources = [box(10.1, 10.1, 10.2, 10.2)] * 10_000
    source = from_shapely_geometries(sources, residency=Residency.DEVICE)
    mask_base = from_shapely_geometries(
        [box(-1.0, -1.0, 1.0, 1.0), box(10.0, 10.0, 11.0, 11.0)],
        residency=Residency.DEVICE,
    )
    mask = type(mask_base)._indexed_view(
        mask_base,
        cp.asarray([1], dtype=cp.int64),
        assume_unique_indices=True,
    )

    geopandas.clear_dispatch_events()
    result = evaluate_binary_predicate(
        "within",
        source,
        mask,
        dispatch_mode=ExecutionMode.GPU,
    )

    assert mask.is_indexed_view
    assert result.values.tolist() == [True] * 10_000
    assert all(
        event.implementation != "gpu_convex_grouped_vertex_containment"
        for event in geopandas.get_dispatch_events(clear=True)
    )


@pytest.mark.gpu
def test_convex_grouped_selector_declines_sparse_indexed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")
    cp = pytest.importorskip("cupy")
    import vibespatial.predicates.polygon as polygon_predicates

    source_base = from_shapely_geometries(
        [box(-0.01, -0.01, 0.01, 0.01)] * 20_000,
        residency=Residency.DEVICE,
    )
    source = type(source_base)._indexed_view(
        source_base,
        cp.arange(10_000, dtype=cp.int64),
        assume_unique_indices=True,
    )
    mask = from_shapely_geometries(
        [box(-1.0, -1.0, 1.0, 1.0)],
        residency=Residency.DEVICE,
    )

    def fail_if_certification_launches(*_args, **_kwargs):
        raise AssertionError("indexed source certification must not launch")

    monkeypatch.setattr(
        polygon_predicates,
        "certify_polygonal_sources_simple_no_holes_gpu",
        fail_if_certification_launches,
    )
    result = polygon_predicates.compute_polygonal_covered_by_single_convex_grouped_gpu(
        source,
        mask,
        query_family=GeometryFamily.POLYGON,
        mask_family=GeometryFamily.POLYGON,
    )

    assert source.is_indexed_view
    assert source.row_count == 10_000
    assert source_base.row_count == 20_000
    assert result is None


@pytest.mark.gpu
def test_gpu_polygon_predicates_preserve_interior_collapsed_ring_semantics() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    target = box(0.0, 0.0, 10.0, 10.0)
    collapsed = Polygon([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (1.0, 1.0)])
    source = from_shapely_geometries([collapsed], residency=Residency.DEVICE)
    mask = from_shapely_geometries([target], residency=Residency.DEVICE)

    for predicate in ("covered_by", "within"):
        result = evaluate_binary_predicate(
            predicate,
            source,
            mask,
            dispatch_mode=ExecutionMode.GPU,
        )
        expected = bool(getattr(shapely, predicate)(collapsed, target))
        assert result.values.tolist() == [expected]
        assert expected is True


@pytest.mark.gpu
def test_convex_grouped_lowering_preserves_inverse_public_predicates() -> None:
    if not has_gpu_runtime():
        pytest.skip("CUDA runtime not available")

    from vibespatial.predicates.binary import _broadcast_right_owned

    target_geometry = box(-1.0, -1.0, 1.0, 1.0)
    source_geometries = [box(-0.1, -0.1, 0.1, 0.1)] * 9_999 + [
        box(0.9, -0.1, 1.1, 0.1)
    ]
    source = from_shapely_geometries(source_geometries, residency=Residency.DEVICE)
    mask = from_shapely_geometries([target_geometry], residency=Residency.DEVICE)
    repeated_mask = _broadcast_right_owned(mask, len(source_geometries))

    for predicate, left, right, expected in (
        (
            "covered_by",
            source,
            mask,
            shapely.covered_by(np.asarray(source_geometries, dtype=object), target_geometry),
        ),
        (
            "within",
            source,
            mask,
            shapely.within(np.asarray(source_geometries, dtype=object), target_geometry),
        ),
        (
            "covers",
            repeated_mask,
            source,
            shapely.covers(target_geometry, np.asarray(source_geometries, dtype=object)),
        ),
        (
            "contains",
            repeated_mask,
            source,
            shapely.contains(target_geometry, np.asarray(source_geometries, dtype=object)),
        ),
    ):
        result = evaluate_binary_predicate(
            predicate,
            left,
            right,
            dispatch_mode=ExecutionMode.GPU,
        )
        assert np.array_equal(result.values, expected), predicate


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
