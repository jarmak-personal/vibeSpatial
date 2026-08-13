from __future__ import annotations

import ast
from pathlib import Path

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    DEFAULT_DATACENTER_PROFILE,
    CompensationMode,
    CoordinateStats,
    ExecutionMode,
    KernelClass,
    PrecisionMode,
    RefinementMode,
    RuntimeSelection,
    select_precision_plan,
)


def test_nearest_precision_overrides_only_flow_through_precision_plan() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "spatial" / "nearest.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    direct_compute_overrides: list[int] = []
    unplanned_fp64_requests: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )
        for keyword in node.keywords:
            value = keyword.value
            is_precision_constant = (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "PrecisionMode"
                and value.attr in {"FP32", "FP64"}
            )
            if keyword.arg == "compute_precision" and is_precision_constant:
                direct_compute_overrides.append(node.lineno)
            if is_precision_constant and value.attr == "FP64":
                if not (function_name == "select_precision_plan" and keyword.arg == "requested"):
                    unplanned_fp64_requests.append(node.lineno)

    # Constructor/function defaults are another way to smuggle an override
    # around PrecisionPlan, so reject those too.
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "PrecisionMode"
            and node.attr == "FP64"
        ):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.keyword):
            grandparent = parents.get(parent)
            if (
                isinstance(grandparent, ast.Call)
                and isinstance(grandparent.func, ast.Name)
                and grandparent.func.id == "select_precision_plan"
                and parent.arg == "requested"
            ):
                continue
        if node.lineno not in unplanned_fp64_requests:
            unplanned_fp64_requests.append(node.lineno)

    assert direct_compute_overrides == []
    assert unplanned_fp64_requests == []


def test_nearest_ambiguity_refinement_has_no_exact_device_compaction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "src" / "vibespatial" / "spatial" / "nearest.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    refiners = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name
        in {
            "_refine_ambiguous_point_family_distances",
            "_dwithin_refine_gpu",
        }
    }

    assert set(refiners) == {
        "_refine_ambiguous_point_family_distances",
        "_dwithin_refine_gpu",
    }
    for name, function in refiners.items():
        flatnonzero_calls = [
            node.value.lineno
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and "ambiguous" in target.id
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
            and any(
                isinstance(descendant, ast.Call)
                and isinstance(descendant.func, ast.Attribute)
                and descendant.func.attr == "flatnonzero"
                for descendant in ast.walk(node.value)
            )
        ]
        ambiguous_size_branches = [
            node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and any(
                isinstance(descendant, ast.Attribute)
                and descendant.attr == "size"
                and isinstance(descendant.value, ast.Name)
                and "ambiguous" in descendant.value.id
                for descendant in ast.walk(node.test)
            )
        ]
        source = ast.get_source_segment(path.read_text(), function) or ""

        assert flatnonzero_calls == [], f"{name} exact-compacts ambiguity at {flatnonzero_calls}"
        assert ambiguous_size_branches == [], (
            f"{name} branches on exact ambiguity size at {ambiguous_size_branches}"
        )
        assert "NativeDeviceSelection.from_mask" in source
        assert ".gather_capacity" in source
        if name == "_refine_ambiguous_point_family_distances":
            assert ".logical_count" in source
        else:
            assert "pair_active=d_ambiguity_active" in source
            assert "source_positions=d_ambiguity_partition" in source

    mixed = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_mixed_distances_gpu_device"
    )
    mixed_source = ast.get_source_segment(path.read_text(), mixed) or ""
    forbidden = (
        "flatnonzero",
        "copy_device_to_host",
        "to_shapely",
        "shapely.distance",
        "NativeDeviceSelection.from_mask",
        "sub_count",
        "sub_mask",
    )
    assert not any(token in mixed_source for token in forbidden)
    assert "NativeRelationFamilyPartition.from_pair_capacity" in mixed_source
    assert "source_positions=group.source_positions" in mixed_source
    assert "launch_capacity = max(1, (pair_count + group_count - 1) // group_count)" in mixed_source


def test_cpu_runtime_forces_native_fp64() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            reason="cpu",
        ),
        kernel_class=KernelClass.COARSE,
    )

    assert plan.storage_precision is PrecisionMode.FP64
    assert plan.compute_precision is PrecisionMode.FP64
    assert plan.compensation is CompensationMode.NONE


def test_explicit_fp32_on_gpu_uses_centered_compensated_plan() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="gpu",
        ),
        requested=PrecisionMode.FP32,
        kernel_class=KernelClass.PREDICATE,
        coordinate_stats=CoordinateStats(max_abs_coord=2_000_000.0, span=100.0),
    )

    assert plan.storage_precision is PrecisionMode.FP64
    assert plan.compute_precision is PrecisionMode.FP32
    assert plan.compensation is CompensationMode.CENTERED
    assert plan.refinement is RefinementMode.SELECTIVE_FP64
    assert plan.center_coordinates is True


def test_auto_prefers_native_fp64_on_datacenter_profile() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            reason="gpu",
        ),
        kernel_class=KernelClass.METRIC,
        device_profile=DEFAULT_DATACENTER_PROFILE,
    )

    assert plan.compute_precision is PrecisionMode.FP64
    assert plan.refinement is RefinementMode.NONE


def test_auto_prefers_staged_fp32_for_consumer_predicates() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            reason="gpu",
        ),
        kernel_class=KernelClass.PREDICATE,
        device_profile=DEFAULT_CONSUMER_PROFILE,
        coordinate_stats=CoordinateStats(max_abs_coord=10_000.0, span=100.0),
    )

    assert plan.compute_precision is PrecisionMode.FP32
    assert plan.refinement is RefinementMode.SELECTIVE_FP64


def test_auto_keeps_constructive_kernels_on_fp64_even_for_consumer_profile() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            reason="gpu",
        ),
        kernel_class=KernelClass.CONSTRUCTIVE,
        device_profile=DEFAULT_CONSUMER_PROFILE,
    )

    assert plan.compute_precision is PrecisionMode.FP64
    assert plan.compensation is CompensationMode.NONE


def test_metric_kernel_uses_kahan_when_fp32_is_forced() -> None:
    plan = select_precision_plan(
        runtime_selection=RuntimeSelection(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            reason="gpu",
        ),
        requested=PrecisionMode.FP32,
        kernel_class=KernelClass.METRIC,
    )

    assert plan.compute_precision is PrecisionMode.FP32
    assert plan.compensation is CompensationMode.KAHAN
    assert plan.refinement is RefinementMode.NONE
