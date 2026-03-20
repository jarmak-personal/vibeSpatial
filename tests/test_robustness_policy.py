from __future__ import annotations

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    ExecutionMode,
    GeometryPresence,
    KernelClass,
    PrecisionMode,
    PredicateFallback,
    RobustnessGuarantee,
    RuntimeSelection,
    TopologyPolicy,
    select_precision_plan,
)
from vibespatial.runtime.robustness import select_robustness_plan


def gpu_selection() -> RuntimeSelection:
    return RuntimeSelection(requested=ExecutionMode.GPU, selected=ExecutionMode.GPU, reason="gpu")


def test_predicate_kernels_require_exact_guarantee() -> None:
    precision_plan = select_precision_plan(
        runtime_selection=gpu_selection(),
        kernel_class=KernelClass.PREDICATE,
        requested=PrecisionMode.FP32,
        device_profile=DEFAULT_CONSUMER_PROFILE,
    )
    plan = select_robustness_plan(
        kernel_class=KernelClass.PREDICATE,
        precision_plan=precision_plan,
        null_state=GeometryPresence.VALUE,
        empty_state=GeometryPresence.VALUE,
    )

    assert plan.guarantee is RobustnessGuarantee.EXACT
    assert plan.predicate_fallback is PredicateFallback.EXPANSION_ARITHMETIC
    assert plan.topology_policy is TopologyPolicy.PRESERVE


def test_constructive_kernels_require_topology_preservation() -> None:
    precision_plan = select_precision_plan(
        runtime_selection=gpu_selection(),
        kernel_class=KernelClass.CONSTRUCTIVE,
        requested=PrecisionMode.FP64,
    )
    plan = select_robustness_plan(kernel_class=KernelClass.CONSTRUCTIVE, precision_plan=precision_plan)

    assert plan.guarantee is RobustnessGuarantee.EXACT
    assert plan.predicate_fallback is PredicateFallback.RATIONAL_RECONSTRUCTION
    assert plan.topology_policy is TopologyPolicy.PRESERVE


def test_metric_kernels_allow_bounded_error() -> None:
    precision_plan = select_precision_plan(
        runtime_selection=gpu_selection(),
        kernel_class=KernelClass.METRIC,
        requested=PrecisionMode.FP32,
    )
    plan = select_robustness_plan(kernel_class=KernelClass.METRIC, precision_plan=precision_plan)

    assert plan.guarantee is RobustnessGuarantee.BOUNDED_ERROR
    assert plan.predicate_fallback is PredicateFallback.NONE


def test_all_robustness_plans_keep_null_and_empty_semantics() -> None:
    precision_plan = select_precision_plan(
        runtime_selection=gpu_selection(),
        kernel_class=KernelClass.COARSE,
        requested=PrecisionMode.FP64,
    )
    plan = select_robustness_plan(
        kernel_class=KernelClass.COARSE,
        precision_plan=precision_plan,
        null_state=GeometryPresence.NULL,
        empty_state=GeometryPresence.EMPTY,
    )

    assert plan.handles_nulls is True
    assert plan.handles_empties is True
