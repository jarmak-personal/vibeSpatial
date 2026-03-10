from __future__ import annotations

from vibespatial import (
    CompensationMode,
    CoordinateStats,
    DEFAULT_CONSUMER_PROFILE,
    DEFAULT_DATACENTER_PROFILE,
    ExecutionMode,
    KernelClass,
    PrecisionMode,
    RefinementMode,
    RuntimeSelection,
    select_precision_plan,
)


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
