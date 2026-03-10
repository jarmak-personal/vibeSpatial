from __future__ import annotations

import pytest

from vibespatial import (
    DEFAULT_CROSSOVER_POLICIES,
    CrossoverPolicy,
    DispatchDecision,
    ExecutionMode,
    KernelClass,
    Residency,
    TransferTrigger,
    default_crossover_policy,
    select_dispatch_for_rows,
    select_residency_plan,
)


def test_user_materialization_is_explicit_and_visible() -> None:
    plan = select_residency_plan(
        current=Residency.DEVICE,
        target=Residency.HOST,
        trigger=TransferTrigger.USER_MATERIALIZATION,
    )

    assert plan.transfer_required is True
    assert plan.visible_to_user is True
    assert plan.zero_copy_eligible is False


def test_interop_view_prefers_zero_copy_when_layouts_align() -> None:
    plan = select_residency_plan(
        current=Residency.DEVICE,
        target=Residency.HOST,
        trigger=TransferTrigger.INTEROP_VIEW,
    )

    assert plan.transfer_required is False
    assert plan.visible_to_user is False
    assert plan.zero_copy_eligible is True


def test_non_user_transfer_stays_visible() -> None:
    plan = select_residency_plan(
        current=Residency.DEVICE,
        target=Residency.HOST,
        trigger=TransferTrigger.UNSUPPORTED_GPU_PATH,
    )

    assert plan.transfer_required is True
    assert plan.visible_to_user is True
    assert "silent host execution" in plan.reason


def test_default_thresholds_are_defined_by_kernel_class() -> None:
    assert DEFAULT_CROSSOVER_POLICIES == {
        KernelClass.COARSE: 1_000,
        KernelClass.METRIC: 5_000,
        KernelClass.PREDICATE: 10_000,
        KernelClass.CONSTRUCTIVE: 50_000,
    }


def test_default_crossover_policy_carries_kernel_context() -> None:
    policy = default_crossover_policy("point_in_polygon", KernelClass.PREDICATE)

    assert policy == CrossoverPolicy(
        kernel_name="point_in_polygon",
        kernel_class=KernelClass.PREDICATE,
        auto_min_rows=10_000,
        reason="provisional auto crossover for predicate kernels is 10000 rows",
    )


def test_auto_dispatches_cpu_below_threshold() -> None:
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=default_crossover_policy("area", KernelClass.METRIC),
        gpu_available=True,
    )

    assert decision is DispatchDecision.CPU


def test_auto_dispatches_gpu_at_threshold() -> None:
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=10_000,
        policy=default_crossover_policy("point_in_polygon", KernelClass.PREDICATE),
        gpu_available=True,
    )

    assert decision is DispatchDecision.GPU


def test_explicit_gpu_bypasses_threshold() -> None:
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.GPU,
        row_count=10,
        policy=default_crossover_policy("bounds", KernelClass.COARSE),
        gpu_available=True,
    )

    assert decision is DispatchDecision.GPU


def test_explicit_gpu_requires_runtime() -> None:
    with pytest.raises(RuntimeError, match="GPU execution was requested"):
        select_dispatch_for_rows(
            requested_mode=ExecutionMode.GPU,
            row_count=100_000,
            policy=default_crossover_policy("bounds", KernelClass.COARSE),
            gpu_available=False,
        )
