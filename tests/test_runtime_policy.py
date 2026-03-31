from __future__ import annotations

import pytest

from vibespatial import (
    DEFAULT_BROADCAST_CROSSOVER_POLICIES,
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
from vibespatial.runtime.workload import WorkloadShape


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
        broadcast_min_rows=1_000,
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


# ---------------------------------------------------------------------------
# Broadcast crossover threshold tests
# ---------------------------------------------------------------------------


def test_crossover_policy_accepts_broadcast_min_rows() -> None:
    """CrossoverPolicy frozen dataclass accepts the broadcast_min_rows field."""
    policy = CrossoverPolicy(
        kernel_name="test_kernel",
        kernel_class=KernelClass.METRIC,
        auto_min_rows=5_000,
        reason="test",
        broadcast_min_rows=500,
    )
    assert policy.broadcast_min_rows == 500


def test_crossover_policy_broadcast_min_rows_defaults_to_none() -> None:
    """broadcast_min_rows is None when not explicitly set."""
    policy = CrossoverPolicy(
        kernel_name="test_kernel",
        kernel_class=KernelClass.METRIC,
        auto_min_rows=5_000,
        reason="test",
    )
    assert policy.broadcast_min_rows is None


def test_default_broadcast_thresholds_are_defined_by_kernel_class() -> None:
    assert DEFAULT_BROADCAST_CROSSOVER_POLICIES == {
        KernelClass.COARSE: 256,
        KernelClass.METRIC: 500,
        KernelClass.PREDICATE: 1_000,
        KernelClass.CONSTRUCTIVE: 500,
    }


def test_default_crossover_policy_populates_broadcast_min_rows() -> None:
    """default_crossover_policy() sets broadcast_min_rows from the class defaults."""
    policy = default_crossover_policy("geometry_distance", KernelClass.METRIC)
    assert policy.broadcast_min_rows == 500

    policy = default_crossover_policy("bbox_coarse", KernelClass.COARSE)
    assert policy.broadcast_min_rows == 256

    policy = default_crossover_policy("pip_check", KernelClass.PREDICATE)
    assert policy.broadcast_min_rows == 1_000

    policy = default_crossover_policy("polygon_union", KernelClass.CONSTRUCTIVE)
    assert policy.broadcast_min_rows == 500


def test_kernel_override_still_gets_broadcast_threshold() -> None:
    """Kernel-specific overrides should still get broadcast_min_rows from class defaults."""
    # geometry_area has a kernel-specific override of 500, class is METRIC
    policy = default_crossover_policy("geometry_area", KernelClass.METRIC)
    assert policy.auto_min_rows == 500  # kernel-specific override
    assert policy.broadcast_min_rows == 500  # METRIC class broadcast default


def test_broadcast_right_uses_lower_threshold() -> None:
    """BROADCAST_RIGHT workload uses broadcast_min_rows, dispatching to GPU at lower N."""
    policy = default_crossover_policy("point_in_polygon", KernelClass.PREDICATE)
    # Pairwise threshold is 10,000; broadcast is 1,000.
    # At 2,000 rows: pairwise -> CPU, broadcast -> GPU.
    decision_pairwise = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.PAIRWISE,
    )
    decision_broadcast = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )

    assert decision_pairwise is DispatchDecision.CPU
    assert decision_broadcast is DispatchDecision.GPU


def test_scalar_right_uses_broadcast_threshold() -> None:
    """SCALAR_RIGHT is treated the same as BROADCAST_RIGHT for crossover."""
    policy = default_crossover_policy("point_in_polygon", KernelClass.PREDICATE)
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.SCALAR_RIGHT,
    )
    assert decision is DispatchDecision.GPU


def test_pairwise_uses_original_threshold() -> None:
    """PAIRWISE workload shape uses auto_min_rows, not the broadcast threshold."""
    policy = default_crossover_policy("area", KernelClass.METRIC)
    # METRIC pairwise threshold is 5,000.  At 2,000 rows: CPU.
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.PAIRWISE,
    )
    assert decision is DispatchDecision.CPU


def test_no_workload_shape_uses_original_threshold() -> None:
    """Omitting workload_shape preserves backward-compatible pairwise behavior."""
    policy = default_crossover_policy("area", KernelClass.METRIC)
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=2_000,
        policy=policy,
        gpu_available=True,
    )
    assert decision is DispatchDecision.CPU

    # Above pairwise threshold: GPU.
    decision_above = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=5_000,
        policy=policy,
        gpu_available=True,
    )
    assert decision_above is DispatchDecision.GPU


def test_broadcast_fallback_when_broadcast_min_rows_is_none() -> None:
    """When broadcast_min_rows is None, fall back to auto_min_rows // 10."""
    policy = CrossoverPolicy(
        kernel_name="custom",
        kernel_class=KernelClass.PREDICATE,
        auto_min_rows=10_000,
        reason="test",
        broadcast_min_rows=None,
    )
    # Fallback threshold is 10_000 // 10 = 1_000.
    # At 500 rows: CPU.  At 1_000 rows: GPU.
    decision_below = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=500,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )
    decision_at = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=1_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )

    assert decision_below is DispatchDecision.CPU
    assert decision_at is DispatchDecision.GPU


def test_broadcast_below_broadcast_threshold_dispatches_cpu() -> None:
    """Broadcast workload below the broadcast threshold stays on CPU."""
    policy = default_crossover_policy("point_in_polygon", KernelClass.PREDICATE)
    # broadcast_min_rows for PREDICATE is 1,000.  At 500 rows: CPU.
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.AUTO,
        row_count=500,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )
    assert decision is DispatchDecision.CPU


def test_explicit_modes_ignore_workload_shape() -> None:
    """Explicit CPU/GPU mode bypasses threshold regardless of workload shape."""
    policy = default_crossover_policy("area", KernelClass.METRIC)

    # Explicit CPU with broadcast shape still returns CPU.
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.CPU,
        row_count=100_000,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )
    assert decision is DispatchDecision.CPU

    # Explicit GPU with broadcast shape still returns GPU.
    decision = select_dispatch_for_rows(
        requested_mode=ExecutionMode.GPU,
        row_count=1,
        policy=policy,
        gpu_available=True,
        workload_shape=WorkloadShape.BROADCAST_RIGHT,
    )
    assert decision is DispatchDecision.GPU
