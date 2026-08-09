from __future__ import annotations

import numpy as np
import pytest

from vibespatial import (
    DEFAULT_BROADCAST_CROSSOVER_POLICIES,
    DEFAULT_CROSSOVER_POLICIES,
    CrossoverPolicy,
    DispatchDecision,
    ExecutionMode,
    KernelClass,
    PhysicalWorkEstimate,
    Residency,
    TransferTrigger,
    default_crossover_policy,
    estimate_grouped_work_from_owned,
    estimate_pairwise_product_work_from_owned,
    estimate_pairwise_work_from_owned,
    estimate_part_pair_work_from_owned,
    estimate_physical_work_from_owned,
    estimate_segment_pair_work_from_owned,
    estimate_spatial_index_work_from_owned,
    select_dispatch_for_estimate,
    select_dispatch_for_rows,
    select_residency_plan,
)
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import WorkloadShape


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


def test_plan_dispatch_selection_reports_scalar_broadcast_threshold_in_reason() -> None:
    plan = plan_dispatch_selection(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=2,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=True,
        workload_shape=WorkloadShape.SCALAR_RIGHT,
    )

    assert plan.selected is ExecutionMode.CPU
    assert plan.reason == "GPU runtime available; below 500-row crossover"


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


def test_physical_work_estimate_dispatches_on_candidate_pairs() -> None:
    policy = default_crossover_policy("point_in_polygon", KernelClass.PREDICATE)
    estimate = PhysicalWorkEstimate(
        row_count=10,
        candidate_pair_count=10_000,
        primary_unit_count=10_000,
        primary_unit_name="candidate-pair",
    )

    decision = select_dispatch_for_estimate(
        requested_mode=ExecutionMode.AUTO,
        work_estimate=estimate,
        policy=policy,
        gpu_available=True,
    )

    assert decision is DispatchDecision.GPU


def test_physical_work_estimate_dispatches_on_relation_pairs() -> None:
    policy = default_crossover_policy("predicate_refine", KernelClass.PREDICATE)
    estimate = PhysicalWorkEstimate.for_relation_pairs(
        row_count=8,
        relation_pair_count=10_000,
        primary_unit_name="refine-relation-pair",
    )

    decision = select_dispatch_for_estimate(
        requested_mode=ExecutionMode.AUTO,
        work_estimate=estimate,
        policy=policy,
        gpu_available=True,
    )

    assert decision is DispatchDecision.GPU
    assert estimate.dispatch_unit_name() == "refine-relation-pair"


@pytest.mark.parametrize(
    ("constructor", "pair_keyword"),
    [
        (PhysicalWorkEstimate.for_candidate_pairs, "candidate_pair_count"),
        (PhysicalWorkEstimate.for_relation_pairs, "relation_pair_count"),
    ],
)
def test_pair_estimates_include_output_and_temporary_byte_pressure(
    constructor,
    pair_keyword: str,
) -> None:
    estimate = constructor(
        row_count=2,
        **{pair_keyword: 3},
        output_byte_count=64_000,
        temporary_byte_count=256_000,
    )

    assert estimate.dispatch_unit_count() == 2_000


def test_owned_physical_work_estimate_uses_columnar_shape_metadata() -> None:
    class _Buffer:
        x = np.arange(12)
        ring_offsets = np.array([0, 5, 12])

    class _Owned:
        row_count = 2
        families = {"polygon": _Buffer()}

    estimate = estimate_physical_work_from_owned(
        _Owned(),
        candidate_pair_count=7,
    )

    assert estimate.row_count == 2
    assert estimate.coordinate_count == 12
    assert estimate.segment_count == 12
    assert estimate.ring_count == 2
    assert estimate.candidate_pair_count == 7


def test_owned_physical_work_estimate_scales_indexed_base_shape_to_logical_rows() -> None:
    class _Buffer:
        x = np.arange(40)
        ring_offsets = np.arange(9)

    class _Base:
        row_count = 8

    class _IndexedOwned:
        row_count = 20
        families = {"polygon": _Buffer()}
        is_indexed_view = True
        _base = _Base()

    estimate = estimate_physical_work_from_owned(_IndexedOwned())

    assert estimate.coordinate_count == 100
    assert estimate.segment_count == 100
    assert estimate.ring_count == 20


def test_spatial_index_work_estimate_dispatches_on_columnar_shape() -> None:
    class _Buffer:
        x = np.arange(12)
        ring_offsets = np.array([0, 5, 12])

    class _Owned:
        row_count = 2
        families = {"polygon": _Buffer()}

    estimate = estimate_spatial_index_work_from_owned(_Owned())

    assert estimate.row_count == 2
    assert estimate.coordinate_count == 12
    assert estimate.segment_count == 12
    assert estimate.dispatch_unit_count() == 12
    assert estimate.dispatch_unit_name() == "spatial-index-unit"


def test_owned_physical_work_estimate_prefers_authoritative_device_buffers() -> None:
    class _HostStub:
        x = np.empty(0)
        ring_offsets = None

    class _DeviceBuffer:
        x = np.arange(25)
        ring_offsets = np.arange(0, 26, 5)

    class _DeviceState:
        families = {"polygon": _DeviceBuffer()}

    class _Owned:
        row_count = 5
        families = {"polygon": _HostStub()}
        device_state = _DeviceState()
        is_indexed_view = False

    estimate = estimate_physical_work_from_owned(_Owned())

    assert estimate.coordinate_count == 25
    assert estimate.segment_count == 25
    assert estimate.ring_count == 5
    assert estimate.dispatch_unit_count() == 25


def test_segment_pair_work_estimate_preserves_quadratic_geometry_shape() -> None:
    class _Buffer:
        x = np.arange(22)
        geometry_offsets = np.array([0, 11, 22])

    class _Owned:
        row_count = 2
        families = {"line": _Buffer()}
        is_indexed_view = False

    estimate = estimate_segment_pair_work_from_owned(_Owned())

    assert estimate.coordinate_count == 22
    assert estimate.segment_pair_count == 90
    assert estimate.dispatch_unit_count() == 90
    assert estimate.dispatch_unit_name() == "segment-pair"
    assert "segment_pairs=90" in estimate.telemetry_detail()


def test_pairwise_product_work_estimate_uses_aligned_coordinate_products() -> None:
    class _LeftBuffer:
        x = np.arange(20)

    class _RightBuffer:
        x = np.arange(12)

    class _Left:
        row_count = 4
        families = {"line": _LeftBuffer()}

    class _Right:
        row_count = 4
        families = {"line": _RightBuffer()}

    estimate = estimate_pairwise_product_work_from_owned(
        _Left(),
        _Right(),
        pair_unit="coordinate",
    )

    assert estimate.coordinate_pair_count == 60
    assert estimate.dispatch_unit_count() == 60
    assert estimate.dispatch_unit_name() == "coordinate-pair"


def test_part_pair_work_estimate_tracks_component_endpoint_graph() -> None:
    class _Buffer:
        x = np.arange(16)
        geometry_offsets = np.array([0, 4, 8])
        part_offsets = np.arange(9) * 2

    class _Owned:
        row_count = 2
        families = {"multiline": _Buffer()}
        is_indexed_view = False

    estimate = estimate_part_pair_work_from_owned(_Owned())

    assert estimate.part_count == 8
    assert estimate.part_pair_count == 12
    assert estimate.dispatch_unit_count() == 16
    assert "part_pairs=12" in estimate.telemetry_detail()


def test_grouped_work_estimate_dispatches_on_segment_shape() -> None:
    class _Grouped:
        resolved_group_count = 4

    class _Buffer:
        x = np.arange(60_000)
        ring_offsets = np.arange(0, 60_001, 5)

    class _Owned:
        row_count = 8
        families = {"polygon": _Buffer()}

    estimate = estimate_grouped_work_from_owned(_Owned(), grouped=_Grouped())

    assert estimate.row_count == 8
    assert estimate.group_count == 4
    assert estimate.output_row_count == 4
    assert estimate.segment_count == 60_000
    assert estimate.dispatch_unit_name() == "grouped-segment"
    assert "groups=4" in estimate.telemetry_detail()

    decision = select_dispatch_for_estimate(
        requested_mode=ExecutionMode.AUTO,
        work_estimate=estimate,
        policy=default_crossover_policy("grouped_union", KernelClass.CONSTRUCTIVE),
        gpu_available=True,
    )

    assert decision is DispatchDecision.GPU


def test_pairwise_work_estimate_multiplies_broadcast_right_shape() -> None:
    class _LeftBuffer:
        x = np.arange(10)

    class _RightBuffer:
        x = np.arange(3)

    class _Left:
        row_count = 4
        families = {"linestring": _LeftBuffer()}

    class _Right:
        row_count = 1
        families = {"polygon": _RightBuffer()}

    estimate = estimate_pairwise_work_from_owned(
        _Left(),
        _Right(),
        workload=WorkloadShape.BROADCAST_RIGHT,
    )

    assert estimate.row_count == 4
    assert estimate.coordinate_count == 22
    assert estimate.segment_count == 22
    assert estimate.output_row_count == 4
    assert estimate.dispatch_unit_count() == 22


def test_plan_dispatch_selection_reports_physical_work_unit_reason() -> None:
    estimate = PhysicalWorkEstimate(
        row_count=10,
        candidate_pair_count=9_999,
        primary_unit_count=9_999,
        primary_unit_name="candidate-pair",
    )

    plan = plan_dispatch_selection(
        kernel_name="point_in_polygon",
        kernel_class=KernelClass.PREDICATE,
        row_count=10,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=True,
        work_estimate=estimate,
    )

    assert plan.selected is ExecutionMode.CPU
    assert plan.reason == "GPU runtime available; below 10000-candidate-pair crossover"
    assert any("candidate_pairs=9999" in detail for detail in plan.diagnostics)


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


def test_auto_device_residency_pins_dispatch_to_gpu_below_threshold() -> None:
    plan = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=1,
        requested_mode=ExecutionMode.AUTO,
        gpu_available=True,
        current_residency=Residency.DEVICE,
    )

    assert plan.selected is ExecutionMode.GPU
    assert plan.dispatch_decision is DispatchDecision.GPU
    assert plan.reason == "GPU runtime available; staying on device-resident buffers"


def test_explicit_cpu_still_overrides_device_residency() -> None:
    plan = plan_dispatch_selection(
        kernel_name="geometry_area",
        kernel_class=KernelClass.METRIC,
        row_count=1,
        requested_mode=ExecutionMode.CPU,
        gpu_available=True,
        current_residency=Residency.DEVICE,
    )

    assert plan.selected is ExecutionMode.CPU
