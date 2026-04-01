from __future__ import annotations

from vibespatial import (
    DEFAULT_CONSUMER_PROFILE,
    CoordinateStats,
    DeviceSnapshot,
    DispatchDecision,
    ExecutionMode,
    KernelClass,
    MonitoringBackend,
    MonitoringSample,
    PrecisionMode,
    Residency,
    WorkloadProfile,
    capture_device_snapshot,
    plan_adaptive_execution,
)
from vibespatial.runtime.adaptive import (
    _detect_device_profile,
    get_cached_snapshot,
    invalidate_snapshot_cache,
)
from vibespatial.runtime.crossover import default_crossover_policy
from vibespatial.runtime.kernel_registry import (
    KernelVariantSpec,
    get_kernel_variants,
    register_kernel_variant,
)


def test_capture_device_snapshot_without_probe_uses_static_heuristics() -> None:
    snapshot = capture_device_snapshot(
        probe=None,
        gpu_available=True,
        device_profile=DEFAULT_CONSUMER_PROFILE,
    )

    assert snapshot.backend is MonitoringBackend.UNAVAILABLE
    assert snapshot.gpu_available is True
    assert "static heuristics" in snapshot.reason


def test_capture_device_snapshot_uses_nvml_probe_when_available() -> None:
    snapshot = capture_device_snapshot(
        probe=lambda: MonitoringSample(
            sm_utilization_pct=71.0,
            memory_utilization_pct=54.0,
            device_name="fake-gpu",
        ),
        gpu_available=True,
        device_profile=DEFAULT_CONSUMER_PROFILE,
    )

    assert snapshot.backend is MonitoringBackend.NVML
    assert snapshot.sm_utilization_pct == 71.0
    assert snapshot.memory_utilization_pct == 54.0


def test_auto_planner_stays_on_cpu_below_crossover() -> None:
    plan = plan_adaptive_execution(
        kernel_name="point_in_polygon",
        kernel_class=KernelClass.PREDICATE,
        workload=WorkloadProfile(row_count=1_000, geometry_families=("point", "polygon")),
        requested_mode=ExecutionMode.AUTO,
        device_snapshot=DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="test snapshot",
        ),
    )

    assert plan.dispatch_decision is DispatchDecision.CPU
    assert plan.runtime_selection.selected is ExecutionMode.CPU
    assert plan.precision_plan.compute_precision is PrecisionMode.FP64


def test_planner_picks_specialized_point_variant_then_replans_for_polygons() -> None:
    point_variant = KernelVariantSpec(
        kernel_name="streaming_predicate",
        variant="gpu-point",
        qualified_name="tests.fake.point_variant",
        kernel_class=KernelClass.PREDICATE,
        execution_modes=(ExecutionMode.GPU,),
        geometry_families=("point",),
        supports_mixed=False,
        preferred_residency=Residency.DEVICE,
        precision_modes=(PrecisionMode.FP32, PrecisionMode.FP64),
        min_rows=None,
        max_rows=None,
        tags=("point-optimized",),
    )
    general_variant = KernelVariantSpec(
        kernel_name="streaming_predicate",
        variant="gpu-general",
        qualified_name="tests.fake.general_variant",
        kernel_class=KernelClass.PREDICATE,
        execution_modes=(ExecutionMode.GPU,),
        geometry_families=("point", "polygon"),
        supports_mixed=True,
        preferred_residency=Residency.DEVICE,
        precision_modes=(PrecisionMode.FP32, PrecisionMode.FP64),
        min_rows=None,
        max_rows=None,
        tags=("general",),
    )
    snapshot = DeviceSnapshot(
        backend=MonitoringBackend.NVML,
        gpu_available=True,
        device_profile=DEFAULT_CONSUMER_PROFILE,
        sm_utilization_pct=30.0,
        memory_utilization_pct=40.0,
        device_name="fake-gpu",
        reason="test snapshot",
    )

    first_plan = plan_adaptive_execution(
        kernel_name="streaming_predicate",
        kernel_class=KernelClass.PREDICATE,
        workload=WorkloadProfile(
            row_count=50_000,
            geometry_families=("point",),
            current_residency=Residency.DEVICE,
            is_streaming=True,
            chunk_index=0,
            coordinate_stats=CoordinateStats(max_abs_coord=10_000.0, span=100.0),
        ),
        device_snapshot=snapshot,
        variants=(point_variant, general_variant),
    )
    second_plan = plan_adaptive_execution(
        kernel_name="streaming_predicate",
        kernel_class=KernelClass.PREDICATE,
        workload=WorkloadProfile(
            row_count=50_000,
            geometry_families=("polygon",),
            current_residency=Residency.DEVICE,
            is_streaming=True,
            chunk_index=1,
            coordinate_stats=CoordinateStats(max_abs_coord=10_000.0, span=100.0),
        ),
        device_snapshot=snapshot,
        variants=(point_variant, general_variant),
    )

    assert first_plan.variant is point_variant
    assert first_plan.replan_after_chunk is True
    assert second_plan.variant is general_variant
    assert second_plan.runtime_selection.selected is ExecutionMode.GPU


def test_memory_pressure_reduces_chunk_hint() -> None:
    plan = plan_adaptive_execution(
        kernel_name="overlay",
        kernel_class=KernelClass.CONSTRUCTIVE,
        workload=WorkloadProfile(row_count=1_000_000, geometry_families=("polygon",)),
        requested_mode=ExecutionMode.AUTO,
        device_snapshot=DeviceSnapshot(
            backend=MonitoringBackend.NVML,
            gpu_available=True,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            sm_utilization_pct=55.0,
            memory_utilization_pct=92.0,
            device_name="fake-gpu",
            reason="test snapshot",
        ),
    )

    assert plan.runtime_selection.selected is ExecutionMode.GPU
    assert plan.chunk_rows == 25_000
    assert any("memory pressure" in detail for detail in plan.diagnostics)


def test_registry_decorator_records_structured_variant_metadata() -> None:
    @register_kernel_variant(
        "decorated_kernel",
        "gpu-default",
        kernel_class=KernelClass.COARSE,
        execution_modes=(ExecutionMode.GPU,),
        geometry_families=("point",),
        supports_mixed=False,
        preferred_residency=Residency.DEVICE,
        precision_modes=(PrecisionMode.FP32, PrecisionMode.FP64),
        min_rows=1_000,
        max_rows=1_000_000,
        tags=("default",),
    )
    def decorated_kernel() -> None:
        return None

    del decorated_kernel
    variants = get_kernel_variants("decorated_kernel")

    assert len(variants) == 1
    assert variants[0].variant == "gpu-default"
    assert variants[0].kernel_class is KernelClass.COARSE
    assert variants[0].preferred_residency is Residency.DEVICE


def test_crossover_override_returns_kernel_specific_threshold() -> None:
    policy = default_crossover_policy("point_clip", KernelClass.CONSTRUCTIVE)
    assert policy.auto_min_rows == 10_000
    assert "kernel-specific" in policy.reason

    policy = default_crossover_policy("point_buffer", KernelClass.CONSTRUCTIVE)
    assert policy.auto_min_rows == 10_000

    policy = default_crossover_policy("segment_classify", KernelClass.CONSTRUCTIVE)
    assert policy.auto_min_rows == 4_096

    policy = default_crossover_policy("flat_index_build", KernelClass.COARSE)
    assert policy.auto_min_rows == 0


def test_crossover_override_falls_through_to_class_default() -> None:
    policy = default_crossover_policy("unknown_kernel", KernelClass.PREDICATE)
    assert policy.auto_min_rows == 10_000
    assert "provisional" in policy.reason

    policy = default_crossover_policy("linestring_buffer", KernelClass.CONSTRUCTIVE)
    assert policy.auto_min_rows == 5_000


def test_cached_snapshot_returns_same_object() -> None:
    invalidate_snapshot_cache()
    first = get_cached_snapshot()
    second = get_cached_snapshot()
    assert first is second


def test_invalidate_snapshot_cache_forces_new_object() -> None:
    invalidate_snapshot_cache()
    first = get_cached_snapshot()
    invalidate_snapshot_cache()
    second = get_cached_snapshot()
    assert first is not second


def test_cached_snapshot_refreshes_when_runtime_availability_changes(monkeypatch) -> None:
    import vibespatial.runtime.adaptive as adaptive_runtime

    invalidate_snapshot_cache()
    monkeypatch.setattr(adaptive_runtime, "has_gpu_runtime", lambda: False)
    first = get_cached_snapshot()
    assert first.gpu_available is False

    monkeypatch.setattr(adaptive_runtime, "has_gpu_runtime", lambda: True)
    second = get_cached_snapshot()
    assert second.gpu_available is True
    assert second is not first


def test_detect_device_profile_returns_device_precision_profile() -> None:
    profile = _detect_device_profile()
    assert hasattr(profile, "fp64_to_fp32_ratio")
    assert profile.fp64_to_fp32_ratio > 0.0
    assert profile.fp64_to_fp32_ratio <= 1.0



# ---------------------------------------------------------------------------
# Explicit GPU mode bypasses crossover threshold (lyy.26)
# ---------------------------------------------------------------------------

_GPU_TEST_SNAPSHOT = DeviceSnapshot(
    backend=MonitoringBackend.UNAVAILABLE,
    gpu_available=True,
    device_profile=DEFAULT_CONSUMER_PROFILE,
    reason="test snapshot with GPU",
)


def test_explicit_gpu_bypasses_constructive_crossover_small_batch() -> None:
    """When dispatch_mode=GPU, small batches must stay on GPU.

    The CONSTRUCTIVE crossover threshold is 50K rows.  With AUTO, a
    1-row batch would be routed to CPU.  With explicit GPU, the
    crossover check is skipped entirely.
    """
    plan = plan_adaptive_execution(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        workload=WorkloadProfile(row_count=1),
        requested_mode=ExecutionMode.GPU,
        device_snapshot=_GPU_TEST_SNAPSHOT,
    )
    assert plan.dispatch_decision is DispatchDecision.GPU
    assert plan.runtime_selection.selected is ExecutionMode.GPU
    assert plan.runtime_selection.requested is ExecutionMode.GPU


def test_auto_mode_respects_constructive_crossover() -> None:
    """AUTO mode routes small CONSTRUCTIVE batches to CPU."""
    plan = plan_adaptive_execution(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        workload=WorkloadProfile(row_count=100),
        requested_mode=ExecutionMode.AUTO,
        device_snapshot=_GPU_TEST_SNAPSHOT,
    )
    assert plan.dispatch_decision is DispatchDecision.CPU
    assert plan.runtime_selection.selected is ExecutionMode.CPU


def test_auto_mode_routes_to_gpu_above_constructive_crossover() -> None:
    """AUTO mode routes large CONSTRUCTIVE batches to GPU."""
    plan = plan_adaptive_execution(
        kernel_name="binary_constructive",
        kernel_class=KernelClass.CONSTRUCTIVE,
        workload=WorkloadProfile(row_count=60_000),
        requested_mode=ExecutionMode.AUTO,
        device_snapshot=_GPU_TEST_SNAPSHOT,
    )
    assert plan.dispatch_decision is DispatchDecision.GPU
    assert plan.runtime_selection.selected is ExecutionMode.GPU


def test_explicit_gpu_bypasses_crossover_for_all_kernel_classes() -> None:
    """Explicit GPU mode skips crossover for every kernel class."""
    for kernel_class in KernelClass:
        plan = plan_adaptive_execution(
            kernel_name=f"test_{kernel_class.value}",
            kernel_class=kernel_class,
            workload=WorkloadProfile(row_count=1),
            requested_mode=ExecutionMode.GPU,
            device_snapshot=_GPU_TEST_SNAPSHOT,
        )
        assert plan.dispatch_decision is DispatchDecision.GPU, (
            f"explicit GPU should bypass crossover for {kernel_class.value}"
        )
        assert plan.runtime_selection.selected is ExecutionMode.GPU
