from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ._runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from .crossover import (
    CrossoverPolicy,
    DispatchDecision,
    default_crossover_policy,
    select_dispatch_for_rows,
)
from .kernel_registry import KernelVariantSpec, get_kernel_variants
from .precision import (
    DEFAULT_CONSUMER_PROFILE,
    CoordinateStats,
    DevicePrecisionProfile,
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from .residency import Residency
from .workload import WorkloadShape


class MonitoringBackend(StrEnum):
    UNAVAILABLE = "unavailable"
    NVML = "nvml"


@dataclass(frozen=True)
class MonitoringSample:
    sm_utilization_pct: float
    memory_utilization_pct: float
    device_name: str = "unknown"


class MonitoringProbe(Protocol):
    def __call__(self) -> MonitoringSample: ...


@dataclass(frozen=True)
class DeviceSnapshot:
    backend: MonitoringBackend
    gpu_available: bool
    device_profile: DevicePrecisionProfile
    sm_utilization_pct: float | None = None
    memory_utilization_pct: float | None = None
    device_name: str = "unknown"
    reason: str = ""

    @property
    def underutilized(self) -> bool:
        return self.sm_utilization_pct is not None and self.sm_utilization_pct < 40.0

    @property
    def under_memory_pressure(self) -> bool:
        return self.memory_utilization_pct is not None and self.memory_utilization_pct >= 85.0


@dataclass(frozen=True)
class WorkloadProfile:
    row_count: int
    geometry_families: tuple[str, ...] = ()
    mixed_geometry: bool = False
    current_residency: Residency = Residency.HOST
    coordinate_stats: CoordinateStats | None = None
    is_streaming: bool = False
    chunk_index: int = 0
    avg_vertices_per_geometry: float = 0.0
    workload_shape: WorkloadShape | None = None


@dataclass(frozen=True)
class AdaptivePlan:
    runtime_selection: RuntimeSelection
    dispatch_decision: DispatchDecision
    crossover_policy: CrossoverPolicy
    precision_plan: PrecisionPlan
    variant: KernelVariantSpec | None
    chunk_rows: int
    replan_after_chunk: bool
    diagnostics: tuple[str, ...]


def capture_device_snapshot(
    *,
    probe: MonitoringProbe | None = None,
    gpu_available: bool | None = None,
    device_profile: DevicePrecisionProfile | None = None,
) -> DeviceSnapshot:
    runtime_available = has_gpu_runtime() if gpu_available is None else gpu_available
    profile = device_profile or DEFAULT_CONSUMER_PROFILE

    if not runtime_available:
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=False,
            device_profile=profile,
            reason="GPU runtime is unavailable; adaptive planner will remain on host",
        )

    if probe is None:
        return DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=True,
            device_profile=profile,
            reason="NVML monitoring unavailable; planner will use static heuristics only",
        )

    sample = probe()
    return DeviceSnapshot(
        backend=MonitoringBackend.NVML,
        gpu_available=True,
        device_profile=profile,
        sm_utilization_pct=sample.sm_utilization_pct,
        memory_utilization_pct=sample.memory_utilization_pct,
        device_name=sample.device_name,
        reason="NVML monitoring sample captured",
    )


def _select_runtime_with_availability(
    requested_mode: ExecutionMode | str,
    *,
    gpu_available: bool,
) -> RuntimeSelection:
    mode = requested_mode if isinstance(requested_mode, ExecutionMode) else ExecutionMode(requested_mode)

    if mode is ExecutionMode.CPU:
        return RuntimeSelection(requested=mode, selected=ExecutionMode.CPU, reason="CPU requested")

    if gpu_available:
        return RuntimeSelection(
            requested=mode,
            selected=ExecutionMode.GPU,
            reason="GPU runtime available",
        )

    if mode is ExecutionMode.GPU:
        raise RuntimeError("GPU execution was requested, but no GPU runtime is available")

    return RuntimeSelection(
        requested=mode,
        selected=ExecutionMode.CPU,
        reason="GPU runtime unavailable; using explicit CPU fallback",
    )


def _score_variant(
    variant: KernelVariantSpec,
    *,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    workload: WorkloadProfile,
) -> int | None:
    if runtime_selection.selected not in variant.execution_modes:
        return None
    if precision_plan.compute_precision not in variant.precision_modes:
        return None
    if workload.mixed_geometry and not variant.supports_mixed:
        return None
    if variant.min_rows is not None and workload.row_count < variant.min_rows:
        return None
    if variant.max_rows is not None and workload.row_count > variant.max_rows:
        return None

    score = 0
    if variant.kernel_class is precision_plan.kernel_class:
        score += 4
    if not variant.geometry_families:
        score += 1
    else:
        shared = len(set(workload.geometry_families).intersection(variant.geometry_families))
        score += shared * 3
        if set(workload.geometry_families).issubset(set(variant.geometry_families)):
            score += 3
    if variant.preferred_residency is workload.current_residency:
        score += 1
    if workload.mixed_geometry and variant.supports_mixed:
        score += 2
    # Geometry-adaptive variant selection: boost cooperative variants
    # for complex geometries (many vertices per row), and boost simple
    # variants for geometries with few vertices.
    is_cooperative = "cooperative" in variant.tags
    avg_verts = workload.avg_vertices_per_geometry
    if avg_verts > 0:
        if is_cooperative and avg_verts >= 64:
            # Strong boost for cooperative variants on complex geometries
            score += 5
        elif not is_cooperative and avg_verts < 64:
            # Mild boost for simple variants on simple geometries
            score += 2
    return score


def select_kernel_variant(
    *,
    kernel_name: str,
    runtime_selection: RuntimeSelection,
    precision_plan: PrecisionPlan,
    workload: WorkloadProfile,
    variants: tuple[KernelVariantSpec, ...] | None = None,
) -> KernelVariantSpec | None:
    candidates = variants if variants is not None else get_kernel_variants(kernel_name)
    best_variant: KernelVariantSpec | None = None
    best_score: int | None = None
    for variant in candidates:
        score = _score_variant(
            variant,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            workload=workload,
        )
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_variant = variant
            best_score = score
    return best_variant


def _select_chunk_rows(workload: WorkloadProfile, snapshot: DeviceSnapshot) -> int:
    if workload.row_count <= 0:
        return 0
    if not snapshot.gpu_available:
        return workload.row_count
    if snapshot.under_memory_pressure:
        return min(workload.row_count, 25_000)
    if snapshot.underutilized and workload.row_count > 100_000:
        return min(workload.row_count, 250_000)
    return min(workload.row_count, 100_000)


def plan_adaptive_execution(
    *,
    kernel_name: str,
    kernel_class: KernelClass | str,
    workload: WorkloadProfile,
    requested_mode: ExecutionMode | str = ExecutionMode.AUTO,
    requested_precision: PrecisionMode | str = PrecisionMode.AUTO,
    device_snapshot: DeviceSnapshot | None = None,
    variants: tuple[KernelVariantSpec, ...] | None = None,
) -> AdaptivePlan:
    normalized_class = kernel_class if isinstance(kernel_class, KernelClass) else KernelClass(kernel_class)
    snapshot = device_snapshot or capture_device_snapshot()
    initial_runtime = _select_runtime_with_availability(
        requested_mode,
        gpu_available=snapshot.gpu_available,
    )
    crossover_policy = default_crossover_policy(kernel_name, normalized_class)
    dispatch_decision = select_dispatch_for_rows(
        requested_mode=requested_mode,
        row_count=workload.row_count,
        policy=crossover_policy,
        gpu_available=snapshot.gpu_available,
        workload_shape=workload.workload_shape,
    )

    if initial_runtime.requested is ExecutionMode.AUTO and dispatch_decision is DispatchDecision.CPU:
        runtime_selection = RuntimeSelection(
            requested=initial_runtime.requested,
            selected=ExecutionMode.CPU,
            reason=f"{initial_runtime.reason}; below {crossover_policy.auto_min_rows}-row crossover",
        )
    elif dispatch_decision is DispatchDecision.GPU:
        runtime_selection = RuntimeSelection(
            requested=initial_runtime.requested,
            selected=ExecutionMode.GPU,
            reason=initial_runtime.reason,
        )
    else:
        runtime_selection = initial_runtime

    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=normalized_class,
        requested=requested_precision,
        coordinate_stats=workload.coordinate_stats,
        device_profile=snapshot.device_profile,
    )
    variant = select_kernel_variant(
        kernel_name=kernel_name,
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        workload=workload,
        variants=variants,
    )
    chunk_rows = _select_chunk_rows(workload, snapshot)
    diagnostics = [
        snapshot.reason,
        runtime_selection.reason,
        crossover_policy.reason,
        f"precision: {precision_plan.reason}",
    ]
    if variant is None:
        diagnostics.append("variant: no compatible specialized variant registered")
    else:
        diagnostics.append(f"variant: {variant.variant} ({variant.qualified_name})")
    if snapshot.under_memory_pressure:
        diagnostics.append("chunking: memory pressure detected; reducing planned chunk size")
    elif snapshot.underutilized and chunk_rows > 100_000:
        diagnostics.append("chunking: low utilization detected; increasing chunk size hint")

    return AdaptivePlan(
        runtime_selection=runtime_selection,
        dispatch_decision=dispatch_decision,
        crossover_policy=crossover_policy,
        precision_plan=precision_plan,
        variant=variant,
        chunk_rows=chunk_rows,
        replan_after_chunk=(
            runtime_selection.selected is ExecutionMode.GPU
            and workload.is_streaming
            and workload.chunk_index == 0
        ),
        diagnostics=tuple(diagnostics),
    )


# ---------------------------------------------------------------------------
# Phase 1: Session-scoped device snapshot caching (ADR-0007)
# ---------------------------------------------------------------------------

def _detect_device_profile() -> DevicePrecisionProfile:
    """Build a DevicePrecisionProfile from the actual hardware fp64:fp32 ratio."""
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime
        runtime = get_cuda_runtime()
        ratio = runtime.fp64_to_fp32_ratio
        name = f"detected-{ratio:.4f}"
        return DevicePrecisionProfile(name=name, fp64_to_fp32_ratio=ratio)
    except Exception:
        return DEFAULT_CONSUMER_PROFILE


def _adapt_nvml_sampler(sampler: object) -> MonitoringProbe | None:
    """Wrap a profiling._NvmlGpuSampler as a MonitoringProbe."""
    if not getattr(sampler, "available", False):
        return None

    def probe() -> MonitoringSample:
        sample = sampler.sample()  # type: ignore[union-attr]
        if sample is None:
            return MonitoringSample(0.0, 0.0, "unknown")
        return MonitoringSample(
            sm_utilization_pct=sample.sm_utilization_pct,
            memory_utilization_pct=sample.memory_utilization_pct,
            device_name=sample.device_name,
        )

    return probe


_cached_snapshot: DeviceSnapshot | None = None


def get_cached_snapshot() -> DeviceSnapshot:
    """Return a session-scoped DeviceSnapshot, creating it on first call."""
    global _cached_snapshot
    gpu_available = has_gpu_runtime()
    if _cached_snapshot is not None:
        if _cached_snapshot.gpu_available == gpu_available:
            return _cached_snapshot
        _cached_snapshot = None

    try:
        from vibespatial.bench.profiling import _NvmlGpuSampler
        sampler = _NvmlGpuSampler()
    except Exception:
        sampler = None  # type: ignore[assignment]

    probe = _adapt_nvml_sampler(sampler) if sampler is not None else None

    profile = _detect_device_profile() if gpu_available else DEFAULT_CONSUMER_PROFILE

    _cached_snapshot = capture_device_snapshot(
        probe=probe,
        gpu_available=gpu_available,
        device_profile=profile,
    )
    return _cached_snapshot


def invalidate_snapshot_cache() -> None:
    """Clear the cached snapshot so the next call to get_cached_snapshot() re-probes."""
    global _cached_snapshot
    _cached_snapshot = None


# ---------------------------------------------------------------------------
# Phase 2: Convenience dispatch functions (ADR-0007)
# ---------------------------------------------------------------------------

def plan_kernel_dispatch(
    *,
    kernel_name: str,
    kernel_class: KernelClass | str,
    row_count: int,
    requested_mode: ExecutionMode | str = ExecutionMode.AUTO,
    requested_precision: PrecisionMode | str = PrecisionMode.AUTO,
    geometry_families: tuple[str, ...] = (),
    mixed_geometry: bool = False,
    current_residency: Residency = Residency.HOST,
    coordinate_stats: CoordinateStats | None = None,
    is_streaming: bool = False,
    chunk_index: int = 0,
    gpu_available: bool | None = None,
    workload_shape: WorkloadShape | None = None,
) -> AdaptivePlan:
    """Plan kernel dispatch with a cached device snapshot.

    This is the recommended entry point for all GPU dispatch decisions.
    It gets (or creates) a session-scoped DeviceSnapshot, builds a
    WorkloadProfile, and calls plan_adaptive_execution().
    """
    if gpu_available is False:
        snapshot = DeviceSnapshot(
            backend=MonitoringBackend.UNAVAILABLE,
            gpu_available=False,
            device_profile=DEFAULT_CONSUMER_PROFILE,
            reason="gpu_available explicitly set to False",
        )
    else:
        snapshot = get_cached_snapshot()
        if gpu_available is True and not snapshot.gpu_available:
            snapshot = DeviceSnapshot(
                backend=snapshot.backend,
                gpu_available=True,
                device_profile=snapshot.device_profile,
                sm_utilization_pct=snapshot.sm_utilization_pct,
                memory_utilization_pct=snapshot.memory_utilization_pct,
                device_name=snapshot.device_name,
                reason=f"{snapshot.reason}; gpu_available overridden to True",
            )

    workload = WorkloadProfile(
        row_count=row_count,
        geometry_families=geometry_families,
        mixed_geometry=mixed_geometry,
        current_residency=current_residency,
        coordinate_stats=coordinate_stats,
        is_streaming=is_streaming,
        chunk_index=chunk_index,
        workload_shape=workload_shape,
    )
    return plan_adaptive_execution(
        kernel_name=kernel_name,
        kernel_class=kernel_class,
        workload=workload,
        requested_mode=requested_mode,
        requested_precision=requested_precision,
        device_snapshot=snapshot,
    )


def plan_dispatch_selection(
    *,
    kernel_name: str,
    kernel_class: KernelClass | str,
    row_count: int,
    requested_mode: ExecutionMode | str = ExecutionMode.AUTO,
    gpu_available: bool | None = None,
    workload_shape: WorkloadShape | None = None,
) -> RuntimeSelection:
    """Thin wrapper: plan dispatch and return just the RuntimeSelection."""
    return plan_kernel_dispatch(
        kernel_name=kernel_name,
        kernel_class=kernel_class,
        row_count=row_count,
        requested_mode=requested_mode,
        gpu_available=gpu_available,
        workload_shape=workload_shape,
    ).runtime_selection


# ---------------------------------------------------------------------------
# Geometry complexity estimation for adaptive variant dispatch
# ---------------------------------------------------------------------------


def estimate_avg_vertices(owned: object) -> float:
    """Estimate average vertices per geometry from an OwnedGeometryArray.

    Used by kernel dispatch to select between simple (1-thread-per-geometry)
    and cooperative (1-block-per-geometry) kernel variants.  Returns 0.0 if
    the geometry array has no families or no coordinates.

    Accepts ``object`` to avoid importing OwnedGeometryArray at module scope
    (which would create a circular import).
    """
    families = getattr(owned, "families", None)
    row_count = getattr(owned, "row_count", 0)
    if not families or row_count == 0:
        return 0.0
    total_coords = 0
    for buf in families.values():
        x = getattr(buf, "x", None)
        if x is not None:
            total_coords += len(x)
    return total_coords / row_count
