from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from vibespatial.runtime import ExecutionMode, RuntimeSelection


class PrecisionMode(StrEnum):
    AUTO = "auto"
    FP32 = "fp32"
    FP64 = "fp64"


class KernelClass(StrEnum):
    COARSE = "coarse"
    METRIC = "metric"
    PREDICATE = "predicate"
    CONSTRUCTIVE = "constructive"


class CompensationMode(StrEnum):
    NONE = "none"
    CENTERED = "centered"
    KAHAN = "kahan"
    DOUBLE_SINGLE = "double-single"


class RefinementMode(StrEnum):
    NONE = "none"
    SELECTIVE_FP64 = "selective-fp64"
    EXACT = "exact"


@dataclass(frozen=True)
class DevicePrecisionProfile:
    name: str
    fp64_to_fp32_ratio: float

    @property
    def favors_native_fp64(self) -> bool:
        return self.fp64_to_fp32_ratio >= 0.25


@dataclass(frozen=True)
class CoordinateStats:
    max_abs_coord: float = 0.0
    span: float = 0.0

    @property
    def needs_centering(self) -> bool:
        if self.max_abs_coord == 0.0:
            return False
        if self.max_abs_coord >= 1_000_000.0:
            return True
        return self.span < (self.max_abs_coord * 0.1)


@dataclass(frozen=True)
class PrecisionPlan:
    storage_precision: PrecisionMode
    compute_precision: PrecisionMode
    kernel_class: KernelClass
    compensation: CompensationMode
    refinement: RefinementMode
    center_coordinates: bool
    reason: str


DEFAULT_CONSUMER_PROFILE = DevicePrecisionProfile(name="consumer-gpu", fp64_to_fp32_ratio=1.0 / 32.0)
DEFAULT_DATACENTER_PROFILE = DevicePrecisionProfile(name="datacenter-gpu", fp64_to_fp32_ratio=0.5)


def normalize_precision_mode(value: PrecisionMode | str) -> PrecisionMode:
    return value if isinstance(value, PrecisionMode) else PrecisionMode(value)


def _native_fp64_plan(kernel_class: KernelClass, reason: str) -> PrecisionPlan:
    return PrecisionPlan(
        storage_precision=PrecisionMode.FP64,
        compute_precision=PrecisionMode.FP64,
        kernel_class=kernel_class,
        compensation=CompensationMode.NONE,
        refinement=RefinementMode.NONE,
        center_coordinates=False,
        reason=reason,
    )


def _consumer_fp32_plan(kernel_class: KernelClass, coordinate_stats: CoordinateStats) -> PrecisionPlan:
    compensation = CompensationMode.KAHAN if kernel_class is KernelClass.METRIC else CompensationMode.CENTERED
    refinement = (
        RefinementMode.NONE
        if kernel_class in {KernelClass.COARSE, KernelClass.METRIC}
        else RefinementMode.SELECTIVE_FP64
    )
    compute_precision = (
        PrecisionMode.FP64 if kernel_class is KernelClass.CONSTRUCTIVE else PrecisionMode.FP32
    )
    if compute_precision is PrecisionMode.FP64:
        compensation = CompensationMode.NONE
        refinement = RefinementMode.NONE
    return PrecisionPlan(
        storage_precision=PrecisionMode.FP64,
        compute_precision=compute_precision,
        kernel_class=kernel_class,
        compensation=compensation,
        refinement=refinement,
        center_coordinates=compute_precision is PrecisionMode.FP32 and coordinate_stats.needs_centering,
        reason="consumer-style fp64 throughput is poor; prefer staged fp32 unless kernel class is constructive",
    )


def select_precision_plan(
    *,
    runtime_selection: RuntimeSelection,
    kernel_class: KernelClass,
    requested: PrecisionMode | str = PrecisionMode.AUTO,
    coordinate_stats: CoordinateStats | None = None,
    device_profile: DevicePrecisionProfile | None = None,
) -> PrecisionPlan:
    mode = normalize_precision_mode(requested)
    stats = coordinate_stats or CoordinateStats()

    if runtime_selection.selected is ExecutionMode.CPU:
        return _native_fp64_plan(kernel_class, "CPU execution uses native fp64 semantics")

    if mode is PrecisionMode.FP64:
        return _native_fp64_plan(kernel_class, "explicit fp64 precision requested")

    if mode is PrecisionMode.FP32:
        return PrecisionPlan(
            storage_precision=PrecisionMode.FP64,
            compute_precision=PrecisionMode.FP32,
            kernel_class=kernel_class,
            compensation=CompensationMode.KAHAN if kernel_class is KernelClass.METRIC else CompensationMode.CENTERED,
            refinement=(
                RefinementMode.NONE
                if kernel_class in {KernelClass.COARSE, KernelClass.METRIC}
                else RefinementMode.SELECTIVE_FP64
            ),
            center_coordinates=stats.needs_centering,
            reason="explicit fp32 precision requested",
        )

    profile = device_profile or DEFAULT_CONSUMER_PROFILE
    if profile.favors_native_fp64:
        return _native_fp64_plan(
            kernel_class,
            f"{profile.name} fp64 throughput is favorable enough for native fp64",
        )

    return _consumer_fp32_plan(kernel_class, stats)
