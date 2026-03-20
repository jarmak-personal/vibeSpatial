from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ._runtime import ExecutionMode
from .precision import KernelClass


class DispatchDecision(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class CrossoverPolicy:
    kernel_name: str
    kernel_class: KernelClass
    auto_min_rows: int
    reason: str


DEFAULT_CROSSOVER_POLICIES: dict[KernelClass, int] = {
    KernelClass.COARSE: 1_000,
    KernelClass.METRIC: 5_000,
    KernelClass.PREDICATE: 10_000,
    KernelClass.CONSTRUCTIVE: 50_000,
}

_KERNEL_CROSSOVER_OVERRIDES: dict[str, int] = {
    "point_clip": 10_000,
    "point_buffer": 10_000,
    "linestring_buffer": 5_000,
    "segment_classify": 4_096,
    "flat_index_build": 0,
    "bbox_overlap_candidates": 0,
    "make_valid_repair": 2_000,
    "polygon_centroid": 500,
    "geometry_area": 500,
    "geometry_length": 500,
}


def default_crossover_policy(
    kernel_name: str,
    kernel_class: KernelClass | str,
) -> CrossoverPolicy:
    normalized_class = kernel_class if isinstance(kernel_class, KernelClass) else KernelClass(kernel_class)
    override = _KERNEL_CROSSOVER_OVERRIDES.get(kernel_name)
    if override is not None:
        return CrossoverPolicy(
            kernel_name=kernel_name,
            kernel_class=normalized_class,
            auto_min_rows=override,
            reason=f"kernel-specific crossover override for {kernel_name} is {override} rows",
        )
    threshold = DEFAULT_CROSSOVER_POLICIES[normalized_class]
    return CrossoverPolicy(
        kernel_name=kernel_name,
        kernel_class=normalized_class,
        auto_min_rows=threshold,
        reason=f"provisional auto crossover for {normalized_class.value} kernels is {threshold} rows",
    )


def select_dispatch_for_rows(
    *,
    requested_mode: ExecutionMode | str,
    row_count: int,
    policy: CrossoverPolicy,
    gpu_available: bool,
) -> DispatchDecision:
    mode = requested_mode if isinstance(requested_mode, ExecutionMode) else ExecutionMode(requested_mode)

    if mode is ExecutionMode.CPU:
        return DispatchDecision.CPU

    if mode is ExecutionMode.GPU:
        if not gpu_available:
            raise RuntimeError("GPU execution was requested, but no GPU runtime is available")
        return DispatchDecision.GPU

    if not gpu_available:
        return DispatchDecision.CPU

    if row_count < policy.auto_min_rows:
        return DispatchDecision.CPU

    return DispatchDecision.GPU
