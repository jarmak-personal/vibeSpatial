from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ._runtime import ExecutionMode
from .precision import KernelClass


class WorkloadShape(StrEnum):
    """Classification of how left and right geometry arrays relate in size.

    PAIRWISE:        left and right have the same length; element-wise ops.
    BROADCAST_RIGHT: right has length 1, left has length > 1; the single
                     right geometry is broadcast against every left row.
    SCALAR_RIGHT:    right is a scalar (not an array); skips pandas index
                     alignment entirely.

    BROADCAST_LEFT is intentionally omitted — no consumer exists today.
    INDEXED is intentionally omitted — gather-evaluate-scatter is a
    different computation model, not a workload shape.
    """

    PAIRWISE = "pairwise"
    BROADCAST_RIGHT = "broadcast_right"
    SCALAR_RIGHT = "scalar_right"


def detect_workload_shape(
    left_count: int,
    right_count: int | None,
) -> WorkloadShape:
    """Classify the workload shape for a binary operation."""
    if right_count is None:
        return WorkloadShape.SCALAR_RIGHT
    if right_count == 1 and left_count > 1:
        return WorkloadShape.BROADCAST_RIGHT
    if left_count == right_count:
        return WorkloadShape.PAIRWISE
    raise ValueError(
        f"Incompatible lengths: left={left_count}, right={right_count}. "
        "Use gpd.sjoin() for many-to-many operations."
    )


class DispatchDecision(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class CrossoverPolicy:
    """Per-kernel crossover thresholds for AUTO dispatch.

    ``auto_min_rows`` is the pairwise threshold (left and right have the
    same length).  ``broadcast_min_rows`` is an optional lower threshold
    for broadcast workload shapes (BROADCAST_RIGHT / SCALAR_RIGHT) where
    the right-side geometry fits in L1 cache and is reused N times,
    making GPU profitable at much smaller N.
    """

    kernel_name: str
    kernel_class: KernelClass
    auto_min_rows: int
    reason: str
    broadcast_min_rows: int | None = None


# Pairwise thresholds by kernel class.
DEFAULT_CROSSOVER_POLICIES: dict[KernelClass, int] = {
    KernelClass.COARSE: 1_000,
    KernelClass.METRIC: 5_000,
    KernelClass.PREDICATE: 10_000,
    KernelClass.CONSTRUCTIVE: 50_000,
}

# Broadcast thresholds by kernel class.  These are lower than the
# pairwise thresholds because broadcast-right has perfect right-side
# data locality: one geometry, read once, reused N times from L1 cache.
DEFAULT_BROADCAST_CROSSOVER_POLICIES: dict[KernelClass, int] = {
    KernelClass.COARSE: 256,
    KernelClass.METRIC: 500,
    KernelClass.PREDICATE: 1_000,
    KernelClass.CONSTRUCTIVE: 500,
}

_KERNEL_CROSSOVER_OVERRIDES: dict[str, int] = {
    "normalize": 500,
    "point_clip": 10_000,
    "point_buffer": 500,
    "linestring_buffer": 5_000,
    "segment_classify": 4_096,
    "flat_index_build": 0,
    "bbox_overlap_candidates": 2_048,
    "point_regular_grid_candidates": 0,
    "point_box_query": 0,
    "spatial_index_knn": 0,
    "make_valid_repair": 2_000,
    "polygon_centroid": 500,
    "geometry_area": 500,
    "geometry_length": 500,
}

_BROADCAST_SHAPES = frozenset({WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT})


def default_crossover_policy(
    kernel_name: str,
    kernel_class: KernelClass | str,
) -> CrossoverPolicy:
    normalized_class = kernel_class if isinstance(kernel_class, KernelClass) else KernelClass(kernel_class)
    override = _KERNEL_CROSSOVER_OVERRIDES.get(kernel_name)
    broadcast_threshold = DEFAULT_BROADCAST_CROSSOVER_POLICIES[normalized_class]
    if override is not None:
        return CrossoverPolicy(
            kernel_name=kernel_name,
            kernel_class=normalized_class,
            auto_min_rows=override,
            reason=f"kernel-specific crossover override for {kernel_name} is {override} rows",
            broadcast_min_rows=broadcast_threshold,
        )
    threshold = DEFAULT_CROSSOVER_POLICIES[normalized_class]
    return CrossoverPolicy(
        kernel_name=kernel_name,
        kernel_class=normalized_class,
        auto_min_rows=threshold,
        reason=f"provisional auto crossover for {normalized_class.value} kernels is {threshold} rows",
        broadcast_min_rows=broadcast_threshold,
    )


def select_dispatch_for_rows(
    *,
    requested_mode: ExecutionMode | str,
    row_count: int,
    policy: CrossoverPolicy,
    gpu_available: bool,
    workload_shape: WorkloadShape | None = None,
) -> DispatchDecision:
    """Select CPU or GPU execution based on row count and crossover policy.

    When *workload_shape* is ``BROADCAST_RIGHT`` or ``SCALAR_RIGHT``, the
    effective threshold is ``policy.broadcast_min_rows`` (or
    ``policy.auto_min_rows // 10`` if the policy does not set a broadcast
    threshold).  This reflects the fact that broadcast workloads have
    perfect right-side data locality and benefit from GPU execution at
    much smaller N than pairwise workloads.
    """
    mode = requested_mode if isinstance(requested_mode, ExecutionMode) else ExecutionMode(requested_mode)

    if mode is ExecutionMode.CPU:
        return DispatchDecision.CPU

    if mode is ExecutionMode.GPU:
        if not gpu_available:
            raise RuntimeError("GPU execution was requested, but no GPU runtime is available")
        return DispatchDecision.GPU

    if not gpu_available:
        return DispatchDecision.CPU

    # Select the effective threshold based on workload shape.
    if workload_shape is not None and workload_shape in _BROADCAST_SHAPES:
        threshold = (
            policy.broadcast_min_rows
            if policy.broadcast_min_rows is not None
            else policy.auto_min_rows // 10
        )
    else:
        threshold = policy.auto_min_rows

    if row_count < threshold:
        return DispatchDecision.CPU

    return DispatchDecision.GPU
