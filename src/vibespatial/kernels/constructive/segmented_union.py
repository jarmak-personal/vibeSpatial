"""Segmented union_all kernel: per-group polygon union via GPU tree-reduce.

ADR-0002: CONSTRUCTIVE class -- fp64 by design on all devices.
ADR-0033: Tier classification -- delegates to overlay pipeline (Tier 1 NVRTC
          + Tier 3a CCCL + Tier 2 CuPy) via overlay_union_owned.
ADR-0034: Inherits overlay pipeline precompilation; no new NVRTC source.

Algorithm
---------
For each group defined by ``group_offsets`` (CSR-style):
  - size 0: produce empty polygon
  - size 1: pass through unchanged
  - size 2: single pairwise overlay union
  - size N>2: binary-tree reduction in ceil(log2(N)) rounds

Tree reduction intermediates stay device-resident throughout.  Final
group results are concatenated at the buffer level (no Shapely
round-trip) via ``_concat_owned_arrays``.

The overlay pipeline is strictly pairwise (left[i] vs right[i]), so each
pair union is a separate GPU dispatch.  Groups are iterated serially with
GPU parallelism *within* each overlay_union_owned call.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from vibespatial.constructive import segmented_union_cpu as _segmented_union_cpu_module
from vibespatial.constructive.segmented_union_host import (
    concat_owned_arrays,
    group_has_only_polygon_families,
    group_indices,
    normalize_group_offsets,
    singleton_indices,
    valid_row_indices,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
)
from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import OVERLAY_GPU_FAILURE_THRESHOLD
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.kernel_registry import register_kernel_variant
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    normalize_precision_mode,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.runtime.robustness import select_robustness_plan

if TYPE_CHECKING:
    from vibespatial.geometry.owned import OwnedGeometryArray

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover
    cp = None

_get_empty_owned = _segmented_union_cpu_module.get_empty_owned
_segmented_union_cpu = _segmented_union_cpu_module.segmented_union_cpu
_segmented_union_pair_cpu = _segmented_union_cpu_module.segmented_union_pair_cpu


def _empty_group_owned_like(source: OwnedGeometryArray) -> OwnedGeometryArray:
    empty = _get_empty_owned().take(singleton_indices(0))
    if source.residency is Residency.DEVICE and cp is not None:
        empty.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="segmented union empty group matches device-resident input",
        )
    return empty


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segmented_union_all(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """Union all geometries within each group.  Returns one geometry per group.

    Parameters
    ----------
    geometries : OwnedGeometryArray
        Input polygons (device- or host-resident).
    group_offsets : array-like
        CSR-style int32/int64 offsets.  Group *i* contains
        ``geometries[group_offsets[i]:group_offsets[i+1]]``.
        Length is ``n_groups + 1``.
    dispatch_mode : ExecutionMode or str
        Execution mode hint (AUTO, GPU, CPU).
    precision : PrecisionMode or str
        Precision mode.  CONSTRUCTIVE kernels stay fp64 per ADR-0002.

    Returns
    -------
    OwnedGeometryArray
        One geometry per group.  May contain MultiPolygon when union
        produces disconnected regions.  Empty groups produce empty Polygon.
    """
    from vibespatial.geometry.owned import from_shapely_geometries

    requested = (
        dispatch_mode
        if isinstance(dispatch_mode, ExecutionMode)
        else ExecutionMode(dispatch_mode)
    )
    precision_mode = normalize_precision_mode(precision)

    group_offsets = normalize_group_offsets(group_offsets)
    n_groups = len(group_offsets) - 1
    if n_groups < 0:
        raise ValueError("group_offsets must have length >= 1")
    if n_groups == 0:
        return from_shapely_geometries([])

    total_geoms = int(group_offsets[-1])

    # Dispatch selection
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=total_geoms,
        requested_mode=requested,
        requested_precision=precision_mode,
        current_residency=combined_residency(geometries),
    )

    # ADR-0002: CONSTRUCTIVE kernels stay fp64.  Precision plan is computed
    # for observability (dispatch event detail) only.
    precision_plan = selection.precision_plan
    select_robustness_plan(
        kernel_class=KernelClass.CONSTRUCTIVE,
        precision_plan=precision_plan,
    )

    if selection.selected is ExecutionMode.GPU:
        result = _segmented_union_gpu(
            geometries,
            group_offsets,
            n_groups=n_groups,
            precision_plan=precision_plan,
        )
        if result is not None:
            record_dispatch_event(
                surface="segmented_union_all",
                operation="segmented_union_all",
                implementation="gpu_tree_reduce_overlay",
                reason=selection.reason,
                detail=(
                    f"groups={n_groups}, total_geoms={total_geoms}, "
                    f"precision={precision_plan.compute_precision.value}"
                ),
                requested=selection.requested,
                selected=ExecutionMode.GPU,
            )
            result.record_runtime_selection(selection)
            return result

    # CPU fallback
    result = _segmented_union_cpu(geometries, group_offsets, n_groups=n_groups)
    record_dispatch_event(
        surface="segmented_union_all",
        operation="segmented_union_all",
        implementation="shapely_union_all",
        reason=selection.reason if selection.selected is ExecutionMode.CPU else "GPU fallback to CPU",
        detail=f"groups={n_groups}, total_geoms={total_geoms}",
        requested=selection.requested,
        selected=ExecutionMode.CPU,
    )
    result.record_runtime_selection(selection)
    return result


# ---------------------------------------------------------------------------
# GPU variant: tree-reduce via overlay_union_owned per group
# ---------------------------------------------------------------------------


@register_kernel_variant(
    "segmented_union_all",
    "gpu-overlay-tree-reduce",
    kernel_class=KernelClass.CONSTRUCTIVE,
    geometry_families=("polygon", "multipolygon"),
    execution_modes=(ExecutionMode.GPU,),
    supports_mixed=True,
    precision_modes=(PrecisionMode.AUTO, PrecisionMode.FP64),
    tags=("constructive", "segmented-union", "gpu", "tree-reduce"),
)
def _segmented_union_gpu_variant(
    geometries: OwnedGeometryArray,
    group_offsets: Any,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> OwnedGeometryArray:
    """GPU variant: tree-reduce pairwise overlay union per group."""
    group_offsets = normalize_group_offsets(group_offsets)
    n_groups = len(group_offsets) - 1
    precision_mode = normalize_precision_mode(precision)
    selection = plan_dispatch_selection(
        kernel_name="segmented_union_all",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=int(group_offsets[-1]),
        requested_mode=dispatch_mode,
        requested_precision=precision_mode,
        current_residency=combined_residency(geometries),
    )
    precision_plan = selection.precision_plan
    return _segmented_union_gpu(
        geometries, group_offsets, n_groups=n_groups, precision_plan=precision_plan,
    )


def _segmented_union_gpu(
    geometries: OwnedGeometryArray,
    group_offsets,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """GPU tree-reduce: union geometries within each group.

    For each group, performs binary-tree reduction using overlay_union_owned.
    Intermediate results stay device-resident (ADR-0005 zero-copy).
    Falls back to None on failure so the caller can retry on CPU.

    ADR-0002: CONSTRUCTIVE class, fp64 (segment intersection precision).
    ADR-0033: Inherits overlay pipeline tiers (NVRTC + CCCL + CuPy).
    """
    from vibespatial.constructive.union_all import union_all_gpu_owned

    # Validate: GPU overlay requires polygon-family geometries.
    polygon_tags = {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    if not group_has_only_polygon_families(geometries, polygon_tags):
        # Non-polygon geometry present: fall back to CPU.
        return None

    # Single-group dissolve is common in workflow benchmarks and should not
    # route through the legacy per-row Python reduction. Reuse the global
    # batched GPU tree-reduce path so each round processes whole batches.
    if n_groups == 1:
        from vibespatial.constructive.union_all import union_all_gpu_owned

        keep = valid_row_indices(geometries)
        if keep.size == 0:
            return _empty_group_owned_like(geometries)
        if keep.size < geometries.row_count:
            geometries = geometries.take(keep)
        if geometries.row_count == 1:
            return geometries
        try:
            return union_all_gpu_owned(
                geometries,
                dispatch_mode=ExecutionMode.GPU,
                precision=precision_plan.compute_precision,
            )
        except Exception:
            return None

    group_results: list[OwnedGeometryArray] = []

    for g in range(n_groups):
        start = int(group_offsets[g])
        end = int(group_offsets[g + 1])
        group_size = end - start

        if group_size == 0:
            group_results.append(_empty_group_owned_like(geometries))
            continue

        if group_size == 1:
            single = geometries.take(singleton_indices(start))
            if not single.validity[0]:
                group_results.append(_empty_group_owned_like(geometries))
            else:
                group_results.append(single)
            continue

        # Extract this group's geometries.
        indices = group_indices(start, end)
        group_owned = geometries.take(indices)

        # Filter out invalid/empty rows (vectorized, not Python loop).
        keep = valid_row_indices(group_owned)
        if keep.size == 0:
            group_results.append(_empty_group_owned_like(geometries))
            continue
        if keep.size < group_owned.row_count:
            group_owned = group_owned.take(keep)

        if group_owned.row_count == 1:
            group_results.append(group_owned)
            continue

        try:
            reduced = union_all_gpu_owned(
                group_owned,
                dispatch_mode=ExecutionMode.GPU,
                precision=precision_plan.compute_precision,
            )
            group_results.append(reduced)
        except Exception:
            # Fall back to CPU for this group on any GPU overlay failure.
            return None

    # Buffer-level concatenation: no Shapely round-trip (zero-copy).
    return concat_owned_arrays(group_results)


def _tree_reduce_group(group_owned: OwnedGeometryArray) -> OwnedGeometryArray:
    """Binary-tree reduction of group_owned via overlay_union_owned.

    Intermediate results stay device-resident.  O(log2(N)) rounds.

    If a GPU overlay fails (e.g. ILLEGAL_ADDRESS from degenerate half-edge
    topology), the pair falls back to Shapely CPU union.  After multiple
    consecutive GPU failures (suggesting CUDA context corruption), the
    rest of the reduction proceeds entirely on CPU.
    """
    from vibespatial.overlay.gpu import overlay_union_owned

    # Split into single-row OwnedGeometryArrays for pairwise reduction.
    current: list[OwnedGeometryArray] = []
    for i in range(group_owned.row_count):
        current.append(group_owned.take(singleton_indices(i)))

    rounds = 0
    max_rounds = int(math.ceil(math.log2(max(len(current), 2)))) + 2
    consecutive_gpu_failures = 0
    while len(current) > 1 and rounds < max_rounds:
        next_round: list[OwnedGeometryArray] = []
        for i in range(0, len(current), 2):
            if i + 1 < len(current):
                gpu_ok = False
                if consecutive_gpu_failures < OVERLAY_GPU_FAILURE_THRESHOLD:
                    try:
                        merged = overlay_union_owned(
                            current[i],
                            current[i + 1],
                            dispatch_mode=ExecutionMode.GPU,
                        )
                        merged = merged.take(singleton_indices(0))
                        next_round.append(merged)
                        gpu_ok = True
                        consecutive_gpu_failures = 0
                    except Exception:
                        consecutive_gpu_failures += 1

                if not gpu_ok:
                    # CPU fallback for this pair.
                    cpu_merged = _segmented_union_pair_cpu(current[i], current[i + 1])
                    next_round.append(cpu_merged)
            else:
                # Odd element passes through.
                next_round.append(current[i])
        # Explicit cleanup: release previous round's intermediates promptly
        # to avoid accumulating device memory across reduction rounds.
        del current
        current = next_round
        rounds += 1

    return current[0]
