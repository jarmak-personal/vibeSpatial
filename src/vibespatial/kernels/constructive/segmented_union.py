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

import numpy as np

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
_SEGMENTED_UNION_SERIAL_SMALL_MAX_GROUP_SIZE = 8
_SEGMENTED_UNION_ROBUST_SNAP_GRID = 1.0e-9
_SEGMENTED_UNION_ROBUST_SNAP_PRE_MAX_COORDS = 4096
_SEGMENTED_UNION_ROBUST_SNAP_RETRY_MAX_PARTS = 128


def _empty_group_owned_like(source: OwnedGeometryArray) -> OwnedGeometryArray:
    empty = _get_empty_owned().take(singleton_indices(0))
    if source.residency is Residency.DEVICE and cp is not None:
        empty.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="segmented union empty group matches device-resident input",
        )
    return empty


def _robust_snap_segmented_union_inputs_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    record: bool,
) -> OwnedGeometryArray | None:
    """Snap grouped-union inputs to a sub-nanometer device grid.

    Grouped dissolves often feed overlay results whose shared seams differ by
    floating-point dust after several constructive stages.  Snap-rounding the
    inputs before reduction closes only those sub-grid seams; the subsequent
    union still does the topology work.
    """
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None
    if geometries.row_count == 0:
        return geometries
    if int(np.diff(group_offsets).max(initial=0)) <= 1:
        return geometries

    from vibespatial.constructive.set_precision import _set_precision_gpu

    try:
        snapped = _set_precision_gpu(
            geometries,
            _SEGMENTED_UNION_ROBUST_SNAP_GRID,
            "pointwise",
        )
    except Exception:
        return None

    if record:
        record_dispatch_event(
            surface="segmented_union_all",
            operation="segmented_union_all_precision_snap",
            implementation="gpu_cupy_pointwise_snap",
            reason="robust grouped union seam snap",
            detail=(
                f"rows={geometries.row_count}, grid_size="
                f"{_SEGMENTED_UNION_ROBUST_SNAP_GRID}"
            ),
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
        )
    return snapped


def _owned_coordinate_count(owned: OwnedGeometryArray) -> int:
    if owned.device_state is not None:
        return sum(int(buf.x.size) for buf in owned.device_state.families.values())
    return sum(int(buf.x.size) for buf in owned.families.values())


def _should_pre_snap_segmented_union_inputs(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
) -> bool:
    if geometries.row_count == 0:
        return False
    if int(np.diff(group_offsets).max(initial=0)) <= 1:
        return False
    return _owned_coordinate_count(geometries) <= _SEGMENTED_UNION_ROBUST_SNAP_PRE_MAX_COORDS


def _polygon_exploded_part_count_gpu(result: OwnedGeometryArray) -> int:
    """Return the number of polygonal parts an explode would expose."""
    if result.device_state is None:
        return 0
    count = 0
    d_state = result.device_state
    polygon_tag = FAMILY_TAGS[GeometryFamily.POLYGON]
    if GeometryFamily.POLYGON in d_state.families:
        count += int(cp.count_nonzero(d_state.tags == polygon_tag).item())
    if GeometryFamily.MULTIPOLYGON not in d_state.families:
        return count
    d_buf = d_state.families[GeometryFamily.MULTIPOLYGON]
    if d_buf.part_offsets is not None:
        count += int(d_buf.part_offsets.size) - 1
    return count


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
    # Validate: GPU overlay requires polygon-family geometries.
    polygon_tags = {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    if not group_has_only_polygon_families(geometries, polygon_tags):
        # Non-polygon geometry present: fall back to CPU.
        return None

    working_geometries = geometries
    pre_snapped = False
    if _should_pre_snap_segmented_union_inputs(geometries, group_offsets):
        snapped = _robust_snap_segmented_union_inputs_gpu(
            geometries,
            group_offsets,
            record=True,
        )
        if snapped is not None:
            working_geometries = snapped
            pre_snapped = True

    result = _segmented_union_gpu_impl(
        working_geometries,
        group_offsets,
        n_groups=n_groups,
        precision_plan=precision_plan,
    )
    original_part_count = _polygon_exploded_part_count_gpu(result) if result is not None else 0
    if (
        not pre_snapped
        and 1 < original_part_count <= _SEGMENTED_UNION_ROBUST_SNAP_RETRY_MAX_PARTS
    ):
        snapped = _robust_snap_segmented_union_inputs_gpu(
            geometries,
            group_offsets,
            record=False,
        )
        if snapped is not None:
            retry = _segmented_union_gpu_impl(
                snapped,
                group_offsets,
                n_groups=n_groups,
                precision_plan=precision_plan,
            )
            if (
                retry is not None
                and _polygon_exploded_part_count_gpu(retry) < original_part_count
            ):
                record_dispatch_event(
                    surface="segmented_union_all",
                    operation="segmented_union_all_precision_snap",
                    implementation="gpu_cupy_pointwise_snap",
                    reason="robust grouped union seam snap reduced exploded parts",
                    detail=(
                        f"rows={geometries.row_count}, grid_size="
                        f"{_SEGMENTED_UNION_ROBUST_SNAP_GRID}, "
                        f"parts={original_part_count}->{_polygon_exploded_part_count_gpu(retry)}"
                    ),
                    requested=ExecutionMode.GPU,
                    selected=ExecutionMode.GPU,
                )
                return retry
    return result


def _segmented_union_gpu_impl(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    from vibespatial.constructive.union_all import union_all_gpu_owned

    # Single-group dissolve is common in workflow benchmarks and should not
    # route through the legacy per-row Python reduction. Reuse the global
    # batched GPU tree-reduce path so each round processes whole batches.
    if n_groups == 1:
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

    if int(np.diff(group_offsets).max(initial=0)) <= _SEGMENTED_UNION_SERIAL_SMALL_MAX_GROUP_SIZE:
        return _segmented_union_serial_gpu(
            geometries,
            group_offsets,
            n_groups=n_groups,
            precision_plan=precision_plan,
        )

    grouped = _segmented_union_grouped_overlay_gpu(
        geometries,
        group_offsets,
        n_groups=n_groups,
        precision_plan=precision_plan,
    )
    if grouped is not None:
        return grouped

    return _segmented_union_batched_gpu(
        geometries,
        group_offsets,
        n_groups=n_groups,
        precision_plan=precision_plan,
    )


def _segmented_union_serial_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """Compatibility path for tiny groups.

    Standalone ``union_all`` has observable MultiPolygon component ordering for
    small public dissolves.  Keep those cases on the existing per-group GPU
    reducer; batched reduction is for the larger workflow shape where launch
    amortization dominates.
    """
    from vibespatial.constructive.union_all import union_all_gpu_owned

    group_results: list[OwnedGeometryArray] = []

    for group_index in range(n_groups):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
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

        group_owned = geometries.take(group_indices(start, end))
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
            return None

    return concat_owned_arrays(group_results)


def _segmented_union_grouped_overlay_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """Union all groups with one row-isolated grouped overlay plan.

    This is the grouped geometry-reduce physical shape: one seed row per
    output group, all remaining rows mapped back to that group, and a single
    overlay union materialization instead of log2(N) pairwise union rounds.
    """
    del precision_plan
    if cp is None:  # pragma: no cover - exercised only on CPU-only installs
        return None

    from vibespatial.constructive.binary_constructive import (
        _regroup_intersection_parts_with_grouped_union_gpu,
    )
    from vibespatial.constructive.union_all import (
        _SPATIAL_LOCALIZE_MIN_ROWS,
        _spatially_localize_polygon_union_inputs,
    )

    group_offsets_arr = np.asarray(group_offsets, dtype=np.int64)
    max_group_size = int(np.diff(group_offsets_arr).max(initial=0))
    if max_group_size < _SPATIAL_LOCALIZE_MIN_ROWS:
        total_rows = int(group_offsets_arr[-1]) if group_offsets_arr.size else 0
        if total_rows != geometries.row_count:
            return None
        if total_rows == 0:
            return concat_owned_arrays([_empty_group_owned_like(geometries) for _ in range(n_groups)])

        d_group_offsets = cp.asarray(group_offsets_arr, dtype=cp.int64)
        d_positions = cp.arange(total_rows, dtype=cp.int64)
        d_source_rows = cp.searchsorted(
            d_group_offsets[1:],
            d_positions,
            side="right",
        ).astype(cp.int64, copy=False)

        state = geometries._ensure_device_state()
        d_valid = state.validity[:total_rows].astype(cp.bool_, copy=False)
        valid_count = int(cp.count_nonzero(d_valid).item())
        if valid_count == 0:
            return concat_owned_arrays([_empty_group_owned_like(geometries) for _ in range(n_groups)])
        if valid_count == total_rows:
            valid_geometries = geometries
            d_valid_source_rows = d_source_rows
        else:
            d_valid_positions = cp.flatnonzero(d_valid).astype(cp.int64, copy=False)
            valid_geometries = geometries.take(d_valid_positions)
            d_valid_source_rows = d_source_rows[d_valid_positions]

        try:
            return _regroup_intersection_parts_with_grouped_union_gpu(
                valid_geometries,
                d_valid_source_rows,
                output_row_count=n_groups,
                dispatch_mode=ExecutionMode.GPU,
            )
        except Exception:
            return None

    group_pieces: list[OwnedGeometryArray] = []
    source_rows: list[Any] = []

    for group_index in range(n_groups):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        if end <= start:
            continue

        group_owned = geometries.take(group_indices(start, end))
        keep = valid_row_indices(group_owned)
        if keep.size == 0:
            continue
        if keep.size < group_owned.row_count:
            group_owned = group_owned.take(keep)
        if group_owned.row_count > 2:
            group_owned = _spatially_localize_polygon_union_inputs(group_owned)

        group_pieces.append(group_owned)
        source_rows.append(
            cp.full(group_owned.row_count, group_index, dtype=cp.int32)
        )

    if not group_pieces:
        return concat_owned_arrays([_empty_group_owned_like(geometries) for _ in range(n_groups)])

    try:
        return _regroup_intersection_parts_with_grouped_union_gpu(
            concat_owned_arrays(group_pieces),
            cp.concatenate(source_rows),
            output_row_count=n_groups,
            dispatch_mode=ExecutionMode.GPU,
        )
    except Exception:
        return None


def _segmented_union_batched_gpu(
    geometries: OwnedGeometryArray,
    group_offsets: np.ndarray,
    *,
    n_groups: int,
    precision_plan: PrecisionPlan,
) -> OwnedGeometryArray | None:
    """Reduce all dissolve groups level-by-level instead of group-by-group.

    Each tree level batches pairwise unions for every active group into one
    binary constructive dispatch.  Group boundaries are preserved by carrying
    odd rows and rebuilding CSR offsets between rounds.
    """
    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.constructive.union_all import _spatially_localize_polygon_union_inputs

    group_pieces: list[OwnedGeometryArray] = []
    group_sizes = np.zeros(n_groups, dtype=np.int64)
    empty_group = _empty_group_owned_like(geometries)

    for group_index in range(n_groups):
        start = int(group_offsets[group_index])
        end = int(group_offsets[group_index + 1])
        if end <= start:
            group_pieces.append(empty_group)
            group_sizes[group_index] = 1
            continue

        group_owned = geometries.take(group_indices(start, end))
        keep = valid_row_indices(group_owned)
        if keep.size == 0:
            group_pieces.append(empty_group)
            group_sizes[group_index] = 1
            continue
        if keep.size < group_owned.row_count:
            group_owned = group_owned.take(keep)
        if group_owned.row_count > 2:
            group_owned = _spatially_localize_polygon_union_inputs(group_owned)

        group_pieces.append(group_owned)
        group_sizes[group_index] = group_owned.row_count

    current = concat_owned_arrays(group_pieces)
    current_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(group_sizes, dtype=np.int64),
        ]
    )

    rounds = 0
    max_group_size = int(group_sizes.max(initial=1))
    max_rounds = int(math.ceil(math.log2(max(max_group_size, 2)))) + 2
    while rounds < max_rounds:
        group_sizes = np.diff(current_offsets)
        if not bool(np.any(group_sizes > 1)):
            return current

        pair_counts = group_sizes // 2
        pair_count = int(pair_counts.sum())
        if pair_count == 0:
            return current

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        carry_parts: list[int] = []
        next_order_parts: list[np.ndarray] = []
        pair_cursor = 0
        carry_cursor = pair_count
        next_sizes = (pair_counts + (group_sizes % 2)).astype(np.int64, copy=False)

        for start, size, group_pair_count in zip(
            current_offsets[:-1], group_sizes, pair_counts, strict=True
        ):
            start = int(start)
            size = int(size)
            group_pair_count = int(group_pair_count)
            if group_pair_count:
                left = start + (np.arange(group_pair_count, dtype=np.int64) * 2)
                left_parts.append(left)
                right_parts.append(left + 1)
                next_order_parts.append(
                    np.arange(pair_cursor, pair_cursor + group_pair_count, dtype=np.int64)
                )
                pair_cursor += group_pair_count
            if size % 2:
                carry_parts.append(start + size - 1)
                next_order_parts.append(np.asarray([carry_cursor], dtype=np.int64))
                carry_cursor += 1

        left_indices = np.concatenate(left_parts) if left_parts else np.asarray([], dtype=np.int64)
        right_indices = np.concatenate(right_parts) if right_parts else np.asarray([], dtype=np.int64)
        try:
            next_round = binary_constructive_owned(
                "union",
                current.take(left_indices),
                current.take(right_indices),
                dispatch_mode=ExecutionMode.GPU,
                precision=precision_plan.compute_precision,
                _skip_polygon_contraction=True,
            )
        except Exception:
            return None

        if carry_parts:
            carry_rows = current.take(np.asarray(carry_parts, dtype=np.int64))
            combined = concat_owned_arrays([next_round, carry_rows])
        else:
            combined = next_round

        next_order = (
            np.concatenate(next_order_parts)
            if next_order_parts
            else np.asarray([], dtype=np.int64)
        )
        current = combined.take(next_order) if next_order.size else combined
        current_offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(next_sizes, dtype=np.int64),
            ]
        )
        rounds += 1

    return None


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
