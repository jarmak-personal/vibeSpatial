"""Geometry-only spatial overlay over native relation and grouped carriers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.dispatch import record_dispatch_event

if TYPE_CHECKING:
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.spatial.indexing import CandidatePairs

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None


_SUPPORTED_OPERATIONS = frozenset(
    {"intersection", "difference", "identity", "symmetric_difference", "union"}
)


def _empty_owned_like(owned: OwnedGeometryArray, *, device: bool) -> OwnedGeometryArray:
    if device:
        return owned._device_indexed_take(cp.empty(0, dtype=cp.int64))
    return owned.take(())


def _device_relation_from_candidate_pairs(
    candidate_pairs: CandidatePairs,
    *,
    left_row_count: int,
    right_row_count: int,
) -> NativeRelation:
    """Lower device candidate columns into the canonical relation carrier."""
    from vibespatial.api._native_relation import NativeRelation

    if candidate_pairs.device_left_indices is None or candidate_pairs.device_right_indices is None:
        raise RuntimeError("admitted spatial overlay requires device relation pairs")
    left_rows = cp.asarray(candidate_pairs.device_left_indices, dtype=cp.int32)
    right_rows = cp.asarray(candidate_pairs.device_right_indices, dtype=cp.int32)
    if int(left_rows.size) != int(right_rows.size):
        raise ValueError("spatial overlay candidate pair columns must have equal length")
    return NativeRelation(
        left_indices=left_rows,
        right_indices=right_rows,
        predicate="bounds_intersects",
        left_row_count=left_row_count,
        right_row_count=right_row_count,
        sorted_by_left=False,
    )


def _device_pair_order(primary: Any, secondary: Any) -> Any:
    """Return exact lexicographic pair order through one packed radix key."""
    from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

    if int(primary.size) == 0:
        return cp.empty(0, dtype=cp.int64)
    if int(primary.size) > (1 << 32) - 1:
        raise OverflowError("spatial overlay relation exceeds radix lane width")
    keys = (primary.astype(cp.uint64, copy=False) << cp.uint64(32)) | secondary.astype(
        cp.uint32, copy=False
    ).astype(cp.uint64, copy=False)
    return sort_pairs(
        keys,
        cp.arange(primary.size, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    ).values.astype(cp.int64, copy=False)


def _grouped_difference_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    pair_left: Any,
    pair_right: Any,
    order: Any,
) -> OwnedGeometryArray:
    """Subtract all related right rows in one full-source grouped topology plan."""
    if int(pair_left.size) == 0:
        return left

    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    source_rows = pair_left[order].astype(cp.int32, copy=False)
    gathered_right = right._device_indexed_take(
        pair_right[order],
        assume_unique_indices=False,
    )
    plan = _build_overlay_execution_plan(
        left,
        gathered_right,
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
        _right_geometry_source_rows=source_rows,
        _right_segment_source_rows=source_rows,
        _include_same_side_splits=True,
    )
    result, selected = _materialize_overlay_execution_plan(
        plan,
        operation="difference",
        requested=ExecutionMode.GPU,
        preserve_row_count=left.row_count,
    )
    if selected is not ExecutionMode.GPU or result.row_count != left.row_count:
        raise RuntimeError("grouped spatial difference did not return full device row capacity")
    return result


def _pair_intersection_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    pair_left: Any,
    pair_right: Any,
    order: Any,
) -> OwnedGeometryArray:
    """Construct all relation-pair intersections in one row-isolated plan."""
    if int(pair_left.size) == 0:
        return _empty_owned_like(left, device=True)

    from vibespatial.overlay.gpu import _overlay_owned

    left_rows = left._device_indexed_take(
        pair_left[order],
        assume_unique_indices=False,
    )
    right_rows = right._device_indexed_take(
        pair_right[order],
        assume_unique_indices=False,
    )
    return _overlay_owned(
        left_rows,
        right_rows,
        operation="intersection",
        dispatch_mode=ExecutionMode.GPU,
        _row_isolated=True,
    )


def _filter_device_output(owned: OwnedGeometryArray) -> OwnedGeometryArray:
    from vibespatial.overlay.gpu import _filter_non_empty_owned_device

    selected = _filter_non_empty_owned_device(owned)
    if selected is None:
        raise RuntimeError("admitted spatial overlay output lacked device metadata")
    return selected


def _spatial_overlay_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    relation: NativeRelation,
    *,
    how: str,
) -> OwnedGeometryArray:
    pair_left = cp.asarray(relation.left_indices, dtype=cp.int64)
    pair_right = cp.asarray(relation.right_indices, dtype=cp.int64)
    if int(pair_left.size) != int(pair_right.size):
        raise ValueError("spatial overlay relation pair columns must have equal length")
    left_order = _device_pair_order(pair_left, pair_right)
    parts: list[OwnedGeometryArray] = []

    if how in {"intersection", "identity", "union"}:
        parts.append(
            _pair_intersection_device(
                left,
                right,
                pair_left,
                pair_right,
                left_order,
            )
        )
    if how in {"difference", "identity", "symmetric_difference", "union"}:
        parts.append(
            _grouped_difference_device(
                left,
                right,
                pair_left,
                pair_right,
                left_order,
            )
        )
    if how in {"symmetric_difference", "union"}:
        right_order = _device_pair_order(pair_right, pair_left)
        parts.append(
            _grouped_difference_device(
                right,
                left,
                pair_right,
                pair_left,
                right_order,
            )
        )

    if not parts:
        return _empty_owned_like(left, device=True)
    result = parts[0] if len(parts) == 1 else OwnedGeometryArray.concat(parts)
    return _filter_device_output(result)


def spatial_overlay_owned(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    how: str = "intersection",
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
) -> OwnedGeometryArray:
    """Execute geometry-only planar overlay from one candidate relation."""
    if how not in _SUPPORTED_OPERATIONS:
        raise ValueError(f"unsupported spatial overlay operation: {how}")
    requested = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    use_device = requested is not ExecutionMode.CPU and has_gpu_runtime()
    if requested is ExecutionMode.GPU and not use_device:
        raise RuntimeError("GPU spatial overlay requested without a CUDA runtime")

    if use_device:
        from vibespatial.runtime.residency import Residency, TransferTrigger

        for side, owned in (("left", left), ("right", right)):
            owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason=f"spatial overlay {side} native relation input",
            )

    from vibespatial.spatial.indexing import generate_bounds_pairs

    candidate_pairs = generate_bounds_pairs(
        left,
        right,
        requested_mode=ExecutionMode.GPU if use_device else ExecutionMode.CPU,
    )
    if use_device:
        relation = _device_relation_from_candidate_pairs(
            candidate_pairs,
            left_row_count=left.row_count,
            right_row_count=right.row_count,
        )
        result = _spatial_overlay_device(
            left,
            right,
            relation,
            how=how,
        )
        selected = ExecutionMode.GPU
        implementation = "native_relation_grouped_spatial_overlay"
        reason = (
            "candidate relation fed pair-capacity intersection and full-source "
            "grouped difference without host pair or geometry assembly"
        )
    else:
        from vibespatial.overlay.host_fallback import spatial_overlay_owned_host

        result = spatial_overlay_owned_host(
            left,
            right,
            candidate_pairs,
            how=how,
        )
        selected = ExecutionMode.CPU
        implementation = "explicit_cpu_planar_spatial_overlay"
        reason = "CPU execution was requested or no CUDA runtime was available"

    detail = (
        f"operation={how}, left_rows={left.row_count}, right_rows={right.row_count}, "
        f"candidate_pairs={candidate_pairs.count}, "
        "physical_shape=relation_pair_capacity+full_source_grouped"
    )
    record_dispatch_event(
        surface="geopandas.spatial_overlay",
        operation=how,
        implementation=implementation,
        reason=reason,
        detail=detail,
        requested=requested,
        selected=selected,
    )
    result.runtime_history.append(
        RuntimeSelection(
            requested=requested,
            selected=selected,
            reason=f"spatial_overlay {how}: {reason}",
        )
    )
    return result
