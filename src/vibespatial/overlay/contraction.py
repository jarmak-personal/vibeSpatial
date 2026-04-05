from __future__ import annotations

from vibespatial.overlay.contract import contract_overlay_microcells
from vibespatial.overlay.contraction_reconstruct import (
    reconstruct_overlay_from_microcells as _reconstruct_overlay_from_microcells,
)
from vibespatial.overlay.microcells import (
    build_aligned_overlay_workload as _build_aligned_overlay_workload,
)
from vibespatial.overlay.microcells import (
    build_and_label_overlay_microcells,
    build_overlay_contraction_summary,
)
from vibespatial.runtime import ExecutionMode


def build_aligned_overlay_workload(left, right):
    return _build_aligned_overlay_workload(left, right)


def summarize_overlay_contraction_canary(
    left,
    right,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
):
    return build_overlay_contraction_summary(left, right, dispatch_mode=dispatch_mode).as_dict()


def reconstruct_overlay_from_microcells(
    labels,
    operation: str,
    *,
    row_count: int | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
):
    return _reconstruct_overlay_from_microcells(
        labels,
        operation,
        row_count=row_count,
        dispatch_mode=dispatch_mode,
    )


def overlay_contraction_owned(
    left,
    right,
    *,
    operation: str,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
):
    if left.row_count != right.row_count:
        raise ValueError("contraction overlay currently requires aligned left/right rows")

    labels = build_and_label_overlay_microcells(
        left,
        right,
        dispatch_mode=dispatch_mode,
        selection_operation=operation,
    )
    components = contract_overlay_microcells(labels)
    return _reconstruct_overlay_from_microcells(
        labels,
        operation,
        components=components,
        row_count=left.row_count,
        dispatch_mode=dispatch_mode,
    )
