from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, OwnedGeometryArray, from_shapely_geometries
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_kernel_dispatch
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan

PointSequence = Sequence[object | None] | OwnedGeometryArray


@dataclass(frozen=True)
class PredicateKernelContext:
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan


def coerce_geometry_array(
    values: PointSequence,
    *,
    arg_name: str,
    expected_families: tuple[GeometryFamily, ...],
) -> OwnedGeometryArray:
    if isinstance(values, OwnedGeometryArray):
        array = values
    elif isinstance(values, Sequence):
        array = from_shapely_geometries(list(values))
    else:
        raise TypeError(f"{arg_name} must be an OwnedGeometryArray or a geometry sequence")

    unexpected = {family.value for family in array.families if family not in expected_families}
    if unexpected:
        joined = ", ".join(sorted(unexpected))
        raise ValueError(f"{arg_name} contains unsupported geometry families: {joined}")
    return array


def extract_point_coordinates(array: OwnedGeometryArray) -> tuple[np.ndarray, np.ndarray]:
    # Ensure host-side buffer data is materialized for device-resident arrays.
    # Device-resident OGAs from GPU I/O have x/y as empty stubs.
    if any(not buf.host_materialized for buf in array.families.values()):
        array._ensure_host_state()
    coords = np.full((array.row_count, 2), np.nan, dtype=np.float64)
    empty_mask = np.zeros(array.row_count, dtype=bool)
    buffer = array.families.get(GeometryFamily.POINT)
    if buffer is None:
        return coords, empty_mask

    point_rows = np.flatnonzero(array.tags == FAMILY_TAGS[GeometryFamily.POINT])
    if point_rows.size == 0:
        return coords, empty_mask

    family_rows = array.family_row_offsets[point_rows]
    row_empty = buffer.empty_mask[family_rows].astype(bool, copy=False)
    empty_mask[point_rows] = row_empty

    live = ~row_empty
    if live.any():
        live_point_rows = point_rows[live]
        live_family_rows = family_rows[live]
        starts = buffer.geometry_offsets[live_family_rows]
        coords[live_point_rows, 0] = buffer.x[starts]
        coords[live_point_rows, 1] = buffer.y[starts]
    return coords, empty_mask


def extract_empty_rows(array: OwnedGeometryArray) -> np.ndarray:
    # Ensure host-side buffer data is materialized for device-resident arrays.
    if any(not buf.host_materialized for buf in array.families.values()):
        array._ensure_host_state()
    empty_mask = np.zeros(array.row_count, dtype=bool)
    for family, buffer in array.families.items():
        family_rows = np.flatnonzero(array.tags == FAMILY_TAGS[family])
        if family_rows.size == 0:
            continue
        empty_mask[family_rows] = buffer.empty_mask
    return empty_mask


def resolve_predicate_context(
    *,
    kernel_name: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    dispatch_mode: ExecutionMode | str,
    precision: PrecisionMode | str,
) -> PredicateKernelContext:
    if left.row_count != right.row_count:
        raise ValueError(
            f"{kernel_name} requires aligned inputs; got {left.row_count} and {right.row_count} rows"
        )

    requested_mode = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    geometry_families = tuple(sorted({family.value for family in left.families | right.families}))
    plan = plan_kernel_dispatch(
        kernel_name=kernel_name,
        kernel_class=KernelClass.PREDICATE,
        row_count=left.row_count,
        geometry_families=geometry_families,
        mixed_geometry=len(geometry_families) > 1,
        current_residency=left.residency,
        requested_mode=requested_mode,
        requested_precision=precision,
    )

    runtime_selection = plan.runtime_selection
    if plan.variant is None:
        if requested_mode is ExecutionMode.GPU:
            raise NotImplementedError(f"{kernel_name} has no GPU variant registered yet")
        if runtime_selection.selected is ExecutionMode.GPU:
            runtime_selection = RuntimeSelection(
                requested=requested_mode,
                selected=ExecutionMode.CPU,
                reason=f"{kernel_name} has no GPU variant registered; using explicit CPU fallback",
            )

    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.PREDICATE,
        requested=precision,
    )
    robustness_plan = select_robustness_plan(
        kernel_class=KernelClass.PREDICATE,
        precision_plan=precision_plan,
    )
    left.record_runtime_selection(runtime_selection)
    if right is not left:
        right.record_runtime_selection(runtime_selection)
    return PredicateKernelContext(
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )
