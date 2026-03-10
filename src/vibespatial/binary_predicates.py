from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import shapely

from vibespatial.adaptive_runtime import plan_kernel_dispatch
from vibespatial.cuda_runtime import get_cuda_runtime
from vibespatial.dispatch import record_dispatch_event
from vibespatial.fallbacks import record_fallback_event
from vibespatial.geometry_buffers import GeometryFamily
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.point_binary_relations import (
    POINT_LOCATION_BOUNDARY,
    POINT_LOCATION_INTERIOR,
    POINT_LOCATION_OUTSIDE,
    classify_point_equals_gpu,
    classify_point_line_gpu,
    classify_point_region_gpu,
)
from vibespatial.owned_geometry import FAMILY_TAGS, OwnedGeometryArray, TAG_FAMILIES, from_shapely_geometries
from vibespatial.precision import KernelClass, PrecisionMode, PrecisionPlan, select_precision_plan
from vibespatial.residency import Residency, TransferTrigger
from vibespatial.robustness import RobustnessPlan, select_robustness_plan
from vibespatial.runtime import ExecutionMode, RuntimeSelection


PredicateInput = OwnedGeometryArray | Sequence[object | None] | np.ndarray


class NullBehavior(StrEnum):
    PROPAGATE = "propagate"
    FALSE = "false"


class CoarseRelation(StrEnum):
    INTERSECTS = "intersects"
    CONTAINS = "contains"
    WITHIN = "within"
    DISJOINT = "disjoint"


@dataclass(frozen=True)
class BinaryPredicateSpec:
    name: str
    coarse_relation: CoarseRelation
    shapely_op: str


@dataclass(frozen=True)
class BinaryPredicateResult:
    predicate: str
    values: np.ndarray
    row_count: int
    candidate_rows: np.ndarray
    coarse_true_rows: np.ndarray
    coarse_false_rows: np.ndarray
    runtime_selection: RuntimeSelection
    precision_plan: PrecisionPlan
    robustness_plan: RobustnessPlan


PREDICATE_SPECS: dict[str, BinaryPredicateSpec] = {
    "intersects": BinaryPredicateSpec("intersects", CoarseRelation.INTERSECTS, "intersects"),
    "within": BinaryPredicateSpec("within", CoarseRelation.WITHIN, "within"),
    "contains": BinaryPredicateSpec("contains", CoarseRelation.CONTAINS, "contains"),
    "touches": BinaryPredicateSpec("touches", CoarseRelation.INTERSECTS, "touches"),
    "covered_by": BinaryPredicateSpec("covered_by", CoarseRelation.WITHIN, "covered_by"),
    "covers": BinaryPredicateSpec("covers", CoarseRelation.CONTAINS, "covers"),
    "crosses": BinaryPredicateSpec("crosses", CoarseRelation.INTERSECTS, "crosses"),
    "contains_properly": BinaryPredicateSpec(
        "contains_properly",
        CoarseRelation.CONTAINS,
        "contains_properly",
    ),
    "overlaps": BinaryPredicateSpec("overlaps", CoarseRelation.INTERSECTS, "overlaps"),
    "disjoint": BinaryPredicateSpec("disjoint", CoarseRelation.DISJOINT, "disjoint"),
}

_LINE_FAMILIES = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_REGION_FAMILIES = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_POINT_TAG = FAMILY_TAGS[GeometryFamily.POINT]
_MP_TAG = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
_LINE_TAGS = tuple(FAMILY_TAGS[family] for family in _LINE_FAMILIES)
_REGION_TAGS = tuple(FAMILY_TAGS[family] for family in _REGION_FAMILIES)
_ALL_SUPPORTED_TAGS = (_POINT_TAG, _MP_TAG) + _LINE_TAGS + _REGION_TAGS


def supports_binary_predicate(name: str) -> bool:
    return name in PREDICATE_SPECS


def _coerce_array(
    values: PredicateInput,
    *,
    arg_name: str,
) -> tuple[np.ndarray | None, OwnedGeometryArray | None]:
    if isinstance(values, OwnedGeometryArray):
        return None, values
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            raise TypeError(f"{arg_name} must be a 1D geometry array or a scalar geometry")
        return np.asarray(values, dtype=object), None
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=object), None
    raise TypeError(f"{arg_name} must be an OwnedGeometryArray or 1D geometry sequence")


def _coerce_right(
    values: object | PredicateInput,
    *,
    expected_len: int,
) -> tuple[np.ndarray | object | None, bool, OwnedGeometryArray | None]:
    if isinstance(values, OwnedGeometryArray):
        if values.row_count != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {values.row_count} rows"
            )
        return None, False, values
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values.item(), True, None
        if len(values) != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {len(values)} rows"
            )
        return np.asarray(values, dtype=object), False, None
    if isinstance(values, (list, tuple)):
        if len(values) != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {len(values)} rows"
            )
        return np.asarray(values, dtype=object), False, None
    return values, True, None


def _ensure_registered_kernel(
    predicate: str,
    requested_mode: ExecutionMode,
    row_count: int,
) -> RuntimeSelection:
    plan = plan_kernel_dispatch(
        kernel_name=predicate,
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=requested_mode,
        requested_precision=PrecisionMode.AUTO,
    )
    selection = plan.runtime_selection
    if plan.variant is None:
        if requested_mode is ExecutionMode.GPU:
            raise NotImplementedError(f"{predicate} has no GPU variant registered yet")
        if selection.selected is ExecutionMode.GPU:
            return RuntimeSelection(
                requested=requested_mode,
                selected=ExecutionMode.CPU,
                reason=f"{predicate} has no GPU variant registered; using explicit CPU fallback",
            )
    return selection


def _record_runtime_selection(
    selection: RuntimeSelection,
    arrays: tuple[OwnedGeometryArray | None, ...],
) -> None:
    for array in arrays:
        if array is not None:
            array.record_runtime_selection(selection)


def _owned_from_values(
    values: np.ndarray | object | None,
    *,
    owned: OwnedGeometryArray | None,
    scalar: bool,
) -> OwnedGeometryArray | None:
    if scalar:
        return None
    if owned is not None:
        return owned
    assert isinstance(values, np.ndarray)
    return from_shapely_geometries(values.tolist())


def _materialize_shapely(values: np.ndarray | None, owned: OwnedGeometryArray | None) -> np.ndarray:
    if values is not None:
        return values
    assert owned is not None
    return np.asarray(owned.to_shapely(), dtype=object)


def _bounds_for(
    values: np.ndarray | object | None,
    *,
    owned: OwnedGeometryArray | None,
    dispatch_mode: ExecutionMode,
    size: int | None = None,
) -> np.ndarray:
    if owned is not None:
        bounds = compute_geometry_bounds(owned, dispatch_mode=dispatch_mode)
        if size is not None and bounds.shape[0] != size:
            raise ValueError(f"expected {size} bounds rows, got {bounds.shape[0]}")
        return bounds
    if size is None:
        return np.asarray(shapely.bounds(values), dtype=np.float64)
    scalar_bounds = np.asarray(shapely.bounds(values), dtype=np.float64)
    return np.broadcast_to(scalar_bounds, (size, 4)).copy()


def _bbox_intersects(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        (left[:, 0] <= right[:, 2])
        & (left[:, 2] >= right[:, 0])
        & (left[:, 1] <= right[:, 3])
        & (left[:, 3] >= right[:, 1])
    )


def _bbox_contains(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        (left[:, 0] <= right[:, 0])
        & (left[:, 1] <= right[:, 1])
        & (left[:, 2] >= right[:, 2])
        & (left[:, 3] >= right[:, 3])
    )


def _bbox_within(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return _bbox_contains(right, left)


def _coarse_candidate_mask(
    relation: CoarseRelation,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = ~(np.isnan(left_bounds).any(axis=1) | np.isnan(right_bounds).any(axis=1))
    if relation is CoarseRelation.INTERSECTS:
        coarse_true = np.zeros(left_bounds.shape[0], dtype=bool)
        candidate = valid & _bbox_intersects(left_bounds, right_bounds)
        return candidate, coarse_true, ~(candidate | ~valid)
    if relation is CoarseRelation.CONTAINS:
        coarse_true = np.zeros(left_bounds.shape[0], dtype=bool)
        candidate = valid & _bbox_contains(left_bounds, right_bounds)
        return candidate, coarse_true, ~(candidate | ~valid)
    if relation is CoarseRelation.WITHIN:
        coarse_true = np.zeros(left_bounds.shape[0], dtype=bool)
        candidate = valid & _bbox_within(left_bounds, right_bounds)
        return candidate, coarse_true, ~(candidate | ~valid)
    if relation is CoarseRelation.DISJOINT:
        bbox_intersects = valid & _bbox_intersects(left_bounds, right_bounds)
        coarse_true = valid & ~bbox_intersects
        candidate = bbox_intersects
        return candidate, coarse_true, ~(candidate | coarse_true | ~valid)
    raise ValueError(f"unsupported coarse relation: {relation}")


def _fill_output(
    size: int,
    *,
    null_behavior: NullBehavior,
    null_mask: np.ndarray,
) -> np.ndarray:
    if null_behavior is NullBehavior.FALSE:
        return np.zeros(size, dtype=bool)
    result = np.zeros(size, dtype=object)
    result[:] = False
    result[null_mask] = None
    return result


def _result_to_bool_array(values: np.ndarray | Sequence[bool], count: int) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.shape != (count,):
        return np.asarray(list(values), dtype=bool)
    return array


def _point_relation_to_predicate(
    predicate: str,
    relation: np.ndarray,
    *,
    point_on_left: bool,
) -> np.ndarray:
    outside = relation == POINT_LOCATION_OUTSIDE
    boundary = relation == POINT_LOCATION_BOUNDARY
    interior = relation == POINT_LOCATION_INTERIOR
    if predicate == "intersects":
        return ~outside
    if predicate == "disjoint":
        return outside
    if predicate == "touches":
        return boundary
    if predicate in {"crosses", "overlaps"}:
        return np.zeros(relation.shape[0], dtype=bool)
    if point_on_left:
        if predicate == "within":
            return interior
        if predicate == "covered_by":
            return ~outside
        return np.zeros(relation.shape[0], dtype=bool)
    if predicate == "contains":
        return interior
    if predicate == "covers":
        return ~outside
    if predicate == "contains_properly":
        return interior
    return np.zeros(relation.shape[0], dtype=bool)


def _point_equals_to_predicate(predicate: str, relation: np.ndarray) -> np.ndarray:
    equal = relation == POINT_LOCATION_INTERIOR
    if predicate in {"intersects", "contains", "within", "covers", "covered_by", "contains_properly"}:
        return equal
    if predicate == "disjoint":
        return ~equal
    return np.zeros(relation.shape[0], dtype=bool)


def _unsupported_gpu_reason(predicate: str, *, scalar_right: bool) -> str:
    if scalar_right:
        return f"{predicate} GPU refine does not support scalar right-hand geometries yet"
    return (
        f"{predicate} GPU refine currently supports only point-centric candidate rows "
        "(point/point, point/line, point/polygon, and inverses)"
    )


def _candidate_pairs_supported(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> bool:
    if candidate_rows.size == 0:
        return True
    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]
    left_is_point = (left_tags == _POINT_TAG) | (left_tags == _MP_TAG)
    right_is_point = (right_tags == _POINT_TAG) | (right_tags == _MP_TAG)
    return bool(
        np.all(
            (left_is_point & np.isin(right_tags, _ALL_SUPPORTED_TAGS))
            | (right_is_point & np.isin(left_tags, _ALL_SUPPORTED_TAGS))
        )
    )


def _apply_relation_rows(
    out: np.ndarray,
    row_ids: np.ndarray,
    predicate_values: np.ndarray,
) -> None:
    if row_ids.size:
        out[row_ids] = predicate_values.astype(bool, copy=False)


def _owned_empty_mask(values: OwnedGeometryArray) -> np.ndarray:
    empty = np.zeros(values.row_count, dtype=bool)
    if not values.validity.any():
        return empty
    valid_rows = np.flatnonzero(values.validity)
    valid_tags = values.tags[valid_rows]
    valid_offsets = values.family_row_offsets[valid_rows]
    for family, tag in FAMILY_TAGS.items():
        family_rows = valid_rows[valid_tags == tag]
        if family_rows.size == 0:
            continue
        offsets = valid_offsets[valid_tags == tag]
        empty[family_rows] = values.families[family].empty_mask[offsets]
    return empty


def _uniform_point_region_orientation(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> tuple[bool, GeometryFamily | None] | None:
    valid_left = left.tags[left.validity]
    valid_right = right.tags[right.validity]
    if valid_left.size == 0 or valid_right.size == 0:
        return None
    if np.all(valid_left == _POINT_TAG) and np.all(np.isin(valid_right, _REGION_TAGS)):
        region_tags = valid_right[np.isin(valid_right, _REGION_TAGS)]
        region_family = TAG_FAMILIES[int(region_tags[0])] if region_tags.size and np.all(region_tags == region_tags[0]) else None
        return True, region_family
    if np.all(valid_right == _POINT_TAG) and np.all(np.isin(valid_left, _REGION_TAGS)):
        region_tags = valid_left[np.isin(valid_left, _REGION_TAGS)]
        region_family = TAG_FAMILIES[int(region_tags[0])] if region_tags.size and np.all(region_tags == region_tags[0]) else None
        return False, region_family
    return None


def _evaluate_gpu_point_region_fast_path(
    predicate: str,
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    null_mask: np.ndarray,
    null_behavior: NullBehavior,
    runtime_selection: RuntimeSelection,
    precision: PrecisionMode | str,
) -> BinaryPredicateResult | None:
    orientation = _uniform_point_region_orientation(left, right)
    if orientation is None:
        return None

    point_on_left, single_region_family = orientation
    points = left if point_on_left else right
    regions = right if point_on_left else left

    points.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for point input",
    )
    regions.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for region input",
    )
    region_state = regions._ensure_device_state()
    if any(
        region_state.families[family].bounds is None
        for family in region_state.families
        if family in _REGION_FAMILIES
    ):
        compute_geometry_bounds(regions, dispatch_mode=ExecutionMode.GPU)

    from vibespatial.kernels.predicates.point_in_polygon import launch_point_region_candidate_rows

    runtime = get_cuda_runtime()
    candidate_result = launch_point_region_candidate_rows(points, regions)
    try:
        candidate_rows = runtime.copy_device_to_host(candidate_result.values).astype(np.int32, copy=False)
    finally:
        runtime.free(candidate_result.values)

    empty_mask = (~null_mask) & (_owned_empty_mask(left) | _owned_empty_mask(right))
    candidate_mask = np.zeros(left.row_count, dtype=bool)
    if candidate_rows.size:
        candidate_mask[candidate_rows] = True

    active_rows = ~(null_mask | empty_mask)
    if PREDICATE_SPECS[predicate].coarse_relation is CoarseRelation.DISJOINT:
        coarse_true_mask = active_rows & ~candidate_mask
        coarse_false_mask = np.zeros(left.row_count, dtype=bool)
    else:
        coarse_true_mask = np.zeros(left.row_count, dtype=bool)
        coarse_false_mask = active_rows & ~candidate_mask
    if empty_mask.any():
        if PREDICATE_SPECS[predicate].coarse_relation is CoarseRelation.DISJOINT:
            coarse_true_mask |= empty_mask
        else:
            coarse_false_mask |= empty_mask

    result = _fill_output(
        left.row_count,
        null_behavior=null_behavior,
        null_mask=null_mask,
    )
    if coarse_true_mask.any():
        result[coarse_true_mask] = True
    if null_mask.any() and null_behavior is NullBehavior.FALSE:
        result[null_mask] = False

    if candidate_rows.size:
        if single_region_family is None:
            exact_values = _evaluate_gpu_point_candidates(
                predicate,
                left,
                right,
                candidate_rows,
            )
        else:
            relation = classify_point_region_gpu(
                candidate_rows,
                points,
                regions,
                region_family=single_region_family,
            )
            exact_values = _point_relation_to_predicate(
                predicate,
                relation,
                point_on_left=point_on_left,
            )
        result[candidate_rows] = exact_values

    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.PREDICATE,
        requested=precision,
    )
    robustness_plan = select_robustness_plan(
        kernel_class=KernelClass.PREDICATE,
        precision_plan=precision_plan,
    )
    return BinaryPredicateResult(
        predicate=predicate,
        values=result,
        row_count=left.row_count,
        candidate_rows=candidate_rows,
        coarse_true_rows=np.flatnonzero(coarse_true_mask).astype(np.int32, copy=False),
        coarse_false_rows=np.flatnonzero(coarse_false_mask).astype(np.int32, copy=False),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def _evaluate_gpu_point_candidates(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    if candidate_rows.size == 0:
        return np.empty(0, dtype=bool)

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for left geometry input",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for right geometry input",
    )

    out = np.zeros(candidate_rows.size, dtype=bool)
    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]

    point_point_mask = (left_tags == _POINT_TAG) & (right_tags == _POINT_TAG)
    if point_point_mask.any():
        rows = candidate_rows[point_point_mask]
        relation = classify_point_equals_gpu(rows, left, right)
        _apply_relation_rows(out, np.flatnonzero(point_point_mask), _point_equals_to_predicate(predicate, relation))

    for line_family, line_tag in zip(_LINE_FAMILIES, _LINE_TAGS, strict=True):
        point_line_mask = (left_tags == _POINT_TAG) & (right_tags == line_tag)
        if point_line_mask.any():
            rows = candidate_rows[point_line_mask]
            relation = classify_point_line_gpu(rows, left, right, line_family=line_family)
            _apply_relation_rows(
                out,
                np.flatnonzero(point_line_mask),
                _point_relation_to_predicate(predicate, relation, point_on_left=True),
            )

        line_point_mask = (left_tags == line_tag) & (right_tags == _POINT_TAG)
        if line_point_mask.any():
            rows = candidate_rows[line_point_mask]
            relation = classify_point_line_gpu(rows, right, left, line_family=line_family)
            _apply_relation_rows(
                out,
                np.flatnonzero(line_point_mask),
                _point_relation_to_predicate(predicate, relation, point_on_left=False),
            )

    for region_family, region_tag in zip(_REGION_FAMILIES, _REGION_TAGS, strict=True):
        point_region_mask = (left_tags == _POINT_TAG) & (right_tags == region_tag)
        if point_region_mask.any():
            rows = candidate_rows[point_region_mask]
            relation = classify_point_region_gpu(rows, left, right, region_family=region_family)
            _apply_relation_rows(
                out,
                np.flatnonzero(point_region_mask),
                _point_relation_to_predicate(predicate, relation, point_on_left=True),
            )

        region_point_mask = (left_tags == region_tag) & (right_tags == _POINT_TAG)
        if region_point_mask.any():
            rows = candidate_rows[region_point_mask]
            relation = classify_point_region_gpu(rows, right, left, region_family=region_family)
            _apply_relation_rows(
                out,
                np.flatnonzero(region_point_mask),
                _point_relation_to_predicate(predicate, relation, point_on_left=False),
            )

    # Multipoint pairs — delegate to the indexed dispatch which handles all
    # multipoint × {point, line, region, multipoint} combinations.
    mp_mask = (left_tags == _MP_TAG) | (right_tags == _MP_TAG)
    if mp_mask.any():
        from vibespatial.point_binary_relations import classify_point_predicates_indexed
        mp_idx = np.flatnonzero(mp_mask)
        mp_rows = candidate_rows[mp_idx]
        mp_result = classify_point_predicates_indexed(
            predicate, left, right, mp_rows, mp_rows,
        )
        _apply_relation_rows(out, mp_idx, mp_result)

    return out


def evaluate_binary_predicate(
    predicate: str,
    left: PredicateInput,
    right: object | PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    null_behavior: NullBehavior | str = NullBehavior.PROPAGATE,
    **kwargs: Any,
) -> BinaryPredicateResult:
    try:
        spec = PREDICATE_SPECS[predicate]
    except KeyError as exc:
        raise ValueError(f"unsupported binary predicate: {predicate}") from exc

    left_values, left_owned = _coerce_array(left, arg_name="left")
    row_count = left_owned.row_count if left_owned is not None else len(left_values)
    right_values, scalar_right, right_owned = _coerce_right(right, expected_len=row_count)
    requested_mode = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    normalized_null_behavior = (
        null_behavior if isinstance(null_behavior, NullBehavior) else NullBehavior(null_behavior)
    )

    runtime_selection = _ensure_registered_kernel(predicate, requested_mode, row_count)
    if left_values is None:
        assert left_owned is not None
        left_missing = ~left_owned.validity
    else:
        left_missing = shapely.is_missing(left_values)
    if scalar_right:
        right_missing = np.full(row_count, bool(right_values is None), dtype=bool)
    elif right_values is None:
        assert right_owned is not None
        right_missing = ~right_owned.validity
    else:
        right_missing = shapely.is_missing(right_values)
    null_mask = left_missing | right_missing

    left_gpu_owned = _owned_from_values(left_values, owned=left_owned, scalar=False)
    right_gpu_owned = _owned_from_values(right_values, owned=right_owned, scalar=scalar_right)

    if (
        runtime_selection.selected is ExecutionMode.GPU
        and not scalar_right
        and left_gpu_owned is not None
        and right_gpu_owned is not None
    ):
        fast_path_result = _evaluate_gpu_point_region_fast_path(
            predicate,
            left=left_gpu_owned,
            right=right_gpu_owned,
            null_mask=null_mask,
            null_behavior=normalized_null_behavior,
            runtime_selection=runtime_selection,
            precision=precision,
        )
        if fast_path_result is not None:
            _record_runtime_selection(runtime_selection, (left_gpu_owned, right_gpu_owned))
            return fast_path_result

    gpu_dispatch_mode = ExecutionMode.GPU if runtime_selection.selected is ExecutionMode.GPU else ExecutionMode.CPU
    left_bounds = _bounds_for(
        left_values,
        owned=left_gpu_owned if gpu_dispatch_mode is ExecutionMode.GPU else left_owned,
        dispatch_mode=gpu_dispatch_mode,
    )
    right_bounds = _bounds_for(
        right_values,
        owned=right_gpu_owned if gpu_dispatch_mode is ExecutionMode.GPU else right_owned,
        dispatch_mode=gpu_dispatch_mode,
        size=row_count if scalar_right else None,
    )
    candidate_mask, coarse_true_mask, coarse_false_mask = _coarse_candidate_mask(
        spec.coarse_relation,
        left_bounds,
        right_bounds,
    )
    empty_mask = (~null_mask) & (np.isnan(left_bounds).any(axis=1) | np.isnan(right_bounds).any(axis=1))
    if empty_mask.any():
        candidate_mask = candidate_mask & ~empty_mask
        if spec.coarse_relation is CoarseRelation.DISJOINT:
            coarse_true_mask = coarse_true_mask | empty_mask
        else:
            coarse_false_mask = coarse_false_mask | empty_mask

    if runtime_selection.selected is ExecutionMode.GPU:
        gpu_reason = _unsupported_gpu_reason(predicate, scalar_right=scalar_right)
        if scalar_right or left_gpu_owned is None or right_gpu_owned is None or not _candidate_pairs_supported(
            left_gpu_owned,
            right_gpu_owned,
            np.flatnonzero(candidate_mask & ~null_mask).astype(np.int32, copy=False),
        ):
            if requested_mode is ExecutionMode.GPU:
                raise NotImplementedError(gpu_reason)
            runtime_selection = RuntimeSelection(
                requested=requested_mode,
                selected=ExecutionMode.CPU,
                reason=f"{gpu_reason}; using explicit CPU fallback",
            )

    _record_runtime_selection(runtime_selection, (left_gpu_owned or left_owned, right_gpu_owned or right_owned))
    precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.PREDICATE,
        requested=precision,
    )
    robustness_plan = select_robustness_plan(
        kernel_class=KernelClass.PREDICATE,
        precision_plan=precision_plan,
    )

    result = _fill_output(
        row_count,
        null_behavior=normalized_null_behavior,
        null_mask=null_mask,
    )
    if coarse_true_mask.any():
        result[coarse_true_mask] = True
    if null_mask.any() and normalized_null_behavior is NullBehavior.FALSE:
        result[null_mask] = False

    candidate_rows = np.flatnonzero(candidate_mask & ~null_mask).astype(np.int32, copy=False)
    if candidate_rows.size:
        if runtime_selection.selected is ExecutionMode.GPU:
            assert left_gpu_owned is not None
            assert right_gpu_owned is not None
            exact_values = _evaluate_gpu_point_candidates(
                predicate,
                left_gpu_owned,
                right_gpu_owned,
                candidate_rows,
            )
            result[candidate_rows] = exact_values
        elif scalar_right:
            left_shapely = _materialize_shapely(left_values, left_owned)
            exact_values = getattr(shapely, spec.shapely_op)(left_shapely[candidate_rows], right_values, **kwargs)
            result[candidate_rows] = _result_to_bool_array(exact_values, candidate_rows.size)
        else:
            left_shapely = _materialize_shapely(left_values, left_owned)
            right_shapely = _materialize_shapely(right_values, right_owned)
            exact_values = getattr(shapely, spec.shapely_op)(
                left_shapely[candidate_rows],
                right_shapely[candidate_rows],
                **kwargs,
            )
            result[candidate_rows] = _result_to_bool_array(exact_values, candidate_rows.size)

    return BinaryPredicateResult(
        predicate=predicate,
        values=result,
        row_count=row_count,
        candidate_rows=candidate_rows,
        coarse_true_rows=np.flatnonzero(coarse_true_mask).astype(np.int32, copy=False),
        coarse_false_rows=np.flatnonzero(coarse_false_mask).astype(np.int32, copy=False),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def evaluate_geopandas_binary_predicate(
    predicate: str,
    left: np.ndarray,
    right: object | np.ndarray,
    **kwargs: Any,
) -> np.ndarray | None:
    if not supports_binary_predicate(predicate):
        record_fallback_event(
            surface=f"geopandas.array.{predicate}",
            reason="predicate is not wired to a repo-owned kernel; using host Shapely path",
            detail="unsupported by repo-owned exact predicate engine",
        )
        return None
    result = evaluate_binary_predicate(
        predicate,
        np.asarray(left, dtype=object),
        right if np.isscalar(right) or right is None else np.asarray(right, dtype=object),
        dispatch_mode=ExecutionMode.AUTO,
        null_behavior=NullBehavior.FALSE,
        **kwargs,
    )
    implementation = (
        "owned_gpu_predicate"
        if result.runtime_selection.selected is ExecutionMode.GPU
        else "owned_cpu_predicate"
    )
    reason = (
        "repo-owned binary predicate engine claimed the GeoPandas surface on GPU"
        if result.runtime_selection.selected is ExecutionMode.GPU
        else "repo-owned binary predicate engine claimed the GeoPandas surface on CPU"
    )
    record_dispatch_event(
        surface=f"geopandas.array.{predicate}",
        operation=predicate,
        implementation=implementation,
        reason=reason,
        detail=(
            f"rows={result.row_count}, candidate_rows={int(result.candidate_rows.size)}, "
            f"selected={result.runtime_selection.selected.value}"
        ),
        requested=result.runtime_selection.requested,
        selected=result.runtime_selection.selected,
    )
    return np.asarray(result.values, dtype=bool)


def benchmark_binary_predicate(
    predicate: str,
    left: PredicateInput,
    right: object | PredicateInput,
    **kwargs: Any,
) -> dict[str, int]:
    result = evaluate_binary_predicate(predicate, left, right, null_behavior=NullBehavior.FALSE, **kwargs)
    return {
        "rows": result.row_count,
        "candidate_rows": int(result.candidate_rows.size),
        "coarse_true_rows": int(result.coarse_true_rows.size),
        "coarse_false_rows": int(result.coarse_false_rows.size),
    }
