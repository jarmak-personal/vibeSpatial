from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import shapely

from vibespatial.cuda._runtime import get_cuda_runtime
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    DeviceMetadataState,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    from_shapely_geometries,
    unique_tag_pairs,
)
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_kernel_dispatch
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.residency import Residency, TransferTrigger
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan
from vibespatial.runtime.workload import WorkloadShape

from .point_relations import (
    POINT_LOCATION_BOUNDARY,
    POINT_LOCATION_INTERIOR,
    POINT_LOCATION_OUTSIDE,
    classify_point_equals_gpu,
    classify_point_line_gpu,
    classify_point_region_gpu,
)

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
    "equals": BinaryPredicateSpec("equals", CoarseRelation.INTERSECTS, "equals"),
}

_LINE_FAMILIES = (GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING)
_REGION_FAMILIES = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
_POINT_TAG = FAMILY_TAGS[GeometryFamily.POINT]
_MP_TAG = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
_LINE_TAGS = tuple(FAMILY_TAGS[family] for family in _LINE_FAMILIES)
_REGION_TAGS = tuple(FAMILY_TAGS[family] for family in _REGION_FAMILIES)
_ALL_SUPPORTED_TAGS = (_POINT_TAG, _MP_TAG) + _LINE_TAGS + _REGION_TAGS
# Tags eligible for GPU DE-9IM refinement (all non-point geometry families).
_DE9IM_TAGS = _LINE_TAGS + _REGION_TAGS
# Predicates that can be evaluated from DE-9IM bitmasks.
_DE9IM_PREDICATES = frozenset({
    "intersects", "contains", "within", "touches",
    "covers", "covered_by", "overlaps", "disjoint",
    "contains_properly",
})


_SPECIAL_PREDICATES = frozenset({"equals", "equals_exact", "equals_identical"})


def supports_binary_predicate(name: str) -> bool:
    return name in PREDICATE_SPECS or name in _SPECIAL_PREDICATES


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
) -> tuple[np.ndarray | object | None, bool, OwnedGeometryArray | None, WorkloadShape]:
    if isinstance(values, OwnedGeometryArray):
        if values.row_count == 1 and expected_len > 1:
            return None, False, values, WorkloadShape.BROADCAST_RIGHT
        if values.row_count != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {values.row_count} rows"
            )
        return None, False, values, WorkloadShape.PAIRWISE
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return values.item(), True, None, WorkloadShape.SCALAR_RIGHT
        if len(values) == 1 and expected_len > 1:
            owned = from_shapely_geometries(list(values))
            return None, False, owned, WorkloadShape.BROADCAST_RIGHT
        if len(values) != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {len(values)} rows"
            )
        return np.asarray(values, dtype=object), False, None, WorkloadShape.PAIRWISE
    if isinstance(values, (list, tuple)):
        if len(values) == 1 and expected_len > 1:
            owned = from_shapely_geometries(list(values))
            return None, False, owned, WorkloadShape.BROADCAST_RIGHT
        if len(values) != expected_len:
            raise ValueError(
                f"binary predicate inputs must be aligned; got {expected_len} and {len(values)} rows"
            )
        return np.asarray(values, dtype=object), False, None, WorkloadShape.PAIRWISE
    return values, True, None, WorkloadShape.SCALAR_RIGHT


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


def _broadcast_right_owned(
    right_1row: OwnedGeometryArray,
    n: int,
) -> OwnedGeometryArray:
    """Build an N-row OwnedGeometryArray that broadcasts a single right geometry.

    The returned array has N-length ``validity``, ``tags``, and
    ``family_row_offsets`` arrays (all constant, pointing to row 0 of the
    single right geometry).  The underlying family coordinate buffers are
    *shared* with *right_1row* -- no duplication of geometry data.

    Memory cost: 6 bytes/row (1B bool + 1B int8 + 4B int32).  At 1M rows
    this is 6 MB -- negligible compared to N-copying the geometry.

    These synthetic arrays are NOT cached on the OwnedGeometryArray because
    the broadcast length *n* depends on the left array and may differ
    between calls.
    """
    assert right_1row.row_count == 1
    src_validity = right_1row.validity
    src_tags = right_1row.tags
    src_offsets = right_1row.family_row_offsets

    validity = np.full(n, src_validity[0], dtype=bool)
    tags = np.full(n, src_tags[0], dtype=np.int8)
    family_row_offsets = np.full(n, src_offsets[0], dtype=np.int32)

    # When the source is device-resident, its host family buffers may be
    # un-materialised stubs (empty x/y with host_materialized=False).
    # We must NOT claim DEVICE residency without a device_state --
    # _ensure_device_state() would re-upload the empty stubs as real data,
    # causing CUDA_ERROR_ILLEGAL_ADDRESS.
    #
    # Instead, share the source's device-side family buffers and upload only
    # the small new metadata arrays (6 bytes/row H->D).
    if right_1row.device_state is not None:
        runtime = get_cuda_runtime()
        d_validity = runtime.from_host(validity)
        d_tags = runtime.from_host(tags)
        d_fro = runtime.from_host(family_row_offsets)
        d_meta = DeviceMetadataState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_fro,
        )
        d_state = OwnedGeometryDeviceState(
            validity=d_validity,
            tags=d_tags,
            family_row_offsets=d_fro,
            families=dict(right_1row.device_state.families),
        )
        return OwnedGeometryArray(
            validity=validity,
            tags=tags,
            family_row_offsets=family_row_offsets,
            families=right_1row.families,
            residency=Residency.DEVICE,
            device_state=d_state,
            device_metadata=d_meta,
        )

    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families=right_1row.families,
        residency=Residency.HOST,
    )


def _materialize_shapely(values: np.ndarray | None, owned: OwnedGeometryArray | None) -> np.ndarray:
    if values is not None:
        return values
    assert owned is not None
    return np.asarray(owned.to_shapely(), dtype=object)



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
    if predicate in {"intersects", "contains", "within", "covers", "covered_by", "contains_properly", "equals"}:
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


def _de9im_candidate_pairs_supported(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
    predicate: str,
) -> bool:
    """Check if non-point candidate pairs can use the GPU DE-9IM kernel."""
    if candidate_rows.size == 0:
        return True
    if predicate not in _DE9IM_PREDICATES:
        return False
    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]
    return bool(
        np.all(np.isin(left_tags, _DE9IM_TAGS))
        and np.all(np.isin(right_tags, _DE9IM_TAGS))
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
        from .point_relations import classify_point_predicates_indexed
        mp_idx = np.flatnonzero(mp_mask)
        mp_rows = candidate_rows[mp_idx]
        mp_result = classify_point_predicates_indexed(
            predicate, left, right, mp_rows, mp_rows,
        )
        _apply_relation_rows(out, mp_idx, mp_result)

    return out


def _evaluate_gpu_de9im_candidates(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> np.ndarray:
    """Evaluate non-point candidate pairs via the GPU DE-9IM kernel.

    Supports all combinations of {LINESTRING, MULTILINESTRING, POLYGON,
    MULTIPOLYGON}.  Groups candidates by (left_family, right_family) tag
    pair, dispatches compute_polygon_de9im_gpu per group, then evaluates
    the predicate from the resulting DE-9IM bitmasks.
    """
    if candidate_rows.size == 0:
        return np.empty(0, dtype=bool)

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} DE-9IM GPU execution for left geometry input",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} DE-9IM GPU execution for right geometry input",
    )

    from .polygon import (
        compute_polygon_de9im_gpu,
        evaluate_predicate_from_de9im,
    )

    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]
    de9im_masks = np.zeros(candidate_rows.size, dtype=np.uint16)

    # Upload candidate indices to device once — avoids per-group H2D in
    # compute_polygon_de9im_gpu (passes through d_left/d_right).
    import cupy as cp
    d_candidate_rows = cp.asarray(candidate_rows)

    # Group by (left_family, right_family) and dispatch the correct kernel.
    for lt, rt in unique_tag_pairs(left_tags, right_tags):
        sub_mask = (left_tags == lt) & (right_tags == rt)
        sub_idx = np.flatnonzero(sub_mask)
        if sub_idx.size == 0:
            continue
        lf = TAG_FAMILIES[lt] if lt in TAG_FAMILIES else None
        rf = TAG_FAMILIES[rt] if rt in TAG_FAMILIES else None
        if lf is None or rf is None:
            continue
        d_sub = d_candidate_rows[sub_idx]
        sub_result = compute_polygon_de9im_gpu(
            left, right,
            candidate_rows[sub_idx], candidate_rows[sub_idx],
            query_family=lf, tree_family=rf,
            d_left=d_sub, d_right=d_sub,
        )
        if sub_result is not None:
            de9im_masks[sub_idx] = sub_result

    return evaluate_predicate_from_de9im(de9im_masks, predicate)


def _evaluate_de9im_device(d_masks: object, predicate: str) -> object:
    """Evaluate a spatial predicate from DE-9IM bitmasks on device.

    CuPy-native mirror of ``polygon.evaluate_predicate_from_de9im`` that
    keeps the entire evaluation on device (Tier 2 — CuPy element-wise).

    Parameters
    ----------
    d_masks : cupy uint16 array of DE-9IM bitmasks (device-resident)
    predicate : one of the supported predicate names

    Returns
    -------
    cupy bool array (device-resident)
    """
    import cupy as cp

    from .polygon import (
        _CONTACT_MASK,
        _PREDICATE_RULES,
        DE9IM_BB,
        DE9IM_BE,
        DE9IM_BI,
        DE9IM_EB,
        DE9IM_EI,
        DE9IM_IB,
        DE9IM_IE,
        DE9IM_II,
    )

    m = d_masks.astype(cp.uint16, copy=False)

    if predicate == "intersects":
        return (m & _CONTACT_MASK).astype(bool)

    if predicate == "touches":
        has_contact = (m & (DE9IM_IB | DE9IM_BI | DE9IM_BB)).astype(bool)
        no_ii = ~(m & DE9IM_II).astype(bool)
        return has_contact & no_ii

    if predicate == "covers":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_EI | DE9IM_EB)).astype(bool)
        return has_contact & no_ext

    if predicate == "covered_by":
        has_contact = (m & _CONTACT_MASK).astype(bool)
        no_ext = ~(m & (DE9IM_IE | DE9IM_BE)).astype(bool)
        return has_contact & no_ext

    rule = _PREDICATE_RULES.get(predicate)
    if rule is None:
        raise ValueError(f"Unsupported predicate for DE-9IM evaluation: {predicate}")

    required_set, required_unset = rule
    result = cp.ones(len(m), dtype=bool)
    if required_set:
        result &= (m & required_set) == required_set
    if required_unset:
        result &= (m & required_unset) == 0
    return result


def _fused_gpu_binary_predicate(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray | None:
    """Fused device-resident pipeline: bounds + coarse filter + DE-9IM.

    Keeps all N-sized intermediaries on device (bounds, validity, bbox
    mask, candidate indices).  Only downloads the small candidate-sized
    result at the end.  Returns None if the fused path is not applicable
    (non-GPU, mixed point/non-point, unsupported predicate).
    """
    if predicate not in _DE9IM_PREDICATES:
        return None
    from vibespatial.runtime import has_gpu_runtime
    if not has_gpu_runtime():
        return None

    # Ensure both arrays are on device with bounds computed.
    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"fused GPU {predicate}: left geometry",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"fused GPU {predicate}: right geometry",
    )

    # Compute bounds on device if not cached.
    # compute_geometry_bounds(dispatch_mode=GPU) may silently fall back to
    # CPU (its GPU path has a bare ``except Exception: pass``).  When that
    # happens, the returned numpy bounds are correct but state.row_bounds
    # remains None.  Recover by uploading the CPU-computed bounds to device
    # so the rest of the fused pipeline can proceed without bailing out.
    import cupy as cp

    for arr, state_ref in ((left, "left"), (right, "right")):
        state = arr._ensure_device_state()
        if state.row_bounds is None:
            host_bounds = compute_geometry_bounds(arr, dispatch_mode=ExecutionMode.GPU)
            state = arr._ensure_device_state()
            if state.row_bounds is None:
                # GPU bounds failed; upload CPU-computed bounds to device.
                # The CPU fallback in compute_geometry_bounds may have moved
                # the array to HOST; restore DEVICE residency so the rest
                # of the fused pipeline can proceed.
                record_fallback_event(
                    surface="vibespatial.predicates.binary._fused_gpu_binary_predicate",
                    reason=f"GPU bounds kernel failed for {state_ref}; uploading CPU bounds to device",
                    d2h_transfer=False,
                )
                arr.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason=f"fused GPU {predicate}: restore {state_ref} residency after bounds fallback",
                )
                state = arr._ensure_device_state()
                state.row_bounds = cp.asarray(
                    host_bounds.reshape(arr.row_count, 4),
                    dtype=np.float64,
                )
    left_state = left._ensure_device_state()
    right_state = right._ensure_device_state()

    if left_state.row_bounds is None or right_state.row_bounds is None:
        return None

    n = left.row_count
    # All N-sized work stays on device (Tier 2 — CuPy element-wise).
    d_lb = cp.asarray(left_state.row_bounds).reshape(n, 4)
    d_rb = cp.asarray(right_state.row_bounds).reshape(n, 4)
    d_val_l = cp.asarray(left_state.validity)
    d_val_r = cp.asarray(right_state.validity)

    # Coarse bbox filter on device.
    d_bbox_hit = (
        (d_lb[:, 0] <= d_rb[:, 2])
        & (d_lb[:, 2] >= d_rb[:, 0])
        & (d_lb[:, 1] <= d_rb[:, 3])
        & (d_lb[:, 3] >= d_rb[:, 1])
    )
    d_valid = d_val_l & d_val_r
    d_cand_mask = d_bbox_hit & d_valid

    # Extract candidate indices on device.
    d_cand_rows = cp.flatnonzero(d_cand_mask).astype(cp.int32)
    if d_cand_rows.size == 0:
        return np.zeros(n, dtype=bool)

    # Check all candidates are DE-9IM eligible (no points).
    d_tags_l = cp.asarray(left_state.tags)
    d_tags_r = cp.asarray(right_state.tags)
    d_cand_ltags = d_tags_l[d_cand_rows]
    d_cand_rtags = d_tags_r[d_cand_rows]
    pt = np.int8(FAMILY_TAGS[GeometryFamily.POINT])
    mpt = np.int8(FAMILY_TAGS[GeometryFamily.MULTIPOINT])
    has_points = bool(
        cp.any((d_cand_ltags == pt) | (d_cand_ltags == mpt))
        or cp.any((d_cand_rtags == pt) | (d_cand_rtags == mpt))
    )
    if has_points:
        return None  # Fall back to the partitioned dispatch.

    # DE-9IM kernel — pass device indices to avoid H2D re-upload.
    from .polygon import compute_polygon_de9im_gpu

    cand_count = int(d_cand_rows.size)
    # DE-9IM mask accumulator lives on device (Tier 2 — CuPy element-wise).
    d_de9im_masks = cp.zeros(cand_count, dtype=cp.uint16)

    # Group by (left_family, right_family) tag pair — unique extraction
    # and sub-masking stay on device; dispatch uses device arrays directly.
    d_group_refs: list[tuple] = []
    for lt, rt in unique_tag_pairs(d_cand_ltags, d_cand_rtags):
        lf = TAG_FAMILIES[lt] if lt in TAG_FAMILIES else None
        rf = TAG_FAMILIES[rt] if rt in TAG_FAMILIES else None
        if lf is None or rf is None:
            continue
        d_sub_mask = (d_cand_ltags == lt) & (d_cand_rtags == rt)
        d_sub_idx = cp.flatnonzero(d_sub_mask)
        if d_sub_idx.size == 0:
            continue
        d_sub_cand = d_cand_rows[d_sub_idx]
        d_group_refs.append((lf, rf, d_sub_idx, d_sub_cand))

    # Dispatch per-group DE-9IM kernels.  Indices stay on device;
    # compute_polygon_de9im_gpu returns device arrays directly via
    # return_device=True — no D->H->D ping-pong per group.
    for lf, rf, d_sub_idx, d_sub_cand in d_group_refs:
        d_sub_result = compute_polygon_de9im_gpu(
            left, right,
            query_family=lf, tree_family=rf,
            d_left=d_sub_cand, d_right=d_sub_cand,
            return_device=True,
        )
        if d_sub_result is not None:
            # Device-resident scatter — no host round-trip.
            d_de9im_masks[d_sub_idx] = d_sub_result

    # Evaluate predicate entirely on device (CuPy bitwise ops).
    d_cand_result = _evaluate_de9im_device(d_de9im_masks, predicate)

    # Scatter candidate results into full-size output.
    # For DISJOINT: non-candidates (no bbox overlap) are True (definitely
    # disjoint).  Candidates get their exact DE-9IM result.
    is_disjoint = predicate == "disjoint"
    out = np.ones(n, dtype=bool) if is_disjoint else np.zeros(n, dtype=bool)
    out[cp.asnumpy(d_cand_rows)] = cp.asnumpy(d_cand_result)
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
    right_values, scalar_right, right_owned, workload_shape = _coerce_right(right, expected_len=row_count)
    requested_mode = dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    normalized_null_behavior = (
        null_behavior if isinstance(null_behavior, NullBehavior) else NullBehavior(null_behavior)
    )

    # --- Scalar-right promotion to broadcast-right ---
    # When the right operand is a scalar geometry (not wrapped in an array),
    # wrap it into a 1-row OwnedGeometryArray so the GPU path can handle it
    # via synthetic indirection arrays.  A None scalar becomes a null-validity
    # 1-row owned array.
    _original_scalar_value = right_values  # stash for CPU fallback path
    if workload_shape is WorkloadShape.SCALAR_RIGHT:
        # Wrap the scalar (which may be None) into a 1-row owned array
        right_owned = from_shapely_geometries([right_values])
        right_values = None

    # For broadcast-right, the right_owned is a 1-row array.  We will build
    # the broadcast N-row view lazily when the GPU path needs it.
    _is_broadcast = workload_shape in (WorkloadShape.BROADCAST_RIGHT, WorkloadShape.SCALAR_RIGHT)

    runtime_selection = _ensure_registered_kernel(predicate, requested_mode, row_count)
    if left_values is None:
        assert left_owned is not None
        left_missing = ~left_owned.validity
    else:
        left_missing = shapely.is_missing(left_values)
    if _is_broadcast:
        # For broadcast/scalar-right, null status is uniform from the single row
        assert right_owned is not None
        right_is_null = not right_owned.validity[0]
        right_missing = np.full(row_count, right_is_null, dtype=bool)
    elif right_values is None:
        assert right_owned is not None
        right_missing = ~right_owned.validity
    else:
        right_missing = shapely.is_missing(right_values)
    null_mask = left_missing | right_missing

    # --- GPU point-region fast path (needs owned arrays) ---
    # Defer OwnedGeometryArray conversion: only create when GPU refine
    # will actually use them.  For the common CPU-fallback case (polygon
    # vs polygon), we skip the expensive from_shapely_geometries entirely
    # and use shapely.bounds() for coarse filtering instead.
    left_gpu_owned: OwnedGeometryArray | None = left_owned
    right_gpu_owned: OwnedGeometryArray | None = None

    if runtime_selection.selected is ExecutionMode.GPU:
        # Point-region fast path requires owned arrays -- build them now
        # only for this check.
        left_gpu_owned = _owned_from_values(left_values, owned=left_owned, scalar=False)
        if _is_broadcast and right_owned is not None:
            # Build broadcast N-row view from the 1-row right owned array
            right_gpu_owned = _broadcast_right_owned(right_owned, row_count)
        elif not scalar_right:
            right_gpu_owned = _owned_from_values(right_values, owned=right_owned, scalar=False)

        if left_gpu_owned is not None and right_gpu_owned is not None:
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
                record_dispatch_event(
                    surface="vibespatial.predicates.binary",
                    operation=predicate,
                    requested=requested_mode,
                    selected=runtime_selection.selected,
                    implementation="gpu_point_region_fast_path",
                    reason=runtime_selection.reason,
                    detail=f"workload_shape={workload_shape.value}",
                )
                _record_runtime_selection(runtime_selection, (left_gpu_owned, right_gpu_owned))
                return fast_path_result

            # --- Fused device-resident pipeline for non-point pairs ---
            # Keeps bounds, coarse filter, and candidate extraction on device.
            # Only downloads the small candidate-count result at the end.
            if not null_mask.any():
                fused = _fused_gpu_binary_predicate(predicate, left_gpu_owned, right_gpu_owned)
                if fused is not None:
                    precision_plan = select_precision_plan(
                        runtime_selection=runtime_selection,
                        kernel_class=KernelClass.PREDICATE,
                        requested=precision,
                    )
                    robustness_plan = select_robustness_plan(
                        kernel_class=KernelClass.PREDICATE,
                        precision_plan=precision_plan,
                    )
                    record_dispatch_event(
                        surface="vibespatial.predicates.binary",
                        operation=predicate,
                        requested=requested_mode,
                        selected=runtime_selection.selected,
                        implementation="fused_gpu_binary_predicate",
                        reason=runtime_selection.reason,
                        detail=f"workload_shape={workload_shape.value}",
                    )
                    _record_runtime_selection(runtime_selection, (left_gpu_owned, right_gpu_owned))
                    result_values = np.empty(row_count, dtype=object)
                    result_values[:] = fused
                    return BinaryPredicateResult(
                        predicate=predicate,
                        values=result_values,
                        row_count=row_count,
                        candidate_rows=np.flatnonzero(fused).astype(np.int32, copy=False),
                        coarse_true_rows=np.empty(0, dtype=np.int32),
                        coarse_false_rows=np.empty(0, dtype=np.int32),
                        runtime_selection=runtime_selection,
                        precision_plan=precision_plan,
                        robustness_plan=robustness_plan,
                    )

    # --- Bounds computation ---
    # Prefer shapely.bounds() (~1ms vectorized C) over compute_geometry_bounds(owned)
    # when raw numpy values are available.  This avoids creating OwnedGeometryArray
    # just to compute bounds.
    if left_values is not None:
        left_bounds = np.asarray(shapely.bounds(left_values), dtype=np.float64)
    elif left_gpu_owned is not None:
        gpu_dispatch_mode = ExecutionMode.GPU if runtime_selection.selected is ExecutionMode.GPU else ExecutionMode.CPU
        left_bounds = compute_geometry_bounds(left_gpu_owned, dispatch_mode=gpu_dispatch_mode)
    else:
        assert left_owned is not None
        left_bounds = compute_geometry_bounds(left_owned, dispatch_mode=ExecutionMode.CPU)

    if _is_broadcast and right_owned is not None:
        # Broadcast: compute bounds from the 1-row right, then broadcast to N rows
        broadcast_bounds = compute_geometry_bounds(right_owned, dispatch_mode=ExecutionMode.CPU)
        right_bounds = np.broadcast_to(broadcast_bounds, (row_count, 4)).copy()
    elif right_values is not None:
        right_bounds = np.asarray(shapely.bounds(right_values), dtype=np.float64)
    elif right_gpu_owned is not None:
        gpu_dispatch_mode = ExecutionMode.GPU if runtime_selection.selected is ExecutionMode.GPU else ExecutionMode.CPU
        right_bounds = compute_geometry_bounds(right_gpu_owned, dispatch_mode=gpu_dispatch_mode)
    else:
        assert right_owned is not None
        right_bounds = compute_geometry_bounds(right_owned, dispatch_mode=ExecutionMode.CPU)

    candidate_mask, coarse_true_mask, coarse_false_mask = _coarse_candidate_mask(
        spec.coarse_relation,
        left_bounds,
        right_bounds,
    )
    # Fast pre-check: skip the expensive np.isnan scan when owned arrays
    # report no empty geometries (the common case for generated/clean data).
    _may_have_empties = True
    if left_owned is not None and right_owned is not None:
        _may_have_empties = any(
            getattr(buf, "empty_mask", None) is not None and buf.empty_mask.any()
            for owned in (left_owned, right_owned)
            for buf in owned.families.values()
        )
    if _may_have_empties:
        empty_mask = (~null_mask) & (np.isnan(left_bounds).any(axis=1) | np.isnan(right_bounds).any(axis=1))
    else:
        empty_mask = np.zeros(row_count, dtype=bool)
    if empty_mask.any():
        candidate_mask = candidate_mask & ~empty_mask
        if spec.coarse_relation is CoarseRelation.DISJOINT:
            coarse_true_mask = coarse_true_mask | empty_mask
        else:
            coarse_false_mask = coarse_false_mask | empty_mask

    # --- GPU refine viability check ---
    # Only build owned arrays and check candidate-pair support if GPU is
    # selected.  For broadcast/scalar-right, the broadcast owned array was
    # already built above.
    if runtime_selection.selected is ExecutionMode.GPU:
        gpu_reason = _unsupported_gpu_reason(predicate, scalar_right=False)
        # Ensure owned arrays exist for candidate-pair check
        if left_gpu_owned is None or left_gpu_owned is left_owned:
            left_gpu_owned = _owned_from_values(left_values, owned=left_owned, scalar=False)
        if right_gpu_owned is None:
            if _is_broadcast and right_owned is not None:
                right_gpu_owned = _broadcast_right_owned(right_owned, row_count)
            elif not scalar_right:
                right_gpu_owned = _owned_from_values(right_values, owned=right_owned, scalar=False)

        if left_gpu_owned is not None and right_gpu_owned is not None:
            _cand = np.flatnonzero(candidate_mask & ~null_mask).astype(np.int32, copy=False)
            _point_ok = _candidate_pairs_supported(left_gpu_owned, right_gpu_owned, _cand)
            _de9im_ok = (
                not _point_ok
                and _de9im_candidate_pairs_supported(left_gpu_owned, right_gpu_owned, _cand, predicate)
            )
            if not _point_ok and not _de9im_ok:
                if requested_mode is ExecutionMode.GPU:
                    raise NotImplementedError(gpu_reason)
                runtime_selection = RuntimeSelection(
                    requested=requested_mode,
                    selected=ExecutionMode.CPU,
                    reason=f"{gpu_reason}; using explicit CPU fallback",
                )
        else:
            # Cannot build owned arrays -- fall back to CPU
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
            # Route point-centric pairs through the point kernel, non-point
            # pairs through the DE-9IM kernel.  For element-wise binary
            # predicates the pairs are typically homogeneous, but we handle
            # mixed cases by partitioning.
            left_cand_tags = left_gpu_owned.tags[candidate_rows]
            right_cand_tags = right_gpu_owned.tags[candidate_rows]
            left_is_point = (left_cand_tags == _POINT_TAG) | (left_cand_tags == _MP_TAG)
            right_is_point = (right_cand_tags == _POINT_TAG) | (right_cand_tags == _MP_TAG)
            point_mask = left_is_point | right_is_point
            de9im_mask = ~point_mask

            if point_mask.any():
                point_idx = np.flatnonzero(point_mask)
                point_rows = candidate_rows[point_idx]
                point_values = _evaluate_gpu_point_candidates(
                    predicate, left_gpu_owned, right_gpu_owned, point_rows,
                )
                result[point_rows] = point_values

            if de9im_mask.any():
                de9im_idx = np.flatnonzero(de9im_mask)
                de9im_rows = candidate_rows[de9im_idx]
                de9im_values = _evaluate_gpu_de9im_candidates(
                    predicate, left_gpu_owned, right_gpu_owned, de9im_rows,
                )
                result[de9im_rows] = de9im_values
        elif _is_broadcast:
            # CPU fallback for scalar-right or broadcast-right: recover
            # the single right geometry and broadcast against left candidates.
            left_shapely = _materialize_shapely(left_values, left_owned)
            if _original_scalar_value is not None:
                scalar_geom = _original_scalar_value
            else:
                assert right_owned is not None
                scalar_geom = right_owned.to_shapely()[0]
            exact_values = getattr(shapely, spec.shapely_op)(left_shapely[candidate_rows], scalar_geom, **kwargs)
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

    record_dispatch_event(
        surface="vibespatial.predicates.binary",
        operation=predicate,
        requested=requested_mode,
        selected=runtime_selection.selected,
        implementation=(
            "gpu_binary_predicate" if runtime_selection.selected is ExecutionMode.GPU
            else "cpu_shapely_fallback"
        ),
        reason=runtime_selection.reason,
        detail=f"workload_shape={workload_shape.value}",
    )

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


def _evaluate_geopandas_equals(
    left: np.ndarray | OwnedGeometryArray,
    right: object | np.ndarray | OwnedGeometryArray,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch topological equals through the normalize-then-compare path.

    Topological equality = structural equality after normalization.  Both
    normalize and equals_exact already have GPU paths.  This function
    composes them: normalize both inputs, then compare with tolerance 1e-12.

    For scalar right operands, falls back to Shapely's vectorized C path
    to avoid O(N) Python-side geometry duplication.
    """
    from vibespatial.geometry.equality import geom_equals_owned
    from vibespatial.runtime import get_requested_mode

    # Scalar right: fall back to Shapely vectorized equals which
    # handles scalar broadcasting natively in C, avoiding O(N) Python-side
    # geometry duplication.
    is_scalar = not isinstance(right, (OwnedGeometryArray, np.ndarray, list, tuple))
    if is_scalar:
        left_shapely = (
            np.asarray(left.to_shapely(), dtype=object)
            if isinstance(left, OwnedGeometryArray)
            else np.asarray(left, dtype=object)
        )
        result = shapely.equals(left_shapely, right)
        record_dispatch_event(
            surface="geopandas.array.equals",
            operation="equals",
            implementation="shapely_scalar_broadcast",
            reason="scalar right-hand operand; Shapely vectorized C path",
            detail=f"rows={len(left_shapely)}",
            selected=ExecutionMode.CPU,
        )
        record_fallback_event(
            surface="geopandas.array.equals",
            reason="scalar right-hand operand requires Shapely broadcast",
            detail=f"rows={len(left_shapely)}",
            pipeline="predicate",
            d2h_transfer=isinstance(left, OwnedGeometryArray),
        )
        return result.astype(bool, copy=False)

    # Coerce inputs to OwnedGeometryArray.
    if isinstance(left, OwnedGeometryArray):
        left_owned = left
    else:
        left_owned = from_shapely_geometries(list(left) if not isinstance(left, list) else left)

    if isinstance(right, OwnedGeometryArray):
        right_owned = right
    else:
        right_owned = from_shapely_geometries(list(right) if not isinstance(right, list) else right)

    dispatch_mode = get_requested_mode()
    result = geom_equals_owned(
        left_owned, right_owned, dispatch_mode=dispatch_mode,
    )
    record_dispatch_event(
        surface="geopandas.array.equals",
        operation="equals",
        implementation="geom_equals_owned",
        reason="normalize-then-compare composition for topological equality",
        detail=f"rows={left_owned.row_count}",
        selected=dispatch_mode,
    )
    return result.astype(bool, copy=False)


def _evaluate_geopandas_equals_exact(
    left: np.ndarray | OwnedGeometryArray,
    right: object | np.ndarray | OwnedGeometryArray,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch equals_exact through the dedicated coordinate-comparison path.

    Unlike standard binary predicates, equals_exact cannot use bbox coarse
    filtering because tolerance expands the match window beyond bbox overlap.
    Routes to geom_equals_exact_owned for GPU/CPU dispatch.
    """
    from vibespatial.geometry.equality import geom_equals_exact_owned
    from vibespatial.runtime import get_requested_mode

    tolerance = kwargs["tolerance"] if "tolerance" in kwargs else 0.0

    # Scalar right: fall back to Shapely vectorized equals_exact which
    # handles scalar broadcasting natively in C, avoiding O(N) Python-side
    # geometry duplication.
    is_scalar = not isinstance(right, (OwnedGeometryArray, np.ndarray, list, tuple))
    if is_scalar:
        left_shapely = (
            np.asarray(left.to_shapely(), dtype=object)
            if isinstance(left, OwnedGeometryArray)
            else np.asarray(left, dtype=object)
        )
        result = shapely.equals_exact(left_shapely, right, tolerance=tolerance)
        record_dispatch_event(
            surface="geopandas.array.equals_exact",
            operation="equals_exact",
            implementation="shapely_scalar_broadcast",
            reason="scalar right-hand operand; Shapely vectorized C path",
            detail=f"rows={len(left_shapely)}, tolerance={tolerance}",
            selected=ExecutionMode.CPU,
        )
        return result.astype(bool, copy=False)

    # Coerce inputs to OwnedGeometryArray.
    if isinstance(left, OwnedGeometryArray):
        left_owned = left
    else:
        left_owned = from_shapely_geometries(list(left) if not isinstance(left, list) else left)

    if isinstance(right, OwnedGeometryArray):
        right_owned = right
    else:
        right_owned = from_shapely_geometries(list(right) if not isinstance(right, list) else right)

    dispatch_mode = get_requested_mode()
    result = geom_equals_exact_owned(
        left_owned, right_owned, tolerance, dispatch_mode=dispatch_mode,
    )
    record_dispatch_event(
        surface="geopandas.array.equals_exact",
        operation="equals_exact",
        implementation="geom_equals_exact_owned",
        reason="dedicated coordinate-comparison dispatch for equals_exact",
        detail=f"rows={left_owned.row_count}, tolerance={tolerance}",
        selected=dispatch_mode,
    )
    return result.astype(bool, copy=False)


def _evaluate_geopandas_equals_identical(
    left: np.ndarray | OwnedGeometryArray,
    right: object | np.ndarray | OwnedGeometryArray,
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch equals_identical through the coordinate-comparison path.

    equals_identical is semantically equals_exact(tolerance=0) for 2D
    coordinate data.  Delegates to geom_equals_identical_owned which in
    turn calls geom_equals_exact_owned with tolerance=0.
    """
    from vibespatial.geometry.equality import geom_equals_identical_owned
    from vibespatial.runtime import get_requested_mode

    # Scalar right: fall back to Shapely vectorized equals_identical which
    # handles scalar broadcasting natively in C, avoiding O(N) Python-side
    # geometry duplication.
    is_scalar = not isinstance(right, (OwnedGeometryArray, np.ndarray, list, tuple))
    if is_scalar:
        left_shapely = (
            np.asarray(left.to_shapely(), dtype=object)
            if isinstance(left, OwnedGeometryArray)
            else np.asarray(left, dtype=object)
        )
        result = shapely.equals_exact(left_shapely, right, tolerance=0.0)
        record_dispatch_event(
            surface="geopandas.array.equals_identical",
            operation="equals_identical",
            implementation="shapely_scalar_broadcast",
            reason="scalar right-hand operand; Shapely vectorized C path",
            detail=f"rows={len(left_shapely)}, tolerance=0.0",
            selected=ExecutionMode.CPU,
        )
        return result.astype(bool, copy=False)

    # Coerce inputs to OwnedGeometryArray.
    if isinstance(left, OwnedGeometryArray):
        left_owned = left
    else:
        left_owned = from_shapely_geometries(list(left) if not isinstance(left, list) else left)

    if isinstance(right, OwnedGeometryArray):
        right_owned = right
    else:
        right_owned = from_shapely_geometries(list(right) if not isinstance(right, list) else right)

    dispatch_mode = get_requested_mode()
    result = geom_equals_identical_owned(
        left_owned, right_owned, dispatch_mode=dispatch_mode,
    )
    record_dispatch_event(
        surface="geopandas.array.equals_identical",
        operation="equals_identical",
        implementation="geom_equals_identical_owned",
        reason="dedicated coordinate-comparison dispatch for equals_identical (tolerance=0)",
        detail=f"rows={left_owned.row_count}, tolerance=0.0",
        selected=dispatch_mode,
    )
    return result.astype(bool, copy=False)


def evaluate_geopandas_binary_predicate(
    predicate: str,
    left: np.ndarray | OwnedGeometryArray,
    right: object | np.ndarray | OwnedGeometryArray,
    **kwargs: Any,
) -> np.ndarray | None:
    from vibespatial.runtime import get_requested_mode
    from vibespatial.runtime.execution_trace import execution_trace

    with execution_trace(f"predicate/{predicate}"):
        if not supports_binary_predicate(predicate):
            record_fallback_event(
                surface=f"geopandas.array.{predicate}",
                reason="predicate is not wired to a repo-owned kernel; using host Shapely path",
                detail="unsupported by repo-owned exact predicate engine",
                pipeline="predicate",
            )
            return None

        # --- equals (topological) special path ---
        # Topological equality = structural equality after normalization.
        # Routes to normalize-then-compare composition in equality.py.
        if predicate == "equals":
            try:
                return _evaluate_geopandas_equals(left, right, **kwargs)
            except NotImplementedError:
                record_fallback_event(
                    surface="geopandas.array.equals",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail="NotImplementedError from from_shapely_geometries",
                    pipeline="predicate",
                )
                return None

        # --- equals_exact special path ---
        # Tolerance invalidates the standard bbox coarse filter (two
        # geometries can match within tolerance even when their bboxes
        # don't overlap).  Route directly to the dedicated coordinate-
        # comparison dispatch in geometry/equality.py.
        if predicate == "equals_exact":
            try:
                return _evaluate_geopandas_equals_exact(left, right, **kwargs)
            except NotImplementedError:
                record_fallback_event(
                    surface="geopandas.array.equals_exact",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail="NotImplementedError from from_shapely_geometries",
                    pipeline="predicate",
                )
                return None

        # --- equals_identical special path ---
        # Strict coordinate-level identity (tolerance=0).  Routes through
        # the same NVRTC kernel infrastructure as equals_exact.
        if predicate == "equals_identical":
            try:
                return _evaluate_geopandas_equals_identical(left, right, **kwargs)
            except NotImplementedError:
                record_fallback_event(
                    surface="geopandas.array.equals_identical",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail="NotImplementedError from from_shapely_geometries",
                    pipeline="predicate",
                )
                return None

        left_coerced = left if isinstance(left, OwnedGeometryArray) else np.asarray(left, dtype=object)
        if isinstance(right, OwnedGeometryArray) or np.isscalar(right) or right is None:
            right_coerced = right
        else:
            right_coerced = np.asarray(right, dtype=object)
        result = evaluate_binary_predicate(
            predicate,
            left_coerced,
            right_coerced,
            dispatch_mode=get_requested_mode(),
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
