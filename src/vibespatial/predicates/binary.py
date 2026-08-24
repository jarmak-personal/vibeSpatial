from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np
import shapely

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    get_cuda_runtime,
)
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import (
    FAMILY_TAGS,
    TAG_FAMILIES,
    OwnedGeometryArray,
    OwnedGeometryDeviceState,
    from_shapely_geometries,
    unique_tag_pairs,
)
from vibespatial.kernels.core.geometry_analysis import (
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import (
    PhysicalWorkEstimate,
    WorkloadShape,
    estimate_pairwise_work_from_owned,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.residency import Residency, TransferTrigger, combined_residency
from vibespatial.runtime.robustness import RobustnessPlan, select_robustness_plan

from .point_relations import (
    POINT_LOCATION_BOUNDARY,
    POINT_LOCATION_INTERIOR,
    POINT_LOCATION_OUTSIDE,
    _point_relation_to_predicate_array,
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
_DE9IM_PREDICATES = frozenset(
    {
        "intersects",
        "contains",
        "within",
        "touches",
        "covers",
        "covered_by",
        "overlaps",
        "disjoint",
        "contains_properly",
    }
)
_POINT_POINT_EQUAL_PREDICATES = frozenset(
    {
        "intersects",
        "within",
        "contains",
        "covered_by",
        "covers",
        "contains_properly",
        "equals",
    }
)
_POINT_POINT_FALSE_PREDICATES = frozenset(
    {
        "touches",
        "crosses",
        "overlaps",
    }
)

_FUSED_DE9IM_ROWSET_KERNEL_NAMES = (
    "build_aligned_bbox_candidate_rows_kernel",
    "filter_candidate_tag_pair_kernel",
    "filter_candidate_bounds_within_kernel",
    "init_predicate_output_kernel",
    "scatter_predicate_output_kernel",
)

_FUSED_DE9IM_ROWSET_KERNEL_SOURCE = r"""
extern "C" __global__ void build_aligned_bbox_candidate_rows_kernel(
    const double* __restrict__ left_bounds,
    const double* __restrict__ right_bounds,
    const unsigned char* __restrict__ left_validity,
    const unsigned char* __restrict__ right_validity,
    int n,
    int* __restrict__ candidate_rows,
    int* __restrict__ candidate_count,
    unsigned char* __restrict__ valid_out
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const bool valid = (left_validity[i] != 0u) && (right_validity[i] != 0u);
    valid_out[i] = valid ? 1u : 0u;
    if (!valid) return;

    const double* lb = left_bounds + (4 * i);
    const double* rb = right_bounds + (4 * i);
    const bool hit = (
        lb[0] <= rb[2] && lb[2] >= rb[0] &&
        lb[1] <= rb[3] && lb[3] >= rb[1]
    );
    if (!hit) return;

    const int pos = atomicAdd(candidate_count, 1);
    candidate_rows[pos] = i;
}

extern "C" __global__ void filter_candidate_tag_pair_kernel(
    const int* __restrict__ candidate_rows,
    const int* __restrict__ candidate_count,
    const signed char* __restrict__ left_tags,
    const signed char* __restrict__ right_tags,
    int capacity,
    int left_tag,
    int right_tag,
    int* __restrict__ sub_rows,
    int* __restrict__ sub_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    const int live_count = *candidate_count;
    if (i >= live_count) return;

    const int row = candidate_rows[i];
    if (left_tags[row] != left_tag || right_tags[row] != right_tag) return;
    const int pos = atomicAdd(sub_count, 1);
    sub_rows[pos] = row;
}

extern "C" __global__ void filter_candidate_bounds_within_kernel(
    const int* __restrict__ candidate_rows,
    const int* __restrict__ candidate_count,
    const double* __restrict__ left_bounds,
    const double* __restrict__ right_bounds,
    int capacity,
    int right_row,
    int* __restrict__ sub_rows,
    int* __restrict__ sub_count
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    const int live_count = *candidate_count;
    if (i >= live_count) return;

    const int row = candidate_rows[i];
    const double* lb = left_bounds + (4 * row);
    const int resolved_right_row = right_row < 0 ? row : right_row;
    const double* rb = right_bounds + (4 * resolved_right_row);
    const bool within = (
        lb[0] >= rb[0] && lb[2] <= rb[2] &&
        lb[1] >= rb[1] && lb[3] <= rb[3]
    );
    if (!within) return;
    const int pos = atomicAdd(sub_count, 1);
    sub_rows[pos] = row;
}

extern "C" __global__ void init_predicate_output_kernel(
    const unsigned char* __restrict__ valid,
    unsigned char* __restrict__ out,
    int n,
    int predicate_code
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (predicate_code == 7 && valid[i] != 0u) ? 1u : 0u;
}

extern "C" __global__ void scatter_predicate_output_kernel(
    const int* __restrict__ rows,
    const int* __restrict__ row_count,
    const unsigned char* __restrict__ values,
    unsigned char* __restrict__ out,
    int capacity
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= capacity) return;
    const int live_count = *row_count;
    if (i >= live_count) return;
    out[rows[i]] = values[i] ? 1u : 0u;
}
"""

request_nvrtc_warmup(
    [
        (
            "binary-fused-de9im-rowset",
            _FUSED_DE9IM_ROWSET_KERNEL_SOURCE,
            _FUSED_DE9IM_ROWSET_KERNEL_NAMES,
        ),
    ]
)


def _fused_de9im_rowset_kernels():
    return compile_kernel_group(
        "binary-fused-de9im-rowset",
        _FUSED_DE9IM_ROWSET_KERNEL_SOURCE,
        _FUSED_DE9IM_ROWSET_KERNEL_NAMES,
    )


_SPECIAL_PREDICATES = frozenset({"equals", "equals_exact", "equals_identical"})
_OWNED_EXACT_GEOMETRY_TYPES = frozenset(
    {
        "Point",
        "LineString",
        "Polygon",
        "MultiPoint",
        "MultiLineString",
        "MultiPolygon",
    }
)


def _runtime_device_to_host(
    device_array: object,
    *,
    reason: str,
    terminal_export: bool = False,
) -> np.ndarray:
    return get_cuda_runtime().copy_device_to_host(
        device_array,
        reason=reason,
        terminal_export=terminal_export,
    )


def _runtime_bool_scalar(device_value: object, *, reason: str) -> bool:
    import cupy as cp

    host = _runtime_device_to_host(cp.asarray(device_value).reshape(1), reason=reason)
    return bool(np.asarray(host).reshape(-1)[0])


def _de9im_predicate_device_code(predicate: str) -> int:
    from .polygon import _PREDICATE_DEVICE_CODES

    code = _PREDICATE_DEVICE_CODES.get(predicate)
    if code is None:
        raise ValueError(f"Unsupported predicate for DE-9IM evaluation: {predicate}")
    return int(code)


def _build_aligned_bbox_candidate_rows_device(left_state, right_state, n: int):
    """Build candidate rows and validity in a native row-aligned kernel."""
    import cupy as cp

    d_candidate_rows = cp.empty(n, dtype=cp.int32)
    d_candidate_count = cp.zeros(1, dtype=cp.int32)
    d_valid = cp.empty(n, dtype=cp.bool_)
    if n == 0:
        return d_candidate_rows, d_candidate_count, d_valid

    runtime = get_cuda_runtime()
    kernels = _fused_de9im_rowset_kernels()
    kernel = kernels["build_aligned_bbox_candidate_rows_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(cp.asarray(left_state.row_bounds)),
                ptr(cp.asarray(right_state.row_bounds)),
                ptr(cp.asarray(left_state.validity)),
                ptr(cp.asarray(right_state.validity)),
                n,
                ptr(d_candidate_rows),
                ptr(d_candidate_count),
                ptr(d_valid),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    return d_candidate_rows, d_candidate_count, d_valid


def _filter_candidate_tag_pair_device(
    d_candidate_rows: object,
    d_candidate_count: object,
    left_state,
    right_state,
    *,
    left_tag: int,
    right_tag: int,
    capacity: int,
):
    """Build a device rowset for one family tag pair from candidate rows."""
    import cupy as cp

    d_sub_rows = cp.empty(capacity, dtype=cp.int32)
    d_sub_count = cp.zeros(1, dtype=cp.int32)
    if capacity == 0:
        return d_sub_rows, d_sub_count

    runtime = get_cuda_runtime()
    kernels = _fused_de9im_rowset_kernels()
    kernel = kernels["filter_candidate_tag_pair_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, capacity)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_candidate_rows),
                ptr(d_candidate_count),
                ptr(cp.asarray(left_state.tags)),
                ptr(cp.asarray(right_state.tags)),
                capacity,
                int(left_tag),
                int(right_tag),
                ptr(d_sub_rows),
                ptr(d_sub_count),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    return d_sub_rows, d_sub_count


def _filter_candidate_bounds_within_device(
    d_candidate_rows: object,
    d_candidate_count: object,
    left_state,
    right_state,
    *,
    capacity: int,
    right_row: int = 0,
):
    """Build a device rowset for candidates whose left bounds fit in one right row."""
    import cupy as cp

    d_sub_rows = cp.empty(capacity, dtype=cp.int32)
    d_sub_count = cp.zeros(1, dtype=cp.int32)
    if capacity == 0:
        return d_sub_rows, d_sub_count

    runtime = get_cuda_runtime()
    kernels = _fused_de9im_rowset_kernels()
    kernel = kernels["filter_candidate_bounds_within_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, capacity)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_candidate_rows),
                ptr(d_candidate_count),
                ptr(cp.asarray(left_state.row_bounds)),
                ptr(cp.asarray(right_state.row_bounds)),
                capacity,
                int(right_row),
                ptr(d_sub_rows),
                ptr(d_sub_count),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        ),
    )
    return d_sub_rows, d_sub_count


def _init_de9im_predicate_output_device(d_valid: object, n: int, predicate: str):
    """Initialize a full row-aligned predicate vector on device."""
    import cupy as cp

    d_out = cp.empty(n, dtype=cp.bool_)
    if n == 0:
        return d_out

    runtime = get_cuda_runtime()
    kernels = _fused_de9im_rowset_kernels()
    kernel = kernels["init_predicate_output_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, n)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(d_valid), ptr(d_out), n, _de9im_predicate_device_code(predicate)),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32),
        ),
    )
    return d_out


def _scatter_de9im_predicate_output_device(
    d_rows: object,
    d_row_count: object,
    d_values: object,
    d_out: object,
    *,
    capacity: int,
) -> None:
    if capacity == 0:
        return

    runtime = get_cuda_runtime()
    kernels = _fused_de9im_rowset_kernels()
    kernel = kernels["scatter_predicate_output_kernel"]
    ptr = runtime.pointer
    grid, block = runtime.launch_config(kernel, capacity)
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (ptr(d_rows), ptr(d_row_count), ptr(d_values), ptr(d_out), capacity),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )


def _single_base_row_owned(owned: OwnedGeometryArray) -> OwnedGeometryArray | None:
    if int(owned.row_count) == 1:
        # A one-row indexed view can select any physical base row.  The direct
        # broadcast carrier currently certifies physical row zero, so decline
        # instead of silently changing the selected mask.
        if owned.is_indexed_view:
            return None
        return owned
    base = getattr(owned, "_base", None)
    if base is not None and int(getattr(base, "row_count", -1)) == 1:
        return base
    return None


def _direct_single_convex_containment_outputs_device(
    predicate_names: tuple[str, ...],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    left_state,
    right_state,
) -> dict[str, object] | None:
    """Lower a measured dense broadcast shape before candidate construction.

    For a convex target, a polygonal source is covered exactly when every
    source vertex is covered.  The grouped classifier already evaluates the
    complete source family, so constructing bounds, a candidate rowset, and a
    scatter plan first would only add row-shaped work around the faster
    vertex-shaped algorithm.
    """
    requested = set(predicate_names)
    if not requested or not requested.issubset({"covered_by", "within"}):
        return None

    mask = _single_base_row_owned(right)
    if mask is None:
        return None

    tag_pairs = _device_state_family_tag_pairs(left_state, right_state)
    if tag_pairs is None or len(tag_pairs) != 1:
        return None
    left_tag, right_tag = tag_pairs[0]
    query_family = TAG_FAMILIES.get(left_tag)
    mask_family = TAG_FAMILIES.get(right_tag)
    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    if query_family not in polygonal_families or mask_family not in polygonal_families:
        return None

    from .polygon import compute_polygonal_covered_by_single_convex_grouped_gpu

    d_family_result = compute_polygonal_covered_by_single_convex_grouped_gpu(
        left,
        mask,
        query_family=query_family,
        mask_family=mask_family,
    )
    if d_family_result is None:
        return None

    from vibespatial.api._native_grouped import _map_bounded_grouped_boolean_rows
    from vibespatial.kernels.predicates.point_in_polygon import (
        _wrap_device_result_with_keepalive,
    )
    d_values = _map_bounded_grouped_boolean_rows(
        d_family_result,
        left_state.validity,
        left_state.tags,
        left_state.family_row_offsets,
        right_state.validity,
        row_count=left.row_count,
        family_tag=left_tag,
    )
    d_values = _wrap_device_result_with_keepalive(
        d_values,
        d_family_result,
        left,
        right,
        mask,
    )
    record_dispatch_event(
        surface="vibespatial.predicates.binary",
        operation="/".join(predicate_names),
        selected=ExecutionMode.GPU,
        implementation="gpu_convex_grouped_vertex_containment",
        reason=(
            "conservative convex certificate selected measured dense broadcast "
            "vertex classification with bounded grouped reduction"
        ),
        detail=(
            f"source_rows={left.row_count}, source_family={query_family.value}, "
            f"mask_family={mask_family.value}"
        ),
    )
    return {predicate: d_values for predicate in predicate_names}


def _fused_polygonal_single_right_predicates_device(
    predicate_names: tuple[str, ...],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    left_state,
    right_state,
    d_cand_rows: object,
    d_cand_count: object,
    d_valid: object,
    tag_pairs,
    capacity: int,
    d_candidate_output_rows: object | None = None,
    d_candidate_active: object | None = None,
    right_bounds_are_exact: bool = False,
) -> dict[str, object] | None:
    """Evaluate broadcast-right polygonal predicates without full DE-9IM.

    Candidate slots carry two independent row domains: ``d_cand_rows`` names
    geometry rows in ``left`` while ``d_candidate_output_rows`` names result
    positions.  Keeping both through device selections lets relation-style
    callers refine indexed source rows without physicalizing candidate
    geometry or scattering by source-row cardinality.
    """
    requested = set(predicate_names)
    if not requested.issubset({"intersects", "covered_by", "within"}):
        return None

    mask = _single_base_row_owned(right)
    if mask is None or int(mask.row_count) != 1:
        return None

    import cupy as cp

    mask_state = mask._ensure_device_state(preserve_indexed_view=True)
    mask_families = [
        family
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if family in mask_state.families
    ]
    if len(mask_families) != 1:
        return None
    mask_family = mask_families[0]
    mask_tag = FAMILY_TAGS[mask_family]

    polygon_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    for lt, rt in tag_pairs:
        lf = TAG_FAMILIES.get(lt)
        rf = TAG_FAMILIES.get(rt)
        if lf not in polygon_families or rf != mask_family or rt != mask_tag:
            return None

    from .polygon import (
        compute_polygonal_covered_by_single_convex_grouped_gpu,
        compute_polygonal_covered_by_single_mask_no_holes_gpu,
        compute_polygonal_intersects_gpu,
    )

    outputs = {
        predicate: _init_de9im_predicate_output_device(d_valid, capacity, predicate)
        for predicate in predicate_names
    }
    if capacity == 0:
        return outputs

    from vibespatial.api._native_rowset import NativeDeviceSelection

    d_slots = cp.arange(capacity, dtype=cp.int32)
    d_live = d_slots < cp.asarray(d_cand_count, dtype=cp.int32).reshape(1)[0]
    if d_candidate_active is not None:
        d_active = cp.asarray(d_candidate_active, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != capacity:
            raise ValueError("candidate activity must match predicate capacity")
        d_live &= d_active

    d_source_rows = cp.asarray(d_cand_rows, dtype=cp.int32)
    if d_source_rows.ndim != 1 or int(d_source_rows.size) != capacity:
        raise ValueError("candidate source rows must match predicate capacity")
    d_source_rows = cp.where(d_live, d_source_rows, cp.int32(0))
    if d_candidate_output_rows is None:
        d_output_rows = d_source_rows
    else:
        d_output_rows = cp.asarray(d_candidate_output_rows, dtype=cp.int32)
        if d_output_rows.ndim != 1 or int(d_output_rows.size) != capacity:
            raise ValueError("candidate output rows must match predicate capacity")
        d_output_rows = cp.where(d_live, d_output_rows, cp.int32(0))

    d_left_tags = cp.asarray(left_state.tags, dtype=cp.int8)[d_source_rows]
    d_left_bounds = cp.asarray(left_state.row_bounds, dtype=cp.float64).reshape(
        left.row_count,
        4,
    )[d_source_rows]
    d_right_bounds = cp.asarray(right_state.row_bounds, dtype=cp.float64).reshape(
        -1,
        4,
    )[0]
    d_bounds_within = (
        (d_left_bounds[:, 0] >= d_right_bounds[0])
        & (d_left_bounds[:, 2] <= d_right_bounds[2])
        & (d_left_bounds[:, 1] >= d_right_bounds[1])
        & (d_left_bounds[:, 3] <= d_right_bounds[3])
    )
    d_right_zero = (
        cp.empty(0, dtype=cp.int32) if capacity == 0 else cp.zeros(capacity, dtype=cp.int32)
    )
    for lt, _rt in tag_pairs:
        lf = TAG_FAMILIES[lt]
        d_family = d_live & (d_left_tags == cp.int8(lt))

        if right_bounds_are_exact:
            within_selection = NativeDeviceSelection.from_mask(
                d_family & d_bounds_within,
                source_row_count=capacity,
            )
            d_within_output = within_selection.gather_capacity(d_output_rows)
            d_within_count = cp.asarray(
                within_selection.logical_count,
                dtype=cp.int32,
            )
            d_true = cp.ones(capacity, dtype=cp.bool_)
            for d_out in outputs.values():
                _scatter_de9im_predicate_output_device(
                    d_within_output,
                    d_within_count,
                    d_true,
                    d_out,
                    capacity=capacity,
                )
            intersects_mask = d_family & ~d_bounds_within
        else:
            intersects_mask = d_family

        intersects_selection = NativeDeviceSelection.from_mask(
            intersects_mask,
            source_row_count=capacity,
        )
        d_intersects_source = intersects_selection.gather_capacity(d_source_rows)
        d_intersects_output = intersects_selection.gather_capacity(d_output_rows)
        d_intersects_count = cp.asarray(
            intersects_selection.logical_count,
            dtype=cp.int32,
        )

        if "intersects" in outputs:
            d_intersects = compute_polygonal_intersects_gpu(
                left,
                mask,
                query_family=lf,
                tree_family=mask_family,
                d_left=d_intersects_source,
                d_right=d_right_zero,
                d_pair_count=d_intersects_count,
                pair_capacity=capacity,
                return_device=True,
            )
            if d_intersects is None:
                return None
            _scatter_de9im_predicate_output_device(
                d_intersects_output,
                d_intersects_count,
                d_intersects,
                outputs["intersects"],
                capacity=capacity,
            )

        containment_outputs = tuple(
            predicate for predicate in ("covered_by", "within") if predicate in outputs
        )
        if containment_outputs and not right_bounds_are_exact:
            covered_selection = NativeDeviceSelection.from_mask(
                d_family & d_bounds_within,
                source_row_count=capacity,
            )
            d_covered_source = covered_selection.gather_capacity(d_source_rows)
            d_covered_output = covered_selection.gather_capacity(d_output_rows)
            d_covered_count = cp.asarray(
                covered_selection.logical_count,
                dtype=cp.int32,
            )
            d_family_grouped = compute_polygonal_covered_by_single_convex_grouped_gpu(
                left,
                mask,
                query_family=lf,
                mask_family=mask_family,
            )
            if d_family_grouped is not None:
                d_family_rows = cp.asarray(
                    left_state.family_row_offsets,
                    dtype=cp.int64,
                )[d_covered_source]
                d_covered_by = cp.asarray(d_family_grouped, dtype=cp.bool_)[
                    cp.maximum(d_family_rows, 0)
                ]
            else:
                d_covered_by = compute_polygonal_covered_by_single_mask_no_holes_gpu(
                    left,
                    mask,
                    query_family=lf,
                    mask_family=mask_family,
                    d_left=d_covered_source,
                    d_pair_count=d_covered_count,
                    pair_capacity=capacity,
                    return_device=True,
                )
            if d_covered_by is None:
                return None
            _scatter_de9im_predicate_output_device(
                d_covered_output,
                d_covered_count,
                d_covered_by,
                outputs[containment_outputs[0]],
                capacity=capacity,
            )
            for predicate in containment_outputs[1:]:
                _scatter_de9im_predicate_output_device(
                    d_covered_output,
                    d_covered_count,
                    d_covered_by,
                    outputs[predicate],
                    capacity=capacity,
                )

    return outputs


def _polygonal_single_right_candidate_predicates_device(
    predicate_names: Sequence[str],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    d_candidate_rows: object,
    *,
    d_candidate_active: object | None = None,
    right_bounds_are_exact: bool = False,
) -> dict[str, object] | None:
    """Evaluate candidate-local polygon predicates over indexed source rows.

    The output domain is candidate capacity, not source-row cardinality.  The
    logical candidate count and all family/bounds refinements stay device-side.
    """
    predicate_names = tuple(dict.fromkeys(predicate_names))
    if not predicate_names or not set(predicate_names).issubset({"intersects", "covered_by"}):
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime() or int(right.row_count) != 1:
        return None

    import cupy as cp

    d_candidate_rows = cp.asarray(d_candidate_rows, dtype=cp.int32)
    if d_candidate_rows.ndim != 1:
        raise ValueError("candidate source rows must be one-dimensional")
    capacity = int(d_candidate_rows.size)
    if d_candidate_active is None:
        d_active = cp.ones(capacity, dtype=cp.bool_)
    else:
        d_active = cp.asarray(d_candidate_active, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != capacity:
            raise ValueError("candidate activity must match candidate rows")

    left_state = _ensure_predicate_device_state(
        left,
        reason="single-right candidate predicate: left geometry",
    )
    right_state = _ensure_predicate_device_state(
        right,
        reason="single-right candidate predicate: right geometry",
    )
    for owned, state in ((left, left_state), (right, right_state)):
        if state.row_bounds is None:
            compute_geometry_bounds_device(owned, preserve_indexed_view=True)
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    if left_state.row_bounds is None or right_state.row_bounds is None:
        return None

    tag_pairs = _device_state_family_tag_pairs(left_state, right_state)
    if tag_pairs is None:
        return None
    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    if any(
        TAG_FAMILIES.get(left_tag) not in polygonal_families
        or TAG_FAMILIES.get(right_tag) not in polygonal_families
        for left_tag, right_tag in tag_pairs
    ):
        return None

    d_safe_rows = cp.where(d_active, d_candidate_rows, cp.int32(0))
    d_valid = (
        d_active
        & cp.asarray(left_state.validity, dtype=cp.bool_)[d_safe_rows]
        & cp.asarray(right_state.validity, dtype=cp.bool_).reshape(-1)[0]
    )
    return _fused_polygonal_single_right_predicates_device(
        predicate_names,
        left,
        right,
        left_state=left_state,
        right_state=right_state,
        d_cand_rows=d_safe_rows,
        d_cand_count=cp.asarray([capacity], dtype=cp.int32),
        d_valid=d_valid,
        tag_pairs=tag_pairs,
        capacity=capacity,
        d_candidate_output_rows=cp.arange(capacity, dtype=cp.int32),
        d_candidate_active=d_valid,
        right_bounds_are_exact=right_bounds_are_exact,
    )


def supports_binary_predicate(name: str) -> bool:
    return name in PREDICATE_SPECS or name in _SPECIAL_PREDICATES


def _unsupported_owned_exact_family(values: PredicateInput) -> str | None:
    if isinstance(values, OwnedGeometryArray):
        return None
    array, owned = _coerce_array(values, arg_name="values")
    if owned is not None:
        return None
    assert array is not None
    missing = shapely.is_missing(array)
    for geometry in array[~missing]:
        geom_type = getattr(geometry, "geom_type", None)
        if geom_type not in _OWNED_EXACT_GEOMETRY_TYPES:
            return str(geom_type)
    return None


def _unsupported_owned_exact_operands(
    left: PredicateInput,
    right: object | PredicateInput,
) -> str | None:
    if not isinstance(right, (OwnedGeometryArray, np.ndarray, list, tuple)):
        return None
    for side, values in (("left", left), ("right", right)):
        unsupported = _unsupported_owned_exact_family(values)
        if unsupported is not None:
            return f"{side} contains unsupported geometry family {unsupported}"
    return None


def _coerce_array(
    values: PredicateInput,
    *,
    arg_name: str,
) -> tuple[np.ndarray | None, OwnedGeometryArray | None]:
    if isinstance(values, OwnedGeometryArray):
        return None, values
    to_owned = getattr(values, "to_owned", None)
    if callable(to_owned):
        return None, to_owned()
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            raise TypeError(f"{arg_name} must be a 1D geometry array or a scalar geometry")
        return np.asarray(values, dtype=object), None
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=object), None
    raise TypeError(f"{arg_name} must be an OwnedGeometryArray or 1D geometry sequence")


def _is_scalar_geometry_operand(values: object) -> bool:
    if isinstance(values, (OwnedGeometryArray, np.ndarray, list, tuple)):
        return False
    return not callable(getattr(values, "to_owned", None))


def _coerce_owned_exact_values(
    values: object | PredicateInput,
    *,
    arg_name: str,
) -> tuple[np.ndarray | None, OwnedGeometryArray | None]:
    array, owned = _coerce_array(values, arg_name=arg_name)
    if owned is not None:
        return None, owned
    assert array is not None
    return None, from_shapely_geometries(array.tolist())


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
    *,
    current_residency: Residency = Residency.HOST,
    workload_shape: WorkloadShape | None = None,
    work_estimate: PhysicalWorkEstimate | None = None,
) -> RuntimeSelection:
    plan = plan_dispatch_selection(
        kernel_name=predicate,
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=requested_mode,
        requested_precision=PrecisionMode.AUTO,
        current_residency=current_residency,
        workload_shape=workload_shape,
        work_estimate=work_estimate,
    )
    selection = plan.runtime_selection
    if plan.variant is None:
        if requested_mode is ExecutionMode.GPU:
            raise NotImplementedError(f"{predicate} has no GPU variant registered yet")
        if selection.selected is ExecutionMode.GPU:
            return _explicit_cpu_fallback_selection(
                predicate=predicate,
                requested_mode=requested_mode,
                row_count=row_count,
                reason=f"{predicate} has no GPU variant registered; using explicit CPU fallback",
            )
    return selection


def _explicit_cpu_fallback_selection(
    *,
    predicate: str,
    requested_mode: ExecutionMode,
    row_count: int,
    reason: str,
    workload_shape: WorkloadShape | None = None,
    work_estimate: PhysicalWorkEstimate | None = None,
) -> RuntimeSelection:
    cpu_plan = plan_dispatch_selection(
        kernel_name=predicate,
        kernel_class=KernelClass.PREDICATE,
        row_count=row_count,
        requested_mode=ExecutionMode.CPU,
        requested_precision=PrecisionMode.AUTO,
        workload_shape=workload_shape,
        work_estimate=work_estimate,
    )
    return replace(
        cpu_plan.runtime_selection,
        requested=requested_mode,
        reason=reason,
    )


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
    if n > 1 and right_1row.residency is Residency.DEVICE and right_1row.device_state is not None:
        import cupy as cp

        return OwnedGeometryArray._indexed_view(
            right_1row,
            cp.zeros(n, dtype=cp.int64),
        )

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
    if predicate in {
        "intersects",
        "contains",
        "within",
        "covers",
        "covered_by",
        "contains_properly",
        "equals",
    }:
        return equal
    if predicate == "disjoint":
        return ~equal
    return np.zeros(relation.shape[0], dtype=bool)


def _unsupported_gpu_reason(predicate: str, *, scalar_right: bool) -> str:
    if scalar_right:
        return f"{predicate} GPU refine does not support scalar right-hand geometries yet"
    return (
        f"{predicate} GPU refine currently supports only point-centric and DE-9IM "
        "line/polygon candidate rows"
    )


def _candidate_pairs_supported(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> bool:
    if candidate_rows.size == 0:
        return True
    if (
        left.row_count == right.row_count
        and left.residency is Residency.DEVICE
        and right.residency is Residency.DEVICE
        and left.device_state is not None
        and right.device_state is not None
    ):
        domain_pairs = _device_state_family_tag_pairs(
            left.device_state,
            right.device_state,
        )
        if domain_pairs is not None:
            domain_supported = True
            for left_tag, right_tag in domain_pairs:
                left_is_point = left_tag in {_POINT_TAG, _MP_TAG}
                right_is_point = right_tag in {_POINT_TAG, _MP_TAG}
                if not (
                    (left_is_point and right_tag in _ALL_SUPPORTED_TAGS)
                    or (right_is_point and left_tag in _ALL_SUPPORTED_TAGS)
                ):
                    domain_supported = False
                    break
            if domain_supported:
                return True
    if (
        left.residency is Residency.DEVICE
        and right.residency is Residency.DEVICE
        and left.device_state is not None
        and right.device_state is not None
        and (getattr(left, "_tags", None) is None or getattr(right, "_tags", None) is None)
    ):
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover - guarded by residency
            cp = None
        if cp is not None:
            left_state = left._ensure_device_state(preserve_indexed_view=True)
            right_state = right._ensure_device_state(preserve_indexed_view=True)
            d_rows = cp.asarray(candidate_rows, dtype=cp.int64)
            left_tags = cp.asarray(left_state.tags)[d_rows]
            right_tags = cp.asarray(right_state.tags)[d_rows]
            left_supported = (left_tags >= 0) & (left_tags < len(FAMILY_TAGS))
            right_supported = (right_tags >= 0) & (right_tags < len(FAMILY_TAGS))
            left_is_point = (left_tags == _POINT_TAG) | (left_tags == _MP_TAG)
            right_is_point = (right_tags == _POINT_TAG) | (right_tags == _MP_TAG)
            supported = (left_is_point & right_supported) | (right_is_point & left_supported)
            d_supported = cp.all(supported)
            return bool(
                np.asarray(
                    get_cuda_runtime().copy_device_to_host(
                        cp.asarray(d_supported).reshape(1),
                        reason="binary predicate point-candidate support scalar fence",
                    ),
                    dtype=bool,
                )[0]
            )
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
        np.all(np.isin(left_tags, _DE9IM_TAGS)) and np.all(np.isin(right_tags, _DE9IM_TAGS))
    )


def _gpu_candidate_pairs_supported(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
    predicate: str,
) -> bool:
    """Whether the mixed candidate batch can be partitioned across GPU refine paths."""
    if candidate_rows.size == 0:
        return True
    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]
    left_is_point = (left_tags == _POINT_TAG) | (left_tags == _MP_TAG)
    right_is_point = (right_tags == _POINT_TAG) | (right_tags == _MP_TAG)
    point_rows = candidate_rows[left_is_point | right_is_point]
    de9im_rows = candidate_rows[~(left_is_point | right_is_point)]
    return _candidate_pairs_supported(left, right, point_rows) and _de9im_candidate_pairs_supported(
        left, right, de9im_rows, predicate
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
    if not (
        GeometryFamily.POINT in left.families
        or GeometryFamily.MULTIPOINT in left.families
        or GeometryFamily.POINT in right.families
        or GeometryFamily.MULTIPOINT in right.families
    ):
        return None
    if not (
        any(family in left.families for family in _REGION_FAMILIES)
        or any(family in right.families for family in _REGION_FAMILIES)
    ):
        return None
    if (
        left.residency is Residency.DEVICE
        and right.residency is Residency.DEVICE
        and left.device_state is not None
        and right.device_state is not None
        and (
            getattr(left, "_validity", None) is None
            or getattr(left, "_tags", None) is None
            or getattr(right, "_validity", None) is None
            or getattr(right, "_tags", None) is None
        )
    ):
        return _uniform_point_region_orientation_device(left, right)

    valid_left = left.tags[left.validity]
    valid_right = right.tags[right.validity]
    if valid_left.size == 0 or valid_right.size == 0:
        return None
    return _orientation_from_valid_tag_domains(
        np.bincount(valid_left.astype(np.int64), minlength=len(FAMILY_TAGS)).astype(bool),
        np.bincount(valid_right.astype(np.int64), minlength=len(FAMILY_TAGS)).astype(bool),
    )


def _device_valid_tag_domain(owned: OwnedGeometryArray) -> np.ndarray | None:
    """Return the valid geometry-family tag domain without exporting row metadata."""
    state = owned._ensure_device_state(preserve_indexed_view=True)
    domain = len(FAMILY_TAGS)
    if state.trusted_homogeneous_family is not None:
        tag = FAMILY_TAGS.get(state.trusted_homogeneous_family)
        if tag is None:
            return None
        present = np.zeros(domain, dtype=bool)
        present[int(tag)] = True
        return present
    if state.trusted_family_domain:
        present = np.zeros(domain, dtype=bool)
        for family in state.trusted_family_domain:
            tag = FAMILY_TAGS.get(family)
            if tag is None:
                return None
            present[int(tag)] = True
        return present
    if getattr(owned, "is_indexed_view", False):
        return None

    present = np.zeros(domain, dtype=bool)
    for family in tuple(state.families):
        tag = FAMILY_TAGS.get(family)
        if tag is None:
            return None
        present[int(tag)] = True
    if not bool(np.any(present)):
        return None
    return present


def _orientation_from_valid_tag_domains(
    left_present: np.ndarray,
    right_present: np.ndarray,
) -> tuple[bool, GeometryFamily | None] | None:
    left_active = np.flatnonzero(left_present)
    right_active = np.flatnonzero(right_present)
    if left_active.size == 0 or right_active.size == 0:
        return None
    region_tags = np.asarray(_REGION_TAGS, dtype=np.int64)
    if (
        np.array_equal(left_active, np.asarray([_POINT_TAG], dtype=np.int64))
        and np.isin(right_active, region_tags).all()
    ):
        region_family = TAG_FAMILIES[int(right_active[0])] if right_active.size == 1 else None
        return True, region_family
    if (
        np.array_equal(right_active, np.asarray([_POINT_TAG], dtype=np.int64))
        and np.isin(left_active, region_tags).all()
    ):
        region_family = TAG_FAMILIES[int(left_active[0])] if left_active.size == 1 else None
        return False, region_family
    return None


def _uniform_point_region_orientation_device(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> tuple[bool, GeometryFamily | None] | None:
    left_present = _device_valid_tag_domain(left)
    right_present = _device_valid_tag_domain(right)
    if left_present is None or right_present is None:
        return None
    return _orientation_from_valid_tag_domains(left_present, right_present)


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

    point_region_boolean_predicate = (
        predicate == "intersects"
        or predicate == "disjoint"
        or (point_on_left and predicate == "covered_by")
        or ((not point_on_left) and predicate == "covers")
    )
    if point_region_boolean_predicate:
        from vibespatial.kernels.predicates.point_in_polygon import point_in_polygon

        pip_values = np.asarray(
            point_in_polygon(
                points,
                regions,
                dispatch_mode=ExecutionMode.GPU,
                precision=precision,
            ),
            dtype=object,
        )
        active_rows = ~null_mask
        bool_values = np.zeros(left.row_count, dtype=bool)
        if active_rows.any():
            bool_values[active_rows] = np.asarray(
                pip_values[active_rows],
                dtype=bool,
            )
        if predicate == "disjoint":
            bool_values[active_rows] = ~bool_values[active_rows]

        if null_behavior is NullBehavior.FALSE:
            result_values = bool_values
        else:
            result_values = _fill_output(
                left.row_count,
                null_behavior=null_behavior,
                null_mask=null_mask,
            )
            result_values[active_rows] = bool_values[active_rows]

        precision_plan = select_precision_plan(
            runtime_selection=runtime_selection,
            kernel_class=KernelClass.PREDICATE,
            requested=precision,
        )
        robustness_plan = select_robustness_plan(
            kernel_class=KernelClass.PREDICATE,
            precision_plan=precision_plan,
        )
        active_row_ids = np.flatnonzero(active_rows).astype(np.int32, copy=False)
        return BinaryPredicateResult(
            predicate=predicate,
            values=result_values,
            row_count=left.row_count,
            candidate_rows=active_row_ids,
            coarse_true_rows=np.empty(0, dtype=np.int32),
            coarse_false_rows=np.empty(0, dtype=np.int32),
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )

    if single_region_family is not None:
        active_row_ids = np.flatnonzero(~null_mask).astype(np.int32, copy=False)
        device_values = _evaluate_gpu_point_candidates_device(
            predicate,
            left,
            right,
            active_row_ids,
        )
        if device_values is not None:
            exact_values = _runtime_device_to_host(
                device_values,
                reason=f"binary predicate point-region {predicate} result host export",
                terminal_export=True,
            ).astype(bool, copy=False)
            result = _fill_output(
                left.row_count,
                null_behavior=null_behavior,
                null_mask=null_mask,
            )
            if active_row_ids.size:
                result[active_row_ids] = exact_values
            if null_mask.any() and null_behavior is NullBehavior.FALSE:
                result[null_mask] = False

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
                candidate_rows=active_row_ids,
                coarse_true_rows=np.empty(0, dtype=np.int32),
                coarse_false_rows=np.empty(0, dtype=np.int32),
                runtime_selection=runtime_selection,
                precision_plan=precision_plan,
                robustness_plan=robustness_plan,
            )

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
        compute_geometry_bounds_device(regions)

    from vibespatial.kernels.predicates.point_in_polygon import launch_point_region_candidate_rows

    runtime = get_cuda_runtime()
    candidate_result = launch_point_region_candidate_rows(points, regions)
    try:
        candidate_rows = runtime.copy_device_to_host(
            candidate_result.values,
            reason=(f"binary predicate point-region candidate-row {predicate} host export"),
        ).astype(np.int32, copy=False)
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
        _apply_relation_rows(
            out, np.flatnonzero(point_point_mask), _point_equals_to_predicate(predicate, relation)
        )

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
            predicate,
            left,
            right,
            mp_rows,
            mp_rows,
        )
        _apply_relation_rows(out, mp_idx, mp_result)

    return out


def _evaluate_gpu_point_candidates_device(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> object | None:
    """Evaluate point-family candidate predicates into a device bool vector."""
    if candidate_rows.size == 0:
        import cupy as cp

        return cp.empty(0, dtype=cp.bool_)
    import cupy as cp

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

    from .point_relations import classify_point_predicates_indexed_device

    d_rows = cp.asarray(candidate_rows, dtype=cp.int32)
    return classify_point_predicates_indexed_device(
        predicate,
        left,
        right,
        d_rows,
        d_rows,
    )


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
            left,
            right,
            candidate_rows[sub_idx],
            candidate_rows[sub_idx],
            query_family=lf,
            tree_family=rf,
            d_left=d_sub,
            d_right=d_sub,
        )
        if sub_result is not None:
            de9im_masks[sub_idx] = sub_result

    return evaluate_predicate_from_de9im(de9im_masks, predicate)


def _evaluate_gpu_de9im_candidates_device(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    candidate_rows: np.ndarray,
) -> object:
    """Evaluate non-point candidate predicates into a device bool vector."""
    import cupy as cp

    if candidate_rows.size == 0:
        return cp.empty(0, dtype=cp.bool_)

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

    from .polygon import compute_polygon_de9im_gpu

    left_tags = left.tags[candidate_rows]
    right_tags = right.tags[candidate_rows]
    d_candidate_rows = cp.asarray(candidate_rows, dtype=cp.int32)
    d_out = cp.zeros(candidate_rows.size, dtype=cp.bool_)

    for lt, rt in unique_tag_pairs(left_tags, right_tags):
        sub_mask = (left_tags == lt) & (right_tags == rt)
        sub_idx = np.flatnonzero(sub_mask)
        if sub_idx.size == 0:
            continue
        lf = TAG_FAMILIES[lt] if lt in TAG_FAMILIES else None
        rf = TAG_FAMILIES[rt] if rt in TAG_FAMILIES else None
        if lf is None or rf is None:
            continue
        d_sub_idx = cp.asarray(sub_idx, dtype=cp.int32)
        d_sub_rows = d_candidate_rows[d_sub_idx]
        d_masks = compute_polygon_de9im_gpu(
            left,
            right,
            query_family=lf,
            tree_family=rf,
            d_left=d_sub_rows,
            d_right=d_sub_rows,
            return_device=True,
        )
        if d_masks is not None:
            d_out[d_sub_idx] = _evaluate_de9im_device(d_masks, predicate)

    return d_out


def _evaluate_de9im_device(d_masks: object, predicate: str) -> object:
    """Evaluate a spatial predicate from DE-9IM bitmasks on device.

    Native mirror of ``polygon.evaluate_predicate_from_de9im`` that keeps the
    candidate-refine result on device without CuPy elementwise module loads.

    Parameters
    ----------
    d_masks : cupy uint16 array of DE-9IM bitmasks (device-resident)
    predicate : one of the supported predicate names

    Returns
    -------
    cupy bool array (device-resident)
    """
    from .polygon import evaluate_predicate_from_de9im_device

    return evaluate_predicate_from_de9im_device(d_masks, predicate)


def _contains_point_family(owned: OwnedGeometryArray) -> bool:
    state = getattr(owned, "device_state", None)
    if state is not None and state.trusted_polygonal_only is True:
        return False
    if not any(
        family in owned.families for family in (GeometryFamily.POINT, GeometryFamily.MULTIPOINT)
    ):
        return False
    if (
        owned.residency is Residency.DEVICE
        and owned.device_state is not None
        and (getattr(owned, "_validity", None) is None or getattr(owned, "_tags", None) is None)
    ):
        from vibespatial.runtime import has_gpu_runtime

        if has_gpu_runtime():
            state = owned._ensure_device_state(preserve_indexed_view=True)
            if state.trusted_homogeneous_family is not None:
                return state.trusted_homogeneous_family in {
                    GeometryFamily.POINT,
                    GeometryFamily.MULTIPOINT,
                }
            return bool(
                GeometryFamily.POINT in state.families
                or GeometryFamily.MULTIPOINT in state.families
            )
    valid_tags = owned.tags[owned.validity]
    if valid_tags.size == 0:
        return False
    return bool(np.any((valid_tags == _POINT_TAG) | (valid_tags == _MP_TAG)))


def _owned_tag_pairs(left: OwnedGeometryArray, right: OwnedGeometryArray) -> list[tuple[int, int]]:
    left_state = left.device_state
    right_state = right.device_state
    if (
        left.row_count == right.row_count
        and left_state is not None
        and right_state is not None
        and left_state.trusted_all_valid is True
        and right_state.trusted_all_valid is True
        and left_state.trusted_homogeneous_family is not None
        and right_state.trusted_homogeneous_family is not None
    ):
        return [
            (
                FAMILY_TAGS[left_state.trusted_homogeneous_family],
                FAMILY_TAGS[right_state.trusted_homogeneous_family],
            )
        ]
    if (
        left.row_count == right.row_count
        and left.residency is Residency.DEVICE
        and right.residency is Residency.DEVICE
        and left.device_state is not None
        and right.device_state is not None
    ):
        device_domain_pairs = _device_state_family_tag_pairs(
            left.device_state,
            right.device_state,
        )
        if device_domain_pairs is not None:
            return device_domain_pairs
    if (
        left.row_count == right.row_count
        and left.residency is Residency.DEVICE
        and right.residency is Residency.DEVICE
        and left.device_state is not None
        and right.device_state is not None
        and (
            getattr(left, "_validity", None) is None
            or getattr(left, "_tags", None) is None
            or getattr(right, "_validity", None) is None
            or getattr(right, "_tags", None) is None
        )
    ):
        from vibespatial.runtime import has_gpu_runtime

        if has_gpu_runtime():
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover - guarded by runtime
                cp = None
            if cp is not None:
                left_state = left.device_state
                right_state = right.device_state
                d_valid = cp.asarray(left_state.validity) & cp.asarray(right_state.validity)
                d_pair_codes = (cp.asarray(left_state.tags).astype(cp.int16) << 8) | (
                    cp.asarray(right_state.tags).astype(cp.int16) & cp.int16(0xFF)
                )
                d_codes = cp.unique(
                    d_pair_codes[d_valid],
                )
                if int(d_codes.size) == 0:
                    return []
                host_codes = np.asarray(
                    get_cuda_runtime().copy_device_to_host(
                        d_codes.astype(cp.int16, copy=False),
                        reason="binary predicate tag-pairs host export",
                    ),
                    dtype=np.int16,
                )
                return [((int(code) >> 8) & 0xFF, int(code) & 0xFF) for code in host_codes]
    valid = left.validity & right.validity
    if not bool(np.any(valid)):
        return []
    return unique_tag_pairs(left.tags[valid], right.tags[valid])


def _device_state_family_tag_pairs(
    left_state,
    right_state,
) -> list[tuple[int, int]] | None:
    """Return possible family tag pairs from resident device metadata.

    The fused DE-9IM path groups candidate rows by actual tag pairs on device.
    It only needs the small family-pair search space on the host; exporting the
    observed tag-pair set is a shape break when device state already names the
    resident family buffers.
    """
    polygonal_families = (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
    left_family_source = tuple(getattr(left_state, "families", {}) or ())
    right_family_source = tuple(getattr(right_state, "families", {}) or ())
    left_families = (
        tuple(family for family in polygonal_families if family in left_family_source)
        if getattr(left_state, "trusted_polygonal_only", None) is True
        else left_family_source
    )
    right_families = (
        tuple(family for family in polygonal_families if family in right_family_source)
        if getattr(right_state, "trusted_polygonal_only", None) is True
        else right_family_source
    )
    if not left_families or not right_families:
        return None
    pairs: list[tuple[int, int]] = []
    for left_family in left_families:
        left_tag = FAMILY_TAGS.get(left_family)
        if left_tag is None:
            return None
        for right_family in right_families:
            right_tag = FAMILY_TAGS.get(right_family)
            if right_tag is None:
                return None
            pairs.append((left_tag, right_tag))
    return pairs


def _ensure_predicate_device_state(
    owned: OwnedGeometryArray,
    *,
    reason: str,
) -> OwnedGeometryDeviceState:
    if owned.is_indexed_view and owned.residency is Residency.DEVICE:
        return owned._ensure_device_state(preserve_indexed_view=True)
    owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=reason,
    )
    return owned._ensure_device_state(preserve_indexed_view=True)


def _fused_gpu_binary_predicate_device(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> object | None:
    """Fused device-resident pipeline: bounds + coarse filter + DE-9IM.

    Physical shape: row-aligned predicate expression backed by candidate-pair
    DE-9IM refinement. N-sized intermediaries and candidate rows stay on
    device; the caller decides whether to feed the bool vector to a native
    consumer or export it to a public host array.
    """
    outputs = _fused_gpu_binary_predicates_device((predicate,), left, right)
    if outputs is None:
        return None
    return outputs.get(predicate)


def _fused_gpu_binary_predicate(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> np.ndarray | None:
    """Fused device-resident pipeline with a final public host export."""
    d_out = _fused_gpu_binary_predicate_device(predicate, left, right)
    if d_out is None:
        return None
    host_result = _runtime_device_to_host(
        d_out,
        reason=f"binary predicate exact {predicate} result host export",
        terminal_export=True,
    )
    return np.asarray(host_result, dtype=bool)


def _evaluate_gpu_point_pair_fast_path(
    predicate: str,
    *,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    null_mask: np.ndarray,
    null_behavior: NullBehavior,
    runtime_selection: RuntimeSelection,
    precision: PrecisionMode | str,
) -> BinaryPredicateResult | None:
    """Evaluate row-aligned point/point predicates without host bounds export."""
    if predicate not in (
        _POINT_POINT_EQUAL_PREDICATES | _POINT_POINT_FALSE_PREDICATES | {"disjoint"}
    ):
        return None
    if left.row_count != right.row_count:
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    left.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for left point input",
    )
    right.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason=f"{predicate} selected GPU execution for right point input",
    )

    # This probe runs before the point-family domain is known.  Preserve a
    # broadcast/indexed operand until the family check rejects it; resolving a
    # one-row polygon mask here would erase reusable native lineage before the
    # polygon containment planner gets a chance to inspect it.
    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    if set(left_state.families) != {GeometryFamily.POINT}:
        return None
    if set(right_state.families) != {GeometryFamily.POINT}:
        return None

    import cupy as cp

    left_points = left_state.families[GeometryFamily.POINT]
    right_points = right_state.families[GeometryFamily.POINT]
    left_rows = cp.asarray(left_state.family_row_offsets).astype(cp.int64, copy=False)
    right_rows = cp.asarray(right_state.family_row_offsets).astype(cp.int64, copy=False)
    safe_left_rows = cp.maximum(left_rows, 0)
    safe_right_rows = cp.maximum(right_rows, 0)

    d_valid = (
        cp.asarray(left_state.validity, dtype=cp.bool_)
        & cp.asarray(right_state.validity, dtype=cp.bool_)
        & (left_rows >= 0)
        & (right_rows >= 0)
    )
    d_non_empty = (
        ~cp.asarray(left_points.empty_mask, dtype=cp.bool_)[safe_left_rows]
        & ~cp.asarray(right_points.empty_mask, dtype=cp.bool_)[safe_right_rows]
    )
    if int(left_points.x.size) == 0 or int(right_points.x.size) == 0:
        d_equal = cp.zeros(left.row_count, dtype=cp.bool_)
    else:
        left_coord_rows = cp.minimum(
            left_points.geometry_offsets[safe_left_rows].astype(cp.int64, copy=False),
            int(left_points.x.size) - 1,
        )
        right_coord_rows = cp.minimum(
            right_points.geometry_offsets[safe_right_rows].astype(cp.int64, copy=False),
            int(right_points.x.size) - 1,
        )
        d_equal = (
            d_valid
            & d_non_empty
            & (left_points.x[left_coord_rows] == right_points.x[right_coord_rows])
            & (left_points.y[left_coord_rows] == right_points.y[right_coord_rows])
        )
    if predicate == "disjoint":
        d_values = d_valid & ~d_equal
    elif predicate in _POINT_POINT_EQUAL_PREDICATES:
        d_values = d_equal
    else:
        d_values = cp.zeros(left.row_count, dtype=cp.bool_)

    host_values = _runtime_device_to_host(
        d_values,
        reason=f"binary predicate point-point {predicate} result host export",
        terminal_export=True,
    ).astype(bool, copy=False)
    result = _fill_output(
        left.row_count,
        null_behavior=null_behavior,
        null_mask=null_mask,
    )
    if null_behavior is NullBehavior.FALSE:
        result[:] = host_values
        if null_mask.any():
            result[null_mask] = False
    else:
        non_null = ~null_mask
        result[non_null] = host_values[non_null]

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
        candidate_rows=np.flatnonzero(host_values & ~null_mask).astype(
            np.int32,
            copy=False,
        ),
        coarse_true_rows=np.empty(0, dtype=np.int32),
        coarse_false_rows=np.empty(0, dtype=np.int32),
        runtime_selection=runtime_selection,
        precision_plan=precision_plan,
        robustness_plan=robustness_plan,
    )


def _evaluate_gpu_point_pair_device(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> object | None:
    """Evaluate row-aligned point/point predicates into a device bool vector.

    Physical shape: aligned pairwise point rows. Work units are rows plus one
    coordinate load per side. This is a Tier 2 CuPy element-wise native
    expression path, so there is no public host export until a caller asks for
    one.
    """
    if predicate not in (
        _POINT_POINT_EQUAL_PREDICATES | _POINT_POINT_FALSE_PREDICATES | {"disjoint"}
    ):
        return None
    if left.row_count != right.row_count:
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    left_state = _ensure_predicate_device_state(
        left,
        reason=f"{predicate} native expression left point input",
    )
    right_state = _ensure_predicate_device_state(
        right,
        reason=f"{predicate} native expression right point input",
    )
    if set(left_state.families) != {GeometryFamily.POINT}:
        return None
    if set(right_state.families) != {GeometryFamily.POINT}:
        return None

    import cupy as cp

    left_points = left_state.families[GeometryFamily.POINT]
    right_points = right_state.families[GeometryFamily.POINT]
    left_rows = cp.asarray(left_state.family_row_offsets).astype(cp.int64, copy=False)
    right_rows = cp.asarray(right_state.family_row_offsets).astype(cp.int64, copy=False)
    safe_left_rows = cp.maximum(left_rows, 0)
    safe_right_rows = cp.maximum(right_rows, 0)

    d_valid = (
        cp.asarray(left_state.validity, dtype=cp.bool_)
        & cp.asarray(right_state.validity, dtype=cp.bool_)
        & (left_rows >= 0)
        & (right_rows >= 0)
    )
    d_non_empty = (
        ~cp.asarray(left_points.empty_mask, dtype=cp.bool_)[safe_left_rows]
        & ~cp.asarray(right_points.empty_mask, dtype=cp.bool_)[safe_right_rows]
    )
    if int(left_points.x.size) == 0 or int(right_points.x.size) == 0:
        d_equal = cp.zeros(left.row_count, dtype=cp.bool_)
    else:
        left_coord_rows = cp.minimum(
            left_points.geometry_offsets[safe_left_rows].astype(cp.int64, copy=False),
            int(left_points.x.size) - 1,
        )
        right_coord_rows = cp.minimum(
            right_points.geometry_offsets[safe_right_rows].astype(cp.int64, copy=False),
            int(right_points.x.size) - 1,
        )
        d_equal = (
            d_valid
            & d_non_empty
            & (left_points.x[left_coord_rows] == right_points.x[right_coord_rows])
            & (left_points.y[left_coord_rows] == right_points.y[right_coord_rows])
        )
    if predicate == "disjoint":
        return d_valid & ~d_equal
    if predicate in _POINT_POINT_EQUAL_PREDICATES:
        return d_equal
    return cp.zeros(left.row_count, dtype=cp.bool_)


def _record_binary_predicate_expression_dispatch(
    *,
    predicate: str,
    expression_operation: str,
    row_count: int,
    implementation: str,
    workload_shape: str,
) -> None:
    record_dispatch_event(
        surface="vibespatial.predicates.binary",
        operation=predicate,
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        implementation=implementation,
        reason="device-resident binary predicate expression",
        detail=(
            f"operation={expression_operation}; "
            f"row_count={row_count}; "
            f"workload_shape={workload_shape}; "
            "carrier=NativeExpression"
        ),
    )


def _align_native_expression_owned_pair(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> tuple[OwnedGeometryArray, OwnedGeometryArray, str] | None:
    """Align owned operands for native expression evaluation.

    Public predicates already support scalar/broadcast-right shapes.  The
    private expression path needs the same carrier so downstream consumers can
    keep broadcast predicate vectors on device instead of falling back to the
    public bool-mask export path.
    """
    if left.row_count == right.row_count:
        return left, right, "aligned_pairwise"
    if right.row_count == 1 and left.row_count > 1:
        return left, _broadcast_right_owned(right, left.row_count), "broadcast_right"
    if left.row_count == 1 and right.row_count > 1:
        return _broadcast_right_owned(left, right.row_count), right, "broadcast_left"
    return None


def _trusted_all_valid_owned_pair(
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> bool:
    left_state = left.device_state
    right_state = right.device_state
    return bool(
        left_state is not None
        and right_state is not None
        and left_state.trusted_all_valid is True
        and right_state.trusted_all_valid is True
    )


def _point_region_predicate_expression_device(
    predicate: str,
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    *,
    point_on_left: bool,
    region_family: GeometryFamily,
):
    """Evaluate point-region predicates with device validity gating.

    The public PIP fused path assumes all rows are valid and can read
    family-row offsets directly.  Native predicate expressions cannot insert a
    scalar all-valid fence before every consumer, so mixed-validity rowsets use
    this compact valid-row carrier and only launch exact point-region work for
    rows whose pair metadata is valid on device.
    """
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    import cupy as cp

    points = left if point_on_left else right
    regions = right if point_on_left else left
    points_state = _ensure_predicate_device_state(
        points,
        reason=f"{predicate} native expression point input",
    )
    regions_state = _ensure_predicate_device_state(
        regions,
        reason=f"{predicate} native expression region input",
    )
    if GeometryFamily.POINT not in points_state.families:
        return None
    if region_family not in regions_state.families:
        return None

    d_valid = cp.asarray(points_state.validity, dtype=cp.bool_) & cp.asarray(
        regions_state.validity,
        dtype=cp.bool_,
    )
    d_rows = cp.flatnonzero(d_valid).astype(cp.int32, copy=False)
    values = cp.zeros(left.row_count, dtype=cp.bool_)
    if d_rows.size == 0:
        return values
    relation = classify_point_region_gpu(
        d_rows,
        points,
        regions,
        region_family=region_family,
        return_device=True,
    )
    values[d_rows] = _point_relation_to_predicate_array(
        predicate,
        relation,
        point_on_left=point_on_left,
    )
    return values


def binary_predicate_expression(
    predicate: str,
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    source_token: str | None = None,
    operation: str | None = None,
):
    """Return row-aligned binary predicate results as a private expression.

    This is the Native* consumer path for admitted binary predicates. It keeps
    the bool vector on device so sanctioned callers can lower directly to a
    ``NativeRowSet`` instead of exporting a public bool Series.
    """
    requested_mode = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    if requested_mode is not ExecutionMode.GPU:
        return None
    if predicate not in PREDICATE_SPECS:
        raise ValueError(f"unsupported binary predicate: {predicate}") from None

    _left_values, left_owned = _coerce_array(left, arg_name="left")
    _right_values, right_owned = _coerce_array(right, arg_name="right")
    if left_owned is None or right_owned is None:
        return None
    aligned = _align_native_expression_owned_pair(left_owned, right_owned)
    if aligned is None:
        return None
    left_owned, right_owned, expression_workload_shape = aligned

    expression_operation = operation or f"binary_predicate.{predicate}"
    point_pair_values = _evaluate_gpu_point_pair_device(
        predicate,
        left_owned,
        right_owned,
    )
    if point_pair_values is not None:
        from vibespatial.api._native_expression import NativeExpression

        _record_binary_predicate_expression_dispatch(
            predicate=predicate,
            expression_operation=expression_operation,
            row_count=left_owned.row_count,
            implementation="native_point_pair_expression_gpu",
            workload_shape=f"{expression_workload_shape}_point_point",
        )
        return NativeExpression(
            operation=expression_operation,
            values=point_pair_values,
            source_token=source_token,
            source_row_count=left_owned.row_count,
            dtype=str(getattr(point_pair_values, "dtype", "bool")),
            precision="predicate",
        )

    orientation = _uniform_point_region_orientation(left_owned, right_owned)
    if orientation is not None:
        point_on_left, single_region_family = orientation
        point_region_boolean_predicate = (
            predicate == "intersects"
            or predicate == "disjoint"
            or (point_on_left and predicate == "covered_by")
            or ((not point_on_left) and predicate == "covers")
        )
        if point_region_boolean_predicate:
            points = left_owned if point_on_left else right_owned
            regions = right_owned if point_on_left else left_owned
            from vibespatial.kernels.predicates.point_in_polygon import (
                point_in_polygon_expression,
            )

            expression = point_in_polygon_expression(
                points,
                regions,
                dispatch_mode=ExecutionMode.GPU,
                precision=precision,
                source_token=source_token,
                operation=expression_operation,
            )
            if expression is not None:
                import cupy as cp

                point_state = points._ensure_device_state(
                    preserve_indexed_view=True,
                )
                region_state = regions._ensure_device_state(
                    preserve_indexed_view=True,
                )
                d_valid = cp.asarray(point_state.validity, dtype=cp.bool_) & cp.asarray(
                    region_state.validity,
                    dtype=cp.bool_,
                )
                d_contains = cp.asarray(expression.values, dtype=cp.bool_)
                values = d_valid & ~d_contains if predicate == "disjoint" else d_valid & d_contains
                from vibespatial.api._native_expression import NativeExpression

                _record_binary_predicate_expression_dispatch(
                    predicate=predicate,
                    expression_operation=expression_operation,
                    row_count=left_owned.row_count,
                    implementation="native_point_region_pip_expression_gpu",
                    workload_shape=f"{expression_workload_shape}_point_region",
                )
                return NativeExpression(
                    operation=expression_operation,
                    values=values,
                    source_token=source_token,
                    source_row_count=left_owned.row_count,
                    dtype=str(getattr(values, "dtype", "bool")),
                    precision="predicate",
                )

        if single_region_family is not None:
            values = _point_region_predicate_expression_device(
                predicate,
                left_owned,
                right_owned,
                point_on_left=point_on_left,
                region_family=single_region_family,
            )
            if values is None:
                return None
            from vibespatial.api._native_expression import NativeExpression

            _record_binary_predicate_expression_dispatch(
                predicate=predicate,
                expression_operation=expression_operation,
                row_count=left_owned.row_count,
                implementation="native_point_region_relation_expression_gpu",
                workload_shape=f"{expression_workload_shape}_point_region",
            )
            return NativeExpression(
                operation=expression_operation,
                values=values,
                source_token=source_token,
                source_row_count=left_owned.row_count,
                dtype=str(getattr(values, "dtype", "bool")),
                precision="predicate",
            )

    point_family_values = None
    if _contains_point_family(left_owned) or _contains_point_family(right_owned):
        import cupy as cp

        row_ids = cp.arange(left_owned.row_count, dtype=cp.int32)
        point_family_values = _evaluate_gpu_point_candidates_device(
            predicate,
            left_owned,
            right_owned,
            row_ids,
        )
    de9im_values = _fused_gpu_binary_predicate_device(
        predicate,
        left_owned,
        right_owned,
    )
    if point_family_values is not None and de9im_values is not None:
        values = point_family_values | de9im_values
        implementation = "native_mixed_family_expression_gpu"
        workload_shape = f"{expression_workload_shape}_point_family_de9im"
    elif point_family_values is not None:
        values = point_family_values
        implementation = "native_point_family_indexed_expression_gpu"
        workload_shape = f"{expression_workload_shape}_point_family"
    elif de9im_values is not None:
        values = de9im_values
        implementation = "native_expression_gpu"
        workload_shape = f"{expression_workload_shape}_de9im"
    else:
        return None

    if point_family_values is not None:
        from vibespatial.api._native_expression import NativeExpression

        _record_binary_predicate_expression_dispatch(
            predicate=predicate,
            expression_operation=expression_operation,
            row_count=left_owned.row_count,
            implementation=implementation,
            workload_shape=workload_shape,
        )
        return NativeExpression(
            operation=expression_operation,
            values=values,
            source_token=source_token,
            source_row_count=left_owned.row_count,
            dtype=str(getattr(values, "dtype", "bool")),
            precision="predicate",
        )

    from vibespatial.api._native_expression import NativeExpression

    _record_binary_predicate_expression_dispatch(
        predicate=predicate,
        expression_operation=expression_operation,
        row_count=left_owned.row_count,
        implementation=implementation,
        workload_shape=workload_shape,
    )
    return NativeExpression(
        operation=expression_operation,
        values=values,
        source_token=source_token,
        source_row_count=left_owned.row_count,
        dtype=str(getattr(values, "dtype", "bool")),
        precision="predicate",
    )


def _fused_gpu_binary_predicates_device(
    predicates: Sequence[str],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> dict[str, object] | None:
    """Evaluate multiple DE-9IM predicates with one device-resident exact pass.

    Mask-clip physical plans often need both ``intersects`` and
    ``covered_by`` for the same pairwise inputs.  Computing the DE-9IM mask
    once and deriving multiple boolean vectors avoids duplicate polygon
    relation kernels.  All candidate rows, DE-9IM masks, and full-size
    predicate vectors stay on device so native callers can wrap them as
    ``NativeExpression`` values.
    """
    predicate_names = tuple(dict.fromkeys(predicates))
    if not predicate_names or any(
        predicate not in _DE9IM_PREDICATES for predicate in predicate_names
    ):
        return None
    if left.row_count != right.row_count:
        return None
    if _contains_point_family(left) or _contains_point_family(right):
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    _ensure_predicate_device_state(
        left,
        reason="fused GPU multi-predicate: left geometry",
    )
    _ensure_predicate_device_state(
        right,
        reason="fused GPU multi-predicate: right geometry",
    )

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    direct_containment = _direct_single_convex_containment_outputs_device(
        predicate_names,
        left,
        right,
        left_state=left_state,
        right_state=right_state,
    )
    if direct_containment is not None:
        return direct_containment

    for arr, state_ref in ((left, "left"), (right, "right")):
        state = arr._ensure_device_state(preserve_indexed_view=True)
        if state.row_bounds is None:
            compute_geometry_bounds_device(arr, preserve_indexed_view=True)
            state = arr._ensure_device_state(preserve_indexed_view=True)
            if state.row_bounds is None:
                record_fallback_event(
                    surface="vibespatial.predicates.binary._evaluate_binary_predicates_fused_gpu",
                    reason=f"GPU bounds kernel unavailable for {state_ref}; using individual predicate path",
                    d2h_transfer=False,
                )
                return None

    left_state = left._ensure_device_state(preserve_indexed_view=True)
    right_state = right._ensure_device_state(preserve_indexed_view=True)
    n = left.row_count
    d_cand_rows, d_cand_count, d_valid = _build_aligned_bbox_candidate_rows_device(
        left_state,
        right_state,
        n,
    )
    left_within_predicates = {"covered_by", "within"}
    right_within_predicates = {"covers", "contains", "contains_properly"}
    if set(predicate_names).issubset(left_within_predicates):
        d_cand_rows, d_cand_count = _filter_candidate_bounds_within_device(
            d_cand_rows,
            d_cand_count,
            left_state,
            right_state,
            capacity=n,
            right_row=-1,
        )
    elif set(predicate_names).issubset(right_within_predicates):
        d_cand_rows, d_cand_count = _filter_candidate_bounds_within_device(
            d_cand_rows,
            d_cand_count,
            right_state,
            left_state,
            capacity=n,
            right_row=-1,
        )
    point_tags = {
        FAMILY_TAGS[GeometryFamily.POINT],
        FAMILY_TAGS[GeometryFamily.MULTIPOINT],
    }
    tag_pairs = _device_state_family_tag_pairs(left_state, right_state)
    if tag_pairs is None:
        tag_pairs = _owned_tag_pairs(left, right)
    tag_pairs = [(lt, rt) for lt, rt in tag_pairs if lt not in point_tags and rt not in point_tags]

    outputs = {
        predicate: _init_de9im_predicate_output_device(d_valid, n, predicate)
        for predicate in predicate_names
    }
    if not tag_pairs:
        return outputs

    single_right_outputs = _fused_polygonal_single_right_predicates_device(
        predicate_names,
        left,
        right,
        left_state=left_state,
        right_state=right_state,
        d_cand_rows=d_cand_rows,
        d_cand_count=d_cand_count,
        d_valid=d_valid,
        tag_pairs=tag_pairs,
        capacity=n,
    )
    if single_right_outputs is not None:
        return single_right_outputs

    inverse_single_left = {
        "covers": "covered_by",
        "contains": "within",
    }
    if set(predicate_names).issubset(inverse_single_left):
        mapped_names = tuple(inverse_single_left[name] for name in predicate_names)
        swapped_outputs = _fused_polygonal_single_right_predicates_device(
            mapped_names,
            right,
            left,
            left_state=right_state,
            right_state=left_state,
            d_cand_rows=d_cand_rows,
            d_cand_count=d_cand_count,
            d_valid=d_valid,
            tag_pairs=[(right_tag, left_tag) for left_tag, right_tag in tag_pairs],
            capacity=n,
        )
        if swapped_outputs is not None:
            return {
                predicate: swapped_outputs[inverse_single_left[predicate]]
                for predicate in predicate_names
            }

    from .polygon import compute_polygon_de9im_gpu

    single_tag_pair = len(tag_pairs) == 1
    for lt, rt in tag_pairs:
        lf = TAG_FAMILIES[lt] if lt in TAG_FAMILIES else None
        rf = TAG_FAMILIES[rt] if rt in TAG_FAMILIES else None
        if lf is None or rf is None:
            return None
        if single_tag_pair:
            d_sub_cand = d_cand_rows
            d_sub_count = d_cand_count
        else:
            d_sub_cand, d_sub_count = _filter_candidate_tag_pair_device(
                d_cand_rows,
                d_cand_count,
                left_state,
                right_state,
                left_tag=lt,
                right_tag=rt,
                capacity=n,
            )
        d_sub_result = compute_polygon_de9im_gpu(
            left,
            right,
            query_family=lf,
            tree_family=rf,
            d_left=d_sub_cand,
            d_right=d_sub_cand,
            d_pair_count=d_sub_count,
            pair_capacity=n,
            return_device=True,
        )
        if d_sub_result is None:
            return None
        for predicate, d_out in outputs.items():
            d_values = _evaluate_de9im_device(d_sub_result, predicate)
            _scatter_de9im_predicate_output_device(
                d_sub_cand,
                d_sub_count,
                d_values,
                d_out,
                capacity=n,
            )
    return outputs


def _binary_predicate_relation_pair_values_device(
    predicates: Sequence[str],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
    d_left_rows,
    d_right_rows,
    *,
    d_pair_active=None,
    operation_prefix: str = "binary_predicate.relation_pair",
    left_pair_bounds=None,
    right_pair_bounds=None,
) -> dict[str, object] | None:
    """Evaluate predicate vectors for explicit native relation pairs.

    Physical shape: relation-pair indices over existing owned geometry
    carriers. The geometry buffers are not gathered or physicalized; DE-9IM
    kernels dereference the provided device row indices directly.
    """
    predicate_names = tuple(dict.fromkeys(predicates))
    if not predicate_names or any(
        predicate not in _DE9IM_PREDICATES for predicate in predicate_names
    ):
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    import cupy as cp

    d_left_rows = cp.asarray(d_left_rows, dtype=cp.int32)
    d_right_rows = cp.asarray(d_right_rows, dtype=cp.int32)
    pair_count = int(d_left_rows.size)
    if pair_count != int(d_right_rows.size):
        return None
    if d_pair_active is None:
        d_active = cp.ones(pair_count, dtype=cp.bool_)
        all_pair_slots_active = True
    else:
        d_active = cp.asarray(d_pair_active, dtype=cp.bool_)
        if d_active.ndim != 1 or int(d_active.size) != pair_count:
            raise ValueError("relation pair activity must match pair capacity")
        all_pair_slots_active = False
    d_left_rows = cp.where(d_active, d_left_rows, cp.int32(0))
    d_right_rows = cp.where(d_active, d_right_rows, cp.int32(0))

    left_state = _ensure_predicate_device_state(
        left,
        reason="relation-pair predicate: left geometry",
    )
    right_state = _ensure_predicate_device_state(
        right,
        reason="relation-pair predicate: right geometry",
    )
    needs_covered_by_bounds = predicate_names == ("covered_by",)
    if needs_covered_by_bounds and left_pair_bounds is None and left_state.row_bounds is None:
        compute_geometry_bounds_device(left, preserve_indexed_view=True)
        left_state = left._ensure_device_state(preserve_indexed_view=True)
    if needs_covered_by_bounds and right_pair_bounds is None and right_state.row_bounds is None:
        compute_geometry_bounds_device(right, preserve_indexed_view=True)
        right_state = right._ensure_device_state(preserve_indexed_view=True)
    tag_pairs = _device_state_family_tag_pairs(left_state, right_state)
    if tag_pairs is None:
        return None
    polygonal_families = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    for left_tag, right_tag in tag_pairs:
        if (
            TAG_FAMILIES.get(left_tag) not in polygonal_families
            or TAG_FAMILIES.get(right_tag) not in polygonal_families
        ):
            return None

    d_left_valid = cp.asarray(left_state.validity, dtype=cp.bool_)[d_left_rows]
    d_right_valid = cp.asarray(right_state.validity, dtype=cp.bool_)[d_right_rows]
    d_valid = d_active & d_left_valid & d_right_valid
    d_refine = d_valid
    if predicate_names == ("covered_by",):
        if left_pair_bounds is not None and right_pair_bounds is not None:
            d_left_bounds = cp.asarray(left_pair_bounds, dtype=cp.float64).reshape(
                pair_count,
                4,
            )
            d_right_bounds = cp.asarray(right_pair_bounds, dtype=cp.float64).reshape(
                pair_count,
                4,
            )
        else:
            if left_state.row_bounds is None or right_state.row_bounds is None:
                return None
            d_left_bounds = cp.asarray(
                left_state.row_bounds,
                dtype=cp.float64,
            ).reshape(
                -1,
                4,
            )[d_left_rows]
            d_right_bounds = cp.asarray(
                right_state.row_bounds,
                dtype=cp.float64,
            ).reshape(
                -1,
                4,
            )[d_right_rows]
        d_refine = (
            d_refine
            & (d_left_bounds[:, 0] >= d_right_bounds[:, 0])
            & (d_left_bounds[:, 2] <= d_right_bounds[:, 2])
            & (d_left_bounds[:, 1] >= d_right_bounds[:, 1])
            & (d_left_bounds[:, 3] <= d_right_bounds[:, 3])
        )
    d_left_tags = cp.asarray(left_state.tags)[d_left_rows]
    d_right_tags = cp.asarray(right_state.tags)[d_right_rows]
    outputs = {
        predicate: _init_de9im_predicate_output_device(
            d_valid,
            pair_count,
            predicate,
        )
        for predicate in predicate_names
    }
    if pair_count == 0:
        return outputs

    from vibespatial.api._native_rowset import NativeDeviceSelection

    from .polygon import compute_polygon_de9im_gpu

    single_pair = len(tag_pairs) == 1
    all_pairs_valid = (
        predicate_names != ("covered_by",)
        and all_pair_slots_active
        and left_state.trusted_all_valid is True
        and right_state.trusted_all_valid is True
    )
    direct_covered_by = predicate_names == ("covered_by",)
    used_full_de9im = False
    d_pair_slots = cp.arange(pair_count, dtype=cp.int32)
    for left_tag, right_tag in tag_pairs:
        left_family = TAG_FAMILIES[left_tag]
        right_family = TAG_FAMILIES[right_tag]
        if single_pair and all_pairs_valid:
            selection = NativeDeviceSelection.identity(pair_count)
        else:
            d_sub_mask = d_refine
            if not single_pair:
                d_sub_mask = d_sub_mask & (d_left_tags == left_tag) & (d_right_tags == right_tag)
            selection = NativeDeviceSelection.from_mask(
                d_sub_mask,
                source_row_count=pair_count,
            )
        d_sub_left = selection.gather_capacity(d_left_rows)
        d_sub_right = selection.gather_capacity(d_right_rows)
        d_sub_idx = selection.gather_capacity(d_pair_slots)
        d_sub_count = cp.asarray(selection.logical_count, dtype=cp.int32)

        if direct_covered_by:
            from .polygon import compute_polygonal_covered_by_pair_rows_no_holes_gpu

            d_values = compute_polygonal_covered_by_pair_rows_no_holes_gpu(
                left,
                right,
                query_family=left_family,
                mask_family=right_family,
                d_left=d_sub_left,
                d_right=d_sub_right,
                d_pair_count=d_sub_count,
                pair_capacity=pair_count,
                return_device=True,
            )
            if d_values is not None:
                _scatter_de9im_predicate_output_device(
                    d_sub_idx,
                    d_sub_count,
                    d_values,
                    outputs["covered_by"],
                    capacity=pair_count,
                )
                continue

        d_masks = compute_polygon_de9im_gpu(
            left,
            right,
            query_family=left_family,
            tree_family=right_family,
            d_left=d_sub_left,
            d_right=d_sub_right,
            d_pair_count=d_sub_count,
            pair_capacity=pair_count,
            return_device=True,
        )
        if d_masks is None:
            return None
        used_full_de9im = True
        for predicate, d_out in outputs.items():
            d_values = _evaluate_de9im_device(d_masks, predicate)
            _scatter_de9im_predicate_output_device(
                d_sub_idx,
                d_sub_count,
                d_values,
                d_out,
                capacity=pair_count,
            )

    if direct_covered_by and not used_full_de9im:
        implementation = "relation_pair_covered_by_no_holes_gpu"
        workload_shape = "relation_pair_covered_by"
        reason = "device-resident relation-pair covered_by evaluation"
    else:
        implementation = "relation_pair_predicate_gpu"
        workload_shape = "relation_pair_de9im"
        reason = "device-resident relation-pair predicate evaluation"
    detail = (
        f"operation={operation_prefix}; row_count={pair_count}; "
        f"workload_shape={workload_shape}; carrier=device_bool"
    )
    for predicate in predicate_names:
        record_dispatch_event(
            surface="vibespatial.predicates.binary",
            operation=predicate,
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            implementation=implementation,
            reason=reason,
            detail=detail,
        )
    return outputs


def binary_predicate_expressions(
    predicates: Sequence[str],
    left: PredicateInput,
    right: PredicateInput,
    *,
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    source_token: str | None = None,
    operation_prefix: str = "binary_predicate",
):
    """Return multiple row-aligned predicate vectors as native expressions.

    Physical shape: aligned pairwise predicate expression.  For all-DE-9IM,
    non-point family pairs, one candidate/refine pass computes all requested
    predicates and scatters full-size device vectors.  Other admitted native
    shapes reuse the single-predicate expression producer per predicate.
    """
    requested_mode = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
    if requested_mode is not ExecutionMode.GPU:
        return None
    predicate_names = tuple(dict.fromkeys(predicates))
    if not predicate_names:
        return None
    for predicate in predicate_names:
        if predicate not in PREDICATE_SPECS:
            raise ValueError(f"unsupported binary predicate: {predicate}") from None

    _left_values, left_owned = _coerce_array(left, arg_name="left")
    _right_values, right_owned = _coerce_array(right, arg_name="right")
    if left_owned is None or right_owned is None:
        return None
    aligned = _align_native_expression_owned_pair(left_owned, right_owned)
    if aligned is None:
        return None
    left_owned, right_owned, expression_workload_shape = aligned

    device_values = _fused_gpu_binary_predicates_device(
        predicate_names,
        left_owned,
        right_owned,
    )
    if device_values is not None:
        from vibespatial.api._native_expression import NativeExpression

        expressions = {}
        for predicate, values in device_values.items():
            record_dispatch_event(
                surface="vibespatial.predicates.binary",
                operation=predicate,
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
                implementation="fused_multi_predicate_expression_gpu",
                reason="device-resident DE-9IM multi-predicate expressions",
                detail=(
                    f"operation={operation_prefix}.{predicate}; "
                    f"row_count={left_owned.row_count}; "
                    f"workload_shape={expression_workload_shape}_de9im; "
                    "carrier=NativeExpression"
                ),
            )
            expressions[predicate] = NativeExpression(
                operation=f"{operation_prefix}.{predicate}",
                values=values,
                source_token=source_token,
                source_row_count=left_owned.row_count,
                dtype=str(getattr(values, "dtype", "bool")),
                precision="predicate",
            )
        return expressions

    expressions = {}
    for predicate in predicate_names:
        expression = binary_predicate_expression(
            predicate,
            left_owned,
            right_owned,
            dispatch_mode=dispatch_mode,
            precision=precision,
            source_token=source_token,
            operation=f"{operation_prefix}.{predicate}",
        )
        if expression is None:
            return None
        expressions[predicate] = expression
    return expressions


def _evaluate_binary_predicates_fused_gpu(
    predicates: Sequence[str],
    left: OwnedGeometryArray,
    right: OwnedGeometryArray,
) -> dict[str, np.ndarray] | None:
    """Evaluate multiple DE-9IM predicates with one exact GPU pass.

    This is the public compatibility wrapper for the native multi-predicate
    expression shape.  Device vectors are copied to host only as the terminal
    public bool-mask export.
    """
    predicate_names = tuple(dict.fromkeys(predicates))
    device_outputs = _fused_gpu_binary_predicates_device(predicate_names, left, right)
    if device_outputs is None:
        return None

    import cupy as cp

    h_predicate_results = _runtime_device_to_host(
        cp.stack([device_outputs[predicate] for predicate in predicate_names], axis=0),
        reason="binary predicate fused predicate-results host export",
        terminal_export=True,
    )
    outputs: dict[str, np.ndarray] = {}
    for predicate_index, predicate in enumerate(predicate_names):
        outputs[predicate] = np.asarray(h_predicate_results[predicate_index], dtype=bool)
    detail = (
        f"predicates={','.join(predicate_names)}; "
        f"row_count={left.row_count}; workload_shape=aligned_pairwise_de9im"
    )
    for predicate in predicate_names:
        record_dispatch_event(
            surface="vibespatial.predicates.binary",
            operation=predicate,
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            implementation="fused_multi_predicate_gpu",
            reason="device-resident DE-9IM multi-predicate evaluation",
            detail=detail,
        )
    return outputs


def _evaluate_covered_by_single_polygonal_mask_device(
    left: OwnedGeometryArray,
    mask: OwnedGeometryArray,
) -> object | None:
    """Exact ``covered_by`` for many polygonal rows against one polygonal mask.

    This supports the mask-clip physical shape where a bbox-filtered polygon
    partition is frequently already fully covered by a dissolved polygonal
    mask. The kernel uses the convex-mask proof when legal and otherwise
    falls through to exact polygon DE-9IM on device.  The returned vector is
    device-resident so native clip/overlay consumers can lower it to rowsets
    without a public bool-mask export.
    """
    if mask.row_count != 1:
        return None
    from vibespatial.runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None

    left_state = _ensure_predicate_device_state(
        left,
        reason="covered_by single-mask GPU probe: left geometry",
    )
    mask_state = _ensure_predicate_device_state(
        mask,
        reason="covered_by single-mask GPU probe: mask geometry",
    )

    import cupy as cp

    mask_families = [
        family
        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON)
        if family in mask_state.families
    ]
    if len(mask_families) != 1:
        return None
    mask_family = mask_families[0]

    left_polygon_families = (
        GeometryFamily.POLYGON,
        GeometryFamily.MULTIPOLYGON,
    )
    active_left_families = [
        family for family in left_polygon_families if family in left_state.families
    ]
    if not active_left_families:
        return None
    active_tags = {FAMILY_TAGS[family] for family in active_left_families}
    left_tags = cp.asarray(left_state.tags)
    d_valid = cp.asarray(left_state.validity)
    supported_tag_mask = cp.isin(left_tags, cp.asarray(tuple(active_tags), dtype=cp.int8))
    if (
        left_state.trusted_homogeneous_family in active_left_families
        and left_state.trusted_all_valid is True
    ):
        has_unsupported_valid = False
    elif left._tags is not None and left._validity is not None:
        host_supported_tags = np.isin(
            np.asarray(left._tags, dtype=np.int8),
            np.asarray(tuple(active_tags), dtype=np.int8),
        )
        has_unsupported_valid = bool(
            np.any((~host_supported_tags) & np.asarray(left._validity, dtype=np.bool_))
        )
    else:
        has_unsupported_valid = _runtime_bool_scalar(
            cp.any((~supported_tag_mask) & d_valid),
            reason="binary predicate covered-by single-mask family-domain scalar fence",
        )
    if has_unsupported_valid:
        return None

    from .polygon import compute_polygonal_covered_by_single_mask_no_holes_gpu

    d_out = cp.zeros(left.row_count, dtype=cp.bool_)
    for family in active_left_families:
        tag = FAMILY_TAGS[family]
        d_rows = cp.flatnonzero((left_tags == tag) & d_valid).astype(cp.int32, copy=False)
        if d_rows.size == 0:
            continue
        d_family_result = compute_polygonal_covered_by_single_mask_no_holes_gpu(
            left,
            mask,
            query_family=family,
            mask_family=mask_family,
            d_left=d_rows,
            return_device=True,
        )
        if d_family_result is None:
            return None
        d_out[d_rows] = d_family_result
    return d_out


def _evaluate_covered_by_single_polygonal_mask_gpu(
    left: OwnedGeometryArray,
    mask: OwnedGeometryArray,
) -> np.ndarray | None:
    """Exact ``covered_by`` single-mask probe with a final host bool export."""
    d_out = _evaluate_covered_by_single_polygonal_mask_device(left, mask)
    if d_out is None:
        return None
    return _runtime_device_to_host(
        d_out,
        reason="binary predicate covered-by single-mask result host export",
        terminal_export=True,
    ).astype(bool, copy=False)


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
    right_values, scalar_right, right_owned, workload_shape = _coerce_right(
        right, expected_len=row_count
    )
    requested_mode = (
        dispatch_mode if isinstance(dispatch_mode, ExecutionMode) else ExecutionMode(dispatch_mode)
    )
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

    runtime_selection = _ensure_registered_kernel(
        predicate,
        requested_mode,
        row_count,
        current_residency=combined_residency(left_owned, right_owned),
        workload_shape=workload_shape,
        work_estimate=(
            estimate_pairwise_work_from_owned(
                left_owned,
                right_owned,
                workload=workload_shape,
                output_row_count=row_count,
                primary_unit_name="predicate-pair-coordinate",
            )
            if left_owned is not None and right_owned is not None
            else None
        ),
    )
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
            # Preserve broadcast lineage as a device indexed view.  Keeping
            # the one-row base visible lets reusable metadata (prepared
            # indexes, convex certificates) amortize across every logical row.
            right_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason=f"{predicate} selected GPU execution for broadcast-right geometry",
            )
            right_gpu_owned = _broadcast_right_owned(right_owned, row_count)
        elif not scalar_right:
            right_gpu_owned = _owned_from_values(right_values, owned=right_owned, scalar=False)

        if left_gpu_owned is not None and right_gpu_owned is not None:
            point_pair_result = _evaluate_gpu_point_pair_fast_path(
                predicate,
                left=left_gpu_owned,
                right=right_gpu_owned,
                null_mask=null_mask,
                null_behavior=normalized_null_behavior,
                runtime_selection=runtime_selection,
                precision=precision,
            )
            if point_pair_result is not None:
                record_dispatch_event(
                    surface="vibespatial.predicates.binary",
                    operation=predicate,
                    requested=requested_mode,
                    selected=runtime_selection.selected,
                    implementation="gpu_point_pair_fast_path",
                    reason=runtime_selection.reason,
                    detail=f"workload_shape={workload_shape.value}",
                )
                _record_runtime_selection(runtime_selection, (left_gpu_owned, right_gpu_owned))
                return point_pair_result

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
                    # A nullable object array is required only when nulls are
                    # present and PROPAGATE is requested.  The native fused
                    # path reaches this branch only for an all-valid batch, so
                    # retain its compact boolean host export instead of paying
                    # a second row-shaped object conversion.
                    result_values = np.asarray(fused, dtype=bool)
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
        gpu_dispatch_mode = (
            ExecutionMode.GPU
            if runtime_selection.selected is ExecutionMode.GPU
            else ExecutionMode.CPU
        )
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
        gpu_dispatch_mode = (
            ExecutionMode.GPU
            if runtime_selection.selected is ExecutionMode.GPU
            else ExecutionMode.CPU
        )
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
        empty_mask = (~null_mask) & (
            np.isnan(left_bounds).any(axis=1) | np.isnan(right_bounds).any(axis=1)
        )
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
            if not _gpu_candidate_pairs_supported(
                left_gpu_owned, right_gpu_owned, _cand, predicate
            ):
                if requested_mode is ExecutionMode.GPU:
                    raise NotImplementedError(gpu_reason)
                runtime_selection = _explicit_cpu_fallback_selection(
                    predicate=predicate,
                    requested_mode=requested_mode,
                    row_count=row_count,
                    reason=f"{gpu_reason}; using explicit CPU fallback",
                    workload_shape=workload_shape,
                )
        else:
            # Cannot build owned arrays -- fall back to CPU
            if requested_mode is ExecutionMode.GPU:
                raise NotImplementedError(gpu_reason)
            runtime_selection = _explicit_cpu_fallback_selection(
                predicate=predicate,
                requested_mode=requested_mode,
                row_count=row_count,
                reason=f"{gpu_reason}; using explicit CPU fallback",
                workload_shape=workload_shape,
            )

    _record_runtime_selection(
        runtime_selection, (left_gpu_owned or left_owned, right_gpu_owned or right_owned)
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
                point_device_values = _evaluate_gpu_point_candidates_device(
                    predicate,
                    left_gpu_owned,
                    right_gpu_owned,
                    point_rows,
                )
                if point_device_values is None:
                    point_values = _evaluate_gpu_point_candidates(
                        predicate,
                        left_gpu_owned,
                        right_gpu_owned,
                        point_rows,
                    )
                else:
                    point_values = _runtime_device_to_host(
                        point_device_values,
                        reason=(f"binary predicate point-candidate {predicate} result host export"),
                        terminal_export=True,
                    ).astype(bool, copy=False)
                result[point_rows] = point_values

            if de9im_mask.any():
                de9im_idx = np.flatnonzero(de9im_mask)
                de9im_rows = candidate_rows[de9im_idx]
                de9im_device_values = _evaluate_gpu_de9im_candidates_device(
                    predicate,
                    left_gpu_owned,
                    right_gpu_owned,
                    de9im_rows,
                )
                de9im_values = _runtime_device_to_host(
                    de9im_device_values,
                    reason=(f"binary predicate de9im-candidate {predicate} result host export"),
                    terminal_export=True,
                ).astype(bool, copy=False)
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
            exact_values = getattr(shapely, spec.shapely_op)(
                left_shapely[candidate_rows], scalar_geom, **kwargs
            )
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
            "gpu_binary_predicate"
            if runtime_selection.selected is ExecutionMode.GPU
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
    """Dispatch topological equality through native mutual coverage.

    Scalar right operands are promoted to the same owned broadcast shape used
    by the main binary-predicate stack so strict-native public equality does
    not escape to Shapely solely because the right side is a scalar.
    """
    from vibespatial.geometry.equality import geom_equals_owned
    from vibespatial.runtime import get_requested_mode

    is_scalar = _is_scalar_geometry_operand(right)
    if is_scalar:
        right_geom_type = getattr(right, "geom_type", None)
        if right is not None and right_geom_type not in _OWNED_EXACT_GEOMETRY_TYPES:
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
                reason="scalar right-hand operand uses unsupported owned equality family",
                detail=f"rows={len(left_shapely)}, geom_type={right_geom_type}",
                selected=ExecutionMode.CPU,
            )
            record_fallback_event(
                surface="geopandas.array.equals",
                reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                detail=f"right contains unsupported geometry family {right_geom_type}",
                pipeline="predicate",
                d2h_transfer=isinstance(left, OwnedGeometryArray),
            )
            return result.astype(bool, copy=False)

        _, left_owned = _coerce_owned_exact_values(left, arg_name="left")
        assert left_owned is not None
        right_owned = from_shapely_geometries([right])
        right_owned = _broadcast_right_owned(right_owned, left_owned.row_count)

        result = geom_equals_owned(
            left_owned,
            right_owned,
            dispatch_mode=get_requested_mode(),
        )
        record_dispatch_event(
            surface="geopandas.array.equals",
            operation="equals",
            implementation="geom_equals_owned_broadcast",
            reason="scalar right-hand operand promoted to native mutual coverage",
            detail=f"rows={left_owned.row_count}",
            selected=get_requested_mode(),
        )
        return result.astype(bool, copy=False)

    # Coerce inputs to OwnedGeometryArray.
    _, left_owned = _coerce_owned_exact_values(left, arg_name="left")
    _, right_owned = _coerce_owned_exact_values(right, arg_name="right")
    assert left_owned is not None
    assert right_owned is not None

    dispatch_mode = get_requested_mode()
    result = geom_equals_owned(
        left_owned,
        right_owned,
        dispatch_mode=dispatch_mode,
    )
    record_dispatch_event(
        surface="geopandas.array.equals",
        operation="equals",
        implementation="geom_equals_owned",
        reason="native mutual coverage for topological equality",
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
    is_scalar = _is_scalar_geometry_operand(right)
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
    _, left_owned = _coerce_owned_exact_values(left, arg_name="left")
    _, right_owned = _coerce_owned_exact_values(right, arg_name="right")
    assert left_owned is not None
    assert right_owned is not None

    dispatch_mode = get_requested_mode()
    result = geom_equals_exact_owned(
        left_owned,
        right_owned,
        tolerance,
        dispatch_mode=dispatch_mode,
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
    is_scalar = _is_scalar_geometry_operand(right)
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
    _, left_owned = _coerce_owned_exact_values(left, arg_name="left")
    _, right_owned = _coerce_owned_exact_values(right, arg_name="right")
    assert left_owned is not None
    assert right_owned is not None

    dispatch_mode = get_requested_mode()
    result = geom_equals_identical_owned(
        left_owned,
        right_owned,
        dispatch_mode=dispatch_mode,
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
        # Topological equality routes through mutual native coverage.
        if predicate == "equals":
            unsupported = _unsupported_owned_exact_operands(left, right)
            if unsupported is not None:
                record_fallback_event(
                    surface="geopandas.array.equals",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail=unsupported,
                    pipeline="predicate",
                )
                return None
            return _evaluate_geopandas_equals(left, right, **kwargs)

        # --- equals_exact special path ---
        # Tolerance invalidates the standard bbox coarse filter (two
        # geometries can match within tolerance even when their bboxes
        # don't overlap).  Route directly to the dedicated coordinate-
        # comparison dispatch in geometry/equality.py.
        if predicate == "equals_exact":
            unsupported = _unsupported_owned_exact_operands(left, right)
            if unsupported is not None:
                record_fallback_event(
                    surface="geopandas.array.equals_exact",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail=unsupported,
                    pipeline="predicate",
                )
                return None
            return _evaluate_geopandas_equals_exact(left, right, **kwargs)

        # --- equals_identical special path ---
        # Strict coordinate-level identity (tolerance=0).  Routes through
        # the same NVRTC kernel infrastructure as equals_exact.
        if predicate == "equals_identical":
            unsupported = _unsupported_owned_exact_operands(left, right)
            if unsupported is not None:
                record_fallback_event(
                    surface="geopandas.array.equals_identical",
                    reason="unsupported geometry type for owned equality path (e.g. GeometryCollection)",
                    detail=unsupported,
                    pipeline="predicate",
                )
                return None
            return _evaluate_geopandas_equals_identical(left, right, **kwargs)

        left_coerced = (
            left if isinstance(left, OwnedGeometryArray) else np.asarray(left, dtype=object)
        )
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
    result = evaluate_binary_predicate(
        predicate, left, right, null_behavior=NullBehavior.FALSE, **kwargs
    )
    return {
        "rows": result.row_count,
        "candidate_rows": int(result.candidate_rows.size),
        "coarse_true_rows": int(result.coarse_true_rows.size),
        "coarse_false_rows": int(result.coarse_false_rows.size),
    }
