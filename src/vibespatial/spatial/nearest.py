from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    lower_bound_counting,
    segmented_reduce_min,
    sort_pairs,
    upper_bound,
    upper_bound_counting,
)
from vibespatial.runtime import ExecutionMode, combined_residency
from vibespatial.runtime.adaptive import plan_dispatch_selection

request_warmup(
    [
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "select_i32",
        "select_i64",
        "radix_sort_i32_i32",
        "radix_sort_u64_i32",
        "lower_bound_i32",
        "lower_bound_u64",
        "upper_bound_i32",
        "upper_bound_u64",
        "segmented_reduce_min_f64",
    ]
)
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.geometry.buffers import GeometryFamily  # noqa: E402
from vibespatial.geometry.owned import (  # noqa: E402
    FAMILY_TAGS,
    TAG_FAMILIES,
    FamilyGeometryBuffer,
    OwnedGeometryArray,
    unique_tag_pairs,
)
from vibespatial.kernels.core.geometry_analysis import (  # noqa: E402
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.kernels.core.spatial_query_kernels import (  # noqa: E402
    _grid_nearest_kernels,
    _spatial_query_kernels,
)
from vibespatial.runtime import has_gpu_runtime  # noqa: E402
from vibespatial.runtime.config import SPATIAL_EPSILON  # noqa: E402
from vibespatial.runtime.crossover import PhysicalWorkEstimate  # noqa: E402
from vibespatial.runtime.precision import (  # noqa: E402
    CoordinateStats,
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
    select_precision_plan,
)
from vibespatial.runtime.residency import Residency, TransferTrigger  # noqa: E402
from vibespatial.runtime.robustness import NumericalErrorEnvelope  # noqa: E402

from .query_candidates import (  # noqa: E402
    _generate_candidates_gpu,
    _generate_distance_pairs,
)
from .query_types import (  # noqa: E402
    RegularGridPointIndex,
    _DeviceCandidates,
)
from .query_utils import (  # noqa: E402
    _as_geometry_array,
    _expand_bounds,
    _gpu_bounds_dispatch_mode,
    _to_owned,
    record_shapely_fallback_event,
)

# ---------------------------------------------------------------------------
# GPU nearest-neighbour refinement (Tier 1 NVRTC + Tier 3a CCCL)
# ---------------------------------------------------------------------------
# ADR-0033: distance computation is geometry-specific (Tier 1 NVRTC), segment
# reduction uses CCCL segmented_reduce_min (Tier 3a), and compaction uses CCCL
# select (Tier 3a).  The whole pipeline stays device-resident to avoid the
# device <-> host round-trips that dominated the previous Shapely-based path.


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_FP32_DISTANCE_ERROR_FACTOR = 16.0


@dataclass(frozen=True)
class _NearestMetricPrecisionContext:
    """Resolved plans and fp32 decision envelope for nearest metrics."""

    coarse_plan: PrecisionPlan
    refinement_plan: PrecisionPlan | None
    error_envelope: NumericalErrorEnvelope

    @property
    def fp32_error_bound(self) -> Any:
        """Compatibility alias for distance consumers during carrier rollout."""
        return self.error_envelope.bound

    def refinement_context(self) -> _NearestMetricPrecisionContext:
        if self.refinement_plan is None:
            return self
        return type(self)(
            coarse_plan=self.refinement_plan,
            refinement_plan=None,
            error_envelope=NumericalErrorEnvelope.exact(quantity="distance"),
        )


def _nearest_coordinate_stats_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
) -> CoordinateStats:
    """Reduce cached device bounds to the compact stats PrecisionPlan needs.

    Bounds are native metadata under ADR-0044.  Reducing those four values per
    row avoids scanning authoritative host coordinate buffers.  Only the five
    planning scalars cross to the host, in one observable transfer.
    """
    import cupy as cp

    max_abs_parts = []
    min_x_parts = []
    max_x_parts = []
    min_y_parts = []
    max_y_parts = []
    for owned in (query_owned, tree_owned):
        owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="nearest precision planning consumes device geometry metadata",
        )
        bounds = cp.asarray(compute_geometry_bounds_device(owned)).reshape(-1, 4)
        finite = cp.isfinite(bounds)
        max_abs_parts.append(cp.max(cp.where(finite, cp.abs(bounds), 0.0)))
        min_x_parts.append(cp.min(cp.where(finite[:, 0], bounds[:, 0], cp.inf)))
        max_x_parts.append(cp.max(cp.where(finite[:, 2], bounds[:, 2], -cp.inf)))
        min_y_parts.append(cp.min(cp.where(finite[:, 1], bounds[:, 1], cp.inf)))
        max_y_parts.append(cp.max(cp.where(finite[:, 3], bounds[:, 3], -cp.inf)))

    packed = cp.stack(
        (
            cp.max(cp.stack(max_abs_parts)),
            cp.min(cp.stack(min_x_parts)),
            cp.max(cp.stack(max_x_parts)),
            cp.min(cp.stack(min_y_parts)),
            cp.max(cp.stack(max_y_parts)),
        )
    )
    host_stats = get_cuda_runtime().copy_device_to_host(
        packed,
        reason="nearest precision device-coordinate-stats planning export",
    )
    max_abs, min_x, max_x, min_y, max_y = (float(value) for value in host_stats)
    if not all(np.isfinite(value) for value in (min_x, max_x, min_y, max_y)):
        return CoordinateStats()
    return CoordinateStats(
        max_abs_coord=max_abs,
        span=max(0.0, max_x - min_x, max_y - min_y),
    )


def _plan_nearest_metric_precision(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    pair_count: int,
) -> _NearestMetricPrecisionContext:
    """Resolve coarse and selective-refinement plans for candidate distances."""
    coordinate_stats = _nearest_coordinate_stats_device(query_owned, tree_owned)
    adaptive_plan = plan_dispatch_selection(
        kernel_name="nearest_point_family_distance",
        kernel_class=KernelClass.METRIC,
        row_count=query_owned.row_count,
        requested_mode=ExecutionMode.GPU,
        current_residency=combined_residency(query_owned, tree_owned),
        coordinate_stats=coordinate_stats,
        gpu_available=True,
        work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
            row_count=query_owned.row_count,
            candidate_pair_count=pair_count,
            primary_unit_name="nearest-distance-candidate-pair",
        ),
    )
    coarse_plan = adaptive_plan.precision_plan
    if coarse_plan.compute_precision is not PrecisionMode.FP32:
        return _NearestMetricPrecisionContext(
            coarse_plan,
            None,
            NumericalErrorEnvelope.exact(quantity="distance"),
        )

    refinement_plan = select_precision_plan(
        runtime_selection=adaptive_plan.runtime_selection,
        kernel_class=KernelClass.METRIC,
        requested=PrecisionMode.FP64,
        coordinate_stats=coordinate_stats,
        device_profile=adaptive_plan.device_profile,
    )
    # Centered fp32 perturbs each coordinate by at most roughly one fp32 ulp.
    # Distance is Lipschitz in both operands; the factor also covers the
    # projection arithmetic in point-to-segment kernels.  Candidates within
    # twice this bound of the coarse minimum are refined below.
    error_bound = (
        _FP32_DISTANCE_ERROR_FACTOR
        * float(np.finfo(np.float32).eps)
        * max(coordinate_stats.span, 1.0)
    )
    return _NearestMetricPrecisionContext(
        coarse_plan,
        refinement_plan,
        NumericalErrorEnvelope(
            bound=error_bound,
            quantity="distance",
            arithmetic_precision=PrecisionMode.FP32,
            derivation="centered fp32 coordinate ulps plus point-segment projection arithmetic",
        ),
    )


def _plan_device_resident_metric_precision(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    pair_count: int,
) -> _NearestMetricPrecisionContext:
    """Resolve metric compute precision without crossing the device boundary.

    This is for strict device-return pipelines whose ordering or threshold
    decisions must remain device-resident.  Point-distance kernels center fp32
    coordinates unconditionally, so host-visible coordinate statistics are not
    needed to execute the selected plan safely; the decision envelope below is
    itself a device scalar.
    """
    adaptive_plan = plan_dispatch_selection(
        kernel_name="device_resident_point_family_distance",
        kernel_class=KernelClass.METRIC,
        row_count=query_owned.row_count,
        requested_mode=ExecutionMode.GPU,
        current_residency=combined_residency(query_owned, tree_owned),
        gpu_available=True,
        work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
            row_count=query_owned.row_count,
            candidate_pair_count=pair_count,
            primary_unit_name="device-distance-candidate-pair",
        ),
    )
    coarse_plan = adaptive_plan.precision_plan
    if coarse_plan.compute_precision is not PrecisionMode.FP32:
        return _NearestMetricPrecisionContext(
            coarse_plan,
            None,
            NumericalErrorEnvelope.exact(quantity="distance"),
        )

    refinement_plan = select_precision_plan(
        runtime_selection=adaptive_plan.runtime_selection,
        kernel_class=KernelClass.METRIC,
        requested=PrecisionMode.FP64,
        device_profile=adaptive_plan.device_profile,
    )
    return _NearestMetricPrecisionContext(
        coarse_plan=coarse_plan,
        refinement_plan=refinement_plan,
        error_envelope=NumericalErrorEnvelope(
            bound=_nearest_fp32_error_bound_device(query_owned, tree_owned),
            quantity="distance",
            arithmetic_precision=PrecisionMode.FP32,
            derivation="device-resident centered fp32 extent envelope",
        ),
    )


def _nearest_fp32_error_bound_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
):
    """Return the centered-fp32 distance envelope as a device scalar."""
    import cupy as cp

    min_x_parts = []
    max_x_parts = []
    min_y_parts = []
    max_y_parts = []
    for owned in (query_owned, tree_owned):
        bounds = cp.asarray(compute_geometry_bounds_device(owned)).reshape(-1, 4)
        finite = cp.isfinite(bounds)
        min_x_parts.append(cp.min(cp.where(finite[:, 0], bounds[:, 0], cp.inf)))
        max_x_parts.append(cp.max(cp.where(finite[:, 2], bounds[:, 2], -cp.inf)))
        min_y_parts.append(cp.min(cp.where(finite[:, 1], bounds[:, 1], cp.inf)))
        max_y_parts.append(cp.max(cp.where(finite[:, 3], bounds[:, 3], -cp.inf)))
    span_x = cp.max(cp.stack(max_x_parts)) - cp.min(cp.stack(min_x_parts))
    span_y = cp.max(cp.stack(max_y_parts)) - cp.min(cp.stack(min_y_parts))
    span = cp.maximum(cp.maximum(span_x, span_y), 1.0)
    return _FP32_DISTANCE_ERROR_FACTOR * float(np.finfo(np.float32).eps) * span


def _nearest_ambiguity_mask_host(
    left_idx: np.ndarray,
    distances: np.ndarray,
    n_queries: int,
    *,
    max_distance: float,
    error_bound: float,
) -> np.ndarray:
    """Select coarse pairs that can alter nearest ordering, ties, or bounds."""
    finite = np.isfinite(distances)
    min_distance = np.full(n_queries, np.inf, dtype=np.float64)
    np.minimum.at(min_distance, left_idx, np.where(finite, distances, np.inf))
    pair_min = min_distance[left_idx]
    tie_tolerance = 1e-8 + 1e-5 * np.abs(pair_min)
    ordering_ambiguous = distances <= pair_min + (2.0 * error_bound) + tie_tolerance
    threshold_ambiguous = (
        np.abs(distances - max_distance) <= error_bound
        if np.isfinite(max_distance)
        else np.zeros(distances.shape, dtype=np.bool_)
    )
    return ~finite | ordering_ambiguous | threshold_ambiguous


def _refine_ambiguous_point_family_distances(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    n_queries: int,
    strategy: DistanceStrategy,
    precision_context: _NearestMetricPrecisionContext,
    *,
    max_distance: float,
    exclusive: bool,
    center_device=None,
) -> None:
    """Recompute an ambiguity prefix at source-pair capacity without a count fence."""
    if precision_context.refinement_plan is None or pair_count == 0:
        return

    import cupy as cp

    seg_starts = lower_bound_counting(d_left, 0, n_queries, dtype=np.int32).astype(
        cp.int32, copy=False
    )
    seg_ends = upper_bound_counting(d_left, 0, n_queries, dtype=np.int32).astype(
        cp.int32, copy=False
    )
    d_finite_distances = cp.where(cp.isfinite(d_distances), d_distances, cp.inf)
    coarse_min = segmented_reduce_min(
        d_finite_distances,
        seg_starts,
        seg_ends,
        num_segments=n_queries,
    ).values
    pair_min = coarse_min[d_left]
    tie_tolerance = 1e-8 + 1e-5 * cp.abs(pair_min)
    error_bound = precision_context.fp32_error_bound
    ordering_ambiguous = d_distances <= pair_min + (2.0 * error_bound) + tie_tolerance
    threshold_ambiguous = (
        cp.abs(d_distances - max_distance) <= error_bound
        if np.isfinite(max_distance)
        else cp.zeros(pair_count, dtype=cp.bool_)
    )
    from vibespatial.api._native_rowset import NativeDeviceSelection

    ambiguity_selection = NativeDeviceSelection.from_mask(
        ~cp.isfinite(d_distances) | ordering_ambiguous | threshold_ambiguous,
        source_row_count=pair_count,
    )
    d_refine_left = ambiguity_selection.gather_capacity(d_left, fill_value=0).astype(
        cp.int32,
        copy=False,
    )
    d_refine_right = ambiguity_selection.gather_capacity(d_right, fill_value=0).astype(
        cp.int32,
        copy=False,
    )

    runtime = get_cuda_runtime()
    d_refined = runtime.allocate((ambiguity_selection.capacity,), np.float64)
    try:
        ok = strategy.compute(
            query_owned,
            tree_owned,
            d_refine_left,
            d_refine_right,
            d_refined,
            ambiguity_selection.capacity,
            exclusive=exclusive,
            precision_plan=precision_context.refinement_plan,
            logical_count=ambiguity_selection.logical_count,
            center_device=center_device,
        )
        if ok:
            d_partition = ambiguity_selection.partition_capacity_positions()
            d_distances[d_partition] = cp.where(
                ambiguity_selection.active_capacity_mask(),
                cp.asarray(d_refined),
                d_distances[d_partition],
            )
    finally:
        runtime.free(d_refined)


def _empty_nearest_result(return_distance: bool):
    """Return a canonical empty nearest result."""
    empty = np.empty((2, 0), dtype=np.intp)
    if return_distance:
        return empty, np.asarray([], dtype=np.float64)
    return empty


def _empty_nearest_result_device(return_distance: bool):
    """Return a canonical empty nearest result backed by device arrays."""
    import cupy as cp

    empty_i = cp.empty(0, dtype=cp.int32)
    if return_distance:
        return (empty_i, empty_i), cp.empty(0, dtype=cp.float64)
    return empty_i, empty_i


def _points_only(owned: OwnedGeometryArray) -> bool:
    if owned._validity is None or owned._tags is None:
        return frozenset(owned.families) <= {GeometryFamily.POINT}
    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    valid = owned.validity
    return bool((not valid.any()) or np.all(owned.tags[valid] == point_tag))


def _device_dense_point_coords(
    owned: OwnedGeometryArray,
):
    import cupy as cp

    state = owned._ensure_device_state()
    points = state.families[GeometryFamily.POINT]
    dense_x = cp.full(owned.row_count, cp.nan, dtype=cp.float64)
    dense_y = cp.full(owned.row_count, cp.nan, dtype=cp.float64)
    valid_mask = state.validity.astype(cp.bool_) & (state.tags == FAMILY_TAGS[GeometryFamily.POINT])
    if int(valid_mask.sum()) == 0:
        return dense_x, dense_y

    global_rows = cp.flatnonzero(valid_mask)
    family_rows = state.family_row_offsets[global_rows]
    non_empty = ~points.empty_mask[family_rows].astype(cp.bool_, copy=False)
    if int(non_empty.sum()) == 0:
        return dense_x, dense_y

    active_rows = global_rows[non_empty]
    active_family_rows = family_rows[non_empty]
    coord_idx = points.geometry_offsets[active_family_rows]
    dense_x[active_rows] = points.x[coord_idx]
    dense_y[active_rows] = points.y[coord_idx]
    return dense_x, dense_y


def _detect_regular_grid_point_index(owned: OwnedGeometryArray) -> RegularGridPointIndex | None:
    point_buffer = owned.families.get(GeometryFamily.POINT)
    if point_buffer is None or len(owned.families) != 1:
        return None
    if owned.row_count == 0 or not np.all(owned.validity) or np.any(point_buffer.empty_mask):
        return None
    if not np.array_equal(
        point_buffer.geometry_offsets, np.arange(owned.row_count + 1, dtype=np.int32)
    ):
        return None

    xs = point_buffer.x
    ys = point_buffer.y
    if xs.size != owned.row_count or ys.size != owned.row_count:
        return None
    if np.isnan(xs).any() or np.isnan(ys).any():
        return None

    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    cols = int(unique_x.size)
    rows = int(unique_y.size)
    if cols <= 0 or rows <= 0:
        return None
    if cols == 1 and rows == 1:
        return RegularGridPointIndex(
            origin_x=float(xs[0]),
            origin_y=float(ys[0]),
            cell_width=1.0,
            cell_height=1.0,
            cols=1,
            rows=1,
            size=owned.row_count,
        )

    cell_width = float(unique_x[1] - unique_x[0]) if cols > 1 else 1.0
    cell_height = float(unique_y[1] - unique_y[0]) if rows > 1 else 1.0
    if cell_width <= 0.0 or cell_height <= 0.0:
        return None
    tol = 1e-9 * max(abs(cell_width), abs(cell_height), 1.0)
    if cols > 1 and not np.allclose(np.diff(unique_x), cell_width, atol=tol, rtol=0.0):
        return None
    if rows > 1 and not np.allclose(np.diff(unique_y), cell_height, atol=tol, rtol=0.0):
        return None

    expected_x = (
        float(unique_x[0]) + (np.arange(owned.row_count, dtype=np.float64) % cols) * cell_width
    )
    expected_y = (
        float(unique_y[0]) + (np.arange(owned.row_count, dtype=np.float64) // cols) * cell_height
    )
    if not np.allclose(xs, expected_x, atol=tol, rtol=0.0):
        return None
    if not np.allclose(ys, expected_y, atol=tol, rtol=0.0):
        return None

    return RegularGridPointIndex(
        origin_x=float(unique_x[0]),
        origin_y=float(unique_y[0]),
        cell_width=cell_width,
        cell_height=cell_height,
        cols=cols,
        rows=rows,
        size=owned.row_count,
    )


def _single_family(owned: OwnedGeometryArray) -> GeometryFamily | None:
    """Return the single geometry family if all valid rows share one, else None."""
    if owned._validity is None or owned._tags is None:
        families = tuple(owned.families)
        return families[0] if len(families) == 1 else None
    valid = owned.validity
    if not valid.any():
        return None
    valid_tags = owned.tags[valid]
    unique_tags = np.unique(valid_tags)
    if len(unique_tags) != 1:
        return None
    return TAG_FAMILIES.get(int(unique_tags[0]))


def _tree_distance_family(tree_owned: OwnedGeometryArray) -> GeometryFamily | None:
    """Return the single non-point geometry family in *tree_owned*, or None.

    Used to dispatch to point-distance kernels when the tree contains a
    single family type (linestring, polygon, etc.) that is supported by
    ``point_distance.compute_point_distance_gpu()``.
    """
    from .point_distance import supported_point_distance_families

    family = _single_family(tree_owned)
    if family is not None and family in supported_point_distance_families():
        return family
    return None


def _point_distance_families() -> frozenset:
    from .point_distance import supported_point_distance_families

    return supported_point_distance_families()


def _supports_device_nearest_refinement(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
) -> bool:
    """Return whether nearest exact refinement can stay device-resident."""
    if _points_only(query_owned) and _points_only(tree_owned):
        return True
    if _points_only(query_owned):
        return _tree_distance_family(tree_owned) is not None
    if _points_only(tree_owned):
        query_family = _single_family(query_owned)
        return query_family is not None and query_family in _point_distance_families()
    return _single_family(query_owned) is not None and _single_family(tree_owned) is not None


def _make_point_owned_from_coords(x: np.ndarray, y: np.ndarray) -> OwnedGeometryArray:
    """Build a lightweight point OwnedGeometryArray from raw coordinate arrays.

    Each coordinate becomes a separate point row.  No Shapely objects are
    created -- this constructs the owned buffers directly.
    """
    from vibespatial.geometry.buffers import POINT_SCHEMA

    n = len(x)
    point_tag = FAMILY_TAGS[GeometryFamily.POINT]
    validity = np.ones(n, dtype=bool)
    tags = np.full(n, point_tag, dtype=np.int8)
    family_row_offsets = np.arange(n, dtype=np.int32)
    point_buffer = FamilyGeometryBuffer(
        family=GeometryFamily.POINT,
        schema=POINT_SCHEMA,
        row_count=n,
        x=np.ascontiguousarray(x, dtype=np.float64),
        y=np.ascontiguousarray(y, dtype=np.float64),
        geometry_offsets=np.arange(n + 1, dtype=np.int32),
        empty_mask=np.zeros(n, dtype=bool),
    )
    return OwnedGeometryArray(
        validity=validity,
        tags=tags,
        family_row_offsets=family_row_offsets,
        families={GeometryFamily.POINT: point_buffer},
    )


# ---------------------------------------------------------------------------
# Shared GPU kernel launch helpers (eliminate point-point distance duplication)
# ---------------------------------------------------------------------------


def _launch_point_point_distance_kernel(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    *,
    exclusive: bool = False,
):
    """Launch the point-point distance pairs kernel (Tier 1 NVRTC).

    Shared by the point-point path in ``_compute_pair_distances_gpu``,
    ``_compute_multipoint_distances_gpu``, and ``_compute_mixed_distances_gpu``.
    Both *query_owned* and *tree_owned* must already be device-resident with
    a POINT family.
    """
    point_family = GeometryFamily.POINT
    runtime = get_cuda_runtime()
    ptr = runtime.pointer
    kernels = _spatial_query_kernels()

    # Pair kernels consume row indirection through family_row_offsets. Keep
    # repeated point candidates as compact indexed views; resolving them here
    # gives a logical_rows * source_coordinates allocation shape (multi-TiB for
    # a few million repeated SpatialBench representatives).
    qs = query_owned._ensure_device_state(preserve_indexed_view=True)
    ts = tree_owned._ensure_device_state(preserve_indexed_view=True)
    qp = qs.families[point_family]
    tp = ts.families[point_family]

    dist_params = (
        (
            ptr(qs.validity),
            ptr(qs.tags),
            ptr(qs.family_row_offsets),
            ptr(qp.geometry_offsets),
            ptr(qp.empty_mask),
            ptr(qp.x),
            ptr(qp.y),
            FAMILY_TAGS[point_family],
            ptr(ts.validity),
            ptr(ts.tags),
            ptr(ts.family_row_offsets),
            ptr(tp.geometry_offsets),
            ptr(tp.empty_mask),
            ptr(tp.x),
            ptr(tp.y),
            FAMILY_TAGS[point_family],
            ptr(d_left),
            ptr(d_right),
            ptr(d_distances),
            1 if exclusive else 0,
            pair_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    dist_grid, dist_block = runtime.launch_config(
        kernels["point_point_distance_pairs_from_owned"], pair_count
    )
    runtime.launch(
        kernels["point_point_distance_pairs_from_owned"],
        grid=dist_grid,
        block=dist_block,
        params=dist_params,
    )


# ---------------------------------------------------------------------------
# Shared nearest refinement pipeline (eliminate 3x duplication)
# ---------------------------------------------------------------------------


def _refine_nearest_from_device_distances(
    d_left,
    d_right,
    d_distances,
    pair_count: int,
    n_queries: int,
    *,
    max_distance: float,
    return_all: bool,
    return_distance: bool,
    return_device: bool = False,
) -> tuple[Any, bool]:
    """Shared segment-reduce + keep-mask + compact pipeline.

    Takes device arrays of sorted (left, right) pairs and computed distances,
    and produces the final nearest result.  Used by the point-point,
    point-family, and segment-family refinement paths.

    Returns ``(result, None)`` where *result* is the nearest indices (and
    optional distances).
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    kernels = _spatial_query_kernels()
    ptr = runtime.pointer

    # Build segments from sorted left_idx (Tier 3a CCCL).
    seg_starts = lower_bound_counting(d_left, 0, n_queries, dtype=np.int32)
    seg_ends = upper_bound_counting(d_left, 0, n_queries, dtype=np.int32)
    seg_starts_i32 = seg_starts.astype(cp.int32, copy=False)
    seg_ends_i32 = seg_ends.astype(cp.int32, copy=False)

    # Segmented min-distance per query (Tier 3a CCCL).
    min_result = segmented_reduce_min(
        d_distances,
        seg_starts_i32,
        seg_ends_i32,
        num_segments=n_queries,
    )
    d_min_distances = min_result.values

    # Build keep mask (Tier 1 NVRTC).
    d_keep = runtime.allocate((pair_count,), np.uint8)
    keep_params = (
        (
            ptr(d_distances),
            ptr(d_min_distances),
            ptr(d_left),
            ptr(d_keep),
            float(max_distance),
            pair_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    keep_grid, keep_block = runtime.launch_config(kernels["nearest_keep_mask"], pair_count)
    runtime.launch(
        kernels["nearest_keep_mask"],
        grid=keep_grid,
        block=keep_block,
        params=keep_params,
    )

    # (return_all=False) Keep only first match per segment.
    if not return_all:
        d_first = runtime.from_host(np.zeros(pair_count, dtype=np.uint8))
        seg_grid = max(1, (n_queries + 255) // 256)
        first_params = (
            (
                ptr(d_keep),
                ptr(d_first),
                ptr(seg_starts_i32),
                ptr(seg_ends_i32),
                n_queries,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        seg_grid, seg_block = runtime.launch_config(kernels["nearest_first_per_segment"], n_queries)
        runtime.launch(
            kernels["nearest_first_per_segment"],
            grid=seg_grid,
            block=seg_block,
            params=first_params,
        )
        d_keep = d_first

    # Compact kept indices (Tier 3a CCCL).
    compacted = compact_indices(d_keep)
    if compacted.count == 0:
        if return_device:
            return _empty_nearest_result_device(return_distance), False
        return _empty_nearest_result(return_distance), False

    # Gather results on device. Private NativeRelation consumers can keep the
    # relation and fp64 distances device-resident; public callers export below.
    kept_idx = compacted.values
    if return_device:
        d_out_left = d_left[kept_idx].astype(cp.int32, copy=False)
        d_out_right = d_right[kept_idx].astype(cp.int32, copy=False)
        if return_distance:
            return (
                (d_out_left, d_out_right),
                d_distances[kept_idx].astype(cp.float64, copy=False),
            ), False
        return (d_out_left, d_out_right), False

    h_left = runtime.copy_device_to_host(
        d_left[kept_idx],
        reason="nearest refined left-index host export",
    ).astype(np.intp, copy=False)
    h_right = runtime.copy_device_to_host(
        d_right[kept_idx],
        reason="nearest refined right-index host export",
    ).astype(np.intp, copy=False)
    indices = np.vstack((h_left, h_right))

    if return_distance:
        h_dist = runtime.copy_device_to_host(
            d_distances[kept_idx],
            reason="nearest refined distance host export",
        )
        return (indices, h_dist), False
    return indices, False


# ---------------------------------------------------------------------------
# Distance strategy classes
# ---------------------------------------------------------------------------


class DistanceStrategy(ABC):
    """Base class for GPU distance computation strategies.

    Each subclass knows how to compute pairwise distances for a specific
    combination of query/tree geometry families.
    """

    @abstractmethod
    def compute(
        self,
        query_owned: OwnedGeometryArray,
        tree_owned: OwnedGeometryArray,
        d_left,
        d_right,
        d_distances,
        pair_count: int,
        *,
        exclusive: bool = False,
        precision_plan: PrecisionPlan | None = None,
        logical_count=None,
        center_device=None,
    ) -> bool:
        """Compute distances for candidate pairs on GPU.

        Writes results into *d_distances*.  Returns True on success,
        False if the family combination is not supported.
        """
        ...

    def move_to_device(
        self,
        query_owned: OwnedGeometryArray,
        tree_owned: OwnedGeometryArray,
        *,
        query_reason: str,
        tree_reason: str,
    ):
        """Move both geometry arrays to device."""
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=query_reason,
        )
        tree_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=tree_reason,
        )


class PointPointDistanceStrategy(DistanceStrategy):
    """Compute point-to-point distances via the inline NVRTC kernel."""

    def compute(
        self,
        query_owned: OwnedGeometryArray,
        tree_owned: OwnedGeometryArray,
        d_left,
        d_right,
        d_distances,
        pair_count: int,
        *,
        exclusive: bool = False,
        precision_plan: PrecisionPlan | None = None,
        logical_count=None,
        center_device=None,
    ) -> bool:
        self.move_to_device(
            query_owned,
            tree_owned,
            query_reason="nearest: GPU point-point distance for query",
            tree_reason="nearest: GPU point-point distance for tree",
        )
        _launch_point_point_distance_kernel(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
            exclusive=exclusive,
        )
        return True


class PointFamilyDistanceStrategy(DistanceStrategy):
    """Compute point-to-{line, polygon, ...} distances via point_distance kernels."""

    def __init__(self, tree_family: GeometryFamily):
        self.tree_family = tree_family

    def compute(
        self,
        query_owned: OwnedGeometryArray,
        tree_owned: OwnedGeometryArray,
        d_left,
        d_right,
        d_distances,
        pair_count: int,
        *,
        exclusive: bool = False,
        precision_plan: PrecisionPlan | None = None,
        logical_count=None,
        center_device=None,
    ) -> bool:
        from .point_distance import compute_point_distance_gpu

        self.move_to_device(
            query_owned,
            tree_owned,
            query_reason="nearest: GPU point-distance refinement for query points",
            tree_reason=f"nearest: GPU point-distance refinement for tree {self.tree_family.name}",
        )
        if precision_plan is None:
            precision_plan = _plan_nearest_metric_precision(
                query_owned,
                tree_owned,
                pair_count,
            ).coarse_plan
        return compute_point_distance_gpu(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
            tree_family=self.tree_family,
            exclusive=exclusive,
            compute_precision=precision_plan.compute_precision,
            logical_count=logical_count,
            center_device=center_device,
        )


class SegmentFamilyDistanceStrategy(DistanceStrategy):
    """Compute non-point-to-non-point distances via segment_distance kernels."""

    def __init__(self, query_family: GeometryFamily, tree_family: GeometryFamily):
        self.query_family = query_family
        self.tree_family = tree_family

    def compute(
        self,
        query_owned: OwnedGeometryArray,
        tree_owned: OwnedGeometryArray,
        d_left,
        d_right,
        d_distances,
        pair_count: int,
        *,
        exclusive: bool = False,
        precision_plan: PrecisionPlan | None = None,
        logical_count=None,
        center_device=None,
    ) -> bool:
        from .segment_distance import compute_segment_distance_gpu

        self.move_to_device(
            query_owned,
            tree_owned,
            query_reason=f"nearest: GPU segment-distance refinement for query {self.query_family.name}",
            tree_reason=f"nearest: GPU segment-distance refinement for tree {self.tree_family.name}",
        )
        return compute_segment_distance_gpu(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
            query_family=self.query_family,
            tree_family=self.tree_family,
            exclusive=exclusive,
        )


# ---------------------------------------------------------------------------
# Unified typed nearest refinement (replaces three near-identical functions)
# ---------------------------------------------------------------------------


def _nearest_refine_gpu_typed(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    n_queries: int,
    strategy: DistanceStrategy,
    *,
    max_distance: float,
    return_all: bool = True,
    exclusive: bool = False,
    return_distance: bool = False,
    return_device: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray | None], bool] | None:
    """GPU nearest refinement for a known geometry family combination.

    Uses the provided *strategy* to compute distances, then runs the shared
    segment-reduce + keep-mask + compact pipeline.
    Returns ``(indices_2xN, distances_or_None)`` on success, or ``None``.
    """
    pair_count = left_idx.size
    runtime = get_cuda_runtime()

    d_left = runtime.from_host(np.ascontiguousarray(left_idx, dtype=np.int32))
    d_right = runtime.from_host(np.ascontiguousarray(right_idx, dtype=np.int32))
    d_distances = runtime.allocate((pair_count,), np.float64)

    try:
        # Sort pairs by left_idx for segment construction.
        sorted_result = sort_pairs(d_left, d_right, synchronize=False)
        d_left = sorted_result.keys
        d_right = sorted_result.values

        precision_context = (
            (
                _plan_device_resident_metric_precision(
                    query_owned,
                    tree_owned,
                    pair_count,
                )
                if return_device
                else _plan_nearest_metric_precision(query_owned, tree_owned, pair_count)
            )
            if isinstance(strategy, PointFamilyDistanceStrategy)
            else None
        )
        center_device = None
        if isinstance(strategy, PointFamilyDistanceStrategy):
            from vibespatial.spatial.point_distance import compute_distance_center_device

            center_device = compute_distance_center_device(query_owned, tree_owned)

        # Compute distances using the strategy and resolved PrecisionPlan.
        ok = strategy.compute(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
            exclusive=exclusive,
            precision_plan=(
                precision_context.coarse_plan if precision_context is not None else None
            ),
            center_device=center_device,
        )
        if not ok:
            return None

        if precision_context is not None:
            _refine_ambiguous_point_family_distances(
                query_owned,
                tree_owned,
                d_left,
                d_right,
                d_distances,
                pair_count,
                n_queries,
                strategy,
                precision_context,
                max_distance=max_distance,
                exclusive=exclusive,
                center_device=center_device,
            )

        # Run shared refinement pipeline.
        return _refine_nearest_from_device_distances(
            d_left,
            d_right,
            d_distances,
            pair_count,
            n_queries,
            max_distance=max_distance,
            return_all=return_all,
            return_distance=return_distance,
            return_device=return_device,
        )
    finally:
        runtime.free(d_left)
        runtime.free(d_right)
        runtime.free(d_distances)


# ---------------------------------------------------------------------------
# GPU candidate generation
# ---------------------------------------------------------------------------


def _generate_point_nearest_candidates_regular_grid_gpu(
    query_owned: OwnedGeometryArray,
    tree_index: RegularGridPointIndex,
    *,
    max_distance: float,
    exclusive: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    import cupy as cp

    runtime = get_cuda_runtime()
    query_x, query_y = _device_dense_point_coords(query_owned)
    counts = cp.empty(query_owned.row_count, dtype=cp.int32)
    offsets = None
    out_left = None
    out_right = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer
        count_params = (
            (
                ptr(query_x),
                ptr(query_y),
                tree_index.origin_x,
                tree_index.origin_y,
                tree_index.cell_width,
                tree_index.cell_height,
                tree_index.cols,
                tree_index.rows,
                tree_index.size,
                float(max_distance),
                1 if exclusive else 0,
                ptr(counts),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(
            kernels["point_regular_grid_nearest_count"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_regular_grid_nearest_count"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = (
            count_scatter_total(
                runtime,
                counts,
                offsets,
                reason="nearest regular-grid point-pair allocation fence",
            )
            if query_owned.row_count > 0
            else 0
        )
        if total_pairs == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty

        out_left = runtime.allocate((total_pairs,), np.int32)
        out_right = runtime.allocate((total_pairs,), np.int32)
        scatter_params = (
            (
                ptr(query_x),
                ptr(query_y),
                tree_index.origin_x,
                tree_index.origin_y,
                tree_index.cell_width,
                tree_index.cell_height,
                tree_index.cols,
                tree_index.rows,
                tree_index.size,
                ptr(offsets),
                float(max_distance),
                1 if exclusive else 0,
                ptr(out_left),
                ptr(out_right),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(
            kernels["point_regular_grid_nearest_scatter"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_regular_grid_nearest_scatter"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        runtime.synchronize()

        left = runtime.copy_device_to_host(
            out_left,
            reason="nearest regular-grid point left-index host export",
        ).astype(np.int32, copy=False)
        right = runtime.copy_device_to_host(
            out_right,
            reason="nearest regular-grid point right-index host export",
        ).astype(np.int32, copy=False)
        return left, right
    finally:
        runtime.free(counts)
        runtime.free(offsets)
        runtime.free(out_left)
        runtime.free(out_right)


# ---------------------------------------------------------------------------
# Zero-copy GPU grid-based nearest neighbour (bypasses _to_owned entirely)
# ---------------------------------------------------------------------------


def _extract_point_coords_for_nearest(
    geom_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Extract dense point coordinates from a Shapely geometry array.

    Returns ``(dense_x, dense_y, global_idx, n_total)`` where the dense
    arrays contain only valid non-empty points and *global_idx* maps back
    to original row indices.  Returns ``None`` if any non-Point geometry
    is present.
    """
    type_ids = shapely.get_type_id(geom_array)
    missing = shapely.is_missing(geom_array)
    empty = shapely.is_empty(geom_array)
    valid = ~missing & ~empty
    # Check all non-missing geometries are Points (type_id == 0).
    non_missing = ~missing
    if non_missing.any() and not np.all(type_ids[non_missing] == 0):
        return None
    if not valid.any():
        return None

    coords = shapely.get_coordinates(geom_array[valid])
    dense_x = np.ascontiguousarray(coords[:, 0])
    dense_y = np.ascontiguousarray(coords[:, 1])
    global_idx = np.flatnonzero(valid).astype(np.intp)
    return dense_x, dense_y, global_idx, len(geom_array)


def _nearest_grid_gpu(
    tree_geometries: np.ndarray,
    query_values: np.ndarray,
    *,
    return_all: bool,
    return_distance: bool,
    exclusive: bool,
    max_distance: float | None,
) -> tuple[Any, str] | None:
    """Zero-copy GPU grid nearest-neighbour for point-point data.

    Extracts coordinates directly from Shapely arrays, builds a uniform
    grid spatial hash on device, and runs ring-expansion search entirely
    on the GPU.  Returns ``None`` to fall through to existing paths for
    non-point or empty inputs.
    """
    if not has_gpu_runtime():
        return None

    tree_data = _extract_point_coords_for_nearest(tree_geometries)
    if tree_data is None:
        return None
    query_data = _extract_point_coords_for_nearest(query_values)
    if query_data is None:
        return None

    tree_x_h, tree_y_h, tree_global_idx, _n_tree_total = tree_data
    query_x_h, query_y_h, query_global_idx, n_query_total = query_data

    n_tree = len(tree_x_h)
    n_query = len(query_x_h)
    if n_tree == 0 or n_query == 0:
        result = _empty_nearest_result(return_distance)
        return result, "owned_gpu_nearest"

    import math

    import cupy as cp

    runtime = get_cuda_runtime()
    ptr = runtime.pointer

    # Guard: int32 kernel parameters cannot exceed 2^31-1.
    if n_tree > np.iinfo(np.int32).max or n_query > np.iinfo(np.int32).max:
        return None

    # --- Upload tree coords to device ---
    d_tree_x = runtime.from_host(tree_x_h)
    d_tree_y = runtime.from_host(tree_y_h)
    d_tree_global_idx = runtime.from_host(tree_global_idx.astype(np.int32))

    # --- Grid build: compute bbox from host arrays (avoids 4 D2H syncs) ---
    min_x = float(tree_x_h.min())
    max_x = float(tree_x_h.max())
    min_y = float(tree_y_h.min())
    max_y = float(tree_y_h.max())
    extent_x = max_x - min_x
    extent_y = max_y - min_y
    extent = max(extent_x, extent_y, SPATIAL_EPSILON)

    cell_size = extent / max(1.0, math.ceil(math.sqrt(n_tree)))
    # Floor: cap grid at 4096 x 4096 = ~16M cells
    cell_size = max(cell_size, extent / 4096.0)
    # Ensure cell_size is positive
    cell_size = max(cell_size, SPATIAL_EPSILON)

    origin_x = min_x - cell_size * 0.5
    origin_y = min_y - cell_size * 0.5
    n_cols = max(1, int(math.ceil((max_x - origin_x) / cell_size)) + 1)
    n_rows = max(1, int(math.ceil((max_y - origin_y) / cell_size)) + 1)
    n_cells = n_cols * n_rows

    # Cap: if n_cells is enormous due to degenerate data, fall through
    if n_cells > 16_777_216:  # 16M cells max
        runtime.free(d_tree_x)
        runtime.free(d_tree_y)
        runtime.free(d_tree_global_idx)
        return None

    d_cell_ids = None
    d_sorted_tree_x = None
    d_sorted_tree_y = None
    d_sorted_global_idx = None
    d_cell_start = None
    d_cell_end = None
    d_query_x = None
    d_query_y = None
    d_min_sq = None
    d_min_idx = None
    d_counts = None
    d_offsets = None
    d_out_left = None
    d_out_right = None

    try:
        kernels = _grid_nearest_kernels()

        # Assign cells
        d_cell_ids = runtime.allocate((n_tree,), np.int32)
        grid_a, block_a = runtime.launch_config(kernels["grid_assign_cells"], n_tree)
        runtime.launch(
            kernels["grid_assign_cells"],
            grid=grid_a,
            block=block_a,
            params=(
                (
                    ptr(d_tree_x),
                    ptr(d_tree_y),
                    origin_x,
                    origin_y,
                    cell_size,
                    n_cols,
                    n_rows,
                    ptr(d_cell_ids),
                    n_tree,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

        # Sort tree points by cell_id
        d_order = cp.arange(n_tree, dtype=cp.int32)
        sorted_result = sort_pairs(d_cell_ids, d_order, synchronize=False)
        d_sorted_cell_ids = sorted_result.keys
        d_sort_order = sorted_result.values

        # Reorder tree coords by sort order
        d_sorted_tree_x = d_tree_x[d_sort_order]
        d_sorted_tree_y = d_tree_y[d_sort_order]
        d_sorted_global_idx = d_tree_global_idx[d_sort_order]

        # Build cell ranges
        d_cell_start = runtime.allocate((n_cells,), np.int32, zero=True)
        d_cell_end = runtime.allocate((n_cells,), np.int32, zero=True)
        grid_r, block_r = runtime.launch_config(kernels["grid_build_cell_ranges"], n_tree)
        runtime.launch(
            kernels["grid_build_cell_ranges"],
            grid=grid_r,
            block=block_r,
            params=(
                (ptr(d_sorted_cell_ids), ptr(d_cell_start), ptr(d_cell_end), n_cells, n_tree),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

        # Free intermediates no longer needed
        runtime.free(d_cell_ids)
        d_cell_ids = None
        runtime.free(d_tree_x)
        d_tree_x = None
        runtime.free(d_tree_y)
        d_tree_y = None
        runtime.free(d_tree_global_idx)
        d_tree_global_idx = None
        del d_order, d_sorted_cell_ids, d_sort_order

        # --- Upload query coords ---
        d_query_x = runtime.from_host(query_x_h)
        d_query_y = runtime.from_host(query_y_h)

        # --- Grid nearest search ---
        d_min_sq = runtime.allocate((n_query,), np.float64)
        d_min_idx = runtime.allocate((n_query,), np.int32)
        grid_s, block_s = runtime.launch_config(kernels["grid_nearest_search"], n_query)
        runtime.launch(
            kernels["grid_nearest_search"],
            grid=grid_s,
            block=block_s,
            params=(
                (
                    ptr(d_query_x),
                    ptr(d_query_y),
                    ptr(d_sorted_tree_x),
                    ptr(d_sorted_tree_y),
                    ptr(d_sorted_global_idx),
                    ptr(d_cell_start),
                    ptr(d_cell_end),
                    n_cols,
                    n_rows,
                    origin_x,
                    origin_y,
                    cell_size,
                    n_tree,
                    1 if exclusive else 0,
                    ptr(d_min_sq),
                    ptr(d_min_idx),
                    n_query,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

        # --- Post-processing ---
        if not return_all:
            # return_all=False: just use min_idx directly (one per query).
            runtime.synchronize()
            h_min_sq = runtime.copy_device_to_host(
                d_min_sq,
                reason="nearest grid min-distance host export",
            )
            h_min_idx = runtime.copy_device_to_host(
                d_min_idx,
                reason="nearest grid min-index host export",
            )

            # Filter: finite distances only, and max_distance
            valid_mask = np.isfinite(h_min_sq)
            if max_distance is not None:
                valid_mask &= np.sqrt(h_min_sq) <= max_distance
            valid_q = np.flatnonzero(valid_mask)
            if valid_q.size == 0:
                if max_distance is None:
                    return None
                result = _empty_nearest_result(return_distance)
                return result, "owned_gpu_nearest"

            left = query_global_idx[valid_q].astype(np.intp)
            right = h_min_idx[valid_q].astype(np.intp)
            if max_distance is None and left.size != query_global_idx.size:
                # The zero-copy grid path must cover every valid query row for
                # unbounded nearest. If it does not, decline and let the exact
                # indexed GPU path take over.
                return None
            indices = np.vstack((left, right))
            if return_distance:
                dists = np.sqrt(h_min_sq[valid_q])
                return (indices, dists), "owned_gpu_nearest"
            return indices, "owned_gpu_nearest"

        # return_all=True: need tie-count + tie-scatter
        # Fast path: check if all counts would be 1 (common case)
        d_counts = runtime.allocate((n_query,), np.int32)
        grid_tc, block_tc = runtime.launch_config(kernels["grid_nearest_tie_count"], n_query)
        runtime.launch(
            kernels["grid_nearest_tie_count"],
            grid=grid_tc,
            block=block_tc,
            params=(
                (
                    ptr(d_query_x),
                    ptr(d_query_y),
                    ptr(d_sorted_tree_x),
                    ptr(d_sorted_tree_y),
                    ptr(d_cell_start),
                    ptr(d_cell_end),
                    n_cols,
                    n_rows,
                    origin_x,
                    origin_y,
                    cell_size,
                    ptr(d_min_sq),
                    1 if exclusive else 0,
                    ptr(d_counts),
                    n_query,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

        d_offsets = exclusive_sum(d_counts)
        total_pairs = (
            count_scatter_total(
                runtime,
                d_counts,
                d_offsets,
                reason="nearest point-window pair allocation fence",
            )
            if n_query > 0
            else 0
        )
        if total_pairs == 0:
            if max_distance is None:
                return None
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"

        d_out_left = runtime.allocate((total_pairs,), np.int32)
        d_out_right = runtime.allocate((total_pairs,), np.int32)
        grid_ts, block_ts = runtime.launch_config(kernels["grid_nearest_tie_scatter"], n_query)
        runtime.launch(
            kernels["grid_nearest_tie_scatter"],
            grid=grid_ts,
            block=block_ts,
            params=(
                (
                    ptr(d_query_x),
                    ptr(d_query_y),
                    ptr(d_sorted_tree_x),
                    ptr(d_sorted_tree_y),
                    ptr(d_sorted_global_idx),
                    ptr(d_cell_start),
                    ptr(d_cell_end),
                    n_cols,
                    n_rows,
                    origin_x,
                    origin_y,
                    cell_size,
                    ptr(d_min_sq),
                    1 if exclusive else 0,
                    ptr(d_offsets),
                    ptr(d_out_left),
                    ptr(d_out_right),
                    n_query,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        runtime.synchronize()

        # D2H transfer of final indices
        h_left_dense = runtime.copy_device_to_host(
            d_out_left,
            reason="nearest grid tie-left-index host export",
        ).astype(np.intp)
        h_right = runtime.copy_device_to_host(
            d_out_right,
            reason="nearest grid tie-right-index host export",
        ).astype(np.intp)

        # Compute per-pair distances from min_sq (one D2H for distance data)
        h_min_sq_all = runtime.copy_device_to_host(
            d_min_sq,
            reason="nearest grid tie-distance host export",
        )
        pair_dists = np.sqrt(h_min_sq_all[h_left_dense])

        # Map dense query indices back to global indices
        left = query_global_idx[h_left_dense]
        right = h_right  # already global tree indices from sorted_tree_global_idx

        # Apply max_distance filter
        if max_distance is not None and left.size > 0:
            within = pair_dists <= max_distance
            left = left[within]
            right = right[within]
            pair_dists = pair_dists[within]

        if left.size == 0:
            if max_distance is None:
                return None
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"
        if max_distance is None and np.unique(left).size != query_global_idx.size:
            # Same contract as above: for unbounded nearest every valid query
            # row must appear at least once in the returned pairs.
            return None

        indices = np.vstack((left, right))
        if return_distance:
            return (indices, pair_dists), "owned_gpu_nearest"
        return indices, "owned_gpu_nearest"

    finally:
        runtime.free(d_tree_x)
        runtime.free(d_tree_y)
        runtime.free(d_tree_global_idx)
        runtime.free(d_cell_ids)
        runtime.free(d_sorted_tree_x)
        runtime.free(d_sorted_tree_y)
        runtime.free(d_sorted_global_idx)
        runtime.free(d_cell_start)
        runtime.free(d_cell_end)
        runtime.free(d_query_x)
        runtime.free(d_query_y)
        runtime.free(d_min_sq)
        runtime.free(d_min_idx)
        runtime.free(d_counts)
        runtime.free(d_offsets)
        runtime.free(d_out_left)
        runtime.free(d_out_right)


def _nearest_indexed_point_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    *,
    return_all: bool,
    return_distance: bool,
    exclusive: bool,
    max_distance: float | None = None,
    return_device: bool = False,
) -> tuple[Any, str] | None:
    """Point nearest index path.

    Physical shape: per-query indexed point search followed by tie scatter.
    Native callers can request device COO pair arrays and fp64 distances for a
    downstream ``NativeRelation`` without public pair export.
    """
    if not has_gpu_runtime():
        return None
    if not _points_only(query_owned) or not _points_only(tree_owned):
        return None
    if (
        GeometryFamily.POINT not in query_owned.families
        or GeometryFamily.POINT not in tree_owned.families
    ):
        return None

    import cupy as cp

    runtime = get_cuda_runtime()
    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="nearest_spatial_index selected indexed GPU nearest for query geometry input",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="nearest_spatial_index selected indexed GPU nearest for tree geometry input",
    )

    query_x, query_y = _device_dense_point_coords(query_owned)
    tree_x, tree_y = _device_dense_point_coords(tree_owned)
    valid_tree_rows = cp.flatnonzero(cp.isfinite(tree_x) & cp.isfinite(tree_y)).astype(
        cp.int32, copy=False
    )
    if int(valid_tree_rows.size) == 0:
        result = (
            _empty_nearest_result_device(return_distance)
            if return_device
            else _empty_nearest_result(return_distance)
        )
        return result, "owned_gpu_nearest"

    sorted_tree = sort_pairs(tree_x[valid_tree_rows], valid_tree_rows, synchronize=False)
    query_probe_x = cp.nan_to_num(query_x, nan=0.0)
    insert_idx = lower_bound(sorted_tree.keys, query_probe_x, synchronize=False).astype(
        cp.int32, copy=False
    )
    min_sq = cp.empty(query_owned.row_count, dtype=cp.float64)
    counts = cp.empty(query_owned.row_count, dtype=cp.int32)
    offsets = None
    out_left_alloc = None
    out_right_alloc = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        min_params = (
            (
                ptr(query_x),
                ptr(query_y),
                ptr(sorted_tree.keys),
                ptr(tree_y),
                ptr(sorted_tree.values),
                ptr(insert_idx),
                int(sorted_tree.keys.size),
                1 if exclusive else 0,
                ptr(min_sq),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        min_grid, min_block = runtime.launch_config(
            kernels["point_nearest_min_sq_from_sorted_x"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_nearest_min_sq_from_sorted_x"],
            grid=min_grid,
            block=min_block,
            params=min_params,
        )

        best = cp.sqrt(min_sq)
        tol = 1e-8 + 1e-5 * cp.abs(best)
        query_min_x = cp.where(cp.isfinite(best), query_x - best - tol, 0.0)
        query_max_x = cp.where(cp.isfinite(best), query_x + best + tol, 0.0)
        start_idx = lower_bound(sorted_tree.keys, query_min_x, synchronize=False).astype(
            cp.int32, copy=False
        )
        end_idx = upper_bound(sorted_tree.keys, query_max_x, synchronize=False).astype(
            cp.int32, copy=False
        )

        count_params = (
            (
                ptr(query_x),
                ptr(query_y),
                ptr(sorted_tree.keys),
                ptr(tree_y),
                ptr(sorted_tree.values),
                ptr(start_idx),
                ptr(end_idx),
                ptr(min_sq),
                1 if exclusive else 0,
                ptr(counts),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(
            kernels["point_nearest_tie_count_from_sorted_x"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_nearest_tie_count_from_sorted_x"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = (
            count_scatter_total(
                runtime,
                counts,
                offsets,
                reason="nearest sorted-x tie-pair allocation fence",
            )
            if query_owned.row_count > 0
            else 0
        )
        if total_pairs == 0:
            result = (
                _empty_nearest_result_device(return_distance)
                if return_device
                else _empty_nearest_result(return_distance)
            )
            return result, "owned_gpu_nearest"

        out_left_alloc = runtime.allocate((total_pairs,), np.int32)
        out_right_alloc = runtime.allocate((total_pairs,), np.int32)
        scatter_params = (
            (
                ptr(query_x),
                ptr(query_y),
                ptr(sorted_tree.keys),
                ptr(tree_y),
                ptr(sorted_tree.values),
                ptr(start_idx),
                ptr(end_idx),
                ptr(offsets),
                ptr(min_sq),
                1 if exclusive else 0,
                ptr(out_left_alloc),
                ptr(out_right_alloc),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(
            kernels["point_nearest_tie_scatter_from_sorted_x"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_nearest_tie_scatter_from_sorted_x"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        if return_device:
            d_left_result = out_left_alloc
            d_right_result = out_right_alloc
            d_distances = best[d_left_result].astype(cp.float64, copy=False)

            if max_distance is not None and d_left_result.size:
                within = d_distances <= float(max_distance)
                d_left_result = d_left_result[within]
                d_right_result = d_right_result[within]
                d_distances = d_distances[within]

            if not return_all and d_left_result.size:
                keep = cp.empty(d_left_result.shape, dtype=cp.bool_)
                keep[0] = True
                if d_left_result.size > 1:
                    keep[1:] = d_left_result[1:] != d_left_result[:-1]
                d_left_result = d_left_result[keep]
                d_right_result = d_right_result[keep]
                d_distances = d_distances[keep]

            d_left_result = d_left_result.astype(cp.int32, copy=True)
            d_right_result = d_right_result.astype(cp.int32, copy=True)
            if return_distance:
                return ((d_left_result, d_right_result), d_distances), "owned_gpu_nearest"
            return (d_left_result, d_right_result), "owned_gpu_nearest"
        runtime.synchronize()

        left = runtime.copy_device_to_host(
            out_left_alloc,
            reason="nearest indexed point left-index host export",
        ).astype(np.intp, copy=False)
        right = runtime.copy_device_to_host(
            out_right_alloc,
            reason="nearest indexed point right-index host export",
        ).astype(np.intp, copy=False)
        best_host = runtime.copy_device_to_host(
            best,
            reason="nearest indexed point best-distance host export",
        )

        # Apply max_distance filter when bounded nearest is requested.
        if max_distance is not None and left.size:
            pair_distances = np.asarray(best_host[left], dtype=np.float64)
            within = pair_distances <= max_distance
            left = left[within]
            right = right[within]

        if not return_all and left.size:
            keep = np.zeros(left.size, dtype=bool)
            _, first_idx = np.unique(left, return_index=True)
            keep[np.asarray(first_idx, dtype=np.intp)] = True
            left = left[keep]
            right = right[keep]

        if left.size == 0:
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"

        indices = np.vstack((left, right))
        if return_distance:
            distances = np.asarray(best_host[left], dtype=np.float64)
            return (indices, distances), "owned_gpu_nearest"
        return indices, "owned_gpu_nearest"
    finally:
        runtime.free(min_sq)
        runtime.free(counts)
        runtime.free(offsets)
        runtime.free(out_left_alloc)
        runtime.free(out_right_alloc)


def _generate_point_nearest_candidates_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    *,
    max_distance: float,
    exclusive: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not has_gpu_runtime() or not np.isfinite(max_distance):
        return None
    if not _points_only(query_owned) or not _points_only(tree_owned):
        return None
    if (
        GeometryFamily.POINT not in query_owned.families
        or GeometryFamily.POINT not in tree_owned.families
    ):
        return None

    selection = plan_dispatch_selection(
        kernel_name="point_nearest_candidates",
        kernel_class=KernelClass.METRIC,
        row_count=query_owned.row_count,
        gpu_available=True,
        current_residency=combined_residency(query_owned, tree_owned),
        work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
            row_count=query_owned.row_count,
            candidate_pair_count=query_owned.row_count * tree_owned.row_count,
            primary_unit_name="nearest-candidate-pair",
        ),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    import cupy as cp

    runtime = get_cuda_runtime()
    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="nearest_spatial_index selected GPU point sweep candidate generation for query geometry input",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="nearest_spatial_index selected GPU point sweep candidate generation for tree geometry input",
    )

    regular_grid_point_index = _detect_regular_grid_point_index(tree_owned)
    if regular_grid_point_index is not None:
        return _generate_point_nearest_candidates_regular_grid_gpu(
            query_owned,
            regular_grid_point_index,
            max_distance=max_distance,
            exclusive=exclusive,
        )

    query_x, query_y = _device_dense_point_coords(query_owned)
    tree_x, tree_y = _device_dense_point_coords(tree_owned)
    valid_tree_rows = cp.flatnonzero(cp.isfinite(tree_x) & cp.isfinite(tree_y)).astype(
        cp.int32, copy=False
    )
    if int(valid_tree_rows.size) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    sorted_tree = sort_pairs(tree_x[valid_tree_rows], valid_tree_rows, synchronize=False)
    query_min_x = cp.nan_to_num(query_x - max_distance, nan=0.0)
    query_max_x = cp.nan_to_num(query_x + max_distance, nan=0.0)
    start_idx = lower_bound(sorted_tree.keys, query_min_x, synchronize=False).astype(
        cp.int32, copy=False
    )
    end_idx = upper_bound(sorted_tree.keys, query_max_x, synchronize=False).astype(
        cp.int32, copy=False
    )
    counts = cp.empty(query_owned.row_count, dtype=cp.int32)
    offsets = None
    out_left = None
    out_right = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        count_params = (
            (
                ptr(query_x),
                ptr(query_y),
                ptr(tree_y),
                ptr(sorted_tree.keys),
                ptr(sorted_tree.values),
                ptr(start_idx),
                ptr(end_idx),
                float(max_distance),
                1 if exclusive else 0,
                ptr(counts),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(
            kernels["point_x_window_count"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_x_window_count"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = (
            count_scatter_total(
                runtime,
                counts,
                offsets,
                reason="nearest x-window pair allocation fence",
            )
            if query_owned.row_count > 0
            else 0
        )
        if total_pairs == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty

        out_left = runtime.allocate((total_pairs,), np.int32)
        out_right = runtime.allocate((total_pairs,), np.int32)
        scatter_params = (
            (
                ptr(query_x),
                ptr(query_y),
                ptr(tree_y),
                ptr(sorted_tree.keys),
                ptr(sorted_tree.values),
                ptr(start_idx),
                ptr(end_idx),
                ptr(offsets),
                float(max_distance),
                1 if exclusive else 0,
                ptr(out_left),
                ptr(out_right),
                query_owned.row_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(
            kernels["point_x_window_scatter"], query_owned.row_count
        )
        runtime.launch(
            kernels["point_x_window_scatter"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        runtime.synchronize()

        left = runtime.copy_device_to_host(
            out_left,
            reason="nearest x-window left-index host export",
        ).astype(np.int32, copy=False)
        right = runtime.copy_device_to_host(
            out_right,
            reason="nearest x-window right-index host export",
        ).astype(np.int32, copy=False)
        return left, right
    finally:
        runtime.free(counts)
        runtime.free(offsets)
        runtime.free(out_left)
        runtime.free(out_right)


# ---------------------------------------------------------------------------
# Distance computation dispatch
# ---------------------------------------------------------------------------


def _compute_pair_distances_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    d_distances,
    pair_count: int,
) -> bool:
    """Compute distances for candidate pairs on GPU.

    Dispatches to the appropriate distance kernel based on the geometry
    families present in *query_owned* and *tree_owned*.  Writes results
    into *d_distances*.  Returns True on success, False if the family
    combination is not supported.
    """
    query_family = _single_family(query_owned)
    tree_family = _single_family(tree_owned)
    if query_family is None or tree_family is None:
        return False

    point_family = GeometryFamily.POINT

    if query_family == point_family and tree_family == point_family:
        # Point x Point -- use the inline kernel from spatial_query.
        query_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="dwithin: GPU point-point distance",
        )
        tree_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="dwithin: GPU point-point distance",
        )
        _launch_point_point_distance_kernel(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
        )
        return True

    if query_family == point_family:
        from .point_distance import compute_point_distance_gpu

        precision_context = _plan_nearest_metric_precision(
            query_owned,
            tree_owned,
            pair_count,
        )
        return compute_point_distance_gpu(
            query_owned,
            tree_owned,
            d_left,
            d_right,
            d_distances,
            pair_count,
            tree_family=tree_family,
            compute_precision=precision_context.coarse_plan.compute_precision,
        )

    if tree_family == point_family:
        from .point_distance import compute_point_distance_gpu

        precision_context = _plan_nearest_metric_precision(
            query_owned,
            tree_owned,
            pair_count,
        )
        return compute_point_distance_gpu(
            tree_owned,
            query_owned,
            d_right,
            d_left,
            d_distances,
            pair_count,
            tree_family=query_family,
            compute_precision=precision_context.coarse_plan.compute_precision,
        )

    # Non-point x non-point
    from .segment_distance import compute_segment_distance_gpu

    return compute_segment_distance_gpu(
        query_owned,
        tree_owned,
        d_left,
        d_right,
        d_distances,
        pair_count,
        query_family=query_family,
        tree_family=tree_family,
    )


# ---------------------------------------------------------------------------
# Multipoint distance computation
# ---------------------------------------------------------------------------


def _compute_multipoint_distances_gpu(
    mp_owned: OwnedGeometryArray,
    target_owned: OwnedGeometryArray,
    mp_idx: np.ndarray,
    target_idx: np.ndarray,
    *,
    target_family: GeometryFamily,
    exclusive: bool = False,
    precision_plan: PrecisionPlan | None = None,
) -> np.ndarray | None:
    """Compute multipoint->geometry distances via coord expansion + segmented min.

    Expands each multipoint into per-coordinate point pairs, computes point
    distances using existing GPU kernels (Tier 1), then reduces per-multipoint
    via CCCL ``segmented_reduce_min`` on device (Tier 3a, ADR-0033).  Falls
    back to a host-side Python loop for small inputs (pair_count <= 256)
    where upload overhead would dominate.

    Returns host float64 distance array (one per input pair), or None if the
    target family is not supported by the point distance kernel.
    """
    mp_family = GeometryFamily.MULTIPOINT
    if mp_family not in mp_owned.families:
        return None

    mp_buffer = mp_owned.families[mp_family]
    mp_offsets = mp_buffer.geometry_offsets
    mp_row_offsets = mp_owned.family_row_offsets

    # Vectorised expansion: build per-coord pairs and segment boundaries.
    pair_count = mp_idx.size
    mp_rows = mp_row_offsets[mp_idx].astype(np.int32)
    coord_starts = mp_offsets[mp_rows]
    coord_ends = mp_offsets[mp_rows + 1]
    coord_counts = coord_ends - coord_starts

    total_expanded = int(coord_counts.sum())
    if total_expanded == 0:
        return np.full(pair_count, np.inf, dtype=np.float64)

    # Build segment start/end arrays for reduction.
    seg_ends_arr = np.cumsum(coord_counts).astype(np.int32)
    seg_starts_arr = np.empty_like(seg_ends_arr)
    seg_starts_arr[0] = 0
    seg_starts_arr[1:] = seg_ends_arr[:-1]

    # Expand: each MP coord -> (coord_index, target_index) pair.
    expanded_point_idx = np.empty(total_expanded, dtype=np.int32)
    expanded_target_idx = np.empty(total_expanded, dtype=np.int32)
    cursor = 0
    for i in range(pair_count):
        cs = int(coord_starts[i])
        n = int(coord_counts[i])
        expanded_point_idx[cursor : cursor + n] = np.arange(cs, cs + n, dtype=np.int32)
        expanded_target_idx[cursor : cursor + n] = target_idx[i]
        cursor += n

    # Create a temporary point OwnedGeometryArray from the MP's coord arrays.
    temp_point_owned = _make_point_owned_from_coords(mp_buffer.x, mp_buffer.y)

    if precision_plan is None:
        precision_plan = _plan_nearest_metric_precision(
            mp_owned,
            target_owned,
            total_expanded,
        ).coarse_plan

    runtime = get_cuda_runtime()
    d_exp_left = runtime.from_host(np.ascontiguousarray(expanded_point_idx))
    d_exp_right = runtime.from_host(np.ascontiguousarray(expanded_target_idx))
    d_exp_dist = runtime.allocate((total_expanded,), np.float64)

    try:
        if target_family == GeometryFamily.POINT:
            # Point x Point inline kernel.
            temp_point_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="multipoint distance: expanded points",
            )
            target_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="multipoint distance: target points",
            )
            _launch_point_point_distance_kernel(
                temp_point_owned,
                target_owned,
                d_exp_left,
                d_exp_right,
                d_exp_dist,
                total_expanded,
                exclusive=exclusive,
            )
            ok = True
        else:
            # Point x {Line, Polygon, ...} via existing point distance kernels.
            from .point_distance import compute_point_distance_gpu

            ok = compute_point_distance_gpu(
                temp_point_owned,
                target_owned,
                d_exp_left,
                d_exp_right,
                d_exp_dist,
                total_expanded,
                tree_family=target_family,
                exclusive=exclusive,
                compute_precision=precision_plan.compute_precision,
            )

        if not ok:
            return None

        # Segmented min reduction on device (CCCL Tier 3a) — avoids
        # downloading the full expanded distance array to host.
        if pair_count > 256:
            d_starts = runtime.from_host(seg_starts_arr.astype(np.int32))
            d_ends = runtime.from_host(seg_ends_arr.astype(np.int32))
            seg_result = segmented_reduce_min(
                d_exp_dist,
                d_starts,
                d_ends,
                num_segments=pair_count,
            )
            result = runtime.copy_device_to_host(
                seg_result.values,
                reason="nearest multipoint segmented-min distance host export",
            )
        else:
            exp_distances = runtime.copy_device_to_host(
                d_exp_dist,
                reason="nearest multipoint expanded-distance host export",
            )
            result = np.full(pair_count, np.inf, dtype=np.float64)
            for i in range(pair_count):
                s, e = int(seg_starts_arr[i]), int(seg_ends_arr[i])
                if s < e:
                    result[i] = exp_distances[s:e].min()
    finally:
        runtime.free(d_exp_left)
        runtime.free(d_exp_right)
        runtime.free(d_exp_dist)

    return result


# ---------------------------------------------------------------------------
# Mixed-family distance computation
# ---------------------------------------------------------------------------


def _compute_mixed_distances_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    exclusive: bool = False,
    device_candidates: object | None = None,
    *,
    record_fallback_event: bool = True,
    precision_context: _NearestMetricPrecisionContext | None = None,
) -> tuple[np.ndarray, bool] | None:
    """Compute distances for candidate pairs with mixed geometry families.

    Groups pairs by (left_tag, right_tag) and dispatches to the appropriate
    distance kernel for each group.  Multipoint pairs (MP x non-MP) are handled
    via coord expansion + point distance kernels + segmented min.  Falls back
    to Shapely only for MP x MP (requires double expansion).

    When *device_candidates* is a :class:`_DeviceCandidates`, sub-arrays
    are extracted on-device via CuPy fancy indexing to avoid redundant
    host->device transfers.

    Returns ``(host_distances, used_shapely_fallback)`` or ``None`` if GPU
    runtime is unavailable.
    """
    pair_count = left_idx.size
    if pair_count == 0:
        return np.empty(0, dtype=np.float64), False

    left_tags = query_owned.tags[left_idx]
    right_tags = tree_owned.tags[right_idx]

    runtime = get_cuda_runtime()
    distances = np.full(pair_count, np.inf, dtype=np.float64)
    used_shapely_fallback = False
    if precision_context is None:
        precision_context = _plan_nearest_metric_precision(
            query_owned,
            tree_owned,
            pair_count,
        )

    _dc = device_candidates
    _use_device_idx = _dc is not None and hasattr(_dc, "d_left")

    point_family = GeometryFamily.POINT

    for lt, rt in unique_tag_pairs(left_tags, right_tags):
        lf = TAG_FAMILIES.get(lt)
        rf = TAG_FAMILIES.get(rt)

        sub_mask = (left_tags == lt) & (right_tags == rt)
        sub_idx = np.flatnonzero(sub_mask)
        sub_left = left_idx[sub_idx]
        sub_right = right_idx[sub_idx]
        sub_count = sub_idx.size

        # Build device sub-arrays: CuPy fancy indexing when device candidates
        # are available, else upload from host.
        _own_sub_device = True
        if _use_device_idx:
            import cupy as cp

            d_sub_idx = cp.asarray(sub_idx.astype(np.int32))
            d_sub_left = _dc.d_left[d_sub_idx]
            d_sub_right = _dc.d_right[d_sub_idx]
            _own_sub_device = False  # CuPy manages these arrays
        else:
            d_sub_left = runtime.from_host(np.ascontiguousarray(sub_left, dtype=np.int32))
            d_sub_right = runtime.from_host(np.ascontiguousarray(sub_right, dtype=np.int32))

        # Try GPU distance kernel for this family pair.
        ok = False
        if lf is None or rf is None:
            pass  # Unknown tag -- will fall to Shapely below.
        elif lf == point_family and rf == point_family:
            # Point x Point: use inline kernel from spatial_query.
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                query_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="mixed nearest: point-point distance",
                )
                tree_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="mixed nearest: point-point distance",
                )
                _launch_point_point_distance_kernel(
                    query_owned,
                    tree_owned,
                    d_sub_left,
                    d_sub_right,
                    d_sub_dist,
                    sub_count,
                    exclusive=exclusive,
                )
                sub_distances = np.empty(sub_count, dtype=np.float64)
                runtime.copy_device_to_host(
                    d_sub_dist,
                    sub_distances,
                    reason="nearest mixed point-point distance host export",
                )
                distances[sub_idx] = sub_distances
                ok = True
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)
        elif (lf == GeometryFamily.MULTIPOINT) != (rf == GeometryFamily.MULTIPOINT):
            # One side is MP, the other is not -- expand MP coords to points,
            # compute point distances via existing kernels, segmented min.
            # Free device sub-arrays first -- MP handler allocates its own.
            if _own_sub_device:
                runtime.free(d_sub_left)
                runtime.free(d_sub_right)
            _own_sub_device = False  # already freed
            if lf == GeometryFamily.MULTIPOINT:
                mp_result = _compute_multipoint_distances_gpu(
                    query_owned,
                    tree_owned,
                    sub_left,
                    sub_right,
                    target_family=rf,
                    exclusive=exclusive,
                    precision_plan=precision_context.coarse_plan,
                )
            else:
                mp_result = _compute_multipoint_distances_gpu(
                    tree_owned,
                    query_owned,
                    sub_right,
                    sub_left,
                    target_family=lf,
                    exclusive=exclusive,
                    precision_plan=precision_context.coarse_plan,
                )
            if mp_result is not None:
                distances[sub_idx] = mp_result
                ok = True
        elif lf == point_family or rf == point_family:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .point_distance import compute_point_distance_gpu

                if lf == point_family:
                    ok = compute_point_distance_gpu(
                        query_owned,
                        tree_owned,
                        d_sub_left,
                        d_sub_right,
                        d_sub_dist,
                        sub_count,
                        tree_family=rf,
                        exclusive=exclusive,
                        compute_precision=precision_context.coarse_plan.compute_precision,
                    )
                else:
                    ok = compute_point_distance_gpu(
                        tree_owned,
                        query_owned,
                        d_sub_right,
                        d_sub_left,
                        d_sub_dist,
                        sub_count,
                        tree_family=lf,
                        exclusive=exclusive,
                        compute_precision=precision_context.coarse_plan.compute_precision,
                    )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(
                        d_sub_dist,
                        sub_distances,
                        reason="nearest mixed point-family distance host export",
                    )
                    distances[sub_idx] = sub_distances
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)
        else:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .segment_distance import compute_segment_distance_gpu

                ok = compute_segment_distance_gpu(
                    query_owned,
                    tree_owned,
                    d_sub_left,
                    d_sub_right,
                    d_sub_dist,
                    sub_count,
                    query_family=lf,
                    tree_family=rf,
                    exclusive=exclusive,
                )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(
                        d_sub_dist,
                        sub_distances,
                        reason="nearest mixed segment-family distance host export",
                    )
                    distances[sub_idx] = sub_distances
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)

        # For unsupported family pairs, fall back to Shapely distance.
        if not ok:
            if record_fallback_event and not used_shapely_fallback:
                record_shapely_fallback_event(
                    surface="vibespatial.spatial.nearest",
                    reason="GPU nearest refinement required Shapely fallback for unsupported geometry families",
                    detail=f"pair={lf.name if lf is not None else 'unknown'}/{rf.name if rf is not None else 'unknown'}",
                    pipeline="gpu_candidates -> shapely_refine",
                    d2h_transfer=True,
                )
                used_shapely_fallback = True
            query_shapely = np.asarray(query_owned.to_shapely(), dtype=object)
            tree_shapely = np.asarray(tree_owned.to_shapely(), dtype=object)
            sub_dists = shapely.distance(query_shapely[sub_left], tree_shapely[sub_right])
            distances[sub_idx] = np.asarray(sub_dists, dtype=np.float64)
            if exclusive:
                eq = np.asarray(
                    shapely.equals(query_shapely[sub_left], tree_shapely[sub_right]), dtype=bool
                )
                distances[sub_idx[eq]] = np.inf

    return distances, used_shapely_fallback


# ---------------------------------------------------------------------------
# dwithin refinement
# ---------------------------------------------------------------------------


def _plan_dwithin_gpu_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    pair_capacity: int,
):
    """Plan global metric precision and centering once for tiled dwithin."""
    if not has_gpu_runtime():
        return None

    precision_context = _plan_device_resident_metric_precision(
        query_owned,
        tree_owned,
        pair_capacity,
    )
    pointset_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    distance_center_device = None
    if (set(query_owned.families) | set(tree_owned.families)) & pointset_families:
        from vibespatial.spatial.point_distance import compute_distance_center_device

        distance_center_device = compute_distance_center_device(query_owned, tree_owned)
    return precision_context, distance_center_device


def _dwithin_mask_gpu_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left,
    d_right,
    per_row_distance,
    *,
    pair_active=None,
    precision_context: _NearestMetricPrecisionContext,
    distance_center_device,
):
    """Return an exact device dwithin mask for one bounded candidate tile."""
    if not has_gpu_runtime():
        return None

    try:
        import cupy as cp
    except ImportError:
        return None

    d_left = cp.asarray(d_left, dtype=cp.int32)
    d_right = cp.asarray(d_right, dtype=cp.int32)
    pair_count = int(d_left.size)
    if pair_count != int(d_right.size):
        raise ValueError("dwithin candidate indices must align")
    d_active = (
        cp.ones(pair_count, dtype=cp.bool_)
        if pair_active is None
        else cp.asarray(pair_active, dtype=cp.bool_)
    )
    if pair_count == 0:
        return cp.empty(0, dtype=cp.bool_), False

    distance_result = _compute_mixed_distances_gpu_device(
        query_owned,
        tree_owned,
        d_left,
        d_right,
        precision_context=precision_context,
        pair_active=d_active,
        center_device=distance_center_device,
    )
    if distance_result is None:
        return None
    d_distances, used_shapely_fallback = distance_result
    if used_shapely_fallback:
        return None

    d_thresholds_all = cp.asarray(per_row_distance, dtype=cp.float64)
    d_thresholds = d_thresholds_all[d_left]
    if precision_context.refinement_plan is not None and distance_center_device is not None:
        from vibespatial.api._native_rowset import NativeDeviceSelection

        query_tags = cp.asarray(query_owned._ensure_device_state().tags, dtype=cp.int8)[
            d_left
        ]
        tree_tags = cp.asarray(tree_owned._ensure_device_state().tags, dtype=cp.int8)[
            d_right
        ]
        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        multipoint_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
        pointset_pair = (
            (query_tags == point_tag)
            | (query_tags == multipoint_tag)
            | (tree_tags == point_tag)
            | (tree_tags == multipoint_tag)
        )
        ambiguity_selection = NativeDeviceSelection.from_mask(
            d_active
            & pointset_pair
            & (
                ~cp.isfinite(d_distances)
                | (
                    cp.abs(d_distances - d_thresholds)
                    <= precision_context.fp32_error_bound
                )
            ),
            source_row_count=pair_count,
        )
        d_ambiguous_left = ambiguity_selection.gather_capacity(
            d_left,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ambiguous_right = ambiguity_selection.gather_capacity(
            d_right,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        refined = _compute_mixed_distances_gpu_device(
            query_owned,
            tree_owned,
            d_ambiguous_left,
            d_ambiguous_right,
            precision_context=precision_context.refinement_context(),
            pair_active=ambiguity_selection.active_capacity_mask(),
            source_positions=ambiguity_selection.partition_capacity_positions(),
            output_distances=d_distances,
            center_device=distance_center_device,
        )
        if refined is None:
            return None
        d_distances, used_shapely_fallback = refined
        if used_shapely_fallback:
            return None

    return d_active & (d_distances <= d_thresholds), False


def _dwithin_refine_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray | None,
    right_idx: np.ndarray | None,
    per_row_distance: np.ndarray,
    device_candidates: _DeviceCandidates | None = None,
    *,
    return_device: bool = False,
) -> tuple[object, bool] | None:
    """GPU dwithin refinement: distance <= threshold filter.

    Device-side pipeline (ADR-0033 Tier 2 CuPy for threshold + selection):
      1. Compute distances on device via mixed-distance kernels (Tier 1)
      2. Build per-pair thresholds on device (Tier 2 CuPy)
      3. Apply distance <= threshold filter on device (Tier 2 CuPy)
      4. Represent surviving indices as a capacity-backed native relation
      5. Compact only at the terminal host export when return_device is false

    When *return_device* is True, a ``NativeRelationSelection`` preserves the
    source-pair capacity and device logical count for downstream native work.

    Returns a ``NativeRelationSelection`` for strict device output, compact
    host index arrays otherwise, or ``None`` when GPU refinement is unavailable.
    """
    if not has_gpu_runtime():
        return None

    try:
        import cupy as cp
    except ImportError:
        return None

    _dc = device_candidates
    _use_device_idx = _dc is not None and hasattr(_dc, "d_left")
    pair_count = (
        left_idx.size if left_idx is not None else (_dc.total_pairs if _use_device_idx else 0)
    )
    if pair_count == 0:
        empty = np.empty(0, dtype=np.int32)
        if return_device:
            from vibespatial.api._native_relation import NativeRelation
            from vibespatial.api._native_rowset import NativeDeviceSelection

            relation = NativeRelation(
                left_indices=cp.asarray(empty, dtype=cp.int32),
                right_indices=cp.asarray(empty, dtype=cp.int32),
                predicate="dwithin",
                left_row_count=query_owned.row_count,
                right_row_count=tree_owned.row_count,
            )
            return relation.filter_pairs_selection(
                NativeDeviceSelection.from_mask(
                    cp.empty(0, dtype=cp.bool_),
                    source_row_count=0,
                )
            ), False
        return (empty, empty), False

    # --- Device-side distance computation ---
    # Accumulate distances in a device CuPy array instead of host numpy.
    precision_context = _plan_device_resident_metric_precision(
        query_owned,
        tree_owned,
        pair_count,
    )
    pointset_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    distance_center_device = None
    if (set(query_owned.families) | set(tree_owned.families)) & pointset_families:
        from vibespatial.spatial.point_distance import compute_distance_center_device

        distance_center_device = compute_distance_center_device(query_owned, tree_owned)
    d_distances_result = _compute_mixed_distances_gpu_device(
        query_owned,
        tree_owned,
        left_idx,
        right_idx,
        device_candidates=device_candidates,
        precision_context=precision_context,
        center_device=distance_center_device,
    )
    if d_distances_result is None:
        if return_device:
            return None
        # Fall back to host-side path.
        if left_idx is None or right_idx is None:
            if not _use_device_idx:
                return None
            left_idx, right_idx = _dc.to_host()
        distances_result = _compute_mixed_distances_gpu(
            query_owned,
            tree_owned,
            left_idx,
            right_idx,
            exclusive=False,
            device_candidates=device_candidates,
        )
        if distances_result is None:
            return None
        distances, used_shapely_fallback = distances_result
        thresholds = per_row_distance[left_idx]
        keep = distances <= thresholds
        return (left_idx[keep], right_idx[keep]), used_shapely_fallback

    d_distances, used_shapely_fallback = d_distances_result

    # --- Device-side threshold filter (Tier 2 CuPy) ---
    d_thresholds_all = cp.asarray(per_row_distance, dtype=cp.float64)
    if _use_device_idx:
        d_left_all = _dc.d_left
        d_right_all = _dc.d_right
    else:
        d_left_all = cp.asarray(left_idx, dtype=cp.int32)
        d_right_all = cp.asarray(right_idx, dtype=cp.int32)
    d_thresholds = d_thresholds_all[d_left_all]

    # Distances close enough to a threshold that centered fp32 rounding could
    # change the boolean decision are selected at source-pair capacity.  Each
    # point-family launch consumes a device logical count, so inactive lanes
    # return before fp64 work and no exact cardinality crosses to Python.
    if precision_context.refinement_plan is not None and distance_center_device is not None:
        from vibespatial.api._native_rowset import NativeDeviceSelection

        query_tags = cp.asarray(query_owned._ensure_device_state().tags, dtype=cp.int8)[
            d_left_all
        ]
        tree_tags = cp.asarray(tree_owned._ensure_device_state().tags, dtype=cp.int8)[
            d_right_all
        ]
        point_tag = FAMILY_TAGS[GeometryFamily.POINT]
        multipoint_tag = FAMILY_TAGS[GeometryFamily.MULTIPOINT]
        pointset_pair = (
            (query_tags == point_tag)
            | (query_tags == multipoint_tag)
            | (tree_tags == point_tag)
            | (tree_tags == multipoint_tag)
        )
        ambiguity_selection = NativeDeviceSelection.from_mask(
            pointset_pair
            & (
                ~cp.isfinite(d_distances)
                | (
                    cp.abs(d_distances - d_thresholds)
                    <= precision_context.fp32_error_bound
                )
            ),
            source_row_count=pair_count,
        )
        d_ambiguous_left = ambiguity_selection.gather_capacity(
            d_left_all,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ambiguous_right = ambiguity_selection.gather_capacity(
            d_right_all,
            fill_value=0,
        ).astype(cp.int32, copy=False)
        d_ambiguity_active = ambiguity_selection.active_capacity_mask()
        d_ambiguity_partition = ambiguity_selection.partition_capacity_positions()
        refined = _compute_mixed_distances_gpu_device(
            query_owned,
            tree_owned,
            d_ambiguous_left,
            d_ambiguous_right,
            precision_context=precision_context.refinement_context(),
            pair_active=d_ambiguity_active,
            source_positions=d_ambiguity_partition,
            output_distances=d_distances,
            center_device=distance_center_device,
        )
        if refined is None:
            return None
        d_distances, _ = refined

    d_keep = d_distances <= d_thresholds

    if return_device:
        from vibespatial.api._native_relation import NativeRelation
        from vibespatial.api._native_rowset import NativeDeviceSelection

        relation = NativeRelation(
            left_indices=d_left_all.astype(cp.int32, copy=False),
            right_indices=d_right_all.astype(cp.int32, copy=False),
            predicate="dwithin",
            left_row_count=query_owned.row_count,
            right_row_count=tree_owned.row_count,
        )
        return relation.filter_pairs_selection(
            NativeDeviceSelection.from_mask(
                d_keep,
                source_row_count=pair_count,
            )
        ), used_shapely_fallback

    # Public/host output is an explicit terminal compaction boundary.
    d_keep_idx = cp.flatnonzero(d_keep)

    # Gather surviving indices on device.
    d_left_result = d_left_all[d_keep_idx]
    d_right_result = d_right_all[d_keep_idx]

    # --- Single D->H transfer of filtered results ---
    runtime = get_cuda_runtime()
    return (
        runtime.copy_device_to_host(
            d_left_result,
            reason="dwithin filtered left-index host export",
        ),
        runtime.copy_device_to_host(
            d_right_result,
            reason="dwithin filtered right-index host export",
        ),
    ), used_shapely_fallback


def _compute_mixed_distances_gpu_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray | None,
    right_idx: np.ndarray | None,
    device_candidates: object | None = None,
    *,
    precision_context: _NearestMetricPrecisionContext | None = None,
    pair_active=None,
    source_positions=None,
    output_distances=None,
    center_device=None,
):
    """Compute one device relation through one all-family grouped partition."""
    try:
        import cupy as cp
    except ImportError:
        return None

    _dc = device_candidates
    _use_device_idx = _dc is not None and hasattr(_dc, "d_left")
    pair_count = (
        left_idx.size if left_idx is not None else (_dc.total_pairs if _use_device_idx else 0)
    )
    if pair_count == 0:
        return cp.empty(0, dtype=cp.float64), False

    all_families = frozenset(FAMILY_TAGS)
    query_families = frozenset(query_owned.families)
    tree_families = frozenset(tree_owned.families)
    if not query_families <= all_families or not tree_families <= all_families:
        return None

    query_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="mixed distance partition consumes device query metadata",
    )
    tree_owned.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="mixed distance partition consumes device tree metadata",
    )
    query_state = query_owned._ensure_device_state()
    tree_state = tree_owned._ensure_device_state()
    d_left = (
        cp.asarray(_dc.d_left, dtype=cp.int32)
        if _use_device_idx
        else cp.asarray(left_idx, dtype=cp.int32)
    )
    d_right = (
        cp.asarray(_dc.d_right, dtype=cp.int32)
        if _use_device_idx
        else cp.asarray(right_idx, dtype=cp.int32)
    )
    d_pair_active = (
        cp.ones(pair_count, dtype=cp.bool_)
        if pair_active is None
        else cp.asarray(pair_active, dtype=cp.bool_)
    )
    d_source_positions = (
        cp.arange(pair_count, dtype=cp.int32)
        if source_positions is None
        else cp.asarray(source_positions, dtype=cp.int32)
    )
    d_distances = (
        cp.full(pair_count, cp.inf, dtype=cp.float64)
        if output_distances is None
        else cp.asarray(output_distances, dtype=cp.float64)
    )
    if precision_context is None:
        precision_context = _plan_device_resident_metric_precision(
            query_owned,
            tree_owned,
            pair_count,
        )

    from vibespatial.api._native_relation import NativeRelationFamilyPartition
    from vibespatial.spatial.point_distance import (
        compute_distance_center_device,
        compute_pointset_distance_gpu,
    )
    from vibespatial.spatial.segment_distance import (
        compute_segment_distance_partition_gpu,
    )

    family_count = len(FAMILY_TAGS)
    partition = NativeRelationFamilyPartition.from_pair_capacity(
        d_left,
        d_right,
        d_pair_active,
        cp.asarray(query_state.tags, dtype=cp.int8),
        cp.asarray(tree_state.tags, dtype=cp.int8),
        family_count=family_count,
        source_positions=d_source_positions,
    )
    group_count = family_count**2
    launch_capacity = max(1, (pair_count + group_count - 1) // group_count)
    pointset_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    if center_device is None and (query_families | tree_families) & pointset_families:
        center_device = compute_distance_center_device(query_owned, tree_owned)

    for left_family in query_families:
        for right_family in tree_families:
            group = partition.family_pair(
                left_family=left_family,
                right_family=right_family,
                left_family_tag=FAMILY_TAGS[left_family],
                right_family_tag=FAMILY_TAGS[right_family],
                launch_capacity=launch_capacity,
            )
            if left_family in pointset_families:
                ok = compute_pointset_distance_gpu(
                    query_owned,
                    tree_owned,
                    group.left_indices,
                    group.right_indices,
                    d_distances,
                    group.capacity,
                    query_family=left_family,
                    tree_family=right_family,
                    source_offset=group.source_offset,
                    logical_count=group.logical_count,
                    source_positions=group.source_positions,
                    center_device=center_device,
                    compute_precision=precision_context.coarse_plan.compute_precision,
                )
            elif right_family in pointset_families:
                ok = compute_pointset_distance_gpu(
                    tree_owned,
                    query_owned,
                    group.right_indices,
                    group.left_indices,
                    d_distances,
                    group.capacity,
                    query_family=right_family,
                    tree_family=left_family,
                    source_offset=group.source_offset,
                    logical_count=group.logical_count,
                    source_positions=group.source_positions,
                    center_device=center_device,
                    compute_precision=precision_context.coarse_plan.compute_precision,
                )
            else:
                ok = compute_segment_distance_partition_gpu(
                    query_owned,
                    tree_owned,
                    group.left_indices,
                    group.right_indices,
                    d_distances,
                    group.capacity,
                    query_family=left_family,
                    tree_family=right_family,
                    source_offset=group.source_offset,
                    logical_count=group.logical_count,
                    source_positions=group.source_positions,
                )
            if not ok:
                return None

    return d_distances, False


# ---------------------------------------------------------------------------
# Host-side nearest from precomputed distances
# ---------------------------------------------------------------------------


def _nearest_from_distances(
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    distances: np.ndarray,
    n_queries: int,
    *,
    max_distance: float,
    return_all: bool = True,
    return_distance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Build nearest result from precomputed distances (host-side reduce)."""
    if left_idx.size == 0:
        return _empty_nearest_result(return_distance)

    min_distance = np.full(n_queries, np.inf, dtype=np.float64)
    np.minimum.at(min_distance, left_idx, distances)
    if return_all:
        keep = np.isclose(distances, min_distance[left_idx])
    else:
        order = np.lexsort((right_idx, left_idx, distances))
        left_sorted = left_idx[order]
        first = np.r_[True, left_sorted[1:] != left_sorted[:-1]]
        keep = np.zeros(left_idx.size, dtype=bool)
        keep[order[first]] = True
    keep &= distances <= max_distance
    indices = np.vstack(
        (
            left_idx[keep].astype(np.intp, copy=False),
            right_idx[keep].astype(np.intp, copy=False),
        )
    )
    if return_distance:
        return indices, distances[keep]
    return indices


# ---------------------------------------------------------------------------
# Full GPU nearest refinement dispatcher
# ---------------------------------------------------------------------------


def _nearest_refine_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    n_queries: int,
    *,
    max_distance: float,
    return_all: bool = True,
    exclusive: bool = False,
    return_distance: bool = False,
    return_device: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray | None], bool] | None:
    """Full GPU nearest refinement pipeline.

    Handles point-point distance, point-to-segment/polygon distance,
    segment-to-segment distance, and mixed-family arrays.
    Returns ``(indices_2xN, distances_or_None)`` on success, or ``None``
    when the GPU path is not applicable.
    """
    if not has_gpu_runtime():
        return None

    # --- Try single-family fast paths first ---
    use_mixed = False

    if _points_only(query_owned) and _points_only(tree_owned):
        if (
            GeometryFamily.POINT not in query_owned.families
            or GeometryFamily.POINT not in tree_owned.families
        ):
            # All rows are null/empty -- return empty result instead of CPU fallback.
            if return_device:
                return _empty_nearest_result_device(return_distance), False
            return _empty_nearest_result(return_distance), False
        # fall through to point-point distance below
    elif _points_only(query_owned):
        tree_family = _tree_distance_family(tree_owned)
        if tree_family is not None:
            return _nearest_refine_gpu_typed(
                query_owned,
                tree_owned,
                left_idx,
                right_idx,
                n_queries,
                PointFamilyDistanceStrategy(tree_family),
                max_distance=max_distance,
                return_all=return_all,
                exclusive=exclusive,
                return_distance=return_distance,
                return_device=return_device,
            )
        use_mixed = True
    elif _points_only(tree_owned):
        query_family = _single_family(query_owned)
        if query_family is not None and query_family in _point_distance_families():
            return _nearest_refine_gpu_typed(
                tree_owned,
                query_owned,
                right_idx,
                left_idx,
                n_queries,
                PointFamilyDistanceStrategy(query_family),
                max_distance=max_distance,
                return_all=return_all,
                exclusive=exclusive,
                return_distance=return_distance,
                return_device=return_device,
            )
        use_mixed = True
    else:
        query_family = _single_family(query_owned)
        tree_family = _single_family(tree_owned)
        if query_family is not None and tree_family is not None:
            return _nearest_refine_gpu_typed(
                query_owned,
                tree_owned,
                left_idx,
                right_idx,
                n_queries,
                SegmentFamilyDistanceStrategy(query_family, tree_family),
                max_distance=max_distance,
                return_all=return_all,
                exclusive=exclusive,
                return_distance=return_distance,
                return_device=return_device,
            )
        use_mixed = True

    # --- Mixed-family fallback: per-pair tag dispatch ---
    if use_mixed:
        if return_device:
            return None
        precision_context = _plan_nearest_metric_precision(
            query_owned,
            tree_owned,
            left_idx.size,
        )
        mixed_distances_result = _compute_mixed_distances_gpu(
            query_owned,
            tree_owned,
            left_idx,
            right_idx,
            exclusive=exclusive,
            precision_context=precision_context,
        )
        if mixed_distances_result is not None:
            mixed_distances, used_shapely_fallback = mixed_distances_result
            if precision_context.refinement_plan is not None:
                ambiguous = _nearest_ambiguity_mask_host(
                    left_idx,
                    mixed_distances,
                    n_queries,
                    max_distance=max_distance,
                    error_bound=precision_context.fp32_error_bound,
                )
                if ambiguous.any():
                    refined_result = _compute_mixed_distances_gpu(
                        query_owned,
                        tree_owned,
                        left_idx[ambiguous],
                        right_idx[ambiguous],
                        exclusive=exclusive,
                        record_fallback_event=False,
                        precision_context=precision_context.refinement_context(),
                    )
                    if refined_result is not None:
                        refined_distances, refined_used_fallback = refined_result
                        mixed_distances[ambiguous] = refined_distances
                        used_shapely_fallback |= refined_used_fallback
            return _nearest_from_distances(
                left_idx,
                right_idx,
                mixed_distances,
                n_queries,
                max_distance=max_distance,
                return_all=return_all,
                return_distance=return_distance,
            ), used_shapely_fallback
        return None

    if (
        GeometryFamily.POINT not in query_owned.families
        or GeometryFamily.POINT not in tree_owned.families
    ):
        # Degenerate: both _points_only but POINT not in families -- all null.
        if return_device:
            return _empty_nearest_result_device(return_distance), False
        return _empty_nearest_result(return_distance), False

    # Point-point path: use PointPointDistanceStrategy via the unified pipeline.
    strategy = PointPointDistanceStrategy()
    return _nearest_refine_gpu_typed(
        query_owned,
        tree_owned,
        left_idx,
        right_idx,
        n_queries,
        strategy,
        max_distance=max_distance,
        return_all=return_all,
        exclusive=exclusive,
        return_distance=return_distance,
        return_device=return_device,
    )


# ---------------------------------------------------------------------------
# Iterative doubling for unbounded nearest on GPU
# ---------------------------------------------------------------------------


def _iterative_nearest_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    initial_distance: float,
    max_diagonal: float,
    *,
    n_queries: int,
    return_all: bool,
    exclusive: bool,
    return_distance: bool,
    return_device: bool = False,
):
    """Iterative doubling nearest: start with *initial_distance*, double until
    every query row has at least one candidate, then refine on GPU.

    Returns ``(result, impl_string)`` on success, or ``None`` when the
    iterative approach should be skipped (falls back to full-diagonal).
    """
    if not has_gpu_runtime():
        return None

    distance = initial_distance
    max_iterations = 8  # at most 2^8 = 256x the initial estimate
    for _ in range(max_iterations):
        if distance >= max_diagonal:
            # Reached full diagonal -- fall back to the caller's
            # full-extent path to avoid redundant work.
            return None

        expanded = _expand_bounds(
            query_bounds,
            np.full(query_bounds.shape[0], distance, dtype=np.float64),
        )
        gpu_candidates = _generate_candidates_gpu(expanded, tree_bounds)
        if gpu_candidates is not None:
            left_idx, right_idx = gpu_candidates
        else:
            per_row_dist = np.full(n_queries, distance, dtype=np.float64)
            left_idx, right_idx = _generate_distance_pairs(
                query_bounds,
                tree_bounds,
                per_row_dist,
            )

        if left_idx.size == 0:
            distance *= 2.0
            continue

        # Check coverage: every valid query row must have at least one
        # candidate.
        covered = np.zeros(n_queries, dtype=bool)
        if left_idx.size > 0:
            covered[left_idx] = True
        valid_queries = ~np.isnan(query_bounds).any(axis=1)
        uncovered = valid_queries & ~covered
        if uncovered.any():
            distance *= 2.0
            continue

        # All queries covered -- run GPU refinement.
        gpu_result = _nearest_refine_gpu(
            query_owned,
            tree_owned,
            left_idx,
            right_idx,
            n_queries,
            max_distance=distance,
            return_all=return_all,
            exclusive=exclusive,
            return_distance=return_distance,
            return_device=return_device,
        )
        if gpu_result is not None:
            result, used_shapely_fallback = gpu_result
            return result, "owned_cpu_nearest" if used_shapely_fallback else "owned_gpu_nearest"

        # GPU refinement declined (unsupported family combo) -- fall back.
        return None

    # Exhausted iterations without full coverage -- fall back.
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nearest_spatial_index(
    tree_geometries: np.ndarray | None,
    geometry: Any,
    *,
    tree_query_nearest,
    return_all: bool = True,
    max_distance: float | None = None,
    return_distance: bool = False,
    exclusive: bool = False,
    k: int = 1,
    tree_owned: OwnedGeometryArray | None = None,
    query_owned: OwnedGeometryArray | None = None,
    return_device: bool = False,
) -> tuple[Any, str]:
    """Find nearest tree geometry for each query geometry.

    Returns ``(result, implementation)`` where *implementation* is one of
    ``"strtree_host"``, ``"owned_gpu_nearest"``, or ``"owned_cpu_nearest"``.
    """
    if query_owned is not None:
        query_values = None
        scalar = False
        n_queries = query_owned.row_count
    else:
        query_values, scalar = _as_geometry_array(geometry)
        n_queries = 0 if query_values is None else len(query_values)
    if query_values is None and query_owned is None:
        result = (
            _empty_nearest_result_device(return_distance)
            if return_device
            else _empty_nearest_result(return_distance)
        )
        return result, "owned_cpu_nearest"

    # --- NEW: Try zero-copy GPU grid nearest (bypasses _to_owned entirely) ---
    if (
        not return_device
        and tree_owned is None
        and query_owned is None
        and tree_geometries is not None
    ):
        grid_result = _nearest_grid_gpu(
            tree_geometries,
            query_values,
            return_all=return_all,
            return_distance=return_distance,
            exclusive=exclusive,
            max_distance=max_distance,
        )
        if grid_result is not None:
            return grid_result

    # --- Convert owned arrays and compute bounds (shared by both paths) ---
    if query_owned is None:
        query_owned = _to_owned(query_values)
    if tree_owned is None:
        if tree_geometries is None:
            raise ValueError("tree_geometries or tree_owned is required for nearest")
        tree_owned = _to_owned(tree_geometries)
    n_queries = query_owned.row_count
    n_tree = tree_owned.row_count

    if return_device and not _supports_device_nearest_refinement(query_owned, tree_owned):
        return None, "owned_cpu_nearest"

    # Try the efficient indexed GPU nearest path for all-Point arrays.
    # Works for both bounded (max_distance != None) and unbounded nearest.
    if k == 1:
        indexed_result = _nearest_indexed_point_gpu(
            query_owned,
            tree_owned,
            return_all=return_all,
            return_distance=return_distance,
            exclusive=exclusive,
            max_distance=max_distance,
            return_device=return_device,
        )
        if indexed_result is not None:
            return indexed_result

    from .spatial_index_knn_device import spatial_index_knn_device

    if return_device and not return_all:
        try:
            query_bounds_device = compute_geometry_bounds_device(query_owned)
            tree_bounds_device = compute_geometry_bounds_device(tree_owned)
        except RuntimeError:
            return None, "owned_cpu_nearest"
        knn_result = spatial_index_knn_device(
            query_owned,
            tree_owned,
            query_bounds_device,
            tree_bounds_device,
            k=k,
            max_distance=max_distance,
            exclusive=exclusive,
            return_all=return_all,
        )
        if knn_result is None:
            return None, "owned_cpu_nearest"
        indices = (knn_result.d_query_idx, knn_result.d_target_idx)
        if return_distance:
            return (indices, knn_result.d_distances), "owned_gpu_nearest"
        return indices, "owned_gpu_nearest"

    query_bounds = compute_geometry_bounds(
        query_owned, dispatch_mode=_gpu_bounds_dispatch_mode(query_owned)
    )
    tree_bounds = compute_geometry_bounds(
        tree_owned, dispatch_mode=_gpu_bounds_dispatch_mode(tree_owned)
    )

    # --- Try device-side k-NN query (vibeSpatial-247.7.2) ---------------------
    # Unified GPU pipeline: candidate generation -> exact distance -> top-k.
    knn_result = None
    if not (return_device and return_all):
        knn_result = spatial_index_knn_device(
            query_owned,
            tree_owned,
            query_bounds,
            tree_bounds,
            k=k,
            max_distance=max_distance,
            exclusive=exclusive,
            return_all=return_all,
        )
    if knn_result is not None and knn_result.total_pairs > 0:
        if return_device:
            indices = (knn_result.d_query_idx, knn_result.d_target_idx)
            if return_distance:
                return (indices, knn_result.d_distances), "owned_gpu_nearest"
            return indices, "owned_gpu_nearest"
        runtime = get_cuda_runtime()
        h_left = runtime.copy_device_to_host(
            knn_result.d_query_idx,
            reason="nearest kNN query-index host export",
        ).astype(np.intp, copy=False)
        h_right = runtime.copy_device_to_host(
            knn_result.d_target_idx,
            reason="nearest kNN target-index host export",
        ).astype(np.intp, copy=False)
        indices = np.vstack((h_left, h_right))
        if return_distance:
            h_dist = runtime.copy_device_to_host(
                knn_result.d_distances,
                reason="nearest kNN distance host export",
            ).astype(np.float64, copy=False)
            return (indices, h_dist), "owned_gpu_nearest"
        return indices, "owned_gpu_nearest"

    # --- Effective max_distance -----------------------------------------------
    # When max_distance is None (unbounded nearest) compute an effective ceiling
    # from the data extent so the bounded candidate-generation pipeline produces
    # ALL valid query x tree pairs.  The downstream keep-mask uses the effective
    # value (INFINITY analog) so no actual filtering occurs for unbounded calls.
    #
    # Unbounded generates O(Q*M) candidates, so apply a crossover check: below
    # the COARSE threshold STRtree kNN is faster (avoids CCCL JIT + all-pairs).
    if max_distance is not None:
        effective_max_distance = float(max_distance)
    else:
        selection = plan_dispatch_selection(
            kernel_name="nearest_knn_brute",
            kernel_class=KernelClass.COARSE,
            row_count=n_queries,
            gpu_available=has_gpu_runtime(),
            current_residency=combined_residency(query_owned, tree_owned),
            work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
                row_count=n_queries,
                candidate_pair_count=n_queries * n_tree,
                primary_unit_name="nearest-candidate-pair",
            ),
        )
        if selection.selected is not ExecutionMode.GPU:
            if return_device:
                return None, "owned_cpu_nearest"
            # Below crossover -- STRtree kNN is more efficient for small data.
            if query_values is None:
                query_values = np.asarray(query_owned.to_shapely(), dtype=object)
            result = tree_query_nearest(
                query_values,
                max_distance=max_distance,
                return_distance=return_distance,
                all_matches=return_all,
                exclusive=exclusive,
            )
            if return_distance:
                indices, distances = result
                return (indices, distances), "strtree_host"
            return result, "strtree_host"

        # Bounding box of ALL valid geometry bounds (query u tree).
        all_bounds = np.vstack((query_bounds, tree_bounds))
        valid_mask = ~np.isnan(all_bounds).any(axis=1)
        if not valid_mask.any():
            result = _empty_nearest_result(return_distance)
            return result, "owned_cpu_nearest"
        valid_bounds = all_bounds[valid_mask]
        extent_dx = float(valid_bounds[:, 2].max() - valid_bounds[:, 0].min())
        extent_dy = float(valid_bounds[:, 3].max() - valid_bounds[:, 1].min())
        full_diagonal = float(np.hypot(extent_dx, extent_dy)) * 1.01 + 1.0

        # Iterative doubling: start with an estimated initial distance based
        # on the average spacing, then double until every query has at least
        # one candidate.  This avoids O(Q*M) candidate pairs for datasets
        # where the nearest neighbour is typically much closer than the full
        # extent diagonal.
        avg_spacing = full_diagonal / max(1.0, float(np.sqrt(n_tree)))
        initial_estimate = max(avg_spacing * 2.0, 1.0)
        iterative_result = _iterative_nearest_gpu(
            query_owned,
            tree_owned,
            query_bounds,
            tree_bounds,
            initial_estimate,
            full_diagonal,
            n_queries=n_queries,
            return_all=return_all,
            exclusive=exclusive,
            return_distance=return_distance,
            return_device=return_device,
        )
        if iterative_result is not None:
            return iterative_result
        # Fall through with full diagonal as last resort.
        effective_max_distance = full_diagonal

    # Try GPU candidate generation with expanded query bounds.
    point_sweep_candidates = _generate_point_nearest_candidates_gpu(
        query_owned,
        tree_owned,
        max_distance=effective_max_distance,
        exclusive=exclusive,
    )
    if point_sweep_candidates is not None:
        left_idx, right_idx = point_sweep_candidates
        impl = "owned_gpu_nearest"
    else:
        expanded_bounds = _expand_bounds(
            query_bounds,
            np.full(query_bounds.shape[0], effective_max_distance, dtype=np.float64),
        )
        gpu_candidates = _generate_candidates_gpu(expanded_bounds, tree_bounds)
        if gpu_candidates is not None:
            left_idx, right_idx = gpu_candidates
            impl = "owned_gpu_nearest"
        else:
            per_row_distance = np.full(n_queries, effective_max_distance, dtype=np.float64)
            left_idx, right_idx = _generate_distance_pairs(
                query_bounds, tree_bounds, per_row_distance
            )
            impl = "owned_cpu_nearest"

    if left_idx.size == 0:
        result = (
            _empty_nearest_result_device(return_distance)
            if return_device
            else _empty_nearest_result(return_distance)
        )
        return result, impl

    # --- GPU nearest refinement (Tier 1 NVRTC + Tier 3a CCCL) ----------------
    # When GPU is available and both arrays contain only points, run the entire
    # distance/reduce/filter pipeline on device to avoid the Shapely host path.
    # Works for both GPU-generated and CPU-generated candidate pairs.
    gpu_result = _nearest_refine_gpu(
        query_owned,
        tree_owned,
        left_idx,
        right_idx,
        n_queries,
        max_distance=effective_max_distance,
        return_all=return_all,
        exclusive=exclusive,
        return_distance=return_distance,
        return_device=return_device,
    )
    if gpu_result is not None:
        result, used_shapely_fallback = gpu_result
        return result, "owned_cpu_nearest" if used_shapely_fallback else "owned_gpu_nearest"

    if return_device:
        return None, impl

    # --- CPU Shapely refinement fallback -------------------------------------
    if impl == "owned_gpu_nearest":
        record_shapely_fallback_event(
            surface="vibespatial.spatial.nearest",
            reason="GPU nearest candidate refinement fell back to host Shapely distance",
            detail=f"max_distance={max_distance!r}, return_all={return_all}, exclusive={exclusive}",
            pipeline="gpu_candidates -> shapely_refine",
            d2h_transfer=False,
        )
    impl = "owned_cpu_nearest"
    if query_values is None:
        query_values = np.asarray(query_owned.to_shapely(), dtype=object)
    if tree_geometries is None:
        tree_geometries = np.asarray(tree_owned.to_shapely(), dtype=object)
    left_values = query_values[left_idx]
    right_values = tree_geometries[right_idx]
    distances = shapely.distance(left_values, right_values)
    distances = np.asarray(distances, dtype=np.float64)
    if exclusive:
        equal_mask = np.asarray(shapely.equals(left_values, right_values), dtype=bool)
        left_idx = left_idx[~equal_mask]
        right_idx = right_idx[~equal_mask]
        distances = distances[~equal_mask]
    if left_idx.size == 0:
        result = _empty_nearest_result(return_distance)
        return result, impl

    min_distance = np.full(n_queries, np.inf, dtype=np.float64)
    np.minimum.at(min_distance, left_idx, distances)
    if return_all:
        keep = np.isclose(distances, min_distance[left_idx])
    else:
        order = np.lexsort((right_idx, left_idx, distances))
        left_sorted = left_idx[order]
        first = np.r_[True, left_sorted[1:] != left_sorted[:-1]]
        keep = np.zeros(left_idx.size, dtype=bool)
        keep[order[first]] = True
    keep &= distances <= effective_max_distance
    indices = np.vstack(
        (left_idx[keep].astype(np.intp, copy=False), right_idx[keep].astype(np.intp, copy=False))
    )
    # ADR-0042: low-level spatial-query kernels still use integer index arrays.
    if __debug__:
        assert isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.integer), (
            f"ADR-0042: nearest indices must be integer ndarray, got {type(indices).__name__} dtype={getattr(indices, 'dtype', None)}"
        )
    if return_distance:
        return (indices, distances[keep]), impl
    return indices, impl
