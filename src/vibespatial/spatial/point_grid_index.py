"""Reusable dense-cell index for conservative point-tree candidates.

The carrier stores each point exactly once. Predicate queries may consume its
cell-aligned bbox superset because an exact predicate immediately refines the
relation. Bbox-only queries continue to use the exact Morton path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite, log2, sqrt

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    compile_kernel_group,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_primitives import PairSortStrategy, exclusive_sum, sort_pairs
from vibespatial.cuda.nvrtc_precompile import request_nvrtc_warmup
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.hotpath_trace import attach_work_amplification, hotpath_stage

from .point_grid_index_kernels import (
    _POINT_GRID_INDEX_SOURCE,
    POINT_GRID_INDEX_KERNEL_NAMES,
)
from .point_partition import (
    PointPartitionDecline,
    PointPartitionPreflight,
    PointPartitionQueryPlan,
    PointPartitionQuerySlice,
    PointPartitionVariant,
    cached_point_partition,
    checked_i32,
    checked_product,
    checked_sum,
    point_partition_all_bounds_finite,
    point_partition_cache_key,
    point_partition_fp64_coarse_plan,
    publish_point_partition,
    query_plan,
    record_point_partition_readiness,
    retain_point_partition_completion,
    wait_for_point_partition,
)
from .query_types import (
    CandidateRelationCapacityError,
    _DeviceCandidates,
    require_device_candidate_pair_capacity,
)

_MIN_POINT_GRID_ROWS = 100_000
_TARGET_POINTS_PER_CELL = 8
_MIN_GRID_SIZE = 64
_MAX_GRID_SIZE = 2_048
_MAX_PLANNING_PACKET_BYTES = 64 * 1024

request_nvrtc_warmup(
    [("point-grid-index", _POINT_GRID_INDEX_SOURCE, POINT_GRID_INDEX_KERNEL_NAMES)]
)


@dataclass(frozen=True)
class PreparedPointGridIndex:
    """Device-resident point rows grouped into a fixed square grid."""

    grid_size: int
    cache_key: object
    precision_plan: object
    readiness: object
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    sorted_tree_rows: object
    cell_counts: object
    cell_offsets: object
    integral_counts: object

    @property
    def device_bytes(self) -> int:
        return sum(
            int(getattr(value, "nbytes", 0))
            for value in (
                self.sorted_tree_rows,
                self.cell_counts,
                self.cell_offsets,
                self.integral_counts,
            )
        )


def _point_grid_preparation_metrics(
    prepared,
    *,
    built: bool,
    cache_hit: bool,
    declined: bool,
    query_count: int,
    pair_budget: int,
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...]]:
    """Return cache evidence from host-owned preparation metadata only."""
    max_metrics = {
        "pair_budget_slots": int(pair_budget),
    }
    if prepared is not None:
        grid_size = int(prepared.grid_size)
        max_metrics.update(
            {
                "source_rows": int(prepared.cache_key.row_count),
                "grid_cells": grid_size * grid_size,
                "persistent_bytes": int(prepared.device_bytes),
            }
        )
    sum_metrics = {
        "preparation_requests": 1,
        "build_count": int(built),
        "declined_preparations": int(declined),
        "query_rows_requested": int(query_count),
    }
    unavailable = [
        "build_seconds",
        "avoidable_rebuild_seconds",
        "invalidation_reason",
    ]
    if declined:
        unavailable.extend(
            (
                "cache_hits",
                "cache_misses",
                "source_rows",
                "grid_cells",
                "persistent_bytes",
            )
        )
    else:
        sum_metrics.update(
            {
                "cache_hits": int(cache_hit),
                "cache_misses": int(built),
            }
        )
    return sum_metrics, max_metrics, tuple(unavailable)


def point_grid_index_kernels():
    return compile_kernel_group(
        "point-grid-index",
        _POINT_GRID_INDEX_SOURCE,
        POINT_GRID_INDEX_KERNEL_NAMES,
    )


def _grid_size_for_rows(row_count: int) -> int:
    target = max(sqrt(max(row_count, 1) / _TARGET_POINTS_PER_CELL), 1.0)
    power_of_two = 1 << int(ceil(log2(target)))
    return min(max(power_of_two, _MIN_GRID_SIZE), _MAX_GRID_SIZE)


def _point_grid_extents(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> tuple[float, float] | None:
    """Return finite positive extents or decline unsafe fp64 normalization."""
    x_extent = xmax - xmin
    y_extent = ymax - ymin
    if (
        not isfinite(x_extent)
        or not isfinite(y_extent)
        or x_extent <= 0.0
        or y_extent <= 0.0
    ):
        return None
    return x_extent, y_extent


def _point_grid_required_bytes(
    row_count: int,
    cell_count: int,
    *,
    needs_device_bounds: bool,
    query_count: int = 0,
    pair_budget: int = 0,
) -> int:
    """Bound every simultaneously-live grid build and query allocation.

    The build no longer uses ``cupy.bincount``: its CUB histogram scratch was
    not bounded by the old estimate and could request 12 GiB at 10M rows.
    Sorted-key lower bounds have an explicit row/cell footprint instead.
    """
    row_count = checked_i32(row_count, name="point-grid row count")
    cell_count = checked_i32(cell_count, name="point-grid cell count")
    query_count = checked_i32(query_count, name="point-grid query count")
    pair_budget = int(pair_budget)
    if pair_budget < 0:
        raise ValueError("point-grid pair budget must be nonnegative")
    # Optional bounds(32), finite(1), x/y(8), ids(8), rows(4), radix
    # outputs(12), conservative radix workspace(64), and allocator slack.
    row_bytes = 161 if needs_device_bounds else 129
    # thresholds(8), lower-bound positions(8), counts(4), offsets(8),
    # integral image(8), and two cumsum workspaces plus slack.
    cell_bytes = 76
    # bounds/counts/offsets/cursors plus guarded candidate tile and error flag.
    query_bytes = 48
    # Candidate columns/guard plus the existing exact-refinement scratch.
    pair_bytes = 73
    return checked_sum(
        checked_product(row_count, row_bytes, name="point-grid row bytes"),
        checked_product(cell_count, cell_bytes, name="point-grid cell bytes"),
        checked_product(query_count, query_bytes, name="point-grid query bytes"),
        checked_product(pair_budget, pair_bytes, name="point-grid pair bytes"),
        1 << 20,
        name="point-grid stage bytes",
    )


def point_grid_preflight(
    native_index,
    *,
    query_count: int,
    pair_budget: int,
    force_eligible: bool = False,
) -> tuple[PointPartitionPreflight | None, PointPartitionDecline | None]:
    """Inspect grid shape and bytes without building or querying the provider."""
    flat_index = native_index.to_flat_index()
    owned = flat_index.geometry_array
    grid_size = _grid_size_for_rows(owned.row_count)
    key = point_partition_cache_key(
        native_index,
        PointPartitionVariant.GRID,
        parameters={"grid_size": grid_size},
    )
    cached = cached_point_partition(native_index, key)
    if (
        (
            cached is None
            and owned.row_count < _MIN_POINT_GRID_ROWS
            and not force_eligible
        )
        or set(owned.families) != {GeometryFamily.POINT}
    ):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid is ineligible for this row count or geometry family",
        )
    if not point_partition_all_bounds_finite(native_index):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid declines nonfinite point rows",
        )
    xmin, ymin, xmax, ymax = map(float, flat_index.total_bounds)
    if not all(isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid declines nonfinite aggregate point bounds",
        )
    if _point_grid_extents(xmin, ymin, xmax, ymax) is None:
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid requires finite positive x and y extents",
        )
    metadata = native_index.metadata
    device_bounds = None if metadata is None else metadata.bounds
    if device_bounds is None or not hasattr(device_bounds, "__cuda_array_interface__"):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid requires device-resident NativeSpatialIndex bounds",
        )
    cell_count = grid_size * grid_size
    required = _point_grid_required_bytes(
        0 if cached is not None else owned.row_count,
        0 if cached is not None else cell_count,
        needs_device_bounds=(
            cached is None and getattr(flat_index, "device_bounds", None) is None
        ),
        query_count=query_count,
        pair_budget=pair_budget,
    )
    return PointPartitionPreflight(
        PointPartitionVariant.GRID,
        required,
        "dense point grid passed structural and complete-memory preflight",
        id(native_index),
        int(query_count),
        int(pair_budget),
        key,
    ), None


def prepare_point_grid_index(
    native_index,
    *,
    query_count: int = 0,
    pair_budget: int = 0,
    force_eligible: bool = False,
    admission=None,
) -> tuple[PreparedPointGridIndex | None, PointPartitionDecline | None]:
    """Build or return the NativeSpatialIndex-owned dense point grid."""
    with hotpath_stage(
        "spatial.point_grid_index.prepare",
        category="setup",
    ) as stage_metadata:
        with native_index.point_partition_lock:
            cache_entries_before = (
                len(native_index.point_partition_cache)
                if stage_metadata is not None
                else None
            )
            prepared, decline = _prepare_point_grid_index_locked(
                native_index,
                query_count=query_count,
                pair_budget=pair_budget,
                force_eligible=force_eligible,
                admission=admission,
            )
            if stage_metadata is not None:
                assert cache_entries_before is not None
                cache_entries_after = len(native_index.point_partition_cache)
                built = prepared is not None and cache_entries_after > cache_entries_before
                cache_hit = (
                    prepared is not None
                    and cache_entries_after == cache_entries_before
                )
                sums, maxima, unavailable = _point_grid_preparation_metrics(
                    prepared,
                    built=built,
                    cache_hit=cache_hit,
                    declined=prepared is None,
                    query_count=query_count,
                    pair_budget=pair_budget,
                )
                attach_work_amplification(
                    stage_metadata,
                    operation="prepare_point_grid_index",
                    metric_family="rebuild",
                    sums=sums,
                    maxima=maxima,
                    unavailable=unavailable,
                    physical_shape="reusable_point_partition_index",
                    consumer_kind="point_partition_query",
                    semantic_contract={
                        "cache_scope": "NativeSpatialIndex point-partition cache",
                        "cache_identity_exported": False,
                        "device_logical_counts_read": False,
                    },
                )
        return prepared, decline


def _prepare_point_grid_index_locked(
    native_index,
    *,
    query_count: int,
    pair_budget: int,
    force_eligible: bool,
    admission,
) -> tuple[PreparedPointGridIndex | None, PointPartitionDecline | None]:
    wait_for_point_partition(native_index.readiness)
    preflight, decline = point_grid_preflight(
        native_index,
        query_count=query_count,
        pair_budget=pair_budget,
        force_eligible=force_eligible,
    )
    if preflight is None:
        return None, decline
    if admission is not None:
        admission.validate_admission(
            native_index,
            PointPartitionVariant.GRID,
            query_count=query_count,
            pair_budget=pair_budget,
            cache_key=preflight.cache_key,
            required_bytes=preflight.required_bytes,
        )
    flat_index = native_index.to_flat_index()
    owned = flat_index.geometry_array
    grid_size = _grid_size_for_rows(owned.row_count)
    key = point_partition_cache_key(
        native_index,
        PointPartitionVariant.GRID,
        parameters={"grid_size": grid_size},
    )
    cached = cached_point_partition(native_index, key)
    if cached is not None:
        query_required = _point_grid_required_bytes(
            0,
            0,
            needs_device_bounds=False,
            query_count=query_count,
            pair_budget=pair_budget,
        )
        if admission is None:
            memory_admission = get_cuda_runtime().admit_device_memory(
                stage="spatial.point_grid_query",
                required_bytes=query_required,
                requested_units=query_count,
            )
            if not memory_admission.admitted:
                return None, PointPartitionDecline(
                    PointPartitionVariant.GRID,
                    (
                        "dense point grid query preflight declined "
                        f"{query_required} bytes with "
                        f"{memory_admission.remaining_bytes} remaining"
                    ),
                    memory_decline=True,
                )
        wait_for_point_partition(cached.readiness)
        return cached, None
    # Eligibility and finite-row checks were completed before any provider work.
    xmin, ymin, xmax, ymax = map(float, flat_index.total_bounds)
    if not all(isfinite(value) for value in (xmin, ymin, xmax, ymax)):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid declines nonfinite aggregate point bounds",
        )
    extents = _point_grid_extents(xmin, ymin, xmax, ymax)
    if extents is None:
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid requires finite positive x and y extents",
        )
    x_extent, y_extent = extents

    import cupy as cp

    cell_count = grid_size * grid_size
    runtime = get_cuda_runtime()
    required_bytes = _point_grid_required_bytes(
        owned.row_count,
        cell_count,
        needs_device_bounds=flat_index.device_bounds is None,
        query_count=query_count,
        pair_budget=pair_budget,
    )
    if admission is None:
        memory_admission = runtime.admit_device_memory(
            stage="spatial.point_grid_index",
            required_bytes=required_bytes,
            requested_units=owned.row_count,
        )
        if not memory_admission.admitted:
            return None, PointPartitionDecline(
                PointPartitionVariant.GRID,
                (
                    "dense point grid preflight declined "
                    f"{required_bytes} bytes with "
                    f"{memory_admission.remaining_bytes} remaining"
                ),
                memory_decline=True,
            )

    metadata = native_index.metadata
    device_bounds = None if metadata is None else metadata.bounds
    if device_bounds is None or not hasattr(device_bounds, "__cuda_array_interface__"):
        return None, PointPartitionDecline(
            PointPartitionVariant.GRID,
            "dense point grid requires device-resident NativeSpatialIndex bounds",
        )
    precision_plan = point_partition_fp64_coarse_plan()
    bounds = cp.asarray(device_bounds, dtype=cp.float64).reshape(-1, 4)
    finite = cp.isfinite(bounds).all(axis=1)
    point_x = cp.clip(bounds[:, 0], xmin, xmax)
    point_y = cp.clip(bounds[:, 1], ymin, ymax)
    cell_x = cp.floor(
        (point_x - xmin) * (grid_size / x_extent)
    ).astype(cp.int32, copy=False)
    cell_y = cp.floor(
        (point_y - ymin) * (grid_size / y_extent)
    ).astype(cp.int32, copy=False)
    cell_x = cp.clip(cell_x, 0, grid_size - 1)
    cell_y = cp.clip(cell_y, 0, grid_size - 1)
    invalid_cell = np.uint64(cell_count)
    cell_ids = cp.where(
        finite,
        cell_y.astype(cp.uint64) * np.uint64(grid_size) + cell_x.astype(cp.uint64),
        invalid_cell,
    )
    rows = cp.arange(owned.row_count, dtype=cp.int32)
    sorted_pairs = sort_pairs(
        cell_ids,
        rows,
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    sorted_cell_ids = cp.asarray(sorted_pairs.keys, dtype=cp.uint64)
    sorted_tree_rows = cp.asarray(sorted_pairs.values, dtype=cp.int32)
    cell_thresholds = cp.arange(cell_count + 1, dtype=cp.uint64)
    cell_boundaries = cp.searchsorted(
        sorted_cell_ids,
        cell_thresholds,
        side="left",
    )
    cell_counts = cp.ascontiguousarray(
        (cell_boundaries[1:] - cell_boundaries[:-1]).astype(cp.int32, copy=False)
    )
    cell_offsets = exclusive_sum(cell_counts.astype(cp.int64), synchronize=False)
    integral_counts = cp.zeros((grid_size + 1, grid_size + 1), dtype=cp.int64)
    interior = integral_counts[1:, 1:]
    interior[...] = cell_counts.reshape(grid_size, grid_size)
    cp.cumsum(interior, axis=0, out=interior)
    cp.cumsum(interior, axis=1, out=interior)
    readiness = record_point_partition_readiness()
    built = PreparedPointGridIndex(
        grid_size=grid_size,
        cache_key=key,
        precision_plan=precision_plan,
        readiness=readiness,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        sorted_tree_rows=sorted_tree_rows,
        cell_counts=cell_counts,
        cell_offsets=cell_offsets,
        integral_counts=integral_counts,
    )
    prepared = publish_point_partition(native_index, key, built)
    retain_point_partition_completion(
        native_index,
        owned,
        bounds,
        finite,
        point_x,
        point_y,
        cell_x,
        cell_y,
        cell_ids,
        rows,
        sorted_cell_ids,
        cell_thresholds,
        cell_boundaries,
        built,
        prepared,
    )
    record_dispatch_event(
        surface="vibespatial.spatial.point_grid_index",
        operation="prepare",
        implementation="dense_point_cell_index_gpu",
        reason=(
            f"grouped {owned.row_count} point rows once into "
            f"{grid_size}x{grid_size} cells"
        ),
        selected=ExecutionMode.GPU,
    )
    return prepared, None


def _point_grid_query_counts(prepared, bounds):
    """Count cell-superset rows for fp64 query bounds on device."""
    import cupy as cp

    wait_for_point_partition(prepared.readiness)
    bounds = cp.ascontiguousarray(cp.asarray(bounds, dtype=cp.float64)).reshape(-1, 4)
    query_count = int(bounds.shape[0])
    counts = cp.empty(query_count, dtype=cp.int64)
    if query_count == 0:
        return bounds, counts
    runtime = get_cuda_runtime()
    kernel = point_grid_index_kernels()["point_grid_query_counts"]
    grid, block = runtime.launch_config(kernel, query_count)
    ptr = runtime.pointer
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(bounds),
                prepared.xmin,
                prepared.ymin,
                prepared.xmax,
                prepared.ymax,
                prepared.grid_size,
                ptr(prepared.integral_counts),
                ptr(counts),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    return bounds, counts


def point_grid_relation_superset_query(native_index, query_bounds):
    """Materialize one admitted full relation candidate superset.

    Relation-producing public APIs are pair-shaped by contract.  They retain a
    single named allocation fence for the exact output capacity, then use the
    same provenance-bound guarded scatter as bounded reducers.
    """
    import cupy as cp

    bounds = cp.ascontiguousarray(cp.asarray(query_bounds, dtype=cp.float64)).reshape(
        -1,
        4,
    )
    query_count = int(bounds.shape[0])
    # The count/scatter kernels reject nonfinite query rows in-device as empty
    # windows.  Full relation output therefore needs no host finite-proof
    # planning packet; null/empty rows simply contribute no candidate pairs.
    prepared, decline = prepare_point_grid_index(
        native_index,
        query_count=query_count,
        pair_budget=0,
    )
    if prepared is None:
        return None, decline
    bounds, counts = _point_grid_query_counts(prepared, bounds)
    if query_count == 0:
        return _DeviceCandidates(
            d_left=cp.empty(0, dtype=cp.int32),
            d_right=cp.empty(0, dtype=cp.int32),
            total_pairs=0,
        ), None
    runtime = get_cuda_runtime()
    offsets = exclusive_sum(counts, synchronize=False)
    try:
        total_pairs = count_scatter_total(
            runtime,
            counts,
            offsets,
            reason="point-grid relation candidate allocation fence",
        )
    finally:
        runtime.free(offsets)
    if total_pairs == 0:
        empty = _DeviceCandidates(
            d_left=cp.empty(0, dtype=cp.int32),
            d_right=cp.empty(0, dtype=cp.int32),
            total_pairs=0,
        )
        retain_point_partition_completion(
            native_index,
            prepared,
            bounds,
            counts,
            empty,
        )
        return empty, None
    relation_required = _point_grid_required_bytes(
        0,
        0,
        needs_device_bounds=False,
        query_count=query_count,
        pair_budget=total_pairs,
    )
    memory_admission = runtime.admit_device_memory(
        stage="spatial.point_grid_relation_complete",
        required_bytes=relation_required,
        requested_units=total_pairs,
    )
    if not memory_admission.admitted:
        raise CandidateRelationCapacityError(
            "dense point-grid relation complete-stage admission declined after "
            "candidate counting submitted; refusing a post-submission Morton "
            f"retry ({relation_required} bytes required with "
            f"{memory_admission.remaining_bytes} remaining)"
        )
    relation_admission = PointPartitionPreflight(
        PointPartitionVariant.GRID,
        relation_required,
        "dense point-grid relation passed complete candidate/exact admission",
        id(native_index),
        query_count,
        total_pairs,
        prepared.cache_key,
        admitted=True,
    )
    partitions = ((0, query_count, total_pairs),)
    plan = query_plan(
        owner=native_index,
        variant=PointPartitionVariant.GRID,
        prepared=prepared,
        query_bounds=bounds,
        query_counts=counts,
        partitions=partitions,
        pair_budget=total_pairs,
        relation_admission=relation_admission,
    )
    candidates = point_grid_superset_query(native_index, next(plan.slices()))
    retain_point_partition_completion(native_index, prepared, plan, bounds, counts)
    return candidates, None


def point_grid_superset_query(
    native_index,
    query_slice: PointPartitionQuerySlice,
) -> _DeviceCandidates | None:
    """Return cell-conservative pairs for immediate exact refinement."""
    prepared = query_slice.plan.prepared
    query_slice.validate(native_index, PointPartitionVariant.GRID, prepared)
    relation_admission = query_slice.plan.relation_admission
    if relation_admission is not None:
        relation_required = _point_grid_required_bytes(
            0,
            0,
            needs_device_bounds=False,
            query_count=int(query_slice.plan.query_bounds.shape[0]),
            pair_budget=int(query_slice.plan.pair_budget),
        )
        relation_admission.validate_admission(
            native_index,
            PointPartitionVariant.GRID,
            query_count=int(query_slice.plan.query_bounds.shape[0]),
            pair_budget=int(query_slice.plan.pair_budget),
            cache_key=prepared.cache_key,
            required_bytes=relation_required,
        )
    wait_for_point_partition(query_slice.plan.readiness)

    import cupy as cp

    runtime = get_cuda_runtime()
    bounds = cp.ascontiguousarray(
        cp.asarray(query_slice.query_bounds, dtype=cp.float64)
    ).reshape(-1, 4)
    query_count = int(bounds.shape[0])
    if query_count == 0:
        return _DeviceCandidates(
            d_left=cp.empty(0, dtype=cp.int32),
            d_right=cp.empty(0, dtype=cp.int32),
            total_pairs=0,
        )
    query_counts = cp.asarray(query_slice.query_counts, dtype=cp.int64)
    if int(query_counts.size) != query_count:
        raise ValueError("point-grid query token counts must align to query rows")
    pair_capacity = int(query_slice.capacity)
    query_offsets = None
    query_cursors = None
    out_left = None
    out_right = None
    kernels = point_grid_index_kernels()
    ptr = runtime.pointer
    try:
        query_offsets = exclusive_sum(query_counts, synchronize=False)
        if relation_admission is None:
            require_device_candidate_pair_capacity(
                pair_capacity,
                relation_name="capacity-backed point-grid candidate tile",
            )
        out_left = cp.zeros(pair_capacity, dtype=cp.int32)
        out_right = cp.zeros(pair_capacity, dtype=cp.int32)
        total_pairs = pair_capacity
        if total_pairs == 0:
            return _DeviceCandidates(
                d_left=cp.empty(0, dtype=cp.int32),
                d_right=cp.empty(0, dtype=cp.int32),
                total_pairs=0,
            )
        query_cursors = query_offsets.astype(cp.uint64, copy=True)
        error_flag = cp.zeros(1, dtype=cp.uint32)
        scatter_kernel = kernels["point_grid_query_scatter"]
        scatter_threads = runtime.optimal_block_size(scatter_kernel)
        runtime.launch(
            scatter_kernel,
            grid=(query_count, 1, 1),
            block=(scatter_threads, 1, 1),
            params=(
                (
                    ptr(bounds),
                    prepared.xmin,
                    prepared.ymin,
                    prepared.xmax,
                    prepared.ymax,
                    prepared.grid_size,
                    ptr(prepared.cell_counts),
                    ptr(prepared.cell_offsets),
                    ptr(prepared.sorted_tree_rows),
                    ptr(query_offsets),
                    ptr(query_counts),
                    ptr(query_cursors),
                    ptr(out_left),
                    ptr(out_right),
                    total_pairs,
                    ptr(error_flag),
                    query_count,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_F64,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I64,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        result = _DeviceCandidates(
            d_left=out_left,
            d_right=out_right,
            total_pairs=total_pairs,
            error_flag=error_flag,
        )
        retain_point_partition_completion(
            native_index,
            query_slice,
            prepared,
            bounds,
            query_counts,
            query_offsets,
            query_cursors,
            error_flag,
            out_left,
            out_right,
        )
        out_left = None
        out_right = None
        return result
    finally:
        runtime.free(query_offsets)
        runtime.free(query_cursors)
        runtime.free(out_left)
        runtime.free(out_right)


def point_grid_query_row_partitions(
    native_index,
    query_bounds,
    *,
    pair_budget: int,
    force_eligible: bool = False,
    admission=None,
) -> tuple[PointPartitionQueryPlan | None, PointPartitionDecline | None]:
    """Plan query-row slices whose point-grid supersets fit a pair budget.

    Fixed-size row blocks are reduced on-device and cross once as compact
    planning metadata. Greedy block slices use exact count sums as capacities,
    so every admitted tile fits ``pair_budget`` without a per-tile allocation
    fence, padded launch, or full row-count export. Oversized blocks are refined
    down to single-row metadata; consumers then scan those rows in bounded
    dense tree-row tiles.
    """
    import cupy as cp

    bounds = cp.ascontiguousarray(cp.asarray(query_bounds, dtype=cp.float64)).reshape(
        -1,
        4,
    )
    query_count = int(bounds.shape[0])
    prepared, decline = prepare_point_grid_index(
        native_index,
        query_count=query_count,
        pair_budget=pair_budget,
        force_eligible=force_eligible,
        admission=admission,
    )
    if prepared is None:
        return None, decline
    if pair_budget <= 0:
        raise ValueError("point-grid pair budget must be positive")

    runtime = get_cuda_runtime()
    if query_count == 0:
        return (
            query_plan(
                owner=native_index,
                variant=PointPartitionVariant.GRID,
                prepared=prepared,
                query_bounds=bounds,
                query_counts=cp.empty(0, dtype=cp.int64),
                partitions=(),
                pair_budget=pair_budget,
            ),
            None,
        )
    bounds, counts = _point_grid_query_counts(prepared, bounds)
    max_packet_values = _MAX_PLANNING_PACKET_BYTES // np.dtype(np.int64).itemsize
    block_size = 32
    while (query_count + block_size - 1) // block_size > max_packet_values:
        block_size *= 2
    while True:
        full_row_count = (query_count // block_size) * block_size
        block_counts_parts = []
        if full_row_count:
            block_counts_parts.append(
                counts[:full_row_count]
                .reshape(-1, block_size)
                .sum(axis=1, dtype=cp.int64)
            )
        if full_row_count < query_count:
            block_counts_parts.append(
                counts[full_row_count:].sum(dtype=cp.int64).reshape(1)
            )
        block_counts = (
            block_counts_parts[0]
            if len(block_counts_parts) == 1
            else cp.concatenate(block_counts_parts)
        )
        if int(block_counts.nbytes) > _MAX_PLANNING_PACKET_BYTES:
            return None, PointPartitionDecline(
                PointPartitionVariant.GRID,
                "dense point grid query planning packet exceeds 64 KiB",
            )
        block_counts_host = runtime.copy_device_to_host(
            block_counts,
            reason="point-grid reduction block-count planning packet",
        )
        partitions = _count_bounded_block_partitions(
            np.asarray(block_counts_host, dtype=np.int64),
            block_size=block_size,
            query_count=query_count,
            pair_budget=pair_budget,
        )
        if partitions is not None:
            return (
                query_plan(
                    owner=native_index,
                    variant=PointPartitionVariant.GRID,
                    prepared=prepared,
                    query_bounds=bounds,
                    query_counts=counts,
                    partitions=partitions,
                    pair_budget=pair_budget,
                ),
                None,
            )
        if block_size == 1:
            partitions = _count_bounded_block_partitions(
                    np.asarray(block_counts_host, dtype=np.int64),
                    block_size=1,
                    query_count=query_count,
                    pair_budget=pair_budget,
                    admit_oversized=True,
            )
            return (
                query_plan(
                    owner=native_index,
                    variant=PointPartitionVariant.GRID,
                    prepared=prepared,
                    query_bounds=bounds,
                    query_counts=counts,
                    partitions=partitions,
                    pair_budget=pair_budget,
                ),
                None,
            )
        block_size //= 2


def _count_bounded_block_partitions(
    block_counts: np.ndarray,
    *,
    block_size: int,
    query_count: int,
    pair_budget: int,
    admit_oversized: bool = False,
) -> tuple[tuple[int, int, int], ...] | None:
    """Greedily pack exact device-reduced row-block counts under a budget."""
    block_counts = np.asarray(block_counts, dtype=np.int64)
    block_size = int(block_size)
    query_count = int(query_count)
    pair_budget = int(pair_budget)
    if (
        block_counts.ndim != 1
        or np.any(block_counts < 0)
        or block_size <= 0
        or query_count < 0
        or pair_budget <= 0
    ):
        raise ValueError("point-grid block partition dimensions must be nonnegative")
    if query_count == 0:
        return ()
    expected_blocks = (query_count + block_size - 1) // block_size
    if int(block_counts.size) != expected_blocks:
        raise ValueError("point-grid block counts must cover every query row")
    if np.any(block_counts > pair_budget) and not admit_oversized:
        return None
    if admit_oversized and block_size != 1:
        raise ValueError("oversized point-grid blocks require single-row metadata")
    partitions: list[tuple[int, int, int]] = []
    block_start = 0
    capacity = 0
    for block, count_value in enumerate(block_counts):
        count = int(count_value)
        if count > pair_budget:
            if block > block_start:
                partitions.append((
                    block_start * block_size,
                    min(block * block_size, query_count),
                    capacity,
                ))
            partitions.append((
                block * block_size,
                min((block + 1) * block_size, query_count),
                count,
            ))
            block_start = block + 1
            capacity = 0
            continue
        if block > block_start and capacity + count > pair_budget:
            partitions.append(
                (
                    block_start * block_size,
                    min(block * block_size, query_count),
                    capacity,
                )
            )
            block_start = block
            capacity = 0
        capacity += count
    if block_start * block_size < query_count:
        partitions.append(
            (
                block_start * block_size,
                query_count,
                capacity,
            )
        )
    return tuple(partitions)


def point_grid_candidate_not_in_other_superset(
    native_index,
    prior_plan: PointPartitionQueryPlan,
    candidate_query_rows,
    candidate_tree_rows,
):
    """Mark candidates absent from an aligned point grid's prior superset.

    This is an internal paired-reduction primitive.  Candidate rows refer to
    two aligned point columns, so ``candidate_tree_rows`` can address the
    geometry owned by ``flat_index`` directly.  Oversized query rows are marked
    already seen because their prior reduction scanned every aligned tree row.
    """
    prepared = prior_plan.prepared
    prior_plan.validate(native_index, PointPartitionVariant.GRID, prepared)
    wait_for_point_partition(prior_plan.readiness)

    import cupy as cp

    d_left = cp.asarray(candidate_query_rows, dtype=cp.int32)
    d_right = cp.asarray(candidate_tree_rows, dtype=cp.int32)
    if d_left.ndim != 1 or d_right.ndim != 1 or d_left.size != d_right.size:
        raise ValueError("point-grid candidate rows must be aligned vectors")
    flat_index = native_index.to_flat_index()
    owned = flat_index.geometry_array
    state = owned._ensure_device_state(preserve_indexed_view=True)
    point_buffer = state.families.get(GeometryFamily.POINT)
    if point_buffer is None:
        return None
    bounds = cp.ascontiguousarray(cp.asarray(prior_plan.query_bounds, dtype=cp.float64)).reshape(
        -1,
        4,
    )
    counts = cp.asarray(prior_plan.query_counts, dtype=cp.int64)
    out = cp.empty(d_left.size, dtype=cp.uint8)
    if d_left.size == 0:
        return out.astype(cp.bool_, copy=False)
    runtime = get_cuda_runtime()
    kernel = point_grid_index_kernels()[
        "point_grid_candidate_not_in_other_superset"
    ]
    grid, block = runtime.launch_config(kernel, int(d_left.size))
    ptr = runtime.pointer
    runtime.launch(
        kernel,
        grid=grid,
        block=block,
        params=(
            (
                ptr(d_left),
                ptr(d_right),
                ptr(bounds),
                ptr(counts),
                ptr(state.family_row_offsets),
                ptr(point_buffer.geometry_offsets),
                ptr(point_buffer.empty_mask),
                ptr(point_buffer.x),
                ptr(point_buffer.y),
                prepared.xmin,
                prepared.ymin,
                prepared.xmax,
                prepared.ymax,
                prepared.grid_size,
                int(prior_plan.pair_budget),
                ptr(out),
                int(d_left.size),
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
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        ),
    )
    result = out.astype(cp.bool_, copy=False)
    retain_point_partition_completion(
        native_index,
        prior_plan,
        prepared,
        d_left,
        d_right,
        bounds,
        counts,
        state,
        point_buffer,
        out,
        result,
    )
    return result
