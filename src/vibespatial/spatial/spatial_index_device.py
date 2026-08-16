"""Device-side spatial index query — unified GPU BVH-style traversal.

Provides ``spatial_index_device_query``, the single entry point for GPU-
accelerated spatial index traversal in sjoin and other bulk query paths.

Strategy selection (transparent to caller):
  - *Brute-force O(N*M)*: each query thread scans all tree rows.
    Optimal when M is small or the total work (N*M) fits in a few waves.
  - *Morton range O(N*log(M)+K)*: uses pre-sorted Morton keys with CCCL
    binary search to narrow the scan range per query, then refines with
    bbox overlap.  Better for large M where most tree rows are distant.

ADR-0002: COARSE kernel class — bbox comparisons stay fp64 (bounds kernels
are memory-bound, not compute-bound, so fp32 provides no throughput
advantage, and fp32 rounding could shrink bounds causing false negatives).

ADR-0033 tier classification:
  - Tier 1 (NVRTC): bbox overlap count/scatter, Morton range computation
  - Tier 3a (CCCL): exclusive_sum, lower_bound, upper_bound, compact_indices
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - exercised on CPU-only installs
    cp = None

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_I64,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    PairSortStrategy,
    compact_indices,
    exclusive_sum,
    lower_bound,
    sort_pairs,
    upper_bound,
)
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS
from vibespatial.kernels.core.spatial_query_kernels import (
    _morton_range_kernels,
    _spatial_query_kernels,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import PhysicalWorkEstimate
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    PrecisionPlan,
)

from .query_types import (
    _POLYGON_DE9IM_PREDICATES,
    SpatialQueryExecution,
    _DeviceCandidates,
    available_device_memory_bytes,
    require_device_candidate_pair_capacity,
)

# Eagerly request CCCL spec warmup at module import (ADR-0034 Level 1).
request_warmup(
    [
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "select_i32",
        "select_i64",
        "lower_bound_u64",
        "upper_bound_u64",
        "radix_sort_i32_i32",
    ]
)


# ---------------------------------------------------------------------------
# Strategy selection thresholds
# ---------------------------------------------------------------------------
# Morton range has higher fixed overhead (6 kernel launches vs 2 for brute-
# force), but scales as O(N*log(M)+K) instead of O(N*M).
# Crossover: when N*M exceeds this, prefer Morton range.  Benchmarked at
# ~1M (roughly 1K x 1K) with warm kernels (ADR-0034) — the overhead delta
# is ~0.5ms, dominated by the binary search + range expansion cost.
_MORTON_RANGE_CROSSOVER = 1_000_000

# Small many-by-few bbox joins are bounded by the pair product.  A device
# pair-mask plus compaction avoids scalarizing the count/scan output size.
_PAIR_MASK_BRUTE_FORCE_MAX_PRODUCT = 262_144
_SEMIJOIN_MAX_CANDIDATES_PER_ROW = 8 * 1024
_SEMIJOIN_MAX_TILE_LANES = 16 * 1024 * 1024
_SEMIJOIN_MAX_SEGMENT_PAIR_LANES = 8 * 1024 * 1024
_SEMIJOIN_TILE_BYTES_PER_LANE = 64
_MORTON_SPAN_BUCKET_UPPER_BOUNDS = (0,) + tuple(1 << exponent for exponent in range(32))


def _spatial_reduction_tile_lane_capacity(
    query_owned,
    tree_owned,
    *,
    predicate: str | None,
    family_admission: tuple[bool, bool, bool] | None,
) -> int:
    """Bound one reduction tile by live bytes and exact-refine work shape."""
    available_bytes = available_device_memory_bytes()
    memory_lanes = (
        _SEMIJOIN_MAX_TILE_LANES
        if available_bytes is None
        else max(
            1,
            int(available_bytes) // 4 // _SEMIJOIN_TILE_BYTES_PER_LANE,
        )
    )
    segment_lanes = _SEMIJOIN_MAX_TILE_LANES
    if predicate is not None and family_admission is not None and not family_admission[0]:
        from vibespatial.geometry.owned import ensure_device_geometry_size_bounds

        query_segment_span = ensure_device_geometry_size_bounds(
            query_owned,
            reason="spatial reduction query segment-span planning packet",
        )
        tree_segment_span = ensure_device_geometry_size_bounds(
            tree_owned,
            reason="spatial reduction tree segment-span planning packet",
        )
        segment_pair_span = max(
            int(query_segment_span) * int(tree_segment_span),
            1,
        )
        segment_lanes = max(
            1,
            _SEMIJOIN_MAX_SEGMENT_PAIR_LANES // segment_pair_span,
        )
    return max(
        1,
        min(_SEMIJOIN_MAX_TILE_LANES, memory_lanes, segment_lanes),
    )


def prefers_pair_mask_spatial_index_query(query_count: int, tree_count: int) -> bool:
    """Return True when device pair-mask query is the preferred physical shape."""
    query_count = int(query_count)
    tree_count = int(tree_count)
    return (
        1 < query_count
        and 1 < tree_count
        and query_count * tree_count <= _PAIR_MASK_BRUTE_FORCE_MAX_PRODUCT
    )


def _is_device_array(value) -> bool:
    return hasattr(value, "__cuda_array_interface__")


def _empty_device_candidates() -> _DeviceCandidates | None:
    if cp is None:  # pragma: no cover - exercised on CPU-only installs
        return None
    return _DeviceCandidates(
        d_left=cp.empty(0, dtype=cp.int32),
        d_right=cp.empty(0, dtype=cp.int32),
        total_pairs=0,
    )


def _prepare_query_bounds_device(
    bounds,
    runtime,
):
    """Return `(device_bounds_flat, temp_allocation)` for kernel launch input."""
    if _is_device_array(bounds):
        if cp is None:  # pragma: no cover - exercised on CPU-only installs
            raise RuntimeError("CuPy is not installed; device bounds are unavailable")
        base = cp.asarray(bounds)
        prepared = base
        if prepared.dtype != cp.float64:
            prepared = prepared.astype(cp.float64, copy=True)
        if not prepared.flags.c_contiguous:
            prepared = cp.ascontiguousarray(prepared)
        return prepared.ravel(), None if prepared is base else prepared

    device_bounds = runtime.from_host(np.ascontiguousarray(bounds, dtype=np.float64).ravel())
    return device_bounds, device_bounds


def _prepare_tree_bounds_device(
    flat_index,
    runtime,
):
    """Return cached device bounds for indexed tree rows.

    Public ``GeometryArray.sindex`` construction deliberately permits a cheap
    host build.  A later GPU query must not interpret that host residency as a
    reason to discard the index shape and scan ``N * M`` pairs.  Hydrate the
    compact fp64 bounds once and retain them on the reusable flat index.
    """
    device_bounds = getattr(flat_index, "device_bounds", None)
    if device_bounds is not None:
        if cp is None:  # pragma: no cover - exercised on CPU-only installs
            raise RuntimeError("CuPy is not installed; device bounds are unavailable")
        base = cp.asarray(device_bounds)
        prepared = base
        if prepared.dtype != cp.float64:
            prepared = prepared.astype(cp.float64, copy=True)
        if not prepared.flags.c_contiguous:
            prepared = cp.ascontiguousarray(prepared)
        return prepared.ravel(), None if prepared is base else prepared

    d_bounds = runtime.from_host(
        np.ascontiguousarray(flat_index.bounds, dtype=np.float64)
    )
    object.__setattr__(flat_index, "device_bounds", d_bounds)
    return d_bounds.ravel(), None


def spatial_index_device_query(
    flat_index,
    query_bounds,
    *,
    distance: np.ndarray | object | None = None,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
    allow_bbox_superset: bool = False,
) -> tuple[_DeviceCandidates | None, SpatialQueryExecution]:
    """GPU-accelerated spatial index query — replaces CPU STRtree traversal.

    Parameters
    ----------
    flat_index : FlatSpatialIndex
        Pre-built spatial index with Morton keys and sorted order.
    query_bounds : np.ndarray, shape (N, 4)
        Query bounding boxes as ``[minx, miny, maxx, maxy]`` rows.
    distance : np.ndarray or None
        Per-row distance expansion for dwithin queries.  When provided,
        query bounds are expanded by the corresponding distance before
        bbox overlap testing.
    precision : PrecisionMode
        Precision mode override.  COARSE class — bounds stay fp64 on all
        devices (memory-bound; fp32 rounding causes false negatives).

    Returns
    -------
    (candidates, execution) : tuple[_DeviceCandidates | None, SpatialQueryExecution]
        ``candidates`` is None when GPU dispatch is skipped. When GPU dispatch
        runs and finds no pairs, ``candidates`` is an empty device-resident
        result. ``execution`` carries the dispatch decision metadata.
    """
    query_count = int(query_bounds.shape[0])
    tree_count = flat_index.size

    # This helper is already the GPU-only candidate-generation path.
    # Use the planner for precision/device-profile wiring, but explicitly
    # pin GPU so the caller's "try the device path first" intent remains
    # visible instead of re-running AUTO crossover selection here.
    try:
        plan = plan_dispatch_selection(
            kernel_name="bbox_overlap_candidates",
            kernel_class=KernelClass.COARSE,
            row_count=query_count,
            requested_mode=ExecutionMode.GPU,
            requested_precision=precision,
            gpu_available=has_gpu_runtime(),
            work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
                row_count=query_count,
                candidate_pair_count=query_count * tree_count,
                primary_unit_name="bbox-candidate-pair",
            ),
        )
    except RuntimeError:
        return None, SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="GPU runtime unavailable; skipping device spatial index query",
        )
    _precision_plan = plan.precision_plan
    # Bounds kernels are memory-bound: fp64 is correct and necessary.
    assert _precision_plan.compute_precision in (
        PrecisionMode.FP64,
        PrecisionMode.FP32,
    ), "PrecisionPlan must resolve to a concrete mode"
    if plan.selected is not ExecutionMode.GPU:
        return None, SpatialQueryExecution(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="device spatial index query planner did not resolve a GPU dispatch",
        )

    if query_count == 0 or tree_count == 0:
        return _empty_device_candidates(), SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="empty input; no candidate pairs to generate",
        )

    # Expand bounds for dwithin if distance is provided.
    effective_bounds = query_bounds
    if distance is not None:
        effective_bounds = _expand_bounds_for_distance(query_bounds, distance)

    if allow_bbox_superset:
        from .point_grid_index import point_grid_superset_query

        result = point_grid_superset_query(flat_index, effective_bounds)
        if result is not None:
            return result, SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason=(
                    "cached point-grid cell superset for immediate exact "
                    f"predicate refinement: N={query_count}, M={tree_count}"
                ),
            )

    # Strategy selection: Morton range vs brute-force.
    n_product = query_count * tree_count
    total_bounds = getattr(flat_index, "total_bounds", None)
    has_morton = (
        total_bounds is not None
        and not np.isnan(total_bounds[0])
        # Regular-grid indexes deliberately store identity keys/order because
        # their exact grid metadata is the index.  Those placeholders are not
        # Morton codes and must never be hydrated into the Morton range path.
        and getattr(flat_index, "regular_grid", None) is None
        and (
            getattr(flat_index, "device_morton_keys", None) is not None
            or (
                getattr(flat_index, "_host_morton_keys", None) is not None
                and getattr(flat_index, "_host_order", None) is not None
            )
        )
    )

    if has_morton and n_product >= _MORTON_RANGE_CROSSOVER:
        result = _morton_range_query(flat_index, query_bounds, effective_bounds)
        if result is not None:
            return result, SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason=(
                    f"Morton range O(N*log(M)+K) query: "
                    f"N={query_count}, M={tree_count}, N*M={n_product:,}"
                ),
            )

    # Fall through to brute-force O(N*M).  A one-row tree has bounded output
    # cardinality (at most one pair per query), so use mask compaction instead
    # of the generic count/scan/scatter path that scalarizes total pairs for
    # allocation.
    if tree_count == 1 and query_count > 1:
        result = _brute_force_single_tree_multi(effective_bounds, flat_index)
    elif prefers_pair_mask_spatial_index_query(query_count, tree_count):
        result = _brute_force_pair_mask_multi(effective_bounds, flat_index)
    elif query_count == 1 and not _is_device_array(effective_bounds):
        result = _brute_force_scalar(effective_bounds[0], flat_index)
    else:
        result = _brute_force_multi(effective_bounds, flat_index)

    if result is None:
        return None, SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="GPU brute-force bbox overlap found zero candidates",
        )

    return result, SpatialQueryExecution(
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
        implementation="owned_gpu_spatial_query",
        reason=(
            f"brute-force O(N*M) bbox overlap query: "
            f"N={query_count}, M={tree_count}, N*M={n_product:,}"
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expand_bounds_for_distance(
    bounds,
    distances: np.ndarray | object,
):
    """Expand query bounds by per-row distances for dwithin."""
    if _is_device_array(bounds):
        if cp is None:  # pragma: no cover - exercised on CPU-only installs
            raise RuntimeError("CuPy is not installed; device bounds are unavailable")
        expanded = cp.array(bounds, dtype=cp.float64, copy=True)
        if np.isscalar(distances):
            dist = float(distances)
        else:
            dist = cp.asarray(distances, dtype=cp.float64)
        expanded[:, 0] -= dist
        expanded[:, 1] -= dist
        expanded[:, 2] += dist
        expanded[:, 3] += dist
        return expanded

    expanded = np.array(bounds, dtype=np.float64, copy=True, order="C")
    if np.isscalar(distances):
        d = float(distances)
        expanded[:, 0] -= d
        expanded[:, 1] -= d
        expanded[:, 2] += d
        expanded[:, 3] += d
    else:
        dist = np.asarray(distances, dtype=np.float64)
        expanded[:, 0] -= dist
        expanded[:, 1] -= dist
        expanded[:, 2] += dist
        expanded[:, 3] += dist
    return expanded


def _brute_force_scalar(
    query_bounds_row: np.ndarray,
    flat_index,
) -> _DeviceCandidates | None:
    """GPU brute-force for Q=1: one thread per tree row."""
    if np.isnan(query_bounds_row).any():
        return _empty_device_candidates()

    import cupy as cp

    runtime = get_cuda_runtime()
    tree_count = flat_index.size
    d_tree_bounds, temp_tree_bounds = _prepare_tree_bounds_device(flat_index, runtime)
    d_mask = runtime.allocate((tree_count,), cp.uint8)
    try:
        kernels = _spatial_query_kernels()
        kernel = kernels["bbox_overlap_tree_mask"]
        ptr = runtime.pointer
        params = (
            (
                ptr(d_tree_bounds),
                float(query_bounds_row[0]),
                float(query_bounds_row[1]),
                float(query_bounds_row[2]),
                float(query_bounds_row[3]),
                ptr(d_mask),
                tree_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        grid, block = runtime.launch_config(kernel, tree_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)

        compacted = compact_indices(d_mask)
        if compacted.values.size == 0:
            return _empty_device_candidates()
        d_right = cp.asarray(compacted.values, dtype=cp.int32)
        d_left = cp.zeros(d_right.size, dtype=cp.int32)
        return _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=int(d_right.size),
        )
    finally:
        runtime.free(temp_tree_bounds)
        runtime.free(d_mask)


def _brute_force_multi(
    query_bounds,
    flat_index,
) -> _DeviceCandidates | None:
    """GPU brute-force for Q>1: count + exclusive_sum + scatter."""
    import cupy as cp

    runtime = get_cuda_runtime()
    query_count = int(query_bounds.shape[0])
    tree_count = flat_index.size

    d_query_bounds, temp_query_bounds = _prepare_query_bounds_device(query_bounds, runtime)
    d_tree_bounds, temp_tree_bounds = _prepare_tree_bounds_device(flat_index, runtime)
    d_counts = runtime.allocate((query_count,), cp.int32)
    d_counts_i64 = None
    d_offsets = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        # Pass 0: count overlapping pairs per query row.
        count_kernel = kernels["bbox_overlap_multi_count"]
        count_params = (
            (
                ptr(d_query_bounds),
                ptr(d_tree_bounds),
                query_count,
                tree_count,
                ptr(d_counts),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        )
        count_grid, count_block = runtime.launch_config(count_kernel, query_count)
        runtime.launch(
            count_kernel,
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        # Exclusive scan for offsets (CCCL Tier 3a).
        d_counts_i64 = cp.asarray(d_counts).astype(cp.int64, copy=False)
        d_offsets = exclusive_sum(d_counts_i64)

        total_pairs = (
            count_scatter_total(
                runtime,
                d_counts_i64,
                d_offsets,
                reason="device spatial-index candidate-pair allocation fence",
            )
            if query_count > 0
            else 0
        )
        if total_pairs == 0:
            return _empty_device_candidates()

        require_device_candidate_pair_capacity(
            total_pairs,
            relation_name="device bbox candidate relation",
        )

        # Pass 1: scatter matching pairs.
        d_left = cp.empty(total_pairs, dtype=cp.int32)
        d_right = cp.empty(total_pairs, dtype=cp.int32)
        scatter_kernel = kernels["bbox_overlap_multi_scatter"]
        scatter_params = (
            (
                ptr(d_query_bounds),
                ptr(d_tree_bounds),
                query_count,
                tree_count,
                ptr(d_offsets),
                ptr(d_left),
                ptr(d_right),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(scatter_kernel, query_count)
        runtime.launch(
            scatter_kernel,
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )

        # Sync before freeing input buffers (scatter kernel reads them).
        runtime.synchronize()

        return _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=total_pairs,
        )
    finally:
        runtime.free(temp_query_bounds)
        runtime.free(temp_tree_bounds)
        runtime.free(d_counts)
        runtime.free(d_counts_i64)
        runtime.free(d_offsets)


def _brute_force_pair_mask_multi(
    query_bounds,
    flat_index,
) -> _DeviceCandidates | None:
    """GPU brute-force for bounded N*M using pair-mask compaction."""
    import cupy as cp

    runtime = get_cuda_runtime()
    query_count = int(query_bounds.shape[0])
    tree_count = flat_index.size
    pair_count = query_count * tree_count
    if pair_count <= 0:
        return _empty_device_candidates()

    d_query_bounds, temp_query_bounds = _prepare_query_bounds_device(
        query_bounds,
        runtime,
    )
    d_tree_bounds, temp_tree_bounds = _prepare_tree_bounds_device(flat_index, runtime)
    d_mask = runtime.allocate((pair_count,), cp.uint8)
    try:
        kernels = _spatial_query_kernels()
        kernel = kernels["bbox_overlap_multi_pair_mask"]
        ptr = runtime.pointer
        params = (
            (
                ptr(d_query_bounds),
                ptr(d_tree_bounds),
                query_count,
                tree_count,
                ptr(d_mask),
                pair_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I64,
            ),
        )
        grid, block = runtime.launch_config(kernel, pair_count)
        runtime.launch(kernel, grid=grid, block=block, params=params)
        compacted = compact_indices(d_mask)
        d_flat = cp.asarray(compacted.values, dtype=cp.int64)
        if int(d_flat.size) == 0:
            return _empty_device_candidates()
        d_left = (d_flat // np.int64(tree_count)).astype(cp.int32, copy=False)
        d_right = (d_flat - d_left.astype(cp.int64) * np.int64(tree_count)).astype(
            cp.int32,
            copy=False,
        )
        return _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=int(d_flat.size),
        )
    finally:
        runtime.free(temp_query_bounds)
        runtime.free(temp_tree_bounds)
        runtime.free(d_mask)


def _brute_force_single_tree_multi(
    query_bounds,
    flat_index,
) -> _DeviceCandidates | None:
    """GPU brute-force for M=1 using bounded mask compaction.

    Each query can contribute at most one candidate pair, so the output shape
    is naturally query-row shaped.  This avoids the generic total-pair
    allocation fence used by the N*M scatter path.
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    query_count = int(query_bounds.shape[0])
    d_query_bounds, temp_query_bounds = _prepare_query_bounds_device(
        query_bounds,
        runtime,
    )
    d_tree_bounds, temp_tree_bounds = _prepare_tree_bounds_device(flat_index, runtime)
    d_counts = runtime.allocate((query_count,), cp.int32)
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer
        count_kernel = kernels["bbox_overlap_multi_count"]
        count_params = (
            (
                ptr(d_query_bounds),
                ptr(d_tree_bounds),
                query_count,
                1,
                ptr(d_counts),
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
            ),
        )
        count_grid, count_block = runtime.launch_config(count_kernel, query_count)
        runtime.launch(
            count_kernel,
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        compacted = compact_indices(d_counts)
        if compacted.values.size == 0:
            return _empty_device_candidates()
        d_left = cp.asarray(compacted.values, dtype=cp.int32)
        d_right = cp.zeros(d_left.size, dtype=cp.int32)
        return _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=int(d_left.size),
        )
    finally:
        runtime.free(temp_query_bounds)
        runtime.free(temp_tree_bounds)
        runtime.free(d_counts)


@dataclass
class _MortonRangeQueryState:
    d_order: object
    d_sorted_keys: object
    d_sorted_tree_bounds: object
    d_query_bounds: object
    d_starts: object
    d_ends: object
    temp_tree_bounds: object | None
    temp_query_bounds: object | None
    temp_expanded_bounds: object | None
    d_range_low: object
    d_range_high: object

    def close(self) -> None:
        """Retire owned temporaries after current-stream work completes."""
        runtime = get_cuda_runtime()
        runtime.free(self.d_sorted_keys)
        runtime.free(self.d_sorted_tree_bounds)
        runtime.free(self.temp_tree_bounds)
        runtime.free(self.temp_query_bounds)
        runtime.free(self.temp_expanded_bounds)
        runtime.free(self.d_range_low)
        runtime.free(self.d_range_high)
        runtime.free(self.d_starts)
        runtime.free(self.d_ends)


def _prepare_morton_range_query(
    flat_index,
    original_bounds,
    effective_bounds,
) -> _MortonRangeQueryState | None:
    """Build reusable device Morton ranges without exporting index columns."""
    import cupy as cp

    runtime = get_cuda_runtime()
    query_count = int(original_bounds.shape[0])
    total_bounds = flat_index.total_bounds
    d_tree_bounds, temp_tree_bounds = _prepare_tree_bounds_device(flat_index, runtime)
    d_tree_bounds = cp.asarray(d_tree_bounds, dtype=cp.float64).reshape(-1, 4)
    d_width = d_tree_bounds[:, 2] - d_tree_bounds[:, 0]
    d_height = d_tree_bounds[:, 3] - d_tree_bounds[:, 1]
    d_extent_summary = cp.stack(
        (
            cp.count_nonzero(cp.isfinite(d_tree_bounds).all(axis=1)),
            cp.max(cp.where(cp.isfinite(d_width), d_width, 0.0)) * 0.5,
            cp.max(cp.where(cp.isfinite(d_height), d_height, 0.0)) * 0.5,
        )
    ).astype(cp.float64, copy=False)
    extent_summary = runtime.copy_device_to_host(
        d_extent_summary,
        reason="device spatial-index tree extent planning fence",
    )
    if int(extent_summary[0]) == 0:
        runtime.free(temp_tree_bounds)
        return None

    expanded_bounds = effective_bounds.copy()
    expanded_bounds[:, 0] -= float(extent_summary[1])
    expanded_bounds[:, 1] -= float(extent_summary[2])
    expanded_bounds[:, 2] += float(extent_summary[1])
    expanded_bounds[:, 3] += float(extent_summary[2])

    device_order = getattr(flat_index, "device_order", None)
    if device_order is None:
        device_order = cp.asarray(flat_index.order, dtype=cp.int32)
        object.__setattr__(flat_index, "device_order", device_order)
    d_order = cp.asarray(device_order, dtype=cp.int32)

    device_morton_keys = getattr(flat_index, "device_morton_keys", None)
    if device_morton_keys is None:
        device_morton_keys = cp.asarray(flat_index.morton_keys, dtype=cp.uint64)
        object.__setattr__(flat_index, "device_morton_keys", device_morton_keys)
    d_unsorted_keys = cp.asarray(device_morton_keys, dtype=cp.uint64)
    d_sorted_keys = cp.ascontiguousarray(d_unsorted_keys[d_order])
    d_sorted_tree_bounds = cp.ascontiguousarray(d_tree_bounds[d_order]).ravel()
    d_query_bounds, temp_query_bounds = _prepare_query_bounds_device(
        effective_bounds,
        runtime,
    )
    d_expanded_bounds, temp_expanded_bounds = _prepare_query_bounds_device(
        expanded_bounds,
        runtime,
    )
    d_range_low = runtime.allocate((query_count,), cp.uint64)
    d_range_high = runtime.allocate((query_count,), cp.uint64)

    kernels = _morton_range_kernels()
    ptr = runtime.pointer
    range_kernel = kernels["morton_range_from_bounds"]
    range_params = (
        (
            ptr(d_expanded_bounds),
            float(total_bounds[0]),
            float(total_bounds[1]),
            float(total_bounds[2]),
            float(total_bounds[3]),
            ptr(d_range_low),
            ptr(d_range_high),
            query_count,
        ),
        (
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
        ),
    )
    range_grid, range_block = runtime.launch_config(range_kernel, query_count)
    runtime.launch(
        range_kernel,
        grid=range_grid,
        block=range_block,
        params=range_params,
    )
    d_starts = lower_bound(d_sorted_keys, d_range_low, synchronize=False)
    d_ends = upper_bound(d_sorted_keys, d_range_high, synchronize=False)
    return _MortonRangeQueryState(
        d_order=d_order,
        d_sorted_keys=d_sorted_keys,
        d_sorted_tree_bounds=d_sorted_tree_bounds,
        d_query_bounds=d_query_bounds,
        d_starts=d_starts,
        d_ends=d_ends,
        temp_tree_bounds=temp_tree_bounds,
        temp_query_bounds=temp_query_bounds,
        temp_expanded_bounds=temp_expanded_bounds,
        d_range_low=d_range_low,
        d_range_high=d_range_high,
    )


def _morton_range_query(
    flat_index,
    original_bounds,
    effective_bounds,
) -> _DeviceCandidates | None:
    """Morton range query — O(N*log(M)+K).

    Uses pre-sorted Morton keys with CCCL binary search to narrow the
    scan window per query, then refines within the window using bbox
    overlap.

    Parameters
    ----------
    flat_index : FlatSpatialIndex
        Must have ``device_morton_keys``, ``device_order``, and valid
        ``total_bounds``.
    original_bounds : np.ndarray
        Original query bounds (used for bbox refinement).
    effective_bounds : np.ndarray
        Possibly distance-expanded bounds (used for Morton range lookup).
    """
    import cupy as cp

    query_count = int(original_bounds.shape[0])
    state = _prepare_morton_range_query(
        flat_index,
        original_bounds,
        effective_bounds,
    )
    if state is None:
        return _empty_device_candidates()
    runtime = get_cuda_runtime()
    d_counts = runtime.allocate((query_count,), cp.int32)
    d_counts_i64 = None
    d_offsets = None
    d_left = None
    d_right = None

    try:
        kernels = _morton_range_kernels()
        ptr = runtime.pointer

        # Step 3: Count bbox overlaps per query within Morton range.
        count_kernel = kernels["morton_range_count"]
        count_params = (
            (
                ptr(state.d_starts),
                ptr(state.d_ends),
                ptr(state.d_sorted_tree_bounds),
                ptr(state.d_query_bounds),
                ptr(d_counts),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(count_kernel, query_count)
        runtime.launch(
            count_kernel,
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        # Step 4: Exclusive scan for output offsets.
        d_counts_i64 = cp.asarray(d_counts).astype(cp.int64, copy=False)
        d_offsets = exclusive_sum(d_counts_i64)

        total_pairs = (
            count_scatter_total(
                runtime,
                d_counts_i64,
                d_offsets,
                reason="device spatial-index refined-pair allocation fence",
            )
            if query_count > 0
            else 0
        )
        if total_pairs == 0:
            return _empty_device_candidates()

        require_device_candidate_pair_capacity(
            total_pairs,
            relation_name="device Morton candidate relation",
        )

        # Step 5: Scatter matching pairs.
        d_left = cp.empty(total_pairs, dtype=cp.int32)
        d_right = cp.empty(total_pairs, dtype=cp.int32)
        scatter_kernel = kernels["morton_range_scatter"]
        scatter_params = (
            (
                ptr(state.d_starts),
                ptr(state.d_ends),
                ptr(state.d_order),
                ptr(state.d_sorted_tree_bounds),
                ptr(state.d_query_bounds),
                ptr(d_offsets),
                ptr(d_left),
                ptr(d_right),
                query_count,
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
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(scatter_kernel, query_count)
        runtime.launch(
            scatter_kernel,
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )

        # Sync before freeing input buffers.
        runtime.synchronize()

        result = _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=total_pairs,
        )
        # Prevent finally from freeing output arrays.
        d_left = None
        d_right = None
        return result
    finally:
        state.close()
        runtime.free(d_counts)
        runtime.free(d_counts_i64)
        runtime.free(d_offsets)
        runtime.free(d_left)
        runtime.free(d_right)


def _classify_homogeneous_reduction_tile(
    predicate,
    query_owned,
    tree_owned,
    d_left,
    d_right,
    *,
    query_family,
    tree_family,
    precision_plan: PrecisionPlan | None = None,
    logical_count=None,
    pair_capacity: int | None = None,
    source_offset=None,
    launch_capacity: int | None = None,
    d_exact_out=None,
    d_relation_scratch=None,
    d_de9im_scratch=None,
):
    """Return one fixed-capacity exact predicate mask for a family pair."""
    import cupy as cp

    if predicate is None:
        return cp.ones(d_left.size, dtype=cp.bool_)

    from vibespatial.geometry.buffers import GeometryFamily

    point_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    if query_family in point_families or tree_family in point_families:
        if precision_plan is None:
            raise TypeError(
                "indexed point-family reduction requires an explicit PrecisionPlan"
            )
        from vibespatial.predicates.point_relations import (
            classify_homogeneous_point_predicates_indexed_device,
        )

        return classify_homogeneous_point_predicates_indexed_device(
            predicate,
            query_owned,
            tree_owned,
            d_left,
            d_right,
            left_family=query_family,
            right_family=tree_family,
            precision_plan=precision_plan,
            logical_count=logical_count,
            source_offset=source_offset,
            launch_capacity=launch_capacity,
            predicate_out=d_exact_out,
            relation_out=d_relation_scratch,
        )

    from vibespatial.predicates.binary import _evaluate_de9im_device
    from vibespatial.predicates.polygon import (
        compute_polygon_de9im_gpu,
        compute_polygonal_intersects_gpu,
    )

    polygonal = {GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON}
    if predicate == "intersects" and query_family in polygonal and tree_family in polygonal:
        return compute_polygonal_intersects_gpu(
            query_owned,
            tree_owned,
            query_family=query_family,
            tree_family=tree_family,
            d_left=d_left,
            d_right=d_right,
            d_pair_count=logical_count,
            pair_capacity=pair_capacity,
            d_pair_offset=source_offset,
            launch_capacity=launch_capacity,
            d_out=d_exact_out,
            return_device=True,
        )
    d_masks = compute_polygon_de9im_gpu(
        query_owned,
        tree_owned,
        query_family=query_family,
        tree_family=tree_family,
        d_left=d_left,
        d_right=d_right,
        d_pair_count=logical_count,
        pair_capacity=pair_capacity,
        d_pair_offset=source_offset,
        launch_capacity=launch_capacity,
        d_mask=d_de9im_scratch,
        return_device=True,
    )
    if d_masks is None:
        return None
    if source_offset is not None:
        from vibespatial.predicates.polygon import evaluate_de9im_grouped_device

        return evaluate_de9im_grouped_device(
            d_masks,
            predicate,
            source_offset=source_offset,
            logical_count=logical_count,
            launch_capacity=launch_capacity,
            out=d_exact_out,
        )
    return _evaluate_de9im_device(d_masks, predicate)


def _family_group_launch_capacities(
    pair_capacity: int,
    family_pair_count: int,
) -> tuple[int, ...]:
    """Bound aggregate grouped classifier launch lanes by one tile plus metadata."""
    pair_capacity = int(pair_capacity)
    family_pair_count = int(family_pair_count)
    if pair_capacity < 0 or family_pair_count < 0:
        raise ValueError("grouped launch dimensions must be nonnegative")
    if family_pair_count == 0:
        return ()
    quotient, remainder = divmod(pair_capacity, family_pair_count)
    return tuple(
        max(1, quotient + (group_index < remainder))
        for group_index in range(family_pair_count)
    )


def _morton_reduction_span_schedule(d_starts, d_ends):
    """Group query rows by power-of-two Morton span on device.

    A fixed 33-value bucket-count packet gives Python a structural launch
    schedule without exporting query rows or data-dependent candidate counts.
    Each nonzero span is scheduled at less than twice its interval length, plus
    bounded final-batch padding within its bucket.
    """
    import cupy as cp

    d_starts = cp.asarray(d_starts, dtype=cp.uint64)
    d_ends = cp.asarray(d_ends, dtype=cp.uint64)
    if int(d_starts.size) != int(d_ends.size):
        raise ValueError("Morton reduction starts and ends must align")
    query_count = int(d_starts.size)
    if query_count == 0:
        return cp.empty(0, dtype=cp.int32), np.zeros(
            len(_MORTON_SPAN_BUCKET_UPPER_BOUNDS),
            dtype=np.int64,
        )

    d_spans = d_ends - d_starts
    d_upper_bounds = cp.asarray(
        _MORTON_SPAN_BUCKET_UPPER_BOUNDS,
        dtype=cp.uint64,
    )
    d_bucket_ids = cp.searchsorted(
        d_upper_bounds,
        d_spans,
        side="left",
    ).astype(cp.int32, copy=False)
    sorted_buckets = sort_pairs(
        d_bucket_ids,
        cp.arange(query_count, dtype=cp.int32),
        strategy=PairSortStrategy.RADIX,
        synchronize=False,
    )
    d_query_order = cp.asarray(sorted_buckets.values, dtype=cp.int32)
    d_bucket_counts = cp.bincount(
        d_bucket_ids,
        minlength=len(_MORTON_SPAN_BUCKET_UPPER_BOUNDS),
    ).astype(cp.int64, copy=False)
    bucket_counts = get_cuda_runtime().copy_device_to_host(
        d_bucket_counts,
        reason="spatial reduction Morton-span bucket planning packet",
    )
    return d_query_order, np.asarray(bucket_counts, dtype=np.int64)


def _spatial_index_device_relation_reduction(
    flat_index,
    query_owned,
    tree_owned,
    query_bounds,
    *,
    predicate: str | None,
    reduction: str,
) -> tuple[object | None, SpatialQueryExecution]:
    """Reduce Morton-range candidate slices without materializing a relation.

    Physical shape: one thread per query row scans only its bounded Morton
    interval slice, then count/scan/scatter produces a capacity-backed candidate
    prefix for exact refinement and row reduction. Query partitions and Morton
    position rounds are derived only from carrier capacity; no device result
    controls Python launch flow or intermediate pair allocation.
    """
    import cupy as cp

    from vibespatial.spatial.query_utils import (
        _owned_gpu_predicate_family_admission,
    )

    query_count = int(query_owned.row_count)
    tree_count = int(tree_owned.row_count)
    if reduction not in {"exists", "right_exists", "count"}:
        raise ValueError(
            "spatial reduction must be 'exists', 'right_exists', or 'count'"
        )
    implementation = {
        "exists": "owned_gpu_spatial_semijoin",
        "right_exists": "owned_gpu_spatial_right_semijoin",
        "count": "owned_gpu_spatial_match_count",
    }[reduction]
    reduction_name = {
        "exists": "left existential semijoin",
        "right_exists": "right existential semijoin",
        "count": "left match count",
    }[reduction]
    execution = SpatialQueryExecution(
        requested=ExecutionMode.GPU,
        selected=ExecutionMode.GPU,
        implementation=implementation,
        reason=f"bounded Morton candidate tiles reduced directly to {reduction_name}",
    )
    from vibespatial.api._native_rowset import NativeDeviceSelection

    if query_count == 0:
        return (
            NativeDeviceSelection.identity(0)
            if reduction != "count"
            else cp.empty(0, dtype=cp.int64)
        ), execution
    if tree_count == 0:
        values = (
            NativeDeviceSelection.from_mask(
                cp.zeros(tree_count if reduction == "right_exists" else query_count, dtype=cp.bool_)
            )
            if reduction != "count"
            else cp.zeros(query_count, dtype=cp.int64)
        )
        return values, execution

    admission = _owned_gpu_predicate_family_admission(query_owned, tree_owned)
    if predicate is not None and (
        admission is None
        or not (
            admission[0]
            or (admission[2] and predicate in _POLYGON_DE9IM_PREDICATES)
        )
    ):
        return None, SpatialQueryExecution(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason=(
                f"native {reduction_name} declined unsupported exact "
                f"predicate families for {predicate!r}"
            ),
        )

    query_families = tuple(
        family for family in query_owned.families if query_owned.family_has_rows(family)
    )
    tree_families = tuple(
        family for family in tree_owned.families if tree_owned.family_has_rows(family)
    )
    homogeneous_family_pair = len(query_families) == 1 and len(tree_families) == 1
    family_pairs = tuple(
        (query_family, tree_family)
        for query_family in query_families
        for tree_family in tree_families
    )
    family_partition_type = None
    if not homogeneous_family_pair:
        from vibespatial.api._native_relation import NativeRelationFamilyPartition

        family_partition_type = NativeRelationFamilyPartition

    point_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    indexed_point_precision_plan = None
    if predicate is not None and (
        any(family in point_families for family in query_families)
        or any(family in point_families for family in tree_families)
    ):
        from vibespatial.predicates.point_relations import (
            _plan_indexed_point_precision,
        )

        indexed_point_precision_plan = _plan_indexed_point_precision(
            PrecisionMode.AUTO,
            runtime_selection=RuntimeSelection(
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.GPU,
                reason=(
                    "indexed point-family candidate refinement preserves "
                    "authoritative fp64 results"
                ),
            ),
        )

    state = _prepare_morton_range_query(flat_index, query_bounds, query_bounds)
    if state is None:
        values = (
            NativeDeviceSelection.from_mask(
                cp.zeros(tree_count if reduction == "right_exists" else query_count, dtype=cp.bool_)
            )
            if reduction != "count"
            else cp.zeros(query_count, dtype=cp.int64)
        )
        return values, execution

    output_count = tree_count if reduction == "right_exists" else query_count
    d_reduced = cp.zeros(output_count, dtype=cp.uint64)
    max_tile_lanes = _spatial_reduction_tile_lane_capacity(
        query_owned,
        tree_owned,
        predicate=predicate,
        family_admission=admission,
    )
    tile_width = min(
        tree_count,
        _SEMIJOIN_MAX_CANDIDATES_PER_ROW,
        max_tile_lanes,
    )
    query_tile_rows = max(1, min(query_count, max_tile_lanes // tile_width))
    tile_count = 0
    family_partition_pass_count = 0

    try:
        runtime = get_cuda_runtime()
        query_state = query_owned._ensure_device_state(preserve_indexed_view=True)
        tree_state = tree_owned._ensure_device_state(preserve_indexed_view=True)
        d_query_tags = cp.asarray(query_state.tags, dtype=cp.int8)
        d_tree_tags = cp.asarray(tree_state.tags, dtype=cp.int8)
        d_query_order, bucket_counts = _morton_reduction_span_schedule(
            state.d_starts,
            state.d_ends,
        )
        kernels = _morton_range_kernels()
        count_kernel = kernels["morton_range_tile_count"]
        scatter_kernel = kernels["morton_range_tile_scatter"]
        ptr = runtime.pointer
        bucket_start = 0
        for bucket_index, bucket_count_raw in enumerate(bucket_counts):
            bucket_count = int(bucket_count_raw)
            bucket_stop = bucket_start + bucket_count
            bucket_span = _MORTON_SPAN_BUCKET_UPPER_BOUNDS[bucket_index]
            if bucket_span == 0 or bucket_count == 0:
                bucket_start = bucket_stop
                continue
            for query_order_start in range(
                bucket_start,
                bucket_stop,
                query_tile_rows,
            ):
                query_order_stop = min(
                    query_order_start + query_tile_rows,
                    bucket_stop,
                )
                query_batch_count = query_order_stop - query_order_start
                for position_start in range(0, bucket_span, tile_width):
                    current_tile_width = min(
                        tile_width,
                        bucket_span - position_start,
                    )
                    pair_capacity = query_batch_count * current_tile_width
                    d_counts = cp.zeros(query_batch_count, dtype=cp.int32)
                    count_grid, count_block = runtime.launch_config(
                        count_kernel,
                        query_batch_count,
                    )
                    runtime.launch(
                        count_kernel,
                        grid=count_grid,
                        block=count_block,
                        params=(
                            (
                                ptr(state.d_starts),
                                ptr(state.d_ends),
                                ptr(d_query_order),
                                ptr(state.d_sorted_tree_bounds),
                                ptr(state.d_query_bounds),
                                ptr(d_counts),
                                query_order_start,
                                query_batch_count,
                                position_start,
                                current_tile_width,
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
                                KERNEL_PARAM_I64,
                                KERNEL_PARAM_I32,
                            ),
                        ),
                    )
                    d_counts_i64 = d_counts.astype(cp.int64, copy=False)
                    d_offsets = exclusive_sum(d_counts_i64, synchronize=False)
                    d_candidate_count_i64 = d_offsets[-1:] + d_counts_i64[-1:]
                    d_candidate_count_i32 = d_candidate_count_i64.astype(
                        cp.int32,
                        copy=False,
                    )
                    d_pair_left = cp.zeros(pair_capacity, dtype=cp.int32)
                    d_pair_right = cp.zeros(pair_capacity, dtype=cp.int32)
                    scatter_grid, scatter_block = runtime.launch_config(
                        scatter_kernel,
                        query_batch_count,
                    )
                    runtime.launch(
                        scatter_kernel,
                        grid=scatter_grid,
                        block=scatter_block,
                        params=(
                            (
                                ptr(state.d_starts),
                                ptr(state.d_ends),
                                ptr(d_query_order),
                                ptr(state.d_order),
                                ptr(state.d_sorted_tree_bounds),
                                ptr(state.d_query_bounds),
                                ptr(d_offsets),
                                ptr(d_pair_left),
                                ptr(d_pair_right),
                                query_order_start,
                                query_batch_count,
                                position_start,
                                current_tile_width,
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
                                KERNEL_PARAM_I32,
                                KERNEL_PARAM_I64,
                                KERNEL_PARAM_I32,
                            ),
                        ),
                    )
                    candidate_selection = NativeDeviceSelection(
                        positions=cp.arange(pair_capacity, dtype=cp.int64),
                        logical_count=d_candidate_count_i64,
                        source_row_count=pair_capacity,
                        full_selection_implies_identity=True,
                    )
                    d_active = candidate_selection.active_capacity_mask()
                    if predicate is None:
                        d_reduce_left = d_pair_left
                        d_reduce_right = d_pair_right
                        d_keep = d_active
                    elif homogeneous_family_pair:
                        d_reduce_left = d_pair_left
                        d_reduce_right = d_pair_right
                        d_exact = _classify_homogeneous_reduction_tile(
                            predicate,
                            query_owned,
                            tree_owned,
                            d_reduce_left,
                            d_reduce_right,
                            query_family=query_families[0],
                            tree_family=tree_families[0],
                            precision_plan=indexed_point_precision_plan,
                            logical_count=d_candidate_count_i32,
                            pair_capacity=pair_capacity,
                        )
                        if d_exact is None:
                            return None, SpatialQueryExecution(
                                requested=ExecutionMode.GPU,
                                selected=ExecutionMode.CPU,
                                implementation="owned_cpu_spatial_query",
                                reason=f"native {reduction_name} exact refinement declined",
                            )
                        d_keep = d_active & cp.asarray(d_exact, dtype=cp.bool_)
                    else:
                        family_partition = family_partition_type.from_pair_capacity(
                            d_pair_left,
                            d_pair_right,
                            d_active,
                            d_query_tags,
                            d_tree_tags,
                            family_count=len(FAMILY_TAGS),
                        )
                        family_partition_pass_count += 1
                        d_exact = cp.zeros(pair_capacity, dtype=cp.bool_)
                        d_relation_scratch = cp.empty(pair_capacity, dtype=cp.uint8)
                        d_de9im_scratch = cp.empty(pair_capacity, dtype=cp.uint16)
                        family_launch_capacities = _family_group_launch_capacities(
                            pair_capacity,
                            len(family_pairs),
                        )
                        for (
                            (query_family, tree_family),
                            family_launch_capacity,
                        ) in zip(
                            family_pairs,
                            family_launch_capacities,
                            strict=True,
                        ):
                            partition = family_partition.family_pair(
                                left_family=query_family,
                                right_family=tree_family,
                                left_family_tag=FAMILY_TAGS[query_family],
                                right_family_tag=FAMILY_TAGS[tree_family],
                                launch_capacity=family_launch_capacity,
                            )
                            d_partition_count_i32 = cp.asarray(
                                partition.logical_count,
                                dtype=cp.int32,
                            )
                            d_exact = _classify_homogeneous_reduction_tile(
                                predicate,
                                query_owned,
                                tree_owned,
                                partition.left_indices,
                                partition.right_indices,
                                query_family=query_family,
                                tree_family=tree_family,
                                precision_plan=indexed_point_precision_plan,
                                logical_count=d_partition_count_i32,
                                pair_capacity=family_launch_capacity,
                                source_offset=partition.source_offset,
                                launch_capacity=family_launch_capacity,
                                d_exact_out=d_exact,
                                d_relation_scratch=d_relation_scratch,
                                d_de9im_scratch=d_de9im_scratch,
                            )
                            if d_exact is None:
                                return None, SpatialQueryExecution(
                                    requested=ExecutionMode.GPU,
                                    selected=ExecutionMode.CPU,
                                    implementation="owned_cpu_spatial_query",
                                    reason=(
                                        f"native {reduction_name} exact family "
                                        "partition refinement declined"
                                    ),
                                )
                        d_reduce_left = family_partition.left_indices
                        d_reduce_right = family_partition.right_indices
                        d_keep = d_exact
                    if reduction == "right_exists":
                        cp.maximum.at(
                            d_reduced,
                            d_reduce_right,
                            d_keep.astype(cp.uint64),
                        )
                    elif reduction == "exists":
                        cp.maximum.at(
                            d_reduced,
                            d_reduce_left,
                            d_keep.astype(cp.uint64),
                        )
                    else:
                        cp.add.at(
                            d_reduced,
                            d_reduce_left,
                            d_keep.astype(cp.uint64),
                        )
                    tile_count += 1
            bucket_start = bucket_stop

        values = (
            NativeDeviceSelection.from_mask(d_reduced != 0)
            if reduction != "count"
            else d_reduced.astype(cp.int64)
        )
        return values, SpatialQueryExecution(
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.GPU,
            implementation=implementation,
            reason=(
                "range-sliced Morton candidate prefixes reduced directly to "
                f"{reduction_name} in {tile_count} structural tiles"
                + (
                    " with one all-family grouped partition pass per mixed tile "
                    f"({family_partition_pass_count} passes)"
                    if family_partition_pass_count
                    else ""
                )
            ),
        )
    finally:
        state.close()


def spatial_index_device_left_semijoin_rows(
    flat_index,
    query_owned,
    tree_owned,
    query_bounds,
    *,
    predicate: str | None,
) -> tuple[object | None, SpatialQueryExecution]:
    """Reduce a dense Morton candidate stream directly into matched rows."""
    return _spatial_index_device_relation_reduction(
        flat_index,
        query_owned,
        tree_owned,
        query_bounds,
        predicate=predicate,
        reduction="exists",
    )


def spatial_index_device_left_match_counts(
    flat_index,
    query_owned,
    tree_owned,
    query_bounds,
    *,
    predicate: str | None,
) -> tuple[object | None, SpatialQueryExecution]:
    """Reduce a dense Morton candidate stream directly into int64 counts."""
    return _spatial_index_device_relation_reduction(
        flat_index,
        query_owned,
        tree_owned,
        query_bounds,
        predicate=predicate,
        reduction="count",
    )


def spatial_index_device_right_semijoin_rows(
    flat_index,
    query_owned,
    tree_owned,
    query_bounds,
    *,
    predicate: str | None,
) -> tuple[object | None, SpatialQueryExecution]:
    """Reduce a dense Morton candidate stream into matched indexed rows."""
    return _spatial_index_device_relation_reduction(
        flat_index,
        query_owned,
        tree_owned,
        query_bounds,
        predicate=predicate,
        reduction="right_exists",
    )
