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

import numpy as np

from vibespatial.cuda._runtime import (
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    upper_bound,
)
from vibespatial.kernels.core.spatial_query_kernels import (
    _morton_range_kernels,
    _spatial_query_kernels,
)
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_kernel_dispatch
from vibespatial.runtime.crossover import DispatchDecision
from vibespatial.runtime.precision import (
    KernelClass,
    PrecisionMode,
    select_precision_plan,
)

from .query_types import SpatialQueryExecution, _DeviceCandidates

# Eagerly request CCCL spec warmup at module import (ADR-0034 Level 1).
request_warmup([
    "exclusive_scan_i32",
    "exclusive_scan_i64",
    "select_i32",
    "select_i64",
    "lower_bound_u64",
    "upper_bound_u64",
])


# ---------------------------------------------------------------------------
# Strategy selection thresholds
# ---------------------------------------------------------------------------
# Morton range has higher fixed overhead (6 kernel launches vs 2 for brute-
# force), but scales as O(N*log(M)+K) instead of O(N*M).
# Crossover: when N*M exceeds this, prefer Morton range.  Benchmarked at
# ~1M (roughly 1K x 1K) with warm kernels (ADR-0034) — the overhead delta
# is ~0.5ms, dominated by the binary search + range expansion cost.
_MORTON_RANGE_CROSSOVER = 1_000_000


def spatial_index_device_query(
    flat_index,
    query_bounds: np.ndarray,
    *,
    distance: np.ndarray | None = None,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
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
        ``candidates`` is None when GPU dispatch is skipped or no pairs
        are found.  ``execution`` carries the dispatch decision metadata.
    """
    if not has_gpu_runtime():
        return None, SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="GPU runtime unavailable; skipping device spatial index query",
        )

    query_count = query_bounds.shape[0]
    tree_count = flat_index.size
    if query_count == 0 or tree_count == 0:
        return None, SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="empty input; no candidate pairs to generate",
        )

    # ADR-0002: wire precision plan for observability.
    # COARSE class on bounds — always fp64 (see module docstring).
    runtime_selection = RuntimeSelection(
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU,
        reason="device spatial index query",
    )
    _precision_plan = select_precision_plan(
        runtime_selection=runtime_selection,
        kernel_class=KernelClass.COARSE,
        requested=precision,
    )
    # Bounds kernels are memory-bound: fp64 is correct and necessary.
    assert _precision_plan.compute_precision in (
        PrecisionMode.FP64,
        PrecisionMode.FP32,
    ), "PrecisionPlan must resolve to a concrete mode"

    # Dispatch check: the crossover threshold for bbox_overlap_candidates
    # is 0, so GPU dispatch is always selected when GPU is available.
    plan = plan_kernel_dispatch(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * tree_count,
        gpu_available=True,
    )
    if plan.dispatch_decision is not DispatchDecision.GPU:
        return None, SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="dispatch decision selected CPU for this workload",
        )

    # Expand bounds for dwithin if distance is provided.
    effective_bounds = query_bounds
    if distance is not None:
        effective_bounds = _expand_bounds_for_distance(query_bounds, distance)

    # Strategy selection: Morton range vs brute-force.
    n_product = query_count * tree_count
    total_bounds = getattr(flat_index, "total_bounds", None)
    has_morton = (
        total_bounds is not None
        and not np.isnan(total_bounds[0])
        and getattr(flat_index, "device_morton_keys", None) is not None
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

    # Fall through to brute-force O(N*M).
    if query_count == 1:
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
    bounds: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Expand query bounds by per-row distances for dwithin."""
    expanded = bounds.copy()
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
        return None

    import cupy as cp

    runtime = get_cuda_runtime()
    tree_bounds = flat_index.bounds
    tree_count = flat_index.size
    d_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    d_mask = runtime.allocate((tree_count,), np.uint8)
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
            return None
        d_right = cp.asarray(compacted.values, dtype=cp.int32)
        d_left = cp.zeros(d_right.size, dtype=cp.int32)
        return _DeviceCandidates(
            d_left=d_left,
            d_right=d_right,
            total_pairs=int(d_right.size),
        )
    finally:
        runtime.free(d_tree_bounds)
        runtime.free(d_mask)


def _brute_force_multi(
    query_bounds: np.ndarray,
    flat_index,
) -> _DeviceCandidates | None:
    """GPU brute-force for Q>1: count + exclusive_sum + scatter."""
    import cupy as cp

    runtime = get_cuda_runtime()
    tree_bounds = flat_index.bounds
    query_count = query_bounds.shape[0]
    tree_count = flat_index.size

    d_query_bounds = runtime.from_host(
        np.ascontiguousarray(query_bounds, dtype=np.float64).ravel()
    )
    d_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    d_counts = runtime.allocate((query_count,), np.int32)
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
        d_offsets = exclusive_sum(d_counts)

        total_pairs = (
            count_scatter_total(runtime, d_counts, d_offsets)
            if query_count > 0
            else 0
        )
        if total_pairs == 0:
            return None

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
        scatter_grid, scatter_block = runtime.launch_config(
            scatter_kernel, query_count
        )
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
        runtime.free(d_query_bounds)
        runtime.free(d_tree_bounds)
        runtime.free(d_counts)
        runtime.free(d_offsets)


def _morton_range_query(
    flat_index,
    original_bounds: np.ndarray,
    effective_bounds: np.ndarray,
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

    runtime = get_cuda_runtime()
    query_count = original_bounds.shape[0]
    total_bounds = flat_index.total_bounds

    # Morton codes encode bbox centers, so a tree geometry whose center is
    # outside the query's Morton range can still overlap if the geometry is
    # large.  Expand query bounds by the maximum tree geometry half-extent
    # to ensure the Morton range is conservative.
    tree_bounds_arr = flat_index.bounds
    valid_tree = tree_bounds_arr[~np.isnan(tree_bounds_arr).any(axis=1)]
    if valid_tree.size == 0:
        return None
    max_half_w = float((valid_tree[:, 2] - valid_tree[:, 0]).max()) / 2.0
    max_half_h = float((valid_tree[:, 3] - valid_tree[:, 1]).max()) / 2.0

    expanded_bounds = effective_bounds.copy()
    expanded_bounds[:, 0] -= max_half_w
    expanded_bounds[:, 1] -= max_half_h
    expanded_bounds[:, 2] += max_half_w
    expanded_bounds[:, 3] += max_half_h

    # Prepare host data in Morton-sorted order.
    sorted_keys_host = flat_index.morton_keys[flat_index.order].astype(np.uint64)
    sorted_tree_bounds_host = np.ascontiguousarray(
        flat_index.bounds[flat_index.order], dtype=np.float64,
    )
    query_bounds_host = np.ascontiguousarray(original_bounds, dtype=np.float64)
    expanded_bounds_host = np.ascontiguousarray(expanded_bounds, dtype=np.float64)
    order_host = np.ascontiguousarray(flat_index.order, dtype=np.int32)

    # Upload to device.
    d_sorted_keys = runtime.from_host(sorted_keys_host)
    d_sorted_tree_bounds = runtime.from_host(sorted_tree_bounds_host.ravel())
    d_query_bounds = runtime.from_host(query_bounds_host.ravel())
    d_expanded_bounds = runtime.from_host(expanded_bounds_host.ravel())
    d_order = runtime.from_host(order_host)
    d_range_low = runtime.allocate((query_count,), np.uint64)
    d_range_high = runtime.allocate((query_count,), np.uint64)
    d_counts = runtime.allocate((query_count,), np.int32)
    d_starts = None
    d_ends = None
    d_offsets = None
    d_left = None
    d_right = None

    try:
        kernels = _morton_range_kernels()
        ptr = runtime.pointer

        # Step 1: Compute Morton ranges from expanded bounds.
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
        range_grid, range_block = runtime.launch_config(
            range_kernel, query_count
        )
        runtime.launch(
            range_kernel,
            grid=range_grid,
            block=range_block,
            params=range_params,
        )

        # Step 2: Binary search on sorted Morton keys (CCCL Tier 3a).
        d_starts = lower_bound(d_sorted_keys, d_range_low, synchronize=False)
        d_ends = upper_bound(d_sorted_keys, d_range_high, synchronize=False)

        # Step 3: Count bbox overlaps per query within Morton range.
        count_kernel = kernels["morton_range_count"]
        count_params = (
            (
                ptr(d_starts),
                ptr(d_ends),
                ptr(d_sorted_tree_bounds),
                ptr(d_query_bounds),
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
        count_grid, count_block = runtime.launch_config(
            count_kernel, query_count
        )
        runtime.launch(
            count_kernel,
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        # Step 4: Exclusive scan for output offsets.
        d_offsets = exclusive_sum(d_counts)

        total_pairs = (
            count_scatter_total(runtime, d_counts, d_offsets)
            if query_count > 0
            else 0
        )
        if total_pairs == 0:
            return None

        # Step 5: Scatter matching pairs.
        d_left = cp.empty(total_pairs, dtype=np.int32)
        d_right = cp.empty(total_pairs, dtype=np.int32)
        scatter_kernel = kernels["morton_range_scatter"]
        scatter_params = (
            (
                ptr(d_starts),
                ptr(d_ends),
                ptr(d_order),
                ptr(d_sorted_tree_bounds),
                ptr(d_query_bounds),
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
        scatter_grid, scatter_block = runtime.launch_config(
            scatter_kernel, query_count
        )
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
        runtime.free(d_sorted_keys)
        runtime.free(d_sorted_tree_bounds)
        runtime.free(d_query_bounds)
        runtime.free(d_expanded_bounds)
        runtime.free(d_order)
        runtime.free(d_range_low)
        runtime.free(d_range_high)
        runtime.free(d_counts)
        runtime.free(d_starts)
        runtime.free(d_ends)
        runtime.free(d_offsets)
        runtime.free(d_left)
        runtime.free(d_right)
