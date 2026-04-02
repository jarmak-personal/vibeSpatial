from __future__ import annotations

import numpy as np

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    upper_bound,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import COARSE_BOUNDS_TILE_SIZE

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "lower_bound_i32", "lower_bound_u64",
    "upper_bound_i32", "upper_bound_u64",
])
from vibespatial.cuda._runtime import (  # noqa: E402
    KERNEL_PARAM_F64,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_PTR,
    count_scatter_total,
    get_cuda_runtime,
)
from vibespatial.kernels.core.spatial_query_kernels import (  # noqa: E402
    _morton_range_kernels,
    _spatial_query_kernels,
)
from vibespatial.runtime import has_gpu_runtime  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402

from .query_types import _DeviceCandidates  # noqa: E402
from .query_utils import _expand_bounds  # noqa: E402


def _generate_distance_pairs(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    distances: np.ndarray,
    *,
    tile_size: int = COARSE_BOUNDS_TILE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    valid_left = np.flatnonzero(~np.isnan(query_bounds).any(axis=1))
    valid_right = np.flatnonzero(~np.isnan(tree_bounds).any(axis=1))
    expanded = _expand_bounds(query_bounds, distances)

    left_out: list[np.ndarray] = []
    right_out: list[np.ndarray] = []
    for left_start in range(0, valid_left.size, tile_size):
        left_chunk = valid_left[left_start : left_start + tile_size]
        left_chunk_bounds = expanded[left_chunk]
        for right_start in range(0, valid_right.size, tile_size):
            right_chunk = valid_right[right_start : right_start + tile_size]
            right_chunk_bounds = tree_bounds[right_chunk]
            intersects = (
                (left_chunk_bounds[:, None, 0] <= right_chunk_bounds[None, :, 2])
                & (left_chunk_bounds[:, None, 2] >= right_chunk_bounds[None, :, 0])
                & (left_chunk_bounds[:, None, 1] <= right_chunk_bounds[None, :, 3])
                & (left_chunk_bounds[:, None, 3] >= right_chunk_bounds[None, :, 1])
            )
            left_local, right_local = np.nonzero(intersects)
            if left_local.size == 0:
                continue
            left_out.append(left_chunk[left_local].astype(np.int32, copy=False))
            right_out.append(right_chunk[right_local].astype(np.int32, copy=False))

    if not left_out:
        empty = np.asarray([], dtype=np.int32)
        return empty, empty
    return np.concatenate(left_out), np.concatenate(right_out)


# ---------------------------------------------------------------------------
# GPU candidate generation (Tier 1 NVRTC + Tier 3a CCCL compaction/scan)
# ---------------------------------------------------------------------------
# ADR-0033: bbox overlap is geometry-specific compute (Tier 1), compaction
# and exclusive scan use CCCL primitives (Tier 3a) per benchmarked defaults.
# This replaces the CPU tiled O(N*M) generate_bounds_pairs path.

def _generate_candidates_gpu_scalar(
    query_bounds_row: np.ndarray,
    tree_bounds: np.ndarray,
    tree_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU candidate gen for Q=1: launch M threads, one per tree row."""
    if np.isnan(query_bounds_row).any():
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    runtime = get_cuda_runtime()
    device_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    device_mask = runtime.allocate((tree_count,), np.uint8)
    try:
        kernels = _spatial_query_kernels()
        kernel = kernels["bbox_overlap_tree_mask"]
        ptr = runtime.pointer
        params = (
            (
                ptr(device_tree_bounds),
                float(query_bounds_row[0]),
                float(query_bounds_row[1]),
                float(query_bounds_row[2]),
                float(query_bounds_row[3]),
                ptr(device_mask),
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

        compacted = compact_indices(device_mask)
        right_host = runtime.copy_device_to_host(compacted.values).astype(np.int32, copy=False)
        left_host = np.zeros(right_host.size, dtype=np.int32)
        return left_host, right_host
    finally:
        runtime.free(device_tree_bounds)
        runtime.free(device_mask)


def _generate_candidates_gpu_multi(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    query_count: int,
    tree_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU candidate gen for Q>1: count + exclusive_scan + scatter."""
    runtime = get_cuda_runtime()
    device_query_bounds = runtime.from_host(
        np.ascontiguousarray(query_bounds, dtype=np.float64).ravel()
    )
    device_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    device_counts = runtime.allocate((query_count,), np.int32)
    device_offsets = None
    device_left = None
    device_right = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        # Count pass
        count_kernel = kernels["bbox_overlap_multi_count"]
        count_params = (
            (
                ptr(device_query_bounds),
                ptr(device_tree_bounds),
                query_count,
                tree_count,
                ptr(device_counts),
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
            count_kernel, grid=count_grid, block=count_block, params=count_params,
        )

        # Exclusive scan for output offsets (CCCL Tier 3a)
        device_offsets = exclusive_sum(device_counts)

        total_pairs = count_scatter_total(runtime, device_counts, device_offsets) if query_count > 0 else 0

        if total_pairs == 0:
            empty = np.empty(0, dtype=np.int32)
            return empty, empty

        # Scatter pass
        device_left = runtime.allocate((total_pairs,), np.int32)
        device_right = runtime.allocate((total_pairs,), np.int32)
        scatter_kernel = kernels["bbox_overlap_multi_scatter"]
        scatter_params = (
            (
                ptr(device_query_bounds),
                ptr(device_tree_bounds),
                query_count,
                tree_count,
                ptr(device_offsets),
                ptr(device_left),
                ptr(device_right),
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
            scatter_kernel, grid=scatter_grid, block=scatter_block, params=scatter_params,
        )
        runtime.synchronize()

        left_host = runtime.copy_device_to_host(device_left).astype(np.int32, copy=False)
        right_host = runtime.copy_device_to_host(device_right).astype(np.int32, copy=False)
        return left_host, right_host
    finally:
        runtime.free(device_query_bounds)
        runtime.free(device_tree_bounds)
        runtime.free(device_counts)
        runtime.free(device_offsets)
        runtime.free(device_left)
        runtime.free(device_right)


def _generate_candidates_gpu_multi_device(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    query_count: int,
    tree_count: int,
) -> _DeviceCandidates | None:
    """GPU candidate gen for Q>1, returning device-resident CuPy arrays.

    Returns _DeviceCandidates with device-resident left/right index arrays,
    or None if there are no candidates. The caller owns the device memory.
    """
    import cupy as cp

    runtime = get_cuda_runtime()
    device_query_bounds = runtime.from_host(
        np.ascontiguousarray(query_bounds, dtype=np.float64).ravel()
    )
    device_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    device_counts = runtime.allocate((query_count,), np.int32)
    device_offsets = None
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        # Count pass
        count_kernel = kernels["bbox_overlap_multi_count"]
        count_params = (
            (
                ptr(device_query_bounds),
                ptr(device_tree_bounds),
                query_count,
                tree_count,
                ptr(device_counts),
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
            count_kernel, grid=count_grid, block=count_block, params=count_params,
        )

        # Exclusive scan for output offsets (CCCL Tier 3a)
        device_offsets = exclusive_sum(device_counts)

        total_pairs = count_scatter_total(runtime, device_counts, device_offsets) if query_count > 0 else 0

        if total_pairs == 0:
            return None

        # Scatter pass
        device_left = cp.empty(total_pairs, dtype=cp.int32)
        device_right = cp.empty(total_pairs, dtype=cp.int32)
        scatter_kernel = kernels["bbox_overlap_multi_scatter"]
        scatter_params = (
            (
                ptr(device_query_bounds),
                ptr(device_tree_bounds),
                query_count,
                tree_count,
                ptr(device_offsets),
                ptr(device_left),
                ptr(device_right),
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
            scatter_kernel, grid=scatter_grid, block=scatter_block, params=scatter_params,
        )

        # Synchronize before the finally block frees input buffers —
        # the scatter kernel may still be reading them asynchronously.
        runtime.synchronize()

        return _DeviceCandidates(
            d_left=device_left,
            d_right=device_right,
            total_pairs=total_pairs,
        )
    finally:
        runtime.free(device_query_bounds)
        runtime.free(device_tree_bounds)
        runtime.free(device_counts)
        runtime.free(device_offsets)


def _generate_candidates_gpu(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GPU bbox overlap candidate generation.

    Returns (left_indices, right_indices) as host np.int32 arrays,
    or None if GPU dispatch is skipped (not available or below crossover).
    """
    query_count = query_bounds.shape[0]
    tree_count = tree_bounds.shape[0]

    if query_count == 0 or tree_count == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    # Crossover check: total bbox overlap work must justify GPU launch
    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * tree_count,
        gpu_available=has_gpu_runtime(),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    if query_count == 1:
        return _generate_candidates_gpu_scalar(query_bounds[0], tree_bounds, tree_count)
    return _generate_candidates_gpu_multi(query_bounds, tree_bounds, query_count, tree_count)


def _generate_distance_pairs_gpu(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    distances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GPU bbox overlap candidate generation for dwithin.

    Expands query bounds by per-row distances, then delegates to the
    standard GPU bbox overlap pipeline.  Returns (left, right) host
    int32 arrays, or None if GPU dispatch is skipped.
    """
    expanded = _expand_bounds(query_bounds, distances)
    return _generate_candidates_gpu(expanded, tree_bounds)


def _generate_distance_pairs_gpu_device(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    distances: np.ndarray,
) -> _DeviceCandidates | None:
    """GPU bbox overlap candidate generation for dwithin, device-resident.

    Expands query bounds by per-row distances, then delegates to
    ``_generate_candidates_gpu_device``.
    """
    expanded = _expand_bounds(query_bounds, distances)
    return _generate_candidates_gpu_device(expanded, tree_bounds)


def _count_candidates_gpu(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
) -> int | None:
    """Count bbox overlap candidates on GPU without materializing pairs.

    Runs only the count kernel + exclusive sum, skipping the scatter pass
    entirely.  Returns the total candidate pair count, or None if GPU
    dispatch is skipped.  This is the optimal path for ``output_format="count"``
    queries where the caller needs only the total, not the pair arrays.
    """
    query_count = query_bounds.shape[0]
    tree_count = tree_bounds.shape[0]

    if query_count == 0 or tree_count == 0:
        return 0

    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * tree_count,
        gpu_available=has_gpu_runtime(),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    runtime = get_cuda_runtime()
    device_query_bounds = runtime.from_host(
        np.ascontiguousarray(query_bounds, dtype=np.float64).ravel()
    )
    device_tree_bounds = runtime.from_host(
        np.ascontiguousarray(tree_bounds, dtype=np.float64).ravel()
    )
    device_counts = runtime.allocate((query_count,), np.int32)
    try:
        kernels = _spatial_query_kernels()
        ptr = runtime.pointer

        count_kernel = kernels["bbox_overlap_multi_count"]
        count_params = (
            (
                ptr(device_query_bounds),
                ptr(device_tree_bounds),
                query_count,
                tree_count,
                ptr(device_counts),
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
            count_kernel, grid=count_grid, block=count_block, params=count_params,
        )

        device_offsets = exclusive_sum(device_counts)
        try:
            total_pairs = count_scatter_total(runtime, device_counts, device_offsets)
            return total_pairs
        finally:
            runtime.free(device_offsets)
    finally:
        # LIFO deallocation order for optimal pool coalescing
        runtime.free(device_counts)
        runtime.free(device_tree_bounds)
        runtime.free(device_query_bounds)


def _generate_candidates_gpu_device(
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
) -> _DeviceCandidates | None:
    """GPU bbox overlap candidate generation, returning device-resident arrays.

    Returns _DeviceCandidates or None if GPU dispatch is skipped.
    """
    query_count = query_bounds.shape[0]
    tree_count = tree_bounds.shape[0]

    if query_count == 0 or tree_count == 0:
        return None

    selection = plan_dispatch_selection(
        kernel_name="bbox_overlap_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * tree_count,
        gpu_available=has_gpu_runtime(),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    if query_count == 1:
        # Scalar path returns host arrays; wrap in DeviceCandidates.
        result = _generate_candidates_gpu_scalar(query_bounds[0], tree_bounds, tree_count)
        if result is None:
            return None
        left_host, right_host = result
        if left_host.size == 0:
            return None
        import cupy as cp
        return _DeviceCandidates(
            d_left=cp.asarray(left_host),
            d_right=cp.asarray(right_host),
            total_pairs=int(left_host.size),
        )

    return _generate_candidates_gpu_multi_device(query_bounds, tree_bounds, query_count, tree_count)


# ---------------------------------------------------------------------------
# Morton range candidate generation (Tier 1 NVRTC + Tier 3a CCCL binary search)
# ---------------------------------------------------------------------------
# ADR-0033: Morton range computation is geometry-specific compute (Tier 1),
# binary search uses CCCL lower_bound/upper_bound (Tier 3a), exclusive scan
# for offset computation uses CCCL exclusive_sum (Tier 3a).
# Replaces O(N*M) brute-force candidate generation with O(N*log(M)+K).

def _generate_candidates_morton_range_gpu(
    flat_index,
    query_bounds: np.ndarray,
) -> _DeviceCandidates | None:
    """Morton range candidate generation on GPU — O(N*log(M)+K).

    Uses the FlatSpatialIndex's sorted Morton keys to narrow candidates via
    LCP-based range binary search, then refines with bbox overlap.
    """
    query_count = query_bounds.shape[0]
    tree_count = flat_index.size
    if query_count == 0 or tree_count == 0:
        return None

    selection = plan_dispatch_selection(
        kernel_name="morton_range_candidates",
        kernel_class=KernelClass.COARSE,
        row_count=query_count * tree_count,
        gpu_available=has_gpu_runtime(),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    total_bounds = flat_index.total_bounds
    if np.isnan(total_bounds[0]):
        return None

    import cupy as cp

    runtime = get_cuda_runtime()

    # Morton codes encode bbox centers, so a tree geometry whose center is
    # outside the query's Morton range can still overlap the query if the
    # geometry is large enough.  Expand query bounds by the maximum tree
    # geometry half-extent to ensure the Morton range is conservative.
    tree_bounds_arr = flat_index.bounds
    valid_tree = tree_bounds_arr[~np.isnan(tree_bounds_arr).any(axis=1)]
    if valid_tree.size == 0:
        return None
    max_half_w = float((valid_tree[:, 2] - valid_tree[:, 0]).max()) / 2.0
    max_half_h = float((valid_tree[:, 3] - valid_tree[:, 1]).max()) / 2.0

    # Expanded bounds for Morton range lookup; original bounds for bbox refinement.
    expanded_bounds = query_bounds.copy()
    expanded_bounds[:, 0] -= max_half_w
    expanded_bounds[:, 1] -= max_half_h
    expanded_bounds[:, 2] += max_half_w
    expanded_bounds[:, 3] += max_half_h

    # Prepare host data: sort tree bounds by Morton order for sequential access.
    sorted_keys_host = flat_index.morton_keys[flat_index.order].astype(np.uint64)
    sorted_tree_bounds_host = np.ascontiguousarray(
        flat_index.bounds[flat_index.order], dtype=np.float64,
    )
    query_bounds_host = np.ascontiguousarray(query_bounds, dtype=np.float64)
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

        # Step 1: Compute Morton ranges from *expanded* bounds (Tier 1 NVRTC).
        range_kernel = kernels["morton_range_from_bounds"]
        range_params = (
            (
                ptr(d_expanded_bounds),
                float(total_bounds[0]), float(total_bounds[1]),
                float(total_bounds[2]), float(total_bounds[3]),
                ptr(d_range_low), ptr(d_range_high),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        range_grid, range_block = runtime.launch_config(range_kernel, query_count)
        runtime.launch(
            range_kernel, grid=range_grid, block=range_block, params=range_params,
        )

        # Step 2: Binary search on sorted Morton keys (Tier 3a CCCL).
        d_starts = lower_bound(d_sorted_keys, d_range_low, synchronize=False)
        d_ends = upper_bound(d_sorted_keys, d_range_high, synchronize=False)

        # Step 3: Count bbox overlaps per query within Morton range (Tier 1).
        count_kernel = kernels["morton_range_count"]
        count_params = (
            (
                ptr(d_starts), ptr(d_ends),
                ptr(d_sorted_tree_bounds), ptr(d_query_bounds),
                ptr(d_counts), query_count,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_I32,
            ),
        )
        count_grid, count_block = runtime.launch_config(count_kernel, query_count)
        runtime.launch(
            count_kernel, grid=count_grid, block=count_block, params=count_params,
        )

        # Step 4: Exclusive scan for output offsets (Tier 3a CCCL).
        d_offsets = exclusive_sum(d_counts)

        total_pairs = count_scatter_total(runtime, d_counts, d_offsets) if query_count > 0 else 0

        if total_pairs == 0:
            return None

        # Step 5: Scatter matching pairs (Tier 1 NVRTC).
        d_left = cp.empty(total_pairs, dtype=np.int32)
        d_right = cp.empty(total_pairs, dtype=np.int32)
        scatter_kernel = kernels["morton_range_scatter"]
        scatter_params = (
            (
                ptr(d_starts), ptr(d_ends),
                ptr(d_order), ptr(d_sorted_tree_bounds),
                ptr(d_query_bounds), ptr(d_offsets),
                ptr(d_left), ptr(d_right),
                query_count,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(scatter_kernel, query_count)
        runtime.launch(
            scatter_kernel, grid=scatter_grid, block=scatter_block,
            params=scatter_params,
        )

        # Synchronize before the finally block frees input buffers —
        # the scatter kernel may still be reading them asynchronously.
        runtime.synchronize()

        result = _DeviceCandidates(
            d_left=d_left, d_right=d_right, total_pairs=total_pairs,
        )
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
