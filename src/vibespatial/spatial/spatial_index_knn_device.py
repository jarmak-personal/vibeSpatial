"""Device-side k-nearest-neighbor spatial query.

Provides ``spatial_index_knn_device``, a GPU-accelerated k-NN spatial query
that replaces the CPU STRtree nearest path in sjoin_nearest.

Pipeline:
  1. Expand query bounds by max_distance (or progressive expansion)
  2. Generate bbox candidate pairs via ``spatial_index_device_query``
  3. Compute exact distances for candidate pairs (reuses nearest.py strategies)
  4. Per-query top-k selection via CCCL segmented_sort
  5. Output: device-resident (query_idx, target_idx, distance) triples

ADR-0002: METRIC kernel class for distance computation -- fp64 required.
ADR-0033 tier classification:
  - Tier 2 (CuPy): element-wise distance filtering, gather/scatter
  - Tier 3a (CCCL): segmented_sort (per-query ranking), exclusive_sum,
    lower_bound, upper_bound, compact_indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
except ModuleNotFoundError:  # pragma: no cover - CPU-only installs
    cp = None

from vibespatial.cuda._runtime import (
    get_cuda_runtime,
)
from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    lower_bound_counting,
    segmented_sort,
    sort_pairs,
    upper_bound_counting,
)
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass, PrecisionMode

from .query_candidates import _generate_candidates_gpu_device
from .query_utils import _expand_bounds

logger = logging.getLogger(__name__)

# Eagerly request CCCL spec warmup at module import (ADR-0034 Level 1).
request_warmup([
    "exclusive_scan_i32",
    "exclusive_scan_i64",
    "select_i32",
    "select_i64",
    "radix_sort_i32_i32",
    "segmented_sort_asc_f64",
    "segmented_reduce_min_f64",
    "lower_bound_i32",
    "upper_bound_i32",
])


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeviceKnnResult:
    """Device-resident k-NN query result.

    All arrays are CuPy device arrays to avoid D->H transfers when the
    result feeds directly into the next GPU pipeline stage (e.g., sjoin
    attribute assembly).

    Attributes
    ----------
    d_query_idx : device int32 array
        Query geometry indices (one per result pair).
    d_target_idx : device int32 array
        Target geometry indices (one per result pair).
    d_distances : device float64 array
        Exact distances for each (query, target) pair.
    total_pairs : int
        Number of result pairs.
    k : int
        Requested k value.
    """
    d_query_idx: Any   # CuPy int32 device array
    d_target_idx: Any  # CuPy int32 device array
    d_distances: Any   # CuPy float64 device array
    total_pairs: int
    k: int

    def to_host(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy result to host as numpy arrays."""
        runtime = get_cuda_runtime()
        query_idx = runtime.copy_device_to_host(self.d_query_idx).astype(np.int32, copy=False)
        target_idx = runtime.copy_device_to_host(self.d_target_idx).astype(np.int32, copy=False)
        distances = runtime.copy_device_to_host(self.d_distances).astype(np.float64, copy=False)
        return query_idx, target_idx, distances


# ---------------------------------------------------------------------------
# Distance computation dispatch (reuses nearest.py infrastructure)
# ---------------------------------------------------------------------------

def _select_distance_strategy(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
):
    """Select the appropriate GPU distance computation strategy.

    Returns a DistanceStrategy instance or None if no GPU path is available.
    """
    from .nearest import (
        PointFamilyDistanceStrategy,
        PointPointDistanceStrategy,
        SegmentFamilyDistanceStrategy,
        _point_distance_families,
        _points_only,
        _single_family,
        _tree_distance_family,
    )

    if _points_only(query_owned) and _points_only(tree_owned):
        from vibespatial.geometry.buffers import GeometryFamily
        if (GeometryFamily.POINT in query_owned.families
                and GeometryFamily.POINT in tree_owned.families):
            return PointPointDistanceStrategy()
        return None

    if _points_only(query_owned):
        tree_family = _tree_distance_family(tree_owned)
        if tree_family is not None:
            return PointFamilyDistanceStrategy(tree_family)
        return None

    if _points_only(tree_owned):
        query_family = _single_family(query_owned)
        if query_family is not None and query_family in _point_distance_families():
            # Reverse: compute distance from tree (points) to query family
            return PointFamilyDistanceStrategy(query_family)
        return None

    query_family = _single_family(query_owned)
    tree_family = _single_family(tree_owned)
    if query_family is not None and tree_family is not None:
        return SegmentFamilyDistanceStrategy(query_family, tree_family)

    return None


def _compute_pair_distances(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_left: Any,
    d_right: Any,
    pair_count: int,
) -> Any | None:
    """Compute exact distances for candidate pairs on device.

    Returns a CuPy float64 device array of shape (pair_count,), or None
    if no GPU distance kernel is available for the geometry families.
    """
    strategy = _select_distance_strategy(query_owned, tree_owned)
    if strategy is None:
        return None

    runtime = get_cuda_runtime()
    d_distances = runtime.allocate((pair_count,), cp.float64)

    # Check if we need to reverse the query/tree for point-to-family distance.
    from .nearest import (
        PointFamilyDistanceStrategy,
        _points_only,
    )
    if (_points_only(tree_owned) and not _points_only(query_owned)
            and isinstance(strategy, PointFamilyDistanceStrategy)):
        # Reverse: strategy computes point->family, so swap left/right
        ok = strategy.compute(
            tree_owned, query_owned,
            d_right, d_left, d_distances, pair_count,
        )
    else:
        ok = strategy.compute(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
        )

    if not ok:
        runtime.free(d_distances)
        return None
    return cp.asarray(d_distances)


# ---------------------------------------------------------------------------
# Per-query top-k selection
# ---------------------------------------------------------------------------

def _topk_per_query(
    d_query_idx: Any,
    d_target_idx: Any,
    d_distances: Any,
    n_queries: int,
    k: int,
    *,
    max_distance: float | None = None,
) -> tuple[Any, Any, Any, int]:
    """Select the k nearest targets per query from unsorted candidate pairs.

    Parameters
    ----------
    d_query_idx, d_target_idx, d_distances
        Device arrays of candidate pairs (not necessarily sorted).
    n_queries : int
        Total number of query geometries.
    k : int
        Number of nearest neighbours to keep per query.
    max_distance : float or None
        If not None, prune candidates beyond this distance before ranking.

    Returns
    -------
    (d_out_query, d_out_target, d_out_dist, total_pairs) : device arrays + count
        The kept (query, target, distance) triples and total pair count.
    """
    pair_count = int(d_query_idx.size)

    # Step 1: Apply max_distance filter if specified.
    if max_distance is not None and np.isfinite(max_distance):
        keep_mask = (d_distances <= max_distance).astype(cp.uint8)
        compacted = compact_indices(keep_mask)
        if compacted.count == 0:
            empty_i = cp.empty(0, dtype=cp.int32)
            empty_f = cp.empty(0, dtype=cp.float64)
            return empty_i, empty_i, empty_f, 0
        kept = compacted.values
        d_query_idx = d_query_idx[kept]
        d_target_idx = d_target_idx[kept]
        d_distances = d_distances[kept]
        pair_count = compacted.count

    if pair_count == 0:
        empty_i = cp.empty(0, dtype=cp.int32)
        empty_f = cp.empty(0, dtype=cp.float64)
        return empty_i, empty_i, empty_f, 0

    # Step 2: Group by query_idx, then segmented-sort distances within each group.
    # Sort pairs by query_idx using CCCL radix sort (Tier 3a).
    pair_indices = cp.arange(pair_count, dtype=cp.int32)
    sorted_by_query = sort_pairs(d_query_idx, pair_indices, synchronize=False)
    d_sorted_query = sorted_by_query.keys
    d_order = sorted_by_query.values
    d_sorted_target = d_target_idx[d_order]
    d_sorted_dist = d_distances[d_order]

    # Step 3: Build segment boundaries from sorted query indices (Tier 3a CCCL).
    seg_starts = lower_bound_counting(
        d_sorted_query, 0, n_queries, dtype=np.int32,
    ).astype(cp.int32, copy=False)
    seg_ends = upper_bound_counting(
        d_sorted_query, 0, n_queries, dtype=np.int32,
    ).astype(cp.int32, copy=False)

    # Step 4: Segmented sort by distance within each query segment (Tier 3a CCCL).
    seg_sort_result = segmented_sort(
        d_sorted_dist,
        values=d_sorted_target,
        starts=seg_starts,
        ends=seg_ends,
        num_segments=n_queries,
        synchronize=False,
    )
    d_segdist = seg_sort_result.keys
    d_segtarget = seg_sort_result.values

    # Step 5: Extract first k per segment (Tier 2 CuPy element-wise).
    # For each position, compute its local offset within the segment;
    # keep only elements where local_offset < k.
    d_positions = cp.arange(pair_count, dtype=cp.int32)
    d_local_offsets = d_positions - seg_starts[d_sorted_query]
    d_keep = (d_local_offsets < k).astype(cp.uint8)

    # Also ensure the element is valid (within segment bounds).
    d_keep &= (d_positions < seg_ends[d_sorted_query]).astype(cp.uint8)

    compacted = compact_indices(d_keep)
    if compacted.count == 0:
        empty_i = cp.empty(0, dtype=cp.int32)
        empty_f = cp.empty(0, dtype=cp.float64)
        return empty_i, empty_i, empty_f, 0

    kept = compacted.values
    d_out_query = d_sorted_query[kept]
    d_out_target = d_segtarget[kept]
    d_out_dist = d_segdist[kept]

    return d_out_query, d_out_target, d_out_dist, compacted.count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spatial_index_knn_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    query_bounds: np.ndarray,
    tree_bounds: np.ndarray,
    *,
    k: int = 1,
    max_distance: float | None = None,
    exclusive: bool = False,
    return_all: bool = True,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> DeviceKnnResult | None:
    """GPU-accelerated k-nearest-neighbor spatial query.

    Replaces the CPU STRtree nearest path with a fully device-resident
    pipeline: candidate generation -> exact distance -> per-query top-k.

    Parameters
    ----------
    query_owned : OwnedGeometryArray
        Query geometries (source of the nearest search).
    tree_owned : OwnedGeometryArray
        Target geometries (the "tree" to search against).
    query_bounds : np.ndarray, shape (Q, 4)
        Pre-computed query bounding boxes.
    tree_bounds : np.ndarray, shape (M, 4)
        Pre-computed target bounding boxes.
    k : int
        Number of nearest neighbours per query.  k=1 is the most common case.
    max_distance : float or None
        Maximum search distance.  Candidates beyond this are pruned.
        When None, an effective distance is computed from the data extent.
    exclusive : bool
        If True, exclude identical geometries from results.
    return_all : bool
        If True, return all k-nearest ties.  If False, return exactly one
        per query (the first nearest).
    precision : PrecisionMode
        Precision mode for distance computation.  METRIC class requires
        fp64 per ADR-0002 on all devices.

    Returns
    -------
    DeviceKnnResult or None
        Device-resident result with (query_idx, target_idx, distance)
        triples, or None if the GPU path is not applicable.
    """
    if not has_gpu_runtime() or cp is None:
        return None

    n_queries = query_bounds.shape[0]
    n_tree = tree_bounds.shape[0]
    if n_queries == 0 or n_tree == 0:
        return DeviceKnnResult(
            d_query_idx=cp.empty(0, dtype=cp.int32),
            d_target_idx=cp.empty(0, dtype=cp.int32),
            d_distances=cp.empty(0, dtype=cp.float64),
            total_pairs=0,
            k=k,
        )

    # Dispatch check: assess whether GPU is profitable for this workload.
    selection = plan_dispatch_selection(
        kernel_name="spatial_index_knn",
        kernel_class=KernelClass.METRIC,
        row_count=n_queries * min(n_tree, 100),  # estimate work per query
        requested_precision=precision,
        gpu_available=True,
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    # Check that we have a GPU distance strategy for this family combination.
    strategy = _select_distance_strategy(query_owned, tree_owned)
    if strategy is None:
        logger.debug(
            "spatial_index_knn_device: no GPU distance strategy for "
            "query=%s / tree=%s families",
            list(query_owned.families.keys()),
            list(tree_owned.families.keys()),
        )
        return None

    # Compute effective max_distance for candidate generation.
    if max_distance is not None and np.isfinite(max_distance):
        effective_max_distance = float(max_distance)
    else:
        # Compute bounding box diagonal of all valid geometry bounds on device.
        d_query_bounds = cp.asarray(query_bounds)
        d_tree_bounds = cp.asarray(tree_bounds)
        d_all_bounds = cp.vstack((d_query_bounds, d_tree_bounds))
        d_valid_mask = ~cp.isnan(d_all_bounds).any(axis=1)
        if not bool(d_valid_mask.any()):
            return DeviceKnnResult(
                d_query_idx=cp.empty(0, dtype=cp.int32),
                d_target_idx=cp.empty(0, dtype=cp.int32),
                d_distances=cp.empty(0, dtype=cp.float64),
                total_pairs=0,
                k=k,
            )
        d_valid_bounds = d_all_bounds[d_valid_mask]
        extent_dx = float(d_valid_bounds[:, 2].max() - d_valid_bounds[:, 0].min())
        extent_dy = float(d_valid_bounds[:, 3].max() - d_valid_bounds[:, 1].min())
        effective_max_distance = float(cp.hypot(extent_dx, extent_dy)) * 1.01 + 1.0

    # --- Candidate generation ------------------------------------------------
    # Expand query bounds by effective_max_distance and find bbox overlaps.
    # Use the device-resident candidate generator to avoid D->H->D round-trip.
    # NOTE: _expand_bounds and _generate_candidates_gpu_device expect host
    # numpy arrays -- do not pass CuPy arrays to them.
    per_row_dist = np.full(n_queries, effective_max_distance, dtype=np.float64)
    expanded_bounds = _expand_bounds(query_bounds, per_row_dist)

    device_candidates = _generate_candidates_gpu_device(expanded_bounds, tree_bounds)
    if device_candidates is not None:
        d_left_cp = device_candidates.d_left
        d_right_cp = device_candidates.d_right
        pair_count = device_candidates.total_pairs
    else:
        # CPU fallback for candidate generation (small workloads).
        from .query_candidates import _generate_distance_pairs
        left_idx_h, right_idx_h = _generate_distance_pairs(
            query_bounds, tree_bounds, per_row_dist,
        )
        if left_idx_h.size == 0:
            return DeviceKnnResult(
                d_query_idx=cp.empty(0, dtype=cp.int32),
                d_target_idx=cp.empty(0, dtype=cp.int32),
                d_distances=cp.empty(0, dtype=cp.float64),
                total_pairs=0,
                k=k,
            )
        pair_count = int(left_idx_h.size)
        d_left_cp = cp.asarray(left_idx_h, dtype=cp.int32)
        d_right_cp = cp.asarray(right_idx_h, dtype=cp.int32)

    if pair_count == 0:
        return DeviceKnnResult(
            d_query_idx=cp.empty(0, dtype=cp.int32),
            d_target_idx=cp.empty(0, dtype=cp.int32),
            d_distances=cp.empty(0, dtype=cp.float64),
            total_pairs=0,
            k=k,
        )

    # --- Exact distance computation ------------------------------------------
    # Sort by left_idx for the distance strategy (some require sorted input).
    sorted_result = sort_pairs(d_left_cp, d_right_cp, synchronize=False)
    d_sorted_left = sorted_result.keys
    d_sorted_right = sorted_result.values

    d_distances = _compute_pair_distances(
        query_owned, tree_owned,
        d_sorted_left, d_sorted_right, pair_count,
    )
    if d_distances is None:
        return None

    # --- Handle exclusive flag: mark self-matches with infinity --------------
    if exclusive:
        self_match = (d_sorted_left == d_sorted_right)
        d_distances = cp.where(self_match, cp.inf, d_distances)

    # --- Per-query top-k selection -------------------------------------------
    d_out_query, d_out_target, d_out_dist, total_pairs = _topk_per_query(
        d_sorted_left,
        d_sorted_right,
        d_distances,
        n_queries,
        k,
        max_distance=effective_max_distance if max_distance is not None else None,
    )

    # When return_all is False and k=1, keep only the first result per query.
    if not return_all and total_pairs > 0:
        # Already sorted by (query, distance); take the first per query.
        if k == 1:
            # k=1 already has at most 1 per query, nothing more to do.
            pass
        else:
            # Deduplicate: keep only the first occurrence of each query_idx.
            unique_mask = cp.zeros(total_pairs, dtype=cp.uint8)
            if total_pairs > 0:
                unique_mask[0] = 1
                unique_mask[1:] = (d_out_query[1:] != d_out_query[:-1]).astype(cp.uint8)
            compacted = compact_indices(unique_mask)
            if compacted.count > 0:
                kept = compacted.values
                d_out_query = d_out_query[kept]
                d_out_target = d_out_target[kept]
                d_out_dist = d_out_dist[kept]
                total_pairs = compacted.count
            else:
                total_pairs = 0

    return DeviceKnnResult(
        d_query_idx=d_out_query,
        d_target_idx=d_out_target,
        d_distances=d_out_dist,
        total_pairs=total_pairs,
        k=k,
    )
