"""Device-side k-nearest-neighbor spatial query.

Provides ``spatial_index_knn_device``, a GPU-accelerated k-NN spatial query
that replaces the CPU STRtree nearest path in sjoin_nearest.

Pipeline:
  1. Admit a capacity-bounded candidate and retained-top-k workspace
  2. Expand unresolved query bounds with bounded query/target streaming
  3. Compute exact/refined distances only for each newly entered radius shell
  4. Merge deterministic per-query top-k rows with CCCL segmented sorts
  5. Output device-resident (query_idx, target_idx, distance) triples

ADR-0002: METRIC precision plan with selective fp64 ranking refinement.
ADR-0033 tier classification:
  - Tier 2 (CuPy): element-wise distance filtering, gather/scatter
  - Tier 3a (CCCL): segmented_sort (per-query ranking), exclusive_sum,
    compact_indices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
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
    exclusive_sum,
    segmented_sort,
    sort_pairs,
)
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, combined_residency, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.crossover import PhysicalWorkEstimate
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.precision import KernelClass, PrecisionMode
from vibespatial.spatial.query_types import (
    CandidateRelationCapacityError,
    _DeviceCandidates,
    _record_device_join_materialization,
)

from .spatial_index_device import spatial_index_device_query

logger = logging.getLogger(__name__)

# Eagerly request CCCL spec warmup at module import (ADR-0034 Level 1).
request_warmup(
    [
        "exclusive_scan_i32",
        "exclusive_scan_i64",
        "select_i32",
        "select_i64",
        "radix_sort_i32_i32",
        "segmented_sort_asc_f64",
        "segmented_sort_asc_i32",
        "segmented_reduce_min_f64",
    ]
)


# This is a simultaneous-live workspace estimate, not an allocator headroom
# guess.  One candidate can participate in candidate COO, distance, query
# ordering, segmented ordering, compaction, and primitive scratch before the
# chunk is reduced to k retained rows.  Keeping the estimate deliberately
# aligned to 128 bytes makes admission conservative and auditable.
_KNN_BYTES_PER_CANDIDATE = 128
_KNN_FIXED_BYTES_PER_QUERY = 128
_KNN_BYTES_PER_RETAINED_SLOT = 64


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

    d_query_idx: Any  # CuPy int32 device array
    d_target_idx: Any  # CuPy int32 device array
    d_distances: Any  # CuPy float64 device array
    total_pairs: int
    k: int
    telemetry: DeviceKnnTelemetry | None = None

    def to_host(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy result to host as numpy arrays."""
        runtime = get_cuda_runtime()
        _record_device_join_materialization(
            self.d_query_idx,
            side="query",
            surface="vibespatial.spatial.spatial_index_knn_device.DeviceKnnResult.to_host",
            operation="device_knn_indices_to_host",
            reason="device kNN query indices crossed to host for public nearest export",
        )
        query_idx = runtime.copy_device_to_host(
            self.d_query_idx,
            reason="device kNN query-index host export",
        ).astype(np.int32, copy=False)
        _record_device_join_materialization(
            self.d_target_idx,
            side="target",
            surface="vibespatial.spatial.spatial_index_knn_device.DeviceKnnResult.to_host",
            operation="device_knn_indices_to_host",
            reason="device kNN target indices crossed to host for public nearest export",
        )
        target_idx = runtime.copy_device_to_host(
            self.d_target_idx,
            reason="device kNN target-index host export",
        ).astype(np.int32, copy=False)
        _record_device_join_materialization(
            self.d_distances,
            side="distances",
            surface="vibespatial.spatial.spatial_index_knn_device.DeviceKnnResult.to_host",
            operation="device_knn_distances_to_host",
            reason="device kNN distances crossed to host for public nearest export",
        )
        distances = runtime.copy_device_to_host(
            self.d_distances,
            reason="device kNN distance host export",
        ).astype(np.float64, copy=False)
        return query_idx, target_idx, distances


@dataclass(slots=True)
class DeviceKnnTelemetry:
    """Host-known control telemetry for one bounded fixed-k execution."""

    query_tile_rows: int
    target_tile_rows: int
    pair_capacity: int
    admitted_workspace_bytes: int
    radius_iterations: int = 0
    candidate_pairs: int = 0
    max_candidate_pairs: int = 0
    query_tiles: int = 0
    target_stream_tiles: int = 0
    scalar_fences: int = 0
    peak_device_bytes: int | None = None
    allocation_count: int | None = None
    d2h_count: int = 0
    d2h_bytes: int = 0
    materialization_count: int = 0

    def detail(self) -> str:
        return (
            f"query_tile_rows={self.query_tile_rows}, "
            f"target_tile_rows={self.target_tile_rows}, "
            f"pair_capacity={self.pair_capacity}, "
            f"workspace_bytes={self.admitted_workspace_bytes}, "
            f"query_tiles={self.query_tiles}, "
            f"target_stream_tiles={self.target_stream_tiles}, "
            f"radius_iterations={self.radius_iterations}, "
            f"candidate_pairs={self.candidate_pairs}, "
            f"max_candidate_pairs={self.max_candidate_pairs}, "
            f"scalar_fences={self.scalar_fences}, "
            f"peak_device_bytes={self.peak_device_bytes}, "
            f"allocation_count={self.allocation_count}, "
            f"d2h_count={self.d2h_count}, "
            f"d2h_bytes={self.d2h_bytes}, "
            f"materialization_count={self.materialization_count}"
        )


class _OperationAllocationMonitor:
    """Measure allocations made inside one fixed-k operation-local RMM scope."""

    def __init__(self) -> None:
        self._statistics = None
        self._active = False
        if not has_gpu_runtime():
            return
        try:
            from rmm import statistics
        except ImportError:
            return
        statistics.enable_statistics()
        statistics.push_statistics()
        self._statistics = statistics
        self._active = True

    def finish(self) -> tuple[int | None, int | None]:
        """Seal the nested scope and return operation-local peak/count."""
        if not self._active or self._statistics is None:
            return None, None
        stats = self._statistics.pop_statistics()
        self._active = False
        if stats is None:
            return None, None
        return int(stats.peak_bytes), int(stats.total_count)


@dataclass(frozen=True, slots=True)
class _KnnWorkspacePlan:
    query_tile_rows: int
    target_tile_rows: int
    pair_capacity: int
    final_output_bytes: int
    tile_fixed_bytes: int
    admitted_workspace_bytes: int


def _plan_knn_workspace(n_queries: int, n_tree: int, k: int) -> _KnnWorkspacePlan:
    """Admit a worst-case fixed-k tile against live device memory."""
    runtime = get_cuda_runtime()
    remaining = int(runtime.query_memory_remaining_bytes())
    retained_slots = int(n_queries) * int(k)
    # Dense retained target/distance state plus the compact terminal relation.
    final_output_bytes = retained_slots * 32
    fixed_per_query = _KNN_FIXED_BYTES_PER_QUERY + (
        int(k) * _KNN_BYTES_PER_RETAINED_SLOT
    )
    minimum = final_output_bytes + fixed_per_query + _KNN_BYTES_PER_CANDIDATE
    if minimum > remaining:
        raise CandidateRelationCapacityError(
            "bounded fixed-k nearest cannot admit its minimum device workspace: "
            f"required={minimum:,} bytes, remaining={remaining:,} bytes, "
            f"queries={n_queries:,}, targets={n_tree:,}, k={k}"
        )

    available_for_tile = remaining - final_output_bytes
    max_pair_capacity = min(
        int(np.iinfo(np.int32).max),
        int(
            (available_for_tile - fixed_per_query)
            // _KNN_BYTES_PER_CANDIDATE
        ),
    )
    full_tree_query_bytes = (
        int(n_tree) * _KNN_BYTES_PER_CANDIDATE + fixed_per_query
    )
    if int(n_tree) <= max_pair_capacity:
        query_tile_rows = max(
            1,
            min(
                int(n_queries),
                available_for_tile // full_tree_query_bytes,
                int(np.iinfo(np.int32).max) // int(n_tree),
            ),
        )
        target_tile_rows = int(n_tree)
        pair_capacity = query_tile_rows * int(n_tree)
    else:
        query_tile_rows = 1
        target_tile_rows = min(max_pair_capacity, int(n_tree))
        pair_capacity = target_tile_rows

    tile_fixed_bytes = query_tile_rows * fixed_per_query
    required_bytes = (
        final_output_bytes
        + tile_fixed_bytes
        + pair_capacity * _KNN_BYTES_PER_CANDIDATE
    )
    admission = runtime.admit_device_memory(
        stage="bounded fixed-k nearest workspace",
        required_bytes=required_bytes,
        requested_units=pair_capacity,
    )
    if not admission.admitted:
        raise CandidateRelationCapacityError(
            "bounded fixed-k nearest workspace changed before admission: "
            f"required={required_bytes:,} bytes, "
            f"remaining={admission.remaining_bytes:,} bytes, "
            f"admitted_pairs={admission.admitted_units:,}"
        )
    return _KnnWorkspacePlan(
        query_tile_rows=query_tile_rows,
        target_tile_rows=target_tile_rows,
        pair_capacity=pair_capacity,
        final_output_bytes=final_output_bytes,
        tile_fixed_bytes=tile_fixed_bytes,
        admitted_workspace_bytes=required_bytes,
    )


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
    grouped_by_query: bool = False,
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

    # Step 1: discard non-finite distances and apply max_distance when set.
    keep_mask = cp.isfinite(d_distances)
    if max_distance is not None and np.isfinite(max_distance):
        keep_mask &= d_distances <= max_distance
    compacted = compact_indices(keep_mask.astype(cp.uint8))
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

    # Step 2: Group by query when the caller did not preserve count/scatter
    # order. Fixed-k candidate chunks arrive grouped and skip this relation
    # sort entirely.
    if grouped_by_query:
        d_sorted_query = d_query_idx.astype(cp.int32, copy=False)
        d_sorted_target = d_target_idx
        d_sorted_dist = d_distances
    else:
        pair_indices = cp.arange(pair_count, dtype=cp.int32)
        sorted_by_query = sort_pairs(d_query_idx, pair_indices, synchronize=False)
        d_sorted_query = sorted_by_query.keys
        d_order = sorted_by_query.values
        d_sorted_target = d_target_idx[d_order]
        d_sorted_dist = d_distances[d_order]

    # Step 3: Dense counts + scan are the authoritative segment boundaries.
    # This avoids lower/upper-bound searches over query identifiers.
    d_counts = cp.bincount(d_sorted_query, minlength=n_queries).astype(
        cp.int32,
        copy=False,
    )
    seg_starts = exclusive_sum(d_counts, synchronize=False).astype(
        cp.int32,
        copy=False,
    )
    seg_ends = seg_starts + d_counts

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

    # CCCL's distance sort does not promise a target-row tie order.  Split the
    # sorted relation into equal-distance runs (also respecting query segment
    # boundaries), then sort target rows inside each run.  This establishes the
    # public deterministic total order (distance, target_row) without changing
    # fp64 distance bits or packing them into a lossy composite key.
    d_positions = cp.arange(pair_count, dtype=cp.int32)
    d_run_start = cp.ones(pair_count, dtype=cp.bool_)
    if pair_count > 1:
        d_run_start[1:] = (d_sorted_query[1:] != d_sorted_query[:-1]) | (
            d_segdist[1:] != d_segdist[:-1]
        )
    d_run_starts = cp.flatnonzero(d_run_start).astype(cp.int32, copy=False)
    run_count = int(d_run_starts.size)
    d_run_ends = cp.empty_like(d_run_starts)
    if run_count > 1:
        d_run_ends[:-1] = d_run_starts[1:]
    d_run_ends[-1] = pair_count
    target_tie_sort = segmented_sort(
        d_segtarget,
        values=d_segdist,
        starts=d_run_starts,
        ends=d_run_ends,
        num_segments=run_count,
        synchronize=False,
    )
    d_segtarget = target_tie_sort.keys
    d_segdist = target_tie_sort.values

    # Step 5: Extract first k per segment (Tier 2 CuPy element-wise).
    # For each position, compute its local offset within the segment;
    # keep only elements where local_offset < k.
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


def _scatter_topk_dense(
    d_query_idx,
    d_target_idx,
    d_distances,
    *,
    n_queries: int,
    k: int,
    out_targets,
    out_distances,
) -> None:
    """Scatter sorted compact top-k rows into reusable dense tile buffers."""
    out_targets.fill(-1)
    out_distances.fill(cp.inf)
    pair_count = int(d_query_idx.size)
    if pair_count == 0:
        return
    d_counts = cp.bincount(d_query_idx, minlength=n_queries).astype(
        cp.int32,
        copy=False,
    )
    seg_starts = exclusive_sum(d_counts, synchronize=False).astype(
        cp.int32,
        copy=False,
    )
    positions = cp.arange(pair_count, dtype=cp.int32)
    slots = positions - seg_starts[d_query_idx]
    out_targets[d_query_idx, slots] = d_target_idx
    out_distances[d_query_idx, slots] = d_distances


def _merge_topk_dense(
    best_targets,
    best_distances,
    d_candidate_query,
    d_candidate_target,
    d_candidate_distances,
    *,
    n_queries: int,
    k: int,
    merge_queries,
    merge_targets,
    merge_distances,
    candidate_targets,
    candidate_distances,
    scratch_targets,
    scratch_distances,
) -> tuple[Any, Any]:
    """Merge one bounded candidate top-k relation into retained tile state."""
    _scatter_topk_dense(
        d_candidate_query,
        d_candidate_target,
        d_candidate_distances,
        n_queries=n_queries,
        k=k,
        out_targets=candidate_targets,
        out_distances=candidate_distances,
    )
    merge_targets[:, :k] = best_targets
    merge_targets[:, k:] = candidate_targets
    merge_distances[:, :k] = best_distances
    merge_distances[:, k:] = candidate_distances
    d_target = merge_targets.reshape(-1)
    d_distance = merge_distances.reshape(-1)
    out_q, out_t, out_d, _ = _topk_per_query(
        merge_queries,
        d_target,
        d_distance,
        n_queries,
        k,
        grouped_by_query=True,
    )
    _scatter_topk_dense(
        out_q,
        out_t,
        out_d,
        n_queries=n_queries,
        k=k,
        out_targets=scratch_targets,
        out_distances=scratch_distances,
    )
    return scratch_targets, scratch_distances


def _device_any(mask, *, reason: str) -> bool:
    """Read one named loop-control scalar without exporting row state."""
    packed = cp.asarray(cp.any(mask), dtype=cp.uint8).reshape(1)
    host = get_cuda_runtime().copy_device_to_host(packed, reason=reason)
    return bool(host[0])


def _bbox_chebyshev_distance(d_query_bounds, d_tree_bounds):
    """Distance at which square-expanded query bounds first admit a pair."""
    dx = cp.maximum(
        cp.maximum(d_query_bounds[:, 0] - d_tree_bounds[:, 2], d_tree_bounds[:, 0] - d_query_bounds[:, 2]),
        0.0,
    )
    dy = cp.maximum(
        cp.maximum(d_query_bounds[:, 1] - d_tree_bounds[:, 3], d_tree_bounds[:, 1] - d_query_bounds[:, 3]),
        0.0,
    )
    return cp.maximum(dx, dy)


def _empty_knn_result(
    k: int,
    telemetry: DeviceKnnTelemetry | None = None,
) -> DeviceKnnResult:
    empty_i = cp.empty(0, dtype=cp.int32)
    return DeviceKnnResult(
        d_query_idx=empty_i,
        d_target_idx=cp.empty(0, dtype=cp.int32),
        d_distances=cp.empty(0, dtype=cp.float64),
        total_pairs=0,
        k=k,
        telemetry=telemetry,
    )


def _initial_search_radius(flat_index, *, n_tree: int, k: int, ceiling: float) -> float:
    """Estimate one conservative starting radius from retained index metadata."""
    total_bounds = flat_index.total_bounds
    if total_bounds is None or not np.all(np.isfinite(total_bounds)):
        return max(float(ceiling) * 1.0e-6, np.finfo(np.float64).eps)
    xmin, ymin, xmax, ymax = (float(value) for value in total_bounds)
    width = max(xmax - xmin, 0.0)
    height = max(ymax - ymin, 0.0)
    diagonal = float(np.hypot(width, height))
    area = width * height
    if area > 0.0:
        radius = np.sqrt(float(k) * area / (np.pi * max(int(n_tree), 1)))
    else:
        radius = diagonal * np.sqrt(float(k) / max(int(n_tree), 1))
    floor = max(diagonal, float(ceiling), 1.0) * 1.0e-9
    return min(max(float(radius), floor), float(ceiling))


def _target_range_flat_view(flat_index, d_tree_bounds, start: int, end: int):
    """Create a non-owning bounds range over retained index state."""
    return replace(
        flat_index,
        _host_order=None,
        _host_morton_keys=None,
        _host_bounds=None,
        regular_grid=None,
        device_morton_keys=None,
        device_order=None,
        device_bounds=d_tree_bounds[start:end],
        point_grid=None,
        _native_spatial_index=None,
    )


def _candidate_batches(
    flat_index,
    native_spatial_index,
    d_active_bounds,
    d_outer_radii,
    d_tree_bounds,
    *,
    workspace: _KnnWorkspacePlan,
    telemetry: DeviceKnnTelemetry,
    candidate_output,
):
    """Yield admitted candidate relations, streaming target ranges when needed."""
    n_tree = int(flat_index.size)
    if n_tree <= workspace.target_tile_rows:
        candidates, execution = spatial_index_device_query(
            flat_index,
            d_active_bounds,
            native_index=native_spatial_index,
            distance=d_outer_radii,
            candidate_output=candidate_output,
        )
        if execution.selected is not ExecutionMode.GPU:
            raise RuntimeError("bounded fixed-k candidate query declined GPU execution")
        if candidates is not None and candidates.total_pairs:
            if candidates.total_pairs > workspace.pair_capacity:
                raise CandidateRelationCapacityError(
                    "spatial index emitted more fixed-k candidates than admitted: "
                    f"pairs={candidates.total_pairs:,}, "
                    f"capacity={workspace.pair_capacity:,}"
                )
            yield candidates
        return

    for target_start in range(0, n_tree, workspace.target_tile_rows):
        target_end = min(n_tree, target_start + workspace.target_tile_rows)
        chunk_index = _target_range_flat_view(
            flat_index,
            d_tree_bounds,
            target_start,
            target_end,
        )
        candidates, execution = spatial_index_device_query(
            chunk_index,
            d_active_bounds,
            distance=d_outer_radii,
            candidate_output=candidate_output,
        )
        telemetry.target_stream_tiles += 1
        if execution.selected is not ExecutionMode.GPU:
            raise RuntimeError("bounded fixed-k target range declined GPU execution")
        if candidates is None or not candidates.total_pairs:
            continue
        if candidates.total_pairs > workspace.pair_capacity:
            raise CandidateRelationCapacityError(
                "target range emitted more fixed-k candidates than admitted: "
                f"pairs={candidates.total_pairs:,}, "
                f"capacity={workspace.pair_capacity:,}"
            )
        d_mapped_right = cp.asarray(candidates.d_right, dtype=cp.int32)
        d_mapped_right += target_start
        yield _DeviceCandidates(
            d_left=candidates.d_left,
            d_right=d_mapped_right,
            total_pairs=candidates.total_pairs,
            error_flag=candidates.error_flag,
        )


def _refined_candidate_topk(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    d_query_global,
    d_query_tile,
    d_target,
    *,
    n_tile_queries: int,
    k: int,
    max_distance: float | None,
    precision_context,
    distance_center_device,
) -> tuple[Any, Any, Any] | None:
    """Return an exact/refined top-k relation for one bounded candidate chunk."""
    from .nearest import _compute_mixed_distances_gpu_device

    pair_count = int(d_query_global.size)
    distance_result = _compute_mixed_distances_gpu_device(
        query_owned,
        tree_owned,
        d_query_global,
        d_target,
        precision_context=precision_context,
        center_device=distance_center_device,
    )
    if distance_result is None:
        return None
    d_distances, used_shapely_fallback = distance_result
    if used_shapely_fallback:
        return None

    # In a staged fp32 plan, do not apply max_distance until candidates near
    # that threshold have been refined.  A coarse distance can round just
    # outside the threshold even when the exact distance is inside it.
    coarse_max_distance = (
        max_distance if precision_context.refinement_plan is None else None
    )
    coarse_q, coarse_t, coarse_d, _ = _topk_per_query(
        d_query_tile,
        d_target,
        d_distances,
        n_tile_queries,
        k,
        max_distance=coarse_max_distance,
        grouped_by_query=True,
    )
    if precision_context.refinement_plan is None:
        return coarse_q, coarse_t, coarse_d

    threshold = cp.full(n_tile_queries, -cp.inf, dtype=cp.float64)
    if int(coarse_q.size):
        cp.maximum.at(threshold, coarse_q, coarse_d)
    pair_threshold = threshold[d_query_tile]
    error_bound = precision_context.fp32_error_bound
    tie_tolerance = 1.0e-8 + 1.0e-5 * cp.abs(pair_threshold)
    finite_distances = cp.isfinite(d_distances)
    ambiguous = ~finite_distances | (
        finite_distances
        & (d_distances <= pair_threshold + (2.0 * error_bound) + tie_tolerance)
    )
    if max_distance is not None and np.isfinite(max_distance):
        ambiguous |= cp.abs(d_distances - float(max_distance)) <= error_bound

    from vibespatial.api._native_rowset import NativeDeviceSelection

    selection = NativeDeviceSelection.from_mask(
        ambiguous,
        source_row_count=pair_count,
    )
    d_refine_query = selection.gather_capacity(
        d_query_global,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    d_refine_target = selection.gather_capacity(
        d_target,
        fill_value=0,
    ).astype(cp.int32, copy=False)
    refined = _compute_mixed_distances_gpu_device(
        query_owned,
        tree_owned,
        d_refine_query,
        d_refine_target,
        precision_context=precision_context.refinement_context(),
        pair_active=selection.active_capacity_mask(),
        source_positions=selection.partition_capacity_positions(),
        output_distances=d_distances,
        center_device=distance_center_device,
    )
    if refined is None or refined[1]:
        return None
    d_distances = refined[0]
    exact_q, exact_t, exact_d, _ = _topk_per_query(
        d_query_tile,
        d_target,
        d_distances,
        n_tile_queries,
        k,
        max_distance=max_distance,
        grouped_by_query=True,
    )
    return exact_q, exact_t, exact_d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _bounds_are_device(*bounds_arrays: Any) -> bool:
    return any(hasattr(bounds, "__cuda_array_interface__") for bounds in bounds_arrays)


def _effective_unbounded_max_distance(
    query_bounds: Any,
    tree_bounds: Any,
    *,
    bounds_on_device: bool,
) -> float | None:
    """Compute the unbounded kNN search ceiling without raw CuPy syncs."""
    if bounds_on_device:
        d_query_bounds = cp.asarray(query_bounds)
        d_tree_bounds = cp.asarray(tree_bounds)
        d_all_bounds = cp.vstack((d_query_bounds, d_tree_bounds))
        d_valid_mask = ~cp.isnan(d_all_bounds).any(axis=1)
        d_extent = cp.empty(5, dtype=cp.float64)
        d_extent[0] = cp.count_nonzero(d_valid_mask).astype(cp.float64)
        d_extent[1] = cp.min(cp.where(d_valid_mask, d_all_bounds[:, 0], cp.inf))
        d_extent[2] = cp.min(cp.where(d_valid_mask, d_all_bounds[:, 1], cp.inf))
        d_extent[3] = cp.max(cp.where(d_valid_mask, d_all_bounds[:, 2], -cp.inf))
        d_extent[4] = cp.max(cp.where(d_valid_mask, d_all_bounds[:, 3], -cp.inf))
        h_extent = get_cuda_runtime().copy_device_to_host(
            d_extent,
            reason="spatial index kNN unbounded extent scalar fence",
        )
        valid_count, xmin, ymin, xmax, ymax = np.asarray(h_extent, dtype=np.float64)
    else:
        all_bounds = np.vstack((np.asarray(query_bounds), np.asarray(tree_bounds)))
        valid_mask = ~np.isnan(all_bounds).any(axis=1)
        valid_count = float(np.count_nonzero(valid_mask))
        if valid_count == 0:
            return None
        valid_bounds = all_bounds[valid_mask]
        xmin = float(valid_bounds[:, 0].min())
        ymin = float(valid_bounds[:, 1].min())
        xmax = float(valid_bounds[:, 2].max())
        ymax = float(valid_bounds[:, 3].max())

    if int(valid_count) == 0:
        return None
    extent_dx = float(xmax - xmin)
    extent_dy = float(ymax - ymin)
    return float(np.hypot(extent_dx, extent_dy)) * 1.01 + 1.0


def _spatial_index_knn_device_impl(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    query_bounds: Any,
    tree_bounds: Any,
    *,
    native_spatial_index: Any | None = None,
    k: int = 1,
    max_distance: float | None = None,
    exclusive: bool = False,
    return_all: bool = False,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> DeviceKnnResult | None:
    """GPU-accelerated k-nearest-neighbor spatial query.

    Replaces the full-candidate plan with a capacity-bounded device pipeline:
    indexed progressive candidate tiles -> exact/refined distance -> retained
    per-query top-k.

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
        Unsupported by this bounded path; callers must decline before entry.
    return_all : bool
        Unsupported by this bounded path. ``False`` returns at most ``k``
        deterministically ordered rows per query.
    precision : PrecisionMode
        Precision mode for distance computation. Consumer GPUs use the METRIC
        staged plan with selective fp64 ranking refinement.

    Returns
    -------
    DeviceKnnResult or None
        Device-resident result with (query_idx, target_idx, distance)
        triples, or None if the GPU path is not applicable.
    """
    if not has_gpu_runtime() or cp is None:
        return None
    if return_all or exclusive:
        logger.debug(
            "bounded fixed-k nearest declined unsupported semantics: "
            "return_all=%s exclusive=%s",
            return_all,
            exclusive,
        )
        return None
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")

    n_queries = int(query_bounds.shape[0])
    n_tree = int(tree_bounds.shape[0])
    bounds_on_device = _bounds_are_device(query_bounds, tree_bounds)
    if n_queries == 0 or n_tree == 0:
        return _empty_knn_result(k)
    if n_queries > np.iinfo(np.int32).max or n_tree > np.iinfo(np.int32).max:
        raise CandidateRelationCapacityError(
            "bounded fixed-k nearest requires int32 query and target row positions"
        )

    if native_spatial_index is None:
        from vibespatial.spatial.indexing import build_flat_spatial_index

        flat_index = build_flat_spatial_index(tree_owned)
        native_spatial_index = flat_index.to_native_spatial_index()
    else:
        native_spatial_index.validate_row_count(tree_owned.row_count)
        if native_spatial_index.geometry is not tree_owned:
            raise ValueError(
                "NativeSpatialIndex geometry does not match fixed-k target geometry"
            )
        flat_index = native_spatial_index.to_flat_index()
    if int(flat_index.size) != n_tree:
        raise ValueError("fixed-k target bounds and retained spatial index disagree")

    workspace = _plan_knn_workspace(n_queries, n_tree, k)
    telemetry = DeviceKnnTelemetry(
        query_tile_rows=workspace.query_tile_rows,
        target_tile_rows=workspace.target_tile_rows,
        pair_capacity=workspace.pair_capacity,
        admitted_workspace_bytes=workspace.admitted_workspace_bytes,
    )
    # Dispatch check: assess whether GPU is profitable for this workload.
    selection = plan_dispatch_selection(
        kernel_name="spatial_index_knn",
        kernel_class=KernelClass.METRIC,
        row_count=n_queries,
        requested_precision=precision,
        gpu_available=True,
        current_residency=combined_residency(query_owned, tree_owned),
        work_estimate=PhysicalWorkEstimate.for_candidate_pairs(
            row_count=n_queries,
            candidate_pair_count=workspace.pair_capacity,
            primary_unit_name="knn-candidate-pair",
        ),
    )
    if selection.selected is not ExecutionMode.GPU:
        return None

    # Compute effective max_distance for candidate generation.
    if max_distance is not None and np.isfinite(max_distance):
        effective_max_distance = float(max_distance)
    else:
        # Compute the bounding-box diagonal of all valid geometry bounds.  Host
        # bounds stay host-known; device bounds use one named extent fence.
        effective_max_distance = _effective_unbounded_max_distance(
            query_bounds,
            tree_bounds,
            bounds_on_device=bounds_on_device,
        )
        if effective_max_distance is None:
            return _empty_knn_result(k, telemetry)

    d_query_bounds = cp.asarray(query_bounds, dtype=cp.float64)
    d_tree_bounds = cp.asarray(tree_bounds, dtype=cp.float64)
    from .nearest import _plan_device_resident_metric_precision

    precision_context = _plan_device_resident_metric_precision(
        query_owned,
        tree_owned,
        workspace.pair_capacity,
        adaptive_plan=selection,
    )
    distance_center_device = None
    from vibespatial.geometry.buffers import GeometryFamily

    pointset_families = {GeometryFamily.POINT, GeometryFamily.MULTIPOINT}
    if (set(query_owned.families) | set(tree_owned.families)) & pointset_families:
        from vibespatial.spatial.point_distance import compute_distance_center_device

        distance_center_device = compute_distance_center_device(
            query_owned,
            tree_owned,
        )

    d_final_targets = cp.full((n_queries, k), -1, dtype=cp.int32)
    d_final_distances = cp.full((n_queries, k), cp.inf, dtype=cp.float64)
    d_candidate_left_workspace = cp.empty(workspace.pair_capacity, dtype=cp.int32)
    d_candidate_right_workspace = cp.empty(workspace.pair_capacity, dtype=cp.int32)

    def _candidate_output(capacity: int):
        if int(capacity) > workspace.pair_capacity:
            raise CandidateRelationCapacityError(
                "fixed-k candidate output exceeded its admitted workspace: "
                f"capacity={capacity:,}, admitted={workspace.pair_capacity:,}"
            )
        return (
            d_candidate_left_workspace[:capacity],
            d_candidate_right_workspace[:capacity],
        )

    initial_radius = _initial_search_radius(
        flat_index,
        n_tree=n_tree,
        k=k,
        ceiling=effective_max_distance,
    )

    for query_start in range(0, n_queries, workspace.query_tile_rows):
        query_end = min(n_queries, query_start + workspace.query_tile_rows)
        tile_rows = query_end - query_start
        telemetry.query_tiles += 1
        d_tile_bounds = d_query_bounds[query_start:query_end]
        d_valid_query = cp.isfinite(d_tile_bounds).all(axis=1)
        d_unresolved = d_valid_query.copy()
        d_radii = cp.full(tile_rows, initial_radius, dtype=cp.float64)
        d_previous_radii = cp.full(tile_rows, -1.0, dtype=cp.float64)
        if max_distance is not None and np.isfinite(max_distance):
            d_radii.fill(float(max_distance))

        best_targets = cp.full((tile_rows, k), -1, dtype=cp.int32)
        best_distances = cp.full((tile_rows, k), cp.inf, dtype=cp.float64)
        scratch_targets = cp.full_like(best_targets, -1)
        scratch_distances = cp.full_like(best_distances, cp.inf)
        candidate_targets = cp.full_like(best_targets, -1)
        candidate_distances = cp.full_like(best_distances, cp.inf)
        merge_targets = cp.empty((tile_rows, 2 * k), dtype=cp.int32)
        merge_distances = cp.empty((tile_rows, 2 * k), dtype=cp.float64)
        merge_queries = cp.repeat(cp.arange(tile_rows, dtype=cp.int32), 2 * k)

        while True:
            telemetry.scalar_fences += 1
            if not _device_any(
                d_unresolved,
                reason="bounded fixed-k unresolved-row scalar fence",
            ):
                break
            telemetry.radius_iterations += 1
            d_active_positions = cp.flatnonzero(d_unresolved).astype(
                cp.int32,
                copy=False,
            )
            d_active_bounds = d_tile_bounds[d_active_positions]
            d_outer_radii = cp.minimum(
                d_radii[d_active_positions],
                effective_max_distance,
            )
            d_inner_radii = d_previous_radii[d_active_positions]

            for candidates in _candidate_batches(
                flat_index,
                native_spatial_index,
                d_active_bounds,
                d_outer_radii,
                d_tree_bounds,
                workspace=workspace,
                telemetry=telemetry,
                candidate_output=_candidate_output,
            ):
                candidates.validate_error_flag()
                d_active_left = cp.asarray(candidates.d_left, dtype=cp.int32)
                d_target = cp.asarray(candidates.d_right, dtype=cp.int32)
                d_query_tile = d_active_positions[d_active_left]
                d_query_global = d_query_tile + query_start

                # The index returns the full outer square.  Keep only pairs
                # whose bbox first enters in this radius shell so exact
                # distances are never recomputed across expansions.
                d_entry_distance = _bbox_chebyshev_distance(
                    d_query_bounds[d_query_global],
                    d_tree_bounds[d_target],
                )
                d_shell = d_entry_distance > d_inner_radii[d_active_left]
                compacted = compact_indices(d_shell.astype(cp.uint8))
                shell_count = compacted.count
                if shell_count == 0:
                    continue
                kept = compacted.values
                d_query_tile = d_query_tile[kept]
                d_query_global = d_query_global[kept]
                d_target = d_target[kept]
                telemetry.candidate_pairs += shell_count
                telemetry.max_candidate_pairs = max(
                    telemetry.max_candidate_pairs,
                    shell_count,
                )

                chunk_topk = _refined_candidate_topk(
                    query_owned,
                    tree_owned,
                    d_query_global,
                    d_query_tile,
                    d_target,
                    n_tile_queries=tile_rows,
                    k=k,
                    max_distance=(
                        float(max_distance)
                        if max_distance is not None and np.isfinite(max_distance)
                        else None
                    ),
                    precision_context=precision_context,
                    distance_center_device=distance_center_device,
                )
                if chunk_topk is None:
                    return None
                chunk_q, chunk_t, chunk_d = chunk_topk
                previous_best_targets = best_targets
                previous_best_distances = best_distances
                best_targets, best_distances = _merge_topk_dense(
                    best_targets,
                    best_distances,
                    chunk_q,
                    chunk_t,
                    chunk_d,
                    n_queries=tile_rows,
                    k=k,
                    merge_queries=merge_queries,
                    merge_targets=merge_targets,
                    merge_distances=merge_distances,
                    candidate_targets=candidate_targets,
                    candidate_distances=candidate_distances,
                    scratch_targets=scratch_targets,
                    scratch_distances=scratch_distances,
                )
                scratch_targets = previous_best_targets
                scratch_distances = previous_best_distances

            if max_distance is not None and np.isfinite(max_distance):
                d_unresolved.fill(False)
                continue

            valid_counts = cp.sum(best_targets >= 0, axis=1)
            kth_covered = (valid_counts >= k) & (
                best_distances[:, k - 1] <= d_radii
            )
            at_ceiling = d_radii >= effective_max_distance
            d_finished = d_unresolved & (kth_covered | at_ceiling)
            d_unresolved &= ~d_finished
            d_previous_radii = cp.where(
                d_unresolved,
                d_radii,
                d_previous_radii,
            )
            d_radii = cp.where(
                d_unresolved,
                cp.minimum(d_radii * 2.0, effective_max_distance),
                d_radii,
            )

        d_final_targets[query_start:query_end] = best_targets
        d_final_distances[query_start:query_end] = best_distances

    d_output_valid = (d_final_targets >= 0) & cp.isfinite(d_final_distances)
    d_query_slots = cp.broadcast_to(
        cp.arange(n_queries, dtype=cp.int32)[:, None],
        (n_queries, k),
    )
    d_out_query = d_query_slots[d_output_valid]
    d_out_target = d_final_targets[d_output_valid]
    d_out_dist = d_final_distances[d_output_valid]
    total_pairs = int(d_out_query.size)

    return DeviceKnnResult(
        d_query_idx=d_out_query,
        d_target_idx=d_out_target,
        d_distances=d_out_dist,
        total_pairs=total_pairs,
        k=k,
        telemetry=telemetry,
    )


def spatial_index_knn_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    query_bounds: Any,
    tree_bounds: Any,
    *,
    native_spatial_index: Any | None = None,
    k: int = 1,
    max_distance: float | None = None,
    exclusive: bool = False,
    return_all: bool = False,
    precision: PrecisionMode | str = PrecisionMode.AUTO,
) -> DeviceKnnResult | None:
    """Run fixed-k nearest with operation-local allocation telemetry."""
    from vibespatial.cuda._runtime import get_d2h_transfer_stats
    from vibespatial.runtime.materialization import get_materialization_events

    monitor = _OperationAllocationMonitor()
    start_d2h_count, start_d2h_bytes = get_d2h_transfer_stats()
    start_materializations = len(get_materialization_events())
    result = None
    try:
        result = _spatial_index_knn_device_impl(
            query_owned,
            tree_owned,
            query_bounds,
            tree_bounds,
            native_spatial_index=native_spatial_index,
            k=k,
            max_distance=max_distance,
            exclusive=exclusive,
            return_all=return_all,
            precision=precision,
        )
    finally:
        peak_device_bytes, allocation_count = monitor.finish()
        if result is not None and result.telemetry is not None:
            end_d2h_count, end_d2h_bytes = get_d2h_transfer_stats()
            result.telemetry.peak_device_bytes = peak_device_bytes
            result.telemetry.allocation_count = allocation_count
            result.telemetry.d2h_count = max(
                end_d2h_count - start_d2h_count,
                0,
            )
            result.telemetry.d2h_bytes = max(
                end_d2h_bytes - start_d2h_bytes,
                0,
            )
            result.telemetry.materialization_count = max(
                len(get_materialization_events()) - start_materializations,
                0,
            )

    if result is not None and result.telemetry is not None:
        record_dispatch_event(
            surface="vibespatial.spatial.spatial_index_knn_device",
            operation="bounded_fixed_k_nearest",
            implementation="owned_gpu_bounded_knn",
            reason=(
                "capacity-admitted progressive NativeSpatialIndex candidate/refine "
                "completed"
            ),
            detail=result.telemetry.detail(),
            selected=ExecutionMode.GPU,
        )
    return result
