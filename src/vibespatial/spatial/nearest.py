from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import shapely

from vibespatial.cuda.cccl_precompile import request_warmup
from vibespatial.cuda.cccl_primitives import (
    compact_indices,
    exclusive_sum,
    lower_bound,
    segmented_reduce_min,
    sort_pairs,
    upper_bound,
)
from vibespatial.runtime.adaptive import plan_kernel_dispatch
from vibespatial.runtime.crossover import DispatchDecision

request_warmup([
    "exclusive_scan_i32", "exclusive_scan_i64",
    "select_i32", "select_i64",
    "radix_sort_i32_i32", "radix_sort_u64_i32",
    "lower_bound_i32", "lower_bound_u64",
    "upper_bound_i32", "upper_bound_u64",
    "segmented_reduce_min_f64",
])
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
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds  # noqa: E402
from vibespatial.kernels.core.spatial_query_kernels import (  # noqa: E402
    _grid_nearest_kernels,
    _spatial_query_kernels,
)
from vibespatial.runtime import has_gpu_runtime  # noqa: E402
from vibespatial.runtime.precision import KernelClass  # noqa: E402
from vibespatial.runtime.residency import Residency, TransferTrigger  # noqa: E402

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

def _empty_nearest_result(return_distance: bool):
    """Return a canonical empty nearest result."""
    empty = np.empty((2, 0), dtype=np.intp)
    if return_distance:
        return empty, np.asarray([], dtype=np.float64)
    return empty


def _points_only(owned: OwnedGeometryArray) -> bool:
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
    non_empty = ~points.empty_mask[family_rows]
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
    if not np.array_equal(point_buffer.geometry_offsets, np.arange(owned.row_count + 1, dtype=np.int32)):
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

    expected_x = float(unique_x[0]) + (np.arange(owned.row_count, dtype=np.float64) % cols) * cell_width
    expected_y = float(unique_y[0]) + (np.arange(owned.row_count, dtype=np.float64) // cols) * cell_height
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

    qs = query_owned._ensure_device_state()
    ts = tree_owned._ensure_device_state()
    qp = qs.families[point_family]
    tp = ts.families[point_family]

    dist_params = (
        (
            ptr(qs.validity), ptr(qs.tags), ptr(qs.family_row_offsets),
            ptr(qp.geometry_offsets), ptr(qp.empty_mask),
            ptr(qp.x), ptr(qp.y),
            FAMILY_TAGS[point_family],
            ptr(ts.validity), ptr(ts.tags), ptr(ts.family_row_offsets),
            ptr(tp.geometry_offsets), ptr(tp.empty_mask),
            ptr(tp.x), ptr(tp.y),
            FAMILY_TAGS[point_family],
            ptr(d_left), ptr(d_right),
            ptr(d_distances),
            1 if exclusive else 0,
            pair_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR,
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,
        ),
    )
    dist_grid, dist_block = runtime.launch_config(kernels["point_point_distance_pairs_from_owned"], pair_count)
    runtime.launch(
        kernels["point_point_distance_pairs_from_owned"],
        grid=dist_grid, block=dist_block, params=dist_params,
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
) -> tuple[Any, None]:
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
    query_keys = cp.arange(n_queries, dtype=cp.int32)
    seg_starts = lower_bound(d_left, query_keys, synchronize=False)
    seg_ends = upper_bound(d_left, query_keys, synchronize=False)
    seg_starts_i32 = seg_starts.astype(cp.int32, copy=False)
    seg_ends_i32 = seg_ends.astype(cp.int32, copy=False)

    # Segmented min-distance per query (Tier 3a CCCL).
    min_result = segmented_reduce_min(
        d_distances, seg_starts_i32, seg_ends_i32, num_segments=n_queries,
    )
    d_min_distances = min_result.values

    # Build keep mask (Tier 1 NVRTC).
    d_keep = runtime.allocate((pair_count,), np.uint8)
    keep_params = (
        (
            ptr(d_distances), ptr(d_min_distances),
            ptr(d_left), ptr(d_keep),
            float(max_distance),
            pair_count,
        ),
        (
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_I32,
        ),
    )
    keep_grid, keep_block = runtime.launch_config(kernels["nearest_keep_mask"], pair_count)
    runtime.launch(
        kernels["nearest_keep_mask"],
        grid=keep_grid, block=keep_block, params=keep_params,
    )

    # (return_all=False) Keep only first match per segment.
    if not return_all:
        d_first = runtime.from_host(np.zeros(pair_count, dtype=np.uint8))
        seg_grid = max(1, (n_queries + 255) // 256)
        first_params = (
            (
                ptr(d_keep), ptr(d_first),
                ptr(seg_starts_i32), ptr(seg_ends_i32),
                n_queries,
            ),
            (
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        seg_grid, seg_block = runtime.launch_config(kernels["nearest_first_per_segment"], n_queries)
        runtime.launch(
            kernels["nearest_first_per_segment"],
            grid=seg_grid, block=seg_block, params=first_params,
        )
        d_keep = d_first

    # Compact kept indices (Tier 3a CCCL).
    compacted = compact_indices(d_keep)
    if compacted.count == 0:
        return _empty_nearest_result(return_distance), None

    # Gather results on device, copy to host.
    kept_idx = compacted.values
    h_left = runtime.copy_device_to_host(d_left[kept_idx]).astype(np.intp, copy=False)
    h_right = runtime.copy_device_to_host(d_right[kept_idx]).astype(np.intp, copy=False)
    indices = np.vstack((h_left, h_right))

    if return_distance:
        h_dist = runtime.copy_device_to_host(d_distances[kept_idx])
        return (indices, h_dist), None
    return indices, None


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
    ) -> bool:
        self.move_to_device(
            query_owned, tree_owned,
            query_reason="nearest: GPU point-point distance for query",
            tree_reason="nearest: GPU point-point distance for tree",
        )
        _launch_point_point_distance_kernel(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
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
    ) -> bool:
        from .point_distance import compute_point_distance_gpu

        self.move_to_device(
            query_owned, tree_owned,
            query_reason="nearest: GPU point-distance refinement for query points",
            tree_reason=f"nearest: GPU point-distance refinement for tree {self.tree_family.name}",
        )
        return compute_point_distance_gpu(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
            tree_family=self.tree_family,
            exclusive=exclusive,
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
    ) -> bool:
        from .segment_distance import compute_segment_distance_gpu

        self.move_to_device(
            query_owned, tree_owned,
            query_reason=f"nearest: GPU segment-distance refinement for query {self.query_family.name}",
            tree_reason=f"nearest: GPU segment-distance refinement for tree {self.tree_family.name}",
        )
        return compute_segment_distance_gpu(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
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
) -> tuple[np.ndarray, np.ndarray | None] | None:
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

        # Compute distances using the strategy.
        ok = strategy.compute(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
            exclusive=exclusive,
        )
        if not ok:
            return None

        # Run shared refinement pipeline.
        return _refine_nearest_from_device_distances(
            d_left, d_right, d_distances, pair_count, n_queries,
            max_distance=max_distance,
            return_all=return_all,
            return_distance=return_distance,
        )
    finally:
        runtime.free(d_left)
        runtime.free(d_right)
        runtime.free(d_distances)


# Backward-compatible wrappers for the two old functions.

def _nearest_refine_gpu_point_distance(
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
    tree_family: GeometryFamily,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """GPU nearest refinement for point queries against non-point trees.

    Uses point-to-segment/polygon distance kernels from ``point_distance``.
    Returns ``(indices_2xN, distances_or_None)`` on success, or ``None``.
    """
    strategy = PointFamilyDistanceStrategy(tree_family)
    return _nearest_refine_gpu_typed(
        query_owned, tree_owned, left_idx, right_idx, n_queries,
        strategy,
        max_distance=max_distance, return_all=return_all,
        exclusive=exclusive, return_distance=return_distance,
    )


def _nearest_refine_gpu_segment_distance(
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
    query_family: GeometryFamily,
    tree_family: GeometryFamily,
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """GPU nearest refinement for non-point geometry pairs.

    Uses segment-to-segment distance kernels from ``segment_distance``.
    Returns ``(indices_2xN, distances_or_None)`` on success, or ``None``.
    """
    strategy = SegmentFamilyDistanceStrategy(query_family, tree_family)
    return _nearest_refine_gpu_typed(
        query_owned, tree_owned, left_idx, right_idx, n_queries,
        strategy,
        max_distance=max_distance, return_all=return_all,
        exclusive=exclusive, return_distance=return_distance,
    )


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
        count_grid, count_block = runtime.launch_config(kernels["point_regular_grid_nearest_count"], query_owned.row_count)
        runtime.launch(
            kernels["point_regular_grid_nearest_count"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = count_scatter_total(runtime, counts, offsets) if query_owned.row_count > 0 else 0
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
        scatter_grid, scatter_block = runtime.launch_config(kernels["point_regular_grid_nearest_scatter"], query_owned.row_count)
        runtime.launch(
            kernels["point_regular_grid_nearest_scatter"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        runtime.synchronize()

        left = runtime.copy_device_to_host(out_left).astype(np.int32, copy=False)
        right = runtime.copy_device_to_host(out_right).astype(np.int32, copy=False)
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
    extent = max(extent_x, extent_y, 1e-12)

    cell_size = extent / max(1.0, math.ceil(math.sqrt(n_tree)))
    # Floor: cap grid at 4096 x 4096 = ~16M cells
    cell_size = max(cell_size, extent / 4096.0)
    # Ensure cell_size is positive
    cell_size = max(cell_size, 1e-12)

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
            grid=grid_a, block=block_a,
            params=(
                (ptr(d_tree_x), ptr(d_tree_y),
                 origin_x, origin_y, cell_size,
                 n_cols, n_rows, ptr(d_cell_ids), n_tree),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
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
            grid=grid_r, block=block_r,
            params=(
                (ptr(d_sorted_cell_ids), ptr(d_cell_start), ptr(d_cell_end),
                 n_cells, n_tree),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32),
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
            grid=grid_s, block=block_s,
            params=(
                (ptr(d_query_x), ptr(d_query_y),
                 ptr(d_sorted_tree_x), ptr(d_sorted_tree_y),
                 ptr(d_sorted_global_idx),
                 ptr(d_cell_start), ptr(d_cell_end),
                 n_cols, n_rows,
                 origin_x, origin_y, cell_size,
                 n_tree, 1 if exclusive else 0,
                 ptr(d_min_sq), ptr(d_min_idx),
                 n_query),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32,
                 KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

        # --- Post-processing ---
        if not return_all:
            # return_all=False: just use min_idx directly (one per query).
            runtime.synchronize()
            h_min_sq = runtime.copy_device_to_host(d_min_sq)
            h_min_idx = runtime.copy_device_to_host(d_min_idx)

            # Filter: finite distances only, and max_distance
            valid_mask = np.isfinite(h_min_sq)
            if max_distance is not None:
                valid_mask &= np.sqrt(h_min_sq) <= max_distance
            valid_q = np.flatnonzero(valid_mask)
            if valid_q.size == 0:
                result = _empty_nearest_result(return_distance)
                return result, "owned_gpu_nearest"

            left = query_global_idx[valid_q].astype(np.intp)
            right = h_min_idx[valid_q].astype(np.intp)
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
            grid=grid_tc, block=block_tc,
            params=(
                (ptr(d_query_x), ptr(d_query_y),
                 ptr(d_sorted_tree_x), ptr(d_sorted_tree_y),
                 ptr(d_cell_start), ptr(d_cell_end),
                 n_cols, n_rows,
                 origin_x, origin_y, cell_size,
                 ptr(d_min_sq),
                 1 if exclusive else 0,
                 ptr(d_counts),
                 n_query),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32,
                 KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )

        d_offsets = exclusive_sum(d_counts)
        total_pairs = count_scatter_total(runtime, d_counts, d_offsets) if n_query > 0 else 0
        if total_pairs == 0:
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"

        d_out_left = runtime.allocate((total_pairs,), np.int32)
        d_out_right = runtime.allocate((total_pairs,), np.int32)
        grid_ts, block_ts = runtime.launch_config(kernels["grid_nearest_tie_scatter"], n_query)
        runtime.launch(
            kernels["grid_nearest_tie_scatter"],
            grid=grid_ts, block=block_ts,
            params=(
                (ptr(d_query_x), ptr(d_query_y),
                 ptr(d_sorted_tree_x), ptr(d_sorted_tree_y),
                 ptr(d_sorted_global_idx),
                 ptr(d_cell_start), ptr(d_cell_end),
                 n_cols, n_rows,
                 origin_x, origin_y, cell_size,
                 ptr(d_min_sq),
                 1 if exclusive else 0,
                 ptr(d_offsets),
                 ptr(d_out_left), ptr(d_out_right),
                 n_query),
                (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32, KERNEL_PARAM_I32,
                 KERNEL_PARAM_F64, KERNEL_PARAM_F64, KERNEL_PARAM_F64,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32,
                 KERNEL_PARAM_PTR,
                 KERNEL_PARAM_PTR, KERNEL_PARAM_PTR,
                 KERNEL_PARAM_I32),
            ),
        )
        runtime.synchronize()

        # D2H transfer of final indices
        h_left_dense = runtime.copy_device_to_host(d_out_left).astype(np.intp)
        h_right = runtime.copy_device_to_host(d_out_right).astype(np.intp)

        # Compute per-pair distances from min_sq (one D2H for distance data)
        h_min_sq_all = runtime.copy_device_to_host(d_min_sq)
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
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"

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
) -> tuple[tuple[np.ndarray, np.ndarray] | np.ndarray, str] | None:
    if not has_gpu_runtime():
        return None
    if not _points_only(query_owned) or not _points_only(tree_owned):
        return None
    if GeometryFamily.POINT not in query_owned.families or GeometryFamily.POINT not in tree_owned.families:
        return None

    plan = plan_kernel_dispatch(
        kernel_name="nearest_knn_indexed",
        kernel_class=KernelClass.COARSE,
        row_count=query_owned.row_count + tree_owned.row_count,
        gpu_available=True,
    )
    dispatch = plan.dispatch_decision
    if dispatch is not DispatchDecision.GPU:
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
    valid_tree_rows = cp.flatnonzero(cp.isfinite(tree_x) & cp.isfinite(tree_y)).astype(cp.int32, copy=False)
    if int(valid_tree_rows.size) == 0:
        result = _empty_nearest_result(return_distance)
        return result, "owned_gpu_nearest"

    sorted_tree = sort_pairs(tree_x[valid_tree_rows], valid_tree_rows, synchronize=False)
    query_probe_x = cp.nan_to_num(query_x, nan=0.0)
    insert_idx = lower_bound(sorted_tree.keys, query_probe_x, synchronize=False).astype(cp.int32, copy=False)
    min_sq = cp.empty(query_owned.row_count, dtype=cp.float64)
    counts = cp.empty(query_owned.row_count, dtype=cp.int32)
    offsets = None
    out_left = None
    out_right = None
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
        min_grid, min_block = runtime.launch_config(kernels["point_nearest_min_sq_from_sorted_x"], query_owned.row_count)
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
        start_idx = lower_bound(sorted_tree.keys, query_min_x, synchronize=False).astype(cp.int32, copy=False)
        end_idx = upper_bound(sorted_tree.keys, query_max_x, synchronize=False).astype(cp.int32, copy=False)

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
        count_grid, count_block = runtime.launch_config(kernels["point_nearest_tie_count_from_sorted_x"], query_owned.row_count)
        runtime.launch(
            kernels["point_nearest_tie_count_from_sorted_x"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = count_scatter_total(runtime, counts, offsets) if query_owned.row_count > 0 else 0
        if total_pairs == 0:
            result = _empty_nearest_result(return_distance)
            return result, "owned_gpu_nearest"

        out_left = runtime.allocate((total_pairs,), np.int32)
        out_right = runtime.allocate((total_pairs,), np.int32)
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
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        scatter_grid, scatter_block = runtime.launch_config(kernels["point_nearest_tie_scatter_from_sorted_x"], query_owned.row_count)
        runtime.launch(
            kernels["point_nearest_tie_scatter_from_sorted_x"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        runtime.synchronize()

        left = runtime.copy_device_to_host(out_left).astype(np.intp, copy=False)
        right = runtime.copy_device_to_host(out_right).astype(np.intp, copy=False)
        best_host = runtime.copy_device_to_host(best)

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
        runtime.free(out_left)
        runtime.free(out_right)


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
    if GeometryFamily.POINT not in query_owned.families or GeometryFamily.POINT not in tree_owned.families:
        return None

    plan = plan_kernel_dispatch(
        kernel_name="point_nearest_candidates",
        kernel_class=KernelClass.METRIC,
        row_count=query_owned.row_count * tree_owned.row_count,
        gpu_available=True,
    )
    dispatch = plan.dispatch_decision
    if dispatch is not DispatchDecision.GPU:
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
    valid_tree_rows = cp.flatnonzero(cp.isfinite(tree_x) & cp.isfinite(tree_y)).astype(cp.int32, copy=False)
    if int(valid_tree_rows.size) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    sorted_tree = sort_pairs(tree_x[valid_tree_rows], valid_tree_rows, synchronize=False)
    query_min_x = cp.nan_to_num(query_x - max_distance, nan=0.0)
    query_max_x = cp.nan_to_num(query_x + max_distance, nan=0.0)
    start_idx = lower_bound(sorted_tree.keys, query_min_x, synchronize=False).astype(cp.int32, copy=False)
    end_idx = upper_bound(sorted_tree.keys, query_max_x, synchronize=False).astype(cp.int32, copy=False)
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
        count_grid, count_block = runtime.launch_config(kernels["point_x_window_count"], query_owned.row_count)
        runtime.launch(
            kernels["point_x_window_count"],
            grid=count_grid,
            block=count_block,
            params=count_params,
        )

        offsets = exclusive_sum(counts)
        total_pairs = count_scatter_total(runtime, counts, offsets) if query_owned.row_count > 0 else 0
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
        scatter_grid, scatter_block = runtime.launch_config(kernels["point_x_window_scatter"], query_owned.row_count)
        runtime.launch(
            kernels["point_x_window_scatter"],
            grid=scatter_grid,
            block=scatter_block,
            params=scatter_params,
        )
        runtime.synchronize()

        left = runtime.copy_device_to_host(out_left).astype(np.int32, copy=False)
        right = runtime.copy_device_to_host(out_right).astype(np.int32, copy=False)
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
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
        )
        return True

    if query_family == point_family:
        from .point_distance import compute_point_distance_gpu
        return compute_point_distance_gpu(
            query_owned, tree_owned,
            d_left, d_right, d_distances, pair_count,
            tree_family=tree_family,
        )

    if tree_family == point_family:
        from .point_distance import compute_point_distance_gpu
        return compute_point_distance_gpu(
            tree_owned, query_owned,
            d_right, d_left, d_distances, pair_count,
            tree_family=query_family,
        )

    # Non-point x non-point
    from .segment_distance import compute_segment_distance_gpu
    return compute_segment_distance_gpu(
        query_owned, tree_owned,
        d_left, d_right, d_distances, pair_count,
        query_family=query_family, tree_family=tree_family,
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
        expanded_point_idx[cursor:cursor + n] = np.arange(cs, cs + n, dtype=np.int32)
        expanded_target_idx[cursor:cursor + n] = target_idx[i]
        cursor += n

    # Create a temporary point OwnedGeometryArray from the MP's coord arrays.
    temp_point_owned = _make_point_owned_from_coords(mp_buffer.x, mp_buffer.y)

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
                temp_point_owned, target_owned,
                d_exp_left, d_exp_right, d_exp_dist, total_expanded,
                exclusive=exclusive,
            )
            ok = True
        else:
            # Point x {Line, Polygon, ...} via existing point distance kernels.
            from .point_distance import compute_point_distance_gpu
            ok = compute_point_distance_gpu(
                temp_point_owned, target_owned,
                d_exp_left, d_exp_right, d_exp_dist, total_expanded,
                tree_family=target_family, exclusive=exclusive,
            )

        if not ok:
            return None

        # Segmented min reduction on device (CCCL Tier 3a) — avoids
        # downloading the full expanded distance array to host.
        if pair_count > 256:
            d_starts = runtime.from_host(seg_starts_arr.astype(np.int32))
            d_ends = runtime.from_host(seg_ends_arr.astype(np.int32))
            seg_result = segmented_reduce_min(
                d_exp_dist, d_starts, d_ends, num_segments=pair_count,
            )
            result = runtime.copy_device_to_host(seg_result.values)
        else:
            exp_distances = runtime.copy_device_to_host(d_exp_dist)
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
) -> np.ndarray | None:
    """Compute distances for candidate pairs with mixed geometry families.

    Groups pairs by (left_tag, right_tag) and dispatches to the appropriate
    distance kernel for each group.  Multipoint pairs (MP x non-MP) are handled
    via coord expansion + point distance kernels + segmented min.  Falls back
    to Shapely only for MP x MP (requires double expansion).

    When *device_candidates* is a :class:`_DeviceCandidates`, sub-arrays
    are extracted on-device via CuPy fancy indexing to avoid redundant
    host->device transfers.

    Returns host float64 array of distances, or None if GPU runtime is
    unavailable.
    """
    pair_count = left_idx.size
    if pair_count == 0:
        return np.empty(0, dtype=np.float64)

    left_tags = query_owned.tags[left_idx]
    right_tags = tree_owned.tags[right_idx]

    runtime = get_cuda_runtime()
    distances = np.full(pair_count, np.inf, dtype=np.float64)

    _dc = device_candidates
    _use_device_idx = _dc is not None and hasattr(_dc, "d_left")

    point_family = GeometryFamily.POINT

    for (lt, rt) in unique_tag_pairs(left_tags, right_tags):
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
                    query_owned, tree_owned,
                    d_sub_left, d_sub_right, d_sub_dist, sub_count,
                    exclusive=exclusive,
                )
                sub_distances = np.empty(sub_count, dtype=np.float64)
                runtime.copy_device_to_host(d_sub_dist, sub_distances)
                distances[sub_idx] = sub_distances
                ok = True
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)
        elif lf == point_family or rf == point_family:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .point_distance import compute_point_distance_gpu
                if lf == point_family:
                    ok = compute_point_distance_gpu(
                        query_owned, tree_owned,
                        d_sub_left, d_sub_right, d_sub_dist, sub_count,
                        tree_family=rf, exclusive=exclusive,
                    )
                else:
                    ok = compute_point_distance_gpu(
                        tree_owned, query_owned,
                        d_sub_right, d_sub_left, d_sub_dist, sub_count,
                        tree_family=lf, exclusive=exclusive,
                    )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(d_sub_dist, sub_distances)
                    distances[sub_idx] = sub_distances
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
                    query_owned, tree_owned, sub_left, sub_right,
                    target_family=rf, exclusive=exclusive,
                )
            else:
                mp_result = _compute_multipoint_distances_gpu(
                    tree_owned, query_owned, sub_right, sub_left,
                    target_family=lf, exclusive=exclusive,
                )
            if mp_result is not None:
                distances[sub_idx] = mp_result
                ok = True
        else:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .segment_distance import compute_segment_distance_gpu
                ok = compute_segment_distance_gpu(
                    query_owned, tree_owned,
                    d_sub_left, d_sub_right, d_sub_dist, sub_count,
                    query_family=lf, tree_family=rf, exclusive=exclusive,
                )
                if ok:
                    sub_distances = np.empty(sub_count, dtype=np.float64)
                    runtime.copy_device_to_host(d_sub_dist, sub_distances)
                    distances[sub_idx] = sub_distances
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)

        # For unsupported family pairs, fall back to Shapely distance.
        if not ok:
            query_shapely = np.asarray(query_owned.to_shapely(), dtype=object)
            tree_shapely = np.asarray(tree_owned.to_shapely(), dtype=object)
            sub_dists = shapely.distance(query_shapely[sub_left], tree_shapely[sub_right])
            distances[sub_idx] = np.asarray(sub_dists, dtype=np.float64)
            if exclusive:
                eq = np.asarray(shapely.equals(query_shapely[sub_left], tree_shapely[sub_right]), dtype=bool)
                distances[sub_idx[eq]] = np.inf

    return distances


# ---------------------------------------------------------------------------
# dwithin refinement
# ---------------------------------------------------------------------------

def _dwithin_refine_gpu(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    per_row_distance: np.ndarray,
    device_candidates: _DeviceCandidates | None = None,
    *,
    return_device: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """GPU dwithin refinement: distance <= threshold filter.

    Device-side pipeline (ADR-0033 Tier 2 CuPy for threshold + compact):
      1. Compute distances on device via mixed-distance kernels (Tier 1)
      2. Build per-pair thresholds on device (Tier 2 CuPy)
      3. Apply distance <= threshold filter on device (Tier 2 CuPy)
      4. Compact surviving indices on device (Tier 2 CuPy flatnonzero)
      5. Single D->H transfer of filtered index arrays (unless return_device)

    When *return_device* is True, the surviving index arrays are returned as
    CuPy device arrays, avoiding a D->H round-trip when the caller will
    immediately re-upload them (e.g. the ``return_device`` path in
    ``query_spatial_index``).

    Returns ``(left_idx, right_idx)`` on success, or ``None``
    when the GPU runtime is unavailable.
    """
    if not has_gpu_runtime():
        return None

    pair_count = left_idx.size
    if pair_count == 0:
        return left_idx, right_idx

    try:
        import cupy as cp
    except ImportError:
        return None

    # --- Device-side distance computation ---
    # Accumulate distances in a device CuPy array instead of host numpy.
    d_distances = _compute_mixed_distances_gpu_device(
        query_owned, tree_owned, left_idx, right_idx,
        device_candidates=device_candidates,
    )
    if d_distances is None:
        # Fall back to host-side path.
        distances = _compute_mixed_distances_gpu(
            query_owned, tree_owned, left_idx, right_idx, exclusive=False,
            device_candidates=device_candidates,
        )
        if distances is None:
            return None
        thresholds = per_row_distance[left_idx]
        keep = distances <= thresholds
        return left_idx[keep], right_idx[keep]

    # --- Device-side threshold filter (Tier 2 CuPy) ---
    d_thresholds = cp.asarray(per_row_distance[left_idx])
    d_keep = d_distances <= d_thresholds

    # --- Device-side compaction (Tier 2 CuPy flatnonzero) ---
    d_keep_idx = cp.flatnonzero(d_keep)

    if d_keep_idx.size == 0:
        empty = np.empty(0, dtype=left_idx.dtype)
        if return_device:
            return cp.asarray(empty, dtype=cp.int32), cp.asarray(empty, dtype=cp.int32)
        return empty, empty

    # Gather surviving indices on device.
    if device_candidates is not None and hasattr(device_candidates, "d_left"):
        # Indices are already on device -- gather directly.
        d_left_result = device_candidates.d_left[d_keep_idx]
        d_right_result = device_candidates.d_right[d_keep_idx]
    else:
        d_left_all = cp.asarray(left_idx)
        d_right_all = cp.asarray(right_idx)
        d_left_result = d_left_all[d_keep_idx]
        d_right_result = d_right_all[d_keep_idx]

    if return_device:
        # Keep results on device -- caller will consume them directly.
        return d_left_result.astype(cp.int32, copy=False), d_right_result.astype(cp.int32, copy=False)

    # --- Single D->H transfer of filtered results ---
    return cp.asnumpy(d_left_result), cp.asnumpy(d_right_result)


def _compute_mixed_distances_gpu_device(
    query_owned: OwnedGeometryArray,
    tree_owned: OwnedGeometryArray,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    device_candidates: object | None = None,
):
    """Compute distances on device, accumulating into a CuPy device array.

    Same dispatch logic as _compute_mixed_distances_gpu but keeps all
    distance results device-resident.  Returns a CuPy float64 array of
    distances, or None if the device pipeline cannot handle all groups.
    """
    try:
        import cupy as cp
    except ImportError:
        return None

    pair_count = left_idx.size
    if pair_count == 0:
        return cp.empty(0, dtype=cp.float64)

    left_tags = query_owned.tags[left_idx]
    right_tags = tree_owned.tags[right_idx]

    runtime = get_cuda_runtime()
    d_distances = cp.full(pair_count, cp.inf, dtype=cp.float64)

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

        # Device sub-index for scatter into d_distances.
        d_sub_idx = cp.asarray(sub_idx.astype(np.int32))

        # Build device sub-arrays for kernel dispatch.
        _own_sub_device = True
        if _use_device_idx:
            d_sub_left = _dc.d_left[d_sub_idx]
            d_sub_right = _dc.d_right[d_sub_idx]
            _own_sub_device = False
        else:
            d_sub_left = runtime.from_host(np.ascontiguousarray(sub_left, dtype=np.int32))
            d_sub_right = runtime.from_host(np.ascontiguousarray(sub_right, dtype=np.int32))

        ok = False
        if lf is None or rf is None:
            pass
        elif lf == point_family and rf == point_family:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                query_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="dwithin device: point-point distance",
                )
                tree_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="dwithin device: point-point distance",
                )
                _launch_point_point_distance_kernel(
                    query_owned, tree_owned,
                    d_sub_left, d_sub_right, d_sub_dist, sub_count,
                )
                # Scatter into device result array (Tier 2 CuPy).
                d_distances[d_sub_idx] = cp.asarray(d_sub_dist)
                ok = True
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)
        elif lf == point_family or rf == point_family:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .point_distance import compute_point_distance_gpu
                if lf == point_family:
                    ok = compute_point_distance_gpu(
                        query_owned, tree_owned,
                        d_sub_left, d_sub_right, d_sub_dist, sub_count,
                        tree_family=rf,
                    )
                else:
                    ok = compute_point_distance_gpu(
                        tree_owned, query_owned,
                        d_sub_right, d_sub_left, d_sub_dist, sub_count,
                        tree_family=lf,
                    )
                if ok:
                    d_distances[d_sub_idx] = cp.asarray(d_sub_dist)
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)
        elif (lf == GeometryFamily.MULTIPOINT) != (rf == GeometryFamily.MULTIPOINT):
            if _own_sub_device:
                runtime.free(d_sub_left)
                runtime.free(d_sub_right)
            _own_sub_device = False
            if lf == GeometryFamily.MULTIPOINT:
                mp_result = _compute_multipoint_distances_gpu(
                    query_owned, tree_owned, sub_left, sub_right,
                    target_family=rf,
                )
            else:
                mp_result = _compute_multipoint_distances_gpu(
                    tree_owned, query_owned, sub_right, sub_left,
                    target_family=lf,
                )
            if mp_result is not None:
                d_distances[d_sub_idx] = cp.asarray(mp_result)
                ok = True
        else:
            d_sub_dist = runtime.allocate((sub_count,), np.float64)
            try:
                from .segment_distance import compute_segment_distance_gpu
                ok = compute_segment_distance_gpu(
                    query_owned, tree_owned,
                    d_sub_left, d_sub_right, d_sub_dist, sub_count,
                    query_family=lf, tree_family=rf,
                )
                if ok:
                    d_distances[d_sub_idx] = cp.asarray(d_sub_dist)
            finally:
                if _own_sub_device:
                    runtime.free(d_sub_left)
                    runtime.free(d_sub_right)
                runtime.free(d_sub_dist)

        if not ok:
            # Unsupported family pair — fall back to Shapely for this group,
            # then upload the sub-group result to device.
            # to_shapely() is cached on OwnedGeometryArray, so repeated
            # calls for multiple unsupported groups are cheap.
            if _own_sub_device:
                runtime.free(d_sub_left)
                runtime.free(d_sub_right)
                _own_sub_device = False
            query_shapely = np.asarray(query_owned.to_shapely(), dtype=object)
            tree_shapely = np.asarray(tree_owned.to_shapely(), dtype=object)
            sub_dists = shapely.distance(query_shapely[sub_left], tree_shapely[sub_right])
            d_distances[d_sub_idx] = cp.asarray(
                np.asarray(sub_dists, dtype=np.float64),
            )

    return d_distances


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
    indices = np.vstack((
        left_idx[keep].astype(np.intp, copy=False),
        right_idx[keep].astype(np.intp, copy=False),
    ))
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
) -> tuple[np.ndarray, np.ndarray | None] | None:
    """Full GPU nearest refinement pipeline.

    Handles point-point distance, point-to-segment/polygon distance,
    segment-to-segment distance, and mixed-family arrays.
    Returns ``(indices_2xN, distances_or_None)`` on success, or ``None``
    when the GPU path is not applicable.
    """
    # --- Try single-family fast paths first ---
    use_mixed = False

    if _points_only(query_owned) and _points_only(tree_owned):
        if GeometryFamily.POINT not in query_owned.families or GeometryFamily.POINT not in tree_owned.families:
            # All rows are null/empty -- return empty result instead of CPU fallback.
            return _empty_nearest_result(return_distance), None
        # fall through to point-point distance below
    elif _points_only(query_owned):
        tree_family = _tree_distance_family(tree_owned)
        if tree_family is not None:
            return _nearest_refine_gpu_typed(
                query_owned, tree_owned, left_idx, right_idx, n_queries,
                PointFamilyDistanceStrategy(tree_family),
                max_distance=max_distance, return_all=return_all,
                exclusive=exclusive, return_distance=return_distance,
            )
        use_mixed = True
    elif _points_only(tree_owned):
        query_family = _single_family(query_owned)
        if query_family is not None and query_family in _point_distance_families():
            return _nearest_refine_gpu_typed(
                tree_owned, query_owned, right_idx, left_idx, n_queries,
                PointFamilyDistanceStrategy(query_family),
                max_distance=max_distance, return_all=return_all,
                exclusive=exclusive, return_distance=return_distance,
            )
        use_mixed = True
    else:
        query_family = _single_family(query_owned)
        tree_family = _single_family(tree_owned)
        if query_family is not None and tree_family is not None:
            return _nearest_refine_gpu_typed(
                query_owned, tree_owned, left_idx, right_idx, n_queries,
                SegmentFamilyDistanceStrategy(query_family, tree_family),
                max_distance=max_distance, return_all=return_all,
                exclusive=exclusive, return_distance=return_distance,
            )
        use_mixed = True

    # --- Mixed-family fallback: per-pair tag dispatch ---
    if use_mixed:
        mixed_distances = _compute_mixed_distances_gpu(
            query_owned, tree_owned, left_idx, right_idx, exclusive=exclusive,
        )
        if mixed_distances is not None:
            return _nearest_from_distances(
                left_idx, right_idx, mixed_distances, n_queries,
                max_distance=max_distance, return_all=return_all,
                return_distance=return_distance,
            ), None
        return None

    if GeometryFamily.POINT not in query_owned.families or GeometryFamily.POINT not in tree_owned.families:
        # Degenerate: both _points_only but POINT not in families -- all null.
        return _empty_nearest_result(return_distance), None

    # Point-point path: use PointPointDistanceStrategy via the unified pipeline.
    strategy = PointPointDistanceStrategy()
    return _nearest_refine_gpu_typed(
        query_owned, tree_owned, left_idx, right_idx, n_queries,
        strategy,
        max_distance=max_distance, return_all=return_all,
        exclusive=exclusive, return_distance=return_distance,
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
                query_bounds, tree_bounds, per_row_dist,
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
            query_owned, tree_owned,
            left_idx, right_idx, n_queries,
            max_distance=distance,
            return_all=return_all,
            exclusive=exclusive,
            return_distance=return_distance,
        )
        if gpu_result is not None:
            result, _ = gpu_result
            return result, "owned_gpu_nearest"

        # GPU refinement declined (unsupported family combo) -- fall back.
        return None

    # Exhausted iterations without full coverage -- fall back.
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def nearest_spatial_index(
    tree_geometries: np.ndarray,
    geometry: Any,
    *,
    tree_query_nearest,
    return_all: bool = True,
    max_distance: float | None = None,
    return_distance: bool = False,
    exclusive: bool = False,
) -> tuple[Any, str]:
    """Find nearest tree geometry for each query geometry.

    Returns ``(result, implementation)`` where *implementation* is one of
    ``"strtree_host"``, ``"owned_gpu_nearest"``, or ``"owned_cpu_nearest"``.
    """
    query_values, scalar = _as_geometry_array(geometry)
    if query_values is None:
        result = _empty_nearest_result(return_distance)
        return result, "owned_cpu_nearest"

    # --- NEW: Try zero-copy GPU grid nearest (bypasses _to_owned entirely) ---
    if has_gpu_runtime():
        grid_result = _nearest_grid_gpu(
            tree_geometries, query_values,
            return_all=return_all, return_distance=return_distance,
            exclusive=exclusive, max_distance=max_distance,
        )
        if grid_result is not None:
            return grid_result

    # --- Convert owned arrays and compute bounds (shared by both paths) ---
    query_owned = _to_owned(query_values)
    tree_owned = _to_owned(tree_geometries)

    # Try the efficient indexed GPU nearest path for all-Point arrays.
    # Works for both bounded (max_distance != None) and unbounded nearest.
    indexed_result = _nearest_indexed_point_gpu(
        query_owned,
        tree_owned,
        return_all=return_all,
        return_distance=return_distance,
        exclusive=exclusive,
        max_distance=max_distance,
    )
    if indexed_result is not None:
        return indexed_result

    query_bounds = compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode())
    tree_bounds = compute_geometry_bounds(tree_owned, dispatch_mode=_gpu_bounds_dispatch_mode())

    # --- Try device-side k-NN query (vibeSpatial-247.7.2) ---------------------
    # Unified GPU pipeline: candidate generation -> exact distance -> top-k.
    if has_gpu_runtime():
        from .spatial_index_knn_device import spatial_index_knn_device

        knn_result = spatial_index_knn_device(
            query_owned,
            tree_owned,
            query_bounds,
            tree_bounds,
            k=1,
            max_distance=max_distance,
            exclusive=exclusive,
            return_all=return_all,
        )
        if knn_result is not None and knn_result.total_pairs > 0:
            runtime = get_cuda_runtime()
            h_left = runtime.copy_device_to_host(
                knn_result.d_query_idx,
            ).astype(np.intp, copy=False)
            h_right = runtime.copy_device_to_host(
                knn_result.d_target_idx,
            ).astype(np.intp, copy=False)
            indices = np.vstack((h_left, h_right))
            if return_distance:
                h_dist = runtime.copy_device_to_host(
                    knn_result.d_distances,
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
        n_queries = len(query_values)
        n_tree = len(tree_geometries)
        plan = plan_kernel_dispatch(
            kernel_name="nearest_knn_brute",
            kernel_class=KernelClass.COARSE,
            row_count=n_queries * n_tree,
            gpu_available=has_gpu_runtime(),
        )
        dispatch = plan.dispatch_decision
        if dispatch is not DispatchDecision.GPU:
            # Below crossover -- STRtree kNN is more efficient for small data.
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
            query_owned, tree_owned,
            query_bounds, tree_bounds,
            initial_estimate, full_diagonal,
            n_queries=n_queries,
            return_all=return_all,
            exclusive=exclusive,
            return_distance=return_distance,
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
            per_row_distance = np.full(len(query_values), effective_max_distance, dtype=np.float64)
            left_idx, right_idx = _generate_distance_pairs(query_bounds, tree_bounds, per_row_distance)
            impl = "owned_cpu_nearest"

    if left_idx.size == 0:
        result = _empty_nearest_result(return_distance)
        return result, impl

    # --- GPU nearest refinement (Tier 1 NVRTC + Tier 3a CCCL) ----------------
    # When GPU is available and both arrays contain only points, run the entire
    # distance/reduce/filter pipeline on device to avoid the Shapely host path.
    # Works for both GPU-generated and CPU-generated candidate pairs.
    if has_gpu_runtime():
        gpu_result = _nearest_refine_gpu(
            query_owned,
            tree_owned,
            left_idx,
            right_idx,
            len(query_values),
            max_distance=effective_max_distance,
            return_all=return_all,
            exclusive=exclusive,
            return_distance=return_distance,
        )
        if gpu_result is not None:
            result, _ = gpu_result
            # Upgrade impl to GPU when refine ran on device.
            return result, "owned_gpu_nearest"

    # --- CPU Shapely refinement fallback -------------------------------------
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

    min_distance = np.full(len(query_values), np.inf, dtype=np.float64)
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
    indices = np.vstack((left_idx[keep].astype(np.intp, copy=False), right_idx[keep].astype(np.intp, copy=False)))
    # ADR-0036: spatial kernels produce integer index arrays only.
    if __debug__:
        assert isinstance(indices, np.ndarray) and np.issubdtype(indices.dtype, np.integer), (
            f"ADR-0036: nearest indices must be integer ndarray, got {type(indices).__name__} dtype={getattr(indices, 'dtype', None)}"
        )
    if return_distance:
        return (indices, distances[keep]), impl
    return indices, impl
