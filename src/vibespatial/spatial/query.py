from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import OwnedGeometryArray
from vibespatial.kernels.core.geometry_analysis import (
    compute_geometry_bounds,
    compute_geometry_bounds_device,
)
from vibespatial.runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.precision import KernelClass
from vibespatial.runtime.residency import Residency

from .indexing import build_flat_spatial_index, generate_bounds_pairs
from .nearest import (  # noqa: F401
    _dwithin_refine_gpu,
    nearest_spatial_index,
)
from .query_box import (  # noqa: F401
    _coords_form_axis_aligned_box,
    _extract_box_query_bounds,
    _extract_box_query_bounds_from_owned,
    _extract_box_query_bounds_shapely,
    _extract_owned_polygon_box_bounds,
    _is_axis_aligned_box,
    _point_box_predicate_mode,
    _query_point_tree_box_index,
    _query_regular_grid_point_index,
    _query_regular_grid_rect_box_index,
)
from .query_candidates import (  # noqa: F401
    _count_candidates_gpu,
    _generate_candidates_gpu,
    _generate_candidates_gpu_device,
    _generate_candidates_gpu_multi,
    _generate_candidates_gpu_multi_device,
    _generate_candidates_gpu_scalar,
    _generate_candidates_morton_range_gpu,
    _generate_distance_pairs,
    _generate_distance_pairs_gpu,
    _generate_distance_pairs_gpu_device,
)
from .query_cpu import shapely_bounds_array

# ---------------------------------------------------------------------------
# Re-exports from decomposed modules for backward compatibility.
# Internal consumers (profile_rails, device_geometry_array, vendor code)
# import private symbols from this module.  These re-exports ensure all
# existing ``from .query import ...`` statements continue
# to work without changes.
# ---------------------------------------------------------------------------
from .query_types import (  # noqa: F401
    _POLYGON_DE9IM_PREDICATES,
    SUPPORTED_GEOM_TYPES,
    DeviceSpatialJoinResult,
    RegularGridPointIndex,
    SpatialJoinIndices,
    SpatialQueryExecution,
    _DeviceCandidates,
    _DeviceJoinResult,
)
from .query_utils import (  # noqa: F401
    _as_geometry_array,
    _expand_bounds,
    _filter_predicate_pairs,
    _filter_predicate_pairs_owned,
    _format_query_indices,
    _gpu_bounds_dispatch_mode,
    _indices_to_dense,
    _indices_to_sparse,
    _planned_query_runtime_selection,
    _reorder_scalar_tree_matches,
    _sort_indices,
    _to_owned,
    record_shapely_fallback_event,
    supports_owned_spatial_input,
)
from .spatial_index_device import (
    spatial_index_device_query,
)


def _device_candidates_to_result(
    device_cands: _DeviceCandidates,
    *,
    sort: bool,
    execution: SpatialQueryExecution,
    return_metadata: bool,
) -> Any:
    """Build a DeviceSpatialJoinResult directly from device-resident candidates.

    Eliminates the D->H->D ping-pong that occurs when regular-grid paths
    call .to_host() and then the return_device block re-uploads via
    cp.asarray().  The candidate arrays stay on device throughout.
    """
    import cupy as _cp

    d_left = device_cands.d_left
    d_right = device_cands.d_right
    if sort and d_left.size > 0:
        order = _cp.lexsort(
            _cp.stack([
                _cp.asarray(d_right, dtype=_cp.int64),
                _cp.asarray(d_left, dtype=_cp.int64),
            ])
        )
        d_left = d_left[order]
        d_right = d_right[order]
    result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
    return (result, execution) if return_metadata else result


def _indices_to_device_result(
    left_idx: Any,
    right_idx: Any,
    *,
    sort: bool,
    execution: SpatialQueryExecution,
    return_metadata: bool,
) -> Any:
    """Build a DeviceSpatialJoinResult from host or device index arrays."""
    import cupy as _cp

    if hasattr(left_idx, "__cuda_array_interface__"):
        d_left = left_idx.astype(_cp.int32, copy=False)
        d_right = right_idx.astype(_cp.int32, copy=False)
    else:
        d_left = _cp.asarray(left_idx, dtype=_cp.int32)
        d_right = _cp.asarray(right_idx, dtype=_cp.int32)

    if sort and d_left.size > 0:
        order = _cp.lexsort(
            _cp.stack([
                _cp.asarray(d_right, dtype=_cp.int64),
                _cp.asarray(d_left, dtype=_cp.int64),
            ])
        )
        d_left = d_left[order]
        d_right = d_right[order]
    result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
    return (result, execution) if return_metadata else result


def query_spatial_index(
    tree_owned: OwnedGeometryArray,
    flat_index,
    geometry: Any,
    *,
    predicate: str | None = None,
    sort: bool = False,
    distance: float | np.ndarray | None = None,
    output_format: str = "indices",
    return_metadata: bool = False,
    return_device: bool = False,
    tree_shapely: np.ndarray | None = None,
    query_shapely: np.ndarray | None = None,
    precomputed_query_bounds: np.ndarray | None = None,
) -> Any:
    execution = SpatialQueryExecution(
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.CPU,
        implementation="owned_cpu_spatial_query",
        reason="repo-owned spatial query engine executed on CPU",
    )
    tree_size = flat_index.size
    if isinstance(geometry, OwnedGeometryArray):
        query_owned = geometry
        query_values = None
        scalar = False
    else:
        query_values, scalar = _as_geometry_array(geometry)
        if query_values is None:
            indices = np.asarray([], dtype=np.intp)
            formatted = _format_query_indices(
                indices,
                tree_size=tree_size,
                query_size=1,
                scalar=True,
                sort=sort,
                output_format=output_format,
            )
            return (formatted, execution) if return_metadata else formatted
        query_owned = None

    # --- Phase A: Regular grid rect box path (Shapely input, no _to_owned) ---
    # When the tree has a regular grid and queries are Shapely arrays, extract
    # bounds directly from Shapely (vectorized, ~2ms) instead of converting to
    # OwnedGeometryArray (~300ms).  This avoids the dominant bottleneck for
    # Shapely-input spatial queries.
    if (
        query_values is not None
        and query_owned is None
        and getattr(flat_index, "regular_grid", None) is not None
        and predicate in (None, "intersects")
    ):
        if precomputed_query_bounds is not None:
            _rg_bounds = precomputed_query_bounds
        elif predicate is None:
            _rg_bounds = shapely_bounds_array(query_values)
        else:
            _rg_bounds = _extract_box_query_bounds_shapely(query_values)
        regular_grid_box_pairs = _query_regular_grid_rect_box_index(
            flat_index, _rg_bounds, predicate=predicate,
        )
        if regular_grid_box_pairs is not None:
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason="repo-owned regular-grid rectangle box query executed on GPU with exact range expansion",
            )
            # Short-circuit: keep indices on device when caller wants device
            # results (ZCOPY001: avoid D->H->D ping-pong in regular-grid path).
            if (
                return_device
                and not scalar
                and output_format == "indices"
                and isinstance(regular_grid_box_pairs, _DeviceCandidates)
            ):
                return _device_candidates_to_result(
                    regular_grid_box_pairs,
                    sort=sort,
                    execution=execution,
                    return_metadata=return_metadata,
                )
            if isinstance(regular_grid_box_pairs, _DeviceCandidates):
                left_idx, right_idx = regular_grid_box_pairs.to_host()
            else:
                left_idx, right_idx = regular_grid_box_pairs
            if scalar:
                if not sort:
                    right_idx = _reorder_scalar_tree_matches(
                        right_idx.astype(np.intp, copy=False),
                        flat_index.order,
                    )
                indices = right_idx.astype(np.intp, copy=False)
            else:
                indices = np.vstack(
                    (
                        left_idx.astype(np.intp, copy=False),
                        right_idx.astype(np.intp, copy=False),
                    )
                )
            query_size = 1 if scalar else len(query_values)
            formatted = _format_query_indices(
                indices,
                tree_size=tree_size,
                query_size=query_size,
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            return (formatted, execution) if return_metadata else formatted

    # --- Phase B: Point-tree box fast path ---
    # Only attempt the expensive per-element box detection when the tree
    # contains points (the only case where this path can succeed).
    _tree_is_point_only = (
        GeometryFamily.POINT in tree_owned.families
        and len(tree_owned.families) == 1
    )

    # For predicate=None with Shapely input, use shapely.bounds() directly
    # (~2ms) instead of _to_owned() (~200-500ms) for bounds extraction.
    # The owned conversion is deferred until predicate refinement needs it.
    if query_values is not None and query_owned is None and predicate is None:
        _shapely_query_bounds = shapely_bounds_array(query_values)
    else:
        _shapely_query_bounds = None

    point_box_bounds = None
    if query_values is not None and _tree_is_point_only:
        if predicate is None and _shapely_query_bounds is not None:
            point_box_bounds = _shapely_query_bounds
        elif predicate is None and query_owned is not None:
            point_box_bounds = compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode(query_owned))
        else:
            point_box_bounds = _extract_box_query_bounds(predicate, query_values)
        point_box_pairs = _query_point_tree_box_index(
            tree_owned,
            predicate=predicate,
            query_row_count=len(query_values),
            box_bounds=point_box_bounds,
        )
        if point_box_pairs is not None:
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason="point-tree box query selected the repo-owned GPU candidate kernel",
            )
            left_idx, right_idx = point_box_pairs
            if scalar:
                if not sort:
                    right_idx = _reorder_scalar_tree_matches(
                        right_idx.astype(np.intp, copy=False),
                        flat_index.order,
                    )
                indices = right_idx.astype(np.intp, copy=False)
            else:
                indices = np.vstack(
                    (
                        left_idx.astype(np.intp, copy=False),
                        right_idx.astype(np.intp, copy=False),
                    )
                )
            formatted = _format_query_indices(
                indices,
                tree_size=tree_size,
                query_size=1 if scalar else len(query_values),
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            return (formatted, execution) if return_metadata else formatted

    # --- Phase C: OwnedGeometryArray paths (conversion if needed) ---
    # Defer _to_owned() as long as possible.  For regular-grid and point-index
    # fast paths that only need bounds, we use shapely.bounds() (vectorized,
    # ~2ms) or the previously extracted _shapely_query_bounds.  _to_owned()
    # is only called when we need the full owned geometry for predicate
    # refinement or when the fast paths all miss.
    if query_values is not None and query_owned is None:
        # Try regular-grid and point-index paths with Shapely bounds first,
        # deferring _to_owned() until after those paths have been tried.
        _deferred_to_owned = True
    else:
        _deferred_to_owned = False

    if _deferred_to_owned:
        # Use Shapely bounds for regular-grid box path.
        if _shapely_query_bounds is not None:
            regular_grid_box_bounds = _shapely_query_bounds
        else:
            regular_grid_box_bounds = shapely_bounds_array(query_values)
    else:
        regular_grid_box_bounds = None
        if query_owned is not None:
            regular_grid_box_bounds = _extract_box_query_bounds_from_owned("intersects", query_owned)

    regular_grid_box_pairs = _query_regular_grid_rect_box_index(
        flat_index,
        regular_grid_box_bounds,
        predicate=predicate,
    )
    if regular_grid_box_pairs is not None:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="repo-owned regular-grid rectangle box query executed on GPU with exact range expansion",
        )
        # Short-circuit: keep indices on device when caller wants device
        # results (ZCOPY001: avoid D->H->D ping-pong in regular-grid path).
        if (
            return_device
            and not scalar
            and output_format == "indices"
            and isinstance(regular_grid_box_pairs, _DeviceCandidates)
        ):
            return _device_candidates_to_result(
                regular_grid_box_pairs,
                sort=sort,
                execution=execution,
                return_metadata=return_metadata,
            )
        if isinstance(regular_grid_box_pairs, _DeviceCandidates):
            left_idx, right_idx = regular_grid_box_pairs.to_host()
        else:
            left_idx, right_idx = regular_grid_box_pairs
        if scalar:
            if not sort:
                right_idx = _reorder_scalar_tree_matches(
                    right_idx.astype(np.intp, copy=False),
                    flat_index.order,
                )
            indices = right_idx.astype(np.intp, copy=False)
        else:
            indices = np.vstack(
                (
                    left_idx.astype(np.intp, copy=False),
                    right_idx.astype(np.intp, copy=False),
                )
            )
        query_size = 1 if scalar else (len(query_values) if query_values is not None else query_owned.row_count)
        formatted = _format_query_indices(
            indices,
            tree_size=tree_size,
            query_size=query_size,
            scalar=scalar,
            sort=sort,
            output_format=output_format,
        )
        return (formatted, execution) if return_metadata else formatted

    # Now convert to owned if still deferred — needed for remaining paths.
    if _deferred_to_owned and query_owned is None:
        query_owned = _to_owned(query_values)

    fast_pairs = _query_regular_grid_point_index(flat_index, query_owned, predicate=predicate)
    if fast_pairs is not None:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="regular-grid point query selected the repo-owned GPU candidate kernel",
        )
        # Short-circuit: keep indices on device when caller wants device
        # results (ZCOPY001: avoid D->H->D ping-pong in regular-grid path).
        if (
            return_device
            and not scalar
            and output_format == "indices"
            and isinstance(fast_pairs, _DeviceCandidates)
        ):
            return _device_candidates_to_result(
                fast_pairs,
                sort=sort,
                execution=execution,
                return_metadata=return_metadata,
            )
        if isinstance(fast_pairs, _DeviceCandidates):
            left_idx, right_idx = fast_pairs.to_host()
        else:
            left_idx, right_idx = fast_pairs
        if scalar:
            if not sort:
                right_idx = _reorder_scalar_tree_matches(
                    right_idx.astype(np.intp, copy=False),
                    flat_index.order,
                )
            indices = right_idx.astype(np.intp, copy=False)
        else:
            indices = np.vstack(
                (
                    left_idx.astype(np.intp, copy=False),
                    right_idx.astype(np.intp, copy=False),
                )
            )
        query_size = 1 if scalar else query_owned.row_count
        formatted = _format_query_indices(
            indices,
            tree_size=tree_size,
            query_size=query_size,
            scalar=scalar,
            sort=sort,
            output_format=output_format,
        )
        return (formatted, execution) if return_metadata else formatted

    point_box_pairs = _query_point_tree_box_index(
        tree_owned,
        predicate=predicate,
        query_row_count=query_owned.row_count,
        box_bounds=_extract_box_query_bounds_from_owned(predicate, query_owned),
    )
    if point_box_pairs is not None:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="point-tree box query selected the repo-owned GPU candidate kernel",
        )
        left_idx, right_idx = point_box_pairs
        if scalar:
            if not sort:
                right_idx = _reorder_scalar_tree_matches(
                    right_idx.astype(np.intp, copy=False),
                    flat_index.order,
                )
            indices = right_idx.astype(np.intp, copy=False)
        else:
            indices = np.vstack(
                (
                    left_idx.astype(np.intp, copy=False),
                    right_idx.astype(np.intp, copy=False),
                )
            )
        query_size = 1 if scalar else query_owned.row_count
        formatted = _format_query_indices(
            indices,
            tree_size=tree_size,
            query_size=query_size,
            scalar=scalar,
            sort=sort,
            output_format=output_format,
        )
        return (formatted, execution) if return_metadata else formatted

    # Compute bounds for candidate generation.  If we already extracted
    # Shapely bounds earlier (no _to_owned needed), reuse them.
    # Otherwise compute from the owned geometry array.
    query_bounds_on_device = False
    if _shapely_query_bounds is not None:
        query_bounds = _shapely_query_bounds
    else:
        if (
            predicate == "dwithin"
            and query_owned.residency is Residency.DEVICE
            and has_gpu_runtime()
        ):
            query_bounds = compute_geometry_bounds_device(query_owned)
            query_bounds_on_device = True
        else:
            query_bounds = compute_geometry_bounds(
                query_owned,
                dispatch_mode=_gpu_bounds_dispatch_mode(query_owned),
            )
    tree_bounds = flat_index.bounds
    query_size = len(query_values) if query_values is not None else query_owned.row_count

    # Shapely arrays for predicate refinement fallback.  GPU DE-9IM and dwithin
    # paths avoid to_shapely() entirely; these are only materialised when the
    # Shapely fallback is actually needed.
    _query_shapely = query_shapely if query_shapely is not None else query_values
    _tree_shapely = tree_shapely   # caller-provided or None

    gpu_candidate_gen = False
    device_indices_materialized_from_gpu = False
    dwithin_used_shapely_fallback = False
    if predicate == "dwithin":
        if distance is None:
            raise ValueError("'distance' parameter is required for 'dwithin' predicate")
        query_size_for_dist = len(query_values) if query_values is not None else query_owned.row_count
        if np.isscalar(distance):
            per_row_distance = np.full(query_size_for_dist, float(distance), dtype=np.float64)
        else:
            per_row_distance = np.asarray(distance, dtype=np.float64)
            if per_row_distance.shape != (query_size_for_dist,):
                raise ValueError("distance array must be broadcastable to the geometry input")
        # Try GPU candidate generation first (device-resident), then GPU refinement.
        device_dist_cands, _dwithin_exec = spatial_index_device_query(
            flat_index, query_bounds, distance=per_row_distance,
        )
        if device_dist_cands is not None:
            gpu_candidate_gen = True
            left_idx = None
            right_idx = None
        elif query_bounds_on_device and _dwithin_exec.selected is ExecutionMode.GPU:
            left_idx = np.empty(0, dtype=np.int32)
            right_idx = np.empty(0, dtype=np.int32)
        else:
            host_query_bounds = (
                compute_geometry_bounds(
                    query_owned,
                    dispatch_mode=_gpu_bounds_dispatch_mode(query_owned),
                )
                if query_bounds_on_device
                else query_bounds
            )
            left_idx, right_idx = _generate_distance_pairs(
                host_query_bounds,
                tree_bounds,
                per_row_distance,
            )

        # Try GPU dwithin refinement first.
        # When return_device is requested, ask _dwithin_refine_gpu to keep
        # results on device (CuPy arrays) to avoid D->H + H->D round-trip.
        gpu_dwithin_result = _dwithin_refine_gpu(
            query_owned, tree_owned, left_idx, right_idx, per_row_distance,
            device_candidates=device_dist_cands if gpu_candidate_gen else None,
            return_device=return_device,
        )
        if gpu_dwithin_result is not None:
            (left_idx, right_idx), used_shapely_fallback = gpu_dwithin_result
            dwithin_used_shapely_fallback = used_shapely_fallback
            refine_selection = _planned_query_runtime_selection(
                kernel_name="dwithin_refine",
                kernel_class=KernelClass.METRIC,
                row_count=left_idx.size,
                requested_mode=ExecutionMode.CPU if used_shapely_fallback else ExecutionMode.GPU,
                gpu_available=True,
                reason=(
                    "GPU dwithin refinement via distance kernels"
                    if not used_shapely_fallback
                    else "GPU candidate generation with Shapely dwithin refinement"
                ),
            )
            # When return_device is True and results are already device-
            # resident CuPy arrays, return early to avoid the generic
            # return_device block which would D->H->D round-trip via
            # cp.asarray.  The _dwithin_refine_gpu(return_device=True)
            # path keeps indices on device throughout.
            if (
                return_device
                and not used_shapely_fallback
                and hasattr(left_idx, "__cuda_array_interface__")
            ):
                import cupy as _cp_dw
                execution = SpatialQueryExecution(
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.GPU,
                    implementation="owned_gpu_spatial_query",
                    reason="repo-owned GPU bbox candidate generation with GPU exact predicate refinement",
                )
                d_left = left_idx.astype(_cp_dw.int32, copy=False)
                d_right = right_idx.astype(_cp_dw.int32, copy=False)
                if sort and d_left.size > 0:
                    order = _cp_dw.lexsort(_cp_dw.stack([d_right.astype(_cp_dw.int64), d_left.astype(_cp_dw.int64)]))
                    d_left = d_left[order]
                    d_right = d_right[order]
                device_result = DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)
                return (device_result, execution) if return_metadata else device_result
        else:
            if gpu_candidate_gen:
                record_shapely_fallback_event(
                    surface="vibespatial.spatial.query",
                    reason="GPU candidate refinement fell back to Shapely dwithin refinement",
                    detail="predicate='dwithin'",
                    pipeline="gpu_candidates -> shapely_refine",
                    d2h_transfer=True,
                )
            # CPU Shapely fallback.
            if left_idx is None or right_idx is None:
                left_idx, right_idx = device_dist_cands.to_host()
                device_indices_materialized_from_gpu = True
            if query_values is None:
                query_values = np.asarray(query_owned.to_shapely(), dtype=object)
                _query_shapely = query_values
            if _tree_shapely is None:
                _tree_shapely = np.asarray(tree_owned.to_shapely(), dtype=object)
            tree_values = _tree_shapely
            left_idx, right_idx, refine_selection = _filter_predicate_pairs(
                predicate,
                query_values,
                tree_values,
                left_idx,
                right_idx,
                distance=per_row_distance,
            )
    else:
        # For count-only queries, skip full predicate refinement and return
        # the candidate pair count directly (ADR-0005: zero D→H transfer).
        # Use the count-only GPU kernel that runs only the bbox overlap
        # count pass without materializing pair arrays or running the
        # scatter kernel.  This avoids the Morton range path which has
        # higher overhead at moderate N*M products (100K x 10K) where
        # the brute-force count kernel is 20-30x faster.
        if output_format == "count":
            gpu_count = _count_candidates_gpu(query_bounds, tree_bounds)
            if gpu_count is not None:
                execution = SpatialQueryExecution(
                    requested=ExecutionMode.AUTO, selected=ExecutionMode.GPU,
                    implementation="owned_gpu_spatial_query",
                    reason="count-only query: GPU count kernel returned without pair materialization",
                )
                return (gpu_count, execution) if return_metadata else gpu_count
            # CPU fallback
            pairs = generate_bounds_pairs(query_owned, flat_index.geometry_array)
            count = int(pairs.count)
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                implementation="owned_gpu_spatial_query",
                reason="count-only query: GPU unavailable, CPU bbox pair count",
            )
            return (count, execution) if return_metadata else count

        # GPU candidate generation via unified spatial_index_device_query.
        # Automatically selects Morton range O(N*log(M)+K) for large inputs
        # and brute-force O(N*M) for small inputs.  Crossover is at ~1M
        # (lowered from 100M for ADR-0034 warm kernels).
        # Falls back to CPU generate_bounds_pairs when GPU is unavailable.
        #
        # Predicate refinement uses indexed access into original owned arrays
        # (no buffer copy via .take()).  When device-resident candidates are
        # available, sub-arrays are extracted on-device via CuPy fancy indexing
        # to avoid redundant host→device transfers.
        device_cands, _sidq_exec = spatial_index_device_query(
            flat_index, query_bounds,
        )
        if device_cands is not None:
            gpu_candidate_gen = True
            # Pass None for host indices — _filter_predicate_pairs_owned will
            # lazily materialise them from device_candidates only when needed
            # (tag classification, Shapely fallback).  This avoids a full
            # D→H transfer when the GPU predicate path handles all pairs.
            left_idx, right_idx, refine_selection = _filter_predicate_pairs_owned(
                predicate,
                query_owned,
                tree_owned,
                None,
                None,
                query_shapely=_query_shapely,
                tree_shapely=_tree_shapely,
                device_candidates=device_cands,
            )
        else:
            # CPU fallback for candidate generation.
            pairs = generate_bounds_pairs(query_owned, flat_index.geometry_array)
            left_idx, right_idx, refine_selection = _filter_predicate_pairs_owned(
                predicate,
                query_owned,
                tree_owned,
                pairs.left_indices,
                pairs.right_indices,
                query_shapely=_query_shapely,
                tree_shapely=_tree_shapely,
            )

    # Execution labeling: GPU candidate generation is the dominant stage,
    # so label the full query as GPU when candidate gen ran on GPU.
    if gpu_candidate_gen:
        if predicate == "dwithin":
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU if dwithin_used_shapely_fallback else ExecutionMode.GPU,
                implementation="owned_cpu_spatial_query" if dwithin_used_shapely_fallback else "owned_gpu_spatial_query",
                reason=(
                    "repo-owned GPU bbox candidate generation with Shapely dwithin refinement"
                    if dwithin_used_shapely_fallback
                    else "repo-owned GPU bbox candidate generation with GPU dwithin refinement"
                ),
            )
        elif predicate is None or refine_selection.selected is ExecutionMode.GPU:
            reason_parts = ["repo-owned GPU bbox candidate generation"]
            if predicate is not None:
                reason_parts.append(
                    f"with {refine_selection.selected.value} exact predicate refinement"
                )
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                implementation="owned_gpu_spatial_query",
                reason=" ".join(reason_parts),
            )
        else:
            execution = SpatialQueryExecution(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                implementation="owned_cpu_spatial_query",
                reason="repo-owned GPU bbox candidate generation with Shapely exact predicate refinement",
            )
    elif refine_selection.selected is ExecutionMode.GPU:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason=(
                "repo-owned spatial query engine selected GPU exact predicate refinement "
                "after owned candidate generation"
            ),
        )
    elif predicate == "dwithin":
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="repo-owned spatial query engine handled dwithin on CPU",
        )
    elif predicate is None:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="repo-owned spatial query engine handled bbox-only candidate generation on CPU",
        )
    else:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
            implementation="owned_cpu_spatial_query",
            reason="repo-owned spatial query engine handled candidate generation and exact refinement on CPU",
        )

    # Fast path for count-only queries: return pair count without building
    # the full index array (ADR-0005: avoid D→H transfer when only the count
    # is needed, e.g. for pipeline stage metadata).
    if output_format == "count":
        count = int(np.unique(left_idx).size) if left_idx.size > 0 else 0
        return (count, execution) if return_metadata else count

    # Phase 2 zero-copy: when caller requests device-resident index arrays
    # (e.g., overlay with both sides owned-backed), keep refined indices on
    # device to feed directly into device_take() — eliminates H->D re-upload.
    # Only applies to non-scalar "indices" format with GPU execution.
    if (
        return_device
        and not scalar
        and output_format == "indices"
        and execution.selected is ExecutionMode.GPU
        and not device_indices_materialized_from_gpu
        and has_gpu_runtime()
    ):
        return _indices_to_device_result(
            left_idx,
            right_idx,
            sort=sort,
            execution=execution,
            return_metadata=return_metadata,
        )

    if scalar:
        indices = right_idx.astype(np.intp, copy=False)
        if not sort:
            indices = _reorder_scalar_tree_matches(indices, flat_index.order)
    else:
        indices = np.vstack(
            (
                left_idx.astype(np.intp, copy=False),
                right_idx.astype(np.intp, copy=False),
            )
        )

    formatted = _format_query_indices(
        indices,
        tree_size=tree_size,
        query_size=query_size,
        scalar=scalar,
        sort=sort,
        output_format=output_format,
    )
    return (formatted, execution) if return_metadata else formatted


def build_owned_spatial_index(geometry: np.ndarray) -> tuple[OwnedGeometryArray, Any]:
    owned = _to_owned(np.asarray(geometry, dtype=object))
    selection = plan_dispatch_selection(
        kernel_name="flat_index_build",
        kernel_class=KernelClass.COARSE,
        row_count=owned.row_count,
    )
    index = build_flat_spatial_index(owned, runtime_selection=selection.runtime_selection)

    # ADR-0034 Level 3: eagerly import the spatial query kernel module so
    # its module-scope request_nvrtc_warmup() fires, then block until all
    # NVRTC and CCCL background compilations finish.  This front-loads JIT
    # cost (~1-2s) into the index build rather than the first query call,
    # eliminating the 400-500x cold-query penalty at small scales (20k).
    if selection.selected is ExecutionMode.GPU:
        import vibespatial.spatial.query_candidates  # noqa: F401 — triggers warmup
        from vibespatial.cuda.cccl_precompile import ensure_pipelines_warm

        ensure_pipelines_warm(timeout=30.0)

    return owned, index
