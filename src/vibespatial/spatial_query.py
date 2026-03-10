from __future__ import annotations

from typing import Any

import numpy as np

from vibespatial.indexing import build_flat_spatial_index, generate_bounds_pairs
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds
from vibespatial.owned_geometry import OwnedGeometryArray
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime

# ---------------------------------------------------------------------------
# Re-exports from decomposed modules for backward compatibility.
# Internal consumers (profile_rails, device_geometry_array, vendor code)
# import private symbols from this module.  These re-exports ensure all
# existing ``from vibespatial.spatial_query import ...`` statements continue
# to work without changes.
# ---------------------------------------------------------------------------

from vibespatial.spatial_query_types import (  # noqa: F401
    SUPPORTED_GEOM_TYPES,
    SpatialQueryExecution,
    SpatialJoinIndices,
    RegularGridPointIndex,
    _DeviceCandidates,
    _DeviceJoinResult,
    _POLYGON_DE9IM_PREDICATES,
)
from vibespatial.spatial_query_utils import (  # noqa: F401
    _as_geometry_array,
    _expand_bounds,
    _filter_predicate_pairs,
    _filter_predicate_pairs_owned,
    _format_query_indices,
    _gpu_bounds_dispatch_mode,
    _indices_to_dense,
    _indices_to_sparse,
    _sort_indices,
    _to_owned,
    supports_owned_spatial_input,
)
from vibespatial.spatial_query_candidates import (  # noqa: F401
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
from vibespatial.spatial_query_box import (  # noqa: F401
    _coords_form_axis_aligned_box,
    _extract_box_query_bounds,
    _extract_box_query_bounds_from_owned,
    _extract_owned_polygon_box_bounds,
    _is_axis_aligned_box,
    _point_box_predicate_mode,
    _query_point_tree_box_index,
    _query_regular_grid_point_index,
    _query_regular_grid_rect_box_index,
)
from vibespatial.spatial_nearest import (  # noqa: F401
    _dwithin_refine_gpu,
    nearest_spatial_index,
)


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
    tree_shapely: np.ndarray | None = None,
    query_shapely: np.ndarray | None = None,
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

    # For predicate=None, build query_owned eagerly so we can use owned
    # bounds instead of shapely.bounds().  Other predicates defer _to_owned
    # to preserve the scalar box fast path (no owned conversion needed).
    if query_values is not None and query_owned is None and predicate is None:
        query_owned = _to_owned(query_values)

    point_box_bounds = None
    if query_values is not None:
        if predicate is None and query_owned is not None:
            point_box_bounds = compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode())
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

    if query_values is not None and query_owned is None:
        query_owned = _to_owned(query_values)

    regular_grid_box_bounds = None
    if query_owned is not None:
        regular_grid_box_bounds = _extract_box_query_bounds_from_owned("intersects", query_owned)
    regular_grid_box_pairs = _query_regular_grid_rect_box_index(
        flat_index,
        regular_grid_box_bounds,
        predicate=predicate,
    )
    if regular_grid_box_pairs is not None:
        if isinstance(regular_grid_box_pairs, _DeviceCandidates):
            left_idx, right_idx = regular_grid_box_pairs.to_host()
        else:
            left_idx, right_idx = regular_grid_box_pairs
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="repo-owned regular-grid rectangle box query executed on GPU with exact range expansion",
        )
        if scalar:
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

    fast_pairs = _query_regular_grid_point_index(flat_index, query_owned, predicate=predicate)
    if fast_pairs is not None:
        execution = SpatialQueryExecution(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU,
            implementation="owned_gpu_spatial_query",
            reason="regular-grid point query selected the repo-owned GPU candidate kernel",
        )
        if isinstance(fast_pairs, _DeviceCandidates):
            left_idx, right_idx = fast_pairs.to_host()
        else:
            left_idx, right_idx = fast_pairs
        if scalar:
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

    # Compute bounds for candidate generation — available from owned geometry
    # without Shapely round-trip.
    query_bounds = compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode())
    tree_bounds = flat_index.bounds
    query_size = len(query_values) if query_values is not None else query_owned.row_count

    # Shapely arrays for predicate refinement fallback.  GPU DE-9IM and dwithin
    # paths avoid to_shapely() entirely; these are only materialised when the
    # Shapely fallback is actually needed.
    _query_shapely = query_shapely if query_shapely is not None else query_values
    _tree_shapely = tree_shapely   # caller-provided or None

    gpu_candidate_gen = False
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
        device_dist_cands = _generate_distance_pairs_gpu_device(
            query_bounds, tree_bounds, per_row_distance,
        )
        if device_dist_cands is not None:
            gpu_candidate_gen = True
            left_idx, right_idx = device_dist_cands.to_host()
        else:
            left_idx, right_idx = _generate_distance_pairs(query_bounds, tree_bounds, per_row_distance)

        # Try GPU dwithin refinement first.
        gpu_dwithin = _dwithin_refine_gpu(
            query_owned, tree_owned, left_idx, right_idx, per_row_distance,
            device_candidates=device_dist_cands if gpu_candidate_gen else None,
        )
        if gpu_dwithin is not None:
            left_idx, right_idx = gpu_dwithin
            refine_selection = RuntimeSelection(
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.GPU,
                reason="GPU dwithin refinement via distance kernels",
            )
        else:
            # CPU Shapely fallback.
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
        # The candidate count is known from the GPU kernel output without
        # materializing the full index arrays.
        if output_format == "count":
            _n_product = query_bounds.shape[0] * tree_bounds.shape[0]
            if _n_product >= 100_000_000:
                device_cands = _generate_candidates_morton_range_gpu(flat_index, query_bounds)
            else:
                device_cands = None
            if device_cands is None:
                device_cands = _generate_candidates_gpu_device(query_bounds, tree_bounds)
            if device_cands is not None:
                count = int(device_cands.total_pairs)
                execution = SpatialQueryExecution(
                    requested=ExecutionMode.AUTO, selected=ExecutionMode.GPU,
                    implementation="owned_gpu_spatial_query",
                    reason="count-only query: GPU candidate count returned without D→H index transfer",
                )
                return (count, execution) if return_metadata else count
            # CPU fallback
            pairs = generate_bounds_pairs(query_owned, flat_index.geometry_array)
            count = int(pairs.count)
            return (count, execution) if return_metadata else count

        # GPU candidate generation: Morton range O(N*log(M)+K) for large
        # inputs, brute-force O(N*M) for small inputs.  Morton range has
        # higher per-call overhead (6 kernel launches vs 2) but scales
        # better — crossover is around N*M ≈ 10 billion (≈100K×100K).
        # Falls back to CPU generate_bounds_pairs when GPU is unavailable.
        #
        # Predicate refinement uses indexed access into original owned arrays
        # (no buffer copy via .take()).  When device-resident candidates are
        # available, sub-arrays are extracted on-device via CuPy fancy indexing
        # to avoid redundant host→device transfers.
        _n_product = query_bounds.shape[0] * tree_bounds.shape[0]
        if _n_product >= 100_000_000:  # ~10K×10K crossover (lowered from 10B for ADR-0034 warm kernels)
            device_cands = _generate_candidates_morton_range_gpu(flat_index, query_bounds)
        else:
            device_cands = None
        if device_cands is None:
            device_cands = _generate_candidates_gpu_device(query_bounds, tree_bounds)
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

    if scalar:
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
        query_size=query_size,
        scalar=scalar,
        sort=sort,
        output_format=output_format,
    )
    return (formatted, execution) if return_metadata else formatted


def build_owned_spatial_index(geometry: np.ndarray) -> tuple[OwnedGeometryArray, Any]:
    owned = _to_owned(np.asarray(geometry, dtype=object))
    use_gpu = has_gpu_runtime()
    selection = RuntimeSelection(
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU if use_gpu else ExecutionMode.CPU,
        reason=(
            "repo-owned spatial index build (GPU morton sort)"
            if use_gpu
            else "repo-owned spatial index build (CPU)"
        ),
    )
    index = build_flat_spatial_index(owned, runtime_selection=selection)
    return owned, index
