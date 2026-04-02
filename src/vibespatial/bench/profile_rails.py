from __future__ import annotations

import numpy as np
from shapely.affinity import translate

from vibespatial.geometry.owned import from_shapely_geometries
from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds, compute_morton_keys
from vibespatial.overlay.reconstruction import OverlayOperation, plan_overlay_reconstruction
from vibespatial.predicates.binary import evaluate_binary_predicate
from vibespatial.runtime import ExecutionMode, RuntimeSelection, has_gpu_runtime, select_runtime
from vibespatial.runtime.config import COARSE_BOUNDS_TILE_SIZE, SEGMENT_TILE_SIZE
from vibespatial.runtime.precision import KernelClass, PrecisionMode, select_precision_plan
from vibespatial.runtime.robustness import select_robustness_plan
from vibespatial.spatial.indexing import build_flat_spatial_index, generate_bounds_pairs
from vibespatial.spatial.query import (
    _as_geometry_array,
    _DeviceCandidates,
    _extract_box_query_bounds,
    _extract_box_query_bounds_from_owned,
    _filter_predicate_pairs,
    _filter_predicate_pairs_owned,
    _format_query_indices,
    _generate_candidates_gpu,
    _generate_distance_pairs,
    _gpu_bounds_dispatch_mode,
    _query_point_tree_box_index,
    _query_regular_grid_point_index,
    _query_regular_grid_rect_box_index,
    _to_owned,
)
from vibespatial.spatial.segment_primitives import (
    _classify_segment_intersections_from_tables,
    _generate_segment_candidates_from_tables,
    extract_segments,
)
from vibespatial.testing.synthetic import SyntheticSpec, generate_polygons

from .profiling import ProfileTrace, StageProfiler


def _build_join_inputs(rows: int, *, overlap_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    tree = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=rows, seed=4)
            ).geometries
        ),
        dtype=object,
    )
    query = tree.copy()
    cutoff = int(rows * overlap_ratio)
    if cutoff < rows:
        query[cutoff:] = np.asarray(
            [translate(geometry, xoff=10_000.0, yoff=10_000.0) for geometry in query[cutoff:]],
            dtype=object,
        )
    return tree, query


def _build_overlay_inputs(rows: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(
        list(
            generate_polygons(
                SyntheticSpec(geometry_type="polygon", distribution="regular-grid", count=rows, seed=10)
            ).geometries
        ),
        dtype=object,
    )
    right = np.asarray([translate(geometry, xoff=0.3, yoff=0.3) for geometry in left], dtype=object)
    return left, right


def profile_join_kernel(
    *,
    rows: int,
    overlap_ratio: float = 0.2,
    tile_size: int = COARSE_BOUNDS_TILE_SIZE,
    predicate: str = "intersects",
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    enable_nvtx: bool = False,
) -> ProfileTrace:
    runtime_selection = select_runtime(dispatch_mode)
    profiler = StageProfiler(
        operation="join",
        dataset=f"polygon-{rows}",
        requested_runtime=dispatch_mode,
        selected_runtime=ExecutionMode.CPU,
        enable_nvtx=enable_nvtx,
    )

    tree_values, query_values = _build_join_inputs(rows, overlap_ratio=overlap_ratio)
    with profiler.stage(
        "build_tree_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="convert tree geometries into owned buffers",
    ) as stage:
        tree_owned = from_shapely_geometries(tree_values.tolist())
        stage.rows_out = tree_owned.row_count

    with profiler.stage(
        "build_query_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="convert query geometries into owned buffers",
    ) as stage:
        query_owned = from_shapely_geometries(query_values.tolist())
        stage.rows_out = query_owned.row_count

    with profiler.stage(
        "compute_tree_bounds",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=tree_owned.row_count,
        detail="derive tree bounds for coarse pruning",
    ) as stage:
        tree_bounds = compute_geometry_bounds(tree_owned)
        stage.rows_out = int(tree_bounds.shape[0])

    with profiler.stage(
        "compute_query_bounds",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=query_owned.row_count,
        detail="derive query bounds for coarse pruning",
    ) as stage:
        query_bounds = compute_geometry_bounds(query_owned)
        stage.rows_out = int(query_bounds.shape[0])

    with profiler.stage(
        "sort_tree_morton",
        category="sort",
        device=ExecutionMode.CPU,
        rows_in=tree_owned.row_count,
        detail="compute Morton keys and stable-sort tree rows",
    ) as stage:
        morton_keys = compute_morton_keys(tree_owned)
        order = np.argsort(morton_keys, kind="stable").astype(np.int32, copy=False)
        stage.rows_out = int(order.size)

    with profiler.stage(
        "coarse_filter",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=int(query_owned.row_count * tree_owned.row_count),
        detail="generate coarse candidate pairs from bounds overlap",
        metadata={"tile_size": tile_size},
    ) as stage:
        pairs = generate_bounds_pairs(query_owned, tree_owned, tile_size=tile_size)
        stage.rows_out = pairs.count
        stage.metadata["pairs_examined"] = pairs.pairs_examined
        stage.metadata["left_bounds_rows"] = int(query_bounds.shape[0])
        stage.metadata["right_bounds_rows"] = int(tree_bounds.shape[0])

    with profiler.stage(
        "refine_predicate",
        category="refine",
        device=ExecutionMode.CPU,
        rows_in=pairs.count,
        detail="evaluate exact predicate on coarse survivors",
    ) as stage:
        refined = evaluate_binary_predicate(
            predicate,
            query_values[pairs.left_indices],
            tree_values[pairs.right_indices],
            dispatch_mode=dispatch_mode,
            null_behavior="false",
        )
        keep = np.asarray(refined.values, dtype=bool)
        left_idx = pairs.left_indices[keep]
        right_idx = pairs.right_indices[keep]
        stage.rows_out = int(keep.sum())
        stage.metadata["predicate"] = predicate

    with profiler.stage(
        "sort_output",
        category="sort",
        device=ExecutionMode.CPU,
        rows_in=int(left_idx.size),
        detail="stable-sort surviving join pairs for deterministic output",
    ) as stage:
        if left_idx.size:
            order = np.lexsort((right_idx, left_idx))
            left_idx = left_idx[order]
            right_idx = right_idx[order]
        stage.rows_out = int(left_idx.size)

    return profiler.finish(
        metadata={
            "rows": rows,
            "overlap_ratio": overlap_ratio,
            "tile_size": tile_size,
            "predicate": predicate,
            "matched_pairs": int(left_idx.size),
            "tree_rows": int(tree_owned.row_count),
            "query_rows": int(query_owned.row_count),
            "planner_selected_runtime": runtime_selection.selected.value,
        }
    )


def profile_overlay_kernel(
    *,
    rows: int,
    tile_size: int = SEGMENT_TILE_SIZE,
    operation: OverlayOperation | str = OverlayOperation.INTERSECTION,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    enable_nvtx: bool = False,
) -> ProfileTrace:
    runtime_selection = select_runtime(dispatch_mode)
    normalized_operation = operation if isinstance(operation, OverlayOperation) else OverlayOperation(operation)
    profiler = StageProfiler(
        operation="overlay",
        dataset=f"polygon-{rows}",
        requested_runtime=dispatch_mode,
        selected_runtime=ExecutionMode.CPU,
        enable_nvtx=enable_nvtx,
    )

    left_values, right_values = _build_overlay_inputs(rows)
    with profiler.stage(
        "build_left_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="convert left overlay input into owned buffers",
    ) as stage:
        left_owned = from_shapely_geometries(left_values.tolist())
        stage.rows_out = left_owned.row_count

    with profiler.stage(
        "build_right_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="convert right overlay input into owned buffers",
    ) as stage:
        right_owned = from_shapely_geometries(right_values.tolist())
        stage.rows_out = right_owned.row_count

    with profiler.stage(
        "extract_left_segments",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=left_owned.row_count,
        detail="extract left edge segments for overlay candidate generation",
    ) as stage:
        left_segments = extract_segments(left_owned)
        stage.rows_out = int(left_segments.count)

    with profiler.stage(
        "extract_right_segments",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=right_owned.row_count,
        detail="extract right edge segments for overlay candidate generation",
    ) as stage:
        right_segments = extract_segments(right_owned)
        stage.rows_out = int(right_segments.count)

    with profiler.stage(
        "filter_segment_candidates",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=int(left_segments.count * right_segments.count),
        detail="coarse-filter segment pairs by segment MBR overlap",
        metadata={"tile_size": tile_size},
    ) as stage:
        candidates = _generate_segment_candidates_from_tables(left_segments, right_segments, tile_size=tile_size)
        stage.rows_out = candidates.count
        stage.metadata["pairs_examined"] = candidates.pairs_examined

    with profiler.stage(
        "refine_intersections",
        category="refine",
        device=ExecutionMode.CPU,
        rows_in=candidates.count,
        detail="classify exact segment intersections on filtered pairs",
    ) as stage:
        precision_plan = select_precision_plan(
            runtime_selection=runtime_selection,
            kernel_class=KernelClass.CONSTRUCTIVE,
            requested=PrecisionMode.AUTO,
        )
        robustness_plan = select_robustness_plan(
            kernel_class=KernelClass.CONSTRUCTIVE,
            precision_plan=precision_plan,
        )
        result = _classify_segment_intersections_from_tables(
            left_segments=left_segments,
            right_segments=right_segments,
            pairs=candidates,
            runtime_selection=runtime_selection,
            precision_plan=precision_plan,
            robustness_plan=robustness_plan,
        )
        stage.rows_out = result.count
        stage.metadata["ambiguous_pairs"] = int(result.ambiguous_rows.size)
        stage.metadata["proper_pairs"] = int(np.count_nonzero(result.kinds == 1))
        stage.metadata["touch_pairs"] = int(np.count_nonzero(result.kinds == 2))
        stage.metadata["overlap_pairs"] = int(np.count_nonzero(result.kinds == 3))

    with profiler.stage(
        "sort_reconstruction_events",
        category="sort",
        device=ExecutionMode.CPU,
        rows_in=result.count,
        detail="stable-sort emitted intersection events for deterministic overlay reconstruction",
    ) as stage:
        if result.count:
            _ = np.lexsort(
                (
                    result.right_segments,
                    result.right_rows,
                    result.left_segments,
                    result.left_rows,
                    result.kinds,
                )
            )
        stage.rows_out = result.count

    plan = plan_overlay_reconstruction(normalized_operation)
    return profiler.finish(
        metadata={
            "rows": rows,
            "tile_size": tile_size,
            "operation": normalized_operation.value,
            "candidate_pairs": int(candidates.count),
            "pairs_examined": int(candidates.pairs_examined),
            "planner_selected_runtime": runtime_selection.selected.value,
            "plan_stages": [stage.name for stage in plan.stages],
            "plan_reason": plan.reason,
        }
    )


def profile_spatial_query_stack(
    *,
    rows: int,
    overlap_ratio: float = 0.2,
    predicate: str | None = "intersects",
    sort: bool = False,
    output_format: str = "indices",
    distance: float | np.ndarray | None = None,
    dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO,
    enable_nvtx: bool = False,
) -> ProfileTrace:
    runtime_selection = select_runtime(dispatch_mode)
    profiler = StageProfiler(
        operation="spatial_query",
        dataset=f"polygon-{rows}",
        requested_runtime=dispatch_mode,
        selected_runtime=runtime_selection.selected,
        enable_nvtx=enable_nvtx,
    )

    with profiler.stage(
        "build_inputs",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=rows,
        detail="generate synthetic tree and query polygon inputs",
        metadata={"overlap_ratio": overlap_ratio},
    ) as stage:
        tree_values, query_values = _build_join_inputs(rows, overlap_ratio=overlap_ratio)
        stage.rows_out = int(len(tree_values) + len(query_values))

    with profiler.stage(
        "build_tree_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=len(tree_values),
        detail="convert tree geometries into owned buffers",
    ) as stage:
        tree_owned = _to_owned(tree_values)
        stage.rows_out = tree_owned.row_count

    with profiler.stage(
        "build_flat_index",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=tree_owned.row_count,
        detail="compute tree bounds and build the flat spatial index",
    ) as stage:
        use_gpu = has_gpu_runtime()
        index_selection = RuntimeSelection(
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU if use_gpu else ExecutionMode.CPU,
            reason=(
                "repo-owned spatial index build (GPU morton sort)"
                if use_gpu
                else "repo-owned spatial index build (CPU)"
            ),
        )
        flat_index = build_flat_spatial_index(tree_owned, runtime_selection=index_selection)
        stage.device = index_selection.selected.value
        stage.rows_out = flat_index.size
        stage.metadata["planner_selected_runtime"] = index_selection.selected.value
        stage.metadata["index_reason"] = index_selection.reason
        stage.metadata["regular_grid_detected"] = flat_index.regular_grid is not None

    with profiler.stage(
        "normalize_query_input",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=len(query_values),
        detail="normalize query geometry input into the internal array form",
    ) as stage:
        normalized_query_values, scalar = _as_geometry_array(query_values)
        query_size = 0 if normalized_query_values is None else int(len(normalized_query_values))
        stage.rows_out = query_size
        stage.metadata["scalar"] = bool(scalar)

    if normalized_query_values is None:
        with profiler.stage(
            "format_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=0,
            detail="format the empty query result",
        ) as stage:
            _format_query_indices(
                np.asarray([], dtype=np.intp),
                tree_size=flat_index.size,
                query_size=1,
                scalar=True,
                sort=sort,
                output_format=output_format,
            )
            stage.rows_out = 0
        return profiler.finish(
            metadata={
                "rows": rows,
                "overlap_ratio": overlap_ratio,
                "predicate": predicate,
                "output_format": output_format,
                "matched_pairs": 0,
                "actual_selected_runtime": "cpu",
                "execution_implementation": "owned_cpu_spatial_query",
                "execution_reason": "empty geometry input",
                "selected_path": "empty_query",
            }
        )

    with profiler.stage(
        "probe_query_point_box",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=query_size,
        detail="check whether the point-tree box GPU query fast path applies",
        metadata={"predicate": predicate},
    ) as stage:
        point_box_bounds = _extract_box_query_bounds(predicate, normalized_query_values)
        point_box_pairs = _query_point_tree_box_index(
            tree_owned,
            predicate=predicate,
            query_row_count=query_size,
            box_bounds=point_box_bounds,
        )
        if point_box_pairs is not None:
            stage.device = ExecutionMode.GPU.value
            stage.rows_out = int(point_box_pairs[0].size)
            stage.metadata["fast_path_hit"] = True
        else:
            stage.rows_out = 0
            stage.metadata["fast_path_hit"] = False

    if point_box_pairs is not None:
        left_idx, right_idx = point_box_pairs
        indices = right_idx.astype(np.intp, copy=False) if scalar else np.vstack((
            left_idx.astype(np.intp, copy=False),
            right_idx.astype(np.intp, copy=False),
        ))
        with profiler.stage(
            "format_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=int(right_idx.size),
            detail="format the fast-path query result",
        ) as stage:
            _format_query_indices(
                indices,
                tree_size=flat_index.size,
                query_size=1 if scalar else query_size,
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            stage.rows_out = int(right_idx.size)
        return profiler.finish(
            metadata={
                "rows": rows,
                "overlap_ratio": overlap_ratio,
                "predicate": predicate,
                "output_format": output_format,
                "matched_pairs": int(right_idx.size),
                "actual_selected_runtime": "gpu",
                "execution_implementation": "owned_gpu_spatial_query",
                "execution_reason": "point-tree box query selected the repo-owned GPU candidate kernel",
                "selected_path": "point_box_query",
            }
        )

    with profiler.stage(
        "build_query_owned",
        category="setup",
        device=ExecutionMode.CPU,
        rows_in=query_size,
        detail="convert query geometries into owned buffers",
    ) as stage:
        query_owned = _to_owned(normalized_query_values)
        stage.rows_out = query_owned.row_count

    with profiler.stage(
        "query_regular_grid_rect_box",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=query_owned.row_count,
        detail="check the regular-grid rectangle GPU fast path",
        metadata={"predicate": predicate},
    ) as stage:
        regular_grid_box_bounds = _extract_box_query_bounds("intersects", normalized_query_values)
        regular_grid_box_pairs = _query_regular_grid_rect_box_index(
            flat_index,
            regular_grid_box_bounds,
            predicate=predicate,
        )
        if regular_grid_box_pairs is not None:
            stage.device = ExecutionMode.GPU.value
            if isinstance(regular_grid_box_pairs, _DeviceCandidates):
                stage.rows_out = regular_grid_box_pairs.total_pairs
            else:
                stage.rows_out = int(regular_grid_box_pairs[0].size)
            stage.metadata["fast_path_hit"] = True
        else:
            stage.rows_out = 0
            stage.metadata["fast_path_hit"] = False

    if regular_grid_box_pairs is not None:
        if isinstance(regular_grid_box_pairs, _DeviceCandidates):
            left_idx, right_idx = regular_grid_box_pairs.to_host()
        else:
            left_idx, right_idx = regular_grid_box_pairs
        indices = right_idx.astype(np.intp, copy=False) if scalar else np.vstack((
            left_idx.astype(np.intp, copy=False),
            right_idx.astype(np.intp, copy=False),
        ))
        with profiler.stage(
            "format_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=int(right_idx.size),
            detail="format the regular-grid rectangle fast-path result",
        ) as stage:
            _format_query_indices(
                indices,
                tree_size=flat_index.size,
                query_size=1 if scalar else query_owned.row_count,
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            stage.rows_out = int(right_idx.size)
        return profiler.finish(
            metadata={
                "rows": rows,
                "overlap_ratio": overlap_ratio,
                "predicate": predicate,
                "output_format": output_format,
                "matched_pairs": int(right_idx.size),
                "actual_selected_runtime": "gpu",
                "execution_implementation": "owned_gpu_spatial_query",
                "execution_reason": (
                    "repo-owned regular-grid rectangle box query executed on GPU with exact range expansion"
                ),
                "selected_path": "regular_grid_rect_box",
            }
        )

    with profiler.stage(
        "query_regular_grid_point",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=query_owned.row_count,
        detail="check the regular-grid point GPU fast path",
        metadata={"predicate": predicate},
    ) as stage:
        fast_pairs = _query_regular_grid_point_index(flat_index, query_owned, predicate=predicate)
        if fast_pairs is not None:
            stage.device = ExecutionMode.GPU.value
            if isinstance(fast_pairs, _DeviceCandidates):
                stage.rows_out = fast_pairs.total_pairs
            else:
                stage.rows_out = int(fast_pairs[0].size)
            stage.metadata["fast_path_hit"] = True
        else:
            stage.rows_out = 0
            stage.metadata["fast_path_hit"] = False

    if fast_pairs is not None:
        if isinstance(fast_pairs, _DeviceCandidates):
            left_idx, right_idx = fast_pairs.to_host()
        else:
            left_idx, right_idx = fast_pairs
        indices = right_idx.astype(np.intp, copy=False) if scalar else np.vstack((
            left_idx.astype(np.intp, copy=False),
            right_idx.astype(np.intp, copy=False),
        ))
        with profiler.stage(
            "format_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=int(right_idx.size),
            detail="format the regular-grid point fast-path result",
        ) as stage:
            _format_query_indices(
                indices,
                tree_size=flat_index.size,
                query_size=1 if scalar else query_owned.row_count,
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            stage.rows_out = int(right_idx.size)
        return profiler.finish(
            metadata={
                "rows": rows,
                "overlap_ratio": overlap_ratio,
                "predicate": predicate,
                "output_format": output_format,
                "matched_pairs": int(right_idx.size),
                "actual_selected_runtime": "gpu",
                "execution_implementation": "owned_gpu_spatial_query",
                "execution_reason": "regular-grid point query selected the repo-owned GPU candidate kernel",
                "selected_path": "regular_grid_point",
            }
        )

    with profiler.stage(
        "probe_owned_point_box",
        category="filter",
        device=ExecutionMode.CPU,
        rows_in=query_owned.row_count,
        detail="check the owned point-tree box GPU fast path",
        metadata={"predicate": predicate},
    ) as stage:
        owned_point_box_pairs = _query_point_tree_box_index(
            tree_owned,
            predicate=predicate,
            query_row_count=query_owned.row_count,
            box_bounds=_extract_box_query_bounds_from_owned(predicate, query_owned),
        )
        if owned_point_box_pairs is not None:
            stage.device = ExecutionMode.GPU.value
            stage.rows_out = int(owned_point_box_pairs[0].size)
            stage.metadata["fast_path_hit"] = True
        else:
            stage.rows_out = 0
            stage.metadata["fast_path_hit"] = False

    if owned_point_box_pairs is not None:
        left_idx, right_idx = owned_point_box_pairs
        indices = right_idx.astype(np.intp, copy=False) if scalar else np.vstack((
            left_idx.astype(np.intp, copy=False),
            right_idx.astype(np.intp, copy=False),
        ))
        with profiler.stage(
            "format_output",
            category="emit",
            device=ExecutionMode.CPU,
            rows_in=int(right_idx.size),
            detail="format the owned point-box fast-path result",
        ) as stage:
            _format_query_indices(
                indices,
                tree_size=flat_index.size,
                query_size=1 if scalar else query_owned.row_count,
                scalar=scalar,
                sort=sort,
                output_format=output_format,
            )
            stage.rows_out = int(right_idx.size)
        return profiler.finish(
            metadata={
                "rows": rows,
                "overlap_ratio": overlap_ratio,
                "predicate": predicate,
                "output_format": output_format,
                "matched_pairs": int(right_idx.size),
                "actual_selected_runtime": "gpu",
                "execution_implementation": "owned_gpu_spatial_query",
                "execution_reason": "point-tree box query selected the repo-owned GPU candidate kernel",
                "selected_path": "owned_point_box_query",
            }
        )

    with profiler.stage(
        "compute_query_bounds",
        category="filter",
        device=_gpu_bounds_dispatch_mode(query_owned),
        rows_in=query_owned.row_count,
        detail="compute query bounds for generic candidate generation",
    ) as stage:
        query_bounds = compute_geometry_bounds(query_owned, dispatch_mode=_gpu_bounds_dispatch_mode(query_owned))
        stage.rows_out = int(query_bounds.shape[0])

    tree_bounds = flat_index.bounds
    query_size = int(query_owned.row_count)
    actual_selected_runtime = "cpu"
    if predicate == "dwithin":
        with profiler.stage(
            "coarse_filter",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(query_bounds.shape[0] * tree_bounds.shape[0]),
            detail="generate distance-aware coarse pairs on CPU",
        ) as stage:
            if distance is None:
                raise ValueError("'distance' parameter is required for 'dwithin' predicate")
            if np.isscalar(distance):
                per_row_distance = np.full(query_size, float(distance), dtype=np.float64)
            else:
                per_row_distance = np.asarray(distance, dtype=np.float64)
            left_idx, right_idx = _generate_distance_pairs(query_bounds, tree_bounds, per_row_distance)
            stage.rows_out = int(left_idx.size)

        with profiler.stage(
            "refine_predicate",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(left_idx.size),
            detail="evaluate exact dwithin pairs",
        ) as stage:
            tree_values_host = np.asarray(tree_owned.to_shapely(), dtype=object)
            left_idx, right_idx, refine_selection = _filter_predicate_pairs(
                predicate,
                normalized_query_values,
                tree_values_host,
                left_idx,
                right_idx,
                distance=per_row_distance,
            )
            stage.rows_out = int(left_idx.size)
            stage.metadata["refine_selected_runtime"] = refine_selection.selected.value
            actual_selected_runtime = refine_selection.selected.value
    else:
        with profiler.stage(
            "coarse_filter",
            category="filter",
            device=ExecutionMode.CPU,
            rows_in=int(query_bounds.shape[0] * tree_bounds.shape[0]),
            detail="generate coarse candidate pairs from bounds overlap",
        ) as stage:
            gpu_candidates = _generate_candidates_gpu(query_bounds, tree_bounds)
            if gpu_candidates is not None:
                left_idx, right_idx = gpu_candidates
                stage.device = ExecutionMode.GPU.value
                stage.metadata["candidate_mode"] = "gpu"
                stage.rows_out = int(left_idx.size)
                actual_selected_runtime = ExecutionMode.GPU.value
            else:
                pairs = generate_bounds_pairs(query_owned, flat_index.geometry_array)
                left_idx = pairs.left_indices
                right_idx = pairs.right_indices
                stage.metadata["candidate_mode"] = "cpu"
                stage.metadata["pairs_examined"] = int(pairs.pairs_examined)
                stage.rows_out = int(pairs.count)

        with profiler.stage(
            "refine_predicate",
            category="refine",
            device=ExecutionMode.CPU,
            rows_in=int(left_idx.size),
            detail="evaluate exact predicate on coarse survivors",
            metadata={"predicate": predicate},
        ) as stage:
            left_idx, right_idx, refine_selection = _filter_predicate_pairs_owned(
                predicate,
                query_owned,
                tree_owned,
                left_idx,
                right_idx,
            )
            stage.rows_out = int(left_idx.size)
            stage.device = refine_selection.selected.value
            stage.metadata["refine_selected_runtime"] = refine_selection.selected.value
            if actual_selected_runtime != ExecutionMode.GPU.value:
                actual_selected_runtime = refine_selection.selected.value

    indices = right_idx.astype(np.intp, copy=False) if scalar else np.vstack((
        left_idx.astype(np.intp, copy=False),
        right_idx.astype(np.intp, copy=False),
    ))
    with profiler.stage(
        "format_output",
        category="emit",
        device=ExecutionMode.CPU,
        rows_in=int(right_idx.size),
        detail="format the query result into the requested output layout",
        metadata={"sort": sort, "output_format": output_format},
    ) as stage:
        _format_query_indices(
            indices,
            tree_size=flat_index.size,
            query_size=query_size,
            scalar=scalar,
            sort=sort,
            output_format=output_format,
        )
        stage.rows_out = int(right_idx.size)

    return profiler.finish(
        metadata={
            "rows": rows,
            "overlap_ratio": overlap_ratio,
            "predicate": predicate,
            "output_format": output_format,
            "matched_pairs": int(right_idx.size),
            "actual_selected_runtime": actual_selected_runtime,
            "execution_implementation": (
                "owned_gpu_spatial_query"
                if actual_selected_runtime == ExecutionMode.GPU.value
                else "owned_cpu_spatial_query"
            ),
            "execution_reason": "profiled full query_spatial_index stack",
            "selected_path": "generic_query_pipeline",
        }
    )
