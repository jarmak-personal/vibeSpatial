from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.runtime._runtime import ExecutionMode, has_gpu_runtime
from vibespatial.runtime.adaptive import plan_dispatch_selection
from vibespatial.runtime.config import (
    OVERLAY_GPU_REMAINDER_THRESHOLD,
    OVERLAY_PAIR_BATCH_THRESHOLD,
)
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event, strict_native_mode_enabled
from vibespatial.runtime.hotpath_trace import hotpath_stage, hotpath_trace_enabled
from vibespatial.runtime.precision import KernelClass
from vibespatial.spatial.indexing import generate_bounds_pairs
from vibespatial.spatial.query_types import DeviceSpatialJoinResult

logger = logging.getLogger(__name__)

_OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS = 262_144
_OVERLAY_FEW_RIGHT_GROUP_MAX = 64
_OVERLAY_FEW_RIGHT_GROUP_MIN_AVG = 8.0
_SHAPELY_TYPE_ID_POLYGON = 3
_SHAPELY_TYPE_ID_MULTIPOLYGON = 6
_SHAPELY_TYPE_ID_GEOMETRYCOLLECTION = 7


def _sync_hotpath() -> None:
    if hotpath_trace_enabled():
        from vibespatial.cuda._runtime import get_cuda_runtime

        get_cuda_runtime().synchronize()


def _geoseries_object_values(series: GeoSeries) -> np.ndarray:
    """Return a fast object array view of a GeoSeries-backed GeometryArray."""
    return np.asarray(series.array, dtype=object)


def _take_geoseries_object_values(series: GeoSeries, rows: np.ndarray) -> np.ndarray:
    """Materialize only the selected rows from a GeoSeries-backed GeometryArray."""
    return np.asarray(series.array.take(rows), dtype=object)


def _empty_owned_result_base(row_count: int, *, device: bool):
    """Build an all-null owned array for scatter assembly."""
    if device:
        try:
            import cupy as cp
        except ModuleNotFoundError:  # pragma: no cover
            device = False
        else:
            from vibespatial.geometry.owned import build_device_resident_owned

            return build_device_resident_owned(
                device_families={},
                row_count=row_count,
                tags=cp.full(row_count, -1, dtype=cp.int8),
                validity=cp.zeros(row_count, dtype=cp.bool_),
                family_row_offsets=cp.full(row_count, -1, dtype=cp.int32),
                execution_mode="gpu",
            )

    from vibespatial.geometry.owned import OwnedGeometryArray
    from vibespatial.runtime.residency import Residency

    return OwnedGeometryArray(
        validity=np.zeros(row_count, dtype=bool),
        tags=np.full(row_count, -1, dtype=np.int8),
        family_row_offsets=np.full(row_count, -1, dtype=np.int32),
        families={},
        residency=Residency.HOST,
    )


def _extract_owned_pair(df1, df2):
    """Return (left_owned, right_owned) if both DataFrames have owned backing, else (None, None)."""
    ga1 = df1.geometry.values
    ga2 = df2.geometry.values
    left_owned = getattr(ga1, '_owned', None)
    right_owned = getattr(ga2, '_owned', None)
    if left_owned is not None and right_owned is not None:
        return left_owned, right_owned
    if (
        has_gpu_runtime()
        and df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
        and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
        and (len(df1) * len(df2)) <= _OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS
    ):
        try:
            if left_owned is None:
                left_owned = ga1.to_owned()
            if right_owned is None:
                right_owned = ga2.to_owned()
        except (AttributeError, NotImplementedError):
            return None, None
        if left_owned is not None and right_owned is not None:
            return left_owned, right_owned
    return None, None


def _coerce_owned_pair_for_strict_overlay(df1, df2, left_owned, right_owned):
    """Materialize owned backing for strict overlay paths when GPU is available.

    Overlay does its spatial join before pairwise constructive work, so this
    coercion stays off the hot-path for non-strict runs. In strict mode we need
    the downstream pairwise overlay operations to use the repo-owned GPU
    dispatch instead of inheriting the generic small-workload crossover.
    """
    if not strict_native_mode_enabled() or not has_gpu_runtime():
        return left_owned, right_owned
    try:
        if left_owned is None:
            left_owned = df1.geometry.values.to_owned()
        if right_owned is None:
            right_owned = df2.geometry.values.to_owned()
    except (AttributeError, NotImplementedError):
        return left_owned, right_owned
    return left_owned, right_owned


# ---- Memory estimation for overlay difference GPU guard ----
#
# The overlay difference owned path gathers all right-side polygons by pair
# index, then runs segmented union + element-wise difference on GPU.  At
# large scale (100K+ rows) the gathered right array can exceed VRAM.
#
# Safety factor: the gathered array is only the first allocation.
# Segmented union and difference each need comparable working memory.
# Use 3x to account for the full pipeline.

# ---- Batched overlay difference constants ----
#
# When the number of spatial-join pairs exceeds what fits in VRAM, process
# groups of left geometries in batches.  Each batch gathers only its own
# right-side pairs, runs segmented union + pairwise difference, then frees
# intermediate memory before the next batch.
#
# VRAM_BUDGET_FRACTION: fraction of *free* device memory to use for a single
# batch's gathered right array.  Conservative because segmented_union and
# binary_constructive each allocate comparable working memory.
_VRAM_BUDGET_FRACTION = 0.3
# MIN_GROUPS_PER_BATCH: lower bound to avoid pathological per-group dispatch
# overhead when groups are very large.
_MIN_GROUPS_PER_BATCH = 64
# MAX_GROUPS_PER_BATCH: upper bound; larger batches reduce dispatch overhead
# but risk OOM on skewed group-size distributions.
_MAX_GROUPS_PER_BATCH = 10_000


def _estimate_bytes_per_pair(right_owned) -> int:
    """Estimate average device bytes consumed per gathered right-side pair.

    The dominant cost is the coordinate buffers (x, y as float64) plus
    offset arrays (geometry_offsets, part_offsets, ring_offsets as int32)
    and per-row metadata (validity, tags, family_row_offsets).

    We estimate from the *source* right_owned array: total family buffer
    bytes / row_count gives average bytes per row.  Since ``take`` gathers
    by duplicating rows, each pair costs approximately one row.
    """
    try:
        total_bytes = 0
        n_rows = max(right_owned.row_count, 1)
        # Per-row metadata overhead (validity: bool, tag: int8, fro: int32)
        total_bytes += 6 * n_rows  # 1 + 1 + 4
        for buf in right_owned.families.values():
            total_bytes += buf.x.nbytes + buf.y.nbytes
            total_bytes += buf.geometry_offsets.nbytes
            if buf.empty_mask is not None:
                total_bytes += buf.empty_mask.nbytes
            if hasattr(buf, 'part_offsets') and buf.part_offsets is not None:
                total_bytes += buf.part_offsets.nbytes
            if hasattr(buf, 'ring_offsets') and buf.ring_offsets is not None:
                total_bytes += buf.ring_offsets.nbytes
        return max(total_bytes // n_rows, 64)  # floor at 64 bytes
    except Exception:
        return 256  # conservative default


def _compute_batch_groups(
    h_group_offsets: np.ndarray,
    total_pairs: int,
    right_owned,
) -> int:
    """Determine how many groups to process per batch.

    Uses ``cupy.cuda.Device().mem_info`` to query free VRAM and
    ``_estimate_bytes_per_pair`` for per-pair cost.  Returns the number
    of groups per batch, clamped to [_MIN_GROUPS_PER_BATCH,
    _MAX_GROUPS_PER_BATCH].

    If VRAM query fails or the total pair count is below
    OVERLAY_PAIR_BATCH_THRESHOLD, returns n_groups (process everything in one batch).
    """
    n_groups = len(h_group_offsets) - 1
    if total_pairs < OVERLAY_PAIR_BATCH_THRESHOLD:
        return n_groups  # single-batch fast path

    try:
        import cupy
        free_bytes, _total = cupy.cuda.Device().mem_info
    except Exception:
        return n_groups  # cannot query VRAM; single-batch fallback

    bytes_per_pair = _estimate_bytes_per_pair(right_owned)
    # Budget: fraction of free VRAM for the gathered right array
    budget_bytes = int(free_bytes * _VRAM_BUDGET_FRACTION)
    if budget_bytes <= 0:
        budget_bytes = 256 * 1024 * 1024  # 256 MiB absolute floor

    max_pairs_per_batch = max(budget_bytes // bytes_per_pair, 1)

    # Average pairs per group to convert pair budget -> group budget
    avg_pairs = max(total_pairs / max(n_groups, 1), 1.0)
    groups_per_batch = int(max_pairs_per_batch / avg_pairs)
    groups_per_batch = max(groups_per_batch, _MIN_GROUPS_PER_BATCH)
    groups_per_batch = min(groups_per_batch, _MAX_GROUPS_PER_BATCH)

    # If the computed batch already covers all groups, use single batch
    if groups_per_batch >= n_groups:
        return n_groups

    return groups_per_batch


def _group_source_rows_from_offsets(group_offsets: np.ndarray) -> np.ndarray:
    """Expand CSR-style group offsets into one logical group id per right row."""
    offsets = np.asarray(group_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("group_offsets must be a 1D array with length >= 1")
    counts = np.diff(offsets)
    if np.any(counts < 0):
        raise ValueError("group_offsets must be monotonically nondecreasing")
    if counts.size == 0:
        return np.empty(0, dtype=np.int32)
    return np.repeat(
        np.arange(counts.size, dtype=np.int32),
        counts.astype(np.int64, copy=False),
    )


def _sequential_grouped_difference_owned(
    left_batch,
    right_batch,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
):
    """Compute grouped exact difference via repeated pairwise exact differences."""
    from vibespatial.constructive.binary_constructive import (
        binary_constructive_owned,
    )
    with hotpath_stage("overlay.diff.group_metadata", category="setup"):
        group_offsets = np.asarray(group_offsets, dtype=np.int64)
        group_lengths = np.diff(group_offsets).astype(np.int64, copy=False)
        max_group_size = int(group_lengths.max(initial=0))
    if max_group_size <= 0:
        return left_batch

    group_starts = group_offsets[:-1].astype(np.int64, copy=False)
    all_rows = np.arange(left_batch.row_count, dtype=np.int64)
    current_owned = left_batch

    for step in range(max_group_size):
        with hotpath_stage("overlay.diff.exact.active_rows", category="filter"):
            active_rows = all_rows[
                (group_lengths > step)
                & np.asarray(current_owned.validity, dtype=bool)
            ]
        if active_rows.size == 0:
            break
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.left_take", category="refine"):
            active_left = current_owned.take(active_rows)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.right_take", category="refine"):
            right_step = right_batch.take(group_starts[active_rows] + step)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.exact.binary_difference", category="refine"):
            active_diff = binary_constructive_owned(
                "difference",
                active_left,
                right_step,
                dispatch_mode=dispatch_mode,
                _prefer_rowwise_polygon_difference_overlay=True,
            )
        _sync_hotpath()
        if active_rows.size == current_owned.row_count:
            current_owned = active_diff
        else:
            from vibespatial.geometry.owned import concat_owned_scatter

            _sync_hotpath()
            with hotpath_stage("overlay.diff.exact.scatter", category="refine"):
                current_owned = concat_owned_scatter(
                    current_owned,
                    active_diff,
                    active_rows,
                )
            _sync_hotpath()

    return current_owned


def _grouped_overlay_difference_owned(
    left_batch,
    right_batch,
    group_offsets,
    *,
    dispatch_mode: ExecutionMode,
):
    """Compute grouped exact difference from one grouped overlay execution plan.

    The grouped workload shape is:
    - one left geometry row per group
    - many right geometry rows packed together

    The overlay planner already supports logical row isolation. By remapping
    every right geometry row to its owning left-group id, the existing
    same-row split, graph, and face-labeling pipeline becomes a true grouped
    exact-difference executor without per-pair replanning.
    """
    from vibespatial.overlay.gpu import (
        _build_overlay_execution_plan,
        _materialize_overlay_execution_plan,
    )

    with hotpath_stage("overlay.diff.group_metadata", category="setup"):
        group_offsets = np.asarray(group_offsets, dtype=np.int64)
        group_lengths = np.diff(group_offsets).astype(np.int64, copy=False)
        max_group_size = int(group_lengths.max(initial=0))
    if max_group_size <= 0:
        return left_batch

    try:
        with hotpath_stage("overlay.diff.group_rows.expand", category="setup"):
            right_group_rows = _group_source_rows_from_offsets(group_offsets)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.grouped_plan.build", category="setup"):
            plan = _build_overlay_execution_plan(
                left_batch,
                right_batch,
                dispatch_mode=dispatch_mode,
                _cached_right_segments=None,
                _row_isolated=True,
                _right_geometry_source_rows=right_group_rows,
            )
        _sync_hotpath()
        with hotpath_stage("overlay.diff.grouped_plan.materialize", category="refine"):
            diff_owned, _selected = _materialize_overlay_execution_plan(
                plan,
                operation="difference",
                requested=ExecutionMode.GPU,
                preserve_row_count=left_batch.row_count,
            )
        _sync_hotpath()
        if diff_owned.row_count != left_batch.row_count:
            raise RuntimeError(
                "grouped overlay difference produced "
                f"{diff_owned.row_count} rows for {left_batch.row_count} groups"
            )
        if strict_native_mode_enabled() and _selected is not ExecutionMode.GPU:
            record_fallback_event(
                surface="geopandas.array.difference",
                reason="grouped exact overlay difference materialized off GPU",
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}"
                ),
                requested=ExecutionMode.GPU,
                selected=_selected,
                pipeline="overlay",
                d2h_transfer=_selected is ExecutionMode.CPU,
            )
        return diff_owned
    except Exception:
        logger.debug(
            "grouped exact overlay plan failed; falling back to sequential exact difference",
            exc_info=True,
        )
        return _sequential_grouped_difference_owned(
            left_batch,
            right_batch,
            group_offsets,
            dispatch_mode=dispatch_mode,
        )


def _batched_overlay_difference_owned(
    left_owned,
    right_owned,
    idx1,
    idx2,
    d_idx1,
    d_idx2,
    _has_device_indices: bool,
    _pairwise_mode,
):
    """Process overlay difference in VRAM-safe batches.

    Splits the unique left indices into batches, and for each batch:
      1. Gathers the right-side pairs for that batch of groups
      2. Runs segmented union on the gathered right geometries
      3. Runs pairwise difference (left_sub - right_unions)
      4. Frees intermediates before the next batch

    Returns (diff_owned, idx1_unique) ready for concat_owned_scatter.
    """
    from vibespatial.constructive.segmented_union_host import (
        concat_owned_arrays,
    )

    xp = np
    if hasattr(idx1, "__cuda_array_interface__"):
        try:
            import cupy
            xp = cupy
        except ImportError:
            pass

    # --- Compute group structure (unique left indices + group offsets) ---
    with hotpath_stage("overlay.diff.group_index_build", category="setup"):
        idx1_unique, idx1_split_at = xp.unique(idx1, return_index=True)
        group_offsets_full = xp.concatenate(
            [idx1_split_at, xp.asarray([len(idx2)])]
        )

    # Bring group structure to host for batch slicing.  These are small
    # metadata arrays (n_groups + 1 elements), so the D->H cost is trivial.
    h_group_offsets = np.asarray(group_offsets_full)
    h_idx1_unique = np.asarray(idx1_unique)
    n_groups = len(h_idx1_unique)
    total_pairs = int(h_group_offsets[-1])

    # Decide batch size
    groups_per_batch = _compute_batch_groups(
        h_group_offsets, total_pairs, right_owned,
    )

    # --- Single-batch fast path (original code, no overhead) ---
    if groups_per_batch >= n_groups:
        _sync_hotpath()
        with hotpath_stage("overlay.diff.single_batch.right_gather", category="refine"):
            if _has_device_indices:
                right_gathered = right_owned.device_take(d_idx2)
            else:
                right_gathered = right_owned.take(idx2)

        _sync_hotpath()
        with hotpath_stage("overlay.diff.single_batch.left_take", category="refine"):
            left_sub = left_owned.take(idx1_unique)
        _sync_hotpath()
        with hotpath_stage("overlay.diff.single_batch.grouped_difference", category="refine"):
            diff_owned = _grouped_overlay_difference_owned(
                left_sub,
                right_gathered,
                h_group_offsets,
                dispatch_mode=_pairwise_mode,
            )
        _sync_hotpath()
        return diff_owned, idx1_unique

    # --- Multi-batch path ---
    logger.info(
        "overlay difference: batching %d groups into batches of %d "
        "(total_pairs=%d, budget_frac=%.1f%%)",
        n_groups, groups_per_batch, total_pairs,
        _VRAM_BUDGET_FRACTION * 100,
    )

    # We need host-side idx2 for slicing.  If indices are on device,
    # bring idx2 to host (one transfer for the full array — unavoidable
    # for per-batch slicing, but we slice *groups* not individual pairs).
    h_idx2 = np.asarray(idx2)

    batch_results = []
    batch_unique_indices = []

    for batch_start in range(0, n_groups, groups_per_batch):
        batch_end = min(batch_start + groups_per_batch, n_groups)

        # Pair range for this batch of groups
        pair_start = int(h_group_offsets[batch_start])
        pair_end = int(h_group_offsets[batch_end])
        batch_n_pairs = pair_end - pair_start

        if batch_n_pairs == 0:
            continue

        # Free cached pool memory before each batch
        from vibespatial.cuda._runtime import maybe_trim_pool_memory

        maybe_trim_pool_memory()

        # Gather right geometries for this batch only
        batch_idx2 = h_idx2[pair_start:pair_end]
        _sync_hotpath()
        with hotpath_stage("overlay.diff.batch.right_gather", category="refine"):
            right_gathered = right_owned.take(batch_idx2)

        # Build local group offsets for this batch (0-based)
        batch_group_starts = h_group_offsets[batch_start:batch_end + 1] - pair_start
        batch_group_offsets = np.asarray(batch_group_starts, dtype=np.int64)

        # Take the corresponding left geometries
        batch_left_indices = h_idx1_unique[batch_start:batch_end]
        _sync_hotpath()
        with hotpath_stage("overlay.diff.batch.left_take", category="refine"):
            left_sub = left_owned.take(batch_left_indices)

        _sync_hotpath()
        with hotpath_stage("overlay.diff.batch.grouped_difference", category="refine"):
            batch_diff = _grouped_overlay_difference_owned(
                left_sub,
                right_gathered,
                batch_group_offsets,
                dispatch_mode=_pairwise_mode,
            )
        _sync_hotpath()
        del right_gathered, left_sub  # free intermediates

        batch_results.append(batch_diff)
        batch_unique_indices.append(batch_left_indices)

    # Concatenate all batch results
    if len(batch_results) == 0:
        # Edge case: all batches were empty (shouldn't happen, but be safe)
        from vibespatial.geometry.owned import from_shapely_geometries
        diff_owned = from_shapely_geometries([])
        all_unique = np.array([], dtype=np.int64)
    elif len(batch_results) == 1:
        diff_owned = batch_results[0]
        all_unique = batch_unique_indices[0]
    else:
        _sync_hotpath()
        with hotpath_stage("overlay.diff.batch.concat", category="refine"):
            diff_owned = concat_owned_arrays(batch_results)
        _sync_hotpath()
        all_unique = np.concatenate(batch_unique_indices)

    # Return in the same format as idx1_unique (host numpy)
    return diff_owned, all_unique


def _make_valid_geoseries(gs):
    """Apply make_valid to polygon rows of a GeoSeries, preferring GPU path.

    When the GeoSeries has owned backing, routes through make_valid_owned to
    keep data device-resident and avoid Shapely materialisation.  Falls back
    to the standard GeoSeries.make_valid() path otherwise.
    """
    ga = gs.values
    owned = getattr(ga, '_owned', None)
    poly_ix = gs.geom_type.isin(POLYGON_GEOM_TYPES)
    if not poly_ix.any():
        return gs

    if owned is not None:
        from vibespatial.runtime.residency import Residency

        if owned.residency is not Residency.DEVICE:
            gs = gs.copy()
            gs.loc[poly_ix] = gs[poly_ix].make_valid()
            try:
                from vibespatial.geometry.owned import from_shapely_geometries

                new_owned = from_shapely_geometries(
                    np.asarray(gs.array, dtype=object),
                    residency=Residency.HOST,
                )
                new_ga = GeometryArray.from_owned(new_owned, crs=ga.crs)
                return GeoSeries(new_ga, index=gs.index)
            except NotImplementedError:
                return gs

        from vibespatial.constructive.make_valid_pipeline import make_valid_owned

        mv_result = make_valid_owned(owned=owned)
        if mv_result.repaired_rows.size > 0:
            # Repair happened — prefer device-resident .owned to avoid D->H.
            if mv_result.owned is not None:
                try:
                    new_ga = GeometryArray.from_owned(mv_result.owned, crs=ga.crs)
                    return GeoSeries(new_ga, index=gs.index)
                except (NotImplementedError, Exception):
                    pass  # fall through to Shapely materialization
            # Fallback: rebuild from Shapely geometries
            try:
                from vibespatial.geometry.owned import from_shapely_geometries

                new_owned = from_shapely_geometries(list(mv_result.geometries))
                new_ga = GeometryArray.from_owned(new_owned, crs=ga.crs)
            except NotImplementedError:
                new_ga = GeometryArray(mv_result.geometries, crs=ga.crs)
            return GeoSeries(new_ga, index=gs.index)
        # All rows already valid — owned backing preserved, return as-is.
        return gs

    # Shapely fallback path: no owned backing available.
    gs = gs.copy()
    gs.loc[poly_ix] = gs[poly_ix].make_valid()
    return gs


def _ensure_geometry_column(df):
    """Ensure that the geometry column is called 'geometry'.

    If another column with that name exists, it will be dropped.
    """
    if not df._geometry_column_name == "geometry":
        if PANDAS_GE_30:
            if "geometry" in df.columns:
                df = df.drop("geometry", axis=1)
            df = df.rename_geometry("geometry")
        else:
            if "geometry" in df.columns:
                df.drop("geometry", axis=1, inplace=True)
            df.rename_geometry("geometry", inplace=True)
    return df


def _intersecting_index_pairs(df1, df2, *, left_owned=None, right_owned=None):
    # ADR-0036 boundary: produces spatial index arrays only.
    # sindex.query has its own owned-dispatch path (sindex.py lines 334-378)
    # that routes through query_spatial_index when both sides support owned.
    #
    # Phase 2 zero-copy: when both DataFrames have owned (device-resident)
    # backing, request device-resident index arrays from the spatial index
    # to eliminate the D->H->D round-trip when downstream take() re-uploads.
    # Returns DeviceSpatialJoinResult when device arrays are available,
    # otherwise returns the standard (2, n) numpy array or (idx1, idx2) tuple.
    if (
        left_owned is not None
        and right_owned is not None
        and df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
        and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
        and (left_owned.row_count * right_owned.row_count) <= _OVERLAY_BBOX_PAIR_FAST_PATH_MAX_PAIRS
    ):
        candidate_pairs = generate_bounds_pairs(left_owned, right_owned)
        left_idx = np.asarray(candidate_pairs.left_indices, dtype=np.int32)
        right_idx = np.asarray(candidate_pairs.right_indices, dtype=np.int32)
        if left_idx.size > 0:
            order = np.lexsort((right_idx, left_idx))
            left_idx = left_idx[order]
            right_idx = right_idx[order]
        record_dispatch_event(
            surface="geopandas.overlay.sindex",
            operation="intersects",
            implementation="gpu_bbox_pairs_fast_path",
            reason=(
                "owned polygon overlay used direct bbox candidate pairs "
                f"for {left_owned.row_count}x{right_owned.row_count} rows"
            ),
        )
        return left_idx, right_idx

    request_device = (
        left_owned is not None
        and right_owned is not None
    )
    result = df2.sindex.query(
        df1.geometry,
        predicate="intersects",
        sort=True,
        return_device=request_device,
    )
    # DeviceSpatialJoinResult flows through directly to the caller.
    if isinstance(result, DeviceSpatialJoinResult):
        return result
    return result


def _reverse_intersecting_index_pairs(index_result):
    """Derive the reverse intersects pair mapping from a forward query result."""
    if isinstance(index_result, DeviceSpatialJoinResult):
        import cupy as cp

        d_left = index_result.d_right_idx.astype(cp.int32, copy=False)
        d_right = index_result.d_left_idx.astype(cp.int32, copy=False)
        if d_left.size > 0:
            order = cp.lexsort(
                cp.stack([
                    cp.asarray(d_right, dtype=cp.int64),
                    cp.asarray(d_left, dtype=cp.int64),
                ])
            )
            d_left = d_left[order]
            d_right = d_right[order]
        return DeviceSpatialJoinResult(d_left_idx=d_left, d_right_idx=d_right)

    if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
        idx1, idx2 = index_result
    else:
        idx1, idx2 = index_result

    left = np.asarray(idx2)
    right = np.asarray(idx1)
    if left.size > 0:
        order = np.lexsort((right, left))
        left = left[order]
        right = right[order]
    return left, right


def _assemble_intersection_attributes(idx1, idx2, df1, df2):
    """ADR-0036 boundary: attribute assembly from index arrays.

    Receives integer index arrays and attribute-only DataFrames (geometry
    columns already dropped).  Returns a merged DataFrame with attributes
    from both sides joined via the spatial index pairs.

    Indices may be CuPy arrays (Phase 3) — materialized to host here since
    pandas DataFrames are inherently host-side.
    """
    h_idx1 = idx1.get() if hasattr(idx1, "get") else idx1
    h_idx2 = idx2.get() if hasattr(idx2, "get") else idx2
    pairs = pd.DataFrame({"__idx1": h_idx1, "__idx2": h_idx2})
    result = pairs.merge(
        df1,
        left_on="__idx1",
        right_index=True,
    )
    result = result.merge(
        df2,
        left_on="__idx2",
        right_index=True,
        suffixes=("_1", "_2"),
    )
    return result


def _assemble_polygon_intersection_rows_with_lower_dim(
    left_pairs: GeoSeries,
    right_pairs: GeoSeries,
    area_pairs: GeoSeries,
) -> GeoSeries:
    """Recover lower-dimensional polygon intersection remnants at the boundary.

    The polygon constructive intersection path returns only polygonal area.
    Public overlay semantics also need line/point remnants when polygon pairs
    touch without area overlap, and GeometryCollections when polygonal area has
    additional lower-dimensional pieces.
    """
    area_geoms = np.asarray(area_pairs, dtype=object)
    left_geoms = np.asarray(left_pairs, dtype=object)
    right_geoms = np.asarray(right_pairs, dtype=object)
    boundary_geoms = np.asarray(
        shapely.intersection(
            shapely.boundary(left_geoms),
            shapely.boundary(right_geoms),
        ),
        dtype=object,
    )

    assembled = np.empty(len(area_geoms), dtype=object)
    for row_index in range(len(area_geoms)):
        area_geom = area_geoms[row_index]
        if area_geom is not None and area_geom.is_empty:
            area_geom = None
        elif area_geom is not None and getattr(area_geom, "area", 0.0) == 0.0:
            area_geom = None

        edge_geom = boundary_geoms[row_index]
        if edge_geom is not None and edge_geom.is_empty:
            edge_geom = None

        if area_geom is not None and edge_geom is not None:
            edge_geom = edge_geom.difference(area_geom.boundary)
            if edge_geom is not None and edge_geom.is_empty:
                edge_geom = None

        if edge_geom is not None:
            edge_parts = [
                part
                for part in shapely.get_parts(np.asarray([edge_geom], dtype=object))
                if not part.is_empty
            ]
            if edge_parts:
                unique_parts = []
                seen_parts = set()
                for part in edge_parts:
                    normalized = shapely.normalize(part)
                    key = normalized.wkb
                    if key in seen_parts:
                        continue
                    seen_parts.add(key)
                    unique_parts.append(normalized)

                if len(unique_parts) == 1:
                    edge_geom = unique_parts[0]
                else:
                    edge_part_types = {part.geom_type for part in unique_parts}
                    merged_edges = shapely.union_all(np.asarray(unique_parts, dtype=object))
                    if edge_part_types <= {"LineString", "LinearRing", "MultiLineString"}:
                        edge_geom = shapely.line_merge(merged_edges)
                    else:
                        edge_geom = merged_edges
                if edge_geom is not None and edge_geom.is_empty:
                    edge_geom = None

        if area_geom is None and edge_geom is None:
            assembled[row_index] = None
            continue
        if area_geom is None:
            assembled[row_index] = edge_geom
            continue
        if edge_geom is None:
            assembled[row_index] = area_geom
            continue

        parts = [area_geom]
        parts.extend(
            part
            for part in shapely.get_parts(np.asarray([edge_geom], dtype=object))
            if not part.is_empty
        )
        assembled[row_index] = GeometryCollection(parts)

    return GeoSeries(assembled, index=area_pairs.index, crs=area_pairs.crs)


def _count_non_polygon_collection_parts(geometries: np.ndarray) -> int:
    """Count dropped non-polygon parts for GeometryCollection warning parity."""
    if len(geometries) == 0:
        return 0
    parts = shapely.get_parts(geometries)
    if len(parts) == 0:
        return 0
    part_type_ids = shapely.get_type_id(parts)
    non_empty_mask = ~shapely.is_empty(parts)
    polygon_mask = (
        (part_type_ids == _SHAPELY_TYPE_ID_POLYGON)
        | (part_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
    )
    return int(np.count_nonzero(non_empty_mask & ~polygon_mask))


def _strip_non_polygon_collection_parts(geometries: np.ndarray) -> np.ndarray:
    """Replace GeometryCollections with polygon-only equivalents."""
    if len(geometries) == 0:
        return geometries

    type_ids = shapely.get_type_id(geometries)
    collection_rows = np.flatnonzero(type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION)
    if collection_rows.size == 0:
        return geometries

    cleaned = geometries.copy()
    parts, parent_index = shapely.get_parts(
        geometries[collection_rows],
        return_index=True,
    )
    if len(parts) == 0:
        cleaned[collection_rows] = None
        return cleaned

    part_type_ids = shapely.get_type_id(parts)
    non_empty_mask = ~shapely.is_empty(parts)
    polygon_mask = (
        (part_type_ids == _SHAPELY_TYPE_ID_POLYGON)
        | (part_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
    )
    keep_parts = parts[non_empty_mask & polygon_mask]
    keep_parent = np.asarray(parent_index[non_empty_mask & polygon_mask], dtype=np.int32)
    if keep_parts.size == 0:
        cleaned[collection_rows] = None
        return cleaned

    unique_parent, split_at = np.unique(keep_parent, return_index=True)
    grouped_parts = np.split(keep_parts, split_at[1:])
    for local_row, polygon_parts in zip(unique_parent, grouped_parts, strict=True):
        row_index = int(collection_rows[int(local_row)])
        if len(polygon_parts) == 1:
            cleaned[row_index] = polygon_parts[0]
        else:
            cleaned[row_index] = shapely.union_all(polygon_parts)

    dropped_parent = np.setdiff1d(
        np.arange(collection_rows.size, dtype=np.int32),
        unique_parent,
        assume_unique=True,
    )
    if dropped_parent.size > 0:
        cleaned[collection_rows[dropped_parent]] = None
    return cleaned


def _filter_polygon_intersection_rows_for_keep_geom_type(
    left_pairs: GeoSeries,
    right_pairs: GeoSeries,
    area_pairs: GeoSeries,
    *,
    keep_geom_type_warning: bool,
) -> tuple[GeoSeries, int, np.ndarray]:
    """Keep polygonal area rows only and classify dropped lower-dimensional remnants.

    This is the fast path for ``keep_geom_type=True`` and the default
    ``keep_geom_type=None`` overlay contract. We already have the exact polygonal
    area intersection result; the remaining work is:

    - keep only polygon rows for the returned GeoDataFrame
    - optionally count dropped lower-dimensional rows/pieces for the warning

    Unlike ``_assemble_polygon_intersection_rows_with_lower_dim``, this path
    does not materialize the dropped line/point geometries unless they are
    needed to count GeometryCollection extras for the warning.
    """
    area_owned = getattr(area_pairs.values, "_owned", None)
    if area_owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        tags = area_owned.tags
        validity = area_owned.validity
        row_offsets = area_owned.family_row_offsets
        keep_mask = np.zeros(len(tags), dtype=bool)
        owned_metadata_consistent = True

        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            family_tag = FAMILY_TAGS[family]
            family_mask = validity & (tags == family_tag)
            if not family_mask.any():
                continue
            family_rows = row_offsets[family_mask]
            empty_mask = np.asarray(area_owned.families[family].empty_mask, dtype=bool)
            if empty_mask.size == 0 or np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
                owned_metadata_consistent = False
                break
            keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

        if owned_metadata_consistent:
            dropped = 0
            if keep_geom_type_warning and len(tags) > 0:
                empty_rows = np.flatnonzero(~keep_mask)
                if empty_rows.size > 0:
                    dropped += int(
                        np.count_nonzero(
                            np.asarray(
                                left_pairs.take(empty_rows).intersects(right_pairs.take(empty_rows)),
                                dtype=bool,
                            )
                        )
                    )

            filtered = area_pairs.take(np.flatnonzero(keep_mask)).reset_index(drop=True)
            return filtered, dropped, keep_mask

    area_values = _geoseries_object_values(area_pairs)
    present_mask = ~(shapely.is_missing(area_values) | shapely.is_empty(area_values))
    keep_mask = present_mask & (shapely.area(area_values) > 0.0)

    dropped = 0
    if keep_geom_type_warning and len(area_values) > 0:
        empty_rows = np.flatnonzero(~keep_mask)
        if empty_rows.size > 0:
            left_values = _take_geoseries_object_values(left_pairs, empty_rows)
            right_values = _take_geoseries_object_values(right_pairs, empty_rows)
            dropped += int(
                np.count_nonzero(
                    shapely.intersects(left_values, right_values)
                )
            )

    filtered_values = area_values[keep_mask].copy()
    if filtered_values.size > 0:
        filtered_values = _strip_non_polygon_collection_parts(filtered_values)
    filtered = GeoSeries(filtered_values, crs=area_pairs.crs)
    return filtered, dropped, keep_mask


def _needs_host_overlay_difference_boundary_rebuild(df1, df2) -> bool:
    """Return True when public overlay difference needs boundary reconstruction.

    The polygon-polygon owned path preserves the overlay face topology we need.
    Mixed-dimensional overlay differences still need GeoPandas/GEOS boundary
    semantics at the public API boundary: polygon boundaries noded by linework,
    lines split into outside pieces, and exact lower-dimensional remnants.
    """
    left_types = set(df1.geometry.geom_type.dropna())
    right_types = set(df2.geometry.geom_type.dropna())
    if not left_types or not right_types:
        return False
    return (
        not left_types.issubset(set(POLYGON_GEOM_TYPES))
        or not right_types.issubset(set(POLYGON_GEOM_TYPES))
    )


def _grouped_overlay_difference_geoms(df1, df2, idx1, idx2) -> np.ndarray:
    """Vectorized grouped Shapely difference for overlay boundary assembly."""
    left_geoms = np.asarray(df1.geometry, dtype=object)
    result_geoms = left_geoms.copy()

    if idx1.size == 0:
        return result_geoms

    h_idx1 = idx1.get() if hasattr(idx1, "get") else idx1
    h_idx2 = idx2.get() if hasattr(idx2, "get") else idx2

    right_geoms = np.asarray(df2.geometry, dtype=object)
    right_unions = np.empty(len(df1), dtype=object)
    right_unions.fill(None)

    idx1_unique, idx1_split_at = np.unique(h_idx1, return_index=True)
    idx2_groups = np.split(h_idx2, idx1_split_at[1:])
    for left_pos, neighbors_idx in zip(idx1_unique, idx2_groups, strict=True):
        neighbors = right_geoms[neighbors_idx]
        if len(neighbors) == 1:
            right_unions[left_pos] = neighbors[0]
        else:
            right_unions[left_pos] = shapely.union_all(neighbors)

    has_neighbors = np.zeros(len(df1), dtype=bool)
    has_neighbors[idx1_unique] = True
    result_geoms[has_neighbors] = shapely.difference(
        left_geoms[has_neighbors],
        right_unions[has_neighbors],
    )
    return result_geoms


def _many_vs_one_intersection_owned(
    left_sub,
    right_owned,
    unique_right_idx,
    *,
    _has_device_indices=False,
    d_idx2=None,
):
    """Optimized many-vs-one intersection using containment bypass + SH batch clip.

    When many left geometries are paired against a single right geometry (the
    many-vs-one pattern), the three-tier strategy from ``spatial_overlay_owned``
    dramatically reduces work:

    1. **Containment bypass**: left polygons fully inside the right polygon
       are returned as-is with zero overlay computation.
    2. **SH batch clip**: simple boundary-crossing polygons are clipped in a
       single batched GPU kernel launch.
    3. **Element-wise overlay**: only the remaining complex polygons go through
       the full ``binary_constructive_owned`` pipeline.

    Parameters
    ----------
    left_sub : OwnedGeometryArray
        Left geometries gathered by spatial-join index pairs (N rows).
    right_owned : OwnedGeometryArray
        Original right geometry array (from the input GeoDataFrame).
    unique_right_idx : int
        The single right-side row index that all pairs map to.

    Returns
    -------
    OwnedGeometryArray
        Result geometries in the same row order as *left_sub* (1:1 with the
        spatial-join pairs).  Contained rows are pass-through; remainder rows
        are clipped or overlaid.
    """
    from vibespatial.constructive.binary_constructive import (
        binary_constructive_owned,
    )

    index_oga_pairs, complex_left, complex_global_positions, right_one, _pairwise_mode = (
        _prepare_many_vs_one_intersection_chunks(
            left_sub,
            right_owned,
            unique_right_idx,
            global_positions=np.arange(left_sub.row_count, dtype=np.intp),
        )
    )

    # ---- Tier 3: Overlay for remaining boundary-crossing polygons ----
    # After containment bypass and SH batch clip, only a small number of
    # complex polygons remain (those crossing the boundary with holes or
    # high vertex counts).  For this small remainder, vectorized Shapely
    # intersection is faster than launching the full GPU overlay pipeline
    # (which has high fixed cost per kernel invocation for segment
    # extraction, split-event classification, and face reconstruction).
    # The crossover to GPU overlay is beneficial when the remainder set
    # is large (>= 1000 rows) AND the clip polygon is moderately complex.
    def _shapely_remainder_intersection(left_rem_oga, right_one_oga):
        """Intersect remainder polygons via vectorized Shapely (CPU)."""
        from vibespatial.geometry.owned import from_shapely_geometries

        left_geoms = np.asarray(left_rem_oga.to_shapely(), dtype=object)
        right_geom = right_one_oga.to_shapely()[0]
        if right_geom is None or shapely.is_empty(right_geom):
            empty = from_shapely_geometries([shapely.Point()])
            return empty.take(np.asarray([], dtype=np.int64))
        right_arr = np.full(len(left_geoms), right_geom, dtype=object)
        raw = shapely.intersection(left_geoms, right_arr)
        record_fallback_event(
            surface="geopandas.overlay.intersection",
            reason=(
                "many-vs-one remainder: vectorized Shapely intersection "
                f"for {len(left_geoms)} boundary-crossing polygons"
            ),
            detail=f"rows={len(left_geoms)}",
            requested=_pairwise_mode,
            selected=ExecutionMode.CPU,
            pipeline="_many_vs_one_intersection_owned",
            d2h_transfer=True,
        )
        return from_shapely_geometries(list(raw))

    def _gpu_remainder_intersection(left_rem_oga, right_one_oga):
        """Intersect remainder polygons via GPU overlay pipeline."""
        from vibespatial.constructive.binary_constructive import (
            _dispatch_polygon_intersection_overlay_rowwise_gpu,
        )
        from vibespatial.geometry.owned import (
            materialize_broadcast,
            tile_single_row,
        )

        right_rep = materialize_broadcast(
            tile_single_row(right_one_oga, left_rem_oga.row_count),
        )
        rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            left_rem_oga,
            right_rep,
            dispatch_mode=_pairwise_mode,
        )
        if rowwise_result is not None and rowwise_result.row_count == left_rem_oga.row_count:
            return rowwise_result
        return binary_constructive_owned(
            "intersection", left_rem_oga, right_rep,
            dispatch_mode=_pairwise_mode,
        )

    def _remainder_intersection(left_rem_oga, right_one_oga):
        """Choose the remainder path based on residency and crossover shape."""
        from vibespatial.runtime.residency import Residency

        if not has_gpu_runtime():
            return _shapely_remainder_intersection(left_rem_oga, right_one_oga)
        if strict_native_mode_enabled() or left_rem_oga.residency is Residency.DEVICE:
            return _gpu_remainder_intersection(left_rem_oga, right_one_oga)
        if left_rem_oga.row_count < OVERLAY_GPU_REMAINDER_THRESHOLD:
            return _shapely_remainder_intersection(left_rem_oga, right_one_oga)
        return _gpu_remainder_intersection(left_rem_oga, right_one_oga)

    complex_result = None
    if complex_left is not None and complex_global_positions is not None:
        complex_result = _remainder_intersection(complex_left, right_one)

    # Release GPU pool memory after overlay on remainder polygons: SH clip
    # and per-pair GPU overlay produce large intermediates that are dead now.
    from vibespatial.cuda._runtime import maybe_trim_pool_memory

    maybe_trim_pool_memory()

    if complex_result is not None and complex_global_positions is not None:
        index_oga_pairs.append((complex_global_positions, complex_result))

    return _assemble_indexed_owned_chunks(index_oga_pairs, left_sub.row_count)


def _assemble_indexed_owned_chunks(index_oga_pairs, row_count: int):
    """Concatenate owned chunks and restore original row order."""
    from vibespatial.geometry.owned import OwnedGeometryArray, from_shapely_geometries

    if not index_oga_pairs:
        empty = from_shapely_geometries([shapely.Point()])
        return empty.take(np.asarray([], dtype=np.int64))

    if len(index_oga_pairs) == 1:
        indices, oga = index_oga_pairs[0]
        if len(indices) == row_count and np.array_equal(indices, np.arange(row_count)):
            return oga

    all_indices = np.concatenate([idx for idx, _ in index_oga_pairs])
    concat_result = OwnedGeometryArray.concat([oga for _, oga in index_oga_pairs])
    inverse_perm = np.empty(row_count, dtype=np.intp)
    inverse_perm[all_indices] = np.arange(len(all_indices), dtype=np.intp)
    return concat_result.take(inverse_perm)


def _prepare_many_vs_one_intersection_chunks(
    left_sub,
    right_owned,
    unique_right_idx,
    *,
    global_positions: np.ndarray,
):
    """Prepare contained/SH chunks and return any complex remainder batch."""
    from vibespatial.runtime.residency import Residency

    n_pairs = left_sub.row_count
    _pairwise_mode = plan_dispatch_selection(
        kernel_name="overlay_pairwise",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n_pairs,
    )
    _pairwise_mode = (
        ExecutionMode.GPU
        if (
            strict_native_mode_enabled()
            or left_sub.residency is Residency.DEVICE
        )
        else _pairwise_mode.selected
    )

    right_one = right_owned.take(np.array([unique_right_idx], dtype=np.intp))

    _contained_result = None
    _remainder_mask = None

    try:
        import cupy as _cp_local
    except ModuleNotFoundError:
        _cp_local = None

    if _cp_local is not None:
        try:
            from vibespatial.overlay.bypass import _containment_bypass_gpu

            _contained_result, _remainder_mask = _containment_bypass_gpu(
                left_sub, right_one, "intersection",
            )
        except Exception:
            _contained_result = None
            _remainder_mask = None

    if _contained_result is not None and _remainder_mask is None:
        record_dispatch_event(
            surface="geopandas.overlay.intersection",
            operation="many_vs_one_containment_bypass",
            implementation="gpu_containment_bypass_all",
            reason=(
                f"many-vs-one: all {n_pairs} polygons fully inside "
                "right polygon (containment bypass)"
            ),
        )
        return [(global_positions, _contained_result)], None, None, right_one, _pairwise_mode

    d_contained_indices = None
    d_remainder_indices = None
    contained_indices = None
    remainder_indices = None

    if _remainder_mask is not None:
        d_contained_indices = _cp_local.flatnonzero(~_remainder_mask).astype(
            _cp_local.int64, copy=False,
        )
        d_remainder_indices = _cp_local.flatnonzero(_remainder_mask).astype(
            _cp_local.int64, copy=False,
        )
        n_contained = int(d_contained_indices.size)
        n_remainder = int(d_remainder_indices.size)
    else:
        contained_indices = np.asarray([], dtype=np.intp)
        remainder_indices = np.arange(n_pairs, dtype=np.intp)
        n_contained = 0
        n_remainder = len(remainder_indices)

    record_dispatch_event(
        surface="geopandas.overlay.intersection",
        operation="many_vs_one_containment_bypass",
        implementation=("gpu_containment_bypass" if n_contained > 0 else "skipped"),
        reason=(
            f"many-vs-one: {n_contained}/{n_pairs} polygons fully inside "
            f"right polygon, {n_remainder} need overlay"
        ),
    )

    from vibespatial.cuda._runtime import maybe_trim_pool_memory

    maybe_trim_pool_memory()

    if n_remainder == 0:
        return [(global_positions, _contained_result)], None, None, right_one, _pairwise_mode

    if d_remainder_indices is not None:
        left_remainder = left_sub.take(d_remainder_indices)
        remainder_global = global_positions[
            _cp_local.asnumpy(d_remainder_indices).astype(np.intp, copy=False)
        ]
    else:
        left_remainder = left_sub.take(remainder_indices)
        remainder_global = global_positions[remainder_indices]

    sh_clip_result = None
    sh_eligible_mask = None
    sh_remainder_indices_local = None

    if _cp_local is not None and left_remainder.row_count > 0:
        try:
            from vibespatial.overlay.bypass import (
                _batched_sh_clip,
                _classify_remainder_sh_eligible,
                _is_clip_polygon_sh_eligible,
            )

            clip_eligible, clip_vert_count = _is_clip_polygon_sh_eligible(right_one)
            if clip_eligible:
                sh_eligible_mask_arr, _complex_mask = _classify_remainder_sh_eligible(
                    left_remainder, clip_vert_count,
                )
                n_sh = int(sh_eligible_mask_arr.sum())
                if n_sh > 0:
                    sh_clip_result = _batched_sh_clip(
                        left_remainder, right_one, sh_eligible_mask_arr,
                    )
                    if sh_clip_result is not None:
                        sh_eligible_mask = sh_eligible_mask_arr
                        sh_remainder_indices_local = np.flatnonzero(~sh_eligible_mask_arr)
        except Exception:
            sh_clip_result = None
            sh_eligible_mask = None

    index_oga_pairs = []
    if n_contained > 0 and _contained_result is not None:
        contained_global = (
            global_positions[_cp_local.asnumpy(d_contained_indices).astype(np.intp, copy=False)]
            if d_contained_indices is not None
            else global_positions[contained_indices]
        )
        index_oga_pairs.append((contained_global, _contained_result))

    complex_left = None
    complex_global_positions = None
    if sh_clip_result is not None and sh_eligible_mask is not None:
        sh_global = remainder_global[np.flatnonzero(sh_eligible_mask).astype(np.intp, copy=False)]
        index_oga_pairs.append((sh_global, sh_clip_result))
        if sh_remainder_indices_local is not None and len(sh_remainder_indices_local) > 0:
            if _cp_local is not None:
                d_complex = _cp_local.asarray(sh_remainder_indices_local).astype(
                    _cp_local.int64,
                    copy=False,
                )
                complex_left = left_remainder.take(d_complex)
            else:
                complex_left = left_remainder.take(sh_remainder_indices_local)
            complex_global_positions = remainder_global[
                np.asarray(sh_remainder_indices_local, dtype=np.intp)
            ]
    else:
        complex_left = left_remainder
        complex_global_positions = remainder_global

    return (
        index_oga_pairs,
        complex_left,
        complex_global_positions,
        right_one,
        _pairwise_mode,
    )


def _few_right_intersection_owned(
    left_owned,
    right_owned,
    idx1,
    idx2,
    *,
    dispatch_mode=ExecutionMode.AUTO,
    _has_device_indices=False,
    d_idx1=None,
):
    """Run grouped many-vs-one intersection with one exact remainder overlay pass."""
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover
        cp = None

    unique_right = np.unique(idx2)
    if unique_right.size <= 1 or unique_right.size > _OVERLAY_FEW_RIGHT_GROUP_MAX:
        return None
    if (len(idx2) / unique_right.size) < _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG:
        return None

    from vibespatial.constructive.binary_constructive import (
        _dispatch_polygon_intersection_overlay_rowwise_gpu,
        binary_constructive_owned,
    )
    from vibespatial.geometry.owned import (
        OwnedGeometryArray,
        materialize_broadcast,
        tile_single_row,
    )

    index_oga_pairs: list[tuple[np.ndarray, OwnedGeometryArray]] = []
    complex_left_chunks: list[OwnedGeometryArray] = []
    complex_right_chunks: list[OwnedGeometryArray] = []
    complex_global_chunks: list[np.ndarray] = []
    complex_mode = dispatch_mode

    for right_idx in unique_right.tolist():
        pair_positions = np.flatnonzero(idx2 == right_idx).astype(np.int64, copy=False)
        if pair_positions.size == 0:
            continue
        pair_positions_host = pair_positions.astype(np.intp, copy=False)
        if _has_device_indices:
            if cp is None or d_idx1 is None:
                return None
            d_pair_positions = cp.asarray(pair_positions, dtype=cp.int64)
            left_group = left_owned.device_take(d_idx1[d_pair_positions])
        else:
            left_group = left_owned.take(np.asarray(idx1[pair_positions]))

        (
            group_pairs,
            complex_left,
            complex_global_positions,
            right_one,
            _group_mode,
        ) = _prepare_many_vs_one_intersection_chunks(
            left_group,
            right_owned,
            int(right_idx),
            global_positions=pair_positions_host,
        )
        index_oga_pairs.extend(group_pairs)
        if complex_left is not None and complex_global_positions is not None:
            complex_left_chunks.append(complex_left)
            complex_right_chunks.append(
                materialize_broadcast(
                    tile_single_row(right_one, complex_left.row_count),
                ),
            )
            complex_global_chunks.append(complex_global_positions)

    if complex_left_chunks:
        complex_left_all = OwnedGeometryArray.concat(complex_left_chunks)
        complex_right_all = OwnedGeometryArray.concat(complex_right_chunks)
        complex_global_all = np.concatenate(complex_global_chunks)
        complex_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            complex_left_all,
            complex_right_all,
            dispatch_mode=complex_mode,
        )
        if (
            complex_result is None
            or complex_result.row_count != complex_left_all.row_count
        ):
            complex_result = binary_constructive_owned(
                "intersection",
                complex_left_all,
                complex_right_all,
                dispatch_mode=complex_mode,
                _prefer_exact_polygon_intersection=True,
            )
        index_oga_pairs.append((complex_global_all, complex_result))

    return _assemble_indexed_owned_chunks(index_oga_pairs, len(idx1))


def _overlay_intersection(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
    _index_result=None,
):
    """Overlay Intersection operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1, df2, left_owned, right_owned,
    )
    _polygon_inputs = (
        df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
        and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
    )
    # Public polygon intersection can legitimately produce lower-dimensional
    # rows or GeometryCollections that must be filtered at the GeoPandas
    # boundary. Unless the strict exact-polygon GPU path is explicitly
    # requested, keep geometry construction on the host boundary and use the
    # owned path only for pairing/index acceleration.
    _defer_polygon_intersection_to_public_boundary = (
        _polygon_inputs
        and not _prefer_exact_polygon_gpu
        and not strict_native_mode_enabled()
    )
    # ADR-0036 boundary: spatial index produces index arrays only.
    # Phase 2: pass owned arrays to request device-resident index pairs.
    #
    # When neither side has owned backing yet (plain Shapely GeoDataFrames),
    # force the Shapely STRtree path to avoid the GPU spatial index which
    # incurs ~6s of CUDA kernel compilation on first use.  The owned
    # coercion happens AFTER the spatial join (see below), where it
    # benefits the overlay kernels (containment bypass, SH batch clip)
    # without penalizing the join.
    _force_strtree = (
        _index_result is None
        and left_owned is None
        and right_owned is None
        and has_gpu_runtime()
    )
    if _force_strtree:
        df2.sindex._ensure_strtree()
        _raw_idx = df2.sindex.query(
            df1.geometry,
            predicate="intersects",
            sort=True,
        )
        if isinstance(_raw_idx, np.ndarray) and _raw_idx.ndim == 2:
            idx1, idx2 = _raw_idx
        else:
            idx1, idx2 = _raw_idx
        idx1 = np.asarray(idx1, dtype=np.int32)
        idx2 = np.asarray(idx2, dtype=np.int32)
        d_idx1, d_idx2 = None, None
        _has_device_indices = False
    else:
        index_result = (
            _index_result
            if _index_result is not None
            else _intersecting_index_pairs(
                df1, df2, left_owned=left_owned, right_owned=right_owned,
            )
        )

        # Unpack result: DeviceSpatialJoinResult (device arrays) or numpy.
        if isinstance(index_result, DeviceSpatialJoinResult):
            d_idx1 = index_result.d_left_idx
            d_idx2 = index_result.d_right_idx
            # Host arrays for attribute assembly (pandas needs numpy).
            idx1, idx2 = index_result.to_host()
            _has_device_indices = True
        else:
            if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
                idx1, idx2 = index_result
            else:
                idx1, idx2 = index_result
            d_idx1, d_idx2 = None, None
            _has_device_indices = False

    used_owned = False
    # Create pairs of geometries in both dataframes to be intersected
    if idx1.size > 0 and idx2.size > 0:
        left_sub = None
        right_sub = None
        # Many-vs-one owned coercion: when the spatial join reveals a
        # many-vs-one pattern (all pairs reference the same single right
        # row) and neither side has owned backing, coerce to
        # OwnedGeometryArray to enable the GPU containment bypass and SH
        # batch clip.  This coercion is restricted to the many-vs-one
        # pattern to avoid changing behavior for general N-vs-M overlay.
        _unique_right_pre = np.unique(idx2)
        _is_many_vs_one_pre = (
            _unique_right_pre.size == 1 and idx1.size > 1
        )
        if (
            _is_many_vs_one_pre
            and (left_owned is None or right_owned is None)
            and has_gpu_runtime()
        ):
            _both_polygon = (
                df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
                and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
            )
            if _both_polygon:
                from vibespatial.geometry.owned import from_shapely_geometries

                if left_owned is None:
                    try:
                        left_owned = from_shapely_geometries(
                            list(df1.geometry),
                        )
                    except (NotImplementedError, ValueError):
                        left_owned = None

                if right_owned is None:
                    try:
                        right_owned = from_shapely_geometries(
                            list(df2.geometry),
                        )
                    except (NotImplementedError, ValueError):
                        right_owned = None

        # Owned-path dispatch: OwnedGeometryArray.take() operates at buffer
        # level (no Shapely materialization), then binary_constructive_owned
        # routes to GPU when available.  GeoSeries.take() breaks the DGA chain
        # by materializing to Shapely, so we bypass it when owned is present.
        # Note: GeometryArray.copy() preserves _owned, and __setitem__
        # invalidates it on mutation, so owned survives _make_valid when
        # all geometries are already valid.
        intersections = None
        if (
            left_owned is not None
            and right_owned is not None
            and not _defer_polygon_intersection_to_public_boundary
        ):
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )

            # Free pool memory before device_take: spatial index and
            # prior pipeline stages leave freed-but-cached blocks in
            # the pool.  Forcing GC here ensures dead CuPy arrays
            # return their blocks before the large gather allocation.
            from vibespatial.cuda._runtime import maybe_trim_pool_memory

            maybe_trim_pool_memory()
            # Keep AUTO behavior for normal runs, but force the repo-owned
            # GPU path when the public overlay contract only needs polygon
            # output. The small-workload CPU crossover can yield
            # GeometryCollection rows that are valid at the constructive
            # layer but wrong for the keep_geom_type=True / default overlay
            # boundary before collection extraction runs.
            _pairwise_mode = (
                ExecutionMode.GPU
                if strict_native_mode_enabled() or _prefer_exact_polygon_gpu
                else ExecutionMode.AUTO
            )

            # ---- Many-vs-one detection ----
            # Check BEFORE device_take: for many-vs-one (all pairs
            # reference the same single right row), gathering the right
            # side duplicates one polygon's ring data N times.  At 1M
            # scale this can be 5+ GiB and exceed VRAM.  The many-vs-one
            # fast path only needs left_sub gathered; right_owned is
            # passed by reference.
            _unique_right = np.unique(idx2)
            _is_many_vs_one = (
                _unique_right.size == 1 and idx1.size > 1
            )
            _is_few_right = (
                _unique_right.size > 1
                and _unique_right.size <= _OVERLAY_FEW_RIGHT_GROUP_MAX
                and (idx1.size / _unique_right.size) >= _OVERLAY_FEW_RIGHT_GROUP_MIN_AVG
                and df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
                and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
            )

            if _is_many_vs_one:
                # Many-vs-one: only gather left side.
                if _has_device_indices:
                    left_sub = left_owned.device_take(d_idx1)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                right_sub = None  # deferred until fallback

                def _finalize_many_vs_one(result_owned):
                    nonlocal intersections, used_owned, left_sub
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True

                try:
                    result_owned = _many_vs_one_intersection_owned(
                        left_sub,
                        right_owned,
                        int(_unique_right[0]),
                        _has_device_indices=_has_device_indices,
                        d_idx2=d_idx2,
                    )
                    _finalize_many_vs_one(result_owned)
                except Exception:
                    logger.debug(
                        "many-vs-one polygon intersection fast path failed; trimming pool and retrying",
                        exc_info=True,
                    )
                    try:
                        from vibespatial.cuda._runtime import get_cuda_runtime

                        get_cuda_runtime().free_pool_memory()
                    except Exception:
                        logger.debug(
                            "many-vs-one polygon intersection pool trim failed",
                            exc_info=True,
                        )
                    try:
                        left_sub = left_owned.take(np.asarray(idx1))
                        result_owned = _many_vs_one_intersection_owned(
                            left_sub,
                            right_owned,
                            int(_unique_right[0]),
                            _has_device_indices=False,
                            d_idx2=None,
                        )
                        _finalize_many_vs_one(result_owned)
                    except Exception:
                        logger.debug(
                            "many-vs-one polygon intersection retry failed; falling back to gathered element-wise path",
                            exc_info=True,
                        )
                        # Fall through to element-wise fallback.
                        intersections = None
            elif _is_few_right:
                try:
                    result_owned = _few_right_intersection_owned(
                        left_owned,
                        right_owned,
                        np.asarray(idx1),
                        np.asarray(idx2),
                        dispatch_mode=_pairwise_mode,
                        _has_device_indices=_has_device_indices,
                        d_idx1=d_idx1,
                    )
                    if result_owned is not None:
                        intersections = GeoSeries(
                            GeometryArray.from_owned(result_owned, crs=df1.crs),
                        )
                        used_owned = True
                except Exception:
                    logger.debug(
                        "few-right grouped polygon intersection fast path failed; "
                        "falling back to gathered element-wise path",
                        exc_info=True,
                    )
                    intersections = None
            else:
                # Phase 2 zero-copy: pass CuPy device arrays directly to
                # device_take() when available, eliminating H->D re-upload.
                if _has_device_indices:
                    left_sub = left_owned.device_take(d_idx1)
                    right_sub = right_owned.device_take(d_idx2)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                    right_sub = right_owned.take(np.asarray(idx2))

            if intersections is None and not (_is_many_vs_one or _is_few_right):
                # Standard element-wise path for N-vs-M patterns.
                result_owned = binary_constructive_owned(
                    "intersection", left_sub, right_sub,
                    dispatch_mode=_pairwise_mode,
                    _prefer_exact_polygon_intersection=_prefer_exact_polygon_gpu,
                )
                intersections = GeoSeries(
                    GeometryArray.from_owned(result_owned, crs=df1.crs),
                )
                used_owned = True

            if intersections is None and _is_few_right:
                if _has_device_indices:
                    left_sub = left_owned.device_take(d_idx1)
                    right_sub = right_owned.device_take(d_idx2)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                    right_sub = right_owned.take(np.asarray(idx2))
                result_owned = binary_constructive_owned(
                    "intersection", left_sub, right_sub,
                    dispatch_mode=_pairwise_mode,
                    _prefer_exact_polygon_intersection=_prefer_exact_polygon_gpu,
                )
                intersections = GeoSeries(
                    GeometryArray.from_owned(result_owned, crs=df1.crs),
                )
                used_owned = True

            if intersections is None and _is_many_vs_one:
                # Many-vs-one fast path failed -- fall back to element-wise.
                # Gather right side now (deferred from above to avoid OOM
                # on the many-vs-one fast path).
                if right_sub is None:
                    if _has_device_indices:
                        right_sub = right_owned.device_take(d_idx2)
                    else:
                        right_sub = right_owned.take(np.asarray(idx2))
                result_owned = binary_constructive_owned(
                    "intersection", left_sub, right_sub,
                    dispatch_mode=_pairwise_mode,
                    _prefer_exact_polygon_intersection=_prefer_exact_polygon_gpu,
                )
                intersections = GeoSeries(
                    GeometryArray.from_owned(result_owned, crs=df1.crs),
                )
                used_owned = True

        if intersections is None:
            # ADR-0036 boundary: geometry operations on geometry arrays.
            if _defer_polygon_intersection_to_public_boundary:
                # Force a plain Shapely-backed public-boundary intersection
                # here. GeoSeries.take() on owned-backed geometry can retain
                # DGA backing and recurse into binary_constructive_owned,
                # which cannot round-trip GeometryCollection outputs.
                left = GeoSeries(list(df1.geometry.take(idx1)), crs=df1.crs)
                right = GeoSeries(list(df2.geometry.take(idx2)), crs=df2.crs)
            else:
                left = df1.geometry.take(idx1)
                left.reset_index(drop=True, inplace=True)
                right = df2.geometry.take(idx2)
                right.reset_index(drop=True, inplace=True)
            intersections = left.intersection(right)

        # Post-intersection make_valid: use GPU path when owned backing is
        # available to avoid Shapely materialisation on the critical path.
        intersections = _make_valid_geoseries(intersections)

        geom_intersect = intersections
        keep_geom_type_applied = False
        if (
            df1.geom_type.isin(POLYGON_GEOM_TYPES).all()
            and df2.geom_type.isin(POLYGON_GEOM_TYPES).all()
        ):
            pair_left = None
            pair_right = None
            if (
                _preserve_lower_dim_polygon_results
                or _warn_on_dropped_lower_dim_polygon_results
                or _prefer_exact_polygon_gpu
            ):
                if left_sub is not None and right_sub is not None:
                    pair_left = GeoSeries(
                        GeometryArray.from_owned(left_sub, crs=df1.crs),
                        index=geom_intersect.index,
                    )
                    pair_right = GeoSeries(
                        GeometryArray.from_owned(right_sub, crs=df1.crs),
                        index=geom_intersect.index,
                    )
                else:
                    pair_left = df1.geometry.take(idx1)
                    pair_left.reset_index(drop=True, inplace=True)
                    pair_right = df2.geometry.take(idx2)
                    pair_right.reset_index(drop=True, inplace=True)

            if _preserve_lower_dim_polygon_results:
                geom_intersect = _assemble_polygon_intersection_rows_with_lower_dim(
                    pair_left,
                    pair_right,
                    geom_intersect,
                )
            elif _warn_on_dropped_lower_dim_polygon_results:
                geom_intersect, num_dropped, keep_mask = _filter_polygon_intersection_rows_for_keep_geom_type(
                    pair_left,
                    pair_right,
                    geom_intersect,
                    keep_geom_type_warning=True,
                )
                idx1 = np.asarray(idx1, dtype=np.int32)[keep_mask]
                idx2 = np.asarray(idx2, dtype=np.int32)[keep_mask]
                if num_dropped > 0:
                    warnings.warn(
                        "`keep_geom_type=True` in overlay resulted in "
                        f"{num_dropped} dropped geometries of different "
                        "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
                        "geometries",
                        UserWarning,
                        stacklevel=4,
                )
                keep_geom_type_applied = True
            elif _prefer_exact_polygon_gpu:
                geom_intersect, _, keep_mask = _filter_polygon_intersection_rows_for_keep_geom_type(
                    pair_left,
                    pair_right,
                    geom_intersect,
                    keep_geom_type_warning=False,
                )
                idx1 = np.asarray(idx1, dtype=np.int32)[keep_mask]
                idx2 = np.asarray(idx2, dtype=np.int32)[keep_mask]
                keep_geom_type_applied = True

        nonempty_mask = ~(geom_intersect.isna() | geom_intersect.is_empty)
        if not nonempty_mask.all():
            keep = np.asarray(nonempty_mask, dtype=bool)
            idx1 = np.asarray(idx1, dtype=np.int32)[keep]
            idx2 = np.asarray(idx2, dtype=np.int32)[keep]
            geom_intersect = geom_intersect[keep].reset_index(drop=True)

        # ADR-0036 boundary: attribute assembly from index arrays.
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = _assemble_intersection_attributes(
            idx1, idx2,
            df1.drop(df1._geometry_column_name, axis=1),
            df2.drop(df2._geometry_column_name, axis=1),
        )

        result = GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs)
        if keep_geom_type_applied:
            result.attrs["_vibespatial_keep_geom_type_applied"] = True
        return result, used_owned
    else:
        result = df1.iloc[:0].merge(
            df2.iloc[:0].drop(df2.geometry.name, axis=1),
            left_index=True,
            right_index=True,
            suffixes=("_1", "_2"),
        )
        result["__idx1"] = np.nan
        result["__idx2"] = np.nan
        return result[
            result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]
        ], used_owned


def _overlay_difference(df1, df2, left_owned=None, right_owned=None, *, _index_result=None):
    """Overlay Difference operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1, df2, left_owned, right_owned,
    )
    # ADR-0036 boundary: spatial index produces index arrays only.
    # Phase 2: pass owned arrays to request device-resident index pairs.
    index_result = (
        _index_result
        if _index_result is not None
        else _intersecting_index_pairs(
            df1, df2, left_owned=left_owned, right_owned=right_owned,
        )
    )

    # Unpack result: DeviceSpatialJoinResult (device arrays) or numpy.
    if isinstance(index_result, DeviceSpatialJoinResult):
        d_idx1 = index_result.d_left_idx
        d_idx2 = index_result.d_right_idx
        idx1, idx2 = index_result.to_host()
        _has_device_indices = True
    else:
        if isinstance(index_result, np.ndarray) and index_result.ndim == 2:
            idx1, idx2 = index_result
        else:
            idx1, idx2 = index_result
        d_idx1, d_idx2 = None, None
        _has_device_indices = False

    n_left = len(df1)
    used_owned = False
    result_geoms = None
    result_owned = None

    # Owned-path dispatch: GPU segmented union + GPU difference when both
    # DataFrames have owned backing.  Avoids Shapely materialization for
    # the union step when the segmented_union_all GPU kernel is available.
    # Phase 18: uses concat_owned_scatter to keep the result device-resident
    # instead of materializing via to_shapely().
    #
    use_lower_dim_boundary_rebuild = _needs_host_overlay_difference_boundary_rebuild(df1, df2)

    if (
        idx1.size > 0
        and left_owned is not None
        and right_owned is not None
        and not use_lower_dim_boundary_rebuild
    ):
        from vibespatial.geometry.owned import concat_owned_scatter
        from vibespatial.runtime.residency import Residency

        # Keep AUTO behavior for normal runs, but in strict-native mode force
        # the repo-owned GPU difference path here so overlay does not die on
        # the generic small-workload crossover before the polygon dispatcher
        # can choose its overlay-based GPU implementation for concave inputs.
        _pairwise_mode = (
            ExecutionMode.GPU
            if strict_native_mode_enabled()
            else ExecutionMode.AUTO
        )

        # Batched overlay difference: splits groups into VRAM-safe
        # batches to prevent OOM at large scale (100K+).  At small
        # scale the fast path processes everything in a single batch.
        diff_owned, idx1_unique = _batched_overlay_difference_owned(
            left_owned,
            right_owned,
            idx1,
            idx2,
            d_idx1,
            d_idx2,
            _has_device_indices,
            _pairwise_mode,
        )

        # Assemble full result without reviving emptied overlap rows.
        # Rows with no spatial-join neighbors keep their original left geometry.
        # Rows in idx1_unique must take the differenced result verbatim, even
        # when that result is null/empty.
        null_base = _empty_owned_result_base(
            n_left,
            device=(
                left_owned.residency is Residency.DEVICE
                or diff_owned.residency is Residency.DEVICE
            ),
        )
        result_owned = null_base
        no_neighbor_idx = np.setdiff1d(
            np.arange(n_left, dtype=np.int64),
            np.asarray(idx1_unique, dtype=np.int64),
            assume_unique=True,
        )
        if no_neighbor_idx.size > 0:
            preserved_left = left_owned.take(no_neighbor_idx)
            result_owned = concat_owned_scatter(
                result_owned,
                preserved_left,
                no_neighbor_idx,
            )
        result_owned = concat_owned_scatter(
            result_owned,
            diff_owned,
            np.asarray(idx1_unique, dtype=np.int64),
        )
        used_owned = True

    if result_owned is not None:
        # Device-resident path: wrap the scattered OwnedGeometryArray
        # directly in a GeoSeries, preserving the owned backing.
        differences = GeoSeries(
            GeometryArray.from_owned(result_owned, crs=df1.crs),
            index=df1.index,
        )
    else:
        if result_geoms is None:
            result_geoms = _grouped_overlay_difference_geoms(df1, df2, idx1, idx2)

        differences = GeoSeries(result_geoms, index=df1.index, crs=df1.crs)

    # Post-difference make_valid: use GPU path when owned backing is
    # available to avoid Shapely materialisation on the critical path.
    differences = _make_valid_geoseries(differences)
    differences_owned = getattr(differences.values, "_owned", None)
    if differences_owned is not None:
        keep_rows = np.asarray(differences_owned.validity, dtype=bool)
        keep_indices = np.flatnonzero(keep_rows).astype(np.int64, copy=False)
        if keep_indices.size > 0:
            geom_diff = GeoSeries(
                GeometryArray.from_owned(
                    differences_owned.take(keep_indices),
                    crs=df1.crs,
                ),
                index=df1.index[keep_rows],
            )
        else:
            geom_diff = GeoSeries([], index=df1.index[:0], crs=df1.crs)
        dfdiff = df1[keep_rows].copy()
    else:
        empty_mask = differences.is_empty
        if empty_mask.any():
            differences = differences.copy()
            differences.loc[empty_mask] = None
        keep_rows = ~empty_mask
        geom_diff = differences[keep_rows].copy()
        dfdiff = df1[keep_rows].copy()
    geo_col = dfdiff._geometry_column_name
    # Use set_geometry to replace the geometry column while preserving
    # owned backing and the original geometry column name.  The plain
    # __setitem__ path (dfdiff[col] = series) destroys _owned.
    geom_diff.name = geo_col
    dfdiff = dfdiff.set_geometry(geom_diff, crs=df1.crs)
    return dfdiff, used_owned


def _overlay_identity(df1, df2, left_owned=None, right_owned=None):
    """Overlay Identity operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    forward_index_result = _intersecting_index_pairs(
        df1, df2, left_owned=left_owned, right_owned=right_owned,
    )
    dfintersection, used_inter = _overlay_intersection(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
    )
    dfdifference, used_diff = _overlay_difference(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
    )
    dfdifference = _ensure_geometry_column(dfdifference)

    # Columns that were suffixed in dfintersection need to be suffixed in dfdifference
    # as well so they can be matched properly in concat.
    new_columns = [
        col if col in dfintersection.columns else f"{col}_1"
        for col in dfdifference.columns
    ]
    dfdifference.columns = new_columns

    # Now we can concatenate the two dataframes
    result = pd.concat([dfintersection, dfdifference], ignore_index=True, sort=False)

    # keep geometry column last
    columns = list(dfintersection.columns)
    columns.remove("geometry")
    columns.append("geometry")
    result = result.reindex(columns=columns)
    if not isinstance(result, GeoDataFrame):
        result = GeoDataFrame(result)
    if result.crs is None and df1.crs is not None:
        result = result.set_crs(df1.crs)
    return result, used_inter or used_diff


def _overlay_symmetric_diff(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _forward_index_result=None,
    _reverse_index_result=None,
):
    """Overlay Symmetric Difference operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    if _forward_index_result is None:
        _forward_index_result = _intersecting_index_pairs(
            df1, df2, left_owned=left_owned, right_owned=right_owned,
        )
    if _reverse_index_result is None:
        _reverse_index_result = _reverse_intersecting_index_pairs(
            _forward_index_result,
        )

    dfdiff1, used1 = _overlay_difference(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_forward_index_result,
    )
    dfdiff2, used2 = _overlay_difference(
        df2,
        df1,
        right_owned,
        left_owned,
        _index_result=_reverse_index_result,
    )
    dfdiff1["__idx1"] = range(len(dfdiff1))
    dfdiff2["__idx2"] = range(len(dfdiff2))
    dfdiff1["__idx2"] = np.nan
    dfdiff2["__idx1"] = np.nan
    # ensure geometry name (otherwise merge goes wrong)
    dfdiff1 = _ensure_geometry_column(dfdiff1)
    dfdiff2 = _ensure_geometry_column(dfdiff2)

    # Check whether both differences carry owned-backed geometry.
    # If so, bypass the merge-then-pick pattern (which destroys _owned)
    # in favour of pd.concat which preserves owned backing via
    # GeometryArray._concat_same_type.
    diff1_owned = getattr(dfdiff1.geometry.values, '_owned', None)
    diff2_owned = getattr(dfdiff2.geometry.values, '_owned', None)

    if diff1_owned is not None and diff2_owned is not None:
        # GeoPandas warns on CRS mismatch at the public overlay boundary but
        # still proceeds with the left CRS. The owned-preserving concat path
        # must mirror that contract instead of delegating the mismatch to
        # GeometryArray._concat_same_type(), which would raise on pd.concat.
        if dfdiff1.crs != dfdiff2.crs:
            dfdiff2 = dfdiff2.set_crs(dfdiff1.crs, allow_override=True)

        # Align columns to match merge(..., suffixes=("_1","_2")) behavior:
        # shared attribute columns get "_1"/"_2" suffix; unique columns
        # keep their original names.
        skip = {"geometry", "__idx1", "__idx2"}
        attr1 = {c for c in dfdiff1.columns if c not in skip}
        attr2 = {c for c in dfdiff2.columns if c not in skip}
        shared = attr1 & attr2
        rename1 = {c: f"{c}_1" for c in shared}
        rename2 = {c: f"{c}_2" for c in shared}
        if rename1:
            dfdiff1 = dfdiff1.rename(columns=rename1)
        if rename2:
            dfdiff2 = dfdiff2.rename(columns=rename2)

        dfsym = pd.concat(
            [dfdiff1, dfdiff2], ignore_index=True, sort=False,
        )
        # keep geometry column last
        columns = [c for c in dfsym.columns if c != "geometry"] + ["geometry"]
        dfsym = dfsym.reindex(columns=columns)
        if not isinstance(dfsym, GeoDataFrame):
            dfsym = GeoDataFrame(dfsym)
        if dfsym.crs is None and df1.crs is not None:
            dfsym = dfsym.set_crs(df1.crs)
        return dfsym, used1 or used2

    # Shapely fallback: merge path (destroys owned backing).
    dfsym = dfdiff1.merge(
        dfdiff2, on=["__idx1", "__idx2"], how="outer", suffixes=("_1", "_2")
    )
    geometry = dfsym.geometry_1.copy()
    geometry.name = "geometry"
    # https://github.com/pandas-dev/pandas/issues/26468 use loc for now
    geometry.loc[dfsym.geometry_1.isnull()] = dfsym.loc[
        dfsym.geometry_1.isnull(), "geometry_2"
    ]
    dfsym.drop(["geometry_1", "geometry_2"], axis=1, inplace=True)
    dfsym.reset_index(drop=True, inplace=True)
    dfsym = GeoDataFrame(dfsym, geometry=geometry, crs=df1.crs)
    return dfsym, used1 or used2


def _overlay_union(df1, df2, left_owned=None, right_owned=None):
    """Overlay Union operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    forward_index_result = _intersecting_index_pairs(
        df1, df2, left_owned=left_owned, right_owned=right_owned,
    )
    dfinter, used_inter = _overlay_intersection(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
    )
    dfsym, used_sym = _overlay_symmetric_diff(
        df1,
        df2,
        left_owned,
        right_owned,
        _forward_index_result=forward_index_result,
    )
    dfunion = pd.concat([dfinter, dfsym], ignore_index=True, sort=False)
    # keep geometry column last
    columns = list(dfunion.columns)
    columns.remove("geometry")
    columns.append("geometry")
    result = dfunion.reindex(columns=columns)
    if not isinstance(result, GeoDataFrame):
        result = GeoDataFrame(result)
    if result.crs is None and df1.crs is not None:
        result = result.set_crs(df1.crs)
    return result, used_inter or used_sym


def overlay(df1, df2, how="intersection", keep_geom_type=None, make_valid=True):
    """Perform spatial overlay between two GeoDataFrames.

    Currently only supports data GeoDataFrames with uniform geometry types,
    i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
    combination of (Multi)LineString and LinearRing shapes.
    Implements several methods that are all effectively subsets of the union.

    See the User Guide page :doc:`../../user_guide/set_operations` for details.

    Parameters
    ----------
    df1 : GeoDataFrame
    df2 : GeoDataFrame
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None,
        which will set keep_geom_type to True but warn upon dropping
        geometries.
    make_valid : bool, default True
        If True, any invalid input geometries are corrected with a call to make_valid(),
        if False, a `ValueError` is raised if any input geometries are invalid.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> polys1 = geopandas.GeoSeries([Polygon([(0,0), (2,0), (2,2), (0,2)]),
    ...                               Polygon([(2,2), (4,2), (4,4), (2,4)])])
    >>> polys2 = geopandas.GeoSeries([Polygon([(1,1), (3,1), (3,3), (1,3)]),
    ...                               Polygon([(3,3), (5,3), (5,5), (3,5)])])
    >>> df1 = geopandas.GeoDataFrame({'geometry': polys1, 'df1_data':[1,2]})
    >>> df2 = geopandas.GeoDataFrame({'geometry': polys2, 'df2_data':[1,2]})

    >>> geopandas.overlay(df1, df2, how='union')
        df1_data  df2_data                                           geometry
    0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    5       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    6       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='intersection')
       df1_data  df2_data                             geometry
    0         1         1  POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2         1  POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2         2  POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))

    >>> geopandas.overlay(df1, df2, how='symmetric_difference')
        df1_data  df2_data                                           geometry
    0       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    1       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
    2       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
    3       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

    >>> geopandas.overlay(df1, df2, how='difference')
                                                geometry  df1_data
    0      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))         1
    1  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...         2

    >>> geopandas.overlay(df1, df2, how='identity')
       df1_data  df2_data                                           geometry
    0         1       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
    1         2       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
    2         2       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
    3         1       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
    4         2       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...

    See Also
    --------
    sjoin : spatial join
    GeoDataFrame.overlay : equivalent method

    Notes
    -----
    Every operation in GeoPandas is planar, i.e. the potential third
    dimension is not taken into account.
    """
    # Allowed operations
    allowed_hows = [
        "intersection",
        "union",
        "identity",
        "symmetric_difference",
        "difference",  # aka erase
    ]
    # Error Messages
    if how not in allowed_hows:
        raise ValueError(f"`how` was '{how}' but is expected to be in {allowed_hows}")

    if isinstance(df1, GeoSeries) or isinstance(df2, GeoSeries):
        raise NotImplementedError(
            "overlay currently only implemented for GeoDataFrames"
        )

    if not _check_crs(df1, df2):
        _crs_mismatch_warn(df1, df2, stacklevel=3)

    if keep_geom_type is None:
        keep_geom_type = True
        keep_geom_type_warning = True
    else:
        keep_geom_type_warning = False

    for i, df in enumerate([df1, df2]):
        poly_check = df.geom_type.isin(POLYGON_GEOM_TYPES).any()
        lines_check = df.geom_type.isin(LINE_GEOM_TYPES).any()
        points_check = df.geom_type.isin(POINT_GEOM_TYPES).any()
        if sum([poly_check, lines_check, points_check]) > 1:
            raise NotImplementedError(f"df{i + 1} contains mixed geometry types.")

    if how == "intersection":
        box_gdf1 = df1.total_bounds
        box_gdf2 = df2.total_bounds

        if not (
            ((box_gdf1[0] <= box_gdf2[2]) and (box_gdf2[0] <= box_gdf1[2]))
            and ((box_gdf1[1] <= box_gdf2[3]) and (box_gdf2[1] <= box_gdf1[3]))
        ):
            result = df1.iloc[:0].merge(
                df2.iloc[:0].drop(df2.geometry.name, axis=1),
                left_index=True,
                right_index=True,
                suffixes=("_1", "_2"),
            )
            return result[
                result.columns.drop(df1.geometry.name).tolist() + [df1.geometry.name]
            ]

    # Computations
    def _make_valid(df):
        df = df.copy()
        if df.geom_type.isin(POLYGON_GEOM_TYPES).all():
            # GPU make_valid path: when owned backing is available, route
            # through make_valid_owned to keep data device-resident and
            # avoid Shapely materialisation on the overlay critical path.
            ga = df.geometry.values
            owned = getattr(ga, '_owned', None)
            if make_valid and owned is not None:
                from vibespatial.constructive.make_valid_pipeline import (
                    make_valid_owned,
                )

                mv_result = make_valid_owned(owned=owned)
                if mv_result.repaired_rows.size > 0:
                    # Repair happened — prefer device-resident .owned
                    # to avoid D->H transfer.
                    new_ga = None
                    if mv_result.owned is not None:
                        try:
                            new_ga = GeometryArray.from_owned(
                                mv_result.owned, crs=df.crs,
                            )
                        except (NotImplementedError, Exception):
                            pass
                    if new_ga is None:
                        try:
                            from vibespatial.geometry.owned import (
                                from_shapely_geometries,
                            )

                            new_owned = from_shapely_geometries(
                                list(mv_result.geometries),
                            )
                            new_ga = GeometryArray.from_owned(
                                new_owned, crs=df.crs,
                            )
                        except NotImplementedError:
                            new_ga = GeometryArray(
                                mv_result.geometries, crs=df.crs,
                            )
                    col = df._geometry_column_name
                    df[col] = GeoSeries(new_ga, index=df.index)
                    df = _collection_extract(
                        df, geom_type="Polygon", keep_geom_type_warning=False
                    )
                # else: all rows already valid — owned backing preserved
                #       by df.copy() above (GeometryArray.copy preserves _owned).
                return df

            mask = ~df.geometry.is_valid
            col = df._geometry_column_name
            if make_valid:
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].make_valid()
                    # Extract only the input geometry type, as make_valid may change it
                    df = _collection_extract(
                        df, geom_type="Polygon", keep_geom_type_warning=False
                    )

            elif mask.any():
                raise ValueError(
                    "You have passed make_valid=False along with "
                    f"{mask.sum()} invalid input geometries. "
                    "Use make_valid=True or make sure that all geometries "
                    "are valid before using overlay."
                )
        return df

    # Determine the geometry type before make_valid, as make_valid may change it
    if keep_geom_type:
        geom_type = df1.geom_type.iloc[0]

    df1 = _make_valid(df1)
    df2 = _make_valid(df2)

    # Extract owned arrays AFTER _make_valid.  GeometryArray.copy() now
    # preserves _owned backing, and __setitem__ invalidates it only for
    # mutated rows.  If _make_valid mutated all rows or dropped rows via
    # _collection_extract, _owned will already be None here.
    left_owned, right_owned = _extract_owned_pair(df1, df2)

    _used_owned = False
    with warnings.catch_warnings():  # CRS checked above, suppress array-level warning
        warnings.filterwarnings("ignore", message="CRS mismatch between the CRS")
        if how == "difference":
            result, _used_owned = _overlay_difference(
                df1, df2, left_owned, right_owned,
            )
        elif how == "intersection":
            result, _used_owned = _overlay_intersection(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(
                    strict_native_mode_enabled() and keep_geom_type is not False
                ),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
            )
        elif how == "symmetric_difference":
            result, _used_owned = _overlay_symmetric_diff(
                df1, df2, left_owned, right_owned,
            )
        elif how == "union":
            result, _used_owned = _overlay_union(df1, df2, left_owned, right_owned)
        elif how == "identity":
            result, _used_owned = _overlay_identity(
                df1, df2, left_owned, right_owned,
            )

    record_dispatch_event(
        surface="geopandas.overlay",
        operation=f"overlay_{how}",
        implementation="owned_dispatch" if _used_owned else "shapely_host",
        reason=(
            f"{how} via owned-path dispatch"
            if _used_owned
            else "no owned backing or host fallback"
        ),
        detail=(
            f"left_rows={len(df1)}, right_rows={len(df2)}, "
            f"how={how}, owned={left_owned is not None}"
        ),
        requested=ExecutionMode.AUTO,
        selected=ExecutionMode.GPU if _used_owned else ExecutionMode.CPU,
    )

    if keep_geom_type and not result.attrs.get("_vibespatial_keep_geom_type_applied", False):
        result_owned = getattr(result.geometry.values, "_owned", None)
        if result_owned is not None:
            result = _collection_extract_owned(result, geom_type, keep_geom_type_warning)
        else:
            result = _collection_extract(result, geom_type, keep_geom_type_warning)

    if result.geometry.isna().any():
        result = result.loc[~result.geometry.isna()].copy()

    if how in ["intersection", "symmetric_difference", "union", "identity"]:
        drop_cols = [col for col in ("__idx1", "__idx2") if col in result.columns]
        if drop_cols:
            result.drop(drop_cols, axis=1, inplace=True)

    result.reset_index(drop=True, inplace=True)
    return result


def _geom_type_to_target_families(geom_type: str) -> set[int] | None:
    """Map a Shapely geom_type string to the set of OwnedGeometryArray family tags to keep.

    Returns ``None`` if *geom_type* is not a recognized polygon, line, or point type.
    Imports are deferred to avoid circular dependencies.
    """
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    if geom_type in POLYGON_GEOM_TYPES:
        return {FAMILY_TAGS[GeometryFamily.POLYGON], FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]}
    if geom_type in LINE_GEOM_TYPES:
        return {
            FAMILY_TAGS[GeometryFamily.LINESTRING],
            FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
        }
    if geom_type in POINT_GEOM_TYPES:
        return {FAMILY_TAGS[GeometryFamily.POINT], FAMILY_TAGS[GeometryFamily.MULTIPOINT]}
    return None


def _collection_extract_owned(df, geom_type, keep_geom_type_warning):
    """Device-resident collection extract: filter by geometry family tag.

    When the result GeoDataFrame's geometry column has OwnedGeometryArray
    backing, we can filter by the ``tags`` array directly -- no Shapely
    materialization, no ``.explode()``, no ``.dissolve()``.

    OwnedGeometryArray does not represent GeometryCollections; constituent
    geometries are stored as individual rows tagged by their concrete family.
    Filtering is a simple mask on the tags array followed by
    ``OwnedGeometryArray.take()``.
    """
    from vibespatial.geometry.owned import NULL_TAG

    ga = df.geometry.values
    owned = ga._owned

    target_tags = _geom_type_to_target_families(geom_type)
    if target_tags is None:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    tags = owned.tags
    keep_mask = np.zeros(len(tags), dtype=bool)
    for tag in target_tags:
        keep_mask |= tags == tag

    num_dropped = int((~keep_mask & (tags != NULL_TAG)).sum())

    if num_dropped > 0 and keep_geom_type_warning:
        warnings.warn(
            "`keep_geom_type=True` in overlay resulted in "
            f"{num_dropped} dropped geometries of different "
            "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
            "geometries",
            UserWarning,
            stacklevel=2,
        )

    # Preserve null geometries only on the default keep_geom_type=None path,
    # which historically keeps bookkeeping rows that later warning-based
    # filtering may expose.  Explicit keep_geom_type=True should continue to
    # behave strictly and drop missing-geometry rows.
    if keep_geom_type_warning:
        keep_mask |= ~owned.validity

    if keep_mask.all():
        return df

    # Filter both the DataFrame rows and the owned geometry array together.
    # Use iloc for positional indexing -- the DataFrame may have a non-default
    # index after concat in overlay sub-operations.
    keep_indices = np.flatnonzero(keep_mask)
    result = df.iloc[keep_indices].copy()
    filtered_owned = owned.take(keep_indices)

    if (
        geom_type in POLYGON_GEOM_TYPES
        and "__idx1" in result.columns
        and "__idx2" in result.columns
        and len(result) > 1
    ):
        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        idx1_col = result["__idx1"]
        idx2_col = result["__idx2"]
        if idx1_col.notna().all() and idx2_col.notna().all():
            idx1 = idx1_col.to_numpy(dtype=np.int64, copy=False)
            idx2 = idx2_col.to_numpy(dtype=np.int64, copy=False)
            order = np.lexsort((idx2, idx1))
            if order.size > 1:
                idx1_sorted = idx1[order]
                idx2_sorted = idx2[order]
                group_starts = np.flatnonzero(
                    (idx1_sorted[1:] != idx1_sorted[:-1]) | (idx2_sorted[1:] != idx2_sorted[:-1])
                ) + 1
                group_offsets = np.concatenate(
                    [np.asarray([0], dtype=np.int64), group_starts.astype(np.int64, copy=False), np.asarray([len(order)], dtype=np.int64)]
                )
                if len(group_offsets) - 1 != len(order):
                    result = result.iloc[order].iloc[group_offsets[:-1]].copy()
                    filtered_owned = segmented_union_all(filtered_owned.take(order), group_offsets)

    # Rebuild the GeoSeries with owned backing to avoid Shapely materialisation.
    geom_col = result._geometry_column_name
    result[geom_col] = GeoSeries(
        GeometryArray.from_owned(filtered_owned, crs=df.crs),
        index=result.index,
    )
    return result


def _collection_extract(df, geom_type, keep_geom_type_warning):
    # Check input
    if geom_type in POLYGON_GEOM_TYPES:
        geom_types = POLYGON_GEOM_TYPES
    elif geom_type in LINE_GEOM_TYPES:
        geom_types = LINE_GEOM_TYPES
    elif geom_type in POINT_GEOM_TYPES:
        geom_types = POINT_GEOM_TYPES
    else:
        raise TypeError(f"`geom_type` does not support {geom_type}.")

    result = df.copy()

    # First we filter the geometry types inside GeometryCollections objects
    # (e.g. GeometryCollection([polygon, point]) -> polygon)
    # we do this separately on only the relevant rows, as this is an expensive
    # operation (an expensive no-op for geometry types other than collections)
    is_collection = result.geom_type == "GeometryCollection"
    if is_collection.any():
        geom_col = result._geometry_column_name
        collections = result.loc[is_collection, [geom_col]]

        exploded = collections.reset_index(drop=True).explode(index_parts=True)
        exploded = exploded.reset_index(level=0)

        orig_num_geoms_exploded = exploded.shape[0]
        exploded.loc[~exploded.geom_type.isin(geom_types), geom_col] = None
        num_dropped_collection = (
            orig_num_geoms_exploded - exploded.geometry.isna().sum()
        )

        # level_0 created with above reset_index operation
        # and represents the original geometry collections
        # TODO avoiding dissolve to call union_all in this case could further
        # improve performance (we only need to collect geometries in their
        # respective Multi version)
        dissolved = exploded.dissolve(by="level_0")
        result.loc[is_collection, geom_col] = dissolved[geom_col].values
    else:
        num_dropped_collection = 0

    # Now we filter all geometries (in theory we don't need to do this
    # again for the rows handled above for GeometryCollections, but filtering
    # them out is probably more expensive as simply including them when this
    # is typically about only a few rows)
    orig_num_geoms = result.shape[0]
    geom_keep_mask = result.geom_type.isin(geom_types)
    if keep_geom_type_warning:
        geom_keep_mask = geom_keep_mask | result.geometry.isna()
    result = result.loc[geom_keep_mask]
    num_dropped = orig_num_geoms - result.shape[0]

    if (num_dropped > 0 or num_dropped_collection > 0) and keep_geom_type_warning:
        warnings.warn(
            "`keep_geom_type=True` in overlay resulted in "
            f"{num_dropped + num_dropped_collection} dropped geometries of different "
            "geometry types than df1 has. Set `keep_geom_type=False` to retain all "
            "geometries",
            UserWarning,
            stacklevel=2,
        )

    return result
