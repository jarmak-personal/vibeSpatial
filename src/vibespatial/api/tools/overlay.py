from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import GeometryCollection

from vibespatial.api import GeoDataFrame, GeoSeries
from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_results import (
    ConcatConstructiveResult,
    GeometryNativeResult,
    LeftConstructiveFragment,
    LeftConstructiveResult,
    PairwiseConstructiveFragment,
    PairwiseConstructiveResult,
    RelationIndexResult,
    SymmetricDifferenceConstructiveResult,
)
from vibespatial.api.geometry_array import (
    LINE_GEOM_TYPES,
    POINT_GEOM_TYPES,
    POLYGON_GEOM_TYPES,
    GeometryArray,
    _check_crs,
    _crs_mismatch_warn,
)
from vibespatial.api.tools._pair_cache import get_cached_intersection_pairs
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
_OVERLAY_FEW_RIGHT_CACHED_SEG_MIN_ROWS = 256
_OVERLAY_ROWWISE_REMAINDER_MAX = 32
_OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX = 128
_OVERLAY_HOST_EXACT_PAIR_BATCH_MAX_ROWS = 128
_MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS = 64
_MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION = 0.05
_MANY_VS_ONE_HOST_EXACT_MAX_ROWS = 4_096
_SHAPELY_TYPE_ID_POLYGON = 3
_SHAPELY_TYPE_ID_MULTIPOLYGON = 6
_SHAPELY_TYPE_ID_GEOMETRYCOLLECTION = 7


def _can_rewrite_single_mask_intersection_to_clip(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    *,
    how: str,
    left_all_polygons: bool,
    right_all_polygons: bool,
) -> bool:
    """Return True when overlay intersection is semantically equivalent to clip.

    A single-row, geometry-only polygon mask on the right has the same public
    result columns and geometry semantics as ``clip(df1, mask)`` while avoiding
    the generic overlay planner entirely.

    On GPU-capable runs, keep this shape on the native overlay path instead of
    routing it through the public clip surface. The current clip boundary still
    carries public-path cleanup that is slower and can diverge from overlay's
    keep-geom-type behavior for polygon intersections.
    """
    if has_gpu_runtime():
        return False
    if how != "intersection":
        return False
    if not left_all_polygons or not right_all_polygons:
        return False
    if len(df2) != 1:
        return False
    return tuple(df2.columns) == (df2.geometry.name,)


def _series_polygon_mask(series: GeoSeries) -> np.ndarray:
    """Return a polygon-or-multipolygon membership mask for a geometry series."""
    owned = getattr(series.values, "_owned", None)
    if owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS

        tags = np.asarray(owned.tags)
        return np.asarray(owned.validity, dtype=bool) & (
            (tags == FAMILY_TAGS[GeometryFamily.POLYGON])
            | (tags == FAMILY_TAGS[GeometryFamily.MULTIPOLYGON])
        )
    return np.asarray(series.geom_type.isin(POLYGON_GEOM_TYPES), dtype=bool)


def _series_all_polygons(series: GeoSeries) -> bool:
    return bool(_series_polygon_mask(series).all())


def _maybe_seed_polygon_validity_cache(spatial) -> None:
    geometry = spatial.geometry if isinstance(spatial, GeoDataFrame) else spatial
    values = geometry.values
    owned = getattr(values, "_owned", None)
    if owned is None:
        return
    if not bool(geometry.geom_type.isin(POLYGON_GEOM_TYPES).all()):
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _seed_all_validity_cache_if_owned(owned) -> None:
    if owned is None:
        return

    from vibespatial.geometry.owned import seed_all_validity_cache

    seed_all_validity_cache(owned)


def _candidate_rows_all_valid(series: GeoSeries, row_indices: np.ndarray) -> bool:
    if row_indices.size == 0:
        return True

    values = series.values
    owned = getattr(values, "_owned", None)
    if owned is not None:
        from vibespatial.constructive.validity import is_valid_owned

        subset = owned.take(np.asarray(row_indices, dtype=np.int64))
        valid_mask = np.asarray(is_valid_owned(subset), dtype=bool)
        if not bool(np.all(subset.validity)):
            valid_mask = valid_mask.copy()
            valid_mask[~subset.validity] = True
        return bool(valid_mask.all())

    return bool(series.iloc[np.asarray(row_indices, dtype=np.intp)].is_valid.all())


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


def _polygon_rect_overlap_mask(geometries: GeoSeries) -> np.ndarray | None:
    """Return normalized rectangle-overlap metadata when present."""
    mask = getattr(geometries.values, "_polygon_rect_boundary_overlap", None)
    if mask is None:
        owned = getattr(geometries.values, "_owned", None)
        if owned is not None:
            mask = getattr(owned, "_polygon_rect_boundary_overlap", None)
    if mask is None:
        return None
    mask = mask.get() if hasattr(mask, "get") else np.asarray(mask)
    mask = np.asarray(mask, dtype=bool)
    if mask.size != len(geometries):
        return None
    return mask


def _polygon_rect_exact_polygon_only_mask(geometries: GeoSeries) -> np.ndarray | None:
    """Return rows whose rectangle fast path is known polygon-complete."""
    mask = getattr(geometries.values, "_polygon_rect_exact_polygon_only", None)
    if mask is None:
        owned = getattr(geometries.values, "_owned", None)
        if owned is not None:
            mask = getattr(owned, "_polygon_rect_exact_polygon_only", None)
    if mask is None:
        return None
    mask = mask.get() if hasattr(mask, "get") else np.asarray(mask)
    mask = np.asarray(mask, dtype=bool)
    if mask.size != len(geometries):
        return None
    return mask


def _can_defer_make_valid_to_rect_repair(geometries: GeoSeries) -> bool:
    """Return True when targeted rectangle repair can replace generic make_valid."""
    return _polygon_rect_overlap_mask(geometries) is not None


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


def _owned_valid_nonempty_mask(owned) -> np.ndarray:
    """Return a public-boundary keep mask without materializing full host geometry."""
    from vibespatial.geometry.owned import FAMILY_TAGS

    validity = np.asarray(owned.validity, dtype=bool)
    if not validity.any():
        return validity

    tags = np.asarray(owned.tags)
    row_offsets = np.asarray(owned.family_row_offsets)
    keep_mask = validity.copy()

    for family in owned.families:
        owned._ensure_host_family_structure(family)
        family_tag = FAMILY_TAGS[family]
        family_mask = validity & (tags == family_tag)
        if not family_mask.any():
            continue
        family_rows = row_offsets[family_mask]
        empty_mask = np.asarray(owned.families[family].empty_mask, dtype=bool)
        if empty_mask.size == 0 or np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
            # Fall back to the GeometryArray path if metadata is inconsistent.
            return validity & ~np.asarray(GeometryArray.from_owned(owned).is_empty, dtype=bool)
        keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

    return keep_mask


def _geometry_native_result_from_geoseries(geoseries: GeoSeries) -> GeometryNativeResult:
    owned = getattr(geoseries.values, "_owned", None)
    if owned is not None:
        return GeometryNativeResult.from_owned(owned, crs=geoseries.crs)
    return GeometryNativeResult.from_geoseries(geoseries)


def _extract_owned_pair(df1, df2):
    """Return (left_owned, right_owned) if both DataFrames have owned backing, else (None, None)."""
    ga1 = df1.geometry.values
    ga2 = df2.geometry.values
    left_owned = getattr(ga1, '_owned', None)
    right_owned = getattr(ga2, '_owned', None)
    if (
        _series_all_polygons(df1.geometry)
        and _series_all_polygons(df2.geometry)
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
            if has_gpu_runtime():
                from vibespatial.runtime.residency import Residency, TransferTrigger

                if (
                    ga1.__class__.__name__ == "DeviceGeometryArray"
                    or ga2.__class__.__name__ == "DeviceGeometryArray"
                    or getattr(left_owned, "residency", None) is Residency.DEVICE
                    or getattr(right_owned, "residency", None) is Residency.DEVICE
                ):
                    if left_owned.residency is not Residency.DEVICE:
                        left_owned = left_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason=(
                                "overlay kept polygon pair on device after one input "
                                "already selected the device-native path"
                            ),
                        )
                    if right_owned.residency is not Residency.DEVICE:
                        right_owned = right_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason=(
                                "overlay kept polygon pair on device after one input "
                                "already selected the device-native path"
                            ),
                        )
            return left_owned, right_owned
    return None, None


def _should_prefer_exact_polygon_gpu(
    df1,
    df2,
    left_owned,
    right_owned,
    *,
    keep_geom_type,
) -> bool:
    """Prefer the exact GPU polygon boundary whenever both polygon inputs have owned backing.

    Once we can represent both sides as owned polygon arrays, the exact
    polygon-intersection path should stay in the native execution family
    instead of dropping back to the host exact-intersection boundary.
    This keeps cheap host-to-owned polygon cases on the GPU path as well.
    """
    if keep_geom_type is False or not has_gpu_runtime():
        return False
    if not (
        _series_all_polygons(df1.geometry)
        and _series_all_polygons(df2.geometry)
    ):
        return False
    if strict_native_mode_enabled():
        return True
    return left_owned is not None and right_owned is not None


def _should_use_owned_constructive_overlay(left_owned, right_owned) -> bool:
    """Use the owned constructive overlay path only when the workflow is truly device-native.

    Host-resident ``_owned`` backings on plain GeoPandas inputs are still a
    transitional cache layer, not a stable public constructive execution model.
    Auto-mode public overlay should only enter the owned constructive path when
    strict-native mode requires it or the data already lives on device.
    """
    if left_owned is None or right_owned is None:
        return False
    if strict_native_mode_enabled():
        return True

    from vibespatial.runtime.residency import Residency, combined_residency

    return combined_residency(left_owned, right_owned) is Residency.DEVICE


def _coerce_owned_pair_for_strict_overlay(df1, df2, left_owned, right_owned):
    """Materialize owned backing for strict overlay paths when GPU is available.

    Overlay does its spatial join before pairwise constructive work, so this
    coercion stays off the hot-path for non-strict runs. In strict mode we need
    the downstream pairwise overlay operations to use the repo-owned GPU
    dispatch instead of inheriting the generic small-workload crossover.
    """
    if not strict_native_mode_enabled() or not has_gpu_runtime():
        return left_owned, right_owned
    from vibespatial.runtime.residency import Residency, TransferTrigger

    try:
        if left_owned is None:
            left_owned = df1.geometry.values.to_owned()
        if right_owned is None:
            right_owned = df2.geometry.values.to_owned()
        if left_owned is not None and left_owned.residency is not Residency.DEVICE:
            left_owned = left_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="strict overlay coerced owned left input to device residency",
            )
        if right_owned is not None and right_owned.residency is not Residency.DEVICE:
            right_owned = right_owned.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="strict overlay coerced owned right input to device residency",
            )
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
    from vibespatial.constructive.binary_constructive import binary_constructive_owned
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

    def _repair_invalid_rows_with_group_union(diff_owned):
        from vibespatial.constructive.binary_constructive import binary_constructive_owned
        from vibespatial.constructive.validity import is_valid_owned
        from vibespatial.geometry.owned import concat_owned_scatter
        from vibespatial.kernels.constructive.segmented_union import segmented_union_all

        invalid_mask = ~np.asarray(is_valid_owned(diff_owned), dtype=bool)
        if not bool(np.any(invalid_mask)):
            return diff_owned, 0

        invalid_rows = np.flatnonzero(invalid_mask).astype(np.int64, copy=False)
        starts = group_offsets[invalid_rows]
        stops = group_offsets[invalid_rows + 1]
        counts = (stops - starts).astype(np.int64, copy=False)
        if not bool(np.all(counts > 0)):
            return diff_owned, 0

        right_indices = np.concatenate(
            [
                np.arange(start, stop, dtype=np.int64)
                for start, stop in zip(starts, stops, strict=True)
            ]
        )
        invalid_group_offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(counts, dtype=np.int64),
            ]
        )
        unioned_right = segmented_union_all(
            right_batch.take(right_indices),
            invalid_group_offsets,
            dispatch_mode=dispatch_mode,
        )
        repaired_rows = binary_constructive_owned(
            "difference",
            left_batch.take(invalid_rows),
            unioned_right,
            dispatch_mode=dispatch_mode,
            _skip_single_row_polygon_difference_exact_correction=True,
        )
        repaired_valid_mask = ~np.asarray(is_valid_owned(repaired_rows), dtype=bool)
        if bool(np.any(repaired_valid_mask)):
            return diff_owned, 0
        return concat_owned_scatter(diff_owned, repaired_rows, invalid_rows), int(
            invalid_rows.size
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
        try:
            with hotpath_stage("overlay.diff.grouped_plan.build", category="setup"):
                plan = _build_overlay_execution_plan(
                    left_batch,
                    right_batch,
                    dispatch_mode=dispatch_mode,
                    _cached_right_segments=None,
                    _row_isolated=True,
                    _right_geometry_source_rows=right_group_rows,
                )
        except Exception as exc:
            logger.debug(
                "grouped exact overlay plan build failed; falling back to sequential exact difference",
                exc_info=True,
            )
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_plan_build_failed_gpu",
                reason=(
                    "grouped exact overlay plan build failed and fell back to sequential exact difference"
                ),
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}, "
                    f"error={type(exc).__name__}: {exc}"
                ),
                requested=dispatch_mode,
                selected=(
                    ExecutionMode.GPU
                    if dispatch_mode is not ExecutionMode.CPU and has_gpu_runtime()
                    else ExecutionMode.CPU
                ),
            )
            return _sequential_grouped_difference_owned(
                left_batch,
                right_batch,
                group_offsets,
                dispatch_mode=dispatch_mode,
            )
        _sync_hotpath()
        try:
            with hotpath_stage("overlay.diff.grouped_plan.materialize", category="refine"):
                diff_owned, _selected = _materialize_overlay_execution_plan(
                    plan,
                    operation="difference",
                    requested=ExecutionMode.GPU,
                    preserve_row_count=left_batch.row_count,
                )
        except Exception as exc:
            logger.debug(
                "grouped exact overlay plan materialization failed; falling back to sequential exact difference",
                exc_info=True,
            )
            record_dispatch_event(
                surface="geopandas.array.difference",
                operation="difference",
                implementation="grouped_overlay_difference_materialize_failed_gpu",
                reason=(
                    "grouped exact overlay plan materialization failed and fell back to sequential exact difference"
                ),
                detail=(
                    f"groups={left_batch.row_count}, "
                    f"pairs={right_batch.row_count}, "
                    f"error={type(exc).__name__}: {exc}"
                ),
                requested=dispatch_mode,
                selected=(
                    ExecutionMode.GPU
                    if dispatch_mode is not ExecutionMode.CPU and has_gpu_runtime()
                    else ExecutionMode.CPU
                ),
            )
            return _sequential_grouped_difference_owned(
                left_batch,
                right_batch,
                group_offsets,
                dispatch_mode=dispatch_mode,
            )
        _sync_hotpath()
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation=(
                "grouped_overlay_difference_gpu"
                if _selected is ExecutionMode.GPU
                else "grouped_overlay_difference_cpu"
            ),
            reason=(
                "grouped exact overlay difference used one row-isolated overlay plan"
                if _selected is ExecutionMode.GPU
                else "grouped exact overlay difference materialized off GPU"
            ),
            detail=(
                f"groups={left_batch.row_count}, "
                f"pairs={right_batch.row_count}"
            ),
            requested=dispatch_mode,
            selected=_selected,
        )
        if diff_owned.row_count != left_batch.row_count:
            raise RuntimeError(
                "grouped overlay difference produced "
                f"{diff_owned.row_count} rows for {left_batch.row_count} groups"
            )
        repaired_invalid_rows = 0
        if diff_owned.row_count > 0:
            diff_owned, repaired_invalid_rows = _repair_invalid_rows_with_group_union(
                diff_owned
            )
            if repaired_invalid_rows > 0:
                record_dispatch_event(
                    surface="geopandas.array.difference",
                    operation="difference",
                    implementation="grouped_union_difference_gpu",
                    reason=(
                        "invalid grouped difference rows were repaired with grouped union "
                        "plus exact difference"
                    ),
                    detail=(
                        f"groups={left_batch.row_count}, "
                        f"pairs={right_batch.row_count}, "
                        f"repaired_rows={repaired_invalid_rows}"
                    ),
                    requested=dispatch_mode,
                    selected=ExecutionMode.GPU,
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
    except Exception as exc:
        logger.debug(
            "grouped exact overlay plan failed; falling back to sequential exact difference",
            exc_info=True,
        )
        record_dispatch_event(
            surface="geopandas.array.difference",
            operation="difference",
            implementation="grouped_overlay_difference_postcheck_failed_gpu",
            reason=(
                "grouped exact overlay plan post-check failed and fell back to sequential exact difference"
            ),
            detail=(
                f"groups={left_batch.row_count}, "
                f"pairs={right_batch.row_count}, "
                f"error={type(exc).__name__}: {exc}"
            ),
            requested=dispatch_mode,
            selected=(
                ExecutionMode.GPU
                if dispatch_mode is not ExecutionMode.CPU and has_gpu_runtime()
                else ExecutionMode.CPU
            ),
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
    from vibespatial.runtime.residency import Residency, TransferTrigger

    xp = np
    if hasattr(idx1, "__cuda_array_interface__"):
        try:
            import cupy
            xp = cupy
        except ImportError:
            pass
    try:
        import cupy as cp
    except ImportError:  # pragma: no cover - exercised on CPU-only installs
        cp = None

    use_device_gather = (
        cp is not None
        and _pairwise_mode is not ExecutionMode.CPU
        and has_gpu_runtime()
    )
    if use_device_gather:
        left_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="overlay difference grouped left gather",
        )
        right_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="overlay difference grouped right gather",
        )

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
            if use_device_gather and _has_device_indices:
                right_gathered = right_owned.device_take(d_idx2)
            elif use_device_gather:
                right_gathered = right_owned.device_take(cp.asarray(idx2, dtype=cp.int64))
            else:
                right_gathered = right_owned.take(idx2)

        _sync_hotpath()
        with hotpath_stage("overlay.diff.single_batch.left_take", category="refine"):
            if use_device_gather:
                left_sub = left_owned.device_take(cp.asarray(idx1_unique, dtype=cp.int64))
            else:
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
            if use_device_gather:
                right_gathered = right_owned.device_take(cp.asarray(batch_idx2, dtype=cp.int64))
            else:
                right_gathered = right_owned.take(batch_idx2)

        # Build local group offsets for this batch (0-based)
        batch_group_starts = h_group_offsets[batch_start:batch_end + 1] - pair_start
        batch_group_offsets = np.asarray(batch_group_starts, dtype=np.int64)

        # Take the corresponding left geometries
        batch_left_indices = h_idx1_unique[batch_start:batch_end]
        _sync_hotpath()
        with hotpath_stage("overlay.diff.batch.left_take", category="refine"):
            if use_device_gather:
                left_sub = left_owned.device_take(cp.asarray(batch_left_indices, dtype=cp.int64))
            else:
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


def _make_valid_geoseries(gs, *, dispatch_mode: ExecutionMode | str = ExecutionMode.AUTO):
    """Apply make_valid to polygon rows of a GeoSeries, preferring GPU path.

    When the GeoSeries has owned backing, routes through make_valid_owned to
    keep data device-resident and avoid Shapely materialisation.  Falls back
    to the standard GeoSeries.make_valid() path otherwise.
    """
    ga = gs.values
    owned = getattr(ga, '_owned', None)
    poly_ix = _series_polygon_mask(gs)
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
        from vibespatial.constructive.validity import is_valid_owned

        valid_mask = np.asarray(is_valid_owned(owned), dtype=bool)
        if not bool(np.all(owned.validity)):
            valid_mask = valid_mask.copy()
            valid_mask[~owned.validity] = True
        if valid_mask.all():
            return gs

        requested_mode = (
            dispatch_mode
            if isinstance(dispatch_mode, ExecutionMode)
            else ExecutionMode(dispatch_mode)
        )
        mv_result = make_valid_owned(owned=owned, dispatch_mode=dispatch_mode)
        if mv_result.repaired_rows.size > 0:
            remaining_invalid = None
            if mv_result.owned is not None:
                remaining_invalid = ~np.asarray(is_valid_owned(mv_result.owned), dtype=bool)
                if not bool(np.all(mv_result.owned.validity)):
                    remaining_invalid = remaining_invalid.copy()
                    remaining_invalid[~mv_result.owned.validity] = False
                if remaining_invalid.any():
                    if requested_mode is ExecutionMode.GPU:
                        record_fallback_event(
                            surface="geopandas.array.make_valid",
                            reason="owned GPU make_valid left invalid polygon rows after repair",
                            detail=(
                                f"rows={owned.row_count}, "
                                f"remaining_invalid={int(np.count_nonzero(remaining_invalid))}"
                            ),
                            requested=dispatch_mode,
                            selected=ExecutionMode.CPU,
                            pipeline="overlay.make_valid",
                            d2h_transfer=True,
                        )
                    else:
                        record_dispatch_event(
                            surface="geopandas.array.make_valid",
                            operation="make_valid",
                            implementation="shapely.make_valid_compatibility",
                            reason=(
                                "owned make_valid left invalid rows after AUTO repair, "
                                "so compatibility cleanup remained on host"
                            ),
                            detail=(
                                f"rows={owned.row_count}, "
                                f"remaining_invalid={int(np.count_nonzero(remaining_invalid))}"
                            ),
                            requested=dispatch_mode,
                            selected=ExecutionMode.CPU,
                        )
                    repaired_values = np.asarray(mv_result.geometries, dtype=object)
                    repaired_values[remaining_invalid] = np.asarray(
                        shapely.make_valid(repaired_values[remaining_invalid]),
                        dtype=object,
                    )
                    record_dispatch_event(
                        surface="geopandas.overlay",
                        operation="make_valid",
                        implementation="shapely.make_valid_fallback",
                        reason="owned GPU make_valid left invalid polygon rows after repair",
                        detail=(
                            f"rows={owned.row_count}, "
                            f"remaining_invalid={int(np.count_nonzero(remaining_invalid))}"
                        ),
                        selected=ExecutionMode.CPU,
                    )
                    return GeoSeries(GeometryArray(repaired_values, crs=ga.crs), index=gs.index)
            # Repair happened — prefer device-resident .owned to avoid D->H.
            if mv_result.owned is not None:
                try:
                    _seed_all_validity_cache_if_owned(mv_result.owned)
                    new_ga = GeometryArray.from_owned(mv_result.owned, crs=ga.crs)
                    return GeoSeries(new_ga, index=gs.index)
                except (NotImplementedError, Exception) as exc:
                    if requested_mode is ExecutionMode.GPU:
                        record_fallback_event(
                            surface="geopandas.array.make_valid",
                            reason=(
                                "owned make_valid result could not be rebuilt without "
                                "host materialization"
                            ),
                            detail=(
                                f"rows={owned.row_count}, "
                                f"error={type(exc).__name__}: {exc}"
                            ),
                            requested=dispatch_mode,
                            selected=ExecutionMode.CPU,
                            pipeline="overlay.make_valid",
                            d2h_transfer=True,
                        )
                    else:
                        record_dispatch_event(
                            surface="geopandas.array.make_valid",
                            operation="make_valid",
                            implementation="host_rewrap_compatibility",
                            reason=(
                                "owned make_valid result could not be rebuilt under AUTO, "
                                "so compatibility materialization stayed on host"
                            ),
                            detail=(
                                f"rows={owned.row_count}, "
                                f"error={type(exc).__name__}: {exc}"
                            ),
                            requested=dispatch_mode,
                            selected=ExecutionMode.CPU,
                        )
            # Fallback: rebuild from Shapely geometries
            try:
                from vibespatial.geometry.owned import from_shapely_geometries

                new_owned = from_shapely_geometries(list(mv_result.geometries))
                _seed_all_validity_cache_if_owned(new_owned)
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
    # ADR-0042 low-level contract: spatial indexing still produces index arrays.
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
    if request_device:
        left_query_geoms = GeoSeries(
            GeometryArray.from_owned(left_owned, crs=df1.crs),
            crs=df1.crs,
        )
        result = df2.sindex.query(
            left_query_geoms,
            predicate="intersects",
            sort=True,
            return_device=True,
        )
        return result

    result = df2.sindex.query(
        df1.geometry,
        predicate="intersects",
        sort=True,
        return_device=False,
    )
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
    """ADR-0042 transitional boundary: attribute assembly from index arrays.

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


def _intersection_attribute_columns(df1: GeoDataFrame, df2: GeoDataFrame) -> list[str]:
    """Return the public intersection attribute schema without materializing rows."""
    empty = np.empty(0, dtype=np.int32)
    columns = _assemble_intersection_attributes(
        empty,
        empty,
        df1.drop(df1._geometry_column_name, axis=1),
        df2.drop(df2._geometry_column_name, axis=1),
    ).columns
    return list(columns)


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
            edge_parts = shapely.get_parts(np.asarray([edge_geom], dtype=object))
            if len(edge_parts) > 0:
                cleaned_parts = np.asarray(
                    shapely.difference(
                        edge_parts,
                        np.full(len(edge_parts), area_geom.boundary, dtype=object),
                    ),
                    dtype=object,
                )
                edge_parts = shapely.get_parts(cleaned_parts)
                edge_parts = edge_parts[~shapely.is_empty(edge_parts)]
                edge_geom = shapely.union_all(edge_parts) if len(edge_parts) > 0 else None
            else:
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


def _count_dropped_polygon_parts_from_exact_values(
    exact_values: np.ndarray,
) -> int:
    """Count dropped lower-dimensional pieces from exact host intersection output."""
    if len(exact_values) == 0:
        return 0

    missing_mask = shapely.is_missing(exact_values) | shapely.is_empty(exact_values)
    type_ids = shapely.get_type_id(exact_values)
    polygon_mask = (
        (type_ids == _SHAPELY_TYPE_ID_POLYGON)
        | (type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
    )
    collection_mask = type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION

    dropped = int(np.count_nonzero(~missing_mask & ~polygon_mask & ~collection_mask))
    if collection_mask.any():
        dropped += _count_non_polygon_collection_parts(exact_values[collection_mask])
    return dropped


def _count_dropped_polygon_intersection_parts(
    left_values: np.ndarray,
    right_values: np.ndarray,
    row_count: int,
    *,
    exact_values: np.ndarray | None = None,
) -> int:
    """Count lower-dimensional exact-intersection output dropped by keep_geom_type.

    Warning parity needs the exact host intersection shape, not just the
    polygon-only area output retained by the fast native path.
    """
    if row_count == 0:
        return 0

    if exact_values is None:
        exact_values = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    else:
        exact_values = np.asarray(exact_values, dtype=object)
    return _count_dropped_polygon_parts_from_exact_values(exact_values)


def _warning_candidate_mask_for_polygon_keep_geom_type(
    left_values: np.ndarray,
    right_values: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Return rows that can affect the polygon keep-geom-type warning count.

    Rows already dropped by the polygon area filter always need exact host
    classification. Rows retained as polygonal area only matter when their
    boundaries still intersect, because that is the only way the exact
    intersection can carry lower-dimensional extras inside a kept row.
    """
    suspect_mask = ~keep_mask
    kept_rows = np.flatnonzero(keep_mask)
    if kept_rows.size == 0:
        return suspect_mask

    kept_left = left_values[kept_rows]
    kept_right = right_values[kept_rows]
    boundary_overlap = shapely.intersects(
        shapely.boundary(kept_left),
        shapely.boundary(kept_right),
    )
    if np.any(boundary_overlap):
        suspect_mask = suspect_mask.copy()
        suspect_mask[kept_rows[np.asarray(boundary_overlap, dtype=bool)]] = True
    return suspect_mask


def _warning_candidate_mask_from_exact_intersection_values(
    exact_values: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    """Classify keep-geom-type warning rows directly from exact host output."""
    exact_values = np.asarray(exact_values, dtype=object)
    keep_mask = np.asarray(keep_mask, dtype=bool)
    if exact_values.size != keep_mask.size:
        raise ValueError("exact_values and keep_mask must have the same length")

    warning_mask = ~keep_mask
    kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
    if kept_rows.size == 0:
        return warning_mask

    kept_values = exact_values[kept_rows]
    missing_mask = shapely.is_missing(kept_values) | shapely.is_empty(kept_values)
    type_ids = shapely.get_type_id(kept_values)
    polygon_mask = (
        (type_ids == _SHAPELY_TYPE_ID_POLYGON)
        | (type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
    )
    collection_mask = type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION
    kept_warning_mask = ~missing_mask & ~polygon_mask & ~collection_mask

    if collection_mask.any():
        for local_row in np.flatnonzero(collection_mask).astype(np.intp, copy=False):
            kept_warning_mask[local_row] = (
                _count_non_polygon_collection_parts(
                    np.asarray([kept_values[local_row]], dtype=object)
                )
                > 0
            )

    warning_mask = warning_mask.copy()
    warning_mask[kept_rows] = kept_warning_mask
    return warning_mask


def _owned_non_empty_row_mask(owned) -> np.ndarray | None:
    """Return a host boolean mask for non-empty rows in an owned array."""
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.geometry.owned import FAMILY_TAGS

    tags = np.asarray(owned.tags)
    validity = np.asarray(owned.validity, dtype=bool)
    row_offsets = np.asarray(owned.family_row_offsets, dtype=np.int64)
    keep_mask = np.zeros(len(tags), dtype=bool)

    for family, buffer in owned.families.items():
        family_tag = FAMILY_TAGS[family]
        family_mask = validity & (tags == family_tag)
        if not family_mask.any():
            continue
        family_rows = row_offsets[family_mask]
        empty_mask = np.asarray(buffer.empty_mask, dtype=bool)
        if empty_mask.size == 0:
            family_count = int(getattr(buffer, "row_count", 0))
            if np.any((family_rows < 0) | (family_rows >= family_count)):
                return None
            if family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
                return None
            keep_mask[np.flatnonzero(family_mask)] = True
            continue
        if np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
            return None
        keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

    return keep_mask


def _device_count_dropped_polygon_intersection_warning_rows(
    area_owned,
    keep_mask: np.ndarray,
    warning_rows: np.ndarray,
    *,
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> int | None:
    """Count dropped lower-dimensional warning pieces without host probing.

    The current polygonal fast path keeps only area output. For keep-geom-type
    warnings we need the lower-dimensional boundary remnants that were dropped.
    When the source polygons and retained area are still device-backed, recover
    those remnants on device via boundary intersection and boundary difference.
    Return ``None`` when the warning rows still require the existing explicit
    host semantic probe. The device path now accepts arbitrary aligned pair
    rows instead of only the old tiny many-vs-one warning subset.
    """
    if (
        warning_rows.size == 0
        or not has_gpu_runtime()
    ):
        return None
    left_source_owned = getattr(left_source.values, "_owned", None) if left_source is not None else None
    right_source_owned = getattr(right_source.values, "_owned", None) if right_source is not None else None
    left_pairs_owned = getattr(left_pairs.values, "_owned", None) if left_pairs is not None else None
    right_pairs_owned = getattr(right_pairs.values, "_owned", None) if right_pairs is not None else None
    if area_owned is None:
        return None

    from vibespatial.constructive.binary_constructive import binary_constructive_owned
    from vibespatial.constructive.boundary import boundary_owned
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.predicates.binary import evaluate_binary_predicate
    from vibespatial.runtime.residency import Residency

    warning_keep_mask = np.asarray(keep_mask[warning_rows], dtype=bool)
    if (
        left_source_owned is not None
        and right_source_owned is not None
        and left_rows is not None
        and right_rows is not None
        and getattr(left_source_owned, "residency", None) is Residency.DEVICE
        and getattr(right_source_owned, "residency", None) is Residency.DEVICE
    ):
        device_left_owned = left_source_owned
        device_right_owned = right_source_owned
        warning_left_rows = np.asarray(left_rows, dtype=np.intp)[warning_rows]
        warning_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
    elif (
        left_pairs_owned is not None
        and right_pairs_owned is not None
        and getattr(left_pairs_owned, "residency", None) is Residency.DEVICE
        and getattr(right_pairs_owned, "residency", None) is Residency.DEVICE
    ):
        device_left_owned = left_pairs_owned
        device_right_owned = right_pairs_owned
        warning_left_rows = warning_rows.astype(np.intp, copy=False)
        warning_right_rows = warning_rows.astype(np.intp, copy=False)
    else:
        return None
    if np.any(warning_keep_mask) and getattr(area_owned, "residency", None) is not Residency.DEVICE:
        return None
    if warning_rows.size > 128:
        # Large keep-geom-type warning batches are advisory only. Once the result
        # is already device-native, keep big rect-overlap refinements on device
        # via the conservative native count instead of risking a mid-helper CPU
        # fallback on mixed lower-dimensional boundary cases.
        return None

    try:
        import cupy as cp

        def _take_owned_rows(owned, rows: np.ndarray):
            rows64 = np.asarray(rows, dtype=np.int64)
            if rows64.size == 0:
                return owned.take(rows64)
            return owned.device_take(cp.asarray(rows64, dtype=cp.int64))

        def _has_only_linear_boundary_families(owned) -> bool:
            return set(owned.families.keys()).issubset(
                {GeometryFamily.LINESTRING, GeometryFamily.MULTILINESTRING}
            )

        dropped = 0
        chunk_rows = 256
        for start in range(0, warning_rows.size, chunk_rows):
            stop = min(start + chunk_rows, warning_rows.size)
            chunk_warning_rows = warning_rows[start:stop]
            chunk_keep_mask = warning_keep_mask[start:stop]
            chunk_left_rows = warning_left_rows[start:stop]
            chunk_right_rows = warning_right_rows[start:stop]

            warning_left = _take_owned_rows(
                device_left_owned,
                chunk_left_rows,
            )
            warning_right = _take_owned_rows(
                device_right_owned,
                chunk_right_rows,
            )
            warning_left_boundary = boundary_owned(warning_left)
            warning_right_boundary = boundary_owned(warning_right)
            if (
                not _has_only_linear_boundary_families(warning_left_boundary)
                or not _has_only_linear_boundary_families(warning_right_boundary)
            ):
                return None
            boundary_overlap = binary_constructive_owned(
                "intersection",
                warning_left_boundary,
                warning_right_boundary,
                dispatch_mode=ExecutionMode.GPU,
            )
            if boundary_overlap.row_count != chunk_warning_rows.size:
                return None

            boundary_non_empty = _owned_non_empty_row_mask(boundary_overlap)
            if boundary_non_empty is None:
                return None

            dropped += int(np.count_nonzero(boundary_non_empty & ~chunk_keep_mask))

            kept_local_rows = np.flatnonzero(
                chunk_keep_mask & boundary_non_empty
            ).astype(np.int64, copy=False)
            if kept_local_rows.size == 0:
                continue

            kept_boundary_overlap = boundary_overlap.device_take(
                cp.asarray(kept_local_rows, dtype=cp.int64)
            )
            kept_area = _take_owned_rows(area_owned, chunk_warning_rows[kept_local_rows])
            kept_area_boundary = boundary_owned(kept_area)
            covered_mask = np.asarray(
                evaluate_binary_predicate(
                    "covered_by",
                    kept_boundary_overlap,
                    kept_area_boundary,
                    dispatch_mode=ExecutionMode.GPU,
                ).values,
                dtype=bool,
            )
            if covered_mask.size != kept_local_rows.size:
                return None
            dropped += int(np.count_nonzero(~covered_mask))
        return dropped
    except Exception:
        logger.debug(
            "device-native keep_geom_type warning count failed; falling back to host semantic probe",
            exc_info=True,
        )
        return None


def _device_polygon_keep_geom_type_cover_mask(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    warning_rows: np.ndarray,
    *,
    area_owned=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> np.ndarray | None:
    """Return rows provably equal to one source polygon from device predicates.

    When one polygon covers the other, the exact polygon-polygon intersection is
    exactly the covered polygon, so keep-geom-type warning classification does
    not need a host semantic probe for that row.
    """
    if warning_rows.size == 0 or not has_gpu_runtime():
        return None

    left_source_owned = getattr(left_source.values, "_owned", None) if left_source is not None else None
    right_source_owned = getattr(right_source.values, "_owned", None) if right_source is not None else None
    left_pairs_owned = getattr(left_pairs.values, "_owned", None) if left_pairs is not None else None
    right_pairs_owned = getattr(right_pairs.values, "_owned", None) if right_pairs is not None else None

    from vibespatial.runtime.residency import Residency

    use_source_rows = (
        left_source is not None
        and right_source is not None
        and left_rows is not None
        and right_rows is not None
        and left_source_owned is not None
        and right_source_owned is not None
        and left_source_owned.residency is Residency.DEVICE
        and right_source_owned.residency is Residency.DEVICE
    )
    use_pair_rows = (
        left_pairs_owned is not None
        and right_pairs_owned is not None
        and left_pairs_owned.residency is Residency.DEVICE
        and right_pairs_owned.residency is Residency.DEVICE
    )
    if not use_source_rows and not use_pair_rows:
        return None

    try:
        import cupy as cp

        from vibespatial.constructive.measurement import area_owned as measure_area_owned
        from vibespatial.predicates.binary import evaluate_binary_predicate

        def _take_owned_rows(owned, rows: np.ndarray):
            if rows.size == 0:
                return owned.take(rows.astype(np.int64, copy=False))
            return owned.device_take(cp.asarray(rows, dtype=cp.int64))

        def _area_candidate_masks(left_owned, right_input) -> tuple[np.ndarray, np.ndarray]:
            left_area = np.asarray(measure_area_owned(left_owned), dtype=np.float64)
            right_area = np.asarray(measure_area_owned(right_input), dtype=np.float64)
            if right_area.size == 1 and left_area.size > 1:
                right_area = np.full(left_area.shape, float(right_area[0]), dtype=np.float64)
            scale = np.maximum(np.abs(left_area), np.abs(right_area))
            tol = np.maximum(scale * 1.0e-12, 1.0e-12)
            maybe_covers = left_area + tol >= right_area
            maybe_covered_by = left_area <= right_area + tol
            return maybe_covers, maybe_covered_by

        def _subset_right_input(right_input, row_mask: np.ndarray):
            if right_input.row_count == 1 or row_mask.all():
                return right_input
            return _take_owned_rows(
                right_input,
                np.flatnonzero(row_mask).astype(np.int64, copy=False),
            )

        skip_covered_by_probe = bool(
            getattr(area_owned, "_many_vs_one_left_containment_bypass_applied", False)
        )

        def _evaluate_cover_mask(left_owned, right_input) -> np.ndarray:
            maybe_covers, maybe_covered_by = _area_candidate_masks(left_owned, right_input)
            cover_mask = np.zeros(left_owned.row_count, dtype=bool)

            if maybe_covers.any():
                left_eval = (
                    left_owned
                    if maybe_covers.all()
                    else _take_owned_rows(
                        left_owned,
                        np.flatnonzero(maybe_covers).astype(np.int64, copy=False),
                    )
                )
                covers = np.asarray(
                    evaluate_binary_predicate(
                        "covers",
                        left_eval,
                        _subset_right_input(right_input, maybe_covers),
                        dispatch_mode=ExecutionMode.GPU,
                    ).values,
                    dtype=bool,
                )
                if maybe_covers.all():
                    cover_mask |= covers
                else:
                    cover_mask[np.flatnonzero(maybe_covers).astype(np.intp, copy=False)] |= covers

            if not skip_covered_by_probe and maybe_covered_by.any():
                left_eval = (
                    left_owned
                    if maybe_covered_by.all()
                    else _take_owned_rows(
                        left_owned,
                        np.flatnonzero(maybe_covered_by).astype(np.int64, copy=False),
                    )
                )
                covered_by = np.asarray(
                    evaluate_binary_predicate(
                        "covered_by",
                        left_eval,
                        _subset_right_input(right_input, maybe_covered_by),
                        dispatch_mode=ExecutionMode.GPU,
                    ).values,
                    dtype=bool,
                )
                if maybe_covered_by.all():
                    cover_mask |= covered_by
                else:
                    cover_mask[np.flatnonzero(maybe_covered_by).astype(np.intp, copy=False)] |= covered_by

            return cover_mask

        if use_source_rows:
            left_owned = left_source_owned
            right_owned = right_source_owned
            source_left_rows = np.asarray(left_rows, dtype=np.intp)[warning_rows]
            source_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
        else:
            left_owned = left_pairs_owned
            right_owned = right_pairs_owned
            source_left_rows = warning_rows.astype(np.intp, copy=False)
            source_right_rows = warning_rows.astype(np.intp, copy=False)

        unique_left_rows = np.unique(source_left_rows)
        unique_right_rows = np.unique(source_right_rows)
        if unique_right_rows.size == 1 and source_left_rows.size > 1:
            left_eval = _take_owned_rows(left_owned, source_left_rows)
            right_one = _take_owned_rows(right_owned, unique_right_rows)
            return _evaluate_cover_mask(left_eval, right_one)

        if unique_left_rows.size == 1 and source_right_rows.size > 1:
            right_eval = _take_owned_rows(right_owned, source_right_rows)
            left_one = _take_owned_rows(left_owned, unique_left_rows)
            return _evaluate_cover_mask(right_eval, left_one)

        left_eval = _take_owned_rows(left_owned, source_left_rows)
        right_eval = _take_owned_rows(right_owned, source_right_rows)
        return _evaluate_cover_mask(left_eval, right_eval)
    except Exception:
        logger.debug(
            "device-native keep_geom_type cover classification failed; "
            "falling back to semantic probe",
            exc_info=True,
        )
        return None


def _clear_device_exact_keep_geom_type_warnings(
    warning_mask: np.ndarray,
    keep_mask: np.ndarray,
    *,
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    area_owned=None,
    left_pairs: GeoSeries | None = None,
    right_pairs: GeoSeries | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop warning rows that device predicates prove do not need host probing."""
    warning_rows = np.flatnonzero(warning_mask).astype(np.intp, copy=False)
    if warning_rows.size == 0:
        return warning_mask, warning_rows

    kept_warning_rows = warning_rows[np.asarray(keep_mask[warning_rows], dtype=bool)]
    if kept_warning_rows.size == 0:
        return warning_mask, warning_rows

    if not _many_vs_one_keep_geom_type_cover_probe_needed(
        left_source,
        right_source,
        left_rows,
        right_rows,
        kept_warning_rows,
        area_owned=area_owned,
    ):
        return warning_mask, warning_rows

    cover_mask = _device_polygon_keep_geom_type_cover_mask(
        left_source,
        right_source,
        left_rows,
        right_rows,
        kept_warning_rows,
        area_owned=area_owned,
        left_pairs=left_pairs,
        right_pairs=right_pairs,
    )
    if cover_mask is None or cover_mask.size != kept_warning_rows.size:
        return warning_mask, warning_rows

    resolved_rows = np.asarray(cover_mask, dtype=bool)
    if not resolved_rows.any():
        return warning_mask, warning_rows

    warning_mask = np.asarray(warning_mask, dtype=bool).copy()
    warning_mask[kept_warning_rows[resolved_rows]] = False
    return warning_mask, np.flatnonzero(warning_mask).astype(np.intp, copy=False)


def _many_vs_one_keep_geom_type_cover_probe_needed(
    left_source: GeoSeries | None,
    right_source: GeoSeries | None,
    left_rows,
    right_rows,
    warning_rows: np.ndarray,
    *,
    area_owned=None,
) -> bool:
    """Return True when kept warning rows might still cover the single right polygon."""
    if warning_rows.size == 0:
        return False
    if not bool(getattr(area_owned, "_many_vs_one_left_containment_bypass_applied", False)):
        return True
    if left_source is None or right_source is None or left_rows is None or right_rows is None:
        return True

    left_source_owned = getattr(left_source.values, "_owned", None)
    right_source_owned = getattr(right_source.values, "_owned", None)
    if left_source_owned is None or right_source_owned is None:
        return True

    from vibespatial.runtime.residency import Residency

    if (
        left_source_owned.residency is not Residency.DEVICE
        or right_source_owned.residency is not Residency.DEVICE
    ):
        return True

    source_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
    unique_right_rows = np.unique(source_right_rows)
    if unique_right_rows.size != 1:
        return True

    try:
        import cupy as cp

        from vibespatial.kernels.core.geometry_analysis import compute_geometry_bounds_device

        compute_geometry_bounds_device(left_source_owned)
        compute_geometry_bounds_device(right_source_owned)
        d_left_bounds = cp.asarray(left_source_owned.device_state.row_bounds).reshape(
            left_source_owned.row_count,
            4,
        )
        d_right_bounds = cp.asarray(right_source_owned.device_state.row_bounds).reshape(
            right_source_owned.row_count,
            4,
        )
        d_warning_left_rows = cp.asarray(
            np.asarray(left_rows, dtype=np.intp)[warning_rows],
            dtype=cp.int64,
        )
        d_warning_left_bounds = d_left_bounds[d_warning_left_rows]
        d_right_bounds_one = d_right_bounds[int(unique_right_rows[0])]
        maybe_covers = (
            (d_warning_left_bounds[:, 0] <= d_right_bounds_one[0])
            & (d_warning_left_bounds[:, 1] <= d_right_bounds_one[1])
            & (d_warning_left_bounds[:, 2] >= d_right_bounds_one[2])
            & (d_warning_left_bounds[:, 3] >= d_right_bounds_one[3])
        )
        return bool(cp.any(maybe_covers))
    except Exception:
        logger.debug(
            "many-vs-one keep_geom_type cover-probe precheck failed; "
            "falling back to full device cover classification",
            exc_info=True,
        )
        return True


def _repair_invalid_polygon_output_rows(geometries: GeoSeries) -> GeoSeries:
    """Repair invalid polygon rows from the rectangle exact path when present.

    The rectangle intersection kernel can emit polygon rows with zero-area
    duplicate-edge spikes on boundary-overlap cases. Those rows are still
    geometrically equivalent after ``make_valid`` but can change convex-hull
    fingerprints and public GeoPandas validity semantics. Repair only rows
    explicitly flagged by the kernel so the normal fast path stays untouched.
    """
    owned = getattr(geometries.values, "_owned", None)
    overlap_mask = getattr(owned, "_polygon_rect_boundary_overlap", None)
    if overlap_mask is None:
        if len(geometries) > 5000:
            return geometries
        suspect_rows = np.arange(len(geometries), dtype=np.intp)
    else:
        overlap_mask = (
            overlap_mask.get() if hasattr(overlap_mask, "get") else np.asarray(overlap_mask)
        )
        overlap_mask = np.asarray(overlap_mask, dtype=bool)
        if overlap_mask.size == len(geometries) and overlap_mask.any():
            suspect_rows = np.flatnonzero(overlap_mask).astype(np.intp, copy=False)
        else:
            if len(geometries) > 5000:
                return geometries
            suspect_rows = np.arange(len(geometries), dtype=np.intp)

    suspect_values: np.ndarray | None = None
    if owned is not None and overlap_mask is None:
        from vibespatial.constructive.validity import is_valid_owned

        suspect_owned = (
            owned
            if suspect_rows.size == len(geometries)
            else owned.take(suspect_rows.astype(np.int64, copy=False))
        )
        invalid_mask = ~np.asarray(is_valid_owned(suspect_owned), dtype=bool)
    else:
        all_values = np.asarray(geometries.values._data, dtype=object)
        suspect_values = all_values[suspect_rows]
        invalid_mask = ~np.asarray(shapely.is_valid(suspect_values), dtype=bool)
    if not invalid_mask.any():
        return geometries

    all_values = np.asarray(geometries.values._data, dtype=object)
    if suspect_values is None:
        suspect_values = all_values[suspect_rows]
    repaired_values = np.asarray(
        shapely.make_valid(suspect_values[invalid_mask]),
        dtype=object,
    )
    all_values[suspect_rows[invalid_mask]] = repaired_values
    repaired = GeoSeries(all_values, index=geometries.index, crs=geometries.crs)
    return _attach_polygon_rect_overlap_mask(repaired, overlap_mask)


def _attach_polygon_rect_overlap_mask(
    geometries: GeoSeries,
    overlap_mask: np.ndarray | None,
) -> GeoSeries:
    if overlap_mask is None:
        return geometries
    overlap_mask = np.asarray(overlap_mask, dtype=bool)
    owned = getattr(geometries.values, "_owned", None)
    if owned is not None:
        owned._polygon_rect_boundary_overlap = overlap_mask
    geometries.values._polygon_rect_boundary_overlap = overlap_mask
    return geometries


def _strip_non_polygon_collection_parts(geometries: np.ndarray) -> np.ndarray:
    """Replace GeometryCollections with polygon-only equivalents."""
    if len(geometries) == 0:
        return geometries

    type_ids = shapely.get_type_id(geometries)
    collection_rows = np.flatnonzero(type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION)
    if collection_rows.size == 0:
        return geometries

    cleaned = geometries.copy()
    for row_index in collection_rows:
        pending = np.asarray([geometries[int(row_index)]], dtype=object)
        polygon_parts: list[object] = []
        while len(pending) > 0:
            pending_type_ids = shapely.get_type_id(pending)
            collection_mask = pending_type_ids == _SHAPELY_TYPE_ID_GEOMETRYCOLLECTION
            non_collection = pending[~collection_mask]
            if non_collection.size > 0:
                non_empty_mask = ~shapely.is_empty(non_collection)
                non_collection = non_collection[non_empty_mask]
                if non_collection.size > 0:
                    non_collection_type_ids = shapely.get_type_id(non_collection)
                    polygon_mask = (
                        (non_collection_type_ids == _SHAPELY_TYPE_ID_POLYGON)
                        | (non_collection_type_ids == _SHAPELY_TYPE_ID_MULTIPOLYGON)
                    )
                    if np.any(polygon_mask):
                        polygon_parts.extend(non_collection[polygon_mask].tolist())
            if not np.any(collection_mask):
                break
            pending = shapely.get_parts(pending[collection_mask])

        if not polygon_parts:
            cleaned[int(row_index)] = None
        elif len(polygon_parts) == 1:
            cleaned[int(row_index)] = polygon_parts[0]
        else:
            cleaned[int(row_index)] = shapely.union_all(np.asarray(polygon_parts, dtype=object))
    return cleaned



def _filter_polygon_intersection_rows_for_keep_geom_type(
    left_pairs: GeoSeries | None,
    right_pairs: GeoSeries | None,
    area_pairs: GeoSeries,
    *,
    keep_geom_type_warning: bool,
    left_source: GeoSeries | None = None,
    right_source: GeoSeries | None = None,
    left_rows: np.ndarray | None = None,
    right_rows: np.ndarray | None = None,
) -> tuple[GeoSeries, int, np.ndarray]:
    """Keep polygonal area rows only and classify dropped lower-dimensional remnants."""
    area_overlap_mask = getattr(area_pairs.values, "_polygon_rect_boundary_overlap", None)
    area_exact_polygon_only_mask = _polygon_rect_exact_polygon_only_mask(area_pairs)
    if area_overlap_mask is not None:
        area_overlap_mask = (
            area_overlap_mask.get()
            if hasattr(area_overlap_mask, "get")
            else np.asarray(area_overlap_mask, dtype=bool)
        )
        area_overlap_mask = np.asarray(area_overlap_mask, dtype=bool)
        if area_overlap_mask.size != len(area_pairs):
            area_overlap_mask = None

    area_owned = getattr(area_pairs.values, "_owned", None)
    if area_overlap_mask is None and area_owned is not None:
        area_overlap_mask = getattr(area_owned, "_polygon_rect_boundary_overlap", None)
        if area_overlap_mask is not None:
            area_overlap_mask = (
                area_overlap_mask.get()
                if hasattr(area_overlap_mask, "get")
                else np.asarray(area_overlap_mask, dtype=bool)
            )
            area_overlap_mask = np.asarray(area_overlap_mask, dtype=bool)
            if area_overlap_mask.size != len(area_pairs):
                area_overlap_mask = None

    area_exact_values = getattr(area_pairs.values, "_exact_intersection_values", None)
    area_exact_mask = getattr(area_pairs.values, "_exact_intersection_value_mask", None)

    if area_owned is not None:
        from vibespatial.geometry.buffers import GeometryFamily
        from vibespatial.geometry.owned import FAMILY_TAGS
        from vibespatial.runtime.residency import Residency, TransferTrigger

        host_probe_fallback_logged = False

        def _requires_device_to_host_probe(*series) -> bool:
            for series_obj in series:
                if series_obj is None:
                    continue
                owned = getattr(series_obj.values, "_owned", None)
                if owned is not None and getattr(owned, "residency", None) is Residency.DEVICE:
                    return True
            return False

        def _record_keep_geom_type_host_probe_fallback(detail: str) -> None:
            nonlocal host_probe_fallback_logged
            if host_probe_fallback_logged or not _requires_device_to_host_probe(
                left_source,
                right_source,
                left_pairs,
                right_pairs,
            ):
                return
            record_fallback_event(
                surface="geopandas.overlay.intersection",
                reason="keep_geom_type semantic probe materialized host geometries",
                detail=detail,
                requested=ExecutionMode.GPU,
                selected=ExecutionMode.CPU,
                pipeline="_filter_polygon_intersection_rows_for_keep_geom_type",
                d2h_transfer=True,
            )
            host_probe_fallback_logged = True

        left_source_owned = getattr(left_source.values, "_owned", None) if left_source is not None else None
        right_source_owned = (
            getattr(right_source.values, "_owned", None) if right_source is not None else None
        )
        if (
            left_source_owned is not None
            and right_source_owned is not None
            and left_source_owned.residency is Residency.DEVICE
            and right_source_owned.residency is Residency.DEVICE
            and area_owned.residency is not Residency.DEVICE
        ):
            try:
                area_owned = area_owned.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="keep_geom_type classification promoted area rows to device",
                )
                area_pairs = GeoSeries(
                    GeometryArray.from_owned(area_owned, crs=area_pairs.crs),
                    index=area_pairs.index,
                    crs=area_pairs.crs,
                )
            except Exception:
                logger.debug(
                    "keep_geom_type area promotion to device failed; "
                    "falling back to existing classification path",
                    exc_info=True,
                )

        tags = area_owned.tags
        validity = area_owned.validity
        row_offsets = area_owned.family_row_offsets
        keep_mask = np.zeros(len(tags), dtype=bool)
        owned_metadata_consistent = True
        rect_overlap_mask = None
        if area_exact_values is None:
            area_exact_values = getattr(area_owned, "_exact_intersection_values", None)
        if area_exact_mask is None:
            area_exact_mask = getattr(area_owned, "_exact_intersection_value_mask", None)

        for family in (GeometryFamily.POLYGON, GeometryFamily.MULTIPOLYGON):
            family_tag = FAMILY_TAGS[family]
            family_mask = validity & (tags == family_tag)
            if not family_mask.any():
                continue
            family_rows = row_offsets[family_mask]
            empty_mask = np.asarray(area_owned.families[family].empty_mask, dtype=bool)
            if empty_mask.size == 0:
                family_count = int(getattr(area_owned.families[family], "row_count", 0))
                if (
                    area_overlap_mask is not None
                    and family_count > 0
                    and not np.any((family_rows < 0) | (family_rows >= family_count))
                ):
                    keep_mask[np.flatnonzero(family_mask)] = True
                    continue
                owned_metadata_consistent = False
                break
            if np.any((family_rows < 0) | (family_rows >= empty_mask.size)):
                owned_metadata_consistent = False
                break
            keep_mask[np.flatnonzero(family_mask)] = ~empty_mask[family_rows]

        if owned_metadata_consistent:
            dropped = 0
            if keep_geom_type_warning and len(tags) > 0:
                if area_exact_values is not None and area_exact_mask is not None:
                    area_exact_values = np.asarray(area_exact_values, dtype=object)
                    area_exact_mask = np.asarray(area_exact_mask, dtype=bool)
                    if area_exact_values.size != len(tags) or area_exact_mask.size != len(tags):
                        area_exact_values = None
                        area_exact_mask = None

                rect_overlap_mask = (
                    getattr(area_owned, "_polygon_rect_boundary_overlap", None)
                    if area_overlap_mask is None
                    else area_overlap_mask
                )
                if rect_overlap_mask is not None:
                    rect_overlap_mask = (
                        rect_overlap_mask.get()
                        if hasattr(rect_overlap_mask, "get")
                        else np.asarray(rect_overlap_mask, dtype=bool)
                    )
                    if rect_overlap_mask.size != len(tags):
                        rect_overlap_mask = None
                    else:
                        rect_overlap_mask = np.asarray(rect_overlap_mask, dtype=bool)

                if bool(np.all(keep_mask)) and rect_overlap_mask is not None and not rect_overlap_mask.any():
                    filtered = area_pairs.reset_index(drop=True)
                    filtered = _attach_polygon_rect_overlap_mask(filtered, rect_overlap_mask)
                    return filtered, 0, keep_mask

                if rect_overlap_mask is not None:
                    warning_mask = (~keep_mask) | rect_overlap_mask
                    if area_exact_values is not None and area_exact_mask is not None:
                        warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                        warning_mask[area_exact_mask] = _warning_candidate_mask_from_exact_intersection_values(
                            area_exact_values[area_exact_mask],
                            keep_mask[area_exact_mask],
                        )
                elif area_exact_values is not None and area_exact_mask is not None:
                    warning_mask = np.zeros(len(tags), dtype=bool)
                    warning_mask[area_exact_mask] = _warning_candidate_mask_from_exact_intersection_values(
                        area_exact_values[area_exact_mask],
                        keep_mask[area_exact_mask],
                    )
                    warning_mask[~area_exact_mask & ~keep_mask] = True
                    probe_rows = np.flatnonzero(~area_exact_mask & keep_mask).astype(np.intp, copy=False)
                    if probe_rows.size > 0:
                        _record_keep_geom_type_host_probe_fallback(
                            f"rows={len(tags)}, probe_rows={probe_rows.size}"
                        )
                        if (
                            left_source is not None
                            and right_source is not None
                            and left_rows is not None
                            and right_rows is not None
                        ):
                            left_values = _take_geoseries_object_values(
                                left_source,
                                np.asarray(left_rows, dtype=np.intp)[probe_rows],
                            )
                            right_values = _take_geoseries_object_values(
                                right_source,
                                np.asarray(right_rows, dtype=np.intp)[probe_rows],
                            )
                        else:
                            if left_pairs is None or right_pairs is None:
                                raise ValueError(
                                    "left_pairs/right_pairs or source rows are required when "
                                    "keep_geom_type_warning=True"
                                )
                            left_values = _take_geoseries_object_values(left_pairs, probe_rows)
                            right_values = _take_geoseries_object_values(right_pairs, probe_rows)
                        probe_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                            left_values,
                            right_values,
                            keep_mask[probe_rows],
                        )
                        warning_mask[probe_rows] = probe_mask
                else:
                    warning_mask = np.zeros(len(tags), dtype=bool)
                    if (
                        left_source is not None
                        and right_source is not None
                        and left_rows is not None
                        and right_rows is not None
                    ):
                        source_left_rows = np.asarray(left_rows, dtype=np.intp)
                        source_right_rows = np.asarray(right_rows, dtype=np.intp)
                        device_source_owned = False
                        if (
                            left_source_owned is not None
                            and right_source_owned is not None
                            and left_source_owned.residency is Residency.DEVICE
                            and right_source_owned.residency is Residency.DEVICE
                        ):
                            device_source_owned = True

                        def _take_owned_rows(owned, rows: np.ndarray):
                            import cupy as cp

                            if rows.size == 0:
                                return owned.take(rows)
                            return owned.device_take(cp.asarray(rows, dtype=cp.int64))

                        empty_rows = np.flatnonzero(~keep_mask).astype(np.intp, copy=False)
                        if empty_rows.size > 0:
                            if device_source_owned:
                                from vibespatial.predicates.binary import evaluate_binary_predicate
                                empty_left = _take_owned_rows(left_source_owned, source_left_rows[empty_rows])
                                empty_right = _take_owned_rows(
                                    right_source_owned,
                                    source_right_rows[empty_rows],
                                )
                                warning_mask[empty_rows] = np.asarray(
                                    evaluate_binary_predicate(
                                        "intersects",
                                        empty_left,
                                        empty_right,
                                        dispatch_mode=ExecutionMode.GPU,
                                    ).values,
                                    dtype=bool,
                                )
                            else:
                                _record_keep_geom_type_host_probe_fallback(
                                    f"rows={len(tags)}, dropped_rows={empty_rows.size}"
                                )
                                empty_left_values = _take_geoseries_object_values(
                                    left_source,
                                    source_left_rows[empty_rows],
                                )
                                empty_right_values = _take_geoseries_object_values(
                                    right_source,
                                    source_right_rows[empty_rows],
                                )
                                warning_mask[empty_rows] = np.asarray(
                                    shapely.intersects(empty_left_values, empty_right_values),
                                    dtype=bool,
                                )

                        kept_rows = np.flatnonzero(keep_mask).astype(np.intp, copy=False)
                        if kept_rows.size > 0:
                            if device_source_owned and area_owned.residency is Residency.DEVICE:
                                from vibespatial.constructive.boundary import boundary_owned
                                from vibespatial.predicates.binary import evaluate_binary_predicate

                                kept_left = _take_owned_rows(
                                    left_source_owned,
                                    source_left_rows[kept_rows],
                                )
                                kept_right = _take_owned_rows(
                                    right_source_owned,
                                    source_right_rows[kept_rows],
                                )
                                kept_left_boundary = boundary_owned(kept_left)
                                kept_right_boundary = boundary_owned(kept_right)
                                warning_mask[kept_rows] = np.asarray(
                                    evaluate_binary_predicate(
                                        "intersects",
                                        kept_left_boundary,
                                        kept_right_boundary,
                                        dispatch_mode=ExecutionMode.GPU,
                                    ).values,
                                    dtype=bool,
                                )
                            else:
                                _record_keep_geom_type_host_probe_fallback(
                                    f"rows={len(tags)}, kept_rows={kept_rows.size}"
                                )
                                kept_left_values = _take_geoseries_object_values(
                                    left_source,
                                    source_left_rows[kept_rows],
                                )
                                kept_right_values = _take_geoseries_object_values(
                                    right_source,
                                    source_right_rows[kept_rows],
                                )
                                warning_mask[kept_rows] = np.asarray(
                                    shapely.intersects(
                                        shapely.boundary(kept_left_values),
                                        shapely.boundary(kept_right_values),
                                    ),
                                    dtype=bool,
                                )
                    else:
                        if left_pairs is None or right_pairs is None:
                            raise ValueError(
                                "left_pairs/right_pairs or source rows are required when "
                                "keep_geom_type_warning=True"
                            )
                        _record_keep_geom_type_host_probe_fallback(
                            f"rows={len(tags)}, warning_rows={len(tags)}"
                        )
                        row_positions = np.arange(len(tags), dtype=np.intp)
                        left_values = _take_geoseries_object_values(left_pairs, row_positions)
                        right_values = _take_geoseries_object_values(right_pairs, row_positions)
                        warning_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                            left_values,
                            right_values,
                            keep_mask,
                        )

                if area_exact_polygon_only_mask is not None:
                    safe_rows = np.asarray(keep_mask, dtype=bool) & area_exact_polygon_only_mask
                    if safe_rows.any():
                        warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                        warning_mask[safe_rows] = False
                warning_rows = np.empty(0, dtype=np.intp)
                if np.any(warning_mask):
                    warning_rows = np.flatnonzero(np.asarray(warning_mask, dtype=bool)).astype(
                        np.intp,
                        copy=False,
                    )
                    warning_rows_have_exact_values = (
                        warning_rows.size > 0
                        and area_exact_values is not None
                        and area_exact_mask is not None
                        and bool(np.all(np.asarray(area_exact_mask[warning_rows], dtype=bool)))
                    )
                    if not warning_rows_have_exact_values:
                        warning_mask, warning_rows = _clear_device_exact_keep_geom_type_warnings(
                            warning_mask,
                            keep_mask,
                            left_source=left_source,
                            right_source=right_source,
                            left_rows=left_rows,
                            right_rows=right_rows,
                            area_owned=area_owned,
                            left_pairs=left_pairs,
                            right_pairs=right_pairs,
                        )
                if warning_rows.size > 0:
                    cached_warning_mask = None
                    rect_warning_count_resolved = False
                    if area_exact_values is not None and area_exact_mask is not None:
                        cached_warning_mask = np.asarray(area_exact_mask[warning_rows], dtype=bool)
                        if cached_warning_mask.any():
                            dropped += _count_dropped_polygon_intersection_parts(
                                np.empty(0, dtype=object),
                                np.empty(0, dtype=object),
                                int(np.count_nonzero(cached_warning_mask)),
                                exact_values=area_exact_values[warning_rows][cached_warning_mask],
                            )
                    if rect_overlap_mask is not None:
                        need_rect_probe_values = (
                            cached_warning_mask is None or (~cached_warning_mask).any()
                        )
                        if need_rect_probe_values:
                            uncached_warning_rows = (
                                warning_rows
                                if cached_warning_mask is None
                                else warning_rows[~cached_warning_mask]
                            )
                            device_uncached_dropped = _device_count_dropped_polygon_intersection_warning_rows(
                                area_owned,
                                keep_mask,
                                uncached_warning_rows,
                                left_source=left_source,
                                right_source=right_source,
                                left_rows=left_rows,
                                right_rows=right_rows,
                                left_pairs=left_pairs,
                                right_pairs=right_pairs,
                            )
                            if device_uncached_dropped is not None:
                                dropped += device_uncached_dropped
                                need_rect_probe_values = False
                                rect_warning_count_resolved = True
                            elif _requires_device_to_host_probe(
                                left_source,
                                right_source,
                                left_pairs,
                                right_pairs,
                            ):
                                # Keep device-selected rect-overlap batches on device even when
                                # the lower-dimensional warning counter cannot refine them yet.
                                # The warning count is advisory; falling back here only burns
                                # wall time after the exact geometry result is already native.
                                dropped += int(uncached_warning_rows.size)
                                need_rect_probe_values = False
                                rect_warning_count_resolved = True
                            elif (
                                left_source is not None
                                and right_source is not None
                                and left_rows is not None
                                and right_rows is not None
                            ):
                                uncached_count = int(uncached_warning_rows.size)
                                _record_keep_geom_type_host_probe_fallback(
                                    f"rows={len(tags)}, warning_rows={uncached_count}"
                                )
                                source_left_rows = np.asarray(left_rows, dtype=np.intp)[warning_rows]
                                source_right_rows = np.asarray(right_rows, dtype=np.intp)[warning_rows]
                                left_values = _take_geoseries_object_values(left_source, source_left_rows)
                                right_values = _take_geoseries_object_values(
                                    right_source,
                                    source_right_rows,
                                )
                            else:
                                if left_pairs is None or right_pairs is None:
                                    raise ValueError(
                                        "left_pairs/right_pairs or source rows are required when "
                                        "keep_geom_type_warning=True"
                                    )
                                uncached_count = int(uncached_warning_rows.size)
                                _record_keep_geom_type_host_probe_fallback(
                                    f"rows={len(tags)}, warning_rows={uncached_count}"
                                )
                                left_values = _take_geoseries_object_values(
                                    left_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                                right_values = _take_geoseries_object_values(
                                    right_pairs,
                                    warning_rows.astype(np.intp, copy=False),
                                )
                    if area_exact_values is not None and area_exact_mask is not None:
                        if (~cached_warning_mask).any():
                            uncached_warning_rows = warning_rows[~cached_warning_mask]
                            if rect_overlap_mask is not None and not need_rect_probe_values:
                                pass
                            elif rect_overlap_mask is not None:
                                left_uncached = left_values[~cached_warning_mask]
                                right_uncached = right_values[~cached_warning_mask]
                            elif (
                                left_source is not None
                                and right_source is not None
                                and left_rows is not None
                                and right_rows is not None
                            ):
                                left_uncached = _take_geoseries_object_values(
                                    left_source,
                                    np.asarray(left_rows, dtype=np.intp)[uncached_warning_rows],
                                )
                                right_uncached = _take_geoseries_object_values(
                                    right_source,
                                    np.asarray(right_rows, dtype=np.intp)[uncached_warning_rows],
                                )
                            else:
                                if left_pairs is None or right_pairs is None:
                                    raise ValueError(
                                        "left_pairs/right_pairs or source rows are required when "
                                        "keep_geom_type_warning=True"
                                    )
                                left_uncached = _take_geoseries_object_values(
                                    left_pairs,
                                    uncached_warning_rows,
                                )
                                right_uncached = _take_geoseries_object_values(
                                    right_pairs,
                                    uncached_warning_rows,
                                )
                            if rect_overlap_mask is None or need_rect_probe_values:
                                dropped += _count_dropped_polygon_intersection_parts(
                                    left_uncached,
                                    right_uncached,
                                    int(np.count_nonzero(~cached_warning_mask)),
                                )
                    else:
                        if rect_overlap_mask is not None and rect_warning_count_resolved:
                            pass
                        else:
                            device_dropped = _device_count_dropped_polygon_intersection_warning_rows(
                                area_owned,
                                keep_mask,
                                warning_rows,
                                left_source=left_source,
                                right_source=right_source,
                                left_rows=left_rows,
                                right_rows=right_rows,
                                left_pairs=left_pairs,
                                right_pairs=right_pairs,
                            )
                            if device_dropped is not None:
                                dropped = device_dropped
                            elif strict_native_mode_enabled() and _requires_device_to_host_probe(
                                left_source,
                                right_source,
                                left_pairs,
                                right_pairs,
                            ):
                                # Strict native mode cannot materialize host geometry just to
                                # refine the warning count. Keep the result fully native and
                                # conservatively count one dropped lower-dimensional remnant per
                                # warning row until the full exact-warning counter is native.
                                dropped = int(warning_rows.size)
                            else:
                                if rect_overlap_mask is None:
                                    if (
                                        left_source is not None
                                        and right_source is not None
                                        and left_rows is not None
                                        and right_rows is not None
                                    ):
                                        _record_keep_geom_type_host_probe_fallback(
                                            f"rows={len(tags)}, warning_rows={warning_rows.size}"
                                        )
                                        left_values = _take_geoseries_object_values(
                                            left_source,
                                            np.asarray(left_rows, dtype=np.intp)[warning_rows],
                                        )
                                        right_values = _take_geoseries_object_values(
                                            right_source,
                                            np.asarray(right_rows, dtype=np.intp)[warning_rows],
                                        )
                                    else:
                                        left_values = left_values[warning_mask]
                                        right_values = right_values[warning_mask]
                                dropped = _count_dropped_polygon_intersection_parts(
                                    left_values,
                                    right_values,
                                    int(warning_rows.size),
                                )

            if bool(np.all(keep_mask)):
                filtered = area_pairs.reset_index(drop=True)
                filtered = _attach_polygon_rect_overlap_mask(filtered, rect_overlap_mask)
                return filtered, dropped, keep_mask
            filtered = area_pairs.take(np.flatnonzero(keep_mask)).reset_index(drop=True)
            filtered = _attach_polygon_rect_overlap_mask(
                filtered,
                rect_overlap_mask[keep_mask] if rect_overlap_mask is not None else None,
            )
            return filtered, dropped, keep_mask

    area_values = _geoseries_object_values(area_pairs)
    present_mask = ~(shapely.is_missing(area_values) | shapely.is_empty(area_values))
    keep_mask = present_mask & (shapely.area(area_values) > 0.0)

    dropped = 0
    if keep_geom_type_warning and len(area_values) > 0:
        if area_overlap_mask is not None:
            warning_mask = (~keep_mask) | area_overlap_mask
        else:
            if (
                left_source is not None
                and right_source is not None
                and left_rows is not None
                and right_rows is not None
            ):
                left_values = _take_geoseries_object_values(
                    left_source,
                    np.asarray(left_rows, dtype=np.intp),
                )
                right_values = _take_geoseries_object_values(
                    right_source,
                    np.asarray(right_rows, dtype=np.intp),
                )
            else:
                if left_pairs is None or right_pairs is None:
                    raise ValueError(
                        "left_pairs/right_pairs or source rows are required when "
                        "keep_geom_type_warning=True"
                    )
                left_values = _geoseries_object_values(left_pairs)
                right_values = _geoseries_object_values(right_pairs)
            warning_mask = _warning_candidate_mask_for_polygon_keep_geom_type(
                left_values,
                right_values,
                keep_mask,
            )
        if area_exact_polygon_only_mask is not None:
            safe_rows = np.asarray(keep_mask, dtype=bool) & area_exact_polygon_only_mask
            if safe_rows.any():
                warning_mask = np.asarray(warning_mask, dtype=bool).copy()
                warning_mask[safe_rows] = False
        warning_rows = np.empty(0, dtype=np.intp)
        if np.any(warning_mask):
            warning_mask, warning_rows = _clear_device_exact_keep_geom_type_warnings(
                warning_mask,
                keep_mask,
                left_source=left_source,
                right_source=right_source,
                left_rows=left_rows,
                right_rows=right_rows,
                area_owned=area_owned,
            )

        if warning_rows.size > 0:
            if area_exact_values is not None and area_exact_mask is not None:
                area_exact_values = np.asarray(area_exact_values, dtype=object)
                area_exact_mask = np.asarray(area_exact_mask, dtype=bool)
                if area_exact_values.size != len(area_pairs) or area_exact_mask.size != len(area_pairs):
                    area_exact_values = None
                    area_exact_mask = None

            if area_overlap_mask is not None:
                if (
                    left_source is not None
                    and right_source is not None
                    and left_rows is not None
                    and right_rows is not None
                ):
                    _record_keep_geom_type_host_probe_fallback(
                        f"rows={len(area_values)}, warning_rows={warning_rows.size}"
                    )
                    left_values = _take_geoseries_object_values(
                        left_source,
                        np.asarray(left_rows, dtype=np.intp)[warning_rows],
                    )
                    right_values = _take_geoseries_object_values(
                        right_source,
                        np.asarray(right_rows, dtype=np.intp)[warning_rows],
                    )
                else:
                    if left_pairs is None or right_pairs is None:
                        raise ValueError(
                            "left_pairs/right_pairs or source rows are required when "
                            "keep_geom_type_warning=True"
                        )
                    _record_keep_geom_type_host_probe_fallback(
                        f"rows={len(area_values)}, warning_rows={warning_rows.size}"
                    )
                    left_values = _take_geoseries_object_values(left_pairs, warning_rows)
                    right_values = _take_geoseries_object_values(right_pairs, warning_rows)
            else:
                left_values = left_values[warning_mask]
                right_values = right_values[warning_mask]

            if area_exact_values is not None and area_exact_mask is not None:
                cached_warning_mask = np.asarray(area_exact_mask[warning_rows], dtype=bool)
                if cached_warning_mask.any():
                    dropped += _count_dropped_polygon_intersection_parts(
                        np.empty(0, dtype=object),
                        np.empty(0, dtype=object),
                        int(np.count_nonzero(cached_warning_mask)),
                        exact_values=area_exact_values[warning_rows][cached_warning_mask],
                    )
                if (~cached_warning_mask).any():
                    dropped += _count_dropped_polygon_intersection_parts(
                        left_values[~cached_warning_mask],
                        right_values[~cached_warning_mask],
                        int(np.count_nonzero(~cached_warning_mask)),
                    )
            else:
                dropped = _count_dropped_polygon_intersection_parts(
                    left_values,
                    right_values,
                    int(warning_rows.size),
                )

    if bool(np.all(keep_mask)):
        return area_pairs.reset_index(drop=True), dropped, keep_mask

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


def _filter_effective_polygon_difference_pairs(
    df1: GeoDataFrame,
    df2: GeoDataFrame,
    idx1,
    idx2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop polygon-polygon difference pairs whose intersection has zero area.

    Boundary-only touches do not change polygon difference results. Filtering
    those pairs keeps strict-native difference from routing harmless boundary
    contacts through the heavier overlay-difference reconstruction path.
    """
    if len(idx1) == 0:
        return np.asarray(idx1, dtype=np.int32), np.asarray(idx2, dtype=np.int32)

    left_rows = np.asarray(idx1, dtype=np.intp)
    right_rows = np.asarray(idx2, dtype=np.intp)
    left_values = _take_geoseries_object_values(df1.geometry, left_rows)
    right_values = _take_geoseries_object_values(df2.geometry, right_rows)
    area_mask = shapely.area(shapely.intersection(left_values, right_values)) > 0.0
    if bool(np.all(area_mask)):
        return left_rows.astype(np.int32, copy=False), right_rows.astype(np.int32, copy=False)
    return (
        left_rows[area_mask].astype(np.int32, copy=False),
        right_rows[area_mask].astype(np.int32, copy=False),
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


def _host_exact_polygon_intersection_owned_batch(
    left_owned,
    right_owned,
    *,
    requested: ExecutionMode | str,
    reason: str,
    detail: str,
):
    """Run an exact host intersection batch and cache raw exact results."""
    from vibespatial.geometry.owned import from_shapely_geometries

    requested_mode = (
        requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested)
    )
    d2h_transfer = (
        getattr(left_owned, "device_state", None) is not None
        or getattr(right_owned, "device_state", None) is not None
    )
    record_fallback_event(
        surface="geopandas.overlay.intersection",
        reason=reason,
        detail=detail,
        requested=requested_mode,
        selected=ExecutionMode.CPU,
        pipeline="_host_exact_polygon_intersection_owned_batch",
        d2h_transfer=d2h_transfer,
    )

    left_values = np.asarray(left_owned.to_shapely(), dtype=object)
    if right_owned.row_count == 1 and left_values.size > 1:
        right_geom = right_owned.to_shapely()[0]
        if right_geom is None or shapely.is_empty(right_geom):
            empty = from_shapely_geometries([shapely.Point()])
            return empty.take(np.asarray([], dtype=np.int64))
        right_values = np.full(len(left_values), right_geom, dtype=object)
    else:
        right_values = np.asarray(right_owned.to_shapely(), dtype=object)

    raw = np.asarray(shapely.intersection(left_values, right_values), dtype=object)
    result = from_shapely_geometries(list(raw))
    result._exact_intersection_values = raw
    result._exact_intersection_value_mask = np.ones(len(raw), dtype=bool)
    return result


def _host_exact_polygon_intersection_series_batch(
    left_source: GeoSeries,
    right_source: GeoSeries,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    *,
    crs,
    requested: ExecutionMode | str,
    reason: str,
):
    """Build an exact host GeoSeries batch and retain raw exact results."""
    from vibespatial.geometry.owned import from_shapely_geometries

    requested_mode = (
        requested if isinstance(requested, ExecutionMode) else ExecutionMode(requested)
    )
    left_rows = np.asarray(left_rows, dtype=np.intp)
    right_rows = np.asarray(right_rows, dtype=np.intp)
    record_fallback_event(
        surface="geopandas.overlay.intersection",
        reason=reason,
        detail=f"rows={left_rows.size}",
        requested=requested_mode,
        selected=ExecutionMode.CPU,
        pipeline="_host_exact_polygon_intersection_series_batch",
        d2h_transfer=True,
    )

    left_values = _take_geoseries_object_values(left_source, left_rows)
    right_values = _take_geoseries_object_values(right_source, right_rows)
    raw = np.asarray(shapely.intersection(left_values, right_values), dtype=object)

    try:
        owned = from_shapely_geometries(list(raw))
    except NotImplementedError:
        result = GeoSeries(raw, crs=crs)
        result.values._exact_intersection_values = raw
        result.values._exact_intersection_value_mask = np.ones(len(raw), dtype=bool)
        return result

    owned._exact_intersection_values = raw
    owned._exact_intersection_value_mask = np.ones(len(raw), dtype=bool)
    return GeoSeries(
        GeometryArray.from_owned(owned, crs=crs),
        crs=crs,
    )
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
        _host_rectangle_polygon_mask,
        binary_constructive_owned,
    )
    n_pairs = int(left_sub.row_count)
    pairwise_selection = plan_dispatch_selection(
        kernel_name="overlay_pairwise",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n_pairs,
    )
    if (
        not strict_native_mode_enabled()
        and pairwise_selection.selected is ExecutionMode.CPU
        and n_pairs >= _MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS
        and n_pairs <= _MANY_VS_ONE_HOST_EXACT_MAX_ROWS
    ):
        right_one = right_owned.take(np.array([unique_right_idx], dtype=np.intp))
        exact_result = _host_exact_polygon_intersection_owned_batch(
            left_sub,
            right_one,
            requested=pairwise_selection.requested,
            reason="many-vs-one CPU-selected exact host batch",
            detail=f"rows={n_pairs}",
        )
        exact_result._polygon_rect_boundary_overlap = np.zeros(left_sub.row_count, dtype=bool)
        return exact_result

    _prepare_result = _prepare_many_vs_one_intersection_chunks(
        left_sub,
        right_owned,
        unique_right_idx,
        global_positions=np.arange(left_sub.row_count, dtype=np.intp),
    )
    if len(_prepare_result) == 5:
        index_oga_pairs, complex_left, complex_global_positions, right_one, _pairwise_mode = (
            _prepare_result
        )
    else:
        (
            index_oga_pairs,
            complex_left,
            complex_global_positions,
            right_one,
            _pairwise_mode,
            _use_direct_full_batch_overlay,
        ) = _prepare_result
        if _use_direct_full_batch_overlay:
            logger.debug(
                "many-vs-one intersection bypass yield too small; using one full exact GPU batch"
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
        return _host_exact_polygon_intersection_owned_batch(
            left_rem_oga,
            right_one_oga,
            requested=_pairwise_mode,
            reason=(
                "many-vs-one remainder: vectorized Shapely intersection "
                f"for {int(left_rem_oga.row_count)} boundary-crossing polygons"
            ),
            detail=f"rows={int(left_rem_oga.row_count)}",
        )

    def _gpu_remainder_intersection(left_rem_oga, right_one_oga):
        """Intersect remainder polygons via GPU overlay pipeline."""
        from vibespatial.constructive.binary_constructive import (
            _broadcast_right_cached_segments,
            _dispatch_polygon_intersection_overlay_broadcast_right_gpu,
            _dispatch_polygon_intersection_overlay_rowwise_gpu,
        )
        from vibespatial.geometry.owned import materialize_broadcast, tile_single_row
        from vibespatial.overlay.gpu import _overlay_owned

        right_rep = materialize_broadcast(tile_single_row(right_one_oga, left_rem_oga.row_count))
        cached_right_segments = _broadcast_right_cached_segments(
            right_one_oga,
            left_rem_oga.row_count,
        )
        try:
            if left_rem_oga.row_count <= _OVERLAY_ROWWISE_REMAINDER_MAX:
                broadcast_result = _dispatch_polygon_intersection_overlay_broadcast_right_gpu(
                    left_rem_oga,
                    right_one_oga,
                    dispatch_mode=_pairwise_mode,
                    _cached_right_segments=cached_right_segments,
                )
                if (
                    broadcast_result is not None
                    and broadcast_result.row_count == left_rem_oga.row_count
                ):
                    return broadcast_result
            if left_rem_oga.row_count <= _OVERLAY_MEDIUM_REMAINDER_ROWWISE_MAX:
                rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                    left_rem_oga,
                    right_rep,
                    dispatch_mode=_pairwise_mode,
                    _cached_right_segments=cached_right_segments,
                )
                if rowwise_result is not None and rowwise_result.row_count == left_rem_oga.row_count:
                    return rowwise_result
            if _pairwise_mode is ExecutionMode.GPU:
                batched_result = _overlay_owned(
                    left_rem_oga,
                    right_rep,
                    operation="intersection",
                    dispatch_mode=ExecutionMode.GPU,
                    _cached_right_segments=cached_right_segments,
                    _row_isolated=True,
                )
                if batched_result.row_count == left_rem_oga.row_count:
                    return batched_result
            rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
                left_rem_oga,
                right_rep,
                dispatch_mode=_pairwise_mode,
                _cached_right_segments=cached_right_segments,
            )
            if rowwise_result is not None and rowwise_result.row_count == left_rem_oga.row_count:
                return rowwise_result
            return binary_constructive_owned(
                "intersection", left_rem_oga, right_rep,
                dispatch_mode=_pairwise_mode,
                _cached_right_segments=cached_right_segments,
            )
        finally:
            if cached_right_segments is not None:
                try:
                    from vibespatial.constructive.binary_constructive import (
                        _free_device_segment_table,
                    )

                    _free_device_segment_table(cached_right_segments)
                except Exception:
                    logger.debug(
                        "many-vs-one cached right-segment cleanup failed",
                        exc_info=True,
                    )

    def _remainder_intersection(left_rem_oga, right_one_oga):
        """Choose the remainder path based on residency and crossover shape."""
        from vibespatial.runtime.residency import Residency, TransferTrigger

        if strict_native_mode_enabled():
            if left_rem_oga.residency is not Residency.DEVICE:
                try:
                    left_rem_oga = left_rem_oga.move_to(
                        Residency.DEVICE,
                        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason="strict many-vs-one remainder promoted left polygons to device",
                    )
                except Exception:
                    logger.debug(
                        "strict many-vs-one left promotion failed before device remainder path",
                        exc_info=True,
                    )
            if right_one_oga.residency is not Residency.DEVICE:
                try:
                    right_one_oga = right_one_oga.move_to(
                        Residency.DEVICE,
                        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                        reason="strict many-vs-one remainder promoted clip polygon to device",
                    )
                except Exception:
                    logger.debug(
                        "strict many-vs-one right promotion failed before device remainder path",
                        exc_info=True,
                    )
            try:
                return _gpu_remainder_intersection(left_rem_oga, right_one_oga)
            except Exception:
                logger.debug(
                    "many-vs-one strict device remainder path failed; "
                    "falling back to host crossover policy",
                    exc_info=True,
                )
        elif left_rem_oga.residency is Residency.DEVICE:
            try:
                return _gpu_remainder_intersection(left_rem_oga, right_one_oga)
            except Exception:
                logger.debug(
                    "many-vs-one device remainder path failed; "
                    "falling back to host crossover policy",
                    exc_info=True,
                )
        if left_rem_oga.row_count >= OVERLAY_GPU_REMAINDER_THRESHOLD:
            try:
                left_device = left_rem_oga.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="many-vs-one remainder promoted left polygons to device",
                )
                right_device = right_one_oga.move_to(
                    Residency.DEVICE,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="many-vs-one remainder promoted clip polygon to device",
                )
                return _gpu_remainder_intersection(left_device, right_device)
            except Exception:
                logger.debug(
                    "many-vs-one remainder device promotion failed; "
                    "falling back to host crossover policy",
                    exc_info=True,
                )
        if left_rem_oga.row_count < OVERLAY_GPU_REMAINDER_THRESHOLD:
            return _shapely_remainder_intersection(left_rem_oga, right_one_oga)
        try:
            return _gpu_remainder_intersection(left_rem_oga, right_one_oga)
        except Exception:
            logger.debug(
                "many-vs-one large remainder GPU path failed after host crossover check; "
                "falling back to vectorized Shapely intersection",
                exc_info=True,
            )
            return _shapely_remainder_intersection(left_rem_oga, right_one_oga)

    complex_result = None
    if complex_left is not None and complex_global_positions is not None:
        complex_result = _remainder_intersection(complex_left, right_one)
        if complex_result is not None:
            try:
                left_rect_mask = _host_rectangle_polygon_mask(complex_left)
                right_rect_mask = _host_rectangle_polygon_mask(right_one)
                if (
                    left_rect_mask is not None
                    and right_rect_mask is not None
                    and right_rect_mask.size == 1
                    and bool(right_rect_mask[0])
                ):
                    left_rect_mask = np.asarray(left_rect_mask, dtype=bool)
                    if left_rect_mask.size == complex_result.row_count and left_rect_mask.any():
                        complex_result._polygon_rect_exact_polygon_only = left_rect_mask
            except Exception:
                logger.debug(
                    "many-vs-one exact-overlay rectangle classification failed",
                    exc_info=True,
                )

    # Release GPU pool memory after overlay on remainder polygons: SH clip
    # and per-pair GPU overlay produce large intermediates that are dead now.
    from vibespatial.cuda._runtime import maybe_trim_pool_memory

    maybe_trim_pool_memory()

    if complex_result is not None and complex_global_positions is not None:
        index_oga_pairs.append((complex_global_positions, complex_result))

    assembled = _assemble_indexed_owned_chunks(index_oga_pairs, left_sub.row_count)

    warning_suspect_mask = np.zeros(left_sub.row_count, dtype=bool)
    if complex_result is not None and complex_global_positions is not None:
        complex_overlap_mask = getattr(complex_result, "_polygon_rect_boundary_overlap", None)
        if complex_overlap_mask is not None:
            complex_overlap_mask = (
                complex_overlap_mask.get()
                if hasattr(complex_overlap_mask, "get")
                else np.asarray(complex_overlap_mask, dtype=bool)
            )
            complex_overlap_mask = np.asarray(complex_overlap_mask, dtype=bool)
            if complex_overlap_mask.size == complex_result.row_count:
                warning_suspect_mask[np.asarray(complex_global_positions, dtype=np.intp)] = (
                    complex_overlap_mask
                )
            else:
                warning_suspect_mask[np.asarray(complex_global_positions, dtype=np.intp)] = True
        else:
            warning_suspect_mask[np.asarray(complex_global_positions, dtype=np.intp)] = True
    assembled._polygon_rect_boundary_overlap = warning_suspect_mask
    assembled._many_vs_one_left_containment_bypass_applied = True
    return assembled


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

    exact_values = None
    exact_value_mask = None
    exact_polygon_only_mask = None
    for indices, oga in index_oga_pairs:
        oga_exact_values = getattr(oga, "_exact_intersection_values", None)
        oga_exact_mask = getattr(oga, "_exact_intersection_value_mask", None)
        if oga_exact_values is None or oga_exact_mask is None:
            pass
        else:
            oga_exact_values = np.asarray(oga_exact_values, dtype=object)
            oga_exact_mask = np.asarray(oga_exact_mask, dtype=bool)
            if (
                oga_exact_values.size == len(indices)
                and oga_exact_mask.size == len(indices)
            ):
                if exact_values is None:
                    exact_values = np.empty(row_count, dtype=object)
                    exact_values[:] = None
                    exact_value_mask = np.zeros(row_count, dtype=bool)
                row_indices = np.asarray(indices, dtype=np.intp)
                exact_values[row_indices] = oga_exact_values
                exact_value_mask[row_indices] = oga_exact_mask

        oga_exact_polygon_only = getattr(oga, "_polygon_rect_exact_polygon_only", None)
        if oga_exact_polygon_only is None:
            continue
        oga_exact_polygon_only = np.asarray(oga_exact_polygon_only, dtype=bool)
        if oga_exact_polygon_only.size != len(indices):
            continue
        if exact_polygon_only_mask is None:
            exact_polygon_only_mask = np.zeros(row_count, dtype=bool)
        exact_polygon_only_mask[np.asarray(indices, dtype=np.intp)] = oga_exact_polygon_only

    all_indices = np.concatenate([idx for idx, _ in index_oga_pairs])
    concat_result = OwnedGeometryArray.concat([oga for _, oga in index_oga_pairs])
    inverse_perm = np.empty(row_count, dtype=np.intp)
    inverse_perm[all_indices] = np.arange(len(all_indices), dtype=np.intp)
    result = concat_result.take(inverse_perm)
    if exact_values is not None and exact_value_mask is not None and exact_value_mask.any():
        result._exact_intersection_values = exact_values
        result._exact_intersection_value_mask = exact_value_mask
    if exact_polygon_only_mask is not None and exact_polygon_only_mask.any():
        result._polygon_rect_exact_polygon_only = exact_polygon_only_mask
    return result


def _host_convex_single_ring_polygon_mask(owned) -> np.ndarray | None:
    """Classify rows that are single-ring convex polygons using host metadata."""
    from vibespatial.constructive.binary_constructive import _host_single_ring_polygon_mask
    from vibespatial.geometry.buffers import GeometryFamily
    from vibespatial.overlay.bypass import _is_convex_ring_xy

    single_ring_mask = _host_single_ring_polygon_mask(owned, max_input_vertices=64)
    if single_ring_mask is None or GeometryFamily.POLYGON not in owned.families:
        return None

    polygon_buf = owned.families[GeometryFamily.POLYGON]
    geom_offsets = np.asarray(polygon_buf.geometry_offsets, dtype=np.int64)
    ring_offsets = np.asarray(polygon_buf.ring_offsets, dtype=np.int64)
    x = np.asarray(polygon_buf.x, dtype=np.float64)
    y = np.asarray(polygon_buf.y, dtype=np.float64)

    convex_mask = np.zeros(owned.row_count, dtype=bool)
    for row in np.flatnonzero(single_ring_mask).astype(np.intp, copy=False):
        ring_row = int(geom_offsets[int(row)])
        start = int(ring_offsets[ring_row])
        end = int(ring_offsets[ring_row + 1])
        convex_mask[int(row)] = _is_convex_ring_xy(x, y, start, end)
    return convex_mask


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
        _contained_result._polygon_rect_exact_polygon_only = np.ones(
            _contained_result.row_count,
            dtype=bool,
        )
        record_dispatch_event(
            surface="geopandas.overlay.intersection",
            operation="many_vs_one_containment_bypass",
            implementation="gpu_containment_bypass_all",
            reason=(
                f"many-vs-one: all {n_pairs} polygons fully inside "
                "right polygon (containment bypass)"
            ),
            selected=ExecutionMode.GPU,
        )
        return [(global_positions, _contained_result)], None, None, right_one, _pairwise_mode, False

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
        selected=ExecutionMode.GPU if n_contained > 0 else ExecutionMode.CPU,
    )

    from vibespatial.cuda._runtime import maybe_trim_pool_memory

    maybe_trim_pool_memory()

    if n_remainder == 0:
        return [(global_positions, _contained_result)], None, None, right_one, _pairwise_mode, False

    direct_full_batch_candidate = (
        n_pairs >= _MANY_VS_ONE_DIRECT_FULL_BATCH_MIN_ROWS
        and n_remainder <= _OVERLAY_ROWWISE_REMAINDER_MAX
        and (n_contained / n_pairs) <= _MANY_VS_ONE_DIRECT_FULL_BATCH_MAX_CONTAINED_FRACTION
    )
    if direct_full_batch_candidate and _cp_local is not None:
        try:
            from vibespatial.overlay.bypass import _is_clip_polygon_sh_eligible

            clip_eligible, _clip_vert_count = _is_clip_polygon_sh_eligible(right_one)
        except Exception:
            clip_eligible = False
        if not clip_eligible:
            record_dispatch_event(
                surface="geopandas.overlay.intersection",
                operation="many_vs_one_containment_bypass",
                implementation="gpu_full_batch_exact_overlay",
                reason=(
                    "many-vs-one: containment bypass would only save "
                    f"{n_contained}/{n_pairs} rows; using one full exact GPU batch"
                ),
                selected=ExecutionMode.GPU,
            )
            return [], left_sub, global_positions, right_one, _pairwise_mode, True

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
        _contained_result._polygon_rect_exact_polygon_only = np.ones(
            _contained_result.row_count,
            dtype=bool,
        )
        contained_global = (
            global_positions[_cp_local.asnumpy(d_contained_indices).astype(np.intp, copy=False)]
            if d_contained_indices is not None
            else global_positions[contained_indices]
        )
        index_oga_pairs.append((contained_global, _contained_result))

    complex_left = None
    complex_global_positions = None
    if sh_clip_result is not None and sh_eligible_mask is not None:
        sh_exact_polygon_only = None
        convex_mask = _host_convex_single_ring_polygon_mask(left_remainder)
        if convex_mask is not None and convex_mask.size == left_remainder.row_count:
            sh_exact_polygon_only = np.asarray(
                convex_mask[np.asarray(sh_eligible_mask, dtype=bool)],
                dtype=bool,
            )
            if sh_exact_polygon_only.size == sh_clip_result.row_count and sh_exact_polygon_only.any():
                sh_clip_result._polygon_rect_exact_polygon_only = sh_exact_polygon_only
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
        False,
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
    d_idx2=None,
):
    """Run few-right intersection as one gathered exact pairwise batch.

    The earlier grouped-by-right shape decomposed a logically single overlay
    into many per-right preparations and gathers. For warmed strict-native
    workloads that was slower than simply gathering the intersecting pairs
    once and running one exact rowwise GPU intersection batch.
    """
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
        _expand_right_segments_for_pair_rows,
        _free_device_segment_table,
        binary_constructive_owned,
    )
    if _has_device_indices:
        if cp is None or d_idx1 is None or d_idx2 is None:
            return None
        left_pairs = left_owned.device_take(d_idx1)
        right_pairs = right_owned.device_take(d_idx2)
    else:
        left_pairs = left_owned.take(np.asarray(idx1))
        right_pairs = right_owned.take(np.asarray(idx2))

    rect_capable = False
    try:
        from vibespatial.kernels.constructive.polygon_rect_intersection import (
            polygon_rect_intersection_can_handle,
        )

        rect_capable = (
            polygon_rect_intersection_can_handle(left_pairs, right_pairs)
            or polygon_rect_intersection_can_handle(right_pairs, left_pairs)
        )
    except Exception:
        rect_capable = False

    if rect_capable:
        return binary_constructive_owned(
            "intersection",
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
            _prefer_exact_polygon_intersection=True,
        )

    cached_right_segments = None
    if left_pairs.row_count >= _OVERLAY_FEW_RIGHT_CACHED_SEG_MIN_ROWS:
        cached_right_segments = _expand_right_segments_for_pair_rows(
            right_owned,
            np.asarray(idx2),
        )
    try:
        rowwise_result = _dispatch_polygon_intersection_overlay_rowwise_gpu(
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
            _cached_right_segments=cached_right_segments,
        )
        if rowwise_result is not None and rowwise_result.row_count == left_pairs.row_count:
            return rowwise_result
        return binary_constructive_owned(
            "intersection",
            left_pairs,
            right_pairs,
            dispatch_mode=dispatch_mode,
            _prefer_exact_polygon_intersection=True,
            _cached_right_segments=cached_right_segments,
        )
    finally:
        if cached_right_segments is not None:
            _free_device_segment_table(cached_right_segments)


def _overlay_intersection_native(
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
    """Build the native intersection result before host-side export.

    Returns
    -------
    tuple[PairwiseConstructiveResult, bool]
        Native constructive result plus whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1, df2, left_owned, right_owned,
    )
    _polygon_inputs = _series_all_polygons(df1.geometry) and _series_all_polygons(df2.geometry)
    prefer_exact_polygon_gpu = _prefer_exact_polygon_gpu
    # Public polygon intersection can legitimately produce lower-dimensional
    # rows or GeometryCollections that must be filtered at the GeoPandas
    # boundary. Unless the strict exact-polygon GPU path is explicitly
    # requested, keep geometry construction on the host boundary and use the
    # owned path only for pairing/index acceleration.
    _use_host_exact_polygon_boundary = (
        _polygon_inputs
        and not prefer_exact_polygon_gpu
        and (
            not strict_native_mode_enabled()
            or (left_owned is None and right_owned is None)
        )
    )
    # ADR-0042 low-level contract: spatial indexing may still emit index arrays.
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

    if (
        _polygon_inputs
        and _warn_on_dropped_lower_dim_polygon_results
        and not strict_native_mode_enabled()
        and idx1.size <= OVERLAY_PAIR_BATCH_THRESHOLD
    ):
        _use_host_exact_polygon_boundary = True

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
        if _is_many_vs_one_pre and has_gpu_runtime():
            _both_polygon = _polygon_inputs
            if _both_polygon:
                from vibespatial.geometry.owned import from_shapely_geometries
                from vibespatial.runtime.residency import (
                    Residency,
                    TransferTrigger,
                    combined_residency,
                )

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
                if left_owned is not None and right_owned is not None:
                    if combined_residency(left_owned, right_owned) is not Residency.DEVICE:
                        left_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason="many-vs-one polygon overlay promoted left input to device",
                        )
                        right_owned.move_to(
                            Residency.DEVICE,
                            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                            reason="many-vs-one polygon overlay promoted right input to device",
                        )
                    prefer_exact_polygon_gpu = True
                    _use_host_exact_polygon_boundary = False

        # Owned-path dispatch: OwnedGeometryArray.take() operates at buffer
        # level (no Shapely materialization), then binary_constructive_owned
        # routes to GPU when available.  GeoSeries.take() breaks the DGA chain
        # by materializing to Shapely, so we bypass it when owned is present.
        # Note: GeometryArray.copy() preserves _owned, and __setitem__
        # invalidates it on mutation, so owned survives _make_valid when
        # all geometries are already valid.
        intersections = None
        if (
            intersections is None
            and _polygon_inputs
            and not strict_native_mode_enabled()
            and not prefer_exact_polygon_gpu
            and idx1.size > 0
            and idx1.size <= _OVERLAY_HOST_EXACT_PAIR_BATCH_MAX_ROWS
            and not has_gpu_runtime()
        ):
            pairwise_selection = plan_dispatch_selection(
                kernel_name="overlay_pairwise",
                kernel_class=KernelClass.CONSTRUCTIVE,
                row_count=idx1.size,
                requested_mode=ExecutionMode.AUTO,
            )
            if pairwise_selection.selected is ExecutionMode.CPU:
                intersections = _host_exact_polygon_intersection_series_batch(
                    df1.geometry,
                    df2.geometry,
                    np.asarray(idx1, dtype=np.intp),
                    np.asarray(idx2, dtype=np.intp),
                    crs=df1.crs,
                    requested=pairwise_selection.requested,
                    reason="small polygon pair batch: CPU-selected exact host batch",
                )
        if (
            left_owned is not None
            and right_owned is not None
            and not _use_host_exact_polygon_boundary
            and intersections is None
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
                if strict_native_mode_enabled() or prefer_exact_polygon_gpu
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
                and _polygon_inputs
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
                        d_idx2=d_idx2,
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
                try:
                    result_owned = binary_constructive_owned(
                            "intersection", left_sub, right_sub,
                            dispatch_mode=_pairwise_mode,
                            _prefer_exact_polygon_intersection=prefer_exact_polygon_gpu,
                        )
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
                except Exception:
                    logger.debug(
                        "owned element-wise polygon intersection failed; "
                        "falling back to boundary path",
                        exc_info=True,
                    )
                    intersections = None

            if intersections is None and _is_few_right:
                if _has_device_indices:
                    left_sub = left_owned.device_take(d_idx1)
                    right_sub = right_owned.device_take(d_idx2)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                    right_sub = right_owned.take(np.asarray(idx2))
                try:
                    result_owned = binary_constructive_owned(
                        "intersection", left_sub, right_sub,
                        dispatch_mode=_pairwise_mode,
                        _prefer_exact_polygon_intersection=prefer_exact_polygon_gpu,
                    )
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
                except Exception:
                    logger.debug(
                        "few-right owned polygon intersection failed; "
                        "falling back to boundary path",
                        exc_info=True,
                    )
                    intersections = None

            if intersections is None and _is_many_vs_one:
                # Many-vs-one fast path failed -- fall back to element-wise.
                # Gather right side now (deferred from above to avoid OOM
                # on the many-vs-one fast path).
                if right_sub is None:
                    if _has_device_indices:
                        right_sub = right_owned.device_take(d_idx2)
                    else:
                        right_sub = right_owned.take(np.asarray(idx2))
                try:
                    result_owned = binary_constructive_owned(
                        "intersection", left_sub, right_sub,
                        dispatch_mode=_pairwise_mode,
                        _prefer_exact_polygon_intersection=prefer_exact_polygon_gpu,
                    )
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
                except Exception:
                    logger.debug(
                        "many-vs-one owned polygon intersection failed; "
                        "falling back to boundary path",
                        exc_info=True,
                    )
                    intersections = None

        if intersections is None:
            # ADR-0042 transitional boundary: host exact path still uses GeoSeries ops.
            if _use_host_exact_polygon_boundary:
                if has_gpu_runtime() and idx1.size <= OVERLAY_PAIR_BATCH_THRESHOLD:
                    from vibespatial.constructive.binary_constructive import (
                        binary_constructive_owned,
                    )

                    try:
                        pair_rows_left = np.asarray(idx1, dtype=np.intp)
                        pair_rows_right = np.asarray(idx2, dtype=np.intp)
                        if left_owned is not None and right_owned is not None:
                            pair_left_owned = left_owned.take(pair_rows_left)
                            pair_right_owned = right_owned.take(pair_rows_right)
                        else:
                            pair_left = df1.geometry.take(pair_rows_left)
                            pair_left.reset_index(drop=True, inplace=True)
                            pair_right = df2.geometry.take(pair_rows_right)
                            pair_right.reset_index(drop=True, inplace=True)
                            pair_left_owned = pair_left.values.to_owned()
                            pair_right_owned = pair_right.values.to_owned()
                        result_owned = binary_constructive_owned(
                            "intersection",
                            pair_left_owned,
                            pair_right_owned,
                            dispatch_mode=ExecutionMode.GPU,
                            _prefer_exact_polygon_intersection=True,
                        )
                        intersections = GeoSeries(
                            GeometryArray.from_owned(result_owned, crs=df1.crs),
                        )
                        used_owned = True
                    except Exception:
                        logger.debug(
                            "pair-owned exact polygon GPU boundary fallback failed; "
                            "falling back to Shapely exact intersection",
                            exc_info=True,
                        )
                        intersections = None

                if intersections is None:
                    left_values = _take_geoseries_object_values(
                        df1.geometry,
                        np.asarray(idx1, dtype=np.intp),
                    )
                    right_values = _take_geoseries_object_values(
                        df2.geometry,
                        np.asarray(idx2, dtype=np.intp),
                    )
                    intersections = GeoSeries(
                        shapely.intersection(left_values, right_values),
                        crs=df1.crs,
                    )
            else:
                left = df1.geometry.take(idx1)
                left.reset_index(drop=True, inplace=True)
                right = df2.geometry.take(idx2)
                right.reset_index(drop=True, inplace=True)
                intersections = left.intersection(right)

        # Post-intersection make_valid must run for both owned/GPU and
        # fallback boundary paths. Some exact polygon fast paths can emit
        # topologically invalid-but-equivalent polygons that need repair
        # before public overlay consumers like dissolve.
        if not (
            _polygon_inputs
            and _can_defer_make_valid_to_rect_repair(intersections)
        ):
            intersections = _make_valid_geoseries(
                intersections,
                dispatch_mode=(
                    ExecutionMode.GPU
                    if (used_owned or prefer_exact_polygon_gpu)
                    else ExecutionMode.AUTO
                ),
            )

        geom_intersect = intersections
        keep_geom_type_applied = False
        if _polygon_inputs:
            pair_left = None
            pair_right = None
            source_idx1 = np.asarray(idx1, dtype=np.intp)
            source_idx2 = np.asarray(idx2, dtype=np.intp)
            left_source_geoms = df1.geometry
            right_source_geoms = df2.geometry
            if left_owned is not None:
                left_source_geoms = GeoSeries(
                    GeometryArray.from_owned(left_owned, crs=df1.crs),
                    crs=df1.crs,
                )
            if right_owned is not None:
                right_source_geoms = GeoSeries(
                    GeometryArray.from_owned(right_owned, crs=df2.crs),
                    crs=df2.crs,
                )
            if (
                _preserve_lower_dim_polygon_results
            ):
                pair_left = df1.geometry.take(idx1)
                pair_left.reset_index(drop=True, inplace=True)
                pair_right = df2.geometry.take(idx2)
                pair_right.reset_index(drop=True, inplace=True)

            if (
                _warn_on_dropped_lower_dim_polygon_results
                or prefer_exact_polygon_gpu
            ):
                pair_left = pair_left if _preserve_lower_dim_polygon_results else None
                pair_right = pair_right if _preserve_lower_dim_polygon_results else None

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
                    left_source=left_source_geoms,
                    right_source=right_source_geoms,
                    left_rows=source_idx1,
                    right_rows=source_idx2,
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
            elif prefer_exact_polygon_gpu:
                geom_intersect, _, keep_mask = _filter_polygon_intersection_rows_for_keep_geom_type(
                    pair_left,
                    pair_right,
                    geom_intersect,
                    keep_geom_type_warning=False,
                    left_source=left_source_geoms,
                    right_source=right_source_geoms,
                    left_rows=source_idx1,
                    right_rows=source_idx2,
                )
                idx1 = np.asarray(idx1, dtype=np.int32)[keep_mask]
                idx2 = np.asarray(idx2, dtype=np.int32)[keep_mask]
                keep_geom_type_applied = True

            geom_intersect = _repair_invalid_polygon_output_rows(geom_intersect)
            if not _preserve_lower_dim_polygon_results:
                geom_values = _geoseries_object_values(geom_intersect)
                if np.any(shapely.get_type_id(geom_values) == 7):
                    geom_intersect = GeoSeries(
                        _strip_non_polygon_collection_parts(geom_values),
                        index=geom_intersect.index,
                        crs=geom_intersect.crs,
                    )

        geom_intersect_owned = getattr(geom_intersect.values, "_owned", None)
        if geom_intersect_owned is not None:
            nonempty_mask = _owned_valid_nonempty_mask(geom_intersect_owned)
        else:
            nonempty_mask = ~(geom_intersect.isna() | geom_intersect.is_empty)
        if not nonempty_mask.all():
            keep = np.asarray(nonempty_mask, dtype=bool)
            idx1 = np.asarray(idx1, dtype=np.int32)[keep]
            idx2 = np.asarray(idx2, dtype=np.int32)[keep]
            geom_intersect = geom_intersect[keep].reset_index(drop=True)

        return (
            PairwiseConstructiveResult(
                geometry=_geometry_native_result_from_geoseries(geom_intersect),
                relation=RelationIndexResult(idx1, idx2),
                keep_geom_type_applied=keep_geom_type_applied,
            ),
            used_owned,
        )

    empty_geometry = GeoSeries([], index=pd.RangeIndex(0), crs=df1.crs, name="geometry")
    return (
        PairwiseConstructiveResult(
            geometry=_geometry_native_result_from_geoseries(empty_geometry),
            relation=RelationIndexResult(
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
            ),
            keep_geom_type_applied=False,
        ),
        used_owned,
    )


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
    export_result, used_owned = _overlay_intersection_export_result(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
        _index_result=_index_result,
    )
    return export_result.to_geodataframe(), used_owned


def _overlay_difference_native(df1, df2, left_owned=None, right_owned=None, *, _index_result=None):
    """Build the native difference result before host-side export.

    Returns
    -------
    tuple[LeftConstructiveResult, bool]
        Native constructive result plus whether the owned dispatch path was used.
    """
    left_owned, right_owned = _coerce_owned_pair_for_strict_overlay(
        df1, df2, left_owned, right_owned,
    )
    # ADR-0042 low-level contract: spatial indexing may still emit index arrays.
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

    polygon_inputs = _series_all_polygons(df1.geometry) and _series_all_polygons(df2.geometry)
    if strict_native_mode_enabled() and polygon_inputs and len(idx1) > 0:
        idx1, idx2 = _filter_effective_polygon_difference_pairs(df1, df2, idx1, idx2)

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
        and _should_use_owned_constructive_overlay(left_owned, right_owned)
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
    differences = _make_valid_geoseries(
        differences,
        dispatch_mode=ExecutionMode.GPU if used_owned else ExecutionMode.AUTO,
    )
    differences_owned = getattr(differences.values, "_owned", None)
    if differences_owned is not None:
        keep_rows = _owned_valid_nonempty_mask(differences_owned)
        keep_indices = np.flatnonzero(keep_rows).astype(np.int64, copy=False)
        if keep_indices.size > 0:
            geometry_result = GeometryNativeResult(
                crs=df1.crs,
                owned=differences_owned.take(keep_indices),
            )
        else:
            geometry_result = GeometryNativeResult.from_geoseries(
                GeoSeries([], index=df1.index[:0], crs=df1.crs),
            )
        keep_positions = keep_indices
    else:
        empty_mask = differences.is_empty
        if empty_mask.any():
            differences = differences.copy()
            differences.loc[empty_mask] = None
        keep_rows = ~empty_mask
        geom_diff = differences[keep_rows].copy()
        geometry_result = GeometryNativeResult.from_geoseries(geom_diff)
        keep_positions = np.flatnonzero(np.asarray(keep_rows, dtype=bool)).astype(np.int64, copy=False)
    return LeftConstructiveResult(
        geometry=geometry_result,
        row_positions=keep_positions,
    ), used_owned


def _overlay_difference(df1, df2, left_owned=None, right_owned=None, *, _index_result=None):
    export_result, used_owned = _overlay_difference_export_result(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_index_result,
    )
    return export_result.to_geodataframe(), used_owned


def _overlay_intersection_export_result(
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
    """Build the deferred public intersection export result before GeoDataFrame assembly."""
    native_result, used_owned = _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
        _index_result=_index_result,
    )
    return (
        PairwiseConstructiveFragment(
            result=native_result,
            left_df=df1.reset_index(drop=True),
            right_df=df2.reset_index(drop=True),
            attribute_assembler=_assemble_intersection_attributes,
            geometry_name="geometry",
            frame_type=GeoDataFrame,
            prefer_native_attribute_projection=True,
        ),
        used_owned,
    )


def _overlay_difference_export_result(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _index_result=None,
):
    """Build the deferred public difference export result before GeoDataFrame assembly."""
    native_result, used_owned = _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_index_result,
    )
    return (
        LeftConstructiveFragment(
            result=native_result,
            df=df1,
            geometry_name=df1._geometry_column_name,
            frame_type=GeoDataFrame,
        ),
        used_owned,
    )


def _overlay_identity_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Build the native identity result before the explicit GeoPandas export."""
    forward_index_result = _intersecting_index_pairs(
        df1, df2, left_owned=left_owned, right_owned=right_owned,
    )
    intersection_native, used_inter = _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    difference_native, used_diff = _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
    )

    intersection_left = df1.reset_index(drop=True)
    intersection_right = df2.reset_index(drop=True)
    intersection_columns = _intersection_attribute_columns(intersection_left, intersection_right)
    difference_rename = {
        column: (column if column in intersection_columns else f"{column}_1")
        for column in df1.drop(df1._geometry_column_name, axis=1).columns
    }

    native_result = ConcatConstructiveResult(
        parts=(
            PairwiseConstructiveFragment(
                result=intersection_native,
                left_df=intersection_left,
                right_df=intersection_right,
                attribute_assembler=_assemble_intersection_attributes,
                geometry_name="geometry",
                frame_type=GeoDataFrame,
                prefer_native_attribute_projection=True,
            ),
            LeftConstructiveFragment(
                result=difference_native,
                df=df1,
                rename_columns=difference_rename,
                geometry_name="geometry",
                frame_type=GeoDataFrame,
            ),
        ),
        geometry_name="geometry",
        frame_type=GeoDataFrame,
        crs=df1.crs,
    )
    return native_result, used_inter or used_diff


def _overlay_identity(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Overlay Identity operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    native_result, used_owned = _overlay_identity_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    return native_result.to_geodataframe(), used_owned


def _overlay_symmetric_diff_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _forward_index_result=None,
    _reverse_index_result=None,
):
    """Build the native symmetric-difference result before explicit export."""
    if _forward_index_result is None:
        _forward_index_result = _intersecting_index_pairs(
            df1, df2, left_owned=left_owned, right_owned=right_owned,
        )
    if _reverse_index_result is None:
        _reverse_index_result = _reverse_intersecting_index_pairs(
            _forward_index_result,
        )

    diff1_native, used1 = _overlay_difference_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=_forward_index_result,
    )
    diff2_native, used2 = _overlay_difference_native(
        df2,
        df1,
        right_owned,
        left_owned,
        _index_result=_reverse_index_result,
    )

    native_result = SymmetricDifferenceConstructiveResult(
        left_result=diff1_native,
        right_result=diff2_native,
        left_df=df1,
        right_df=df2,
        geometry_name="geometry",
        frame_type=GeoDataFrame,
        crs=df1.crs,
    )
    return native_result, used1 or used2


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
    native_result, used_owned = _overlay_symmetric_diff_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _forward_index_result=_forward_index_result,
        _reverse_index_result=_reverse_index_result,
    )
    return native_result.to_geodataframe(), used_owned


def _overlay_union_native(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Build the native union result before the explicit GeoPandas export."""
    forward_index_result = _intersecting_index_pairs(
        df1, df2, left_owned=left_owned, right_owned=right_owned,
    )
    intersection_native, used_inter = _overlay_intersection_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _index_result=forward_index_result,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    symmetric_native, used_sym = _overlay_symmetric_diff_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _forward_index_result=forward_index_result,
    )
    native_result = ConcatConstructiveResult(
        parts=(
            PairwiseConstructiveFragment(
                result=intersection_native,
                left_df=df1.reset_index(drop=True),
                right_df=df2.reset_index(drop=True),
                attribute_assembler=_assemble_intersection_attributes,
                geometry_name="geometry",
                frame_type=GeoDataFrame,
                prefer_native_attribute_projection=True,
            ),
            symmetric_native,
        ),
        geometry_name="geometry",
        frame_type=GeoDataFrame,
        crs=df1.crs,
    )
    return native_result, used_inter or used_sym


def _overlay_union(
    df1,
    df2,
    left_owned=None,
    right_owned=None,
    *,
    _prefer_exact_polygon_gpu: bool = False,
    _preserve_lower_dim_polygon_results: bool = False,
    _warn_on_dropped_lower_dim_polygon_results: bool = False,
):
    """Overlay Union operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    native_result, used_owned = _overlay_union_native(
        df1,
        df2,
        left_owned,
        right_owned,
        _prefer_exact_polygon_gpu=_prefer_exact_polygon_gpu,
        _preserve_lower_dim_polygon_results=_preserve_lower_dim_polygon_results,
        _warn_on_dropped_lower_dim_polygon_results=_warn_on_dropped_lower_dim_polygon_results,
    )
    return native_result.to_geodataframe(), used_owned


def _reset_overlay_result_index(result: GeoDataFrame) -> GeoDataFrame:
    """Drop a non-default index without shedding owned geometry backing."""
    if isinstance(result.index, pd.RangeIndex):
        if result.index.start == 0 and result.index.step == 1 and result.index.stop == len(result):
            _maybe_seed_polygon_validity_cache(result)
            return result

    geom_name = result._geometry_column_name
    geom_values = result.geometry.values
    attrs = result.attrs.copy()

    if getattr(geom_values, "_owned", None) is not None:
        attrs_df = result.drop(columns=[geom_name]).reset_index(drop=True)
        geom_series = GeoSeries(
            geom_values,
            index=attrs_df.index,
            name=geom_name,
            crs=result.crs,
        )
        reset = GeoDataFrame(attrs_df).set_geometry(
            geom_series,
            crs=result.crs,
        )
    else:
        reset = result.reset_index(drop=True)
        if not isinstance(reset, GeoDataFrame):
            reset = GeoDataFrame(reset)
        if reset.crs is None and result.crs is not None:
            reset = reset.set_crs(result.crs)

    reset.attrs.update(attrs)
    _maybe_seed_polygon_validity_cache(reset)
    return reset


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

    left_polygon_mask = _series_polygon_mask(df1.geometry)
    right_polygon_mask = _series_polygon_mask(df2.geometry)
    left_all_polygons = bool(left_polygon_mask.all())
    right_all_polygons = bool(right_polygon_mask.all())

    for i, df in enumerate([df1, df2]):
        poly_check = bool(left_polygon_mask.any()) if i == 0 else bool(right_polygon_mask.any())
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
    boundary_left_owned = getattr(df1.geometry.values, "_owned", None)
    boundary_right_owned = getattr(df2.geometry.values, "_owned", None)
    boundary_prefers_exact_polygon_gpu = _should_prefer_exact_polygon_gpu(
        df1,
        df2,
        boundary_left_owned,
        boundary_right_owned,
        keep_geom_type=keep_geom_type,
    )
    boundary_make_valid_mode = (
        ExecutionMode.GPU if boundary_prefers_exact_polygon_gpu else ExecutionMode.AUTO
    )

    def _make_valid(df, *, dispatch_mode: ExecutionMode | str, all_polygons: bool):
        df = df.copy()
        if all_polygons:
            # GPU make_valid path: when owned backing is available, route
            # through make_valid_owned to keep data device-resident and
            # avoid Shapely materialisation on the overlay critical path.
            ga = df.geometry.values
            owned = getattr(ga, '_owned', None)
            if make_valid and owned is not None:
                from vibespatial.constructive.make_valid_pipeline import (
                    make_valid_owned,
                )
                from vibespatial.constructive.validity import is_valid_owned

                valid_mask = np.asarray(is_valid_owned(owned), dtype=bool)
                if not bool(np.all(owned.validity)):
                    valid_mask = valid_mask.copy()
                    valid_mask[~owned.validity] = True
                if valid_mask.all():
                    return df

                mv_result = make_valid_owned(
                    owned=owned,
                    dispatch_mode=dispatch_mode,
                )
                if mv_result.repaired_rows.size > 0:
                    # Repair happened — prefer device-resident .owned
                    # to avoid D->H transfer.
                    new_ga = None
                    if mv_result.owned is not None:
                        try:
                            new_ga = GeometryArray.from_owned(
                                mv_result.owned, crs=df.crs,
                            )
                        except Exception as exc:
                            record_fallback_event(
                                surface="geopandas.array.make_valid",
                                reason=(
                                    "owned make_valid output could not be rewrapped; "
                                    "host materialization required"
                                ),
                                detail=(
                                    f"rewrap_error={type(exc).__name__}: {exc}"
                                ),
                                requested=dispatch_mode,
                                selected=ExecutionMode.CPU,
                                pipeline="overlay._make_valid",
                                d2h_transfer=True,
                            )
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

    cached_intersection_index_result = None
    if how == "intersection":
        cached_intersection_index_result = get_cached_intersection_pairs(df1, df2)
    if cached_intersection_index_result is not None:
        cached_left_rows = np.unique(np.asarray(cached_intersection_index_result[0], dtype=np.int32))
        cached_right_rows = np.unique(np.asarray(cached_intersection_index_result[1], dtype=np.int32))
    else:
        cached_left_rows = np.asarray([], dtype=np.int32)
        cached_right_rows = np.asarray([], dtype=np.int32)
    reuse_cached_intersection_index = (
        cached_intersection_index_result is not None
        and make_valid
        and left_all_polygons
        and right_all_polygons
        and _candidate_rows_all_valid(df1.geometry, cached_left_rows)
        and _candidate_rows_all_valid(df2.geometry, cached_right_rows)
    )
    if reuse_cached_intersection_index:
        df1 = df1.copy()
        df2 = df2.copy()
    else:
        cached_intersection_index_result = None
        df1 = _make_valid(
            df1,
            dispatch_mode=boundary_make_valid_mode,
            all_polygons=left_all_polygons,
        )
        df2 = _make_valid(
            df2,
            dispatch_mode=boundary_make_valid_mode,
            all_polygons=right_all_polygons,
        )

    if _can_rewrite_single_mask_intersection_to_clip(
        df1,
        df2,
        how=how,
        left_all_polygons=left_all_polygons,
        right_all_polygons=right_all_polygons,
    ):
        from vibespatial.api.tools.clip import clip as _clip_surface

        result = _clip_surface(
            df1,
            df2.geometry.iloc[0],
            keep_geom_type=keep_geom_type,
            sort=False,
        )
        geometry_values = result.geometry.values
        _used_owned = (
            getattr(geometry_values, "_owned", None) is not None
            or geometry_values.__class__.__name__ == "DeviceGeometryArray"
        )
        record_dispatch_event(
            surface="geopandas.overlay",
            operation="overlay_intersection",
            implementation="clip_rewrite",
            reason=(
                "single-row geometry-only right mask rewrote overlay intersection to clip"
            ),
            detail=(
                f"left_rows={len(df1)}, right_rows={len(df2)}, how={how}, "
                f"used_owned={_used_owned}"
            ),
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.GPU if _used_owned else ExecutionMode.CPU,
        )
        return _reset_overlay_result_index(result)

    # Extract owned arrays AFTER _make_valid.  GeometryArray.copy() now
    # preserves _owned backing, and __setitem__ invalidates it only for
    # mutated rows.  If _make_valid mutated all rows or dropped rows via
    # _collection_extract, _owned will already be None here.
    left_owned, right_owned = _extract_owned_pair(df1, df2)
    prefer_exact_polygon_gpu = _should_prefer_exact_polygon_gpu(
        df1,
        df2,
        left_owned,
        right_owned,
        keep_geom_type=keep_geom_type,
    )

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
                    prefer_exact_polygon_gpu
                ),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
                _index_result=cached_intersection_index_result,
            )
        elif how == "symmetric_difference":
            result, _used_owned = _overlay_symmetric_diff(
                df1, df2, left_owned, right_owned,
            )
        elif how == "union":
            result, _used_owned = _overlay_union(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(
                    prefer_exact_polygon_gpu
                ),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
            )
        elif how == "identity":
            result, _used_owned = _overlay_identity(
                df1,
                df2,
                left_owned,
                right_owned,
                _prefer_exact_polygon_gpu=(
                    prefer_exact_polygon_gpu
                ),
                _preserve_lower_dim_polygon_results=(keep_geom_type is False),
                _warn_on_dropped_lower_dim_polygon_results=keep_geom_type_warning,
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

    return _reset_overlay_result_index(result)


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
