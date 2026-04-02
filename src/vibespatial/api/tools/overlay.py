from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import shapely

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
from vibespatial.runtime.precision import KernelClass
from vibespatial.spatial.query_types import DeviceSpatialJoinResult

logger = logging.getLogger(__name__)


def _extract_owned_pair(df1, df2):
    """Return (left_owned, right_owned) if both DataFrames have owned backing, else (None, None)."""
    ga1 = df1.geometry.values
    ga2 = df2.geometry.values
    left_owned = getattr(ga1, '_owned', None)
    right_owned = getattr(ga2, '_owned', None)
    if left_owned is not None and right_owned is not None:
        return left_owned, right_owned
    return None, None


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
    from vibespatial.constructive.binary_constructive import (
        binary_constructive_owned,
    )
    from vibespatial.constructive.segmented_union_host import (
        concat_owned_arrays,
    )
    from vibespatial.kernels.constructive.segmented_union import (
        segmented_union_all,
    )

    xp = np
    if hasattr(idx1, "__cuda_array_interface__"):
        try:
            import cupy
            xp = cupy
        except ImportError:
            pass

    # --- Compute group structure (unique left indices + group offsets) ---
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
        if _has_device_indices:
            right_gathered = right_owned.device_take(d_idx2)
        else:
            right_gathered = right_owned.take(idx2)

        right_unions_owned = segmented_union_all(right_gathered, group_offsets_full)
        left_sub = left_owned.take(idx1_unique)
        diff_owned = binary_constructive_owned(
            "difference", left_sub, right_unions_owned,
            dispatch_mode=_pairwise_mode,
        )
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
        try:
            from vibespatial.cuda._runtime import get_cuda_runtime
            get_cuda_runtime().free_pool_memory()
        except Exception:
            pass

        # Gather right geometries for this batch only
        batch_idx2 = h_idx2[pair_start:pair_end]
        right_gathered = right_owned.take(batch_idx2)

        # Build local group offsets for this batch (0-based)
        batch_group_starts = h_group_offsets[batch_start:batch_end + 1] - pair_start
        batch_group_offsets = np.asarray(batch_group_starts, dtype=np.int64)

        # Segmented union: one union per group in this batch
        right_unions = segmented_union_all(right_gathered, batch_group_offsets)
        del right_gathered  # free gathered array before difference

        # Take the corresponding left geometries
        batch_left_indices = h_idx1_unique[batch_start:batch_end]
        left_sub = left_owned.take(batch_left_indices)

        # Pairwise difference
        batch_diff = binary_constructive_owned(
            "difference", left_sub, right_unions,
            dispatch_mode=_pairwise_mode,
        )
        del right_unions, left_sub  # free intermediates

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
        diff_owned = concat_owned_arrays(batch_results)
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
    if not strict_native_mode_enabled():
        request_device = left_owned is not None and right_owned is not None
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

    idx1, idx2 = df2.sindex._tree.query(
        np.asarray(df1.geometry.array, dtype=object),
        predicate="intersects",
    )
    idx1 = np.asarray(idx1, dtype=np.int32)
    idx2 = np.asarray(idx2, dtype=np.int32)
    # Sort by idx1 to match the non-strict path (sort=True) contract.
    # _overlay_difference relies on sorted idx1 for np.split grouping.
    order = np.argsort(idx1, kind="stable")
    return idx1[order], idx2[order]


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

    n_pairs = left_sub.row_count
    _pairwise_mode = plan_dispatch_selection(
        kernel_name="overlay_pairwise",
        kernel_class=KernelClass.CONSTRUCTIVE,
        row_count=n_pairs,
    )
    _pairwise_mode = _pairwise_mode.selected

    # Build the single-row right OGA for containment bypass / SH clip.
    right_one = right_owned.take(np.array([unique_right_idx], dtype=np.intp))

    # ---- Tier 1: Containment bypass (GPU vertex-in-polygon) ----
    # Identifies left polygons fully inside the right polygon.
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
        # All polygons are fully contained -- return left_sub as-is.
        record_dispatch_event(
            surface="geopandas.overlay.intersection",
            operation="many_vs_one_containment_bypass",
            implementation="gpu_containment_bypass_all",
            reason=(
                f"many-vs-one: all {n_pairs} polygons fully inside "
                "right polygon (containment bypass)"
            ),
        )
        return _contained_result

    # Determine which rows are contained vs remainder.
    if _remainder_mask is not None:
        # _remainder_mask is a CuPy bool array: True = needs overlay.
        h_remainder_mask = _cp_local.asnumpy(_remainder_mask)
        contained_indices = np.flatnonzero(~h_remainder_mask)
        remainder_indices = np.flatnonzero(h_remainder_mask)
    else:
        # No containment bypass (no GPU or bypass failed) -- all need overlay.
        contained_indices = np.array([], dtype=np.intp)
        remainder_indices = np.arange(n_pairs, dtype=np.intp)

    n_contained = len(contained_indices)
    n_remainder = len(remainder_indices)

    record_dispatch_event(
        surface="geopandas.overlay.intersection",
        operation="many_vs_one_containment_bypass",
        implementation=(
            "gpu_containment_bypass" if n_contained > 0 else "skipped"
        ),
        reason=(
            f"many-vs-one: {n_contained}/{n_pairs} polygons fully inside "
            f"right polygon, {n_remainder} need overlay"
        ),
    )

    # Release GPU pool memory after containment bypass: bounds check + PIP
    # can produce large intermediates that are no longer needed.
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup

    if n_remainder == 0:
        # Should not happen (handled above), but be safe.
        return _contained_result

    # Subset left_sub to remainder rows only.
    if _cp_local is not None:
        d_remainder = _cp_local.asarray(remainder_indices).astype(_cp_local.int64)
        left_remainder = left_sub.take(d_remainder)
    else:
        left_remainder = left_sub.take(remainder_indices)

    # ---- Tier 2: SH batch clip for simple boundary-crossing polygons ----
    sh_clip_result = None
    sh_eligible_mask = None
    sh_remainder_indices_local = None  # indices into left_remainder

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
                        sh_remainder_indices_local = np.flatnonzero(
                            ~sh_eligible_mask_arr
                        )
        except Exception:
            sh_clip_result = None
            sh_eligible_mask = None

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
        right_rep = right_one_oga.take(
            np.zeros(left_rem_oga.row_count, dtype=np.intp),
        )
        return binary_constructive_owned(
            "intersection", left_rem_oga, right_rep,
            dispatch_mode=_pairwise_mode,
        )

    def _remainder_intersection(left_rem_oga, right_one_oga):
        """Choose one remainder path up front instead of falling back by exception."""
        if (
            left_rem_oga.row_count < OVERLAY_GPU_REMAINDER_THRESHOLD
            or not has_gpu_runtime()
        ):
            return _shapely_remainder_intersection(left_rem_oga, right_one_oga)
        return _gpu_remainder_intersection(left_rem_oga, right_one_oga)

    if sh_clip_result is not None and sh_eligible_mask is not None:
        # Some polygons were SH-clipped; overlay only the complex remainder.
        if len(sh_remainder_indices_local) > 0:
            if _cp_local is not None:
                d_complex = _cp_local.asarray(
                    sh_remainder_indices_local,
                ).astype(_cp_local.int64)
                left_complex = left_remainder.take(d_complex)
            else:
                left_complex = left_remainder.take(sh_remainder_indices_local)
            complex_result = _remainder_intersection(left_complex, right_one)
        else:
            complex_result = None
    else:
        # No SH clip -- overlay all remainder rows.
        complex_result = _remainder_intersection(left_remainder, right_one)

    # Release GPU pool memory after overlay on remainder polygons: SH clip
    # and per-pair GPU overlay produce large intermediates that are dead now.
    try:
        from vibespatial.cuda._runtime import get_cuda_runtime
        get_cuda_runtime().free_pool_memory()
    except Exception:
        pass  # best-effort cleanup

    # ---- Reassemble results in original pair order ----
    # Build a result array with one geometry per pair in the original order.
    # contained_indices -> left_sub geometry (identity)
    # remainder with SH clip -> sh_clip_result (in SH eligible order)
    # remainder without SH clip -> complex_result (in complex order)
    from vibespatial.geometry.owned import OwnedGeometryArray

    # Track (global_indices, oga) for each chunk to enable correct ordering.
    index_oga_pairs: list[tuple[np.ndarray, OwnedGeometryArray]] = []

    if n_contained > 0 and _contained_result is not None:
        index_oga_pairs.append((contained_indices, _contained_result))

    if sh_clip_result is not None and sh_eligible_mask is not None:
        # SH clip results correspond to the SH-eligible rows of left_remainder.
        sh_eligible_local = np.flatnonzero(sh_eligible_mask)
        sh_global = remainder_indices[sh_eligible_local]
        index_oga_pairs.append((sh_global, sh_clip_result))

        if complex_result is not None and sh_remainder_indices_local is not None:
            complex_global = remainder_indices[sh_remainder_indices_local]
            index_oga_pairs.append((complex_global, complex_result))
    elif complex_result is not None:
        index_oga_pairs.append((remainder_indices, complex_result))

    if not index_oga_pairs:
        # Edge case: no results at all.
        from vibespatial.geometry.owned import from_shapely_geometries

        empty = from_shapely_geometries([shapely.Point()])
        return empty.take(np.asarray([], dtype=np.int64))

    if len(index_oga_pairs) == 1:
        # Only one chunk -- check if it covers all pairs in order.
        indices, oga = index_oga_pairs[0]
        if len(indices) == n_pairs and np.array_equal(
            indices, np.arange(n_pairs),
        ):
            return oga

    # Multiple chunks or non-trivial ordering: concat and reorder.
    # Build a permutation array: result[perm[i]] = concat_result[i].
    all_indices = np.concatenate([idx for idx, _ in index_oga_pairs])
    all_ogas = [oga for _, oga in index_oga_pairs]
    concat_result = OwnedGeometryArray.concat(all_ogas)

    # all_indices[i] is the target position in the output for concat row i.
    # We need: output[all_indices[i]] = concat_result[i]
    # Equivalently: output = concat_result.take(inverse_perm)
    # where inverse_perm[all_indices[i]] = i.
    inverse_perm = np.empty(n_pairs, dtype=np.intp)
    inverse_perm[all_indices] = np.arange(len(all_indices), dtype=np.intp)
    return concat_result.take(inverse_perm)


def _overlay_intersection(df1, df2, left_owned=None, right_owned=None):
    """Overlay Intersection operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
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
        left_owned is None
        and right_owned is None
        and has_gpu_runtime()
    )
    if _force_strtree:
        df2.sindex._ensure_strtree()
        _raw_idx = df2.sindex._tree.query(
            np.asarray(df1.geometry.array, dtype=object),
            predicate="intersects",
        )
        idx1 = np.asarray(_raw_idx[0], dtype=np.int32)
        idx2 = np.asarray(_raw_idx[1], dtype=np.int32)
        order = np.argsort(idx1, kind="stable")
        idx1, idx2 = idx1[order], idx2[order]
        d_idx1, d_idx2 = None, None
        _has_device_indices = False
    else:
        index_result = _intersecting_index_pairs(
            df1, df2, left_owned=left_owned, right_owned=right_owned,
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
        if left_owned is not None and right_owned is not None:
            from vibespatial.constructive.binary_constructive import (
                binary_constructive_owned,
            )

            # Free pool memory before device_take: spatial index and
            # prior pipeline stages leave freed-but-cached blocks in
            # the pool.  Forcing GC here ensures dead CuPy arrays
            # return their blocks before the large gather allocation.
            try:
                from vibespatial.cuda._runtime import get_cuda_runtime
                get_cuda_runtime().free_pool_memory()
            except Exception:
                pass
            # AUTO dispatch: the GPU polygon_intersection kernel uses
            # Sutherland-Hodgman, which is only correct for convex clip
            # polygons.  AUTO lets the crossover policy route small
            # batches (< 50K rows) to the CPU/Shapely path that handles
            # concave polygons.  The validity-bitmap winding-direction
            # bug has been fixed, so GPU dispatch IS correct when the
            # crossover threshold selects it for convex-only workloads.
            _pairwise_mode = ExecutionMode.AUTO

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

            if _is_many_vs_one:
                # Many-vs-one: only gather left side.
                if _has_device_indices:
                    left_sub = left_owned.device_take(d_idx1)
                else:
                    left_sub = left_owned.take(np.asarray(idx1))
                right_sub = None  # deferred until fallback

                try:
                    result_owned = _many_vs_one_intersection_owned(
                        left_sub,
                        right_owned,
                        int(_unique_right[0]),
                        _has_device_indices=_has_device_indices,
                        d_idx2=d_idx2,
                    )
                    intersections = GeoSeries(
                        GeometryArray.from_owned(result_owned, crs=df1.crs),
                    )
                    used_owned = True
                except Exception:
                    # Fall through to element-wise fallback.
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

            if intersections is None and not _is_many_vs_one:
                # Standard element-wise path for N-vs-M patterns.
                result_owned = binary_constructive_owned(
                    "intersection", left_sub, right_sub,
                    dispatch_mode=_pairwise_mode,
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
                )
                intersections = GeoSeries(
                    GeometryArray.from_owned(result_owned, crs=df1.crs),
                )
                used_owned = True

        if intersections is None:
            # ADR-0036 boundary: geometry operations on geometry arrays.
            left = df1.geometry.take(idx1)
            left.reset_index(drop=True, inplace=True)
            right = df2.geometry.take(idx2)
            right.reset_index(drop=True, inplace=True)
            intersections = left.intersection(right)

        # Post-intersection make_valid: use GPU path when owned backing is
        # available to avoid Shapely materialisation on the critical path.
        intersections = _make_valid_geoseries(intersections)

        geom_intersect = intersections

        # ADR-0036 boundary: attribute assembly from index arrays.
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
        dfinter = _assemble_intersection_attributes(
            idx1, idx2,
            df1.drop(df1._geometry_column_name, axis=1),
            df2.drop(df2._geometry_column_name, axis=1),
        )

        return GeoDataFrame(dfinter, geometry=geom_intersect, crs=df1.crs), used_owned
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


def _overlay_difference(df1, df2, left_owned=None, right_owned=None):
    """Overlay Difference operation used in overlay function.

    Returns
    -------
    tuple[GeoDataFrame, bool]
        Result GeoDataFrame and whether the owned dispatch path was used.
    """
    # ADR-0036 boundary: spatial index produces index arrays only.
    # Phase 2: pass owned arrays to request device-resident index pairs.
    index_result = _intersecting_index_pairs(
        df1, df2, left_owned=left_owned, right_owned=right_owned,
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
    if idx1.size > 0 and left_owned is not None and right_owned is not None:
        from vibespatial.geometry.owned import concat_owned_scatter

        # The GPU polygon_intersection kernel uses Sutherland-Hodgman
        # which only handles convex clip polygons.  The difference
        # pipeline may produce concave intermediates (e.g., union of
        # overlapping circles), so keep AUTO until the kernel supports
        # concave clipping.
        _pairwise_mode = ExecutionMode.AUTO

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

        # Assemble full result: scatter differenced rows into the
        # original left owned array.  No to_shapely() materialisation.
        result_owned = concat_owned_scatter(
            left_owned, diff_owned, idx1_unique,
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
            # Vectorized grouped-union approach: for each left geometry, compute
            # left_i - union(overlapping right geometries).  Replaces per-geometry
            # Python loop with grouped shapely.union_all + vectorized
            # shapely.difference.
            left_geoms = np.asarray(df1.geometry, dtype=object)
            result_geoms = left_geoms.copy()

            if idx1.size > 0:
                # Ensure host arrays for Shapely fallback path — indices may
                # be CuPy when the owned path above raised an exception.
                h_idx1 = idx1.get() if hasattr(idx1, "get") else idx1
                h_idx2 = idx2.get() if hasattr(idx2, "get") else idx2

                right_geoms = np.asarray(df2.geometry, dtype=object)
                right_unions = np.empty(n_left, dtype=object)
                right_unions.fill(None)

                # O(N log N) grouping via np.split — avoids O(K*N) per-group
                # mask scan.
                idx1_unique, idx1_split_at = np.unique(h_idx1, return_index=True)
                idx2_groups = np.split(h_idx2, idx1_split_at[1:])
                for left_pos, neighbors_idx in zip(idx1_unique, idx2_groups):
                    neighbors = right_geoms[neighbors_idx]
                    if len(neighbors) == 1:
                        right_unions[left_pos] = neighbors[0]
                    else:
                        right_unions[left_pos] = shapely.union_all(neighbors)  # CPU-only mode

                has_neighbors = np.zeros(n_left, dtype=bool)
                has_neighbors[idx1_unique] = True
                result_geoms[has_neighbors] = shapely.difference(  # CPU-only mode
                    left_geoms[has_neighbors], right_unions[has_neighbors],
                )

        differences = GeoSeries(result_geoms, index=df1.index, crs=df1.crs)

    # Post-difference make_valid: use GPU path when owned backing is
    # available to avoid Shapely materialisation on the critical path.
    differences = _make_valid_geoseries(differences)
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
    dfintersection, used_inter = _overlay_intersection(df1, df2, left_owned, right_owned)
    dfdifference, used_diff = _overlay_difference(df1, df2, left_owned, right_owned)
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


def _overlay_symmetric_diff(df1, df2, left_owned=None, right_owned=None):
    """Overlay Symmetric Difference operation used in overlay function.

    Returns (GeoDataFrame, bool) -- result and whether any sub-op used owned dispatch.
    """
    dfdiff1, used1 = _overlay_difference(df1, df2, left_owned, right_owned)
    dfdiff2, used2 = _overlay_difference(df2, df1, right_owned, left_owned)
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
    dfinter, used_inter = _overlay_intersection(df1, df2, left_owned, right_owned)
    dfsym, used_sym = _overlay_symmetric_diff(df1, df2, left_owned, right_owned)
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
                df1, df2, left_owned, right_owned,
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

        if how in ["intersection", "symmetric_difference", "union", "identity"]:
            result.drop(["__idx1", "__idx2"], axis=1, inplace=True)

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

    if keep_geom_type:
        result_owned = getattr(result.geometry.values, "_owned", None)
        if result_owned is not None:
            result = _collection_extract_owned(result, geom_type, keep_geom_type_warning)
        else:
            result = _collection_extract(result, geom_type, keep_geom_type_warning)

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
