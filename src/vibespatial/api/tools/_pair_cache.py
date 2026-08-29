from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from vibespatial.api._native_state import get_native_state

_MAX_INTERSECTION_PAIR_CACHE_ENTRIES = 64
_MAX_INTERSECTION_PAIR_CACHE_BYTES = 256 * 1024 * 1024


def _is_device_array(values) -> bool:
    return hasattr(values, "__cuda_array_interface__")


@dataclass(frozen=True)
class _CachedIntersectionPairs:
    left_indices: np.ndarray | None
    right_indices: np.ndarray | None
    left_index_values: np.ndarray | None
    left_index_unique: bool
    left_frame: object
    right_frame: object
    device_left_indices: object | None = None
    device_right_indices: object | None = None


_INTERSECTION_PAIR_CACHE: OrderedDict[
    tuple[tuple[str, int, int], tuple[str, int, int]],
    _CachedIntersectionPairs,
] = OrderedDict()


def _pair_cache_array_bytes(values) -> int:
    if values is None:
        return 0
    nbytes = getattr(values, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    return int(np.asarray(values).nbytes)


def _intersection_pair_cache_bytes() -> int:
    """Return retained pair-column bytes without double-counting reverse views."""
    seen: set[int] = set()
    total = 0
    for entry in _INTERSECTION_PAIR_CACHE.values():
        for values in (
            entry.left_indices,
            entry.right_indices,
            entry.device_left_indices,
            entry.device_right_indices,
        ):
            if values is None or id(values) in seen:
                continue
            seen.add(id(values))
            total += _pair_cache_array_bytes(values)
    return total


def _trim_intersection_pair_cache() -> None:
    while _INTERSECTION_PAIR_CACHE and (
        len(_INTERSECTION_PAIR_CACHE) > _MAX_INTERSECTION_PAIR_CACHE_ENTRIES
        or _intersection_pair_cache_bytes() > _MAX_INTERSECTION_PAIR_CACHE_BYTES
    ):
        key, _entry = _INTERSECTION_PAIR_CACHE.popitem(last=False)
        _INTERSECTION_PAIR_CACHE.pop((key[1], key[0]), None)


def pair_cache_token(df) -> tuple[str, int, int]:
    values = df.geometry.values
    owned = getattr(values, "_owned", None)
    if owned is not None:
        return ("owned", id(owned), len(df))
    return ("values", id(values), len(df))


def _cache_index_metadata(df, *, prefer_lazy: bool) -> tuple[np.ndarray | None, bool]:
    """Return host labels only when they are already a cheap public shape.

    Device relation caches are useful to downstream native overlay paths even
    when public index labels are still represented by ``NativeIndexPlan``.
    Materializing those labels only to support the optional subset-remap cache
    path is hidden fallback debt, so lazy/native indexes deliberately keep just
    their uniqueness contract here.
    """
    state = get_native_state(df)
    if state is not None:
        index_plan = state.index_plan
        if prefer_lazy and (
            index_plan.device_labels is not None
            or index_plan.take_positions is not None
        ):
            return None, not bool(index_plan.has_duplicates)
        if index_plan.index is not None:
            return (
                np.asarray(index_plan.index.to_numpy(copy=False), dtype=object),
                not bool(index_plan.has_duplicates),
            )
        return None, not bool(index_plan.has_duplicates)

    return np.asarray(df.index.to_numpy(copy=False), dtype=object), bool(df.index.is_unique)


def cache_intersection_pairs(left_df, right_df, left_indices, right_indices) -> None:
    left_key = (pair_cache_token(left_df), pair_cache_token(right_df))
    right_key = (left_key[1], left_key[0])
    is_device = hasattr(left_indices, "__cuda_array_interface__") or hasattr(
        right_indices,
        "__cuda_array_interface__",
    )
    left_index_values, left_index_unique = _cache_index_metadata(
        left_df,
        prefer_lazy=is_device,
    )
    right_index_values, right_index_unique = _cache_index_metadata(
        right_df,
        prefer_lazy=is_device,
    )

    def _merge_entry(
        key,
        *,
        host_left,
        host_right,
        device_left,
        device_right,
        index_values,
        index_unique,
        frame,
        other_frame,
    ) -> None:
        existing = _INTERSECTION_PAIR_CACHE.get(key)
        if existing is not None:
            if host_left is None:
                host_left = existing.left_indices
            if host_right is None:
                host_right = existing.right_indices
            if device_left is None:
                device_left = existing.device_left_indices
            if device_right is None:
                device_right = existing.device_right_indices
        _INTERSECTION_PAIR_CACHE[key] = _CachedIntersectionPairs(
            left_indices=host_left,
            right_indices=host_right,
            left_index_values=index_values,
            left_index_unique=bool(index_unique),
            left_frame=frame,
            right_frame=other_frame,
            device_left_indices=device_left,
            device_right_indices=device_right,
        )

    host_left_indices = None if is_device else np.asarray(left_indices, dtype=np.int32)
    host_right_indices = None if is_device else np.asarray(right_indices, dtype=np.int32)
    device_left_indices = left_indices if is_device else None
    device_right_indices = right_indices if is_device else None
    _merge_entry(
        left_key,
        host_left=host_left_indices,
        host_right=host_right_indices,
        device_left=device_left_indices,
        device_right=device_right_indices,
        index_values=left_index_values,
        index_unique=left_index_unique,
        frame=left_df,
        other_frame=right_df,
    )
    _merge_entry(
        right_key,
        host_left=host_right_indices,
        host_right=host_left_indices,
        device_left=device_right_indices,
        device_right=device_left_indices,
        index_values=right_index_values,
        index_unique=right_index_unique,
        frame=right_df,
        other_frame=left_df,
    )
    _INTERSECTION_PAIR_CACHE.move_to_end(left_key)
    _INTERSECTION_PAIR_CACHE.move_to_end(right_key)
    _trim_intersection_pair_cache()


def _device_intersection_pairs(left_indices, right_indices):
    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU callers
        return None

    from vibespatial.spatial.query_types import DeviceSpatialJoinResult

    return DeviceSpatialJoinResult(
        d_left_idx=cp.asarray(left_indices, dtype=cp.int32),
        d_right_idx=cp.asarray(right_indices, dtype=cp.int32),
    )


def _cached_frames_have_compatible_subset_surface(df1, cached_left_df) -> bool:
    if cached_left_df is None:
        return False
    if tuple(cached_left_df.columns) != tuple(df1.columns):
        return False
    if getattr(cached_left_df.geometry, "name", None) != getattr(df1.geometry, "name", None):
        return False
    return getattr(cached_left_df, "crs", None) == getattr(df1, "crs", None)


def _device_subset_remap_result(df1, cached: _CachedIntersectionPairs):
    """Remap cached source-frame relation rows through native subset positions."""
    if cached.device_left_indices is None or cached.device_right_indices is None:
        return None
    cached_left_df = cached.left_frame
    if not _cached_frames_have_compatible_subset_surface(df1, cached_left_df):
        return None

    source_state = get_native_state(cached_left_df)
    subset_state = get_native_state(df1)
    if not bool(cached.left_index_unique):
        return None
    if source_state is not None and subset_state is not None:
        plan = subset_state.index_plan
        if bool(getattr(plan, "has_duplicates", False)):
            return None
        selection_positions = getattr(plan, "selection_positions", None)
        if selection_positions is None:
            return None
        if getattr(plan, "selection_source_token", None) != source_state.lineage_token:
            return None
        if getattr(plan, "selection_source_row_count", None) != source_state.row_count:
            return None
        source_row_count = int(source_state.row_count)
        left_token = subset_state.lineage_token
    else:
        cached_values = getattr(getattr(cached_left_df, "geometry", None), "values", None)
        subset_values = getattr(getattr(df1, "geometry", None), "values", None)
        cached_owned = getattr(cached_values, "_owned", None)
        subset_owned = getattr(subset_values, "_owned", None)
        selection_source_owned = getattr(
            subset_values,
            "_selection_source_owned",
            None,
        )
        selection_positions = getattr(subset_values, "_selection_positions", None)
        has_public_selection_provenance = (
            cached_owned is not None
            and selection_source_owned is cached_owned
            and _is_device_array(selection_positions)
        )
        has_owned_view_provenance = (
            cached_owned is not None
            and subset_owned is not None
            and bool(getattr(subset_owned, "is_indexed_view", False))
            and getattr(subset_owned, "_base", None) is cached_owned
            and bool(getattr(subset_owned, "_index_map_unique", False))
        )
        if not (has_public_selection_provenance or has_owned_view_provenance):
            return None
        if not has_public_selection_provenance:
            selection_positions = getattr(subset_owned, "_index_map", None)
        source_row_count = int(cached_owned.row_count)
        left_token = None
    if not _is_device_array(selection_positions):
        return None

    try:
        import cupy as cp
    except ModuleNotFoundError:  # pragma: no cover - guarded by GPU callers
        return None

    d_selection = cp.asarray(selection_positions, dtype=cp.int64)
    if int(d_selection.size) != len(df1):
        return None
    d_cached_left = cp.asarray(cached.device_left_indices, dtype=cp.int64)
    d_cached_right = cp.asarray(cached.device_right_indices, dtype=cp.int32)
    if int(d_cached_left.size) != int(d_cached_right.size):
        return None

    d_source_to_subset = cp.full(source_row_count, -1, dtype=cp.int32)
    d_subset_rows = cp.arange(len(df1), dtype=cp.int32)
    d_source_to_subset[d_selection] = d_subset_rows
    d_mapped_left = d_source_to_subset[d_cached_left]
    from vibespatial.api._native_relation import (
        NativeRelation,
        NativeRelationSelection,
    )
    from vibespatial.api._native_rowset import NativeDeviceSelection

    right_state = get_native_state(cached.right_frame)
    d_keep = d_mapped_left >= 0
    relation = NativeRelation(
        left_indices=d_mapped_left.astype(cp.int32, copy=False),
        right_indices=d_cached_right,
        left_token=left_token,
        right_token=(None if right_state is None else right_state.lineage_token),
        predicate="intersects",
        left_row_count=len(df1),
        right_row_count=len(cached.right_frame),
        sorted_by_left=False,
        origin="intersection-pair-cache-subset",
    )
    return NativeRelationSelection(
        relation=relation,
        selection=NativeDeviceSelection.from_mask(
            d_keep,
            source_row_count=int(d_keep.size),
        ),
    )


def _cached_entry_result(cached: _CachedIntersectionPairs, *, return_device: bool):
    if return_device:
        if cached.device_left_indices is not None and cached.device_right_indices is not None:
            return _device_intersection_pairs(
                cached.device_left_indices,
                cached.device_right_indices,
            )
        if cached.left_indices is not None and cached.right_indices is not None:
            return _device_intersection_pairs(cached.left_indices, cached.right_indices)
        return None
    if cached.left_indices is None or cached.right_indices is None:
        return None
    return cached.left_indices, cached.right_indices


def get_cached_intersection_pairs(df1, df2, *, return_device: bool = False):
    key = (pair_cache_token(df1), pair_cache_token(df2))
    cached = _INTERSECTION_PAIR_CACHE.get(key)
    if cached is not None:
        _INTERSECTION_PAIR_CACHE.move_to_end(key)
        return _cached_entry_result(cached, return_device=return_device)

    if not bool(df1.index.is_unique):
        return None

    left_token, right_token = key
    subset_index_values = None
    for cached_key in reversed(_INTERSECTION_PAIR_CACHE):
        cached_left_token, cached_right_token = cached_key
        if cached_right_token != right_token:
            continue
        if cached_left_token[2] <= left_token[2]:
            continue
        entry = _INTERSECTION_PAIR_CACHE[cached_key]
        if not entry.left_index_unique:
            continue

        if return_device:
            device_result = _device_subset_remap_result(df1, entry)
            if device_result is not None:
                _INTERSECTION_PAIR_CACHE.move_to_end(cached_key)
                return device_result

        if entry.left_indices is None or entry.right_indices is None:
            continue
        if entry.left_index_values is None:
            continue
        cached_left_df = entry.left_frame
        if not _cached_frames_have_compatible_subset_surface(df1, cached_left_df):
            continue
        if subset_index_values is None:
            subset_index_values, subset_unique = _cache_index_metadata(
                df1,
                prefer_lazy=True,
            )
            if subset_index_values is None or not bool(subset_unique):
                continue
        subset_positions = {
            label: position
            for position, label in enumerate(subset_index_values.tolist())
        }
        try:
            cached_subset = cached_left_df.loc[subset_index_values]
        except Exception:
            continue
        cached_subset_index = np.asarray(
            cached_subset.index.to_numpy(copy=False),
            dtype=object,
        )
        if len(cached_subset) != len(df1) or not np.array_equal(
            cached_subset_index,
            subset_index_values,
        ):
            continue
        geometry_name = df1.geometry.name
        cached_attrs = cached_subset.drop(columns=[geometry_name], errors="ignore")
        df1_attrs = df1.drop(columns=[geometry_name], errors="ignore")
        if not cached_attrs.equals(df1_attrs):
            continue
        cached_labels = entry.left_index_values[entry.left_indices.astype(np.intp, copy=False)]
        mapped = np.fromiter(
            (subset_positions.get(label, -1) for label in cached_labels.tolist()),
            dtype=np.int64,
            count=entry.left_indices.size,
        )
        keep = mapped >= 0
        if not keep.any():
            continue
        _INTERSECTION_PAIR_CACHE.move_to_end(cached_key)
        left = mapped[keep].astype(np.int32, copy=False)
        right = entry.right_indices[keep]
        if return_device:
            return _device_intersection_pairs(left, right)
        return left, right
    return None
