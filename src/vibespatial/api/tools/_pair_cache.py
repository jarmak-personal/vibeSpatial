from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

_MAX_INTERSECTION_PAIR_CACHE_ENTRIES = 64


@dataclass(frozen=True)
class _CachedIntersectionPairs:
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_index_values: np.ndarray
    left_index_unique: bool
    left_frame: object


_INTERSECTION_PAIR_CACHE: OrderedDict[
    tuple[tuple[str, int, int], tuple[str, int, int]],
    _CachedIntersectionPairs,
] = OrderedDict()


def pair_cache_token(df) -> tuple[str, int, int]:
    values = df.geometry.values
    owned = getattr(values, "_owned", None)
    if owned is not None:
        return ("owned", id(owned), len(df))
    return ("values", id(values), len(df))


def cache_intersection_pairs(left_df, right_df, left_indices, right_indices) -> None:
    left_key = (pair_cache_token(left_df), pair_cache_token(right_df))
    right_key = (left_key[1], left_key[0])
    left_index_values = np.asarray(left_df.index.to_numpy(copy=False), dtype=object)
    right_index_values = np.asarray(right_df.index.to_numpy(copy=False), dtype=object)
    _INTERSECTION_PAIR_CACHE[left_key] = _CachedIntersectionPairs(
        left_indices=np.asarray(left_indices, dtype=np.int32),
        right_indices=np.asarray(right_indices, dtype=np.int32),
        left_index_values=left_index_values,
        left_index_unique=bool(left_df.index.is_unique),
        left_frame=left_df,
    )
    _INTERSECTION_PAIR_CACHE[right_key] = _CachedIntersectionPairs(
        left_indices=np.asarray(right_indices, dtype=np.int32),
        right_indices=np.asarray(left_indices, dtype=np.int32),
        left_index_values=right_index_values,
        left_index_unique=bool(right_df.index.is_unique),
        left_frame=right_df,
    )
    _INTERSECTION_PAIR_CACHE.move_to_end(left_key)
    _INTERSECTION_PAIR_CACHE.move_to_end(right_key)
    while len(_INTERSECTION_PAIR_CACHE) > _MAX_INTERSECTION_PAIR_CACHE_ENTRIES:
        _INTERSECTION_PAIR_CACHE.popitem(last=False)


def get_cached_intersection_pairs(df1, df2):
    key = (pair_cache_token(df1), pair_cache_token(df2))
    cached = _INTERSECTION_PAIR_CACHE.get(key)
    if cached is not None:
        _INTERSECTION_PAIR_CACHE.move_to_end(key)
        return cached.left_indices, cached.right_indices

    if not bool(df1.index.is_unique):
        return None

    left_token, right_token = key
    subset_positions = {
        label: position
        for position, label in enumerate(df1.index.to_list())
    }
    for cached_key in reversed(_INTERSECTION_PAIR_CACHE):
        cached_left_token, cached_right_token = cached_key
        if cached_right_token != right_token:
            continue
        if cached_left_token[2] <= left_token[2]:
            continue
        entry = _INTERSECTION_PAIR_CACHE[cached_key]
        if not entry.left_index_unique:
            continue
        cached_left_df = entry.left_frame
        if cached_left_df is None:
            continue
        if tuple(cached_left_df.columns) != tuple(df1.columns):
            continue
        if getattr(cached_left_df.geometry, "name", None) != getattr(df1.geometry, "name", None):
            continue
        if getattr(cached_left_df, "crs", None) != getattr(df1, "crs", None):
            continue
        try:
            cached_subset = cached_left_df.loc[df1.index]
        except Exception:
            continue
        if len(cached_subset) != len(df1) or not cached_subset.index.equals(df1.index):
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
        return (
            mapped[keep].astype(np.int32, copy=False),
            entry.right_indices[keep],
        )
    return None
