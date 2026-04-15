from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.api._native_result_core import (
    GeometryNativeResult,
    NativeAttributeTable,
    NativeGeometryColumn,
    NativeReadProvenance,
    NativeTabularResult,
    _host_array,
    _materialize_attribute_geometry_frame,
    _replace_geometry_column_preserving_backing,
    _set_active_geometry_name,
    native_attribute_table_from_arrow_table,  # noqa: F401
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.fallbacks import record_fallback_event

if TYPE_CHECKING:
    from vibespatial.api.geodataframe import GeoDataFrame
    from vibespatial.api.geoseries import GeoSeries


@dataclass(frozen=True)
class RelationIndexResult:
    """Low-level relation result carrying row-pair indices."""

    left_indices: Any
    right_indices: Any

    def to_host(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            _host_array(self.left_indices, dtype=np.int32),
            _host_array(self.right_indices, dtype=np.int32),
        )

    @property
    def size(self) -> int:
        left, _right = self.to_host()
        return int(left.size)


@dataclass(frozen=True)
class PairwiseConstructiveResult:
    """Constructive result assembled from left/right relation pairs."""

    geometry: GeometryNativeResult
    relation: RelationIndexResult
    keep_geom_type_applied: bool = False

    def to_geodataframe(
        self,
        df1: GeoDataFrame,
        df2: GeoDataFrame,
        *,
        attribute_assembler: Callable[[Any, Any, pd.DataFrame, pd.DataFrame], pd.DataFrame],
    ) -> GeoDataFrame:
        from vibespatial.api.geodataframe import GeoDataFrame

        left_idx, right_idx = self.relation.to_host()
        attrs = attribute_assembler(
            left_idx,
            right_idx,
            df1.drop(df1._geometry_column_name, axis=1),
            df2.drop(df2._geometry_column_name, axis=1),
        )
        geom = self.geometry.to_geoseries(index=attrs.index, name="geometry")
        result = pd.DataFrame(attrs, copy=False)
        result["geometry"] = pd.Series(geom.values, index=attrs.index, copy=False, name="geometry")
        result.__class__ = GeoDataFrame
        result._geometry_column_name = "geometry"
        result = _replace_geometry_column_preserving_backing(
            result,
            geom.values,
            crs=self.geometry.crs,
        )
        if self.keep_geom_type_applied:
            result.attrs["_vibespatial_keep_geom_type_applied"] = True
        return result


def _reset_index_with_suffix(df, suffix, other):
    """Equivalent of ``df.reset_index()`` with suffixed auto index column names."""
    index_original = df.index.names
    if PANDAS_GE_30:
        df_reset = df.reset_index()
    else:
        df_reset = df
        df_reset.reset_index(inplace=True)
    column_names = df_reset.columns.to_numpy(copy=True)
    for i, label in enumerate(index_original):
        if label is None:
            new_label = column_names[i]
            if "level" in new_label:
                lev = new_label.split("_")[1]
                new_label = f"index_{suffix}{lev}"
            else:
                new_label = f"index_{suffix}"
            if new_label in df.columns or new_label in other.columns:
                raise ValueError(
                    f"'{new_label}' cannot be a column name in the frames being joined"
                )
            column_names[i] = new_label
    return df_reset, pd.Index(column_names)


def _process_column_names_with_suffix(
    left: pd.Index, right: pd.Index, suffixes, left_df, right_df
):
    """Add suffixes to overlapping labels while preserving the active geometry name."""
    to_rename = left.intersection(right)
    if len(to_rename) == 0:
        return left, right

    lsuffix, rsuffix = suffixes

    if not lsuffix and not rsuffix:
        raise ValueError(f"columns overlap but no suffix specified: {to_rename}")

    def renamer(x, suffix, geometry):
        if x in to_rename and x != geometry and suffix is not None:
            return f"{x}_{suffix}"
        return x

    lrenamer = partial(
        renamer,
        suffix=lsuffix,
        geometry=getattr(left_df, "_geometry_column_name", None),
    )
    rrenamer = partial(
        renamer,
        suffix=rsuffix,
        geometry=getattr(right_df, "_geometry_column_name", None),
    )

    left_renamed = pd.Index([lrenamer(lab) for lab in left])
    right_renamed = pd.Index([rrenamer(lab) for lab in right])

    dups = []
    if not left_renamed.is_unique:
        dups = left_renamed[
            (left_renamed.duplicated()) & (~left.duplicated())
        ].tolist()
    if not right_renamed.is_unique:
        dups.extend(
            right_renamed[
                (right_renamed.duplicated()) & (~right.duplicated())
            ].tolist()
        )
    if dups:
        warnings.warn(
            f"Passing 'suffixes' which cause duplicate columns {set(dups)} in the "
            f"result is deprecated and will raise a MergeError in a future version.",
            FutureWarning,
            stacklevel=4,
        )

    return left_renamed, right_renamed


def _restore_index(joined, index_names, index_names_original):
    """Restore the original index names, including unnamed levels."""
    if PANDAS_GE_30:
        joined = joined.set_index(list(index_names))
    else:
        joined.set_index(list(index_names), inplace=True)

    joined_index_names = list(joined.index.names)
    for i, label in enumerate(index_names_original):
        if label is None:
            joined_index_names[i] = None
    joined.index.names = joined_index_names
    return joined


def _adjust_join_indexers(indices, distances, left_length, right_length, how):
    """Adjust relation indices for outer, left, and right join semantics."""
    if how == "inner":
        return indices, distances

    l_idx, r_idx = indices

    if how == "outer":
        if l_idx.size:
            order = np.lexsort((r_idx, l_idx))
            l_idx = l_idx[order]
            r_idx = r_idx[order]
            if distances is not None:
                distances = distances[order]
        matched_left = np.zeros(left_length, dtype=bool)
        matched_right = np.zeros(right_length, dtype=bool)
        if l_idx.size:
            matched_left[l_idx] = True
            matched_right[r_idx] = True
        right_missing = np.flatnonzero(~matched_right)
        counts = (
            np.bincount(l_idx, minlength=left_length)
            if l_idx.size
            else np.zeros(left_length, dtype=np.intp)
        )
        repeats = np.maximum(counts, 1)
        left_row_count = int(repeats.sum())
        left_positions = np.arange(left_length, dtype=np.intp)
        outer_l_idx = np.repeat(left_positions, repeats)
        outer_r_idx = np.full(left_row_count, -1, dtype=np.intp)
        outer_distances = (
            None
            if distances is None
            else np.full(left_row_count, np.nan, dtype=np.float64)
        )
        if l_idx.size:
            match_offsets = np.cumsum(counts, dtype=np.intp) - counts
            rank_in_left = (
                np.arange(l_idx.size, dtype=np.intp)
                - np.repeat(match_offsets, counts)
            )
            row_starts = np.cumsum(repeats, dtype=np.intp) - repeats
            target = row_starts[l_idx] + rank_in_left
            outer_r_idx[target] = r_idx
            if outer_distances is not None:
                outer_distances[target] = distances
        l_idx = np.concatenate(
            (outer_l_idx, np.full(right_missing.size, -1, dtype=np.intp))
        )
        r_idx = np.concatenate((outer_r_idx, right_missing.astype(np.intp, copy=False)))
        if distances is not None:
            distances = np.concatenate(
                (
                    outer_distances,
                    np.full(right_missing.size, np.nan, dtype=np.float64),
                )
            )
        return (l_idx, r_idx), distances

    if how == "right":
        indexer = np.lexsort((l_idx, r_idx))
        l_idx, r_idx = l_idx[indexer], r_idx[indexer]
        if distances is not None:
            distances = distances[indexer]
        r_idx, l_idx = l_idx, r_idx

    original_length = right_length if how == "right" else left_length
    idx = np.arange(original_length)
    l_idx_missing = idx[~np.isin(idx, l_idx)]
    insert_idx = np.searchsorted(l_idx, l_idx_missing)
    l_idx = np.insert(l_idx, insert_idx, l_idx_missing)
    r_idx = np.insert(r_idx, insert_idx, -1)
    if distances is not None:
        distances = np.insert(distances, insert_idx, np.nan)

    if how == "right":
        l_idx, r_idx = r_idx, l_idx

    return (l_idx, r_idx), distances


def _reassemble_outer_geometry(left_geometry, right_geometry, l_idx, r_idx, new_index):
    """Reassemble outer-join geometry using only geometry frames and index arrays."""
    left_geometry_series = left_geometry._reindex_with_indexers(
        {0: (new_index, l_idx)}
    ).iloc[:, 0]
    right_geometry_series = right_geometry._reindex_with_indexers(
        {0: (new_index, r_idx)}
    ).iloc[:, 0]
    return left_geometry_series.where(
        left_geometry_series.notna(), right_geometry_series
    )
def _drop_private_attribute_columns(attributes: pd.DataFrame) -> pd.DataFrame:
    """Strip internal provenance columns from the shared export boundary."""
    private_columns = [
        column for column in ("__idx1", "__idx2") if column in attributes.columns
    ]
    if not private_columns:
        return attributes
    return attributes.drop(columns=private_columns)


def _native_attribute_table_from_projected_frames(
    frames: list[pd.DataFrame],
    *,
    index_override: pd.Index,
    storage: str = "arrow",
) -> NativeAttributeTable:
    if storage == "pandas":
        if not frames:
            return NativeAttributeTable(dataframe=pd.DataFrame(index=index_override))
        normalized_frames: list[pd.DataFrame] = []
        for frame in frames:
            if frame.index.equals(index_override):
                normalized_frames.append(frame)
                continue
            rebased = frame.copy(deep=False)
            rebased.index = index_override
            normalized_frames.append(rebased)
        concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
        frame = pd.concat(normalized_frames, axis=1, **concat_kwargs)
        return NativeAttributeTable(dataframe=frame)

    if storage != "arrow":
        raise ValueError(f"Unsupported projected-frame storage: {storage!r}")

    import pyarrow as pa

    tables = []
    declared_names: list[Any] = []
    for frame in frames:
        if frame.shape[1] == 0:
            continue
        tables.append(pa.Table.from_pandas(frame, preserve_index=False))
        declared_names.extend(list(frame.columns))

    if not tables:
        return NativeAttributeTable(dataframe=pd.DataFrame(index=index_override))

    arrays = []
    names: list[str] = []
    for table in tables:
        arrays.extend(table.columns)
        names.extend(table.column_names)
    return NativeAttributeTable(
        arrow_table=pa.Table.from_arrays(arrays, names=names),
        index_override=index_override,
        column_override=tuple(declared_names),
    )


def _projected_frames_to_native_tabular_parts(
    frames: list[pd.DataFrame],
    *,
    index_override: pd.Index,
    storage: str = "arrow",
    dropped_column_names: set[str] | frozenset[str] = frozenset(),
) -> tuple[NativeAttributeTable, tuple[NativeGeometryColumn, ...], tuple[str, ...]]:
    """Split projected frames into attribute storage plus explicit secondary geometry."""
    from vibespatial.api.geo_base import _is_geometry_like_dtype

    attribute_frames: list[pd.DataFrame] = []
    secondary_geometry: list[NativeGeometryColumn] = []
    column_entries: list[tuple[str, Any]] = []
    geometry_names: set[Any] = set()

    for frame in frames:
        if frame.shape[1] == 0:
            continue
        geometry_mask = np.asarray(frame.dtypes.map(_is_geometry_like_dtype), dtype=bool)
        attribute_columns: list[Any] = []
        for position, column in enumerate(frame.columns):
            column_name = column
            if column_name in dropped_column_names:
                continue
            if geometry_mask[position]:
                if column_name in geometry_names:
                    raise ValueError(
                        "duplicate projected geometry columns are not supported in the native tabular boundary"
                    )
                series = frame.iloc[:, position]
                if not series.index.equals(index_override):
                    series = series.copy(deep=False)
                    series.index = index_override
                secondary_geometry.append(
                    NativeGeometryColumn(
                        column_name,
                        GeometryNativeResult.from_values(
                            series.values,
                            crs=getattr(series, "crs", None),
                            index=index_override,
                            name=column_name,
                        ),
                    )
                )
                geometry_names.add(column_name)
                column_entries.append(("geometry", column_name))
                continue
            attribute_columns.append(column)
            column_entries.append(("attribute", column_name))

        if not attribute_columns:
            continue
        attrs = frame.loc[:, attribute_columns]
        if not attrs.index.equals(index_override):
            attrs = attrs.copy(deep=False)
            attrs.index = index_override
        attribute_frames.append(attrs)

    attributes = _native_attribute_table_from_projected_frames(
        attribute_frames,
        index_override=index_override,
        storage=storage,
    )
    attribute_column_iter = iter(attributes.columns)
    column_order: list[Any] = []
    for kind, column_name in column_entries:
        if kind == "attribute":
            column_order.append(next(attribute_column_iter))
        else:
            column_order.append(column_name)
    return attributes, tuple(secondary_geometry), tuple(column_order)


def _rename_native_geometry_columns(
    columns: tuple[NativeGeometryColumn, ...],
    mapping: dict[str, str] | None,
) -> tuple[NativeGeometryColumn, ...]:
    if not mapping:
        return columns
    return tuple(
        NativeGeometryColumn(mapping.get(column.name, column.name), column.geometry)
        for column in columns
    )


def _rename_output_column_order(
    column_order: tuple[str, ...],
    mapping: dict[str, str] | None,
) -> tuple[str, ...]:
    if not mapping:
        return column_order
    return tuple(mapping.get(column_name, column_name) for column_name in column_order)


def _left_constructive_output_column_order(
    *,
    df: GeoDataFrame,
    geometry_name: str,
    attributes: NativeAttributeTable,
    secondary_geometry: tuple[NativeGeometryColumn, ...],
) -> tuple[str, ...]:
    from vibespatial.api.geo_base import _is_geometry_like_dtype

    source_geometry_name = df._geometry_column_name
    source_non_geometry = df.drop(source_geometry_name, axis=1).copy(deep=False)
    secondary_iter = iter(secondary_geometry)
    attribute_iter = iter(attributes.columns)
    order: list[str] = []
    for column_name in df.columns:
        if column_name == source_geometry_name:
            order.append(geometry_name)
            continue
        if column_name == geometry_name:
            continue
        if (
            column_name in source_non_geometry.columns
            and _is_geometry_like_dtype(source_non_geometry[column_name].dtype)
        ):
            order.append(next(secondary_iter).name)
            continue
        order.append(next(attribute_iter))
    return tuple(order)


def _native_pairwise_attribute_table(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    left_idx,
    right_idx,
    *,
    lsuffix: str = "1",
    rsuffix: str = "2",
) -> NativeAttributeTable:
    """Project pairwise attribute rows directly into the native table boundary."""
    h_left_idx = left_idx.get() if hasattr(left_idx, "get") else left_idx
    h_right_idx = right_idx.get() if hasattr(right_idx, "get") else right_idx
    row_count = len(h_left_idx)
    out_index = pd.RangeIndex(row_count)

    left_attrs = left_df.drop(left_df._geometry_column_name, axis=1).copy(deep=False)
    right_attrs = right_df.drop(right_df._geometry_column_name, axis=1).copy(deep=False)
    left_columns, right_columns = _process_column_names_with_suffix(
        left_attrs.columns,
        right_attrs.columns,
        (lsuffix, rsuffix),
        left_attrs,
        right_attrs,
    )
    left_attrs.columns = left_columns
    right_attrs.columns = right_columns

    left_projected = left_attrs._reindex_with_indexers({0: (out_index, h_left_idx)})
    right_projected = right_attrs._reindex_with_indexers({0: (out_index, h_right_idx)})
    return _native_attribute_table_from_projected_frames(
        [left_projected, right_projected],
        index_override=out_index,
    )


def _project_pairwise_attribute_frame(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    left_idx,
    right_idx,
    *,
    lsuffix: str = "1",
    rsuffix: str = "2",
) -> pd.DataFrame:
    """Project pairwise attribute rows directly into a pandas frame.

    This is the public GeoPandas export fast path: keep the projection
    shape from the native pairwise boundary, but skip the Arrow round-trip
    when the caller is going to materialize a GeoDataFrame immediately.
    """
    h_left_idx = left_idx.get() if hasattr(left_idx, "get") else left_idx
    h_right_idx = right_idx.get() if hasattr(right_idx, "get") else right_idx
    row_count = len(h_left_idx)
    out_index = pd.RangeIndex(row_count)

    left_attrs = left_df.drop(left_df._geometry_column_name, axis=1).copy(deep=False)
    right_attrs = right_df.drop(right_df._geometry_column_name, axis=1).copy(deep=False)
    left_columns, right_columns = _process_column_names_with_suffix(
        left_attrs.columns,
        right_attrs.columns,
        (lsuffix, rsuffix),
        left_attrs,
        right_attrs,
    )
    left_attrs.columns = left_columns
    right_attrs.columns = right_columns

    left_projected = left_attrs._reindex_with_indexers({0: (out_index, h_left_idx)})
    right_projected = right_attrs._reindex_with_indexers({0: (out_index, h_right_idx)})
    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
    return pd.concat([left_projected, right_projected], axis=1, **concat_kwargs)


def _relation_join_output_layout(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    *,
    how: str,
    lsuffix: str,
    rsuffix: str,
    on_attribute=None,
) -> tuple[str, Any, tuple[str, ...]]:
    """Compute joined column order without materializing joined rows."""
    left_geometry_name = left_df.geometry.name
    left_crs = left_df.crs
    right_geometry_name = right_df.geometry.name
    right_crs = right_df.crs

    left_schema = left_df
    right_schema = right_df
    if on_attribute:
        right_schema = right_schema.drop(on_attribute, axis=1)

    if how in ("inner", "left"):
        right_schema = right_schema.drop(right_geometry_name, axis=1)
    elif how == "right":
        left_schema = left_schema.drop(left_geometry_name, axis=1)
    else:
        left_schema = left_schema.drop(left_geometry_name, axis=1)
        right_schema = right_schema.drop(right_geometry_name, axis=1)

    left_schema = left_schema.copy(deep=False)
    left_nlevels = left_schema.index.nlevels
    left_index_original = left_schema.index.names
    left_schema, left_column_names = _reset_index_with_suffix(left_schema, lsuffix, right_schema)

    right_schema = right_schema.copy(deep=False)
    right_nlevels = right_schema.index.nlevels
    right_index_original = right_schema.index.names
    right_schema, right_column_names = _reset_index_with_suffix(right_schema, rsuffix, left_schema)

    left_column_names, right_column_names = _process_column_names_with_suffix(
        left_column_names,
        right_column_names,
        (lsuffix, rsuffix),
        left_schema,
        right_schema,
    )
    left_schema.columns = left_column_names
    right_schema.columns = right_column_names
    left_index = left_schema.columns[:left_nlevels]
    right_index = right_schema.columns[:right_nlevels]

    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
    joined = pd.concat([left_schema.iloc[:0], right_schema.iloc[:0]], axis=1, **concat_kwargs)

    if how in ("inner", "left"):
        joined = _restore_index(joined, left_index, left_index_original)
        geometry_name = left_geometry_name
        crs = left_crs
    elif how == "right":
        joined = _restore_index(joined, right_index, right_index_original)
        geometry_name = right_geometry_name
        crs = right_crs
    else:
        joined[ left_geometry_name ] = pd.Series(dtype=object)
        geometry_name = left_geometry_name
        crs = left_crs

    return geometry_name, crs, tuple(joined.columns)


def _native_relation_join_parts(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    distances,
    *,
    how: str,
    lsuffix: str,
    rsuffix: str,
    on_attribute=None,
    attribute_storage: str = "arrow",
) -> tuple[
    NativeAttributeTable,
    GeometryNativeResult,
    str,
    Any,
    tuple[str, ...],
    tuple[NativeGeometryColumn, ...],
    np.ndarray | None,
]:
    """Build native relation-join export parts without constructing a joined DataFrame."""
    left_geometry_name = left_df.geometry.name
    left_crs = left_df.crs
    right_geometry_name = right_df.geometry.name
    right_crs = right_df.crs
    geometry_name, crs, output_layout = _relation_join_output_layout(
        left_df,
        right_df,
        how=how,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        on_attribute=on_attribute,
    )

    left_geometry = None
    right_geometry = None
    right_attr_df = right_df
    if on_attribute:
        right_attr_df = right_attr_df.drop(on_attribute, axis=1)

    if how in ("inner", "left"):
        left_attr_df = left_df.drop(left_geometry_name, axis=1)
        right_attr_df = right_attr_df.drop(right_geometry_name, axis=1)
    elif how == "right":
        left_attr_df = left_df.drop(left_geometry_name, axis=1)
        right_attr_df = right_attr_df.drop(right_geometry_name, axis=1)
    else:
        left_geometry = left_df[[left_geometry_name]].copy(deep=False)
        right_geometry = right_df[[right_geometry_name]].copy(deep=False)
        left_attr_df = left_df.drop(left_geometry_name, axis=1)
        right_attr_df = right_attr_df.drop(right_geometry_name, axis=1)

    left_attr_df = left_attr_df.copy(deep=False)
    left_nlevels = left_attr_df.index.nlevels
    left_index_original = left_attr_df.index.names
    left_attr_df, left_column_names = _reset_index_with_suffix(left_attr_df, lsuffix, right_attr_df)

    right_attr_df = right_attr_df.copy(deep=False)
    right_nlevels = right_attr_df.index.nlevels
    right_index_original = right_attr_df.index.names
    right_attr_df, right_column_names = _reset_index_with_suffix(right_attr_df, rsuffix, left_attr_df)

    left_column_names, right_column_names = _process_column_names_with_suffix(
        left_column_names,
        right_column_names,
        (lsuffix, rsuffix),
        left_attr_df,
        right_attr_df,
    )
    left_attr_df.columns = left_column_names
    right_attr_df.columns = right_column_names
    left_index = left_attr_df.columns[:left_nlevels]
    right_index = right_attr_df.columns[:right_nlevels]

    (l_idx, r_idx), distances = _adjust_join_indexers(
        indices, distances, len(left_attr_df), len(right_attr_df), how
    )
    new_index = pd.RangeIndex(len(l_idx))
    direct_take = (
        how == "inner"
        and len(l_idx) > 0
        and np.all(np.asarray(l_idx) >= 0)
        and np.all(np.asarray(r_idx) >= 0)
    )
    if direct_take:
        l_take = np.asarray(l_idx, dtype=np.intp)
        r_take = np.asarray(r_idx, dtype=np.intp)
        left_projected = left_attr_df.take(l_take)
        right_projected = right_attr_df.take(r_take)
        left_projected.index = new_index
        right_projected.index = new_index
    else:
        left_projected = left_attr_df._reindex_with_indexers({0: (new_index, l_idx)})
        right_projected = right_attr_df._reindex_with_indexers({0: (new_index, r_idx)})

    if how in ("inner", "left"):
        left_projected = _restore_index(left_projected, left_index, left_index_original)
        joined_index = left_projected.index
        geometry_name = left_geometry_name
        crs = left_crs
        if direct_take:
            geometry_values = left_df.geometry.values.take(np.asarray(l_idx, dtype=np.intp))
        else:
            geometry_values = left_df[[left_geometry_name]]._reindex_with_indexers(
                {0: (new_index, l_idx)}
            ).iloc[:, 0].values
        geometry_result = GeometryNativeResult.from_values(
            geometry_values,
            crs=left_crs,
            index=joined_index,
            name=left_geometry_name,
        )
    elif how == "right":
        right_projected = _restore_index(right_projected, right_index, right_index_original)
        joined_index = right_projected.index
        geometry_name = right_geometry_name
        crs = right_crs
        if direct_take:
            geometry_values = right_df.geometry.values.take(np.asarray(r_idx, dtype=np.intp))
        else:
            geometry_values = right_df[[right_geometry_name]]._reindex_with_indexers(
                {0: (new_index, r_idx)}
            ).iloc[:, 0].values
        geometry_result = GeometryNativeResult.from_values(
            geometry_values,
            crs=right_crs,
            index=joined_index,
            name=right_geometry_name,
        )
    else:
        joined_index = new_index
        geometry_series = _reassemble_outer_geometry(
            left_geometry,
            right_geometry,
            l_idx,
            r_idx,
            new_index,
        )
        geometry_result = GeometryNativeResult.from_values(
            geometry_series.values,
            crs=left_geometry.crs,
            index=joined_index,
            name=left_geometry_name,
        )

    attributes, secondary_geometry, _projected_order = _projected_frames_to_native_tabular_parts(
        [left_projected, right_projected],
        index_override=joined_index,
        storage=attribute_storage,
        dropped_column_names={geometry_name},
    )
    known_columns = set(attributes.columns) | {geometry_name} | {
        column.name for column in secondary_geometry
    }
    column_order = tuple(
        column_name for column_name in output_layout if column_name in known_columns
    )
    if geometry_name not in column_order:
        column_order = tuple([geometry_name, *column_order])
    return (
        attributes,
        geometry_result,
        geometry_name,
        crs,
        column_order,
        secondary_geometry,
        distances,
    )


def _materialize_relation_join_parts(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    distances,
    *,
    how: str,
    lsuffix: str,
    rsuffix: str,
    on_attribute=None,
) -> tuple[pd.DataFrame, GeometryNativeResult, str, Any, tuple[str, ...], np.ndarray | None]:
    """Explicit relation-join assembly before final GeoPandas export."""
    left_geometry_name = left_df.geometry.name
    left_crs = left_df.crs
    right_geometry_name = right_df.geometry.name
    right_crs = right_df.crs

    left_geometry = None
    right_geometry = None
    if on_attribute:
        right_df = right_df.drop(on_attribute, axis=1)

    if how in ("inner", "left"):
        right_df = right_df.drop(right_df.geometry.name, axis=1)
    elif how == "right":
        left_df = left_df.drop(left_df.geometry.name, axis=1)
    else:
        left_geometry = left_df[[left_df.geometry.name]].copy(deep=False)
        right_geometry = right_df[[right_df.geometry.name]].copy(deep=False)
        left_df = left_df.drop(left_df.geometry.name, axis=1)
        right_df = right_df.drop(right_df.geometry.name, axis=1)

    left_df = left_df.copy(deep=False)
    left_nlevels = left_df.index.nlevels
    left_index_original = left_df.index.names
    left_df, left_column_names = _reset_index_with_suffix(left_df, lsuffix, right_df)

    right_df = right_df.copy(deep=False)
    right_nlevels = right_df.index.nlevels
    right_index_original = right_df.index.names
    right_df, right_column_names = _reset_index_with_suffix(right_df, rsuffix, left_df)

    left_column_names, right_column_names = _process_column_names_with_suffix(
        left_column_names,
        right_column_names,
        (lsuffix, rsuffix),
        left_df,
        right_df,
    )
    left_df.columns = left_column_names
    right_df.columns = right_column_names
    left_index = left_df.columns[:left_nlevels]
    right_index = right_df.columns[:right_nlevels]

    (l_idx, r_idx), distances = _adjust_join_indexers(
        indices, distances, len(left_df), len(right_df), how
    )
    new_index = pd.RangeIndex(len(l_idx))
    left = left_df._reindex_with_indexers({0: (new_index, l_idx)})
    right = right_df._reindex_with_indexers({0: (new_index, r_idx)})
    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
    joined = pd.concat([left, right], axis=1, **concat_kwargs)

    if how in ("inner", "left"):
        joined = _restore_index(joined, left_index, left_index_original)
        geometry_name = left_geometry_name
        crs = left_crs
    elif how == "right":
        joined = _restore_index(joined, right_index, right_index_original)
        geometry_name = right_geometry_name
        crs = right_crs
    else:
        geometry = _reassemble_outer_geometry(
            left_geometry, right_geometry, l_idx, r_idx, new_index
        )
        joined[left_geometry.columns[0]] = geometry
        geometry_name = left_geometry.columns[0]
        crs = left_geometry.crs

    column_order = tuple(joined.columns)
    geometry_result = GeometryNativeResult.from_values(
        joined[geometry_name].values,
        crs=crs,
        index=joined.index,
        name=geometry_name,
    )
    attributes = joined.drop(columns=[geometry_name])
    return attributes, geometry_result, geometry_name, crs, column_order, distances


def _materialize_relation_join(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    distances,
    *,
    how: str,
    lsuffix: str,
    rsuffix: str,
    on_attribute=None,
) -> tuple[GeoDataFrame, np.ndarray | None]:
    """Explicit GeoPandas export boundary for relation-style join results."""
    attributes, geometry, geometry_name, crs, column_order, distances = _materialize_relation_join_parts(
        left_df,
        right_df,
        indices,
        distances,
        how=how,
        lsuffix=lsuffix,
        rsuffix=rsuffix,
        on_attribute=on_attribute,
    )
    joined = _materialize_attribute_geometry_frame(
        attributes,
        (NativeGeometryColumn(geometry_name, geometry),),
        geometry_name=geometry_name,
        column_order=column_order,
    )
    return joined, distances


@dataclass(frozen=True)
class RelationJoinResult:
    """Join result that stays as native relation indices until explicit export."""

    relation: RelationIndexResult
    distances: Any | None = None

    def to_host_distances(self) -> np.ndarray | None:
        if self.distances is None:
            return None
        return _host_array(self.distances, dtype=np.float64)

    def materialize(
        self,
        left_df: GeoDataFrame,
        right_df: GeoDataFrame,
        *,
        how: str,
        lsuffix: str,
        rsuffix: str,
        on_attribute=None,
    ) -> tuple[GeoDataFrame, np.ndarray | None]:
        return _materialize_relation_join(
            left_df,
            right_df,
            self.relation.to_host(),
            self.to_host_distances(),
            how=how,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            on_attribute=on_attribute,
        )

    def to_geodataframe(
        self,
        left_df: GeoDataFrame,
        right_df: GeoDataFrame,
        *,
        how: str,
        lsuffix: str,
        rsuffix: str,
        on_attribute=None,
        distance_col: str | None = None,
    ) -> GeoDataFrame:
        return _relation_join_export_result_to_native_tabular_result(
            RelationJoinExportResult(
                relation_result=self,
                left_df=left_df,
                right_df=right_df,
                how=how,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                on_attribute=on_attribute,
                distance_col=distance_col,
            ),
            attribute_storage="pandas",
        ).to_geodataframe()


@dataclass(frozen=True)
class RelationJoinExportResult:
    """Deferred GeoPandas export wrapper for relation joins."""

    relation_result: RelationJoinResult
    left_df: GeoDataFrame
    right_df: GeoDataFrame
    how: str
    lsuffix: str
    rsuffix: str
    on_attribute: list | None = None
    distance_col: str | None = None

    def materialize(self) -> tuple[GeoDataFrame, np.ndarray | None]:
        return self.relation_result.materialize(
            self.left_df,
            self.right_df,
            how=self.how,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            on_attribute=self.on_attribute,
        )

    def to_geodataframe(self) -> GeoDataFrame:
        return _relation_join_export_result_to_native_tabular_result(
            self,
            attribute_storage="pandas",
        ).to_geodataframe()


@dataclass(frozen=True)
class LeftConstructiveResult:
    """Constructive result that preserves a subset of left-hand rows."""

    geometry: GeometryNativeResult
    row_positions: np.ndarray

    def to_geodataframe(self, df: GeoDataFrame) -> GeoDataFrame:
        attrs = df.iloc[np.asarray(self.row_positions, dtype=np.intp)].copy()
        geom_name = attrs._geometry_column_name
        geom = self.geometry.to_geoseries(index=attrs.index, name=geom_name)
        return _replace_geometry_column_preserving_backing(
            attrs,
            geom.values,
            crs=self.geometry.crs,
        )

    def to_geoseries(self, series: GeoSeries) -> GeoSeries:
        row_positions = np.asarray(self.row_positions, dtype=np.intp)
        index = series.index.take(row_positions)
        return self.geometry.to_geoseries(
            index=index,
            name=getattr(series, "name", None),
        )

    def materialize(self, source):
        from vibespatial.api.geodataframe import GeoDataFrame
        from vibespatial.api.geoseries import GeoSeries

        if isinstance(source, GeoDataFrame):
            return self.to_geodataframe(source)
        if isinstance(source, GeoSeries):
            return self.to_geoseries(source)
        raise TypeError(
            "LeftConstructiveResult.materialize() expects GeoDataFrame or GeoSeries"
        )


def _coerce_constructive_export_frame(
    frame,
    *,
    geometry_name: str,
    frame_type: type | None,
):
    from vibespatial.api.geodataframe import GeoDataFrame

    frame = _set_active_geometry_name(frame, geometry_name)
    target_type = frame_type or type(frame)
    if not isinstance(frame, target_type):
        frame_attrs = frame.attrs.copy()
        frame = target_type(frame, geometry=geometry_name, crs=frame.crs)
        if frame_attrs:
            frame.attrs.update(frame_attrs)
    elif isinstance(frame, GeoDataFrame) and frame._geometry_column_name != geometry_name:
        frame._geometry_column_name = geometry_name
    return frame


@dataclass(frozen=True)
class GroupedConstructiveResult:
    """Grouped constructive result that exports to GeoPandas only at the boundary."""

    geometry: GeometryNativeResult
    attributes: pd.DataFrame
    geometry_name: str
    as_index: bool
    frame_type: type | None = None

    def to_geodataframe(self) -> GeoDataFrame:
        from vibespatial.api.geodataframe import GeoDataFrame

        frame_type = self.frame_type or GeoDataFrame
        aggregated = _grouped_constructive_result_to_native_tabular_result(
            self,
        ).to_geodataframe()
        if not isinstance(aggregated, frame_type):
            aggregated.__class__ = frame_type
            aggregated._geometry_column_name = self.geometry_name
        return aggregated


def _spatial_to_native_tabular_result(spatial) -> NativeTabularResult:
    geometry_name = getattr(spatial, "_geometry_column_name", None)
    if geometry_name is not None:
        from vibespatial.api.geo_base import _is_geometry_like_dtype

        geometry_columns = [
            str(column)
            for column in spatial.columns[
                np.asarray(spatial.dtypes.map(_is_geometry_like_dtype), dtype=bool)
            ]
        ]
        if geometry_name not in geometry_columns:
            geometry_columns.append(geometry_name)

        def build_geometry_result(column_name: str) -> GeometryNativeResult:
            series = spatial[column_name]
            return GeometryNativeResult.from_values(
                series.values,
                crs=getattr(series, "crs", spatial.crs),
                index=spatial.index,
                name=column_name,
            )

        geometry = build_geometry_result(geometry_name)
        secondary_geometry = tuple(
            NativeGeometryColumn(column_name, build_geometry_result(column_name))
            for column_name in geometry_columns
            if column_name != geometry_name
        )
        return NativeTabularResult(
            attributes=spatial.drop(columns=geometry_columns).copy(deep=False),
            geometry=geometry,
            geometry_name=geometry_name,
            column_order=tuple(spatial.columns),
            attrs=spatial.attrs.copy() or None,
            secondary_geometry=secondary_geometry,
        )

    geometry_name = getattr(spatial, "name", None) or "geometry"
    geometry = GeometryNativeResult.from_values(
        spatial.values,
        crs=spatial.crs,
        index=spatial.index,
        name=geometry_name,
    )
    return NativeTabularResult(
        attributes=pd.DataFrame(index=spatial.index),
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
        attrs=spatial.attrs.copy() or None,
    )


def _geometry_native_result_to_native_tabular_result(
    result: GeometryNativeResult,
    *,
    geometry_name: str = "geometry",
) -> NativeTabularResult:
    return NativeTabularResult(
        attributes=pd.DataFrame(index=pd.RangeIndex(result.row_count)),
        geometry=result,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
    )


def _grouped_constructive_result_to_native_tabular_result(
    result: GroupedConstructiveResult,
) -> NativeTabularResult:
    return _grouped_constructive_to_native_tabular_result(
        geometry=result.geometry,
        attributes=result.attributes,
        geometry_name=result.geometry_name,
        as_index=result.as_index,
    )


def _grouped_constructive_to_native_tabular_result(
    *,
    geometry: GeometryNativeResult,
    attributes: pd.DataFrame,
    geometry_name: str,
    as_index: bool,
) -> NativeTabularResult:
    attributes = attributes.copy(deep=False)
    leading_columns: list[str] = []
    trailing_columns = list(attributes.columns)
    if not as_index:
        attributes = attributes.reset_index()
        leading_count = len(attributes.columns) - len(trailing_columns)
        leading_columns = list(attributes.columns[:leading_count])
        trailing_columns = list(attributes.columns[leading_count:])
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*leading_columns, geometry_name, *trailing_columns]),
    )


def _rename_native_tabular_result(
    result: NativeTabularResult,
    mapping: dict[str, str] | None,
    *,
    geometry_name: str | None = None,
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    renamed_attributes = (
        result.attributes.rename_columns(mapping)
        if mapping
        else result.attributes
    )
    renamed_geometry_name = geometry_name or (
        mapping.get(result.geometry_name, result.geometry_name)
        if mapping
        else result.geometry_name
    )
    renamed_secondary_geometry = _rename_native_geometry_columns(
        result.secondary_geometry,
        mapping,
    )
    if renamed_secondary_geometry:
        renamed_secondary_geometry = tuple(
            column
            for column in renamed_secondary_geometry
            if column.name != renamed_geometry_name
        )

    renamed_column_order = result.resolved_column_order
    if mapping:
        renamed_column_order = _rename_output_column_order(
            renamed_column_order,
            mapping,
        )
    if geometry_name is not None and geometry_name != result.geometry_name:
        renamed_column_order = tuple(
            geometry_name if column_name == result.geometry_name else column_name
            for column_name in renamed_column_order
        )
    known_columns = set(renamed_attributes.columns) | {renamed_geometry_name} | {
        column.name for column in renamed_secondary_geometry
    }
    filtered_column_order: list[str] = []
    for column_name in renamed_column_order:
        if column_name not in known_columns or column_name in filtered_column_order:
            continue
        filtered_column_order.append(column_name)
    if renamed_geometry_name not in filtered_column_order:
        filtered_column_order.append(renamed_geometry_name)

    merged_attrs = result.attrs.copy() if result.attrs else {}
    if attrs:
        merged_attrs.update(attrs)
    return NativeTabularResult(
        attributes=renamed_attributes,
        geometry=result.geometry,
        geometry_name=renamed_geometry_name,
        column_order=tuple(filtered_column_order),
        attrs=merged_attrs or None,
        secondary_geometry=renamed_secondary_geometry,
        provenance=result.provenance,
    )


def _pairwise_constructive_to_native_tabular_result(
    *,
    geometry: GeometryNativeResult,
    relation: RelationIndexResult,
    keep_geom_type_applied: bool,
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    geometry_name: str = "geometry",
    rename_columns: dict[str, str] | None = None,
    assign_columns: dict[str, Any] | None = None,
    frame_attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    left_idx, right_idx = relation.to_host()
    h_left_idx = left_idx.get() if hasattr(left_idx, "get") else left_idx
    h_right_idx = right_idx.get() if hasattr(right_idx, "get") else right_idx
    row_count = len(h_left_idx)
    out_index = pd.RangeIndex(row_count)

    left_attrs = left_df.drop(left_df._geometry_column_name, axis=1).copy(deep=False)
    right_attrs = right_df.drop(right_df._geometry_column_name, axis=1).copy(deep=False)
    left_columns, right_columns = _process_column_names_with_suffix(
        left_attrs.columns,
        right_attrs.columns,
        ("1", "2"),
        left_attrs,
        right_attrs,
    )
    left_attrs.columns = left_columns
    right_attrs.columns = right_columns

    left_projected = left_attrs._reindex_with_indexers({0: (out_index, h_left_idx)})
    right_projected = right_attrs._reindex_with_indexers({0: (out_index, h_right_idx)})
    attributes, secondary_geometry, column_order = _projected_frames_to_native_tabular_parts(
        [left_projected, right_projected],
        index_override=out_index,
        dropped_column_names={"geometry"},
    )
    result_attrs: dict[str, Any] = {}
    if keep_geom_type_applied:
        result_attrs["_vibespatial_keep_geom_type_applied"] = True
    if frame_attrs:
        result_attrs.update(frame_attrs)
    if rename_columns:
        attributes = attributes.rename_columns(rename_columns)
        secondary_geometry = _rename_native_geometry_columns(
            secondary_geometry,
            rename_columns,
        )
        column_order = _rename_output_column_order(
            column_order,
            rename_columns,
        )
    trailing_columns: list[str] = []
    if assign_columns:
        for column, values in assign_columns.items():
            attributes = attributes.with_column(column, values)
        trailing_columns = list(assign_columns)
    trailing_columns = [column for column in trailing_columns if column in attributes.columns]
    base_columns = [column for column in column_order if column not in trailing_columns]
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*base_columns, geometry_name, *trailing_columns]),
        attrs=result_attrs or None,
        secondary_geometry=secondary_geometry,
    )


def _pairwise_constructive_result_to_native_tabular_result(
    result: PairwiseConstructiveResult,
    *,
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    geometry_name: str = "geometry",
    rename_columns: dict[str, str] | None = None,
    assign_columns: dict[str, Any] | None = None,
    frame_attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    return _pairwise_constructive_to_native_tabular_result(
        geometry=result.geometry,
        relation=result.relation,
        keep_geom_type_applied=result.keep_geom_type_applied,
        left_df=left_df,
        right_df=right_df,
        geometry_name=geometry_name,
        rename_columns=rename_columns,
        assign_columns=assign_columns,
        frame_attrs=frame_attrs,
    )


def _left_constructive_to_native_tabular_result(
    *,
    geometry: GeometryNativeResult,
    row_positions: np.ndarray,
    df: GeoDataFrame,
    geometry_name: str = "geometry",
    rename_columns: dict[str, str] | None = None,
    assign_columns: dict[str, Any] | None = None,
    frame_attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    row_positions = np.asarray(row_positions, dtype=np.intp)
    source_attrs = df.drop(df._geometry_column_name, axis=1).copy(deep=False)
    projected = source_attrs._reindex_with_indexers(
        {0: (pd.RangeIndex(len(row_positions)), row_positions)}
    )
    attributes, secondary_geometry, _column_order = _projected_frames_to_native_tabular_parts(
        [projected],
        index_override=source_attrs.index.take(row_positions),
        dropped_column_names={geometry_name},
    )
    if rename_columns:
        attributes = attributes.rename_columns(rename_columns)
        secondary_geometry = _rename_native_geometry_columns(
            secondary_geometry,
            rename_columns,
        )
    trailing_columns: list[str] = []
    if assign_columns:
        for column, values in assign_columns.items():
            attributes = attributes.with_column(column, values)
        trailing_columns = list(assign_columns)
    trailing_columns = [column for column in trailing_columns if column in attributes.columns]
    base_columns = [
        column
        for column in _left_constructive_output_column_order(
            df=df,
            geometry_name=geometry_name,
            attributes=attributes,
            secondary_geometry=secondary_geometry,
        )
        if column not in trailing_columns
    ]
    result_attrs = frame_attrs
    if result_attrs is None and df.attrs:
        result_attrs = df.attrs.copy()
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*base_columns, *trailing_columns]),
        attrs=result_attrs or None,
        secondary_geometry=secondary_geometry,
    )


def _left_constructive_result_to_native_tabular_result(
    result: LeftConstructiveResult,
    *,
    df: GeoDataFrame,
    geometry_name: str = "geometry",
    rename_columns: dict[str, str] | None = None,
    assign_columns: dict[str, Any] | None = None,
    frame_attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    return _left_constructive_to_native_tabular_result(
        geometry=result.geometry,
        row_positions=result.row_positions,
        df=df,
        geometry_name=geometry_name,
        rename_columns=rename_columns,
        assign_columns=assign_columns,
        frame_attrs=frame_attrs,
    )


def _pairwise_constructive_fragment_to_native_tabular_result(fragment) -> NativeTabularResult:
    """Compatibility shim for legacy pairwise fragment lowering."""
    return _pairwise_constructive_to_native_tabular_result(
        geometry=fragment.result.geometry,
        relation=fragment.result.relation,
        keep_geom_type_applied=fragment.result.keep_geom_type_applied,
        left_df=fragment.left_df,
        right_df=fragment.right_df,
        geometry_name=getattr(fragment, "geometry_name", "geometry"),
        rename_columns=getattr(fragment, "rename_columns", None),
        assign_columns=getattr(fragment, "assign_columns", None),
        frame_attrs=None,
    )


def _left_constructive_fragment_to_native_tabular_result(fragment) -> NativeTabularResult:
    """Compatibility shim for legacy left-row fragment lowering."""
    frame_attrs = fragment.df.attrs.copy() if fragment.df.attrs else None
    return _left_constructive_to_native_tabular_result(
        geometry=fragment.result.geometry,
        row_positions=fragment.result.row_positions,
        df=fragment.df,
        geometry_name=getattr(fragment, "geometry_name", "geometry"),
        rename_columns=getattr(fragment, "rename_columns", None),
        assign_columns=getattr(fragment, "assign_columns", None),
        frame_attrs=frame_attrs,
    )


def _reorder_concat_positions(
    concat_row_positions: np.ndarray,
    ordered_row_positions: np.ndarray,
) -> np.ndarray:
    if concat_row_positions.size <= 1:
        return np.arange(concat_row_positions.size, dtype=np.intp)
    sorter = np.argsort(concat_row_positions, kind="stable")
    return sorter[
        np.searchsorted(
            concat_row_positions[sorter],
            ordered_row_positions,
        )
    ].astype(np.intp, copy=False)


def _assemble_indexed_owned_parts(index_oga_pairs, row_count: int):
    from vibespatial.geometry.owned import OwnedGeometryArray

    if not index_oga_pairs:
        return None
    if len(index_oga_pairs) == 1:
        indices, oga = index_oga_pairs[0]
        if len(indices) == row_count and np.array_equal(indices, np.arange(row_count)):
            return oga

    all_indices = np.concatenate([idx for idx, _ in index_oga_pairs]).astype(np.intp, copy=False)
    concat_result = OwnedGeometryArray.concat([oga for _, oga in index_oga_pairs])
    inverse_perm = np.empty(row_count, dtype=np.intp)
    inverse_perm[all_indices] = np.arange(len(all_indices), dtype=np.intp)
    return concat_result.take(inverse_perm)


def _clip_owned_geometry_native_result(result, *, crs):
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.geometry.owned import FAMILY_TAGS, GeometryFamily, OwnedGeometryArray
    from vibespatial.runtime.residency import Residency

    if not result.parts or not all(part.geometry.owned is not None for part in result.parts):
        return None, None

    row_parts = [np.asarray(part.row_positions, dtype=np.intp) for part in result.parts]
    owned_parts = [part.geometry.owned for part in result.parts]
    row_positions = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.intp)
    combined_owned = (
        owned_parts[0]
        if len(owned_parts) == 1
        else OwnedGeometryArray.concat(owned_parts)
    )
    initial_residency = combined_owned.residency

    if row_positions.size > 1:
        reorder = _reorder_concat_positions(
            row_positions,
            np.asarray(result.ordered_row_positions, dtype=np.intp),
        )
        row_positions = row_positions[reorder]
        combined_owned = combined_owned.take(reorder)

    geometry_array = GeometryArray.from_owned(combined_owned, crs=crs)
    keep = np.asarray(combined_owned.validity, dtype=bool) & ~np.asarray(
        geometry_array.is_empty,
        dtype=bool,
    )
    if not keep.all():
        keep_rows = np.flatnonzero(keep).astype(np.intp, copy=False)
        row_positions = row_positions[keep_rows]
        combined_owned = combined_owned.take(keep_rows)
        geometry_array = GeometryArray.from_owned(combined_owned, crs=crs)

    if (
        not result.clipping_by_rectangle
        and result.has_non_point_candidates
        and combined_owned.row_count > 0
    ):
        tags = np.asarray(combined_owned.tags, dtype=np.int8)
        polygon_mask = np.isin(
            tags,
            [
                FAMILY_TAGS[GeometryFamily.POLYGON],
                FAMILY_TAGS[GeometryFamily.MULTIPOLYGON],
            ],
        )
        if np.any(polygon_mask):
            keep = np.ones(combined_owned.row_count, dtype=bool)
            nonpositive_area = ~(
                np.asarray(
                    geometry_array.area[polygon_mask],
                    dtype=np.float64,
                ) > 0.0
            )
            polygon_bounds = np.asarray(
                geometry_array.bounds,
                dtype=np.float64,
            )[polygon_mask]
            pointlike_zero_area = (
                nonpositive_area
                & (np.abs(polygon_bounds[:, 2] - polygon_bounds[:, 0]) <= SPATIAL_EPSILON)
                & (np.abs(polygon_bounds[:, 3] - polygon_bounds[:, 1]) <= SPATIAL_EPSILON)
            )
            keep[polygon_mask] = ~nonpositive_area | pointlike_zero_area
            if not keep.all():
                keep_rows = np.flatnonzero(keep).astype(np.intp, copy=False)
                row_positions = row_positions[keep_rows]
                combined_owned = combined_owned.take(keep_rows)
                geometry_array = GeometryArray.from_owned(combined_owned, crs=crs)
                tags = np.asarray(combined_owned.tags, dtype=np.int8)

        line_mask = np.isin(
            tags,
            [
                FAMILY_TAGS[GeometryFamily.LINESTRING],
                FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
            ],
        )
        if np.any(line_mask):
            degenerate = line_mask & (
                np.asarray(geometry_array.length, dtype=np.float64) == 0.0
            )
            if np.any(degenerate):
                degenerate_rows = np.flatnonzero(degenerate).astype(np.intp, copy=False)
                repaired = geometry_array.take(degenerate_rows).make_valid()
                repaired_owned = getattr(repaired, "_owned", None)
                if repaired_owned is None or repaired_owned.row_count != degenerate_rows.size:
                    return None, None
                keep_rows = np.flatnonzero(~degenerate).astype(np.intp, copy=False)
                index_oga_pairs = []
                if keep_rows.size:
                    index_oga_pairs.append((keep_rows, combined_owned.take(keep_rows)))
                index_oga_pairs.append((degenerate_rows, repaired_owned))
                combined_owned = _assemble_indexed_owned_parts(
                    index_oga_pairs,
                    combined_owned.row_count,
                )

    if initial_residency is Residency.DEVICE and combined_owned.residency is not Residency.DEVICE:
        from vibespatial.runtime.residency import TransferTrigger

        combined_owned = combined_owned.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason=(
                "clip native-tabular boundary restored device-backed output after "
                "geometry analysis and cleanup"
            ),
        )

    return GeometryNativeResult.from_owned(combined_owned, crs=crs), row_positions


def _clip_constructive_parts_to_native_tabular_result(
    *,
    source,
    parts: tuple[LeftConstructiveResult, ...],
    ordered_row_positions: np.ndarray,
    clipping_by_rectangle: bool,
    has_non_point_candidates: bool,
    keep_geom_type: bool,
    spatial_materializer: Callable[[], Any] | None = None,
) -> NativeTabularResult:
    if not parts:
        if hasattr(source, "_geometry_column_name"):
            geometry_name = source._geometry_column_name
            attributes = NativeAttributeTable(
                dataframe=source.drop(columns=[geometry_name]).iloc[:0].copy(deep=False)
            )
        else:
            geometry_name = getattr(source, "name", None) or "geometry"
            attributes = NativeAttributeTable(dataframe=pd.DataFrame(index=source.iloc[:0].index))
        return NativeTabularResult(
            attributes=attributes,
            geometry=_empty_geometry_native_result(geometry_name=geometry_name, crs=source.crs),
            geometry_name=geometry_name,
            column_order=_clip_source_column_order(
                source,
                geometry_name=geometry_name,
                attributes=attributes,
            ),
            attrs=source.attrs.copy() or None,
        )

    if len(parts) == 1 and not has_non_point_candidates and not keep_geom_type:
        part = parts[0]
        row_positions = np.asarray(part.row_positions, dtype=np.intp)
        if hasattr(source, "_geometry_column_name"):
            geometry_name = source._geometry_column_name
            source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
            projected = source_attrs.iloc[row_positions].copy(deep=False)
            attributes = _native_attribute_table_from_projected_frames(
                [projected],
                index_override=source_attrs.index.take(row_positions),
            )
        else:
            geometry_name = getattr(source, "name", None) or "geometry"
            attributes = NativeAttributeTable(
                dataframe=pd.DataFrame(index=source.index.take(row_positions))
            )
        return NativeTabularResult(
            attributes=attributes,
            geometry=part.geometry,
            geometry_name=geometry_name,
            column_order=_clip_source_column_order(
                source,
                geometry_name=geometry_name,
                attributes=attributes,
            ),
            attrs=source.attrs.copy() or None,
        )

    if not keep_geom_type:
        geometry_name = (
            source._geometry_column_name
            if hasattr(source, "_geometry_column_name")
            else getattr(source, "name", None) or "geometry"
        )
        owned_geometry, owned_rows = _clip_owned_geometry_native_result(
            SimpleNamespace(
                source=source,
                parts=parts,
                ordered_row_positions=ordered_row_positions,
                clipping_by_rectangle=clipping_by_rectangle,
                has_non_point_candidates=has_non_point_candidates,
                keep_geom_type=keep_geom_type,
            ),
            crs=source.crs,
        )
        if owned_geometry is not None and owned_rows is not None:
            if hasattr(source, "_geometry_column_name"):
                source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
                projected = source_attrs.iloc[owned_rows].copy(deep=False)
                attributes = _native_attribute_table_from_projected_frames(
                    [projected],
                    index_override=source.index.take(owned_rows),
                )
            else:
                attributes = NativeAttributeTable(
                    dataframe=pd.DataFrame(index=source.index.take(owned_rows))
                )
            return NativeTabularResult(
                attributes=attributes,
                geometry=owned_geometry,
                geometry_name=geometry_name,
                column_order=_clip_source_column_order(
                    source,
                    geometry_name=geometry_name,
                    attributes=attributes,
                ),
                attrs=source.attrs.copy() or None,
            )

        if any(part.geometry.owned is not None for part in parts):
            record_fallback_event(
                surface="geopandas.clip",
                reason="clip native-tabular export requires host semantic cleanup before materialization",
                detail=(
                    "owned clip rows could not be preserved in the native boundary, so "
                    "host shapely/object cleanup is the explicit export fallback"
                ),
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                pipeline="clip.to_native_tabular_result",
                d2h_transfer=True,
            )
        row_parts: list[np.ndarray] = []
        geometry_parts: list[np.ndarray] = []
        for part in parts:
            row_positions = np.asarray(part.row_positions, dtype=np.intp)
            row_parts.append(row_positions)
            if part.geometry.owned is not None:
                geometry_parts.append(np.asarray(part.geometry.owned.to_shapely(), dtype=object))
            else:
                geometry_parts.append(np.asarray(part.geometry.series, dtype=object))

        row_positions = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.intp)
        geometry_values = (
            np.concatenate(geometry_parts).astype(object, copy=False)
            if geometry_parts
            else np.empty(0, dtype=object)
        )
        if row_positions.size > 1:
            ordered_rows = np.asarray(ordered_row_positions, dtype=np.intp)
            sorter = np.argsort(row_positions, kind="stable")
            reorder = sorter[np.searchsorted(row_positions[sorter], ordered_rows)]
            row_positions = row_positions[reorder]
            geometry_values = geometry_values[reorder]

        if geometry_values.size > 0:
            keep = ~(
                shapely.is_missing(geometry_values)
                | shapely.is_empty(geometry_values)
            )
            row_positions = row_positions[keep]
            geometry_values = geometry_values[keep]

        if not clipping_by_rectangle and has_non_point_candidates and geometry_values.size > 0:
            type_ids = np.asarray(shapely.get_type_id(geometry_values), dtype=np.int32)
            polygon_mask = (type_ids == 3) | (type_ids == 6)
            if np.any(polygon_mask):
                nonpositive_area = np.asarray(
                    shapely.area(geometry_values[polygon_mask]),
                    dtype=np.float64,
                ) <= 0.0
                if np.any(nonpositive_area):
                    polygon_bounds = np.asarray(
                        shapely.bounds(geometry_values[polygon_mask]),
                        dtype=np.float64,
                    ).reshape(-1, 4)
                    pointlike_zero_area = (
                        nonpositive_area
                        & (np.abs(polygon_bounds[:, 2] - polygon_bounds[:, 0]) <= SPATIAL_EPSILON)
                        & (np.abs(polygon_bounds[:, 3] - polygon_bounds[:, 1]) <= SPATIAL_EPSILON)
                    )
                    keep = np.ones(len(geometry_values), dtype=bool)
                    keep[np.flatnonzero(polygon_mask)[nonpositive_area & ~pointlike_zero_area]] = False
                    row_positions = row_positions[keep]
                    geometry_values = geometry_values[keep]
                    type_ids = type_ids[keep]

            line_mask = (type_ids == 1) | (type_ids == 2) | (type_ids == 5)
            if np.any(line_mask):
                degenerate_lines = np.asarray(
                    shapely.length(geometry_values[line_mask]),
                    dtype=np.float64,
                ) == 0.0
                if np.any(degenerate_lines):
                    repaired_values = geometry_values.copy()
                    line_rows = np.flatnonzero(line_mask)[degenerate_lines]
                    repaired_values[line_rows] = shapely.make_valid(
                        geometry_values[line_mask][degenerate_lines]
                    )
                    geometry_values = repaired_values

        if hasattr(source, "_geometry_column_name"):
            geometry_name = source._geometry_column_name
            source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
            projected = source_attrs.iloc[row_positions].copy(deep=False)
            attributes = _native_attribute_table_from_projected_frames(
                [projected],
                index_override=source.index.take(row_positions),
            )
        else:
            geometry_name = getattr(source, "name", None) or "geometry"
            attributes = NativeAttributeTable(
                dataframe=pd.DataFrame(index=source.index.take(row_positions))
            )

        geometry = (
            _empty_geometry_native_result(geometry_name=geometry_name, crs=source.crs)
            if len(row_positions) == 0
            else GeometryNativeResult.from_values(
                geometry_values,
                crs=source.crs,
                index=source.index.take(row_positions),
                name=geometry_name,
            )
        )
        return NativeTabularResult(
            attributes=attributes,
            geometry=geometry,
            geometry_name=geometry_name,
            column_order=_clip_source_column_order(
                source,
                geometry_name=geometry_name,
                attributes=attributes,
            ),
            attrs=source.attrs.copy() or None,
        )

    if spatial_materializer is None:
        raise ValueError(
            "clip native-tabular export requires a spatial materializer when keep_geom_type cleanup is needed"
        )
    clipped = spatial_materializer()
    return _spatial_to_native_tabular_result(clipped)


def _clip_native_result_to_native_tabular_result(result) -> NativeTabularResult:
    return _clip_constructive_parts_to_native_tabular_result(
        source=result.source,
        parts=result.parts,
        ordered_row_positions=result.ordered_row_positions,
        clipping_by_rectangle=result.clipping_by_rectangle,
        has_non_point_candidates=result.has_non_point_candidates,
        keep_geom_type=result.keep_geom_type,
        spatial_materializer=result.to_spatial,
    )


def _relation_join_export_result_to_native_tabular_result(
    result: RelationJoinExportResult,
    *,
    attribute_storage: str = "arrow",
) -> NativeTabularResult:
    attributes, geometry, geometry_name, _crs, column_order, secondary_geometry, distances = _native_relation_join_parts(
        result.left_df,
        result.right_df,
        result.relation_result.relation.to_host(),
        result.relation_result.to_host_distances(),
        how=result.how,
        lsuffix=result.lsuffix,
        rsuffix=result.rsuffix,
        on_attribute=result.on_attribute,
        attribute_storage=attribute_storage,
    )
    if result.distance_col is not None:
        attributes = attributes.with_column(result.distance_col, distances)
        column_order = tuple([*column_order, result.distance_col])
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=column_order,
        attrs=result.left_df.attrs.copy() or None,
        secondary_geometry=secondary_geometry,
    )


def _empty_geometry_native_result(*, geometry_name: str, crs) -> GeometryNativeResult:
    from vibespatial.api.geoseries import GeoSeries

    return GeometryNativeResult.from_geoseries(
        GeoSeries([], index=pd.RangeIndex(0), crs=crs, name=geometry_name),
    )


def _clip_source_column_order(
    source,
    *,
    geometry_name: str,
    attributes: NativeAttributeTable,
) -> tuple[Any, ...]:
    if not hasattr(source, "columns"):
        return (geometry_name,)
    return tuple(source.columns)


def _concat_geometry_result_sequence(
    geometries: list[GeometryNativeResult],
    *,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    if not geometries:
        return _empty_geometry_native_result(geometry_name=geometry_name, crs=crs)

    if all(geometry.owned is not None for geometry in geometries):
        owned_arrays = [geometry.owned for geometry in geometries]
        same_residency = len({owned.residency for owned in owned_arrays}) == 1
        if same_residency:
            try:
                from vibespatial.geometry.owned import concatenate_owned_arrays

                return GeometryNativeResult.from_owned(
                    concatenate_owned_arrays(owned_arrays),
                    crs=crs,
                )
            except Exception:
                pass

    series_parts = [
        geometry.to_geoseries(
            index=pd.RangeIndex(geometry.row_count),
            name=geometry_name,
        )
        for geometry in geometries
    ]
    if crs is not None:
        series_parts = [
            series.set_crs(crs, allow_override=True)
            if series.crs != crs
            else series
            for series in series_parts
        ]
    combined = pd.concat(series_parts, ignore_index=True)
    return GeometryNativeResult.from_values(
        combined.values,
        crs=crs,
        index=combined.index,
        name=geometry_name,
    )


def _concat_geometry_native_results(
    results: list[NativeTabularResult],
    *,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    return _concat_geometry_result_sequence(
        [result.geometry for result in results],
        geometry_name=geometry_name,
        crs=crs,
    )


def _concat_native_tabular_results(
    results: list[NativeTabularResult],
    *,
    geometry_name: str,
    crs,
    attrs: dict[str, Any] | None = None,
    provenance: NativeReadProvenance | None = None,
) -> NativeTabularResult:
    if not results:
        return NativeTabularResult(
            attributes=pd.DataFrame(index=pd.RangeIndex(0)),
            geometry=_empty_geometry_native_result(geometry_name=geometry_name, crs=crs),
            geometry_name=geometry_name,
            column_order=(geometry_name,),
            attrs=attrs,
            provenance=provenance,
        )

    merged_attrs: dict[str, Any] = {}
    for result in results:
        if result.attrs:
            merged_attrs.update(result.attrs)
    if attrs:
        merged_attrs.update(attrs)

    attributes = NativeAttributeTable.concat(
        [result.attributes for result in results],
        ignore_index=True,
        sort=False,
    )
    geometry = _concat_geometry_native_results(
        results,
        geometry_name=geometry_name,
        crs=crs,
    )
    secondary_names = [column.name for column in results[0].secondary_geometry]
    for result in results[1:]:
        if [column.name for column in result.secondary_geometry] != secondary_names:
            raise ValueError("cannot concatenate native tabular results with mismatched secondary geometry columns")
    secondary_geometry = tuple(
        NativeGeometryColumn(
            column_name,
            _concat_geometry_result_sequence(
                [
                    next(column.geometry for column in result.secondary_geometry if column.name == column_name)
                    for result in results
                ],
                geometry_name=column_name,
                crs=next(
                    column.geometry.crs
                    for column in results[0].secondary_geometry
                    if column.name == column_name
                ),
            ),
        )
        for column_name in secondary_names
    )
    column_order: list[str] = []
    for result in results:
        for column_name in result.resolved_column_order:
            if column_name not in column_order:
                column_order.append(column_name)
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple(column_order),
        attrs=merged_attrs or None,
        secondary_geometry=secondary_geometry,
        provenance=provenance,
    )


def _symmetric_difference_native_tabular_results(
    left_result: NativeTabularResult,
    right_result: NativeTabularResult,
    *,
    geometry_name: str,
    crs,
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    shared = set(left_result.attributes.columns) & set(right_result.attributes.columns)
    rename1 = {column: f"{column}_1" for column in shared}
    rename2 = {column: f"{column}_2" for column in shared}
    if rename1:
        left_result = _rename_native_tabular_result(
            left_result,
            rename1,
            geometry_name=geometry_name,
        )
    elif left_result.geometry_name != geometry_name:
        left_result = _rename_native_tabular_result(
            left_result,
            None,
            geometry_name=geometry_name,
        )
    if rename2:
        right_result = _rename_native_tabular_result(
            right_result,
            rename2,
            geometry_name=geometry_name,
        )
    elif right_result.geometry_name != geometry_name:
        right_result = _rename_native_tabular_result(
            right_result,
            None,
            geometry_name=geometry_name,
        )

    merged = _concat_native_tabular_results(
        [left_result, right_result],
        geometry_name=geometry_name,
        crs=crs,
        attrs=attrs,
    )
    merged_column_order = tuple(
        [
            *[
                column_name
                for column_name in merged.resolved_column_order
                if column_name != geometry_name
            ],
            geometry_name,
        ]
    )
    return NativeTabularResult(
        attributes=merged.attributes,
        geometry=merged.geometry,
        geometry_name=geometry_name,
        column_order=merged_column_order,
        attrs=merged.attrs,
        secondary_geometry=merged.secondary_geometry,
        provenance=merged.provenance,
    )


def _symmetric_difference_constructive_result_to_native_tabular_result(result) -> NativeTabularResult:
    """Compatibility shim for legacy symmetric-difference wrapper lowering."""
    left_result = _left_constructive_result_to_native_tabular_result(
        result.left_result,
        df=result.left_df,
        geometry_name=result.geometry_name,
    )
    right_result = _left_constructive_result_to_native_tabular_result(
        result.right_result,
        df=result.right_df,
        geometry_name=result.geometry_name,
    )
    return _symmetric_difference_native_tabular_results(
        left_result,
        right_result,
        geometry_name=result.geometry_name,
        crs=result.crs if result.crs is not None else left_result.geometry.crs,
    )


def _concat_constructive_result_to_native_tabular_result(result) -> NativeTabularResult:
    """Compatibility shim for legacy concat wrapper lowering."""
    results = []
    for part in result.parts:
        payload = to_native_tabular_result(part)
        if payload is None:
            payload = _spatial_to_native_tabular_result(part.to_geodataframe())
        results.append(payload)
    return _concat_native_tabular_results(
        results,
        geometry_name=result.geometry_name,
        crs=result.crs if result.crs is not None else (results[0].geometry.crs if results else None),
        attrs=result.result_attrs,
    )


def to_native_tabular_result(result) -> NativeTabularResult | None:
    """Convert supported native result families into the shared tabular boundary."""
    from vibespatial.api.tools.clip import ClipNativeResult

    if isinstance(result, NativeTabularResult):
        return result
    if isinstance(result, GeometryNativeResult):
        return _geometry_native_result_to_native_tabular_result(result)
    if isinstance(result, GroupedConstructiveResult):
        return _grouped_constructive_result_to_native_tabular_result(result)
    if isinstance(result, ClipNativeResult):
        return _clip_native_result_to_native_tabular_result(result)
    if isinstance(result, RelationJoinExportResult):
        return _relation_join_export_result_to_native_tabular_result(result)
    return None
