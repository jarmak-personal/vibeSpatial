from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
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
    NativeGeometryProvenance,
    NativeReadProvenance,
    NativeTabularResult,
    _host_array,
    _is_admissible_pandas_numeric_series,
    _materialize_attribute_geometry_frame,
    _replace_geometry_column_preserving_backing,
    _set_active_geometry_name,
    native_attribute_table_from_arrow_table,  # noqa: F401
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.config import SPATIAL_EPSILON
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency, combined_residency

if TYPE_CHECKING:
    from vibespatial.api.geodataframe import GeoDataFrame
    from vibespatial.api.geoseries import GeoSeries


_PUBLIC_SJOIN_PANDAS_EXPORT_MAX_ROWS = 50_000


def _index_array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    size = getattr(values, "size", None)
    if size is not None:
        return int(size)
    return len(values)


def _cached_geometry_metadata(geometry: GeometryNativeResult):
    owned = getattr(geometry, "owned", None)
    if owned is None:
        return None
    from vibespatial.api._native_metadata import NativeGeometryMetadata

    return NativeGeometryMetadata.from_cached_owned(owned)


def _index_host_detail(values: Any, *, side: str) -> str:
    try:
        size = _index_array_size(values)
    except Exception:
        return f"side={side}"
    itemsize = int(getattr(getattr(values, "dtype", None), "itemsize", 0))
    byte_detail = f", bytes={size * itemsize}" if itemsize else ""
    return f"side={side}, rows={size}{byte_detail}"


def _pairwise_index_to_host(
    values: Any,
    *,
    side: str,
    surface: str,
    operation: str,
    reason: str,
    dtype=np.intp,
    strict_disallowed: bool = False,
) -> np.ndarray:
    return _host_array(
        values,
        dtype=dtype,
        strict_disallowed=strict_disallowed,
        surface=surface,
        operation=operation,
        reason=reason,
        detail=_index_host_detail(values, side=side),
    )


@dataclass(frozen=True)
class RelationIndexResult:
    """Low-level relation result carrying row-pair indices."""

    left_indices: Any
    right_indices: Any

    def to_host(
        self,
        *,
        surface: str = "vibespatial.api.RelationIndexResult.to_host",
        operation: str = "relation_indices_to_host",
        reason: str = "relation pair indices crossed to host at an explicit compatibility boundary",
        dtype=np.int32,
        strict_disallowed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            _pairwise_index_to_host(
                self.left_indices,
                side="left",
                surface=surface,
                operation=operation,
                reason=reason,
                dtype=dtype,
                strict_disallowed=strict_disallowed,
            ),
            _pairwise_index_to_host(
                self.right_indices,
                side="right",
                surface=surface,
                operation=operation,
                reason=reason,
                dtype=dtype,
                strict_disallowed=strict_disallowed,
            ),
        )

    @property
    def size(self) -> int:
        left_size = _index_array_size(self.left_indices)
        right_size = _index_array_size(self.right_indices)
        if left_size != right_size:
            raise ValueError(
                f"RelationIndexResult pair length mismatch: {left_size} != {right_size}"
            )
        return left_size

    def __len__(self) -> int:
        return self.size


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

        left_idx, right_idx = self.relation.to_host(
            surface="vibespatial.api.PairwiseConstructiveResult.to_geodataframe",
            operation="pairwise_constructive_relation_indices_to_host",
            reason="public pairwise constructive export needs pandas-compatible row indexers",
            dtype=np.intp,
        )
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
    from vibespatial.api.geoseries import GeoSeries

    left_column = left_geometry.iloc[:, 0]
    right_column = right_geometry.iloc[:, 0]
    left_values = left_column.values.take(
        np.asarray(l_idx, dtype=np.intp),
        allow_fill=True,
        fill_value=None,
    )
    right_values = right_column.values.take(
        np.asarray(r_idx, dtype=np.intp),
        allow_fill=True,
        fill_value=None,
    )
    left_objects = np.asarray(left_values, dtype=object)
    right_objects = np.asarray(right_values, dtype=object)
    assembled = left_objects.copy()
    missing_left = pd.isna(assembled)
    assembled[missing_left] = right_objects[missing_left]
    return GeoSeries(
        assembled,
        index=new_index,
        crs=getattr(left_column, "crs", None),
        name=left_column.name,
    )
def _drop_private_attribute_columns(attributes: pd.DataFrame) -> pd.DataFrame:
    """Strip internal provenance columns from the shared export boundary."""
    private_columns = [
        column for column in ("__idx1", "__idx2") if column in attributes.columns
    ]
    if not private_columns:
        return attributes
    return attributes.drop(columns=private_columns)


def _same_index_with_names(left: pd.Index, right: pd.Index) -> bool:
    return bool(left.equals(right) and list(left.names) == list(right.names))


def _reduction_source_column_map(columns) -> dict[Any, Any]:
    if isinstance(columns, Mapping):
        return dict(columns)
    return {column: column for column in columns}


def _mapped_native_state_numeric_columns(
    state,
    columns,
) -> dict[Any, Any] | None:
    column_map = _reduction_source_column_map(columns)
    table = NativeAttributeTable.from_value(state.attributes)
    source_values = table.numeric_column_arrays(column_map.values())
    if source_values is None:
        return None
    return {
        output_name: source_values[source_name]
        for output_name, source_name in column_map.items()
    }


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
            if _same_index_with_names(frame.index, index_override):
                normalized_frames.append(frame)
                continue
            rebased = frame.copy(deep=False)
            rebased.index = index_override
            normalized_frames.append(rebased)
        concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
        frame = pd.concat(normalized_frames, axis=1, **concat_kwargs)
        return NativeAttributeTable(dataframe=frame)

    if storage not in {"arrow", "device"}:
        raise ValueError(f"Unsupported projected-frame storage: {storage!r}")

    import pyarrow as pa

    tables = []
    declared_names: list[Any] = []
    device_admissible = storage == "device"
    for frame in frames:
        if frame.shape[1] == 0:
            continue
        if device_admissible:
            for column in frame.columns:
                series = frame[column]
                if not isinstance(series, pd.Series) or not _is_admissible_pandas_numeric_series(
                    series,
                ):
                    device_admissible = False
                    break
        tables.append(pa.Table.from_pandas(frame, preserve_index=False))
        declared_names.extend(list(frame.columns))

    if not tables:
        return NativeAttributeTable(dataframe=pd.DataFrame(index=index_override))

    arrays = []
    names: list[str] = []
    for table in tables:
        arrays.extend(table.columns)
        names.extend(table.column_names)
    arrow_table = pa.Table.from_arrays(arrays, names=names)
    if (
        storage == "device"
        and device_admissible
        and len(set(declared_names)) == len(declared_names)
    ):
        try:
            import pylibcudf as plc
        except ModuleNotFoundError:
            pass
        else:
            return NativeAttributeTable(
                device_table=plc.Table.from_arrow(arrow_table),
                index_override=index_override,
                column_override=tuple(declared_names),
                schema_override=arrow_table.schema,
            )
    return NativeAttributeTable(
        arrow_table=arrow_table,
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
                if not _same_index_with_names(series.index, index_override):
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
        if not _same_index_with_names(attrs.index, index_override):
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
    h_left_idx = _pairwise_index_to_host(
        left_idx,
        side="left",
        surface="vibespatial.api._native_results._native_pairwise_attribute_table",
        operation="pairwise_attribute_indices_to_host",
        reason="native pairwise attribute projection still needs pandas-compatible row indexers",
    )
    h_right_idx = _pairwise_index_to_host(
        right_idx,
        side="right",
        surface="vibespatial.api._native_results._native_pairwise_attribute_table",
        operation="pairwise_attribute_indices_to_host",
        reason="native pairwise attribute projection still needs pandas-compatible row indexers",
    )
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
    h_left_idx = _pairwise_index_to_host(
        left_idx,
        side="left",
        surface="vibespatial.api._native_results._project_pairwise_attribute_frame",
        operation="public_pairwise_attribute_indices_to_host",
        reason="public pairwise attribute export needs pandas-compatible row indexers",
    )
    h_right_idx = _pairwise_index_to_host(
        right_idx,
        side="right",
        surface="vibespatial.api._native_results._project_pairwise_attribute_frame",
        operation="public_pairwise_attribute_indices_to_host",
        reason="public pairwise attribute export needs pandas-compatible row indexers",
    )
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


def _single_index_reset_column_name(
    df: GeoDataFrame,
    other_df: GeoDataFrame,
    *,
    suffix: str,
) -> Any | None:
    """Return the reset-index column name for the common single-index case."""
    if df.index.nlevels != 1:
        return None
    label = df.index.names[0]
    if label is not None:
        if label in df.columns:
            return None
        return label
    candidate = f"index_{suffix}0" if "index" in df.columns else f"index_{suffix}"
    if candidate in df.columns or candidate in other_df.columns:
        return None
    return candidate


def _has_secondary_geometry_columns(df: GeoDataFrame, geometry_name: str) -> bool:
    from vibespatial.api.geo_base import _is_geometry_like_dtype

    non_geometry = df.drop(geometry_name, axis=1)
    if non_geometry.shape[1] == 0:
        return False
    return bool(np.asarray(non_geometry.dtypes.map(_is_geometry_like_dtype), dtype=bool).any())


def _native_relation_join_parts_fast_pandas_inner(
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    indices: tuple[np.ndarray, np.ndarray],
    distances,
    *,
    lsuffix: str,
    rsuffix: str,
    on_attribute=None,
) -> tuple[
    NativeAttributeTable,
    GeometryNativeResult,
    str,
    Any,
    tuple[str, ...],
    tuple[NativeGeometryColumn, ...],
    np.ndarray | None,
] | None:
    """Fast public GeoDataFrame export for simple inner relation joins.

    Arrow/native sinks keep using the general native-tabular path.  This path
    only removes schema/reset-index work when the caller is already asking for a
    pandas-backed public GeoDataFrame.
    """
    if on_attribute:
        return None
    left_geometry_name = left_df.geometry.name
    right_geometry_name = right_df.geometry.name
    if _has_secondary_geometry_columns(left_df, left_geometry_name):
        return None
    if _has_secondary_geometry_columns(right_df, right_geometry_name):
        return None

    left_index_name = _single_index_reset_column_name(
        left_df,
        right_df,
        suffix=lsuffix,
    )
    right_index_name = _single_index_reset_column_name(
        right_df,
        left_df,
        suffix=rsuffix,
    )
    if left_index_name is None or right_index_name is None:
        return None

    l_idx, r_idx = indices
    l_take = np.asarray(l_idx, dtype=np.intp)
    r_take = np.asarray(r_idx, dtype=np.intp)

    left_attr_df = left_df.drop(left_geometry_name, axis=1)
    right_attr_df = right_df.drop(right_geometry_name, axis=1)
    left_reset_columns = pd.Index([left_index_name, *left_attr_df.columns])
    right_reset_columns = pd.Index([right_index_name, *right_attr_df.columns])
    left_columns, right_columns = _process_column_names_with_suffix(
        left_reset_columns,
        right_reset_columns,
        (lsuffix, rsuffix),
        left_df,
        right_df,
    )
    left_output_columns = list(left_columns[1:])
    right_output_columns = list(right_columns)
    joined_index = left_df.index.take(l_take)
    joined_index_name = None if left_df.index.names[0] is None else left_columns[0]
    joined_index = joined_index.rename(joined_index_name)

    frames: list[pd.DataFrame] = []
    if left_attr_df.shape[1] > 0:
        left_projected = left_attr_df.take(l_take)
        left_projected.index = joined_index
        left_projected.columns = left_output_columns
        frames.append(left_projected)

    right_index_values = right_df.index.take(r_take)
    frames.append(
        pd.DataFrame(
            {right_output_columns[0]: right_index_values},
            index=joined_index,
            copy=False,
        )
    )
    if right_attr_df.shape[1] > 0:
        right_projected = right_attr_df.take(r_take)
        right_projected.index = joined_index
        right_projected.columns = right_output_columns[1:]
        frames.append(right_projected)

    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
    attributes = pd.concat(frames, axis=1, **concat_kwargs)
    attributes.index = joined_index

    geometry_values = left_df.geometry.values.take(l_take)
    geometry_result = GeometryNativeResult.from_values(
        geometry_values,
        crs=left_df.crs,
        index=joined_index,
        name=left_geometry_name,
    )

    left_column_iter = iter(left_output_columns)
    column_order: list[Any] = []
    for column_name in left_df.columns:
        if column_name == left_geometry_name:
            column_order.append(left_geometry_name)
        else:
            column_order.append(next(left_column_iter))
    column_order.extend(right_output_columns)

    return (
        NativeAttributeTable(dataframe=attributes),
        geometry_result,
        left_geometry_name,
        left_df.crs,
        tuple(column_order),
        (),
        distances,
    )


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
    if how == "inner" and attribute_storage == "pandas":
        fast_parts = _native_relation_join_parts_fast_pandas_inner(
            left_df,
            right_df,
            indices,
            distances,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            on_attribute=on_attribute,
        )
        if fast_parts is not None:
            return fast_parts

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
        if direct_take or np.all(np.asarray(l_idx) >= 0):
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
        if direct_take or np.all(np.asarray(r_idx) >= 0):
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

    def __len__(self) -> int:
        return len(self.relation)

    def to_native_relation(
        self,
        *,
        left_token: str | None = None,
        right_token: str | None = None,
        predicate: str | None = None,
        left_row_count: int | None = None,
        right_row_count: int | None = None,
    ):
        from vibespatial.api._native_relation import NativeRelation

        return NativeRelation.from_relation_index_result(
            self.relation,
            left_token=left_token,
            right_token=right_token,
            predicate=predicate,
            distances=self.distances,
            left_row_count=left_row_count,
            right_row_count=right_row_count,
        )

    def to_host_distances(self) -> np.ndarray | None:
        if self.distances is None:
            return None
        return _host_array(
            self.distances,
            dtype=np.float64,
            strict_disallowed=False,
            surface="vibespatial.api.RelationJoinResult.to_host_distances",
            operation="relation_distances_to_host",
            reason="relation distance column crossed to host at an explicit compatibility boundary",
        )

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
            self.relation.to_host(
                surface="vibespatial.api.RelationJoinResult.materialize",
                operation="relation_join_indices_to_host",
                reason="public relation join materialization needs pandas-compatible row indexers",
                dtype=np.intp,
            ),
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


def _device_attribute_table_from_column(
    name: Any,
    values: Any,
    *,
    row_count: int,
    index_override: pd.Index,
) -> NativeAttributeTable | None:
    try:
        import cupy as cp
        import pyarrow as pa
        import pylibcudf as plc
    except ModuleNotFoundError:
        return None

    d_values = cp.asarray(values)
    if d_values.ndim != 1 or int(d_values.size) != int(row_count):
        return None
    dtype = np.dtype(d_values.dtype)
    if not (
        np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.bool_)
    ):
        return None
    column = plc.Column.from_cuda_array_interface(d_values)
    return NativeAttributeTable(
        device_table=plc.Table([column]),
        index_override=index_override,
        column_override=(name,),
        schema_override=pa.schema([pa.field(str(name), column.type().to_arrow())]),
    )


def _attribute_table_for_device_position_take(
    table: NativeAttributeTable,
) -> NativeAttributeTable | None:
    if len(table.columns) == 0:
        return table
    if table.device_table is not None or table.arrow_table is not None:
        return table
    if table.dataframe is None:
        return None
    try:
        return _native_attribute_table_from_projected_frames(
            [table.dataframe],
            index_override=table.index,
            storage="arrow",
        )
    except (ImportError, TypeError, ValueError):
        return None


def _relation_join_export_result_to_native_frame_state_device(
    result: RelationJoinExportResult,
    *,
    strict_index_materialization: bool = True,
):
    """Build an inner relation-join frame state without host pair export."""
    if result.how != "inner" or result.on_attribute:
        return None

    from vibespatial.api._native_state import NativeFrameState, get_native_state

    left_state = get_native_state(result.left_df)
    right_state = get_native_state(result.right_df)
    if left_state is None or right_state is None:
        return None

    left_geometry_name = result.left_df.geometry.name
    right_geometry_name = result.right_df.geometry.name
    if left_state.geometry_name != left_geometry_name:
        return None
    if right_state.geometry_name != right_geometry_name:
        return None
    if left_state.secondary_geometry or right_state.secondary_geometry:
        return None
    if _has_secondary_geometry_columns(result.left_df, left_geometry_name):
        return None
    if _has_secondary_geometry_columns(result.right_df, right_geometry_name):
        return None

    left_owned = getattr(left_state.geometry, "owned", None)
    if left_owned is None or getattr(left_owned, "residency", None) is not Residency.DEVICE:
        return None

    right_index_name = _single_index_reset_column_name(
        result.right_df,
        result.left_df,
        suffix=result.rsuffix,
    )
    left_index_name = _single_index_reset_column_name(
        result.left_df,
        result.right_df,
        suffix=result.lsuffix,
    )
    if left_index_name is None or right_index_name is None:
        return None

    left_attrs = NativeAttributeTable.from_value(left_state.attributes)
    right_attrs = NativeAttributeTable.from_value(right_state.attributes)
    left_attrs = _attribute_table_for_device_position_take(left_attrs)
    right_attrs = _attribute_table_for_device_position_take(right_attrs)
    if left_attrs is None or right_attrs is None:
        return None

    def _can_gather(table: NativeAttributeTable, row_positions) -> bool:
        if len(table.columns) == 0:
            return True
        if hasattr(row_positions, "__cuda_array_interface__"):
            return table.device_table is not None or table.arrow_table is not None
        return True

    def _can_gather_from_device_positions(table: NativeAttributeTable) -> bool:
        return (
            len(table.columns) == 0
            or table.device_table is not None
            or table.arrow_table is not None
        )

    if not _can_gather_from_device_positions(left_attrs) or not _can_gather_from_device_positions(
        right_attrs,
    ):
        return None

    relation = result.to_native_relation()
    pair_count = len(relation)
    out_index = pd.RangeIndex(pair_count)
    try:
        import cupy as cp
    except ModuleNotFoundError:
        return None
    left_indices = cp.asarray(relation.left_indices, dtype=cp.int64)
    right_indices = cp.asarray(relation.right_indices, dtype=cp.int64)

    if not _can_gather(left_attrs, left_indices) or not _can_gather(
        right_attrs,
        right_indices,
    ):
        return None

    left_reset_columns = pd.Index([left_index_name, *left_attrs.columns])
    right_reset_columns = pd.Index([right_index_name, *right_attrs.columns])
    left_columns, right_columns = _process_column_names_with_suffix(
        left_reset_columns,
        right_reset_columns,
        (result.lsuffix, result.rsuffix),
        result.left_df,
        result.right_df,
    )
    left_output_columns = tuple(left_columns[1:])
    right_output_columns = tuple(right_columns)
    left_mapping = {
        old: new
        for old, new in zip(left_attrs.columns, left_output_columns, strict=True)
        if old != new
    }
    right_mapping = {
        old: new
        for old, new in zip(right_attrs.columns, right_output_columns[1:], strict=True)
        if old != new
    }

    left_projected = left_attrs.take(
        left_indices,
        preserve_index=False,
    ).rename_columns(left_mapping)
    right_projected = right_attrs.take(
        right_indices,
        preserve_index=False,
    ).rename_columns(right_mapping)

    right_index_plan = right_state.index_plan.take(
        right_indices,
        preserve_index=True,
        unique=False,
        strict_disallowed=strict_index_materialization,
    )
    if right_index_plan.device_labels is not None:
        right_index_values = right_index_plan.device_labels
    elif right_index_plan.index is not None:
        right_index_values = right_index_plan.index.to_numpy(copy=False)
    else:
        return None
    right_index_attributes = _device_attribute_table_from_column(
        right_output_columns[0],
        right_index_values,
        row_count=pair_count,
        index_override=out_index,
    )
    if right_index_attributes is None:
        return None

    attributes = NativeAttributeTable.combine_columns(
        [left_projected, right_index_attributes, right_projected],
        index_override=out_index,
    )
    if attributes is None:
        return None

    geometry = left_state.geometry.take(left_indices)
    index_plan = left_state.index_plan.take(
        left_indices,
        preserve_index=True,
        unique=False,
        strict_disallowed=strict_index_materialization,
    )
    column_order: list[Any] = []
    left_column_iter = iter(left_output_columns)
    for column_name in result.left_df.columns:
        if column_name == left_geometry_name:
            column_order.append(left_geometry_name)
        else:
            column_order.append(next(left_column_iter))
    column_order.extend(right_output_columns)
    if result.distance_col is not None:
        if result.relation_result.distances is None:
            return None
        updated_attributes = attributes.assign_columns(
            {result.distance_col: result.relation_result.distances},
            columns=tuple([*attributes.columns, result.distance_col]),
        )
        if updated_attributes is None:
            return None
        attributes = updated_attributes
        column_order.append(result.distance_col)

    geometry_metadata_cache = (
        None
        if left_state.geometry_metadata_cache is None
        else left_state.geometry_metadata_cache.take(left_indices)
    )
    return NativeFrameState(
        attributes=attributes,
        geometry=geometry,
        geometry_name=left_geometry_name,
        column_order=tuple(column_order),
        index_plan=index_plan,
        row_count=pair_count,
        secondary_geometry=(),
        attrs=result.left_df.attrs.copy(),
        provenance=None,
        geometry_metadata_cache=geometry_metadata_cache,
        residency=combined_residency(getattr(geometry, "owned", None)),
        readiness=left_state.readiness,
    )


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
    predicate: str | None = None

    def materialize(self) -> tuple[GeoDataFrame, np.ndarray | None]:
        return self.relation_result.materialize(
            self.left_df,
            self.right_df,
            how=self.how,
            lsuffix=self.lsuffix,
            rsuffix=self.rsuffix,
            on_attribute=self.on_attribute,
        )

    def _prefer_pandas_public_export(self) -> bool:
        """Return whether public GeoDataFrame export should use pandas assembly.

        The device-native joined-frame route avoids host relation export, but
        public GeoDataFrame materialization still has to export the index and
        attributes.  For small and medium relation outputs, measured shootout
        costs are lower if we cross the public relation boundary directly and
        skip the device-table-to-Arrow public export.
        """
        try:
            pair_count = len(self.relation_result.relation)
        except Exception:
            return False
        return pair_count <= _PUBLIC_SJOIN_PANDAS_EXPORT_MAX_ROWS

    def to_geodataframe(self) -> GeoDataFrame:
        if not self._prefer_pandas_public_export():
            state = _relation_join_export_result_to_native_frame_state_device(
                self,
                strict_index_materialization=False,
            )
            if state is not None:
                return state.to_native_tabular_result().to_geodataframe()
        return _relation_join_export_result_to_native_tabular_result(
            self,
            attribute_storage="pandas",
        ).to_geodataframe()

    def to_native_tabular_result(
        self,
        *,
        attribute_storage: str = "arrow",
    ) -> NativeTabularResult:
        """Lower relation-join rows to the private native tabular boundary.

        Physical shape: relation-pair row projection.  Native input carrier is
        ``NativeRelation``/``RelationIndexResult`` plus the source frames; the
        output carrier is ``NativeTabularResult`` for sanctioned downstream
        native consumers.  This is not a public lazy join result.
        """
        return _relation_join_export_result_to_native_tabular_result(
            self,
            attribute_storage=attribute_storage,
        )

    def to_native_frame_state(
        self,
        *,
        attribute_storage: str = "arrow",
    ):
        """Return the private joined-frame carrier without GeoDataFrame export."""
        if attribute_storage == "device":
            state = _relation_join_export_result_to_native_frame_state_device(self)
            if state is not None:
                return state
        return self.to_native_tabular_result(
            attribute_storage=attribute_storage,
        ).to_native_frame_state()

    def to_native_relation(self):
        from vibespatial.api._native_state import get_native_state

        left_state = get_native_state(self.left_df)
        right_state = get_native_state(self.right_df)
        return self.relation_result.to_native_relation(
            left_token=(
                left_state.lineage_token
                if left_state is not None
                else f"gdf:{id(self.left_df)}"
            ),
            right_token=(
                right_state.lineage_token
                if right_state is not None
                else f"gdf:{id(self.right_df)}"
            ),
            predicate=self.predicate,
            left_row_count=(
                left_state.row_count if left_state is not None else len(self.left_df)
            ),
            right_row_count=(
                right_state.row_count if right_state is not None else len(self.right_df)
            ),
        )

    def left_semijoin_native_frame(
        self,
        *,
        order: str = "sorted",
        preserve_index: bool = False,
    ):
        """Return matched left rows as private native state when admissible."""
        from vibespatial.api._native_state import get_native_state

        left_state = get_native_state(self.left_df)
        if left_state is None:
            return None
        rowset = self.to_native_relation().left_semijoin_rowset(order=order)
        return left_state.take(rowset, preserve_index=preserve_index)

    def left_unique_label_semijoin_native_frame(self):
        """Return matched left rows for public unique-label semijoin shapes.

        This is the private admitted equivalent of:
        ``joined = sjoin(left, right); left.loc[joined.index.unique()]``.
        It deliberately declines duplicate-label and MultiIndex sources because
        public `.loc` expands duplicate labels and has level-specific semantics.
        """
        from vibespatial.api._native_state import get_native_state

        left_state = get_native_state(self.left_df)
        if left_state is None:
            return None
        if not left_state.index_plan.admits_unique_label_selection:
            return None
        rowset = self.to_native_relation().left_semijoin_rowset(order="first")
        return left_state.take(rowset, preserve_index=True)

    def left_antijoin_native_frame(
        self,
        *,
        preserve_index: bool = False,
    ):
        """Return unmatched left rows as private native state when admissible."""
        from vibespatial.api._native_state import get_native_state

        left_state = get_native_state(self.left_df)
        if left_state is None:
            return None
        rowset = self.to_native_relation().left_antijoin_rowset()
        return left_state.take(rowset, preserve_index=preserve_index)

    def right_semijoin_native_frame(
        self,
        *,
        order: str = "sorted",
        preserve_index: bool = False,
    ):
        """Return matched right rows as private native state when admissible."""
        from vibespatial.api._native_state import get_native_state

        right_state = get_native_state(self.right_df)
        if right_state is None:
            return None
        rowset = self.to_native_relation().right_semijoin_rowset(order=order)
        return right_state.take(rowset, preserve_index=preserve_index)

    def right_antijoin_native_frame(
        self,
        *,
        preserve_index: bool = False,
    ):
        """Return unmatched right rows as private native state when admissible."""
        from vibespatial.api._native_state import get_native_state

        right_state = get_native_state(self.right_df)
        if right_state is None:
            return None
        rowset = self.to_native_relation().right_antijoin_rowset()
        return right_state.take(rowset, preserve_index=preserve_index)

    def left_match_count_expression(self):
        """Return per-left-row relation match counts as a private expression."""
        return self.to_native_relation().left_match_count_expression()

    def right_match_count_expression(self):
        """Return per-right-row relation match counts as a private expression."""
        return self.to_native_relation().right_match_count_expression()

    def left_reduce_right_numeric_columns(
        self,
        right_columns,
        reducers,
        *,
        left_row_count: int | None = None,
    ):
        """Reduce attached right-source numeric attributes by left relation rows."""
        from vibespatial.api._native_state import get_native_state

        right_state = get_native_state(self.right_df)
        if right_state is None:
            return None
        mapped_columns = _mapped_native_state_numeric_columns(
            right_state,
            right_columns,
        )
        if mapped_columns is None:
            return None
        return self.to_native_relation().left_reduce_right_numeric_columns(
            mapped_columns,
            reducers,
            left_row_count=left_row_count,
        )

    def right_reduce_left_numeric_columns(
        self,
        left_columns,
        reducers,
        *,
        right_row_count: int | None = None,
    ):
        """Reduce attached left-source numeric attributes by right relation rows."""
        from vibespatial.api._native_state import get_native_state

        left_state = get_native_state(self.left_df)
        if left_state is None:
            return None
        mapped_columns = _mapped_native_state_numeric_columns(
            left_state,
            left_columns,
        )
        if mapped_columns is None:
            return None
        return self.to_native_relation().right_reduce_left_numeric_columns(
            mapped_columns,
            reducers,
            right_row_count=right_row_count,
        )


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
    attributes: NativeAttributeTable | pd.DataFrame
    geometry_name: str
    as_index: bool
    frame_type: type | None = None
    provenance: NativeGeometryProvenance | None = None

    def to_native_tabular_result(self) -> NativeTabularResult:
        """Lower grouped constructive output to the private native boundary.

        Physical shape: grouped geometry output rows plus grouped attributes.
        Native input carrier is the grouped constructive result; the output
        carrier is ``NativeTabularResult`` for sanctioned downstream native
        consumers before public export.
        """
        return _grouped_constructive_result_to_native_tabular_result(self)

    def to_native_frame_state(self):
        """Return the grouped constructive frame carrier without GeoDataFrame export."""
        return self.to_native_tabular_result().to_native_frame_state()

    def to_geodataframe(self) -> GeoDataFrame:
        from vibespatial.api.geodataframe import GeoDataFrame

        frame_type = self.frame_type or GeoDataFrame
        aggregated = self.to_native_tabular_result().to_geodataframe()
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
            geometry_metadata=_cached_geometry_metadata(geometry),
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
        geometry_metadata=_cached_geometry_metadata(geometry),
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
        geometry_metadata=_cached_geometry_metadata(result),
    )


def _default_constructive_source_rows(row_count: int, *, prefer_device: bool):
    if prefer_device:
        import cupy as cp

        return cp.arange(row_count, dtype=cp.int32)
    return np.arange(row_count, dtype=np.intp)


def _owned_row_family_tags(owned):
    if getattr(owned, "residency", None) is Residency.DEVICE:
        state = owned._ensure_device_state()
        return state.tags
    return owned.tags


def _unary_constructive_owned_to_native_tabular_result(
    owned,
    *,
    operation: str,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    """Lower row-aligned unary constructive owned output to the native boundary."""
    row_count = int(owned.row_count)
    if source_rows is None:
        source_rows = _default_constructive_source_rows(
            row_count,
            prefer_device=getattr(owned, "residency", None) is Residency.DEVICE,
        )
    geometry = GeometryNativeResult.from_owned(owned, crs=crs)
    return NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count))),
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
        attrs=attrs,
        provenance=NativeGeometryProvenance(
            operation=operation,
            row_count=row_count,
            source_rows=source_rows,
            source_tokens=source_tokens,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
    )


def _constructive_part_source_rows(
    source_rows,
    part_source_rows,
    *,
    source_row_count: int,
    output_row_count: int,
    prefer_device: bool,
):
    if source_rows is None:
        if part_source_rows is not None:
            return part_source_rows
        return _default_constructive_source_rows(
            output_row_count,
            prefer_device=prefer_device,
        )
    if _index_array_size(source_rows) != int(source_row_count):
        raise ValueError("source_rows must match the input row count")
    if part_source_rows is None:
        return _default_constructive_source_rows(
            output_row_count,
            prefer_device=prefer_device,
        )
    if (
        hasattr(source_rows, "__cuda_array_interface__")
        or hasattr(part_source_rows, "__cuda_array_interface__")
    ):
        import cupy as cp

        return cp.asarray(source_rows)[cp.asarray(part_source_rows, dtype=cp.int64)]
    return np.asarray(source_rows)[np.asarray(part_source_rows, dtype=np.int64)]


def _polygonal_parts_constructive_to_native_tabular_result(
    owned,
    *,
    operation: str,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    """Lower non-row-aligned polygonal part output to the native boundary."""
    from vibespatial.constructive.binary_constructive import (
        _explode_polygonal_rows_to_polygons_gpu,
    )

    exploded = _explode_polygonal_rows_to_polygons_gpu(owned)
    prefer_device = getattr(owned, "residency", None) is Residency.DEVICE
    if exploded is None:
        row_count = 0
        geometry = _empty_geometry_native_result(geometry_name=geometry_name, crs=crs)
        part_source_rows = None
    else:
        parts, part_source_rows = exploded
        row_count = int(parts.row_count)
        geometry = GeometryNativeResult.from_owned(parts, crs=crs)

    output_source_rows = _constructive_part_source_rows(
        source_rows,
        part_source_rows,
        source_row_count=int(owned.row_count),
        output_row_count=row_count,
        prefer_device=prefer_device,
    )
    return NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count))),
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
        attrs=attrs,
        provenance=NativeGeometryProvenance(
            operation=operation,
            row_count=row_count,
            source_rows=output_source_rows,
            part_family_tags=_owned_row_family_tags(parts) if row_count else None,
            source_tokens=source_tokens,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
    )


def _point_parts_constructive_to_native_tabular_result(
    owned,
    *,
    operation: str,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    """Lower non-row-aligned point part output to the native boundary."""
    from vibespatial.constructive.binary_constructive import (
        _explode_point_rows_to_points_gpu,
    )

    exploded = _explode_point_rows_to_points_gpu(owned)
    prefer_device = getattr(owned, "residency", None) is Residency.DEVICE
    if exploded is None:
        row_count = 0
        geometry = _empty_geometry_native_result(geometry_name=geometry_name, crs=crs)
        part_source_rows = None
    else:
        parts, part_source_rows = exploded
        row_count = int(parts.row_count)
        geometry = GeometryNativeResult.from_owned(parts, crs=crs)

    output_source_rows = _constructive_part_source_rows(
        source_rows,
        part_source_rows,
        source_row_count=int(owned.row_count),
        output_row_count=row_count,
        prefer_device=prefer_device,
    )
    return NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count))),
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
        attrs=attrs,
        provenance=NativeGeometryProvenance(
            operation=operation,
            row_count=row_count,
            source_rows=output_source_rows,
            part_family_tags=_owned_row_family_tags(parts) if row_count else None,
            source_tokens=source_tokens,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
    )


def _lineal_parts_constructive_to_native_tabular_result(
    owned,
    *,
    operation: str,
    crs=None,
    geometry_name: str = "geometry",
    source_rows=None,
    source_tokens: tuple[str, ...] = (),
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    """Lower non-row-aligned lineal part output to the native boundary."""
    from vibespatial.constructive.binary_constructive import (
        _explode_lineal_rows_to_lines_gpu,
    )

    exploded = _explode_lineal_rows_to_lines_gpu(owned)
    prefer_device = getattr(owned, "residency", None) is Residency.DEVICE
    if exploded is None:
        row_count = 0
        geometry = _empty_geometry_native_result(geometry_name=geometry_name, crs=crs)
        part_source_rows = None
    else:
        parts, part_source_rows = exploded
        row_count = int(parts.row_count)
        geometry = GeometryNativeResult.from_owned(parts, crs=crs)

    output_source_rows = _constructive_part_source_rows(
        source_rows,
        part_source_rows,
        source_row_count=int(owned.row_count),
        output_row_count=row_count,
        prefer_device=prefer_device,
    )
    return NativeTabularResult(
        attributes=NativeAttributeTable(dataframe=pd.DataFrame(index=pd.RangeIndex(row_count))),
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(geometry_name,),
        attrs=attrs,
        provenance=NativeGeometryProvenance(
            operation=operation,
            row_count=row_count,
            source_rows=output_source_rows,
            part_family_tags=_owned_row_family_tags(parts) if row_count else None,
            source_tokens=source_tokens,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
    )


def _grouped_constructive_result_to_native_tabular_result(
    result: GroupedConstructiveResult,
) -> NativeTabularResult:
        return _grouped_constructive_to_native_tabular_result(
            geometry=result.geometry,
            attributes=result.attributes,
            geometry_name=result.geometry_name,
            as_index=result.as_index,
            provenance=result.provenance,
        )


def _grouped_constructive_to_native_tabular_result(
    *,
    geometry: GeometryNativeResult,
    attributes: NativeAttributeTable | pd.DataFrame,
    geometry_name: str,
    as_index: bool,
    provenance: NativeGeometryProvenance | None = None,
) -> NativeTabularResult:
    attributes = NativeAttributeTable.from_value(attributes)
    leading_columns: list[str] = []
    trailing_columns = list(attributes.columns)
    if not as_index:
        attributes, reset_leading, reset_trailing = attributes.reset_index_deferred()
        leading_columns = list(reset_leading)
        trailing_columns = list(reset_trailing)
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*leading_columns, geometry_name, *trailing_columns]),
        provenance=provenance,
        geometry_metadata=_cached_geometry_metadata(geometry),
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
        geometry_metadata=result.geometry_metadata,
    )


def _pairwise_constructive_native_state_attribute_parts(
    *,
    relation: RelationIndexResult,
    left_df: GeoDataFrame,
    right_df: GeoDataFrame,
    geometry_name: str,
) -> tuple[NativeAttributeTable, tuple[NativeGeometryColumn, ...], tuple[Any, ...]] | None:
    """Gather pairwise constructive attributes from attached native frame states."""
    from vibespatial.api._native_relation import NativeRelation
    from vibespatial.api._native_state import get_native_state

    left_state = get_native_state(left_df)
    right_state = get_native_state(right_df)
    if left_state is None or right_state is None:
        return None
    if left_state.geometry_name != left_df._geometry_column_name:
        return None
    if right_state.geometry_name != right_df._geometry_column_name:
        return None
    if left_state.secondary_geometry or right_state.secondary_geometry:
        return None
    if _has_secondary_geometry_columns(left_df, left_state.geometry_name):
        return None
    if _has_secondary_geometry_columns(right_df, right_state.geometry_name):
        return None

    left_attributes = NativeAttributeTable.from_value(left_state.attributes)
    right_attributes = NativeAttributeTable.from_value(right_state.attributes)
    relation_uses_device_rows = hasattr(
        relation.left_indices,
        "__cuda_array_interface__",
    ) or hasattr(relation.right_indices, "__cuda_array_interface__")
    if (
        not relation_uses_device_rows
        and (
            left_attributes.device_table is not None
            or right_attributes.device_table is not None
        )
    ):
        try:
            import cupy as cp
        except ModuleNotFoundError:
            return None
        relation = RelationIndexResult(
            cp.asarray(relation.left_indices, dtype=cp.int64),
            cp.asarray(relation.right_indices, dtype=cp.int64),
        )

    native_relation = NativeRelation.from_relation_index_result(
        relation,
        left_token=left_state.lineage_token,
        right_token=right_state.lineage_token,
        left_row_count=left_state.row_count,
        right_row_count=right_state.row_count,
    )
    _validate_relation_constructive_state(
        native_relation,
        left_state,
        right_state,
    )
    attributes, column_order = _relation_constructive_attribute_parts(
        relation=native_relation,
        left_state=left_state,
        right_state=right_state,
        geometry_name=geometry_name,
    )
    expected_attribute_count = len(left_attributes.columns) + len(right_attributes.columns)
    if len(attributes.columns) != expected_attribute_count:
        return None
    return attributes, (), column_order


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
    row_count = len(relation)
    native_parts = _pairwise_constructive_native_state_attribute_parts(
        relation=relation,
        left_df=left_df,
        right_df=right_df,
        geometry_name=geometry_name,
    )
    if native_parts is None:
        h_left_idx, h_right_idx = relation.to_host(
            surface="vibespatial.api._native_results._pairwise_constructive_to_native_tabular_result",
            operation="pairwise_constructive_relation_indices_to_host",
            reason="native pairwise constructive tabular assembly still needs pandas-compatible attribute indexers",
            dtype=np.intp,
        )
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
    else:
        attributes, secondary_geometry, column_order = native_parts

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
    base_columns = [
        column
        for column in column_order
        if column != geometry_name and column not in trailing_columns
    ]
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*base_columns, geometry_name, *trailing_columns]),
        attrs=result_attrs or None,
        secondary_geometry=secondary_geometry,
        provenance=NativeGeometryProvenance(
            operation="pairwise_constructive",
            row_count=row_count,
            left_rows=relation.left_indices,
            right_rows=relation.right_indices,
            keep_geom_type_applied=keep_geom_type_applied,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
    )


def _validate_relation_constructive_state(
    relation,
    left_state,
    right_state,
) -> None:
    if relation.left_token is not None and relation.left_token != left_state.lineage_token:
        raise ValueError("NativeRelation left source token does not match left state")
    if relation.right_token is not None and relation.right_token != right_state.lineage_token:
        raise ValueError("NativeRelation right source token does not match right state")
    if (
        relation.left_row_count is not None
        and int(relation.left_row_count) != int(left_state.row_count)
    ):
        raise ValueError("NativeRelation left row count does not match left state")
    if (
        relation.right_row_count is not None
        and int(relation.right_row_count) != int(right_state.row_count)
    ):
        raise ValueError("NativeRelation right row count does not match right state")


def _relation_constructive_attribute_parts(
    *,
    relation,
    left_state,
    right_state,
    geometry_name: str,
) -> tuple[NativeAttributeTable, tuple[Any, ...]]:
    """Gather relation-aligned attributes when doing so keeps pair rows native."""
    pair_count = len(relation)
    out_index = pd.RangeIndex(pair_count)
    left_attrs = NativeAttributeTable.from_value(left_state.attributes)
    right_attrs = NativeAttributeTable.from_value(right_state.attributes)
    left_positions = relation.left_indices
    right_positions = relation.right_indices
    if (
        not hasattr(left_positions, "__cuda_array_interface__")
        and (
            left_attrs.device_table is not None
            or right_attrs.device_table is not None
        )
    ):
        try:
            import cupy as cp
        except ModuleNotFoundError:
            pass
        else:
            left_positions = cp.asarray(left_positions, dtype=cp.int64)
            right_positions = cp.asarray(right_positions, dtype=cp.int64)

    def _can_gather(table: NativeAttributeTable, row_positions) -> bool:
        if len(table.columns) == 0:
            return True
        if hasattr(row_positions, "__cuda_array_interface__"):
            return table.device_table is not None or table.arrow_table is not None
        return True

    if not _can_gather(left_attrs, left_positions) or not _can_gather(
        right_attrs,
        right_positions,
    ):
        return NativeAttributeTable(dataframe=pd.DataFrame(index=out_index)), (
            geometry_name,
        )

    left_columns, right_columns = _process_column_names_with_suffix(
        left_attrs.columns,
        right_attrs.columns,
        ("1", "2"),
        SimpleNamespace(_geometry_column_name=left_state.geometry_name),
        SimpleNamespace(_geometry_column_name=right_state.geometry_name),
    )
    left_mapping = {
        old: new
        for old, new in zip(left_attrs.columns, left_columns, strict=True)
        if old != new
    }
    right_mapping = {
        old: new
        for old, new in zip(right_attrs.columns, right_columns, strict=True)
        if old != new
    }
    left_projected = left_attrs.take(
        left_positions,
        preserve_index=False,
    ).rename_columns(left_mapping)
    right_projected = right_attrs.take(
        right_positions,
        preserve_index=False,
    ).rename_columns(right_mapping)
    combined = NativeAttributeTable.combine_columns(
        [left_projected, right_projected],
        index_override=out_index,
    )
    if combined is None:
        return NativeAttributeTable(dataframe=pd.DataFrame(index=out_index)), (
            geometry_name,
        )
    return combined, tuple([*combined.columns, geometry_name])


def _relation_constructive_to_native_tabular_result(
    *,
    op: str,
    relation,
    left_state,
    right_state,
    geometry_name: str = "geometry",
    dispatch_mode: ExecutionMode | str = ExecutionMode.GPU,
    frame_attrs: dict[str, Any] | None = None,
) -> NativeTabularResult | None:
    """Lower relation-pair constructive geometry to the native tabular boundary.

    Physical shape: relation-pair consume.  The input carriers are a
    ``NativeRelation`` plus left/right ``NativeFrameState`` geometry columns.
    Pair rows are gathered without formatting public pair indices, then the
    existing pairwise constructive kernel produces row-aligned output geometry.
    Relation-aligned attributes are gathered only when the source attribute
    carriers can be taken without hidden device pair materialization.
    """
    _validate_relation_constructive_state(relation, left_state, right_state)
    left_owned = getattr(left_state.geometry, "owned", None)
    right_owned = getattr(right_state.geometry, "owned", None)
    if left_owned is None or right_owned is None:
        return None

    left_pairs = left_owned.take(relation.left_indices)
    right_pairs = right_owned.take(relation.right_indices)
    from vibespatial.constructive.binary_constructive import binary_constructive_owned

    constructed = binary_constructive_owned(
        op,
        left_pairs,
        right_pairs,
        dispatch_mode=dispatch_mode,
        _allow_rectangle_intersection_fast_path=False,
    )
    pair_count = len(relation)
    if int(constructed.row_count) != int(pair_count):
        raise ValueError(
            "relation constructive output row count must match relation pair count"
        )
    crs = left_state.geometry.crs
    geometry = GeometryNativeResult.from_owned(constructed, crs=crs)
    attributes, column_order = _relation_constructive_attribute_parts(
        relation=relation,
        left_state=left_state,
        right_state=right_state,
        geometry_name=geometry_name,
    )
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=column_order,
        attrs=frame_attrs,
        provenance=NativeGeometryProvenance(
            operation=f"relation_{op}",
            row_count=pair_count,
            left_rows=relation.left_indices,
            right_rows=relation.right_indices,
            source_tokens=(
                left_state.lineage_token,
                right_state.lineage_token,
            ),
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
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
    row_count = int(row_positions.size)
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
        provenance=NativeGeometryProvenance(
            operation="left_constructive",
            row_count=row_count,
            source_rows=row_positions,
        ),
        geometry_metadata=_cached_geometry_metadata(geometry),
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
    sorted_positions = concat_row_positions[sorter]
    insertion = np.searchsorted(
        sorted_positions,
        ordered_row_positions,
    )
    valid = insertion < sorted_positions.size
    if np.any(valid):
        valid_indices = insertion[valid]
        valid[valid] = sorted_positions[valid_indices] == ordered_row_positions[valid]
    return sorter[insertion[valid]].astype(np.intp, copy=False)


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


def _clip_source_attributes_for_rows(
    source,
    *,
    geometry_name: str,
    row_positions,
) -> NativeAttributeTable | None:
    """Gather clip source attributes from an attached native frame when admissible."""
    if not hasattr(source, "_geometry_column_name"):
        return None
    if source._geometry_column_name != geometry_name:
        return None

    from vibespatial.api._native_state import get_native_state

    source_state = get_native_state(source)
    if source_state is None or source_state.geometry_name != geometry_name:
        return None
    if source_state.secondary_geometry or _has_secondary_geometry_columns(
        source,
        geometry_name,
    ):
        return None

    attributes = NativeAttributeTable.from_value(source_state.attributes)
    expected_columns = tuple(column for column in source.columns if column != geometry_name)
    if tuple(attributes.columns) != expected_columns:
        projected = attributes.project_columns(expected_columns)
        if projected is None:
            return None
        attributes = projected

    h_rows = np.asarray(row_positions, dtype=np.int64)
    index_override = source.index.take(h_rows)
    if attributes.device_table is not None:
        try:
            import cupy as cp
        except ModuleNotFoundError:
            return None
        gathered = attributes.take(
            cp.asarray(h_rows, dtype=cp.int64),
            preserve_index=False,
        )
        return gathered.with_index(index_override)

    return attributes.take(h_rows, preserve_index=True).with_index(index_override)


def _clip_project_source_attributes(
    source,
    *,
    geometry_name: str,
    row_positions,
) -> NativeAttributeTable:
    native_attributes = _clip_source_attributes_for_rows(
        source,
        geometry_name=geometry_name,
        row_positions=row_positions,
    )
    if native_attributes is not None:
        return native_attributes

    row_positions = np.asarray(row_positions, dtype=np.intp)
    if hasattr(source, "_geometry_column_name"):
        source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
        projected = source_attrs.iloc[row_positions].copy(deep=False)
        return _native_attribute_table_from_projected_frames(
            [projected],
            index_override=source.index.take(row_positions),
        )

    return NativeAttributeTable(
        dataframe=pd.DataFrame(index=source.index.take(row_positions))
    )


def _clip_owned_geometry_native_result(result, *, crs):
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.geometry.owned import FAMILY_TAGS, GeometryFamily, OwnedGeometryArray
    from vibespatial.runtime.residency import Residency

    if not result.parts or not all(part.geometry.owned is not None for part in result.parts):
        return None, None, None

    row_parts = [np.asarray(part.row_positions, dtype=np.intp) for part in result.parts]
    owned_parts = [part.geometry.owned for part in result.parts]
    semantic_cleanup_done = all(
        bool(getattr(owned, "_clip_semantically_clean", False))
        for owned in owned_parts
    )
    row_positions = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.intp)
    combined_owned = (
        owned_parts[0]
        if len(owned_parts) == 1
        else OwnedGeometryArray.concat(owned_parts)
    )
    initial_residency = combined_owned.residency

    def _take_owned_rows(owned, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.intp)
        if rows.size == owned.row_count and np.array_equal(
            rows,
            np.arange(owned.row_count, dtype=rows.dtype),
        ):
            return owned
        if owned.residency is Residency.DEVICE:
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover - guarded by residency
                cp = None
            if cp is not None:
                return owned.device_take(
                    cp.asarray(rows, dtype=cp.int64),
                    host_indices_for_sizing=np.asarray(rows, dtype=np.int64),
                )
        return owned.take(rows)

    def _valid_nonempty_mask(owned) -> np.ndarray:
        if owned.residency is Residency.DEVICE and owned.device_state is not None:
            try:
                import cupy as cp
            except ModuleNotFoundError:  # pragma: no cover - guarded by residency
                cp = None
            if cp is not None:
                d_state = owned.device_state
                d_keep = d_state.validity.astype(cp.bool_, copy=True)
                for family, d_buf in d_state.families.items():
                    d_family_rows = cp.flatnonzero(
                        d_keep & (d_state.tags == FAMILY_TAGS[family])
                    ).astype(cp.int64, copy=False)
                    if int(d_family_rows.size) == 0:
                        continue
                    d_local_rows = d_state.family_row_offsets[d_family_rows]
                    d_keep[d_family_rows] &= ~d_buf.empty_mask[
                        d_local_rows.astype(cp.int64, copy=False)
                    ].astype(cp.bool_, copy=False)
                return _host_array(
                    d_keep,
                    dtype=bool,
                    strict_disallowed=False,
                    surface="vibespatial.api._native_results._clip_constructive_parts_to_native_tabular_result",
                    operation="clip_valid_nonempty_mask_to_host",
                    reason="clip native tabular cleanup still needs a host row mask before final geometry take",
                    detail=f"rows={int(d_keep.size)}, bytes={int(d_keep.size)}",
                ).astype(bool, copy=False)

        geometry_array = GeometryArray.from_owned(owned, crs=crs)
        return np.asarray(owned.validity, dtype=bool) & ~np.asarray(
            geometry_array.is_empty,
            dtype=bool,
        )

    if row_positions.size > 1:
        reorder = _reorder_concat_positions(
            row_positions,
            np.asarray(result.ordered_row_positions, dtype=np.intp),
        )
        if not np.array_equal(reorder, np.arange(reorder.size, dtype=reorder.dtype)):
            row_positions = row_positions[reorder]
            combined_owned = _take_owned_rows(combined_owned, reorder)

    geometry_array = None
    if not semantic_cleanup_done:
        keep = _valid_nonempty_mask(combined_owned)
        if not keep.all():
            keep_rows = np.flatnonzero(keep).astype(np.intp, copy=False)
            row_positions = row_positions[keep_rows]
            combined_owned = _take_owned_rows(combined_owned, keep_rows)
            geometry_array = None

    repaired_mask = None
    if (
        not semantic_cleanup_done
        and not result.clipping_by_rectangle
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
            polygon_rows = np.flatnonzero(polygon_mask).astype(np.intp, copy=False)
            polygon_owned = _take_owned_rows(combined_owned, polygon_rows)
            from vibespatial.constructive.measurement import area_owned
            from vibespatial.geometry.device_array import _compute_bounds_from_owned

            nonpositive_area = ~(
                np.asarray(
                    area_owned(polygon_owned),
                    dtype=np.float64,
                ) > 0.0
            )
            polygon_bounds = np.asarray(
                _compute_bounds_from_owned(polygon_owned),
                dtype=np.float64,
            ).reshape(-1, 4)
            pointlike_zero_area = (
                nonpositive_area
                & (np.abs(polygon_bounds[:, 2] - polygon_bounds[:, 0]) <= SPATIAL_EPSILON)
                & (np.abs(polygon_bounds[:, 3] - polygon_bounds[:, 1]) <= SPATIAL_EPSILON)
            )
            keep[polygon_mask] = ~nonpositive_area | pointlike_zero_area
            if not keep.all():
                keep_rows = np.flatnonzero(keep).astype(np.intp, copy=False)
                row_positions = row_positions[keep_rows]
                combined_owned = _take_owned_rows(combined_owned, keep_rows)
                geometry_array = None
                tags = np.asarray(combined_owned.tags, dtype=np.int8)

        line_mask = np.isin(
            tags,
            [
                FAMILY_TAGS[GeometryFamily.LINESTRING],
                FAMILY_TAGS[GeometryFamily.MULTILINESTRING],
            ],
        )
        if np.any(line_mask):
            if geometry_array is None:
                geometry_array = GeometryArray.from_owned(combined_owned, crs=crs)
            degenerate = line_mask & (
                np.asarray(geometry_array.length, dtype=np.float64) == 0.0
            )
            if np.any(degenerate):
                degenerate_rows = np.flatnonzero(degenerate).astype(np.intp, copy=False)
                repaired = geometry_array.take(degenerate_rows).make_valid()
                repaired_owned = getattr(repaired, "_owned", None)
                if repaired_owned is None or repaired_owned.row_count != degenerate_rows.size:
                    return None, None, None
                keep_rows = np.flatnonzero(~degenerate).astype(np.intp, copy=False)
                index_oga_pairs = []
                if keep_rows.size:
                    index_oga_pairs.append((keep_rows, _take_owned_rows(combined_owned, keep_rows)))
                index_oga_pairs.append((degenerate_rows, repaired_owned))
                combined_owned = _assemble_indexed_owned_parts(
                    index_oga_pairs,
                    combined_owned.row_count,
                )
                repaired_mask = np.zeros(combined_owned.row_count, dtype=bool)
                repaired_mask[degenerate_rows] = True

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

    return GeometryNativeResult.from_owned(combined_owned, crs=crs), row_positions, repaired_mask


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
        else:
            geometry_name = getattr(source, "name", None) or "geometry"
        attributes = _clip_project_source_attributes(
            source,
            geometry_name=geometry_name,
            row_positions=np.empty(0, dtype=np.intp),
        )
        geometry = _empty_geometry_native_result(geometry_name=geometry_name, crs=source.crs)
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
            provenance=NativeGeometryProvenance(
                operation="clip",
                row_count=0,
                source_rows=np.empty(0, dtype=np.intp),
                keep_geom_type_applied=keep_geom_type,
            ),
            geometry_metadata=_cached_geometry_metadata(geometry),
        )

    if len(parts) == 1 and not has_non_point_candidates and not keep_geom_type:
        part = parts[0]
        row_positions = np.asarray(part.row_positions, dtype=np.intp)
        if hasattr(source, "_geometry_column_name"):
            geometry_name = source._geometry_column_name
        else:
            geometry_name = getattr(source, "name", None) or "geometry"
        attributes = _clip_project_source_attributes(
            source,
            geometry_name=geometry_name,
            row_positions=row_positions,
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
            provenance=NativeGeometryProvenance(
                operation="clip",
                row_count=int(row_positions.size),
                source_rows=row_positions,
                keep_geom_type_applied=keep_geom_type,
            ),
            geometry_metadata=_cached_geometry_metadata(part.geometry),
        )

    if not keep_geom_type:
        geometry_name = (
            source._geometry_column_name
            if hasattr(source, "_geometry_column_name")
            else getattr(source, "name", None) or "geometry"
        )
        owned_geometry, owned_rows, repaired_mask = _clip_owned_geometry_native_result(
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
            attributes = _clip_project_source_attributes(
                source,
                geometry_name=geometry_name,
                row_positions=owned_rows,
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
                provenance=NativeGeometryProvenance(
                    operation="clip",
                    row_count=int(np.asarray(owned_rows).size),
                    source_rows=owned_rows,
                    repaired_mask=repaired_mask,
                    keep_geom_type_applied=keep_geom_type,
                ),
                geometry_metadata=_cached_geometry_metadata(owned_geometry),
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
            reorder = _reorder_concat_positions(row_positions, ordered_rows)
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
            repaired_mask = None
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
                    if repaired_mask is not None:
                        repaired_mask = repaired_mask[keep]

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
                    repaired_mask = np.zeros(len(geometry_values), dtype=bool)
                    repaired_mask[line_rows] = True
        else:
            repaired_mask = None

        geometry_name = (
            source._geometry_column_name
            if hasattr(source, "_geometry_column_name")
            else getattr(source, "name", None) or "geometry"
        )
        attributes = _clip_project_source_attributes(
            source,
            geometry_name=geometry_name,
            row_positions=row_positions,
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
            provenance=NativeGeometryProvenance(
                operation="clip",
                row_count=int(row_positions.size),
                source_rows=row_positions,
                repaired_mask=repaired_mask,
                keep_geom_type_applied=keep_geom_type,
            ),
            geometry_metadata=_cached_geometry_metadata(geometry),
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
    if attribute_storage == "device":
        state = _relation_join_export_result_to_native_frame_state_device(result)
        if state is not None:
            return state.to_native_tabular_result()

    relation = result.relation_result.relation
    if len(relation) == 0:
        relation_indices = (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )
        relation_distances = (
            None
            if result.relation_result.distances is None
            else np.empty(0, dtype=np.float64)
        )
    else:
        relation_indices = relation.to_host(
            surface="vibespatial.api._native_results._relation_join_export_result_to_native_tabular_result",
            operation="relation_join_indices_to_host",
            reason="native relation join tabular assembly still needs pandas-compatible attribute indexers",
            dtype=np.intp,
        )
        relation_distances = result.relation_result.to_host_distances()

    attributes, geometry, geometry_name, _crs, column_order, secondary_geometry, distances = _native_relation_join_parts(
        result.left_df,
        result.right_df,
        relation_indices,
        relation_distances,
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
        geometry_metadata=_cached_geometry_metadata(geometry),
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
    object_parts = [np.asarray(series.values, dtype=object) for series in series_parts]
    if len(object_parts) == 1:
        combined_values = object_parts[0]
    else:
        combined_values = np.concatenate(object_parts)
    return GeometryNativeResult.from_values(
        combined_values,
        crs=crs,
        index=pd.RangeIndex(int(combined_values.size)),
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
    ignore_index: bool = True,
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

    if provenance is None and all(
        isinstance(result.provenance, NativeGeometryProvenance)
        for result in results
    ):
        provenance = NativeGeometryProvenance.concat(
            [result.provenance for result in results],
            operation="concat_constructive",
        )

    merged_attrs: dict[str, Any] = {}
    for result in results:
        if result.attrs:
            merged_attrs.update(result.attrs)
    if attrs:
        merged_attrs.update(attrs)

    attributes = NativeAttributeTable.concat(
        [result.attributes for result in results],
        ignore_index=ignore_index,
        sort=False,
    )
    geometry = _concat_geometry_native_results(
        results,
        geometry_name=geometry_name,
        crs=crs,
    )
    geometry_metadata = None
    if all(result.geometry_metadata is not None for result in results):
        from vibespatial.api._native_metadata import NativeGeometryMetadata

        geometry_metadata = NativeGeometryMetadata.concat(
            [result.geometry_metadata for result in results],
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
        geometry_metadata=geometry_metadata,
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
        geometry_metadata=merged.geometry_metadata,
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
