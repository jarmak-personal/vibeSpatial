from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import shapely

from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.fallbacks import record_fallback_event

if TYPE_CHECKING:
    from vibespatial.api.geodataframe import GeoDataFrame
    from vibespatial.api.geoseries import GeoSeries


def _host_array(values: Any, *, dtype) -> np.ndarray:
    host_values = values.get() if hasattr(values, "get") else values
    return np.asarray(host_values, dtype=dtype)


@dataclass(frozen=True)
class NativeAttributeTable:
    """Attribute payload that can stay columnar without requiring pandas storage."""

    dataframe: pd.DataFrame | None = None
    arrow_table: Any | None = None
    index_override: pd.Index | None = None

    def __post_init__(self) -> None:
        if (self.dataframe is None) == (self.arrow_table is None):
            raise ValueError(
                "NativeAttributeTable requires exactly one of dataframe or arrow_table"
            )
        if self.arrow_table is not None and self.index_override is None:
            object.__setattr__(
                self,
                "index_override",
                pd.RangeIndex(int(self.arrow_table.num_rows)),
            )

    @classmethod
    def from_value(cls, value) -> NativeAttributeTable:
        if isinstance(value, NativeAttributeTable):
            return value
        if isinstance(value, pd.DataFrame):
            return cls(dataframe=value)

        try:
            import pyarrow as pa
        except ImportError:  # pragma: no cover - pyarrow is present in normal test envs
            pa = None
        if pa is not None and isinstance(value, pa.Table):
            return cls(arrow_table=value)

        raise TypeError("NativeAttributeTable expects pandas DataFrame or pyarrow.Table")

    @property
    def index(self) -> pd.Index:
        if self.dataframe is not None:
            return self.dataframe.index
        return self.index_override

    @property
    def columns(self) -> pd.Index:
        if self.dataframe is not None:
            return self.dataframe.columns
        return pd.Index(self.arrow_table.column_names)

    def to_pandas(self, *, copy: bool = False) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy(deep=copy) if copy else self.dataframe

        frame = self.arrow_table.to_pandas()
        frame.index = self.index
        return frame.copy(deep=copy) if copy else frame

    def to_arrow(self, *, index: bool | None = None, columns=None):
        import pyarrow as pa

        requested_columns = None if columns is None else list(columns)
        can_skip_index = (
            index is False
            or (
                index is None
                and isinstance(self.index_override, pd.RangeIndex)
                and self.index_override.start == 0
                and self.index_override.step == 1
                and list(self.index_override.names) == [None]
            )
        )
        if self.arrow_table is not None and can_skip_index:
            table = self.arrow_table
            if requested_columns is not None:
                table = table.select(requested_columns)
            return table

        frame = self.to_pandas(copy=False)
        if requested_columns is not None:
            frame = frame.loc[:, requested_columns]
        return pa.Table.from_pandas(frame, preserve_index=index)

    def with_column(self, name: str, values) -> NativeAttributeTable:
        if self.arrow_table is not None:
            import pyarrow as pa

            table = self.arrow_table.append_column(name, pa.array(values))
            return type(self)(arrow_table=table, index_override=self.index)

        frame = self.dataframe.copy(deep=False)
        frame[name] = values
        return type(self)(dataframe=frame)

    def rename_columns(self, mapping: dict[str, str]) -> NativeAttributeTable:
        if not mapping:
            return self
        if self.arrow_table is not None:
            renamed = [mapping.get(name, name) for name in self.arrow_table.column_names]
            return type(self)(arrow_table=self.arrow_table.rename_columns(renamed), index_override=self.index)
        return type(self)(dataframe=self.dataframe.rename(columns=mapping, copy=False))

    @classmethod
    def concat(
        cls,
        tables: list[NativeAttributeTable],
        *,
        ignore_index: bool = True,
        sort: bool = False,
    ) -> NativeAttributeTable:
        if not tables:
            return cls(dataframe=pd.DataFrame(index=pd.RangeIndex(0)))

        try:
            import pyarrow as pa
        except ImportError:  # pragma: no cover - pyarrow present in normal envs
            pa = None

        if pa is None:
            frames = [table.to_pandas(copy=False) for table in tables]
            return cls(dataframe=pd.concat(frames, ignore_index=ignore_index, sort=sort))

        arrow_tables = [table.to_arrow(index=False) for table in tables]
        ordered_columns: list[str] = []
        column_types: dict[str, Any] = {}
        for table in arrow_tables:
            for field in table.schema:
                if field.name not in column_types:
                    column_types[field.name] = field.type
                    ordered_columns.append(field.name)

        aligned_tables = []
        for table in arrow_tables:
            table_columns = set(table.column_names)
            arrays = []
            for name in ordered_columns:
                if name in table_columns:
                    arrays.append(table[name])
                else:
                    arrays.append(pa.nulls(table.num_rows, type=column_types[name]))
            aligned_tables.append(pa.table(arrays, names=ordered_columns))

        concatenated = pa.concat_tables(aligned_tables)
        if ignore_index:
            index_override = pd.RangeIndex(concatenated.num_rows)
        else:
            index_override = tables[0].index
            for table in tables[1:]:
                index_override = index_override.append(table.index)
        return cls(arrow_table=concatenated, index_override=index_override)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key):
        if self.dataframe is not None:
            return self.dataframe[key]
        if isinstance(key, (list, tuple)):
            return self.to_pandas(copy=False).loc[:, list(key)]
        if isinstance(key, pd.Index):
            return self.to_pandas(copy=False).loc[:, list(key)]
        return self.to_pandas(copy=False)[key]

    def __getattr__(self, name: str):
        return getattr(self.to_pandas(copy=False), name)


def _set_active_geometry_name(frame, geometry_name: str):
    """Force the active geometry column name, replacing stale inactive collisions."""
    current_name = frame._geometry_column_name
    if current_name == geometry_name:
        return frame
    if geometry_name in frame.columns:
        frame = frame.drop(columns=[geometry_name])
    return frame.rename_geometry(geometry_name)


def _replace_geometry_column_preserving_backing(frame, values, *, crs):
    """Replace a GeoDataFrame geometry column without demoting DGA-backed data."""
    from vibespatial.api.geometry_array import GeometryArray
    from vibespatial.api.geoseries import GeoSeries
    from vibespatial.geometry.device_array import DeviceGeometryArray

    geom_name = frame._geometry_column_name
    if isinstance(values, GeometryArray | DeviceGeometryArray):
        geometry_series = pd.Series(values, index=frame.index, copy=False, name=geom_name)
    else:
        geometry_series = GeoSeries(values, index=frame.index, crs=crs, name=geom_name)

    data_columns = {
        column_name: (geometry_series if column_name == geom_name else frame[column_name])
        for column_name in frame.columns
    }
    rebuilt = pd.DataFrame(data_columns, index=frame.index, copy=False)
    rebuilt.__class__ = type(frame)
    rebuilt._geometry_column_name = geom_name
    rebuilt.attrs = frame.attrs.copy()
    return rebuilt


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
class GeometryNativeResult:
    """Geometry result that stays native until explicitly materialized."""

    crs: Any
    owned: Any | None = None
    series: GeoSeries | None = None

    def __post_init__(self) -> None:
        if (self.owned is None) == (self.series is None):
            raise ValueError("GeometryNativeResult requires exactly one of owned or series")

    @classmethod
    def from_owned(cls, owned, *, crs) -> GeometryNativeResult:
        return cls(crs=crs, owned=owned)

    @classmethod
    def from_geoseries(cls, series: GeoSeries) -> GeometryNativeResult:
        owned = getattr(series.values, "_owned", None)
        if owned is not None:
            return cls.from_owned(owned, crs=series.crs)
        return cls(crs=series.crs, series=series)

    @classmethod
    def from_values(
        cls,
        values,
        *,
        crs,
        index=None,
        name: str | None = None,
    ) -> GeometryNativeResult:
        from vibespatial.api.geometry_array import GeometryArray
        owned = getattr(values, "_owned", None)
        if "DeviceGeometryArray" in type(values).__name__ and owned is not None:
            return cls.from_owned(owned, crs=crs)

        from vibespatial.api.geoseries import GeoSeries

        if isinstance(values, GeometryArray):
            if index is None:
                index = pd.RangeIndex(len(values))
            return cls.from_geoseries(GeoSeries(values, index=index, name=name, crs=crs))

        if owned is not None:
            return cls.from_owned(owned, crs=crs)
        if index is None:
            index = pd.RangeIndex(len(values))
        return cls.from_geoseries(GeoSeries(values, index=index, name=name, crs=crs))

    def to_geoseries(self, *, index, name: str) -> GeoSeries:
        if self.owned is not None:
            from vibespatial.api.geometry_array import GeometryArray
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.io.geoarrow import geoseries_from_owned
            from vibespatial.runtime.residency import Residency

            if self.owned.residency is Residency.DEVICE:
                return geoseries_from_owned(
                    self.owned,
                    name=name,
                    crs=self.crs,
                    index=index,
                )
            return GeoSeries(
                GeometryArray.from_owned(self.owned, crs=self.crs),
                index=index,
                name=name,
                crs=self.crs,
            )
        from vibespatial.api.geoseries import GeoSeries

        values = self.series.values
        return GeoSeries(values, index=index, name=name, crs=self.crs)

    @property
    def row_count(self) -> int:
        if self.owned is not None:
            return int(self.owned.row_count)
        return int(len(self.series))


@dataclass(frozen=True)
class NativeTabularResult:
    """Device-native tabular export boundary for geometry plus attributes."""

    attributes: NativeAttributeTable | pd.DataFrame
    geometry: GeometryNativeResult
    geometry_name: str
    column_order: tuple[str, ...]
    attrs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            NativeAttributeTable.from_value(self.attributes),
        )

    def to_geodataframe(self) -> GeoDataFrame:
        frame = _materialize_attribute_geometry_frame(
            self.attributes,
            self.geometry,
            geometry_name=self.geometry_name,
            crs=self.geometry.crs,
            column_order=self.column_order,
        )
        if self.attrs:
            frame.attrs.update(self.attrs)
        return frame

    def to_arrow(
        self,
        *,
        index: bool | None = None,
        geometry_encoding: str = "WKB",
        interleaved: bool = True,
        include_z: bool | None = None,
    ):
        from vibespatial.api.io._geoarrow import ArrowTable
        from vibespatial.io.geoarrow import native_tabular_to_arrow

        table, _geometry_encoding = native_tabular_to_arrow(
            self,
            index=index,
            geometry_encoding=geometry_encoding,
            interleaved=interleaved,
            include_z=include_z,
        )
        return ArrowTable(table)

    def to_parquet(
        self,
        path,
        *,
        index: bool | None = None,
        compression: str | None = "snappy",
        geometry_encoding: str = "WKB",
        write_covering_bbox: bool = False,
        schema_version: str | None = None,
        **kwargs,
    ) -> None:
        from vibespatial.io.geoparquet import write_geoparquet

        write_geoparquet(
            self,
            path,
            index=index,
            compression=compression,
            geometry_encoding=geometry_encoding,
            write_covering_bbox=write_covering_bbox,
            schema_version=schema_version,
            **kwargs,
        )

    def to_feather(
        self,
        path,
        *,
        index: bool | None = None,
        compression: str | None = None,
        schema_version: str | None = None,
        **kwargs,
    ) -> None:
        from vibespatial.api.io.arrow import _to_feather

        _to_feather(
            self,
            path,
            index=index,
            compression=compression,
            schema_version=schema_version,
            **kwargs,
        )


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


def _materialize_attribute_geometry_frame(
    attributes: NativeAttributeTable | pd.DataFrame,
    geometry: GeometryNativeResult,
    *,
    geometry_name: str,
    crs,
    column_order: tuple[str, ...] | None = None,
):
    """Explicit host export for attribute tables plus native geometry."""
    from vibespatial.api.geodataframe import GeoDataFrame

    attributes = NativeAttributeTable.from_value(attributes)
    geom = geometry.to_geoseries(index=attributes.index, name=geometry_name)
    frame = attributes.to_pandas(copy=False)
    if column_order is None:
        ordered_columns = [*frame.columns, geometry_name]
    else:
        ordered_columns = list(column_order)

    placeholder_geometry = pd.Series(
        np.empty(len(attributes.index), dtype=object),
        index=attributes.index,
        copy=False,
        name=geometry_name,
    )
    data_columns = {
        column_name: (
            placeholder_geometry if column_name == geometry_name else frame[column_name]
        )
        for column_name in ordered_columns
    }
    rebuilt = pd.DataFrame(data_columns, index=attributes.index, copy=False)
    rebuilt.__class__ = GeoDataFrame
    rebuilt._geometry_column_name = geometry_name
    rebuilt.attrs = frame.attrs.copy()
    return _replace_geometry_column_preserving_backing(
        rebuilt,
        geom.values,
        crs=crs,
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
    for frame in frames:
        if frame.shape[1] == 0:
            continue
        tables.append(pa.Table.from_pandas(frame, preserve_index=False))

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
    )


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
) -> tuple[NativeAttributeTable, GeometryNativeResult, str, Any, tuple[str, ...], np.ndarray | None]:
    """Build native relation-join export parts without constructing a joined DataFrame."""
    left_geometry_name = left_df.geometry.name
    left_crs = left_df.crs
    right_geometry_name = right_df.geometry.name
    right_crs = right_df.crs

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
        geometry_name = left_geometry_name
        crs = left_crs
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

    attributes = _native_attribute_table_from_projected_frames(
        [left_projected, right_projected],
        index_override=joined_index,
        storage=attribute_storage,
    )
    column_order = tuple([*attributes.columns, geometry_name])
    return attributes, geometry_result, geometry_name, crs, column_order, distances


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
        geometry,
        geometry_name=geometry_name,
        crs=crs,
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


@dataclass(frozen=True)
class SymmetricDifferenceConstructiveResult:
    """Deferred symmetric-difference export that preserves GeoPandas semantics."""

    left_result: LeftConstructiveResult
    right_result: LeftConstructiveResult
    left_df: GeoDataFrame
    right_df: GeoDataFrame
    geometry_name: str = "geometry"
    frame_type: type | None = None
    crs: Any | None = None

    def to_geodataframe(self) -> GeoDataFrame:
        from vibespatial.api.geodataframe import GeoDataFrame

        dfdiff1 = self.left_result.to_geodataframe(self.left_df).copy(deep=False)
        dfdiff2 = self.right_result.to_geodataframe(self.right_df).copy(deep=False)
        dfdiff1["__idx1"] = range(len(dfdiff1))
        dfdiff2["__idx2"] = range(len(dfdiff2))
        dfdiff1["__idx2"] = np.nan
        dfdiff2["__idx1"] = np.nan
        dfdiff1 = _set_active_geometry_name(dfdiff1, self.geometry_name)
        dfdiff2 = _set_active_geometry_name(dfdiff2, self.geometry_name)

        diff1_owned = getattr(dfdiff1.geometry.values, "_owned", None)
        diff2_owned = getattr(dfdiff2.geometry.values, "_owned", None)
        base_crs = self.crs if self.crs is not None else dfdiff1.crs

        if diff1_owned is not None and diff2_owned is not None:
            if dfdiff1.crs != dfdiff2.crs:
                dfdiff2 = dfdiff2.set_crs(dfdiff1.crs, allow_override=True)

            skip = {self.geometry_name, "__idx1", "__idx2"}
            attr1 = {column for column in dfdiff1.columns if column not in skip}
            attr2 = {column for column in dfdiff2.columns if column not in skip}
            shared = attr1 & attr2
            rename1 = {column: f"{column}_1" for column in shared}
            rename2 = {column: f"{column}_2" for column in shared}
            if rename1:
                dfdiff1 = dfdiff1.rename(columns=rename1)
            if rename2:
                dfdiff2 = dfdiff2.rename(columns=rename2)

            result = pd.concat([dfdiff1, dfdiff2], ignore_index=True, sort=False)
            columns = [column for column in result.columns if column != self.geometry_name]
            columns.append(self.geometry_name)
            result = result.reindex(columns=columns)
            frame_type = self.frame_type or type(dfdiff1)
            if not isinstance(result, frame_type):
                result = frame_type(result, geometry=self.geometry_name, crs=base_crs)
            elif result.crs is None and base_crs is not None:
                result = result.set_crs(base_crs, allow_override=True)
            return result

        dfsym = dfdiff1.merge(
            dfdiff2,
            on=["__idx1", "__idx2"],
            how="outer",
            suffixes=("_1", "_2"),
        )
        geometry_1 = np.asarray(dfsym[f"{self.geometry_name}_1"], dtype=object)
        geometry_2 = np.asarray(dfsym[f"{self.geometry_name}_2"], dtype=object)
        mask = pd.isna(geometry_1)
        combined_geometry = geometry_1.copy()
        combined_geometry[mask] = geometry_2[mask]
        dfsym.drop(
            [f"{self.geometry_name}_1", f"{self.geometry_name}_2"],
            axis=1,
            inplace=True,
        )
        dfsym.reset_index(drop=True, inplace=True)
        frame_type = self.frame_type or GeoDataFrame
        return frame_type(
            dfsym,
            geometry=pd.Series(combined_geometry, name=self.geometry_name),
            crs=base_crs,
        )


@dataclass(frozen=True)
class PairwiseConstructiveFragment:
    """Deferred export fragment for a pairwise constructive native result."""

    result: PairwiseConstructiveResult
    left_df: GeoDataFrame
    right_df: GeoDataFrame
    attribute_assembler: Callable[[Any, Any, pd.DataFrame, pd.DataFrame], pd.DataFrame]
    rename_columns: dict[str, str] | None = None
    assign_columns: dict[str, Any] | None = None
    geometry_name: str = "geometry"
    frame_type: type | None = None
    prefer_native_attribute_projection: bool = False

    def to_geodataframe(self) -> GeoDataFrame:
        if self.prefer_native_attribute_projection:
            left_idx, right_idx = self.result.relation.to_host()
            attributes = _project_pairwise_attribute_frame(
                self.left_df,
                self.right_df,
                left_idx,
                right_idx,
                lsuffix="1",
                rsuffix="2",
            )
            frame_attrs: dict[str, Any] = {}
            if self.result.keep_geom_type_applied:
                frame_attrs["_vibespatial_keep_geom_type_applied"] = True
            if self.rename_columns:
                attributes = attributes.rename(columns=self.rename_columns, copy=False)
            trailing_columns: list[str] = []
            if self.assign_columns:
                attributes = attributes.copy(deep=False)
                for column, values in self.assign_columns.items():
                    attributes[column] = values
                trailing_columns = list(self.assign_columns)
            trailing_columns = [
                column for column in trailing_columns if column in attributes.columns
            ]
            base_columns = [
                column for column in attributes.columns if column not in trailing_columns
            ]
            frame = _materialize_attribute_geometry_frame(
                attributes,
                self.result.geometry,
                geometry_name=self.geometry_name,
                crs=self.result.geometry.crs,
                column_order=tuple([*base_columns, self.geometry_name, *trailing_columns]),
            )
            if frame_attrs:
                frame.attrs.update(frame_attrs)
        else:
            frame = self.result.to_geodataframe(
                self.left_df,
                self.right_df,
                attribute_assembler=self.attribute_assembler,
            )
            if self.rename_columns:
                frame = frame.rename(columns=self.rename_columns)
            if self.assign_columns:
                frame = frame.copy(deep=False)
                for column, values in self.assign_columns.items():
                    frame[column] = values
        frame = _set_active_geometry_name(frame, self.geometry_name)
        frame_type = self.frame_type or type(frame)
        if not isinstance(frame, frame_type):
            frame = frame_type(frame, geometry=self.geometry_name, crs=frame.crs)
        return frame


@dataclass(frozen=True)
class LeftConstructiveFragment:
    """Deferred export fragment for a left-row-preserving constructive result."""

    result: LeftConstructiveResult
    df: GeoDataFrame
    rename_columns: dict[str, str] | None = None
    assign_columns: dict[str, Any] | None = None
    geometry_name: str = "geometry"
    frame_type: type | None = None

    def to_geodataframe(self) -> GeoDataFrame:
        frame = self.result.to_geodataframe(self.df)
        if self.rename_columns:
            frame = frame.rename(columns=self.rename_columns)
        if self.assign_columns:
            frame = frame.copy(deep=False)
            for column, values in self.assign_columns.items():
                frame[column] = values
        frame = _set_active_geometry_name(frame, self.geometry_name)
        frame_type = self.frame_type or type(frame)
        if not isinstance(frame, frame_type):
            frame = frame_type(frame, geometry=self.geometry_name, crs=frame.crs)
        return frame


@dataclass(frozen=True)
class ConcatConstructiveResult:
    """Concatenate deferred constructive fragments and export once at the boundary."""

    parts: tuple[Any, ...]
    geometry_name: str = "geometry"
    frame_type: type | None = None
    crs: Any | None = None
    result_attrs: dict[str, Any] | None = None

    def to_geodataframe(self) -> GeoDataFrame:
        from vibespatial.api.geodataframe import GeoDataFrame

        frames = [part.to_geodataframe() for part in self.parts]
        if not frames:
            frame_type = self.frame_type or GeoDataFrame
            result = frame_type(geometry=self.geometry_name)
            if self.crs is not None:
                result = result.set_crs(self.crs, allow_override=True)
            if self.result_attrs:
                result.attrs.update(self.result_attrs)
            return result

        base_crs = self.crs if self.crs is not None else frames[0].crs
        aligned_frames: list[GeoDataFrame] = []
        for frame in frames:
            frame = _set_active_geometry_name(frame, self.geometry_name)
            if base_crs is not None and frame.crs != base_crs:
                frame = frame.set_crs(base_crs, allow_override=True)
            aligned_frames.append(frame)

        result = pd.concat(aligned_frames, ignore_index=True, sort=False)
        columns = [column for column in result.columns if column != self.geometry_name]
        columns.append(self.geometry_name)
        result = result.reindex(columns=columns)
        frame_type = self.frame_type or type(aligned_frames[0])
        if not isinstance(result, frame_type):
            result = frame_type(result, geometry=self.geometry_name, crs=base_crs)
        elif result.crs is None and base_crs is not None:
            result = result.set_crs(base_crs, allow_override=True)
        merged_attrs: dict[str, Any] = {}
        for frame in aligned_frames:
            merged_attrs.update(frame.attrs)
        if self.result_attrs:
            merged_attrs.update(self.result_attrs)
        if merged_attrs:
            result.attrs.update(merged_attrs)
        return result


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
        geometry_frame = frame_type(
            {
                self.geometry_name: self.geometry.to_geoseries(
                    index=self.attributes.index,
                    name=self.geometry_name,
                )
            },
            geometry=self.geometry_name,
            index=self.attributes.index,
            crs=self.geometry.crs,
        )
        aggregated = geometry_frame.join(self.attributes)
        if not self.as_index:
            aggregated = aggregated.reset_index()
        if not isinstance(aggregated, frame_type):
            aggregated = frame_type(
                aggregated,
                geometry=self.geometry_name,
                crs=self.geometry.crs,
            )
        return aggregated


def _spatial_to_native_tabular_result(spatial) -> NativeTabularResult:
    geometry_name = getattr(spatial, "_geometry_column_name", None)
    if geometry_name is not None:
        geometry = GeometryNativeResult.from_values(
            spatial[geometry_name].values,
            crs=spatial.crs,
            index=spatial.index,
            name=geometry_name,
        )
        return NativeTabularResult(
            attributes=spatial.drop(columns=[geometry_name]).copy(deep=False),
            geometry=geometry,
            geometry_name=geometry_name,
            column_order=tuple(spatial.columns),
            attrs=spatial.attrs.copy() or None,
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
    attributes = result.attributes.copy(deep=False)
    leading_columns: list[str] = []
    trailing_columns = list(attributes.columns)
    if not result.as_index:
        attributes = attributes.reset_index()
        leading_count = len(attributes.columns) - len(trailing_columns)
        leading_columns = list(attributes.columns[:leading_count])
        trailing_columns = list(attributes.columns[leading_count:])
    return NativeTabularResult(
        attributes=attributes,
        geometry=result.geometry,
        geometry_name=result.geometry_name,
        column_order=tuple([*leading_columns, result.geometry_name, *trailing_columns]),
    )


def _pairwise_constructive_fragment_to_native_tabular_result(
    fragment: PairwiseConstructiveFragment,
) -> NativeTabularResult:
    left_idx, right_idx = fragment.result.relation.to_host()
    attrs = _native_pairwise_attribute_table(
        fragment.left_df,
        fragment.right_df,
        left_idx,
        right_idx,
        lsuffix="1",
        rsuffix="2",
    )
    frame_attrs: dict[str, Any] = {}
    if fragment.result.keep_geom_type_applied:
        frame_attrs["_vibespatial_keep_geom_type_applied"] = True
    if fragment.rename_columns:
        attrs = attrs.rename_columns(fragment.rename_columns)
    trailing_columns: list[str] = []
    if fragment.assign_columns:
        for column, values in fragment.assign_columns.items():
            attrs = attrs.with_column(column, values)
        trailing_columns = list(fragment.assign_columns)
    trailing_columns = [column for column in trailing_columns if column in attrs.columns]
    base_columns = [column for column in attrs.columns if column not in trailing_columns]
    return NativeTabularResult(
        attributes=attrs,
        geometry=fragment.result.geometry,
        geometry_name=fragment.geometry_name,
        column_order=tuple([*base_columns, fragment.geometry_name, *trailing_columns]),
        attrs=frame_attrs or None,
    )


def _left_constructive_fragment_to_native_tabular_result(
    fragment: LeftConstructiveFragment,
) -> NativeTabularResult:
    row_positions = np.asarray(fragment.result.row_positions, dtype=np.intp)
    source_attrs = fragment.df.drop(fragment.df._geometry_column_name, axis=1).copy(deep=False)
    projected = source_attrs._reindex_with_indexers(
        {0: (pd.RangeIndex(len(row_positions)), row_positions)}
    )
    attrs = _native_attribute_table_from_projected_frames(
        [projected],
        index_override=source_attrs.index.take(row_positions),
    )
    if fragment.rename_columns:
        attrs = attrs.rename_columns(fragment.rename_columns)
    trailing_columns: list[str] = []
    if fragment.assign_columns:
        for column, values in fragment.assign_columns.items():
            attrs = attrs.with_column(column, values)
        trailing_columns = list(fragment.assign_columns)
    trailing_columns = [column for column in trailing_columns if column in attrs.columns]
    base_columns = [column for column in attrs.columns if column not in trailing_columns]
    return NativeTabularResult(
        attributes=attrs,
        geometry=fragment.result.geometry,
        geometry_name=fragment.geometry_name,
        column_order=tuple([*base_columns, fragment.geometry_name, *trailing_columns]),
        attrs=fragment.df.attrs.copy() or None,
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
            keep[polygon_mask] = np.asarray(
                geometry_array.area[polygon_mask],
                dtype=np.float64,
            ) > 0.0
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

    return GeometryNativeResult.from_owned(combined_owned, crs=crs), row_positions


def _clip_native_result_to_native_tabular_result(result) -> NativeTabularResult:
    source = result.source

    if not result.parts:
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
            column_order=tuple([*attributes.columns, geometry_name]),
            attrs=source.attrs.copy() or None,
        )

    if len(result.parts) == 1 and not result.has_non_point_candidates and not result.keep_geom_type:
        part = result.parts[0]
        row_positions = np.asarray(part.row_positions, dtype=np.intp)
        if hasattr(source, "_geometry_column_name"):
            geometry_name = source._geometry_column_name
            source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
            projected = source_attrs._reindex_with_indexers(
                {0: (pd.RangeIndex(len(row_positions)), row_positions)}
            )
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
            column_order=tuple([*attributes.columns, geometry_name]),
            attrs=source.attrs.copy() or None,
        )

    if not result.keep_geom_type:
        geometry_name = (
            source._geometry_column_name
            if hasattr(source, "_geometry_column_name")
            else getattr(source, "name", None) or "geometry"
        )
        owned_geometry, owned_rows = _clip_owned_geometry_native_result(
            result,
            crs=source.crs,
        )
        if owned_geometry is not None and owned_rows is not None:
            if hasattr(source, "_geometry_column_name"):
                source_attrs = source.drop(columns=[geometry_name]).copy(deep=False)
                projected = source_attrs._reindex_with_indexers(
                    {0: (pd.RangeIndex(len(owned_rows)), owned_rows)}
                )
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
                column_order=tuple([*attributes.columns, geometry_name]),
                attrs=source.attrs.copy() or None,
            )

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
        for part in result.parts:
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
            ordered_rows = np.asarray(result.ordered_row_positions, dtype=np.intp)
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

        if not result.clipping_by_rectangle and result.has_non_point_candidates and geometry_values.size > 0:
            type_ids = np.asarray(shapely.get_type_id(geometry_values), dtype=np.int32)
            polygon_mask = (type_ids == 3) | (type_ids == 6)
            if np.any(polygon_mask):
                nonpositive_area = np.asarray(
                    shapely.area(geometry_values[polygon_mask]),
                    dtype=np.float64,
                ) <= 0.0
                if np.any(nonpositive_area):
                    keep = np.ones(len(geometry_values), dtype=bool)
                    keep[np.flatnonzero(polygon_mask)[nonpositive_area]] = False
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
            projected = source_attrs._reindex_with_indexers(
                {0: (pd.RangeIndex(len(row_positions)), row_positions)}
            )
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
            column_order=tuple([*attributes.columns, geometry_name]),
            attrs=source.attrs.copy() or None,
        )

    clipped = result._materialize_parts()
    clipped = result._normalize_geometry_backing(clipped)
    clipped = result._filter_result(clipped)
    clipped = result._apply_keep_geom_type(clipped)
    return _spatial_to_native_tabular_result(clipped)


def _relation_join_export_result_to_native_tabular_result(
    result: RelationJoinExportResult,
    *,
    attribute_storage: str = "arrow",
) -> NativeTabularResult:
    attributes, geometry, geometry_name, _crs, column_order, distances = _native_relation_join_parts(
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
    )


def _empty_geometry_native_result(*, geometry_name: str, crs) -> GeometryNativeResult:
    from vibespatial.api.geoseries import GeoSeries

    return GeometryNativeResult.from_geoseries(
        GeoSeries([], index=pd.RangeIndex(0), crs=crs, name=geometry_name),
    )


def _concat_geometry_native_results(
    results: list[NativeTabularResult],
    *,
    geometry_name: str,
    crs,
) -> GeometryNativeResult:
    if not results:
        return _empty_geometry_native_result(geometry_name=geometry_name, crs=crs)

    geometries = [result.geometry for result in results]
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
        result.geometry.to_geoseries(
            index=result.attributes.index,
            name=geometry_name,
        )
        for result in results
    ]
    combined = pd.concat(series_parts, ignore_index=True)
    return GeometryNativeResult.from_values(
        combined.values,
        crs=crs,
        index=combined.index,
        name=geometry_name,
    )


def _concat_native_tabular_results(
    results: list[NativeTabularResult],
    *,
    geometry_name: str,
    crs,
    attrs: dict[str, Any] | None = None,
) -> NativeTabularResult:
    if not results:
        return NativeTabularResult(
            attributes=pd.DataFrame(index=pd.RangeIndex(0)),
            geometry=_empty_geometry_native_result(geometry_name=geometry_name, crs=crs),
            geometry_name=geometry_name,
            column_order=(geometry_name,),
            attrs=attrs,
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
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=tuple([*attributes.columns, geometry_name]),
        attrs=merged_attrs or None,
    )


def _symmetric_difference_constructive_result_to_native_tabular_result(
    result: SymmetricDifferenceConstructiveResult,
) -> NativeTabularResult:
    left_result = _left_constructive_fragment_to_native_tabular_result(
        LeftConstructiveFragment(
            result=result.left_result,
            df=result.left_df,
            geometry_name=result.geometry_name,
            frame_type=result.frame_type,
        )
    )
    right_result = _left_constructive_fragment_to_native_tabular_result(
        LeftConstructiveFragment(
            result=result.right_result,
            df=result.right_df,
            geometry_name=result.geometry_name,
            frame_type=result.frame_type,
        )
    )

    left_attrs = left_result.attributes
    right_attrs = right_result.attributes
    shared = set(left_attrs.columns) & set(right_attrs.columns)
    rename1 = {column: f"{column}_1" for column in shared}
    rename2 = {column: f"{column}_2" for column in shared}
    if rename1:
        left_attrs = left_attrs.rename_columns(rename1)
    if rename2:
        right_attrs = right_attrs.rename_columns(rename2)

    left_result = NativeTabularResult(
        attributes=left_attrs,
        geometry=left_result.geometry,
        geometry_name=result.geometry_name,
        column_order=tuple([*left_attrs.columns, result.geometry_name]),
        attrs=left_result.attrs,
    )
    right_result = NativeTabularResult(
        attributes=right_attrs,
        geometry=right_result.geometry,
        geometry_name=result.geometry_name,
        column_order=tuple([*right_attrs.columns, result.geometry_name]),
        attrs=right_result.attrs,
    )

    base_crs = result.crs if result.crs is not None else left_result.geometry.crs
    return _concat_native_tabular_results(
        [left_result, right_result],
        geometry_name=result.geometry_name,
        crs=base_crs,
    )


def _constructive_result_to_native_tabular_result(result) -> NativeTabularResult:
    if isinstance(result, PairwiseConstructiveFragment):
        return _pairwise_constructive_fragment_to_native_tabular_result(result)
    if isinstance(result, LeftConstructiveFragment):
        return _left_constructive_fragment_to_native_tabular_result(result)
    if isinstance(result, SymmetricDifferenceConstructiveResult):
        return _symmetric_difference_constructive_result_to_native_tabular_result(result)
    if isinstance(result, ConcatConstructiveResult):
        return _concat_constructive_result_to_native_tabular_result(result)
    return _spatial_to_native_tabular_result(result.to_geodataframe())


def _concat_constructive_result_to_native_tabular_result(
    result: ConcatConstructiveResult,
) -> NativeTabularResult:
    results = [_constructive_result_to_native_tabular_result(part) for part in result.parts]
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
    if isinstance(result, PairwiseConstructiveFragment):
        return _pairwise_constructive_fragment_to_native_tabular_result(result)
    if isinstance(result, LeftConstructiveFragment):
        return _left_constructive_fragment_to_native_tabular_result(result)
    if isinstance(result, SymmetricDifferenceConstructiveResult):
        return _symmetric_difference_constructive_result_to_native_tabular_result(result)
    if isinstance(result, ConcatConstructiveResult):
        return _concat_constructive_result_to_native_tabular_result(result)
    if isinstance(result, RelationJoinExportResult):
        return _relation_join_export_result_to_native_tabular_result(result)
    return None
