from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    record_materialization_event,
)

if TYPE_CHECKING:
    from vibespatial.api.geodataframe import GeoDataFrame
    from vibespatial.api.geoseries import GeoSeries


def _host_array(values: Any, *, dtype) -> np.ndarray:
    if hasattr(values, "get"):
        record_materialization_event(
            surface="vibespatial.api._native_result_core._host_array",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="array_to_host",
            reason="device-like array exposed a get() host materialization path",
            d2h_transfer=True,
        )
        host_values = values.get()
    else:
        host_values = values
    return np.asarray(host_values, dtype=dtype)


def _normalize_row_selection(row_positions):
    if hasattr(row_positions, "__cuda_array_interface__"):
        import cupy as cp

        d_positions = cp.asarray(row_positions)
        if d_positions.dtype == cp.bool_ or d_positions.dtype == bool:
            return cp.flatnonzero(d_positions).astype(cp.int64, copy=False)
        return d_positions.astype(cp.int64, copy=False)

    positions = np.asarray(row_positions)
    if positions.dtype == bool:
        positions = np.flatnonzero(positions)
    return np.asarray(positions, dtype=np.int64)


def _host_row_positions(row_positions) -> np.ndarray:
    normalized = _normalize_row_selection(row_positions)
    if hasattr(normalized, "__cuda_array_interface__"):
        import cupy as cp

        item_count = int(getattr(normalized, "size", len(normalized)))
        itemsize = int(getattr(getattr(normalized, "dtype", None), "itemsize", 0))
        record_materialization_event(
            surface="vibespatial.api._native_result_core._host_row_positions",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="row_positions_to_host",
            reason="device row positions were normalized on host",
            detail=f"rows={item_count}, bytes={item_count * itemsize}",
            d2h_transfer=True,
        )
        return cp.asnumpy(normalized)
    return np.asarray(normalized, dtype=np.int64)


def _attribute_storage_label(attributes: NativeAttributeTable | pd.DataFrame) -> str:
    table = NativeAttributeTable.from_value(attributes)
    if table.dataframe is not None:
        return "pandas"
    if table.arrow_table is not None:
        return "arrow"
    if table.device_table is not None:
        return "device"
    if table.loader is not None:
        return "loader"
    return "unknown"


def _geometry_storage_label(geometry: GeometryNativeResult) -> str:
    owned = getattr(geometry, "owned", None)
    if owned is not None:
        residency = getattr(getattr(owned, "residency", None), "value", None)
        return f"owned:{residency or 'unknown'}"
    return "geoseries"


def _device_table_row_count(table: Any) -> int:
    num_rows = getattr(table, "num_rows", None)
    if callable(num_rows):
        return int(num_rows())
    if num_rows is not None:
        return int(num_rows)
    shape = getattr(table, "shape", None)
    if shape is not None:
        return int(shape[0])
    raise TypeError("device attribute table does not expose a row count")


def _rename_device_arrow_table(table: Any, column_override, *, schema) -> Any:
    import pyarrow as pa

    names = [str(name) for name in (column_override or table.column_names)]
    if schema is None:
        return pa.Table.from_arrays(
            [table.column(index) for index in range(table.num_columns)],
            names=names,
            metadata=table.schema.metadata,
        )
    fields = []
    for index, name in enumerate(names):
        try:
            fields.append(schema.field(name))
        except KeyError:
            fields.append(pa.field(name, table.column(index).type))
    return pa.Table.from_arrays(
        [table.column(index) for index in range(table.num_columns)],
        schema=pa.schema(fields, metadata=schema.metadata),
    )


def _rename_schema_fields(schema: Any, old_names, new_names) -> Any:
    if schema is None:
        return None

    import pyarrow as pa

    fields = []
    for index, (old_name, new_name) in enumerate(
        zip(tuple(old_names), tuple(new_names), strict=True)
    ):
        try:
            field = schema.field(str(old_name))
        except KeyError:
            field = schema.field(index)
        fields.append(
            pa.field(
                str(new_name),
                field.type,
                nullable=field.nullable,
                metadata=field.metadata,
            )
        )
    return pa.schema(fields, metadata=schema.metadata)


def _renamed_pandas_columns_like(columns: pd.Index, renamed_logical) -> pd.Index:
    if isinstance(columns, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(renamed_logical, names=columns.names)
    return pd.Index(renamed_logical, name=columns.name)


def _is_admissible_pandas_numeric_series(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        not series.hasnans
        and (
            pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
        )
    )


def _is_admissible_arrow_numeric_type(dtype) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_integer(dtype)
        or pa.types.is_floating(dtype)
        or pa.types.is_boolean(dtype)
    )


def _column_position_map(columns) -> dict[Any, int] | None:
    names = tuple(columns)
    if len(set(names)) != len(names):
        return None
    return {name: index for index, name in enumerate(names)}


def _pylibcudf_numeric_column_view(column):
    if column.offset() != 0 or column.null_count() != 0:
        return None
    arrow_type = column.type().to_arrow()
    if not _is_admissible_arrow_numeric_type(arrow_type):
        return None

    import cupy as cp

    dtype = np.dtype(column.type().typestr)
    return cp.asarray(column.data()).view(dtype)[: int(column.size())]


@dataclass(frozen=True)
class NativeAttributeTable:
    """Attribute payload that can stay columnar without requiring pandas storage."""

    dataframe: pd.DataFrame | None = None
    arrow_table: Any | None = None
    device_table: Any | None = None
    loader: Callable[[], pd.DataFrame] | None = None
    index_override: pd.Index | None = None
    column_override: tuple[Any, ...] | None = None
    schema_override: Any | None = None
    to_pandas_kwargs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        provided = sum(
            value is not None
            for value in (
                self.dataframe,
                self.arrow_table,
                self.device_table,
                self.loader,
            )
        )
        if provided != 1:
            raise ValueError(
                "NativeAttributeTable requires exactly one of dataframe, arrow_table, "
                "device_table, or loader"
            )
        if self.to_pandas_kwargs is None:
            object.__setattr__(self, "to_pandas_kwargs", {})
        if self.arrow_table is not None and self.index_override is None:
            object.__setattr__(
                self,
                "index_override",
                pd.RangeIndex(int(self.arrow_table.num_rows)),
            )
        if self.arrow_table is not None and self.column_override is None:
            object.__setattr__(self, "column_override", tuple(self.arrow_table.column_names))
        if self.arrow_table is not None and self.schema_override is None:
            object.__setattr__(self, "schema_override", self.arrow_table.schema)
        if self.device_table is not None:
            row_count = _device_table_row_count(self.device_table)
            if self.index_override is None:
                object.__setattr__(self, "index_override", pd.RangeIndex(row_count))
            if self.column_override is None:
                schema = self.schema_override
                if schema is None:
                    raise ValueError(
                        "NativeAttributeTable device_table requires column_override "
                        "or schema_override"
                    )
                object.__setattr__(self, "column_override", tuple(schema.names))
        if self.loader is not None:
            if self.index_override is None:
                raise ValueError("NativeAttributeTable loader requires index_override")
            if self.column_override is None:
                object.__setattr__(self, "column_override", tuple())

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

    @classmethod
    def from_loader(
        cls,
        loader: Callable[[], pd.DataFrame],
        *,
        index_override: pd.Index,
        columns: tuple[Any, ...] | list[Any] | None = None,
        to_pandas_kwargs: dict[str, Any] | None = None,
    ) -> NativeAttributeTable:
        return cls(
            loader=loader,
            index_override=index_override,
            column_override=None if columns is None else tuple(columns),
            to_pandas_kwargs=to_pandas_kwargs,
        )

    @property
    def index(self) -> pd.Index:
        if self.dataframe is not None:
            return self.dataframe.index
        return self.index_override

    @property
    def columns(self) -> pd.Index:
        if self.dataframe is not None:
            return self.dataframe.columns
        if self.loader is not None:
            return pd.Index(self.column_override)
        return pd.Index(self.column_override)

    def _materialize_loaded_frame(self) -> pd.DataFrame:
        if self.loader is None:
            if self.dataframe is None:
                raise ValueError("loader materialization requires a loader-backed table")
            return self.dataframe

        frame = self.loader()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "NativeAttributeTable loader must return a pandas DataFrame"
            )
        if not frame.index.equals(self.index):
            if len(frame) != len(self.index):
                raise ValueError(
                    "NativeAttributeTable loader returned a DataFrame with the wrong row count"
                )
            frame = frame.copy(deep=False)
            frame.index = self.index
        if self.column_override:
            expected = tuple(self.column_override)
            actual = tuple(frame.columns)
            if actual != expected:
                raise ValueError(
                    "NativeAttributeTable loader returned columns that do not match the declared schema"
                )
        object.__setattr__(self, "dataframe", frame)
        object.__setattr__(self, "loader", None)
        object.__setattr__(self, "column_override", tuple(frame.columns))
        return frame

    def numeric_column_arrays(self, columns) -> dict[Any, Any] | None:
        """Return all-valid numeric columns without crossing an export boundary."""
        requested = tuple(dict.fromkeys(columns))
        column_positions = _column_position_map(self.columns)
        if column_positions is None or any(
            column not in column_positions for column in requested
        ):
            return None
        if self.loader is not None:
            return None

        if self.dataframe is not None:
            out: dict[Any, Any] = {}
            for column in requested:
                series = self.dataframe[column]
                if not isinstance(series, pd.Series):
                    return None
                if not _is_admissible_pandas_numeric_series(series):
                    return None
                values = series.to_numpy(copy=False)
                if not np.issubdtype(values.dtype, np.number) and not np.issubdtype(
                    values.dtype,
                    np.bool_,
                ):
                    return None
                out[column] = values
            return out

        if self.arrow_table is not None:
            arrow = self.to_arrow(index=False, columns=requested)
            out = {}
            for logical_name, physical_name in zip(
                requested,
                arrow.column_names,
                strict=True,
            ):
                chunked = arrow.column(physical_name)
                if chunked.null_count or not _is_admissible_arrow_numeric_type(
                    chunked.type
                ):
                    return None
                out[logical_name] = chunked.combine_chunks().to_numpy(
                    zero_copy_only=False
                )
            return out

        if self.device_table is not None:
            source_columns = self.device_table.columns()
            out = {}
            for column in requested:
                values = _pylibcudf_numeric_column_view(
                    source_columns[column_positions[column]]
                )
                if values is None:
                    return None
                out[column] = values
            return out

        return None

    def host_column_series(self, columns) -> dict[Any, pd.Series] | None:
        """Return host pandas columns without crossing an export boundary."""
        requested = tuple(dict.fromkeys(columns))
        column_positions = _column_position_map(self.columns)
        if column_positions is None or any(
            column not in column_positions for column in requested
        ):
            return None
        if self.loader is not None or self.dataframe is None:
            return None

        out: dict[Any, pd.Series] = {}
        for column in requested:
            series = self.dataframe[column]
            if not isinstance(series, pd.Series):
                return None
            out[column] = series
        return out

    def to_pandas(self, *, copy: bool = False, **kwargs) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy(deep=copy) if copy else self.dataframe
        if self.loader is not None:
            frame = self._materialize_loaded_frame()
            return frame.copy(deep=copy) if copy else frame
        if self.device_table is not None:
            frame = self.to_arrow(index=False).to_pandas(
                **{**(self.to_pandas_kwargs or {}), **kwargs}
            )
            frame.index = self.index
            if self.column_override is not None:
                frame.columns = pd.Index(self.column_override)
            return frame.copy(deep=copy) if copy else frame

        to_pandas_kwargs = dict(self.to_pandas_kwargs or {})
        to_pandas_kwargs.update(kwargs)
        frame = self.arrow_table.to_pandas(**to_pandas_kwargs)
        frame.index = self.index
        if self.column_override is not None:
            frame.columns = pd.Index(self.column_override)
        return frame.copy(deep=copy) if copy else frame

    def to_arrow(self, *, index: bool | None = None, columns=None):
        import pyarrow as pa

        if self.loader is not None:
            frame = self.to_pandas(copy=False)
            requested_columns = None if columns is None else list(columns)
            if requested_columns is not None:
                frame = frame.loc[:, requested_columns]
            return pa.Table.from_pandas(frame, preserve_index=index)

        requested_columns = None if columns is None else list(columns)
        if self.device_table is not None:
            if index not in (None, False):
                frame = self.to_pandas(copy=False)
                if requested_columns is not None:
                    frame = frame.loc[:, requested_columns]
                return pa.Table.from_pandas(frame, preserve_index=index)
            record_materialization_event(
                surface="vibespatial.api.NativeAttributeTable.to_arrow",
                boundary=MaterializationBoundary.USER_EXPORT,
                operation="device_attributes_to_arrow",
                reason="device attribute table exported to host Arrow",
                detail=(
                    f"rows={len(self)}, columns={len(self.columns)}, "
                    f"bytes=unknown"
                ),
                d2h_transfer=True,
            )
            table = self.device_table.to_arrow()
            table = _rename_device_arrow_table(
                table,
                self.column_override,
                schema=self.schema_override,
            )
            if requested_columns is not None:
                table = table.select([str(column) for column in requested_columns])
            return table
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

    def to_pylibcudf_columns(self, columns) -> list[Any]:
        import pylibcudf as plc

        requested_columns = list(columns)
        if self.device_table is None:
            table = plc.Table.from_arrow(
                self.to_arrow(index=False, columns=requested_columns)
            )
            return table.columns()

        source_columns = self.device_table.columns()
        by_name = {
            column_name: source_columns[index]
            for index, column_name in enumerate(self.column_override or ())
        }
        return [by_name[column] for column in requested_columns]

    def arrow_schema_for_columns(self, columns):
        import pyarrow as pa

        requested_columns = [str(column) for column in columns]
        schema = self.schema_override
        if schema is None and self.arrow_table is not None:
            schema = self.arrow_table.schema
        if schema is None:
            return pa.schema([pa.field(column, pa.null()) for column in requested_columns])
        fields = []
        for column in requested_columns:
            try:
                fields.append(schema.field(column))
            except KeyError:
                fields.append(pa.field(column, pa.null()))
        return pa.schema(fields, metadata=schema.metadata)

    def with_column(self, name: str, values) -> NativeAttributeTable:
        if self.loader is not None:
            declared_columns = tuple(self.column_override or ())
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).copy(deep=False)
                frame[name] = values
                return frame

            return type(self).from_loader(
                _load,
                index_override=self.index,
                columns=tuple([*declared_columns, name]),
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        if self.arrow_table is not None:
            import pyarrow as pa

            logical_columns = tuple([*(self.column_override or ()), name])
            table = self.arrow_table.append_column(str(name), pa.array(values))
            return type(self)(
                arrow_table=table,
                index_override=self.index,
                column_override=logical_columns,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        frame = self.dataframe.copy(deep=False)
        frame[name] = values
        return type(self)(dataframe=frame)

    def assign_columns(
        self,
        values_by_name: dict[Any, Any],
        *,
        columns: tuple[Any, ...],
    ) -> NativeAttributeTable | None:
        """Return attributes with assigned columns and exact logical order."""
        requested = tuple(columns)
        if len(set(requested)) != len(requested):
            return None
        assigned = {
            name: values
            for name, values in values_by_name.items()
            if name in requested
        }
        known = set(self.columns)
        if any(name not in known and name not in assigned for name in requested):
            return None
        if not assigned and requested == tuple(self.columns):
            return self

        if self.loader is not None:
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).copy(deep=False)
                for name, values in assigned.items():
                    frame[name] = values
                return frame.loc[:, list(requested)]

            return type(self).from_loader(
                _load,
                index_override=self.index,
                columns=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        if self.arrow_table is not None:
            import pyarrow as pa

            source = {
                name: self.arrow_table.column(position)
                for position, name in enumerate(self.column_override or ())
            }
            arrays = []
            for name in requested:
                if name in assigned:
                    arrays.append(pa.array(assigned[name]))
                elif name in source:
                    arrays.append(source[name])
                else:
                    return None
            return type(self)(
                arrow_table=pa.Table.from_arrays(
                    arrays,
                    names=[str(name) for name in requested],
                    metadata=self.arrow_table.schema.metadata,
                ),
                index_override=self.index,
                column_override=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        if self.device_table is not None:
            return None

        frame = self.dataframe.copy(deep=False)
        for name, values in assigned.items():
            frame[name] = values
        return type(self)(dataframe=frame.loc[:, list(requested)])

    def with_index(self, index: pd.Index) -> NativeAttributeTable:
        """Return the same attribute payload with a compatibility index."""
        if self.index.equals(index):
            return self
        if self.dataframe is not None:
            frame = self.dataframe.copy(deep=False)
            frame.index = index
            return type(self)(dataframe=frame)
        if self.loader is not None:
            return type(self).from_loader(
                self.loader,
                index_override=index,
                columns=self.column_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        return type(self)(
            arrow_table=self.arrow_table,
            device_table=self.device_table,
            index_override=index,
            column_override=self.column_override,
            schema_override=self.schema_override,
            to_pandas_kwargs=self.to_pandas_kwargs,
        )

    def reset_index_deferred(
        self,
    ) -> tuple[NativeAttributeTable, tuple[Any, ...], tuple[Any, ...]]:
        """Return ``reset_index()`` attributes without forcing eager export.

        The zero-row prototype preserves pandas' column naming and conflict
        checks while avoiding materializing full grouped reducer payloads.
        """
        prototype = pd.DataFrame(columns=self.columns, index=self.index[:0])
        reset_columns = tuple(prototype.reset_index().columns)
        leading_count = len(reset_columns) - len(self.columns)
        leading_columns = reset_columns[:leading_count]
        trailing_columns = reset_columns[leading_count:]

        if self.dataframe is not None:
            return (
                type(self)(dataframe=self.dataframe.reset_index()),
                leading_columns,
                trailing_columns,
            )

        parent = self

        def _load() -> pd.DataFrame:
            return parent.to_pandas(copy=False).reset_index()

        return (
            type(self).from_loader(
                _load,
                index_override=pd.RangeIndex(len(self)),
                columns=reset_columns,
                to_pandas_kwargs=self.to_pandas_kwargs,
            ),
            leading_columns,
            trailing_columns,
        )

    def rename_columns(self, mapping: dict[Any, Any]) -> NativeAttributeTable:
        if not mapping:
            return self
        renamed_logical = tuple(mapping.get(name, name) for name in self.columns)
        if self.loader is not None:
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).copy(deep=False)
                frame.columns = _renamed_pandas_columns_like(
                    parent.columns,
                    renamed_logical,
                )
                return frame

            return type(self).from_loader(
                _load,
                index_override=self.index,
                columns=renamed_logical,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        if self.arrow_table is not None:
            return type(self)(
                arrow_table=self.arrow_table.rename_columns([str(name) for name in renamed_logical]),
                index_override=self.index,
                column_override=renamed_logical,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        if self.device_table is not None:
            return type(self)(
                device_table=self.device_table,
                index_override=self.index,
                column_override=renamed_logical,
                schema_override=_rename_schema_fields(
                    self.schema_override,
                    self.columns,
                    renamed_logical,
                ),
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        frame = self.dataframe.copy(deep=False)
        frame.columns = _renamed_pandas_columns_like(
            self.dataframe.columns,
            renamed_logical,
        )
        return type(self)(dataframe=frame)

    def take(self, row_positions, *, preserve_index: bool = True) -> NativeAttributeTable:
        normalized = _normalize_row_selection(row_positions)
        if hasattr(normalized, "__cuda_array_interface__"):
            device_taken = self._device_take(normalized, preserve_index=preserve_index)
            if device_taken is not None:
                return device_taken
        host_positions = _host_row_positions(normalized)
        index_override = (
            self.index.take(host_positions)
            if preserve_index
            else pd.RangeIndex(int(host_positions.size))
        )
        if self.loader is not None:
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).take(host_positions)
                if not preserve_index:
                    frame = frame.copy(deep=False)
                    frame.index = index_override
                return frame

            return type(self).from_loader(
                _load,
                index_override=index_override,
                columns=self.column_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        if self.arrow_table is not None:
            import pyarrow as pa

            return type(self)(
                arrow_table=self.arrow_table.take(pa.array(host_positions, type=pa.int64())),
                index_override=index_override,
                column_override=self.column_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        frame = self.to_pandas(copy=False).take(host_positions)
        if not preserve_index:
            frame = frame.copy(deep=False)
            frame.index = index_override
        return type(self)(dataframe=frame)

    def _device_take(
        self,
        row_positions,
        *,
        preserve_index: bool,
    ) -> NativeAttributeTable | None:
        if preserve_index:
            return None
        try:
            import cupy as cp
            import pylibcudf as plc
        except ModuleNotFoundError:
            return None
        if self.device_table is not None:
            source = self.device_table
            schema = self.schema_override
        elif self.arrow_table is not None:
            source = plc.Table.from_arrow(self.to_arrow(index=False))
            schema = self.arrow_table.schema
        else:
            return None
        d_positions = cp.asarray(row_positions)
        target_dtype = cp.int32 if len(self) <= np.iinfo(np.int32).max else cp.int64
        gather_map = plc.Column.from_cuda_array_interface(
            d_positions.astype(target_dtype, copy=False)
        )
        gathered = plc.copying.gather(
            source,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        )
        return type(self)(
            device_table=gathered,
            index_override=pd.RangeIndex(len(row_positions)),
            column_override=self.column_override,
            schema_override=schema,
            to_pandas_kwargs=self.to_pandas_kwargs,
        )

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
        logical_columns_per_table = [list(table.columns) for table in tables]
        ordered_columns: list[Any] = []
        column_types: dict[Any, Any] = {}

        def _promote_arrow_types(left, right):
            if left == right:
                return left
            if pa.types.is_null(left):
                return right
            if pa.types.is_null(right):
                return left
            if (
                (pa.types.is_string(left) and pa.types.is_large_string(right))
                or (pa.types.is_large_string(left) and pa.types.is_string(right))
            ):
                return pa.large_string()
            if (
                (pa.types.is_binary(left) and pa.types.is_large_binary(right))
                or (pa.types.is_large_binary(left) and pa.types.is_binary(right))
            ):
                return pa.large_binary()
            return left

        for table, logical_columns in zip(arrow_tables, logical_columns_per_table, strict=True):
            for field, logical_name in zip(table.schema, logical_columns, strict=True):
                if logical_name not in column_types:
                    column_types[logical_name] = field.type
                    ordered_columns.append(logical_name)
                else:
                    column_types[logical_name] = _promote_arrow_types(
                        column_types[logical_name],
                        field.type,
                    )

        if not ordered_columns:
            if ignore_index:
                index_override = pd.RangeIndex(sum(len(table) for table in tables))
            else:
                index_override = tables[0].index
                for table in tables[1:]:
                    index_override = index_override.append(table.index)
            return cls(dataframe=pd.DataFrame(index=index_override))

        aligned_tables = []
        for table, logical_columns in zip(arrow_tables, logical_columns_per_table, strict=True):
            physical_by_logical = {
                logical_name: field.name
                for field, logical_name in zip(table.schema, logical_columns, strict=True)
            }
            arrays = []
            for logical_name in ordered_columns:
                physical_name = physical_by_logical.get(logical_name)
                if physical_name is not None:
                    column = table[physical_name]
                    target_type = column_types[logical_name]
                    if column.type != target_type:
                        column = column.cast(target_type)
                    arrays.append(column)
                else:
                    arrays.append(pa.nulls(table.num_rows, type=column_types[logical_name]))
            aligned_tables.append(pa.table(arrays, names=[str(name) for name in ordered_columns]))

        concatenated = pa.concat_tables(aligned_tables)
        if ignore_index:
            index_override = pd.RangeIndex(concatenated.num_rows)
        else:
            index_override = tables[0].index
            for table in tables[1:]:
                index_override = index_override.append(table.index)
        common_kwargs = tables[0].to_pandas_kwargs
        if any(table.to_pandas_kwargs != common_kwargs for table in tables[1:]):
            common_kwargs = {}
        return cls(
            arrow_table=concatenated,
            index_override=index_override,
            column_override=tuple(ordered_columns),
            to_pandas_kwargs=common_kwargs,
        )

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


def _pandas_metadata_field_name_map(metadata: dict[str, Any]) -> dict[str, str | None]:
    return {
        str(column_meta["field_name"]): column_meta.get("name")
        for column_meta in metadata.get("columns", [])
        if column_meta.get("field_name") is not None
    }


def _arrow_index_override_from_pandas_metadata(
    table,
    *,
    to_pandas_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.Index, Any]:
    metadata = table.schema.metadata or {}
    pandas_metadata_raw = metadata.get(b"pandas")
    if pandas_metadata_raw is None:
        return pd.RangeIndex(table.num_rows), table

    pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
    index_columns = pandas_metadata.get("index_columns") or []
    if not index_columns:
        return pd.RangeIndex(table.num_rows), table

    if len(index_columns) == 1 and isinstance(index_columns[0], dict):
        range_spec = index_columns[0]
        if range_spec.get("kind") == "range":
            return (
                pd.RangeIndex(
                    start=int(range_spec.get("start", 0)),
                    stop=int(range_spec.get("stop", table.num_rows)),
                    step=int(range_spec.get("step", 1)),
                    name=range_spec.get("name"),
                ),
                table,
            )

    field_name_map = _pandas_metadata_field_name_map(pandas_metadata)
    index_field_names = [str(name) for name in index_columns]
    index_table = table.select(index_field_names).replace_schema_metadata(None)
    index_frame = index_table.to_pandas(**(to_pandas_kwargs or {}))
    index_names = [field_name_map.get(name, name) for name in index_field_names]
    if len(index_field_names) == 1:
        index = pd.Index(index_frame.iloc[:, 0].array, name=index_names[0])
    else:
        index_frame.columns = index_names
        index = pd.MultiIndex.from_frame(index_frame)
    return index, table.drop(index_field_names)


def native_attribute_table_from_arrow_table(
    table,
    *,
    to_pandas_kwargs: dict[str, Any] | None = None,
) -> NativeAttributeTable:
    index_override, attr_table = _arrow_index_override_from_pandas_metadata(
        table,
        to_pandas_kwargs=to_pandas_kwargs,
    )
    return NativeAttributeTable(
        arrow_table=attr_table,
        index_override=index_override,
        to_pandas_kwargs=to_pandas_kwargs,
    )


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
    rebuilt = frame.copy(deep=False)
    pd.DataFrame.__setitem__(rebuilt, geom_name, geometry_series)
    rebuilt.__class__ = type(frame)
    rebuilt._geometry_column_name = geom_name
    rebuilt.attrs = frame.attrs.copy()
    return rebuilt


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

    def take(self, row_positions) -> GeometryNativeResult:
        normalized = _normalize_row_selection(row_positions)
        if self.owned is not None:
            return type(self).from_owned(self.owned.take(normalized), crs=self.crs)
        host_positions = _host_row_positions(normalized)
        return type(self).from_geoseries(self.series.take(host_positions))

    @property
    def row_count(self) -> int:
        if self.owned is not None:
            return int(self.owned.row_count)
        return int(len(self.series))


@dataclass(frozen=True)
class NativeGeometryColumn:
    name: str
    geometry: GeometryNativeResult


@dataclass(frozen=True)
class NativeReadProvenance:
    surface: str
    format_name: str
    source: str | None = None
    backend: str | None = None
    selected_row_groups: tuple[int, ...] | None = None
    bbox: tuple[float, float, float, float] | None = None
    metadata_source: str | None = None
    planner_strategy: str | None = None
    chunk_rows: int | None = None


@dataclass(frozen=True)
class NativeTabularResult:
    """Device-native tabular export boundary for geometry plus attributes."""

    attributes: NativeAttributeTable | pd.DataFrame
    geometry: GeometryNativeResult
    geometry_name: str
    column_order: tuple[str, ...]
    attrs: dict[str, Any] | None = None
    secondary_geometry: tuple[NativeGeometryColumn, ...] = ()
    provenance: NativeReadProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            NativeAttributeTable.from_value(self.attributes),
        )
        row_count = len(self.attributes)
        if self.geometry.row_count != row_count:
            raise ValueError("primary geometry row count must match attribute row count")
        secondary_names = [column.name for column in self.secondary_geometry]
        if self.geometry_name in secondary_names:
            raise ValueError("secondary geometry columns must not repeat the primary geometry name")
        if len(secondary_names) != len(set(secondary_names)):
            raise ValueError("secondary geometry column names must be unique")
        for column in self.secondary_geometry:
            if column.geometry.row_count != row_count:
                raise ValueError("secondary geometry row counts must match attribute row count")
        geometry_names = {self.geometry_name, *secondary_names}
        missing = [name for name in geometry_names if name not in self.column_order]
        if missing:
            raise ValueError(
                "column_order must include every geometry column in NativeTabularResult"
            )
        overlapping = [name for name in self.attributes.columns if name in geometry_names]
        if overlapping:
            raise ValueError(
                "attribute columns must not reuse primary or secondary geometry column names"
            )

    @property
    def geometry_columns(self) -> tuple[NativeGeometryColumn, ...]:
        return (
            NativeGeometryColumn(self.geometry_name, self.geometry),
            *self.secondary_geometry,
        )

    @property
    def resolved_column_order(self) -> tuple[str, ...]:
        ordered = list(self.column_order)
        geometry_names = {column.name for column in self.geometry_columns}
        attr_columns = list(self.attributes.columns)
        if not attr_columns and self.attributes.loader is not None:
            attr_columns = list(self.attributes.to_pandas(copy=False).columns)
        missing_attr_columns = [
            column for column in attr_columns if column not in ordered and column not in geometry_names
        ]
        if not missing_attr_columns:
            return self.column_order
        insert_at = next(
            (index for index, name in enumerate(ordered) if name in geometry_names),
            len(ordered),
        )
        ordered[insert_at:insert_at] = missing_attr_columns
        return tuple(ordered)

    def to_geodataframe(self) -> GeoDataFrame:
        attributes = NativeAttributeTable.from_value(self.attributes)
        record_materialization_event(
            surface="vibespatial.api.NativeTabularResult.to_geodataframe",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="native_tabular_to_geodataframe",
            reason="native tabular result exported to GeoDataFrame compatibility surface",
            detail=(
                f"rows={len(attributes)}, attribute_columns={len(attributes.columns)}, "
                f"attribute_storage={_attribute_storage_label(attributes)}, "
                f"geometry_storage={_geometry_storage_label(self.geometry)}, "
                f"secondary_geometry={len(self.secondary_geometry)}"
            ),
        )
        frame = _materialize_attribute_geometry_frame(
            attributes,
            self.geometry_columns,
            geometry_name=self.geometry_name,
            column_order=self.resolved_column_order,
        )
        if self.attrs:
            frame.attrs.update(self.attrs)
        from vibespatial.api._native_state import attach_native_state_from_native_tabular_result

        attach_native_state_from_native_tabular_result(frame, self)
        return frame

    def to_arrow(
        self,
        *,
        index: bool | None = None,
        geometry_encoding: str = "WKB",
        interleaved: bool = True,
        include_z: bool | None = None,
        force_device_geometry_encode: bool = False,
    ):
        from vibespatial.api.io._geoarrow import ArrowTable
        from vibespatial.io.geoarrow import native_tabular_to_arrow

        table, _geometry_encoding = native_tabular_to_arrow(
            self,
            index=index,
            geometry_encoding=geometry_encoding,
            interleaved=interleaved,
            include_z=include_z,
            force_device_geometry_encode=force_device_geometry_encode,
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

    def take(self, row_positions, *, preserve_index: bool = True) -> NativeTabularResult:
        normalized = _normalize_row_selection(row_positions)
        return type(self)(
            attributes=self.attributes.take(normalized, preserve_index=preserve_index),
            geometry=self.geometry.take(normalized),
            geometry_name=self.geometry_name,
            column_order=self.column_order,
            attrs=self.attrs,
            secondary_geometry=tuple(
                NativeGeometryColumn(column.name, column.geometry.take(normalized))
                for column in self.secondary_geometry
            ),
            provenance=self.provenance,
        )


def _materialize_attribute_geometry_frame(
    attributes: NativeAttributeTable | pd.DataFrame,
    geometry_columns: tuple[NativeGeometryColumn, ...],
    *,
    geometry_name: str,
    column_order: tuple[str, ...] | None = None,
):
    """Explicit host export for attribute tables plus native geometry columns."""
    from vibespatial.api.geodataframe import GeoDataFrame

    attributes = NativeAttributeTable.from_value(attributes)
    frame = attributes.to_pandas(copy=False)
    geometry_names = [column.name for column in geometry_columns]
    overlap = [name for name in frame.columns if name in geometry_names]
    if overlap:
        raise ValueError(
            "attribute columns must not overlap exported geometry column names"
        )
    geometry_series = {
        column.name: column.geometry.to_geoseries(index=attributes.index, name=column.name)
        for column in geometry_columns
    }
    active_geometry = geometry_series[geometry_name]

    requested_order = list(column_order) if column_order is not None else []
    if requested_order:
        normalized_columns = []
        changed = False
        for column in frame.columns:
            logical_name = next(
                (requested for requested in requested_order if str(column) == str(requested)),
                column,
            )
            if logical_name != column:
                changed = True
            normalized_columns.append(logical_name)
        if changed:
            frame = frame.copy(deep=False)
            frame.columns = normalized_columns
    known_columns = set(frame.columns) | set(geometry_series)
    unknown = [name for name in requested_order if name not in known_columns]
    if unknown:
        raise ValueError(
            f"column_order contains columns that are not present in the export payload: {unknown}"
        )

    ordered_columns: list[str] = []
    seen: set[str] = set()

    def append_column(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        ordered_columns.append(name)

    for name in requested_order:
        append_column(name)
    for name in frame.columns:
        append_column(name)
    for name in geometry_names:
        append_column(name)

    ordered_values = {
        name: geometry_series[name] if name in geometry_series else frame[name]
        for name in ordered_columns
    }
    rebuilt = GeoDataFrame(
        pd.DataFrame(ordered_values, index=attributes.index, copy=False),
        geometry=geometry_name,
        crs=active_geometry.crs,
        copy=False,
    )
    rebuilt.attrs = frame.attrs.copy()
    return _replace_geometry_column_preserving_backing(
        rebuilt,
        active_geometry.values,
        crs=active_geometry.crs,
    )


__all__ = [
    "GeometryNativeResult",
    "NativeAttributeTable",
    "NativeGeometryColumn",
    "NativeReadProvenance",
    "NativeTabularResult",
    "_host_array",
    "_host_row_positions",
    "_materialize_attribute_geometry_frame",
    "_normalize_row_selection",
    "_replace_geometry_column_preserving_backing",
    "_set_active_geometry_name",
    "native_attribute_table_from_arrow_table",
]
