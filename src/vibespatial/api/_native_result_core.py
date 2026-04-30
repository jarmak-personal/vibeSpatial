from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vibespatial.api._compat import PANDAS_GE_30
from vibespatial.runtime.materialization import (
    MaterializationBoundary,
    NativeExportBoundary,
    record_materialization_event,
    record_native_export_boundary,
)
from vibespatial.runtime.residency import Residency

if TYPE_CHECKING:
    from vibespatial.api.geodataframe import GeoDataFrame
    from vibespatial.api.geoseries import GeoSeries


def _host_array(
    values: Any,
    *,
    dtype,
    strict_disallowed: bool = True,
    surface: str = "vibespatial.api._native_result_core._host_array",
    operation: str = "array_to_host",
    reason: str = "device-like array exposed a get() host materialization path",
    detail: str = "",
) -> np.ndarray:
    if hasattr(values, "get"):
        record_materialization_event(
            surface=surface,
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation=operation,
            reason=reason,
            detail=detail,
            d2h_transfer=True,
            strict_disallowed=strict_disallowed,
        )
        if hasattr(values, "__cuda_array_interface__"):
            from vibespatial.cuda._runtime import get_cuda_runtime

            host_values = get_cuda_runtime().copy_device_to_host(
                values,
                reason=f"{surface}::{operation}",
            )
        elif type(values).__module__.startswith("cupy"):
            from vibespatial.cuda._runtime import get_cuda_runtime

            host_values = get_cuda_runtime().copy_device_to_host(
                values,
                reason=f"{surface}::{operation}",
            )
        else:
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


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _row_aligned_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    size = getattr(values, "size", None)
    if size is not None:
        return int(size)
    return len(values)


def _row_aligned_residency(*values: Any | None) -> Residency:
    return (
        Residency.DEVICE
        if any(value is not None and _is_device_array(value) for value in values)
        else Residency.HOST
    )


def _gather_row_aligned_optional(values: Any | None, row_positions: Any) -> Any | None:
    if values is None:
        return None
    if _is_device_array(values) or _is_device_array(row_positions):
        import cupy as cp

        return cp.asarray(values)[cp.asarray(row_positions, dtype=cp.int64)]
    return np.asarray(values)[np.asarray(row_positions, dtype=np.int64)]


def _concat_row_aligned_optional(values: list[Any | None]) -> Any | None:
    if not values or all(value is None for value in values):
        return None
    if any(value is None for value in values):
        return None
    if any(_is_device_array(value) for value in values):
        import cupy as cp

        return cp.concatenate([cp.asarray(value) for value in values])
    return np.concatenate([np.asarray(value) for value in values])


def _concat_optional_repaired_masks(
    provenances: list[NativeGeometryProvenance],
) -> Any | None:
    masks = [provenance.repaired_mask for provenance in provenances]
    if not masks or all(mask is None for mask in masks):
        return None
    if any(_is_device_array(mask) for mask in masks if mask is not None):
        import cupy as cp

        return cp.concatenate(
            [
                cp.zeros(provenance.row_count, dtype=cp.bool_)
                if mask is None
                else cp.asarray(mask, dtype=cp.bool_)
                for provenance, mask in zip(provenances, masks, strict=True)
            ]
        )
    return np.concatenate(
        [
            np.zeros(provenance.row_count, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
            for provenance, mask in zip(provenances, masks, strict=True)
        ]
    )


def _host_row_positions(
    row_positions,
    *,
    strict_disallowed: bool = True,
) -> np.ndarray:
    normalized = _normalize_row_selection(row_positions)
    if hasattr(normalized, "__cuda_array_interface__"):
        item_count = int(getattr(normalized, "size", len(normalized)))
        itemsize = int(getattr(getattr(normalized, "dtype", None), "itemsize", 0))
        record_materialization_event(
            surface="vibespatial.api._native_result_core._host_row_positions",
            boundary=MaterializationBoundary.INTERNAL_HOST_CONVERSION,
            operation="row_positions_to_host",
            reason="device row positions were normalized on host",
            detail=f"rows={item_count}, bytes={item_count * itemsize}",
            d2h_transfer=True,
            strict_disallowed=strict_disallowed,
        )
        from vibespatial.cuda._runtime import get_cuda_runtime

        return get_cuda_runtime().copy_device_to_host(
            normalized,
            reason="vibespatial.api._native_result_core._host_row_positions::row_positions_to_host",
        )
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

    def _normalize_pandas_range_metadata(output):
        metadata = output.schema.metadata
        if metadata is None or b"pandas" not in metadata:
            return output
        metadata = dict(metadata)
        try:
            pandas_metadata = json.loads(metadata[b"pandas"].decode("utf-8"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return output
        index_columns = pandas_metadata.get("index_columns") or []
        if len(index_columns) != 1 or not isinstance(index_columns[0], dict):
            return output
        range_spec = dict(index_columns[0])
        if range_spec.get("kind") != "range":
            return output
        start = int(range_spec.get("start", 0))
        stop = int(range_spec.get("stop", output.num_rows))
        step = int(range_spec.get("step", 1))
        if step == 0:
            expected_rows = -1
        else:
            expected_rows = len(range(start, stop, step))
        if expected_rows == int(output.num_rows):
            return output
        range_spec["start"] = 0
        range_spec["stop"] = int(output.num_rows)
        range_spec["step"] = 1
        pandas_metadata["index_columns"] = [range_spec]
        metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
        return output.replace_schema_metadata(metadata)

    names = [str(name) for name in (column_override or table.column_names)]
    if schema is None:
        return _normalize_pandas_range_metadata(pa.Table.from_arrays(
            [table.column(index) for index in range(table.num_columns)],
            names=names,
            metadata=table.schema.metadata,
        ))
    fields = []
    for index, name in enumerate(names):
        try:
            fields.append(schema.field(name))
        except KeyError:
            fields.append(pa.field(name, table.column(index).type))
    return _normalize_pandas_range_metadata(pa.Table.from_arrays(
        [table.column(index) for index in range(table.num_columns)],
        schema=pa.schema(fields, metadata=schema.metadata),
    ))


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


def _native_expression_device_column(value, *, row_count: int):
    from vibespatial.api._native_expression import NativeExpression

    if not isinstance(value, NativeExpression):
        return None
    if value.source_row_count is not None and int(value.source_row_count) != int(row_count):
        raise ValueError("NativeExpression column row count must match attributes")
    if not value.is_device:
        return None

    import cupy as cp
    import pylibcudf as plc

    values = cp.asarray(value.values)
    if int(values.size) != int(row_count):
        raise ValueError("NativeExpression column row count must match attributes")
    dtype = np.dtype(values.dtype)
    if not (
        np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.bool_)
    ):
        return None
    return plc.Column.from_cuda_array_interface(values)


def _assigned_device_column(value, *, row_count: int):
    expression_column = _native_expression_device_column(value, row_count=row_count)
    if expression_column is not None:
        return expression_column
    from vibespatial.api._native_expression import NativeExpression

    if isinstance(value, NativeExpression):
        return None

    if _is_device_array(value):
        import cupy as cp
        import pylibcudf as plc

        values = cp.asarray(value)
        if values.ndim != 1 or int(values.size) != int(row_count):
            raise ValueError("assigned column row count must match attributes")
        dtype = np.dtype(values.dtype)
        if not (
            np.issubdtype(dtype, np.number)
            or np.issubdtype(dtype, np.bool_)
        ):
            return None
        return plc.Column.from_cuda_array_interface(values)

    if pd.api.types.is_scalar(value):
        host_values = np.full(int(row_count), value)
    elif isinstance(value, pd.Series):
        if len(value) != int(row_count):
            raise ValueError("assigned column row count must match attributes")
        if not _is_admissible_pandas_numeric_series(value):
            return None
        host_values = value.to_numpy(copy=False)
    else:
        host_values = np.asarray(value)
        if host_values.ndim != 1 or int(host_values.size) != int(row_count):
            raise ValueError("assigned column row count must match attributes")

    dtype = np.dtype(host_values.dtype)
    if not (
        np.issubdtype(dtype, np.number)
        or np.issubdtype(dtype, np.bool_)
    ):
        return None
    if bool(pd.isna(host_values).any()):
        return None

    import cupy as cp
    import pylibcudf as plc

    return plc.Column.from_cuda_array_interface(cp.asarray(host_values))


def _field_for_device_column(name: Any, column, schema):
    import pyarrow as pa

    if schema is not None:
        try:
            return schema.field(str(name))
        except KeyError:
            pass
    return pa.field(str(name), column.type().to_arrow())


@dataclass(frozen=True)
class NativeAttributeColumnPolicy:
    """Private dtype contract for a device-backed attribute column."""

    category: str
    arrow_type: str
    null_count: int
    can_project_take: bool
    can_compute_numeric: bool


def _device_attribute_column_policy(arrow_type, *, null_count: int) -> NativeAttributeColumnPolicy:
    import pyarrow as pa

    if _is_admissible_arrow_numeric_type(arrow_type):
        if int(null_count) == 0:
            return NativeAttributeColumnPolicy(
                category="all-valid-numeric-bool",
                arrow_type=str(arrow_type),
                null_count=int(null_count),
                can_project_take=True,
                can_compute_numeric=True,
            )
        return NativeAttributeColumnPolicy(
            category="nullable-numeric-bool-movement-only",
            arrow_type=str(arrow_type),
            null_count=int(null_count),
            can_project_take=True,
            can_compute_numeric=False,
        )
    if pa.types.is_dictionary(arrow_type):
        category = "categorical-movement-only"
    elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        category = "string-movement-only"
    elif pa.types.is_temporal(arrow_type):
        category = "datetime-movement-only"
    elif pa.types.is_null(arrow_type):
        category = "null-movement-only"
    else:
        category = "device-movement-only"
    return NativeAttributeColumnPolicy(
        category=category,
        arrow_type=str(arrow_type),
        null_count=int(null_count),
        can_project_take=True,
        can_compute_numeric=False,
    )


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
            policies = self.device_column_policies(requested)
            out = {}
            for column in requested:
                policy = policies.get(column)
                if policy is None or not policy.can_compute_numeric:
                    return None
                values = _pylibcudf_numeric_column_view(
                    source_columns[column_positions[column]]
                )
                if values is None:
                    return None
                out[column] = values
            return out

        return None

    def device_column_policies(
        self,
        columns=None,
    ) -> dict[Any, NativeAttributeColumnPolicy]:
        """Return explicit movement/compute contracts for device columns."""
        if self.device_table is None or not hasattr(self.device_table, "columns"):
            return {}
        requested = (
            tuple(self.columns)
            if columns is None
            else tuple(dict.fromkeys(columns))
        )
        positions = _column_position_map(self.columns)
        if positions is None or any(column not in positions for column in requested):
            return {}
        source_columns = self.device_table.columns()
        return {
            column: _device_attribute_column_policy(
                _field_for_device_column(
                    column,
                    source_columns[positions[column]],
                    self.schema_override,
                ).type,
                null_count=int(source_columns[positions[column]].null_count()),
            )
            for column in requested
        }

    def grouped_device_take_columns(
        self,
        grouped,
        reducers: Mapping[Any, str],
    ) -> NativeAttributeTable | None:
        """Reduce all-valid device columns with grouped first/last gathers.

        This is the non-numeric device-attribute compute contract: string,
        datetime, categorical, and numeric/bool columns may participate only
        when the reducer is positional (`first`/`last`), every group has at
        least one row, and the source column has no nulls. Null-skipping and
        missing-group semantics still decline to the exact host/export path.
        """
        if self.device_table is None or not hasattr(self.device_table, "columns"):
            return None
        if not reducers:
            return None
        normalized_reducers = {
            column: reducer.lower()
            for column, reducer in reducers.items()
            if isinstance(reducer, str)
        }
        if set(normalized_reducers) != set(reducers) or any(
            reducer not in {"first", "last"}
            for reducer in normalized_reducers.values()
        ):
            return None
        if not getattr(grouped, "is_device", False):
            return None
        group_count = int(grouped.resolved_group_count)
        if group_count == 0:
            return None
        group_ids = getattr(grouped, "group_ids", None)
        group_offsets = getattr(grouped, "group_offsets", None)
        sorted_order = getattr(grouped, "sorted_order", None)
        if group_ids is None or group_offsets is None or sorted_order is None:
            return None
        if int(getattr(group_ids, "size", 0)) != group_count:
            return None

        requested = tuple(normalized_reducers)
        positions = _column_position_map(self.columns)
        if positions is None or any(column not in positions for column in requested):
            return None
        policies = self.device_column_policies(requested)
        if any(
            (policy := policies.get(column)) is None
            or not policy.can_project_take
            or policy.null_count != 0
            for column in requested
        ):
            return None

        try:
            import cupy as cp
            import pyarrow as pa
            import pylibcudf as plc
        except ModuleNotFoundError:
            return None

        d_offsets = cp.asarray(group_offsets, dtype=cp.int64)
        d_sorted_order = cp.asarray(sorted_order, dtype=cp.int64)
        first_positions = d_sorted_order[d_offsets[:-1]]
        last_positions = d_sorted_order[d_offsets[1:] - 1]
        target_dtype = cp.int32 if len(self) <= np.iinfo(np.int32).max else cp.int64
        source_columns = self.device_table.columns()
        output_columns = []
        fields = []
        for column in requested:
            selected_rows = (
                first_positions
                if normalized_reducers[column] == "first"
                else last_positions
            )
            gather_map = plc.Column.from_cuda_array_interface(
                selected_rows.astype(target_dtype, copy=False)
            )
            gathered = plc.copying.gather(
                plc.Table([source_columns[positions[column]]]),
                gather_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            output_column = gathered.columns()[0]
            output_columns.append(output_column)
            fields.append(_field_for_device_column(column, output_column, self.schema_override))

        index_override = (
            grouped.output_index_plan.index
            if grouped.output_index_plan is not None
            and grouped.output_index_plan.index is not None
            else pd.RangeIndex(group_count)
        )
        return type(self)(
            device_table=plc.Table(output_columns),
            index_override=index_override,
            column_override=requested,
            schema_override=pa.schema(
                fields,
                metadata=None if self.schema_override is None else self.schema_override.metadata,
            ),
            to_pandas_kwargs=self.to_pandas_kwargs,
        )

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
        if self.device_table is not None:
            logical_columns = tuple([*self.columns, name])
            assigned = self.assign_columns({name: values}, columns=logical_columns)
            if assigned is not None:
                return assigned
            frame = self.to_pandas(copy=False).copy(deep=False)
            frame[name] = values
            return type(self)(dataframe=frame)

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
            row_count = len(self)
            assigned_columns = {}
            for name, value in assigned.items():
                column = _assigned_device_column(value, row_count=row_count)
                if column is None:
                    return None
                assigned_columns[name] = column
            if not hasattr(self.device_table, "columns"):
                return None
            source_columns = self.device_table.columns()
            positions = _column_position_map(self.columns)
            if positions is None:
                return None
            import pyarrow as pa
            import pylibcudf as plc

            output_columns = []
            fields = []
            for name in requested:
                if name in assigned_columns:
                    column = assigned_columns[name]
                elif name in positions:
                    column = source_columns[positions[name]]
                else:
                    return None
                output_columns.append(column)
                fields.append(_field_for_device_column(name, column, self.schema_override))
            return type(self)(
                device_table=plc.Table(output_columns),
                index_override=self.index,
                column_override=requested,
                schema_override=pa.schema(
                    fields,
                    metadata=(
                        None if self.schema_override is None else self.schema_override.metadata
                    ),
                ),
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        frame = self.dataframe.copy(deep=False)
        for name, values in assigned.items():
            frame[name] = values
        return type(self)(dataframe=frame.loc[:, list(requested)])

    def project_columns(self, columns: tuple[Any, ...]) -> NativeAttributeTable | None:
        """Return attributes projected to an exact logical column order."""
        requested = tuple(columns)
        positions = _column_position_map(self.columns)
        if positions is None or any(column not in positions for column in requested):
            return None
        if requested == tuple(self.columns):
            return self

        if self.loader is not None:
            parent = self

            def _load_projected() -> pd.DataFrame:
                return parent.to_pandas(copy=False).loc[:, list(requested)]

            return type(self).from_loader(
                _load_projected,
                index_override=self.index,
                columns=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        if self.arrow_table is not None:
            return type(self)(
                arrow_table=self.to_arrow(index=False, columns=requested),
                index_override=self.index,
                column_override=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        if self.device_table is not None:
            if not hasattr(self.device_table, "columns"):
                return None
            import pyarrow as pa
            import pylibcudf as plc

            source_columns = self.device_table.columns()
            output_columns = [source_columns[positions[name]] for name in requested]
            fields = [
                _field_for_device_column(name, column, self.schema_override)
                for name, column in zip(requested, output_columns, strict=True)
            ]
            return type(self)(
                device_table=plc.Table(output_columns),
                index_override=self.index,
                column_override=requested,
                schema_override=pa.schema(
                    fields,
                    metadata=(
                        None if self.schema_override is None else self.schema_override.metadata
                    ),
                ),
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        frame = self.dataframe.loc[:, list(requested)].copy(deep=False)
        return type(self)(dataframe=frame)

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
        if self.device_table is not None and hasattr(self.device_table, "columns"):
            try:
                import cupy as cp
                import pyarrow as pa
                import pylibcudf as plc
            except ModuleNotFoundError:
                pass
            else:
                if isinstance(self.index, pd.MultiIndex):
                    index_frame = self.index.to_frame(index=False)
                else:
                    index_frame = pd.DataFrame(
                        {leading_columns[0]: self.index},
                        index=pd.RangeIndex(len(self)),
                    )
                index_columns = []
                index_fields = []
                for position, column_name in enumerate(leading_columns):
                    series = index_frame.iloc[:, position]
                    if not _is_admissible_pandas_numeric_series(series):
                        index_columns = []
                        break
                    values = cp.asarray(series.to_numpy(copy=False))
                    column = plc.Column.from_cuda_array_interface(values)
                    index_columns.append(column)
                    index_fields.append(pa.field(str(column_name), column.type().to_arrow()))
                if index_columns:
                    source_columns = self.device_table.columns()
                    output_columns = [*index_columns, *source_columns]
                    fields = [
                        *index_fields,
                        *(
                            _field_for_device_column(
                                name,
                                column,
                                self.schema_override,
                            )
                            for name, column in zip(
                                self.columns,
                                source_columns,
                                strict=True,
                            )
                        ),
                    ]
                    return (
                        type(self)(
                            device_table=plc.Table(output_columns),
                            index_override=pd.RangeIndex(len(self)),
                            column_override=reset_columns,
                            schema_override=pa.schema(
                                fields,
                                metadata=(
                                    None
                                    if self.schema_override is None
                                    else self.schema_override.metadata
                                ),
                            ),
                            to_pandas_kwargs=self.to_pandas_kwargs,
                        ),
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
        if self.dataframe is not None and self.dataframe.shape[1] == 0:
            return type(self)(
                dataframe=pd.DataFrame(
                    index=pd.RangeIndex(_row_aligned_size(row_positions))
                )
            )
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
    def combine_columns(
        cls,
        tables: list[NativeAttributeTable],
        *,
        index_override: pd.Index | None = None,
    ) -> NativeAttributeTable | None:
        """Return a column-wise combination without crossing device tables to host."""
        if not tables:
            index = pd.RangeIndex(0) if index_override is None else index_override
            return cls(dataframe=pd.DataFrame(index=index))

        row_count = len(tables[0])
        if any(len(table) != row_count for table in tables[1:]):
            raise ValueError("NativeAttributeTable column combine requires equal row counts")
        if index_override is None:
            index_override = tables[0].index
            if any(not table.index.equals(index_override) for table in tables[1:]):
                return None

        logical_columns = tuple(
            column for table in tables for column in tuple(table.columns)
        )
        if len(set(logical_columns)) != len(logical_columns):
            return None

        non_empty = [table for table in tables if len(table.columns) > 0]
        if not non_empty:
            return cls(dataframe=pd.DataFrame(index=index_override))

        common_kwargs = non_empty[0].to_pandas_kwargs
        if any(table.to_pandas_kwargs != common_kwargs for table in non_empty[1:]):
            common_kwargs = {}

        if all(table.device_table is not None for table in non_empty):
            try:
                import pyarrow as pa
                import pylibcudf as plc
            except ModuleNotFoundError:
                return None

            output_columns = []
            fields = []
            for table in non_empty:
                source_columns = table.device_table.columns()
                for column_name, column in zip(
                    tuple(table.columns),
                    source_columns,
                    strict=True,
                ):
                    output_columns.append(column)
                    fields.append(
                        _field_for_device_column(
                            column_name,
                            column,
                            table.schema_override,
                        )
                    )
            return cls(
                device_table=plc.Table(output_columns),
                index_override=index_override,
                column_override=logical_columns,
                schema_override=pa.schema(fields),
                to_pandas_kwargs=common_kwargs,
            )

        if all(table.arrow_table is not None for table in non_empty):
            try:
                import pyarrow as pa
            except ModuleNotFoundError:
                return None

            arrays = []
            fields = []
            for table in non_empty:
                arrow = table.to_arrow(index=False)
                for logical_name, physical_name in zip(
                    tuple(table.columns),
                    arrow.column_names,
                    strict=True,
                ):
                    column = arrow.column(physical_name)
                    arrays.append(column)
                    fields.append(pa.field(str(logical_name), column.type))
            return cls(
                arrow_table=pa.Table.from_arrays(
                    arrays,
                    schema=pa.schema(fields),
                ),
                index_override=index_override,
                column_override=logical_columns,
                to_pandas_kwargs=common_kwargs,
            )

        if all(table.device_table is None for table in non_empty):
            frames = [table.to_pandas(copy=False) for table in non_empty]
            concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
            combined = pd.concat(frames, axis=1, **concat_kwargs)
            if not combined.index.equals(index_override):
                combined = combined.copy(deep=False)
                combined.index = index_override
            return cls(dataframe=combined)

        return None

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

        device_tables = [table for table in tables if table.device_table is not None]
        if len(device_tables) == len(tables):
            common_columns = tuple(tables[0].columns)
            common_schema = tables[0].schema_override
            if all(tuple(table.columns) == common_columns for table in tables):
                try:
                    import pylibcudf as plc
                except ModuleNotFoundError:  # pragma: no cover - optional GPU dependency
                    plc = None
                if plc is not None:
                    concatenated = plc.concatenate.concatenate(
                        [table.device_table for table in tables]
                    )
                    if ignore_index:
                        index_override = pd.RangeIndex(_device_table_row_count(concatenated))
                    else:
                        index_override = tables[0].index
                        for table in tables[1:]:
                            index_override = index_override.append(table.index)
                    if any(table.schema_override != common_schema for table in tables[1:]):
                        common_schema = None
                    common_kwargs = tables[0].to_pandas_kwargs
                    if any(table.to_pandas_kwargs != common_kwargs for table in tables[1:]):
                        common_kwargs = {}
                    return cls(
                        device_table=concatenated,
                        index_override=index_override,
                        column_override=common_columns,
                        schema_override=common_schema,
                        to_pandas_kwargs=common_kwargs,
                    )

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

    def with_crs(self, crs) -> GeometryNativeResult:
        if self.crs == crs:
            return self
        return type(self)(crs=crs, owned=self.owned, series=self.series)

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
class NativeGeometryProvenance:
    """Row-aligned source lineage for constructive geometry outputs."""

    operation: str
    row_count: int
    source_rows: Any | None = None
    left_rows: Any | None = None
    right_rows: Any | None = None
    part_family_tags: Any | None = None
    repaired_mask: Any | None = None
    source_tokens: tuple[str, ...] = ()
    keep_geom_type_applied: bool = False
    residency: Residency = dataclass_field(init=False, default=Residency.HOST)

    def __post_init__(self) -> None:
        if int(self.row_count) < 0:
            raise ValueError("NativeGeometryProvenance row_count must be non-negative")
        for name in (
            "source_rows",
            "left_rows",
            "right_rows",
            "part_family_tags",
            "repaired_mask",
        ):
            values = getattr(self, name)
            if values is not None and _row_aligned_size(values) != int(self.row_count):
                raise ValueError(
                    f"NativeGeometryProvenance {name} length must match row_count"
                )
        object.__setattr__(
            self,
            "residency",
            _row_aligned_residency(
                self.source_rows,
                self.left_rows,
                self.right_rows,
                self.part_family_tags,
                self.repaired_mask,
            ),
        )

    @property
    def is_device(self) -> bool:
        return self.residency is Residency.DEVICE

    def validate_row_count(self, row_count: int) -> None:
        if int(row_count) != int(self.row_count):
            raise ValueError(
                f"NativeGeometryProvenance row count mismatch: expected "
                f"{self.row_count}, got {row_count}"
            )

    def take(self, row_positions: Any) -> NativeGeometryProvenance:
        positions = _normalize_row_selection(row_positions)
        return type(self)(
            operation=self.operation,
            row_count=_row_aligned_size(positions),
            source_rows=_gather_row_aligned_optional(self.source_rows, positions),
            left_rows=_gather_row_aligned_optional(self.left_rows, positions),
            right_rows=_gather_row_aligned_optional(self.right_rows, positions),
            part_family_tags=_gather_row_aligned_optional(
                self.part_family_tags,
                positions,
            ),
            repaired_mask=_gather_row_aligned_optional(self.repaired_mask, positions),
            source_tokens=self.source_tokens,
            keep_geom_type_applied=self.keep_geom_type_applied,
        )

    @classmethod
    def concat(
        cls,
        provenances: list[NativeGeometryProvenance],
        *,
        operation: str = "concat",
    ) -> NativeGeometryProvenance | None:
        if not provenances:
            return None
        source_tokens = tuple(
            dict.fromkeys(
                token
                for provenance in provenances
                for token in provenance.source_tokens
            )
        )
        return cls(
            operation=operation,
            row_count=sum(int(provenance.row_count) for provenance in provenances),
            source_rows=_concat_row_aligned_optional(
                [provenance.source_rows for provenance in provenances],
            ),
            left_rows=_concat_row_aligned_optional(
                [provenance.left_rows for provenance in provenances],
            ),
            right_rows=_concat_row_aligned_optional(
                [provenance.right_rows for provenance in provenances],
            ),
            part_family_tags=_concat_row_aligned_optional(
                [provenance.part_family_tags for provenance in provenances],
            ),
            repaired_mask=_concat_optional_repaired_masks(provenances),
            source_tokens=source_tokens,
            keep_geom_type_applied=any(
                provenance.keep_geom_type_applied for provenance in provenances
            ),
        )


@dataclass(frozen=True)
class NativeTabularResult:
    """Device-native tabular export boundary for geometry plus attributes."""

    attributes: NativeAttributeTable | pd.DataFrame
    geometry: GeometryNativeResult
    geometry_name: str
    column_order: tuple[str, ...]
    attrs: dict[str, Any] | None = None
    secondary_geometry: tuple[NativeGeometryColumn, ...] = ()
    provenance: NativeReadProvenance | NativeGeometryProvenance | None = None
    geometry_metadata: Any | None = None
    index_plan: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            NativeAttributeTable.from_value(self.attributes),
        )
        row_count = len(self.attributes)
        if self.geometry.row_count != row_count:
            raise ValueError("primary geometry row count must match attribute row count")
        if self.index_plan is not None:
            self.index_plan.validate_length(row_count)
        if self.geometry_metadata is not None:
            self.geometry_metadata.validate_row_count(row_count)
        if self.provenance is not None and hasattr(self.provenance, "validate_row_count"):
            self.provenance.validate_row_count(row_count)
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

    def to_native_frame_state(self):
        """Return the private frame carrier without public materialization."""
        from vibespatial.api._native_state import NativeFrameState

        return NativeFrameState.from_native_tabular_result(self)

    def attributes_for_export(
        self,
        *,
        surface: str,
        include_index: bool = True,
        strict_disallowed: bool = False,
    ) -> NativeAttributeTable:
        """Return attributes indexed for an explicit public export boundary."""
        attributes = NativeAttributeTable.from_value(self.attributes)
        if not include_index or self.index_plan is None:
            return attributes
        public_index = self.index_plan.to_public_index(
            surface=surface,
            strict_disallowed=strict_disallowed,
        )
        if attributes.index.equals(public_index):
            return attributes
        return attributes.with_index(public_index)

    def to_geodataframe(self) -> GeoDataFrame:
        attributes = self.attributes_for_export(
            surface="vibespatial.api.NativeTabularResult.to_geodataframe",
            include_index=True,
            strict_disallowed=False,
        )
        record_native_export_boundary(NativeExportBoundary(
            surface="vibespatial.api.NativeTabularResult.to_geodataframe",
            operation="native_tabular_to_geodataframe",
            target="geodataframe",
            reason="native tabular result exported to GeoDataFrame compatibility surface",
            detail=(
                f"attribute_columns={len(attributes.columns)}, "
                f"attribute_storage={_attribute_storage_label(attributes)}, "
                f"geometry_storage={_geometry_storage_label(self.geometry)}, "
                f"secondary_geometry={len(self.secondary_geometry)}"
            ),
            row_count=len(attributes),
        ))
        frame = _materialize_attribute_geometry_frame(
            attributes,
            self.geometry_columns,
            geometry_name=self.geometry_name,
            column_order=self.resolved_column_order,
        )
        if self.attrs:
            frame.attrs.update(self.attrs)
        from vibespatial.api._native_state import attach_native_state_from_native_tabular_result

        attach_result = replace(
            self,
            attributes=attributes,
            column_order=tuple(frame.columns),
        )
        attach_native_state_from_native_tabular_result(frame, attach_result)
        return frame

    def to_arrow(
        self,
        *,
        index: bool | None = None,
        geometry_encoding: str = "WKB",
        interleaved: bool = True,
        include_z: bool | None = None,
        force_device_geometry_encode: bool = False,
        record_export_boundary: bool = True,
    ):
        from vibespatial.api.io._geoarrow import ArrowTable
        from vibespatial.io.geoarrow import native_tabular_to_arrow

        if record_export_boundary:
            record_native_export_boundary(NativeExportBoundary(
                surface="vibespatial.api.NativeTabularResult.to_arrow",
                operation="native_tabular_to_arrow",
                target="arrow",
                reason="native tabular result exported to Arrow compatibility surface",
                detail=(
                    f"attribute_columns={len(self.attributes.columns)}, "
                    f"geometry_encoding={geometry_encoding}, "
                    f"secondary_geometry={len(self.secondary_geometry)}"
                ),
                row_count=len(self.attributes),
            ))
        table, _geometry_encoding = native_tabular_to_arrow(
            self,
            index=index,
            geometry_encoding=geometry_encoding,
            interleaved=interleaved,
            include_z=include_z,
            force_device_geometry_encode=force_device_geometry_encode,
            record_export_boundary=False,
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
        record_export_boundary: bool = True,
        **kwargs,
    ) -> None:
        from vibespatial.io.geoparquet import write_geoparquet

        if record_export_boundary:
            record_native_export_boundary(NativeExportBoundary(
                surface="vibespatial.api.NativeTabularResult.to_parquet",
                operation="native_tabular_to_parquet",
                target="geoparquet",
                reason="native tabular result exported to GeoParquet writer boundary",
                detail=(
                    f"attribute_columns={len(self.attributes.columns)}, "
                    f"geometry_encoding={geometry_encoding}, "
                    f"secondary_geometry={len(self.secondary_geometry)}"
                ),
                row_count=len(self.attributes),
            ))
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
        record_export_boundary: bool = True,
        **kwargs,
    ) -> None:
        from vibespatial.api.io.arrow import _to_feather

        if record_export_boundary:
            record_native_export_boundary(NativeExportBoundary(
                surface="vibespatial.api.NativeTabularResult.to_feather",
                operation="native_tabular_to_feather",
                target="feather",
                reason="native tabular result exported to Feather writer boundary",
                detail=(
                    f"attribute_columns={len(self.attributes.columns)}, "
                    f"secondary_geometry={len(self.secondary_geometry)}"
                ),
                row_count=len(self.attributes),
            ))
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
        index_plan = self.index_plan
        if preserve_index and index_plan is None:
            from vibespatial.api._native_rowset import NativeIndexPlan

            index_plan = NativeIndexPlan.from_index(self.attributes.index)
        attributes_preserve_index = preserve_index
        if (
            preserve_index
            and _is_device_array(normalized)
            and index_plan is not None
            and index_plan.kind in {"range", "device-labels"}
        ):
            attributes_preserve_index = False
        taken_index_plan = (
            None
            if index_plan is None
            else index_plan.take(
                normalized,
                preserve_index=preserve_index,
                unique=False,
            )
        )
        return type(self)(
            attributes=self.attributes.take(
                normalized,
                preserve_index=attributes_preserve_index,
            ),
            geometry=self.geometry.take(normalized),
            geometry_name=self.geometry_name,
            column_order=self.column_order,
            attrs=self.attrs,
            secondary_geometry=tuple(
                NativeGeometryColumn(column.name, column.geometry.take(normalized))
                for column in self.secondary_geometry
            ),
            provenance=(
                self.provenance.take(normalized)
                if self.provenance is not None and hasattr(self.provenance, "take")
                else self.provenance
            ),
            geometry_metadata=(
                None
                if self.geometry_metadata is None
                else self.geometry_metadata.take(normalized)
            ),
            index_plan=taken_index_plan,
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
    if len(geometry_columns) == 1 and geometry_columns[0].name == geometry_name:
        simple_order = requested_order or [*list(frame.columns), geometry_name]
        try:
            simple_order_set = set(simple_order)
            frame_column_set = set(frame.columns)
        except TypeError:
            simple_order_set = None
            frame_column_set = None
        if (
            simple_order_set is not None
            and frame.columns.is_unique
            and len(simple_order) == len(simple_order_set)
            and geometry_name in simple_order_set
            and simple_order_set == frame_column_set | {geometry_name}
        ):
            attribute_order = [name for name in simple_order if name != geometry_name]
            ordered_frame = (
                frame.loc[:, attribute_order].copy(deep=False)
                if attribute_order
                else pd.DataFrame(index=attributes.index)
            )
            ordered_frame.index = attributes.index
            geometry_position = simple_order.index(geometry_name)
            geometry_values = pd.Series(
                active_geometry.values,
                index=attributes.index,
                name=geometry_name,
                copy=False,
            )
            if geometry_position == len(attribute_order):
                ordered_frame[geometry_name] = geometry_values
            else:
                ordered_frame.insert(
                    geometry_position,
                    geometry_name,
                    geometry_values,
                )
            ordered_frame.__class__ = GeoDataFrame
            ordered_frame._geometry_column_name = geometry_name
            ordered_frame.attrs = frame.attrs.copy()
            return ordered_frame
    attribute_positions: dict[str, list[int]] = {}
    for position, name in enumerate(frame.columns):
        attribute_positions.setdefault(name, []).append(position)
    consumed_attribute_positions: set[int] = set()
    consumed_geometry_names: set[str] = set()
    ordered_pieces: list[pd.DataFrame] = []

    def append_attribute(name: str) -> bool:
        for position in attribute_positions.get(name, ()):
            if position in consumed_attribute_positions:
                continue
            piece = frame.iloc[:, [position]].copy(deep=False)
            piece.columns = [name]
            ordered_pieces.append(piece)
            consumed_attribute_positions.add(position)
            return True
        return False

    def append_geometry(name: str) -> bool:
        if name not in geometry_series or name in consumed_geometry_names:
            return False
        ordered_pieces.append(geometry_series[name].to_frame(name=name))
        consumed_geometry_names.add(name)
        return True

    unknown: list[str] = []
    for name in requested_order:
        if append_geometry(name) or append_attribute(name):
            continue
        unknown.append(name)
    if unknown:
        raise ValueError(
            f"column_order contains columns that are not present in the export payload: {unknown}"
        )
    for position, name in enumerate(frame.columns):
        if position not in consumed_attribute_positions:
            append_attribute(name)
    for name in geometry_names:
        append_geometry(name)

    ordered_frame = (
        pd.concat(ordered_pieces, axis=1)
        if ordered_pieces
        else pd.DataFrame(index=attributes.index)
    )
    ordered_frame.index = attributes.index
    rebuilt = GeoDataFrame(
        ordered_frame,
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
    "NativeGeometryProvenance",
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
