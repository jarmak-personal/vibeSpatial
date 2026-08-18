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
from vibespatial.runtime.residency import Residency, combined_residency

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
        if d_positions.dtype.kind == "i":
            return d_positions
        return d_positions.astype(cp.int64, copy=False)

    positions = np.asarray(row_positions)
    if positions.dtype == bool:
        positions = np.flatnonzero(positions)
    return np.asarray(positions, dtype=np.int64)


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _pandas_assignment_values(values: Any) -> Any:
    from vibespatial.api._native_expression import NativeExpression

    if isinstance(values, NativeExpression):
        values = values.values
    if _is_device_array(values):
        return _host_array(
            values,
            dtype=None,
            strict_disallowed=False,
            surface="vibespatial.api.NativeAttributeTable",
            operation="attribute_assignment_to_host",
            reason="device attribute values materialized for pandas column assignment",
        )
    return values


def _arrow_compatible_pandas_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower lazy native public arrays to physical dtypes for Arrow export.

    PyArrow records pandas extension dtype names in schema metadata.  The
    private ``vibespatial_*`` dtypes are execution carriers, not persistent
    interchange types, so exporting them through ``Table.from_pandas`` creates
    files that a later Arrow reader cannot reconstruct.  Materialize only
    those native arrays at this terminal boundary and preserve every ordinary
    pandas extension dtype unchanged.
    """
    native_module = "vibespatial.api._native_public_arrays"
    native_columns = [
        column for column in frame.columns if type(frame[column].array).__module__ == native_module
    ]
    index_values = getattr(frame.index, "_values", None)
    native_index = type(index_values).__module__ == native_module
    if not native_columns and not native_index:
        return frame

    result = frame.copy(deep=False)
    for column in native_columns:
        result[column] = frame[column].array.to_numpy(copy=False)
    if native_index:
        result.index = pd.Index(
            index_values.to_numpy(copy=False),
            name=frame.index.name,
        )
    return result


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
    if table.is_device_backed:
        return "device"
    if table.dataframe is not None:
        return "pandas"
    if table.arrow_table is not None:
        return "arrow"
    if table.loader is not None:
        return "loader"
    return "unknown"


def _geometry_storage_label(geometry: GeometryNativeResult) -> str:
    owned = getattr(geometry, "owned", None)
    if owned is not None:
        residency = getattr(getattr(owned, "residency", None), "value", None)
        return f"owned:{residency or 'unknown'}"
    composition = getattr(geometry, "composition", None)
    if composition is not None:
        residency = getattr(composition.residency, "value", composition.residency)
        return f"composition:{residency}"
    return "geoseries"


def _attribute_column_table(
    attributes: NativeAttributeTable,
    column: Any,
) -> NativeAttributeTable | None:
    if attributes.parts is None:
        return attributes if column in tuple(attributes.columns) else None
    for part in attributes.parts:
        if column in tuple(part.columns):
            return part
    return None


def _lazy_public_attribute_frame(
    attributes: NativeAttributeTable,
    *,
    surface: str,
) -> pd.DataFrame | None:
    """Build public attribute columns without exporting device tables eagerly."""
    if attributes.to_pandas_kwargs:
        # ``to_pandas_kwargs`` is an explicit public compatibility request.  A
        # native extension facade would hide the requested pandas dtype mapping.
        return None
    try:
        import pyarrow as pa
    except ImportError:  # pragma: no cover - pyarrow is present in normal envs
        pa = None
    if pa is not None:
        tables = attributes.parts or (attributes,)
        if any(
            table.schema_override is not None
            and any(pa.types.is_nested(field.type) for field in table.schema_override)
            for table in tables
        ):
            # Preserve Arrow's terminal pandas semantics for nested columns,
            # including its documented TypeError without a compatible mapper.
            return None
    try:
        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_public_arrays import (
            NativeAttributeColumnArray,
            NativeNumericExpressionArray,
        )
    except Exception:
        return None

    if attributes.device_table is None and attributes.parts is None:
        return None

    columns: dict[Any, Any] = {}
    for column in attributes.columns:
        table = _attribute_column_table(attributes, column)
        if table is None:
            return None
        if table.device_table is None:
            if table.dataframe is not None:
                columns[column] = table.dataframe[column]
                continue
            return None
        numeric_arrays = table.numeric_column_arrays((column,))
        values = None if numeric_arrays is None else numeric_arrays.get(column)
        if values is not None and _is_device_array(values):
            columns[column] = pd.Series(
                NativeNumericExpressionArray(
                    NativeExpression(
                        operation=f"attribute.{column}",
                        values=values,
                        source_token=None,
                        source_row_count=len(table),
                        dtype=str(getattr(values, "dtype", "")) or None,
                        precision="source",
                    ),
                    export_surface=surface,
                    export_operation="native_attribute_column_to_public_series",
                ),
                index=attributes.index,
                name=column,
            )
        else:
            columns[column] = pd.Series(
                NativeAttributeColumnArray(
                    table,
                    column,
                    export_surface=surface,
                    export_operation="native_attribute_column_to_public_series",
                ),
                index=attributes.index,
                name=column,
            )
    frame = pd.DataFrame(index=attributes.index)
    for column, series in columns.items():
        frame[column] = series
    return frame


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


def _append_pandas_index_to_arrow(table: Any, index: pd.Index, preserve_index):
    if preserve_index is False:
        return table
    import pyarrow as pa
    import pyarrow.pandas_compat as pandas_compat

    index_table = pa.Table.from_pandas(
        pd.DataFrame(index=index),
        preserve_index=preserve_index,
    )
    result = table
    for field, column in zip(index_table.schema, index_table.columns, strict=True):
        if field.name in result.column_names:
            continue
        result = result.append_column(field, column)

    index_metadata = index_table.schema.metadata or {}
    try:
        pandas_index_metadata = json.loads(index_metadata[b"pandas"].decode("utf-8"))
        index_descriptors = pandas_index_metadata["index_columns"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return result

    if isinstance(index, pd.MultiIndex):
        index_levels = [index.get_level_values(level) for level in range(index.nlevels)]
    else:
        index_levels = [index]
    if len(index_levels) != len(index_descriptors):
        return result

    attribute_fields = list(table.schema)
    columns_to_convert = []
    for field in attribute_fields:
        try:
            dtype = field.type.to_pandas_dtype()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            dtype = object
        try:
            columns_to_convert.append(pd.Series([], dtype=dtype, name=field.name))
        except (TypeError, ValueError):
            columns_to_convert.append(pd.Series([], dtype=object, name=field.name))

    column_names = [field.name for field in attribute_fields]
    pandas_metadata = pandas_compat.construct_metadata(
        columns_to_convert,
        pd.DataFrame(index=pd.RangeIndex(0), columns=column_names),
        column_names,
        index_levels,
        index_descriptors,
        preserve_index,
        [
            *(field.type for field in attribute_fields),
            *(field.type for field in index_table.schema),
        ],
        column_field_names=column_names,
    )
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update(pandas_metadata)
    return result.replace_schema_metadata(schema_metadata)


def _is_lazy_native_public_index(index: pd.Index) -> bool:
    values = getattr(index, "_values", None)
    return values.__class__.__name__ == "NativeIndexLabelsArray"


def _index_equals_without_lazy_native_export(left: pd.Index, right: pd.Index) -> bool:
    if left is right:
        return True
    if _is_lazy_native_public_index(left) or _is_lazy_native_public_index(right):
        return False
    return left.equals(right)


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
        return _normalize_pandas_range_metadata(
            pa.Table.from_arrays(
                [table.column(index) for index in range(table.num_columns)],
                names=names,
                metadata=table.schema.metadata,
            )
        )
    fields = []
    for index, name in enumerate(names):
        try:
            fields.append(schema.field(name))
        except KeyError:
            fields.append(pa.field(name, table.column(index).type))
    return _normalize_pandas_range_metadata(
        pa.Table.from_arrays(
            [table.column(index) for index in range(table.num_columns)],
            schema=pa.schema(fields, metadata=schema.metadata),
        )
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


def _take_pandas_frame_rows(
    frame: pd.DataFrame,
    row_positions,
    *,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Take dataframe rows through column arrays, preserving lazy native arrays.

    Pandas' block-manager take/reindex paths assume ndarray-shaped internal
    blocks. Public native extension arrays intentionally do not expose those
    block shapes, so native terminal/export paths must project rows through the
    ExtensionArray protocol instead of private pandas internals.
    """
    positions = np.asarray(row_positions, dtype=np.intp)
    out_index = pd.RangeIndex(int(positions.size)) if index is None else index
    if frame.shape[1] == 0:
        return pd.DataFrame(index=out_index)

    series_parts: list[pd.Series] = []
    for column_position, column_name in enumerate(frame.columns):
        column_values = frame.iloc[:, column_position]
        array = getattr(column_values, "array", None)
        taken = None
        if array is not None and hasattr(array, "take"):
            try:
                taken = array.take(positions, allow_fill=False)
            except Exception:
                taken = None
        if taken is None:
            values = column_values.to_numpy(copy=False)
            taken = np.asarray(values)[positions]
        series_parts.append(pd.Series(taken, index=out_index, name=column_name))

    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
    result = pd.concat(series_parts, axis=1, **concat_kwargs)
    result.index = out_index
    result.columns = frame.columns
    return result


def _is_admissible_pandas_numeric_series(series: pd.Series) -> bool:
    dtype = series.dtype
    return bool(
        not series.hasnans
        and (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype))
    )


def _is_admissible_arrow_numeric_type(dtype) -> bool:
    import pyarrow as pa

    return bool(
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)
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


def _pylibcudf_column_from_device(values):
    from vibespatial.cuda._runtime import pylibcudf_column_from_device

    return pylibcudf_column_from_device(values)


def _pylibcudf_table_from_arrow(table):
    from vibespatial.cuda._runtime import pylibcudf_table_from_arrow

    return pylibcudf_table_from_arrow(table)


def _native_expression_device_column(value, *, row_count: int):
    from vibespatial.api._native_expression import NativeExpression

    if not isinstance(value, NativeExpression):
        return None
    if value.source_row_count is not None and int(value.source_row_count) != int(row_count):
        raise ValueError("NativeExpression column row count must match attributes")
    if not value.is_device:
        return None

    import cupy as cp
    values = cp.asarray(value.values)
    if int(values.size) != int(row_count):
        raise ValueError("NativeExpression column row count must match attributes")
    dtype = np.dtype(values.dtype)
    if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
        return None
    return _pylibcudf_column_from_device(values)


def _assigned_device_column(value, *, row_count: int):
    from vibespatial.runtime._runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None
    expression_column = _native_expression_device_column(value, row_count=row_count)
    if expression_column is not None:
        return expression_column
    from vibespatial.api._native_expression import NativeExpression

    if isinstance(value, NativeExpression):
        return None

    if _is_device_array(value):
        import cupy as cp
        values = cp.asarray(value)
        if values.ndim != 1 or int(values.size) != int(row_count):
            raise ValueError("assigned column row count must match attributes")
        dtype = np.dtype(values.dtype)
        if not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)):
            return None
        return _pylibcudf_column_from_device(values)

    if pd.api.types.is_scalar(value):
        host_values = np.full(int(row_count), value)
    elif isinstance(value, pd.Series):
        if len(value) != int(row_count):
            raise ValueError("assigned column row count must match attributes")
        if not _is_admissible_pandas_numeric_series(value):
            try:
                import pyarrow as pa
            except ModuleNotFoundError:
                return None
            try:
                arrow_array = pa.array(value)
                if int(len(arrow_array)) != int(row_count):
                    raise ValueError("assigned column row count must match attributes")
                return _pylibcudf_table_from_arrow(
                    pa.table({"__assigned__": arrow_array})
                ).columns()[0]
            except Exception:
                return None
        host_values = value.to_numpy(copy=False)
    else:
        host_values = np.asarray(value)
        if host_values.ndim != 1 or int(host_values.size) != int(row_count):
            raise ValueError("assigned column row count must match attributes")

    dtype = np.dtype(host_values.dtype)
    numeric_or_bool = np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
    if not numeric_or_bool or bool(pd.isna(host_values).any()):
        try:
            import pyarrow as pa
        except ModuleNotFoundError:
            return None
        try:
            return _pylibcudf_table_from_arrow(
                pa.table({"__assigned__": pa.array(host_values)})
            ).columns()[0]
        except Exception:
            return None

    import cupy as cp
    if numeric_or_bool:
        return _pylibcudf_column_from_device(cp.asarray(host_values))

    return None


def _assigned_device_attribute_table(
    assigned: Mapping[Any, Any],
    *,
    row_count: int,
    index_override: pd.Index,
    to_pandas_kwargs: dict[str, Any] | None,
) -> NativeAttributeTable | None:
    if not assigned:
        return None
    try:
        import pyarrow as pa
        import pylibcudf as plc
    except ModuleNotFoundError:
        return None

    output_columns = []
    fields = []
    for name, value in assigned.items():
        column = _assigned_device_column(value, row_count=row_count)
        if column is None:
            return None
        output_columns.append(column)
        fields.append(pa.field(str(name), column.type().to_arrow()))
    return NativeAttributeTable(
        device_table=plc.Table(output_columns),
        index_override=index_override,
        column_override=tuple(assigned),
        schema_override=pa.schema(fields),
        to_pandas_kwargs=to_pandas_kwargs,
    )


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
    can_sort: bool


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
                can_sort=True,
            )
        return NativeAttributeColumnPolicy(
            category="nullable-numeric-bool-movement-only",
            arrow_type=str(arrow_type),
            null_count=int(null_count),
            can_project_take=True,
            can_compute_numeric=False,
            can_sort=True,
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
    can_sort = bool(
        pa.types.is_decimal(arrow_type)
        or pa.types.is_dictionary(arrow_type)
        or pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_temporal(arrow_type)
        or pa.types.is_duration(arrow_type)
        or pa.types.is_null(arrow_type)
    )
    return NativeAttributeColumnPolicy(
        category=category,
        arrow_type=str(arrow_type),
        null_count=int(null_count),
        can_project_take=True,
        can_compute_numeric=False,
        can_sort=can_sort,
    )


@dataclass(frozen=True)
class NativeAttributeTable:
    """Attribute payload that can stay columnar without requiring pandas storage."""

    dataframe: pd.DataFrame | None = None
    arrow_table: Any | None = None
    device_table: Any | None = None
    loader: Callable[[], pd.DataFrame] | None = None
    parts: tuple[NativeAttributeTable, ...] | None = None
    index_override: pd.Index | None = None
    column_override: tuple[Any, ...] | None = None
    schema_override: Any | None = None
    to_pandas_kwargs: dict[str, Any] | None = None
    row_positions: Any | None = None

    @property
    def is_device_backed(self) -> bool:
        """Whether all concrete attribute storage remains on device."""
        if self.device_table is not None:
            return True
        return self.parts is not None and all(part.is_device_backed for part in self.parts)

    def __post_init__(self) -> None:
        provided = sum(
            value is not None
            for value in (
                self.dataframe,
                self.arrow_table,
                self.device_table,
                self.loader,
                self.parts,
            )
        )
        if provided != 1:
            raise ValueError(
                "NativeAttributeTable requires exactly one of dataframe, arrow_table, "
                "device_table, loader, or parts"
            )
        if self.to_pandas_kwargs is None:
            object.__setattr__(self, "to_pandas_kwargs", {})
        if self.parts is not None:
            if not self.parts:
                raise ValueError("NativeAttributeTable parts cannot be empty")
            row_count = len(self.parts[0])
            index = self.index_override if self.index_override is not None else self.parts[0].index
            columns = []
            for part in self.parts:
                if len(part) != row_count:
                    raise ValueError("NativeAttributeTable parts must have equal row counts")
                if not part.index.equals(index):
                    raise ValueError("NativeAttributeTable parts must share an index")
                columns.extend(tuple(part.columns))
            if len(set(columns)) != len(columns):
                raise ValueError("NativeAttributeTable parts must not repeat columns")
            if self.index_override is None:
                object.__setattr__(self, "index_override", index)
            if self.column_override is None:
                object.__setattr__(self, "column_override", tuple(columns))
        if (
            self.arrow_table is not None
            and self.index_override is None
            and self.row_positions is None
        ):
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
            if type(self.device_table).__module__.startswith("pylibcudf."):
                from vibespatial.cuda._runtime import pylibcudf_mark_produced

                pylibcudf_mark_produced(self.device_table)
            row_count = _device_table_row_count(self.device_table)
            if self.index_override is None and self.row_positions is None:
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
        if self.row_positions is not None:
            positions = _normalize_row_selection(self.row_positions)
            object.__setattr__(self, "row_positions", positions)
            row_count = _row_aligned_size(positions)
            if self.index_override is None:
                object.__setattr__(self, "index_override", pd.RangeIndex(row_count))
            elif len(self.index_override) != row_count:
                raise ValueError(
                    "NativeAttributeTable row_positions length must match index_override"
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
        if self.parts is not None:
            return self.index_override
        return self.index_override

    @property
    def columns(self) -> pd.Index:
        if self.dataframe is not None:
            return self.dataframe.columns
        if self.parts is not None:
            return pd.Index(self.column_override)
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
            raise TypeError("NativeAttributeTable loader must return a pandas DataFrame")
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
        if column_positions is None or any(column not in column_positions for column in requested):
            return None
        if self.parts is not None:
            out: dict[Any, Any] = {}
            remaining = set(requested)
            for part in self.parts:
                part_requested = tuple(column for column in requested if column in part.columns)
                if not part_requested:
                    continue
                arrays = part.numeric_column_arrays(part_requested)
                if arrays is None:
                    return None
                out.update(arrays)
                remaining.difference_update(arrays)
            if remaining:
                return None
            return {column: out[column] for column in requested}
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
                if chunked.null_count or not _is_admissible_arrow_numeric_type(chunked.type):
                    return None
                out[logical_name] = chunked.combine_chunks().to_numpy(zero_copy_only=False)
            return out

        if self.device_table is not None:
            source_columns = self.device_table.columns()
            policies = self.device_column_policies(requested)
            out = {}
            for column in requested:
                policy = policies.get(column)
                if policy is None or not policy.can_compute_numeric:
                    return None
                values = _pylibcudf_numeric_column_view(source_columns[column_positions[column]])
                if values is None:
                    return None
                if self.row_positions is not None:
                    import cupy as cp

                    values = cp.asarray(values)[cp.asarray(self.row_positions, dtype=cp.int64)]
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
        requested = tuple(self.columns) if columns is None else tuple(dict.fromkeys(columns))
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
        if self.row_positions is not None:
            return self._physicalize_device_row_view().grouped_device_take_columns(
                grouped,
                reducers,
            )
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
            reducer not in {"first", "last"} for reducer in normalized_reducers.values()
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
        from vibespatial.cuda._runtime import pylibcudf_current_stream

        stream = pylibcudf_current_stream(self.device_table)
        for column in requested:
            selected_rows = (
                first_positions if normalized_reducers[column] == "first" else last_positions
            )
            gather_map = _pylibcudf_column_from_device(
                selected_rows.astype(target_dtype, copy=False)
            )
            gathered = plc.copying.gather(
                plc.Table([source_columns[positions[column]]]),
                gather_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )
            output_column = gathered.columns()[0]
            output_columns.append(output_column)
            fields.append(_field_for_device_column(column, output_column, self.schema_override))

        index_override = (
            grouped.output_index_plan.index
            if grouped.output_index_plan is not None and grouped.output_index_plan.index is not None
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
        if column_positions is None or any(column not in column_positions for column in requested):
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

    def _numeric_device_table_to_pandas(self) -> pd.DataFrame | None:
        """Export all-valid numeric device columns without Arrow conversion."""
        if self.device_table is None or not hasattr(self.device_table, "columns"):
            return None
        requested = tuple(self.columns)
        if not requested:
            return pd.DataFrame(index=self.index)
        policies = self.device_column_policies(requested)
        device_arrays: dict[Any, Any] = {}
        total_bytes = 0
        arrays = self.numeric_column_arrays(requested)
        if arrays is None:
            return None
        for column in requested:
            policy = policies.get(column)
            if policy is None or not policy.can_compute_numeric:
                return None
            device_values = arrays.get(column)
            if device_values is None:
                return None
            device_arrays[column] = device_values
            total_bytes += int(device_values.size) * int(device_values.dtype.itemsize)

        record_materialization_event(
            surface="vibespatial.api.NativeAttributeTable.to_pandas",
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="device_numeric_attributes_to_pandas",
            reason="device numeric attribute table exported to pandas",
            detail=(f"rows={len(self)}, columns={len(requested)}, bytes={total_bytes}"),
            d2h_transfer=True,
        )
        from vibespatial.cuda._runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        columns = {
            column: np.asarray(
                runtime.copy_device_to_host(
                    values,
                    reason="device attribute numeric column host export",
                )
            )
            for column, values in device_arrays.items()
        }
        return pd.DataFrame(columns, index=self.index)

    def to_pandas(self, *, copy: bool = False, **kwargs) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe.copy(deep=copy) if copy else self.dataframe
        if self.parts is not None:
            frames = [part.to_pandas(copy=False, **kwargs) for part in self.parts]
            concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
            frame = pd.concat(frames, axis=1, **concat_kwargs)
            if self.column_override is not None:
                frame = frame.loc[:, list(self.column_override)]
            if not frame.index.equals(self.index):
                frame = frame.copy(deep=False)
                frame.index = self.index
            return frame.copy(deep=copy) if copy else frame
        if self.loader is not None:
            frame = self._materialize_loaded_frame()
            return frame.copy(deep=copy) if copy else frame
        if self.device_table is not None:
            to_pandas_kwargs = {**(self.to_pandas_kwargs or {}), **kwargs}
            numeric_frame = (
                None if to_pandas_kwargs else self._numeric_device_table_to_pandas()
            )
            if numeric_frame is not None:
                return numeric_frame.copy(deep=copy) if copy else numeric_frame
            frame = self.to_arrow(index=False).to_pandas(**to_pandas_kwargs)
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

        if self.parts is not None:
            requested_columns = tuple(self.columns if columns is None else columns)
            if all(part.device_table is not None for part in self.parts):
                try:
                    import pylibcudf as plc
                except ModuleNotFoundError:
                    pass
                else:
                    record_materialization_event(
                        surface="vibespatial.api.NativeAttributeTable.to_arrow",
                        boundary=MaterializationBoundary.USER_EXPORT,
                        operation="device_attributes_to_arrow",
                        reason="device attribute table exported to host Arrow",
                        detail=(
                            f"rows={len(self)}, columns={len(requested_columns)}, bytes=unknown"
                        ),
                        d2h_transfer=True,
                    )
                    columns_by_name = {}
                    fields_by_name = {}
                    for part in self.parts:
                        part_requested = tuple(
                            column for column in requested_columns if column in tuple(part.columns)
                        )
                        if not part_requested:
                            continue
                        part_table = part._gathered_device_table(part_requested)
                        for logical_name, column in zip(
                            part_requested,
                            part_table.columns(),
                            strict=True,
                        ):
                            columns_by_name[logical_name] = column
                            fields_by_name[logical_name] = _field_for_device_column(
                                logical_name,
                                column,
                                part.schema_override,
                            )
                    if all(column in columns_by_name for column in requested_columns):
                        output_columns = [columns_by_name[column] for column in requested_columns]
                        fields = [fields_by_name[column] for column in requested_columns]
                        from vibespatial.cuda._runtime import pylibcudf_to_arrow

                        table = pylibcudf_to_arrow(plc.Table(output_columns))
                        table = _rename_device_arrow_table(
                            table,
                            requested_columns,
                            schema=pa.schema(fields),
                        )
                        return _append_pandas_index_to_arrow(
                            table,
                            self.index,
                            index,
                        )
            frame = self.to_pandas(copy=False)
            requested_columns = None if columns is None else list(columns)
            if requested_columns is not None:
                frame = frame.loc[:, requested_columns]
            return pa.Table.from_pandas(
                _arrow_compatible_pandas_frame(frame),
                preserve_index=index,
            )

        if self.loader is not None:
            frame = self.to_pandas(copy=False)
            requested_columns = None if columns is None else list(columns)
            if requested_columns is not None:
                frame = frame.loc[:, requested_columns]
            return pa.Table.from_pandas(
                _arrow_compatible_pandas_frame(frame),
                preserve_index=index,
            )

        requested_columns = None if columns is None else list(columns)
        if self.device_table is not None:
            if index not in (None, False):
                frame = self.to_pandas(copy=False)
                if requested_columns is not None:
                    frame = frame.loc[:, requested_columns]
                return pa.Table.from_pandas(
                    _arrow_compatible_pandas_frame(frame),
                    preserve_index=index,
                )
            record_materialization_event(
                surface="vibespatial.api.NativeAttributeTable.to_arrow",
                boundary=MaterializationBoundary.USER_EXPORT,
                operation="device_attributes_to_arrow",
                reason="device attribute table exported to host Arrow",
                detail=(f"rows={len(self)}, columns={len(self.columns)}, bytes=unknown"),
                d2h_transfer=True,
            )
            table = (
                self._gathered_device_table(requested_columns)
                if self.row_positions is not None
                else self.device_table
            )
            from vibespatial.cuda._runtime import pylibcudf_to_arrow

            table = pylibcudf_to_arrow(table)
            table = _rename_device_arrow_table(
                table,
                tuple(requested_columns) if requested_columns is not None else self.column_override,
                schema=self.schema_override,
            )
            if requested_columns is not None and self.row_positions is None:
                table = table.select([str(column) for column in requested_columns])
            return table
        can_skip_index = index is False or (
            index is None
            and isinstance(self.index_override, pd.RangeIndex)
            and self.index_override.start == 0
            and self.index_override.step == 1
            and list(self.index_override.names) == [None]
        )
        if self.arrow_table is not None and can_skip_index:
            table = self.arrow_table
            if requested_columns is not None:
                table = table.select(requested_columns)
            return table

        frame = self.to_pandas(copy=False)
        if requested_columns is not None:
            frame = frame.loc[:, requested_columns]
        return pa.Table.from_pandas(
            _arrow_compatible_pandas_frame(frame),
            preserve_index=index,
        )

    def to_pylibcudf_columns(self, columns) -> list[Any]:
        requested_columns = list(columns)
        if self.device_table is None:
            table = _pylibcudf_table_from_arrow(
                self.to_arrow(index=False, columns=requested_columns)
            )
            return table.columns()

        from vibespatial.cuda._runtime import pylibcudf_current_stream

        source_columns = self.device_table.columns()
        by_name = {
            column_name: source_columns[index]
            for index, column_name in enumerate(self.column_override or ())
        }
        output_columns = [by_name[column] for column in requested_columns]
        if self.row_positions is None:
            pylibcudf_current_stream(self.device_table)
            return output_columns
        return self._gathered_device_table(requested_columns).columns()

    def _gathered_device_table(self, columns=None):
        """Materialize this row-indirected device view as a pylibcudf table."""
        if self.device_table is None:
            raise ValueError("device gather requires a device-backed attribute table")
        import cupy as cp
        import pylibcudf as plc

        from vibespatial.cuda._runtime import pylibcudf_current_stream

        stream = pylibcudf_current_stream(self.device_table)
        source_columns = self.device_table.columns()
        requested = tuple(self.columns if columns is None else columns)
        positions = _column_position_map(self.columns)
        if positions is None or any(column not in positions for column in requested):
            raise KeyError("requested columns are not present in native device table")
        output_columns = [source_columns[positions[column]] for column in requested]
        if self.row_positions is None:
            return plc.Table(output_columns)
        row_positions = cp.asarray(self.row_positions, dtype=cp.int64)
        target_dtype = (
            cp.int32
            if _device_table_row_count(self.device_table) <= np.iinfo(np.int32).max
            else cp.int64
        )
        gather_map = _pylibcudf_column_from_device(
            row_positions.astype(target_dtype, copy=False)
        )
        return plc.copying.gather(
            plc.Table(output_columns),
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
        )

    def _physicalize_device_row_view(self, columns=None) -> NativeAttributeTable:
        """Return a device table with row_positions applied."""
        if self.row_positions is None:
            return self.project_columns(tuple(self.columns if columns is None else columns)) or self
        import pyarrow as pa

        requested = tuple(self.columns if columns is None else columns)
        gathered = self._gathered_device_table(requested)
        fields = [
            _field_for_device_column(name, column, self.schema_override)
            for name, column in zip(requested, gathered.columns(), strict=True)
        ]
        return type(self)(
            device_table=gathered,
            index_override=self.index,
            column_override=requested,
            schema_override=pa.schema(
                fields,
                metadata=None if self.schema_override is None else self.schema_override.metadata,
            ),
            to_pandas_kwargs=self.to_pandas_kwargs,
        )

    def promote_numeric_to_device(self) -> NativeAttributeTable | None:
        """Return all-valid numeric/bool attributes as a device table."""
        if self.device_table is not None:
            return self
        if self.loader is not None:
            return None
        from vibespatial.runtime._runtime import has_gpu_runtime

        if not has_gpu_runtime():
            return None

        requested = tuple(self.columns)
        if not requested:
            return self
        arrays = self.numeric_column_arrays(requested)
        if arrays is None or any(column not in arrays for column in requested):
            return None

        try:
            import cupy as cp
            import pyarrow as pa
            import pylibcudf as plc
        except ModuleNotFoundError:
            return None

        output_columns = []
        fields = []
        for column in requested:
            device_values = cp.asarray(arrays[column])
            if device_values.ndim != 1 or int(device_values.size) != len(self):
                return None
            device_column = _pylibcudf_column_from_device(device_values)
            output_columns.append(device_column)
            fields.append(_field_for_device_column(column, device_column, self.schema_override))

        return type(self)(
            device_table=plc.Table(output_columns),
            index_override=self.index,
            column_override=requested,
            schema_override=pa.schema(
                fields,
                metadata=None if self.schema_override is None else self.schema_override.metadata,
            ),
            to_pandas_kwargs=self.to_pandas_kwargs,
        )

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
        if self.parts is not None:
            logical_columns = tuple([*self.columns, name])
            assigned = self.assign_columns({name: values}, columns=logical_columns)
            if assigned is None:
                raise ValueError("mixed native attribute table could not represent assigned column")
            return assigned
        if self.loader is not None:
            declared_columns = tuple(self.column_override or ())
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).copy(deep=False)
                frame[name] = _pandas_assignment_values(values)
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
            assigned = self.assign_columns({name: values}, columns=logical_columns)
            if assigned is not None:
                return assigned
            table = self.arrow_table.append_column(
                str(name),
                pa.array(_pandas_assignment_values(values)),
            )
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
        frame[name] = _pandas_assignment_values(values)
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
        assigned = {name: values for name, values in values_by_name.items() if name in requested}
        known = set(self.columns)
        if any(name not in known and name not in assigned for name in requested):
            return None
        if not assigned and requested == tuple(self.columns):
            return self

        from vibespatial.api._native_expression import NativeExpression

        if self.device_table is None and any(
            _is_device_array(value) or (isinstance(value, NativeExpression) and value.is_device)
            for value in assigned.values()
        ):
            if self.parts is None and self.loader is None:
                promoted = self.promote_numeric_to_device()
                if promoted is not None and promoted.device_table is not None:
                    return promoted.assign_columns(values_by_name, columns=requested)
            assigned_table = _assigned_device_attribute_table(
                assigned,
                row_count=len(self),
                index_override=self.index,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
            if assigned_table is not None:
                base_columns = tuple(
                    column
                    for column in self.columns
                    if column in requested and column not in assigned
                )
                base = self.project_columns(base_columns) if base_columns else None
                tables = [table for table in (base, assigned_table) if table is not None]
                combined = type(self).combine_columns(
                    tables,
                    index_override=self.index,
                )
                if combined is not None:
                    return combined.project_columns(requested)

        if self.parts is not None:
            if not assigned:
                return self.project_columns(requested)

            assigned_frame = pd.DataFrame(index=self.index)
            for name, values in assigned.items():
                assigned_frame[name] = _pandas_assignment_values(values)
            assigned_table = type(self)(
                dataframe=assigned_frame.loc[:, list(assigned)],
            )
            base_columns = tuple(
                column for column in self.columns if column in requested and column not in assigned
            )
            base = self.project_columns(base_columns) if base_columns else None
            tables = [table for table in (base, assigned_table) if table is not None]
            combined = type(self).combine_columns(
                tables,
                index_override=self.index,
            )
            if combined is None:
                return None
            return combined.project_columns(requested)

        if self.loader is not None:
            parent = self

            def _load() -> pd.DataFrame:
                frame = parent.to_pandas(copy=False).copy(deep=False)
                for name, values in assigned.items():
                    frame[name] = _pandas_assignment_values(values)
                return frame.loc[:, list(requested)]

            return type(self).from_loader(
                _load,
                index_override=self.index,
                columns=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

        if self.arrow_table is not None:
            import pyarrow as pa

            if any(
                _is_device_array(value) or (isinstance(value, NativeExpression) and value.is_device)
                for value in assigned.values()
            ):
                promoted = self.promote_numeric_to_device()
                if promoted is None:
                    return None
                return promoted.assign_columns(values_by_name, columns=requested)

            source = {
                name: self.arrow_table.column(position)
                for position, name in enumerate(self.column_override or ())
            }
            arrays = []
            for name in requested:
                if name in assigned:
                    value = assigned[name]
                    if isinstance(value, NativeExpression):
                        value = np.asarray(value.values)
                    arrays.append(pa.array(value))
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
            if self.row_positions is not None:
                materialized = self._physicalize_device_row_view()
                return materialized.assign_columns(values_by_name, columns=requested)
            row_count = len(self)
            assigned_columns = {}
            can_represent_assigned_on_device = True
            for name, value in assigned.items():
                column = _assigned_device_column(value, row_count=row_count)
                if column is None:
                    can_represent_assigned_on_device = False
                    break
                assigned_columns[name] = column
            if not can_represent_assigned_on_device:
                if any(
                    isinstance(value, NativeExpression) or _is_device_array(value)
                    for value in assigned.values()
                ):
                    return None
                assigned_frame = pd.DataFrame(index=self.index)
                for name, values in assigned.items():
                    assigned_frame[name] = _pandas_assignment_values(values)
                assigned_table = type(self)(
                    dataframe=assigned_frame.loc[:, list(assigned)],
                )
                base_columns = tuple(
                    column
                    for column in self.columns
                    if column in requested and column not in assigned
                )
                base = self.project_columns(base_columns) if base_columns else None
                tables = [table for table in (base, assigned_table) if table is not None]
                combined = type(self).combine_columns(
                    tables,
                    index_override=self.index,
                )
                if combined is None:
                    return None
                return combined.project_columns(requested)
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
            frame[name] = _pandas_assignment_values(values)
        return type(self)(dataframe=frame.loc[:, list(requested)])

    def project_columns(self, columns: tuple[Any, ...]) -> NativeAttributeTable | None:
        """Return attributes projected to an exact logical column order."""
        requested = tuple(columns)
        positions = _column_position_map(self.columns)
        if positions is None or any(column not in positions for column in requested):
            return None
        if requested == tuple(self.columns):
            return self

        if self.parts is not None:
            projected_parts = []
            for part in self.parts:
                part_columns = tuple(column for column in requested if column in part.columns)
                if not part_columns:
                    continue
                projected = part.project_columns(part_columns)
                if projected is None:
                    return None
                projected_parts.append(projected)
            if not projected_parts:
                return type(self)(dataframe=pd.DataFrame(index=self.index))
            return type(self)(
                parts=tuple(projected_parts),
                index_override=self.index,
                column_override=requested,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )

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
                row_positions=self.row_positions,
            )

        frame = self.dataframe.loc[:, list(requested)].copy(deep=False)
        return type(self)(dataframe=frame)

    def with_index(self, index: pd.Index) -> NativeAttributeTable:
        """Return the same attribute payload with a compatibility index."""
        if _index_equals_without_lazy_native_export(self.index, index):
            return self
        if self.parts is not None:
            return type(self)(
                parts=tuple(part.with_index(index) for part in self.parts),
                index_override=index,
                column_override=self.column_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
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
            row_positions=self.row_positions,
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

        if self.row_positions is not None:
            return self._physicalize_device_row_view().reset_index_deferred()

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
                    column = _pylibcudf_column_from_device(values)
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
        if self.parts is not None:
            return type(self)(
                parts=tuple(part.rename_columns(mapping) for part in self.parts),
                index_override=self.index,
                column_override=renamed_logical,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
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
                arrow_table=self.arrow_table.rename_columns(
                    [str(name) for name in renamed_logical]
                ),
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
                row_positions=self.row_positions,
            )
        frame = self.dataframe.copy(deep=False)
        frame.columns = _renamed_pandas_columns_like(
            self.dataframe.columns,
            renamed_logical,
        )
        return type(self)(dataframe=frame)

    def take(self, row_positions, *, preserve_index: bool = True) -> NativeAttributeTable:
        normalized = _normalize_row_selection(row_positions)
        if self.device_table is not None and not preserve_index:
            import cupy as cp

            if self.row_positions is None:
                selected_positions = cp.asarray(normalized)
            else:
                selected_positions = cp.asarray(self.row_positions)[
                    cp.asarray(normalized)
                ]
            return type(self)(
                device_table=self.device_table,
                index_override=pd.RangeIndex(_row_aligned_size(selected_positions)),
                column_override=self.column_override,
                schema_override=self.schema_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
                row_positions=selected_positions,
            )
        if self.parts is not None:
            taken_parts = tuple(
                part.take(normalized, preserve_index=preserve_index) for part in self.parts
            )
            index_override = (
                taken_parts[0].index
                if preserve_index
                else pd.RangeIndex(_row_aligned_size(normalized))
            )
            return type(self)(
                parts=taken_parts,
                index_override=index_override,
                column_override=self.column_override,
                to_pandas_kwargs=self.to_pandas_kwargs,
            )
        if hasattr(normalized, "__cuda_array_interface__"):
            device_taken = self._device_take(normalized, preserve_index=preserve_index)
            if device_taken is not None:
                return device_taken
            if not preserve_index and self.dataframe is not None:
                promoted = self.promote_numeric_to_device()
                if promoted is not None and promoted is not self:
                    device_taken = promoted._device_take(
                        normalized,
                        preserve_index=False,
                    )
                    if device_taken is not None:
                        return device_taken
            if not preserve_index and (self.loader is not None or self.dataframe is not None):
                parent = self
                device_positions = normalized
                row_count = _row_aligned_size(device_positions)
                index_override = pd.RangeIndex(row_count)

                def _load() -> pd.DataFrame:
                    host_positions = _host_row_positions(
                        device_positions,
                        strict_disallowed=False,
                    )
                    return _take_pandas_frame_rows(
                        parent.to_pandas(copy=False),
                        host_positions,
                        index=pd.RangeIndex(int(host_positions.size)),
                    )

                return type(self).from_loader(
                    _load,
                    index_override=index_override,
                    columns=tuple(self.columns),
                    to_pandas_kwargs=self.to_pandas_kwargs,
                )
        host_positions = _host_row_positions(normalized)
        index_override = (
            self.index.take(host_positions)
            if preserve_index
            else pd.RangeIndex(int(host_positions.size))
        )
        if self.device_table is not None:
            try:
                import cupy as cp
            except ModuleNotFoundError:
                pass
            else:
                device_taken = self._device_take(
                    cp.asarray(host_positions, dtype=cp.int64),
                    preserve_index=False,
                )
                if device_taken is not None:
                    return device_taken.with_index(index_override)
        if self.loader is not None:
            parent = self

            def _load() -> pd.DataFrame:
                return _take_pandas_frame_rows(
                    parent.to_pandas(copy=False),
                    host_positions,
                    index=index_override,
                )

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
        frame = _take_pandas_frame_rows(
            self.to_pandas(copy=False),
            host_positions,
            index=index_override,
        )
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

            from vibespatial.cuda._runtime import pylibcudf_current_stream
        except ModuleNotFoundError:
            return None
        if self.dataframe is not None and self.dataframe.shape[1] == 0:
            return type(self)(
                dataframe=pd.DataFrame(index=pd.RangeIndex(_row_aligned_size(row_positions)))
            )
        if self.device_table is not None:
            source = self.device_table
            schema = self.schema_override
        elif self.arrow_table is not None:
            source = _pylibcudf_table_from_arrow(self.to_arrow(index=False))
            schema = self.arrow_table.schema
        else:
            return None
        d_positions = cp.asarray(row_positions)
        if self.row_positions is not None:
            d_positions = cp.asarray(self.row_positions)[d_positions]
        source_row_count = _device_table_row_count(source)
        target_dtype = cp.int32 if source_row_count <= np.iinfo(np.int32).max else cp.int64
        gather_map = _pylibcudf_column_from_device(
            d_positions.astype(target_dtype, copy=False)
        )
        stream = pylibcudf_current_stream(source)
        gathered = plc.copying.gather(
            source,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
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

        logical_columns = tuple(column for table in tables for column in tuple(table.columns))
        if len(set(logical_columns)) != len(logical_columns):
            return None

        non_empty = [table for table in tables if len(table.columns) > 0]
        if not non_empty:
            return cls(dataframe=pd.DataFrame(index=index_override))

        common_kwargs = non_empty[0].to_pandas_kwargs
        if any(table.to_pandas_kwargs != common_kwargs for table in non_empty[1:]):
            common_kwargs = {}

        if all(table.device_table is not None for table in non_empty):
            if any(table.row_positions is not None for table in non_empty):
                return cls(
                    parts=tuple(non_empty),
                    index_override=index_override,
                    column_override=logical_columns,
                    to_pandas_kwargs=common_kwargs,
                )
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

        if any(table.parts is not None for table in non_empty):
            flattened = []
            for table in non_empty:
                if table.parts is not None:
                    flattened.extend(table.parts)
                else:
                    flattened.append(table)
            return cls.combine_columns(flattened, index_override=index_override)

        if all(table.device_table is None for table in non_empty):
            if any(table.loader is not None for table in non_empty):
                parent_tables = tuple(non_empty)
                output_index = index_override
                output_columns = logical_columns
                output_kwargs = common_kwargs

                def _load() -> pd.DataFrame:
                    frames = [table.to_pandas(copy=False) for table in parent_tables]
                    concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
                    combined = pd.concat(frames, axis=1, **concat_kwargs)
                    if not combined.index.equals(output_index):
                        combined = combined.copy(deep=False)
                        combined.index = output_index
                    return combined

                return cls.from_loader(
                    _load,
                    index_override=output_index,
                    columns=output_columns,
                    to_pandas_kwargs=output_kwargs,
                )

            frames = [table.to_pandas(copy=False) for table in non_empty]
            concat_kwargs = {} if PANDAS_GE_30 else {"copy": False}
            combined = pd.concat(frames, axis=1, **concat_kwargs)
            if not combined.index.equals(index_override):
                combined = combined.copy(deep=False)
                combined.index = index_override
            return cls(dataframe=combined)

        return cls(
            parts=tuple(non_empty),
            index_override=index_override,
            column_override=logical_columns,
            to_pandas_kwargs=common_kwargs,
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
                    from vibespatial.cuda._runtime import pylibcudf_current_stream

                    sources = [table.device_table for table in tables]
                    concatenated = plc.concatenate.concatenate(
                        sources,
                        stream=pylibcudf_current_stream(*sources),
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
            if (pa.types.is_string(left) and pa.types.is_large_string(right)) or (
                pa.types.is_large_string(left) and pa.types.is_string(right)
            ):
                return pa.large_string()
            if (pa.types.is_binary(left) and pa.types.is_large_binary(right)) or (
                pa.types.is_large_binary(left) and pa.types.is_binary(right)
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

    def _without_index_pandas_metadata(arrow_table):
        table_metadata = arrow_table.schema.metadata
        if table_metadata is None or b"pandas" not in table_metadata:
            return arrow_table
        pandas_metadata = json.loads(table_metadata[b"pandas"].decode("utf-8"))
        remaining_fields = set(arrow_table.column_names)
        pandas_metadata["index_columns"] = []
        pandas_metadata["columns"] = [
            column
            for column in pandas_metadata.get("columns", [])
            if column.get("field_name") in remaining_fields
        ]
        projected_metadata = dict(table_metadata)
        projected_metadata[b"pandas"] = json.dumps(pandas_metadata).encode("utf-8")
        return arrow_table.replace_schema_metadata(projected_metadata)

    if pandas_metadata_raw is None:
        return pd.RangeIndex(table.num_rows), table

    pandas_metadata = json.loads(pandas_metadata_raw.decode("utf-8"))
    index_columns = pandas_metadata.get("index_columns") or []
    if not index_columns:
        return pd.RangeIndex(table.num_rows), _without_index_pandas_metadata(table)

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
                _without_index_pandas_metadata(table),
            )

    field_name_map = _pandas_metadata_field_name_map(pandas_metadata)
    index_field_names = [str(name) for name in index_columns]
    missing_index_fields = [name for name in index_field_names if name not in table.column_names]
    if missing_index_fields:
        return pd.RangeIndex(table.num_rows), _without_index_pandas_metadata(table)
    index_table = table.select(index_field_names).replace_schema_metadata(None)
    index_frame = index_table.to_pandas(**(to_pandas_kwargs or {}))
    index_names = [field_name_map.get(name, name) for name in index_field_names]
    if len(index_field_names) == 1:
        index = pd.Index(index_frame.iloc[:, 0].array, name=index_names[0])
    else:
        index_frame.columns = index_names
        index = pd.MultiIndex.from_frame(index_frame)
    return index, _without_index_pandas_metadata(table.drop(index_field_names))


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


def _copy_public_frame_attrs(frame) -> dict[Any, Any]:
    attrs = getattr(frame, "attrs", None)
    return attrs.copy() if isinstance(attrs, dict) else {}


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
    rebuilt.attrs = _copy_public_frame_attrs(frame)
    return rebuilt


@dataclass(frozen=True)
class GeometryNativeResult:
    """Geometry result that stays native until explicitly materialized."""

    crs: Any
    owned: Any | None = None
    series: GeoSeries | None = None
    composition: NativeGeometryComposition | None = None

    def __post_init__(self) -> None:
        storage_count = sum(
            value is not None for value in (self.owned, self.series, self.composition)
        )
        if storage_count != 1:
            raise ValueError(
                "GeometryNativeResult requires exactly one of owned, series, or composition"
            )
        if self.composition is not None and self.composition.crs != self.crs:
            raise ValueError("geometry composition CRS must match GeometryNativeResult CRS")

    @classmethod
    def from_owned(cls, owned, *, crs) -> GeometryNativeResult:
        return cls(crs=crs, owned=owned)

    @classmethod
    def from_composition(
        cls,
        composition: NativeGeometryComposition,
        *,
        crs,
    ) -> GeometryNativeResult:
        return cls(crs=crs, composition=composition.with_crs(crs))

    def with_crs(self, crs) -> GeometryNativeResult:
        if self.crs == crs:
            return self
        composition = None if self.composition is None else self.composition.with_crs(crs)
        return type(self)(
            crs=crs,
            owned=self.owned,
            series=self.series,
            composition=composition,
        )

    def cached_owned(self):
        """Return existing owned storage without certifying or physicalizing."""
        if self.owned is not None:
            return self.owned
        if self.composition is not None:
            return self.composition._singular_owned_cache
        return None

    def apply_rowset_proofs(self, rowset) -> None:
        """Apply semantic row-selection proofs to existing owned storage."""
        owned = self.cached_owned()
        if owned is None:
            return
        if getattr(owned, "_is_lazy_grouped_union_owned", False):
            return
        state = getattr(owned, "device_state", None)
        if state is None and getattr(owned, "residency", None) is Residency.DEVICE:
            state = owned._ensure_device_state(preserve_indexed_view=True)
        if state is None:
            return
        if rowset.trusted_all_valid_rows is True:
            state.trusted_all_valid = True
        family_domain = rowset.geometry_family_domain
        if not family_domain:
            return
        state.trusted_family_domain = tuple(family_domain)
        if rowset.trusted_all_valid_rows is True and len(family_domain) == 1:
            state.trusted_homogeneous_family = family_domain[0]
        from vibespatial.geometry.buffers import GeometryFamily

        if set(family_domain) <= {
            GeometryFamily.POLYGON,
            GeometryFamily.MULTIPOLYGON,
        }:
            state.trusted_polygonal_only = True

    @classmethod
    def from_geoseries(cls, series: GeoSeries) -> GeometryNativeResult:
        values = series.values
        cached_owned = getattr(values, "cached_owned", None)
        owned = cached_owned() if callable(cached_owned) else getattr(values, "_owned", None)
        if owned is not None:
            return cls.from_owned(owned, crs=series.crs)
        composition = getattr(values, "native_composition", None)
        if composition is not None:
            return cls.from_composition(composition, crs=series.crs)
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

        cached_owned = getattr(values, "cached_owned", None)
        owned = cached_owned() if callable(cached_owned) else getattr(values, "_owned", None)
        if callable(cached_owned) and owned is not None:
            return cls.from_owned(owned, crs=crs)
        composition = getattr(values, "native_composition", None)
        if composition is not None:
            return cls.from_composition(composition, crs=crs)

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
        if self.composition is not None:
            return self.composition.to_geoseries(index=index, name=name)
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

    def valid_nonempty_mask_device(self):
        """Return a logical-row device mask for concrete nonempty geometry."""
        if self.owned is not None:
            from vibespatial.geometry.owned import device_valid_nonempty_mask

            return device_valid_nonempty_mask(self.owned)
        if self.composition is not None:
            import cupy as cp

            result = cp.zeros(self.row_count, dtype=cp.bool_)
            if self.composition.contiguous_row_partitions:
                for part in self.composition.parts:
                    part_mask = part.geometry.valid_nonempty_mask_device()
                    if part_mask is None:
                        return None
                    result[cp.asarray(part.output_rows, dtype=cp.int64)] = cp.asarray(
                        part_mask,
                        dtype=cp.bool_,
                    )
                return result
            scatter_result = cp.zeros(self.row_count, dtype=cp.uint32)
            for part in self.composition.parts:
                part_mask = part.geometry.valid_nonempty_mask_device()
                if part_mask is None:
                    return None
                output_rows = cp.asarray(part.output_rows, dtype=cp.int64)
                d_active = cp.asarray(part_mask, dtype=cp.bool_)
                cp.maximum.at(
                    scatter_result,
                    output_rows,
                    d_active.astype(cp.uint32),
                )
            return scatter_result != 0
        return None

    def select_family_domain_device(self, families):
        """Mask non-target concrete parts and return logical keep/drop metadata."""
        import cupy as cp

        from vibespatial.geometry.owned import FAMILY_TAGS

        target_tags = tuple(FAMILY_TAGS[family] for family in families)

        def _owned_selection(owned):
            state = owned._ensure_device_state(preserve_indexed_view=True)
            d_valid_nonempty = type(self).from_owned(
                owned,
                crs=self.crs,
            ).valid_nonempty_mask_device()
            if d_valid_nonempty is None:
                return None
            d_target = cp.zeros(owned.row_count, dtype=cp.bool_)
            d_tags = cp.asarray(state.tags, dtype=cp.int8)
            for tag in target_tags:
                d_target |= d_tags == cp.int8(tag)
            d_keep = cp.asarray(d_valid_nonempty, dtype=cp.bool_) & d_target
            d_drop = cp.asarray(d_valid_nonempty, dtype=cp.bool_) & ~d_target
            return d_keep, d_drop

        if self.owned is not None:
            selection = _owned_selection(self.owned)
            if selection is None:
                return None
            d_keep, d_drop = selection
            return self.mask_capacity(d_keep), d_keep, cp.sum(d_drop, dtype=cp.int64)

        if self.composition is None:
            return None

        d_logical_keep_bits = cp.zeros(self.row_count, dtype=cp.uint32)
        d_drop_count = cp.zeros(1, dtype=cp.int64)
        selected_parts = []
        for part in self.composition.parts:
            owned = part.geometry.owned
            if owned is None:
                return None
            selection = _owned_selection(owned)
            if selection is None:
                return None
            d_part_keep, d_part_drop = selection
            d_rows = cp.asarray(part.output_rows, dtype=cp.int64)
            cp.maximum.at(
                d_logical_keep_bits,
                d_rows,
                d_part_keep.astype(cp.uint32),
            )
            d_drop_count += cp.sum(d_part_drop, dtype=cp.int64).reshape(1)
            selected_parts.append(
                NativeGeometryCompositionPart(
                    geometry=part.geometry.mask_capacity(d_part_keep),
                    output_rows=part.output_rows,
                    collection_position=part.collection_position,
                )
            )

        selected = type(self).from_composition(
            NativeGeometryComposition(
                parts=tuple(selected_parts),
                row_count=self.row_count,
                crs=self.crs,
                trusted_all_ogc_valid=self.composition.trusted_all_ogc_valid,
                contiguous_row_partitions=False,
                trusted_singular_rows=self.composition.trusted_singular_rows,
                trusted_family_domain=tuple(families),
            ),
            crs=self.crs,
        )
        d_logical_keep = d_logical_keep_bits != 0
        return selected, d_logical_keep, d_drop_count[0]

    def mask_capacity(
        self,
        active_mask,
        *,
        preserve_row_bounds: bool = True,
    ) -> GeometryNativeResult:
        """Mask logical capacity lanes without compacting native geometry."""
        import cupy as cp

        d_active = cp.asarray(active_mask, dtype=cp.bool_)
        if int(d_active.size) != self.row_count:
            raise ValueError("geometry capacity mask must match logical row count")
        if self.owned is not None:
            from vibespatial.geometry.owned import device_mask_owned_capacity

            return type(self).from_owned(
                device_mask_owned_capacity(
                    self.owned,
                    d_active,
                    preserve_row_bounds=preserve_row_bounds,
                ),
                crs=self.crs,
            )
        if self.composition is not None:
            parts = tuple(
                NativeGeometryCompositionPart(
                    geometry=part.geometry.mask_capacity(
                        d_active[cp.asarray(part.output_rows, dtype=cp.int64)],
                        preserve_row_bounds=preserve_row_bounds,
                    ),
                    output_rows=part.output_rows,
                    collection_position=part.collection_position,
                )
                for part in self.composition.parts
            )
            return type(self).from_composition(
                NativeGeometryComposition(
                    parts=parts,
                    row_count=self.row_count,
                    crs=self.crs,
                    trusted_all_ogc_valid=self.composition.trusted_all_ogc_valid,
                    contiguous_row_partitions=(
                        self.composition.contiguous_row_partitions
                    ),
                    trusted_singular_rows=self.composition.trusted_singular_rows,
                    trusted_family_domain=self.composition.trusted_family_domain,
                ),
                crs=self.crs,
            )
        raise TypeError("host geometry series cannot be masked as device capacity")

    def permute_capacity(self, row_order) -> GeometryNativeResult:
        """Permute a full logical row capacity without compacting composition parts."""
        import cupy as cp

        d_order = cp.asarray(row_order, dtype=cp.int64)
        if int(d_order.size) != self.row_count:
            raise ValueError("geometry capacity permutation must include every row")
        if self.owned is not None:
            return type(self).from_owned(
                self.owned._device_indexed_take(
                    d_order,
                    assume_unique_indices=True,
                ),
                crs=self.crs,
            )
        if self.composition is not None:
            d_old_to_new = cp.empty(self.row_count, dtype=cp.int64)
            d_old_to_new[d_order] = cp.arange(self.row_count, dtype=cp.int64)
            return type(self).from_composition(
                NativeGeometryComposition(
                    parts=tuple(
                        NativeGeometryCompositionPart(
                            geometry=part.geometry,
                            output_rows=d_old_to_new[cp.asarray(part.output_rows, dtype=cp.int64)],
                            collection_position=part.collection_position,
                        )
                        for part in self.composition.parts
                    ),
                    row_count=self.row_count,
                    crs=self.crs,
                    trusted_all_ogc_valid=self.composition.trusted_all_ogc_valid,
                    contiguous_row_partitions=False,
                    trusted_singular_rows=self.composition.trusted_singular_rows,
                    trusted_family_domain=self.composition.trusted_family_domain,
                ),
                crs=self.crs,
            )
        raise TypeError("host geometry series cannot use a device capacity permutation")

    def take(
        self,
        row_positions,
        *,
        unique: bool = False,
        defer_device_metadata: bool = False,
    ) -> GeometryNativeResult:
        normalized = (
            row_positions
            if (
                defer_device_metadata
                and _is_device_array(row_positions)
                and getattr(getattr(row_positions, "dtype", None), "kind", None)
                in {"i", "u"}
            )
            else _normalize_row_selection(row_positions)
        )
        if self.composition is not None:
            return type(self).from_composition(
                self.composition.take(normalized, unique=unique),
                crs=self.crs,
            )
        if self.owned is not None:
            if _is_device_array(normalized) and defer_device_metadata:
                taken = self.owned._device_indexed_take(
                    normalized,
                    assume_unique_indices=unique,
                    defer_device_metadata=True,
                )
            else:
                taken = (
                    self.owned.device_take(
                        normalized,
                        assume_unique_indices=unique,
                    )
                    if _is_device_array(normalized)
                    else self.owned.take(normalized)
                )
            return type(self).from_owned(taken, crs=self.crs)
        host_positions = _host_row_positions(normalized)
        return type(self).from_geoseries(self.series.take(host_positions))

    @property
    def row_count(self) -> int:
        if self.composition is not None:
            return int(self.composition.row_count)
        if self.owned is not None:
            return int(self.owned.row_count)
        return int(len(self.series))

    @property
    def residency(self) -> Residency:
        if self.composition is not None:
            return self.composition.residency
        return combined_residency(self.owned)


@dataclass(frozen=True)
class NativeGeometryCompositionPart:
    """Concrete geometry rows mapped into logical composition output rows."""

    geometry: GeometryNativeResult
    output_rows: Any
    collection_position: int | None = None

    def __post_init__(self) -> None:
        if self.geometry.composition is not None:
            raise ValueError("geometry composition parts must use concrete storage")
        if _row_aligned_size(self.output_rows) != self.geometry.row_count:
            raise ValueError("geometry composition part rows must match concrete geometry rows")
        if self.collection_position is not None and self.collection_position < 0:
            raise ValueError("geometry collection positions must be non-negative")

    @property
    def residency(self) -> Residency:
        return combined_residency(self.geometry, self.output_rows)

    def with_crs(self, crs) -> NativeGeometryCompositionPart:
        if self.geometry.crs == crs:
            return self
        return type(self)(
            self.geometry.with_crs(crs),
            self.output_rows,
            self.collection_position,
        )


def _composition_part_take_relation(part_rows: Any, selected_rows: Any):
    """Join concrete part rows to selected logical rows as a native relation."""
    use_device = _is_device_array(part_rows) or _is_device_array(selected_rows)
    if use_device:
        import cupy as cp
        import pylibcudf as plc

        from vibespatial.api._native_relation import NativeRelation
        from vibespatial.cuda._runtime import pylibcudf_current_stream

        rows = cp.asarray(part_rows, dtype=cp.int64)
        selected = cp.asarray(selected_rows, dtype=cp.int64)
        if int(rows.size) == 0 or int(selected.size) == 0:
            empty = cp.empty(0, dtype=cp.int64)
            return NativeRelation(
                left_indices=empty,
                right_indices=empty,
                left_row_count=int(rows.size),
                right_row_count=int(selected.size),
            )

        concrete_column, selected_column = plc.join.inner_join(
            plc.Table([_pylibcudf_column_from_device(rows)]),
            plc.Table([_pylibcudf_column_from_device(selected)]),
            plc.types.NullEquality.EQUAL,
            stream=pylibcudf_current_stream(),
        )

        def _join_indices(column):
            return (
                cp.asarray(column.data())
                .view(cp.int32)[: int(column.size())]
                .astype(
                    cp.int64,
                    copy=False,
                )
            )

        concrete_positions = _join_indices(concrete_column)
        output_rows = _join_indices(selected_column)
        if int(concrete_positions.size) > 1:
            from vibespatial.cuda.cccl_primitives import PairSortStrategy, sort_pairs

            relation_keys = (
                output_rows.astype(cp.uint64, copy=False) << cp.uint64(32)
            ) | concrete_positions.astype(cp.uint64, copy=False)
            order = sort_pairs(
                relation_keys,
                cp.arange(int(concrete_positions.size), dtype=cp.int32),
                strategy=PairSortStrategy.RADIX,
                synchronize=False,
            ).values
            concrete_positions = concrete_positions[order]
            output_rows = output_rows[order]
        return NativeRelation(
            left_indices=concrete_positions,
            right_indices=output_rows,
            left_row_count=int(rows.size),
            right_row_count=int(selected.size),
            duplicate_policy="preserve",
        )
    else:
        xp = np

    rows = xp.asarray(part_rows, dtype=xp.int64)
    selected = xp.asarray(selected_rows, dtype=xp.int64)
    if int(rows.size) == 0 or int(selected.size) == 0:
        empty = xp.empty(0, dtype=xp.int64)
        from vibespatial.api._native_relation import NativeRelation

        return NativeRelation(
            left_indices=empty,
            right_indices=empty,
            left_row_count=int(rows.size),
            right_row_count=int(selected.size),
        )

    order = xp.argsort(rows).astype(xp.int64, copy=False)
    sorted_rows = rows[order]
    starts = xp.searchsorted(sorted_rows, selected, side="left").astype(
        xp.int64,
        copy=False,
    )
    stops = xp.searchsorted(sorted_rows, selected, side="right").astype(
        xp.int64,
        copy=False,
    )
    counts = stops - starts
    remapped_output_rows = xp.repeat(
        xp.arange(int(selected.size), dtype=xp.int64),
        counts,
    )
    repeated_starts = xp.repeat(starts, counts)
    group_starts = xp.cumsum(counts, dtype=xp.int64) - counts
    local_positions = xp.arange(
        int(remapped_output_rows.size),
        dtype=xp.int64,
    ) - xp.repeat(group_starts, counts)
    concrete_positions = order[repeated_starts + local_positions]
    from vibespatial.api._native_relation import NativeRelation

    return NativeRelation(
        left_indices=concrete_positions,
        right_indices=remapped_output_rows,
        left_row_count=int(rows.size),
        right_row_count=int(selected.size),
        duplicate_policy="preserve",
    )


def _composition_part_take_unique_capacity(
    part: NativeGeometryCompositionPart,
    selected_rows: Any,
) -> NativeGeometryCompositionPart:
    """Map concrete rows through a unique selection at fixed part capacity."""
    if _is_device_array(part.output_rows) or _is_device_array(selected_rows):
        import cupy as xp
    else:
        xp = np

    rows = xp.asarray(part.output_rows, dtype=xp.int64)
    selected = xp.asarray(selected_rows, dtype=xp.int64)
    if int(selected.size) == 0:
        active = xp.zeros(int(rows.size), dtype=xp.bool_)
        output_rows = xp.zeros(int(rows.size), dtype=xp.int64)
    else:
        order = xp.argsort(selected).astype(xp.int64, copy=False)
        sorted_selected = selected[order]
        locations = xp.searchsorted(sorted_selected, rows, side="left").astype(
            xp.int64,
            copy=False,
        )
        safe_locations = xp.minimum(locations, int(selected.size) - 1)
        active = (locations < int(selected.size)) & (sorted_selected[safe_locations] == rows)
        output_rows = xp.where(
            active,
            order[safe_locations],
            xp.int64(0),
        )
    geometry = (
        part.geometry.mask_capacity(active)
        if _is_device_array(active)
        else part.geometry.take(np.flatnonzero(active))
    )
    if not _is_device_array(active):
        output_rows = output_rows[active]
    return NativeGeometryCompositionPart(
        geometry=geometry,
        output_rows=output_rows,
        collection_position=part.collection_position,
    )


@dataclass(frozen=True)
class NativeGeometryComposition:
    """Logical geometry rows composed from concrete native geometry parts.

    Each part carries concrete geometry storage plus row indirection into the
    public output. Zero parts represent a missing row, one part remains a
    concrete geometry, and multiple heterogeneous parts become a
    GeometryCollection only at an explicit compatibility export boundary.
    """

    parts: tuple[NativeGeometryCompositionPart, ...]
    row_count: int
    crs: Any
    trusted_all_ogc_valid: bool | None = None
    contiguous_row_partitions: bool = False
    trusted_singular_rows: bool = False
    trusted_family_domain: tuple[Any, ...] | None = None
    _singular_owned_cache: Any = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _singular_partitioned_cache: Any = dataclass_field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if int(self.row_count) < 0:
            raise ValueError("geometry composition row_count must be non-negative")
        concrete_parts = []
        null_parts = []
        for part in self.parts:
            if part.geometry.crs != self.crs:
                raise ValueError("geometry composition part CRS must match composition CRS")
            if not _is_device_array(part.output_rows):
                rows = np.asarray(part.output_rows, dtype=np.int64)
                if rows.size > 0 and (
                    np.any(rows < 0) or np.any(rows >= int(self.row_count))
                ):
                    raise ValueError("geometry composition output rows are out of bounds")
            owned = part.geometry.owned
            if owned is not None:
                if owned.residency is Residency.DEVICE:
                    state = owned.device_state
                    family_empty = state is not None and not state.families
                else:
                    family_empty = not owned.families
                if family_empty and part.collection_position is None:
                    null_parts.append(part)
                    continue
            concrete_parts.append(part)
        normalized_parts = tuple(concrete_parts or null_parts[:1])
        if len(normalized_parts) != len(self.parts) or any(
            normalized is not original
            for normalized, original in zip(
                normalized_parts,
                self.parts,
                strict=False,
            )
        ):
            object.__setattr__(self, "parts", normalized_parts)
        if self.trusted_family_domain is None:
            domains = []
            for part in normalized_parts:
                owned = part.geometry.owned
                if owned is None:
                    domains.append(None)
                elif owned.residency is Residency.DEVICE:
                    state = owned.device_state
                    domains.append(
                        None
                        if state is None
                        else getattr(state, "trusted_family_domain", None)
                        or tuple(state.families)
                    )
                else:
                    domains.append(tuple(owned.families))
            if all(domain is not None for domain in domains):
                object.__setattr__(
                    self,
                    "trusted_family_domain",
                    tuple(
                        dict.fromkeys(
                            family
                            for domain in domains
                            for family in domain
                        )
                    ),
                )

    @property
    def residency(self) -> Residency:
        return combined_residency(*self.parts)

    def with_crs(self, crs) -> NativeGeometryComposition:
        if self.crs == crs:
            return self
        return type(self)(
            parts=tuple(part.with_crs(crs) for part in self.parts),
            row_count=self.row_count,
            crs=crs,
            trusted_all_ogc_valid=self.trusted_all_ogc_valid,
            contiguous_row_partitions=self.contiguous_row_partitions,
            trusted_singular_rows=self.trusted_singular_rows,
            trusted_family_domain=self.trusted_family_domain,
        )

    def ordered_contiguous_device_parts(self):
        """Return exact ordered output spans without joining part row arrays.

        This is the terminal-export view of a singular composition.  Each
        returned owned array covers one contiguous logical output interval and
        the intervals cover the full result in order.  Certification reduces
        only compact part-layout metadata to host; geometry rows and
        coordinates remain device-resident.
        """
        if (
            not self.trusted_singular_rows
            or self.residency is not Residency.DEVICE
            or any(part.collection_position is not None for part in self.parts)
        ):
            return None

        owned_parts = []
        total_rows = 0
        for part in self.parts:
            owned = part.geometry.owned
            if owned is None:
                return None
            part_rows = int(owned.row_count)
            if part_rows != _row_aligned_size(part.output_rows):
                return None
            if part_rows:
                owned_parts.append((total_rows, total_rows + part_rows, owned, part.output_rows))
            total_rows += part_rows
        if total_rows != int(self.row_count):
            return None

        if not self.contiguous_row_partitions and owned_parts:
            import cupy as cp

            checks = []
            for start, stop, _owned, output_rows in owned_parts:
                rows = cp.asarray(output_rows, dtype=cp.int64)
                internally_contiguous = (
                    cp.asarray(True)
                    if int(rows.size) == 1
                    else cp.all(rows[1:] == rows[:-1] + cp.int64(1))
                )
                checks.append(
                    internally_contiguous
                    & (rows[0] == cp.int64(start))
                    & (rows[-1] == cp.int64(stop - 1))
                )
            certified = _host_array(
                cp.stack(checks),
                dtype=np.bool_,
                strict_disallowed=False,
                surface="vibespatial.api.NativeGeometryComposition",
                operation="ordered_contiguous_partition_certification",
                reason=(
                    "terminal native export certified compact geometry-part "
                    "layout metadata"
                ),
                detail=f"rows={self.row_count}, parts={len(owned_parts)}",
            )
            if not bool(np.all(certified)):
                return None
            object.__setattr__(self, "contiguous_row_partitions", True)

        return tuple(
            (start, stop, owned)
            for start, stop, owned, _output_rows in owned_parts
        )

    def take(self, row_positions, *, unique: bool = False) -> NativeGeometryComposition:
        singular_partitioned = self._singular_partitioned_cache
        if singular_partitioned is not None:
            return singular_partitioned.take(row_positions, unique=unique)
        selected = _normalize_row_selection(row_positions)
        if _row_aligned_size(selected) == 0:
            return type(self)(
                parts=(),
                row_count=0,
                crs=self.crs,
                trusted_all_ogc_valid=self.trusted_all_ogc_valid,
                trusted_singular_rows=self.trusted_singular_rows,
                trusted_family_domain=self.trusted_family_domain,
            )
        taken_parts = []
        for part in self.parts:
            if unique:
                taken_parts.append(_composition_part_take_unique_capacity(part, selected))
                continue
            take_relation = _composition_part_take_relation(
                part.output_rows,
                selected,
            )
            concrete_positions = take_relation.left_indices
            output_rows = take_relation.right_indices
            if int(concrete_positions.size) == 0:
                continue
            if _is_device_array(concrete_positions) and part.geometry.owned is not None:
                taken_geometry = GeometryNativeResult.from_owned(
                    part.geometry.owned._device_indexed_take(concrete_positions),
                    crs=part.geometry.crs,
                )
            else:
                taken_geometry = part.geometry.take(concrete_positions)
            taken_parts.append(
                NativeGeometryCompositionPart(
                    geometry=taken_geometry,
                    output_rows=output_rows,
                    collection_position=part.collection_position,
                )
            )
        result = type(self)(
            parts=tuple(taken_parts),
            row_count=_row_aligned_size(selected),
            crs=self.crs,
            trusted_all_ogc_valid=self.trusted_all_ogc_valid,
            trusted_singular_rows=self.trusted_singular_rows,
            trusted_family_domain=self.trusted_family_domain,
        )
        cached = self._singular_owned_cache
        if cached is not None:
            if _is_device_array(selected):
                taken_cache = cached._device_indexed_take(
                    selected,
                    assume_unique_indices=unique,
                )
            else:
                import cupy as cp

                taken_cache = cached._device_indexed_take(
                    cp.asarray(selected, dtype=cp.int64),
                    assume_unique_indices=unique,
                )
            object.__setattr__(result, "_singular_owned_cache", taken_cache)
        return result

    def _singular_partitioned_device(self):
        """Select one concrete device part per row without joining buffers."""
        cached = self._singular_partitioned_cache
        if cached is not None:
            return cached
        if self.residency is not Residency.DEVICE:
            return None
        if any(part.collection_position is not None for part in self.parts):
            return None
        if self.trusted_singular_rows and self.contiguous_row_partitions:
            return self
        if self.row_count == 0:
            result = type(self)(
                parts=(),
                row_count=0,
                crs=self.crs,
                trusted_all_ogc_valid=self.trusted_all_ogc_valid,
                trusted_singular_rows=True,
                trusted_family_domain=self.trusted_family_domain,
            )
            object.__setattr__(self, "_singular_partitioned_cache", result)
            return result

        import cupy as cp

        concrete = []
        sentinel_row = cp.int64(self.row_count)
        d_counts_with_sentinel = cp.zeros(self.row_count + 1, dtype=cp.int32)
        for part in self.parts:
            owned = part.geometry.owned
            if owned is None:
                return None
            state = owned._ensure_device_state(preserve_indexed_view=True)
            d_valid = cp.asarray(state.validity, dtype=cp.bool_)
            d_nonempty = part.geometry.valid_nonempty_mask_device()
            if d_nonempty is None:
                return None
            d_nonempty = cp.asarray(d_nonempty, dtype=cp.bool_) & d_valid
            d_rows = cp.asarray(part.output_rows, dtype=cp.int64)
            if int(d_valid.size) != int(d_rows.size):
                raise ValueError("composition concrete validity must align with output rows")
            cp.add.at(
                d_counts_with_sentinel,
                cp.where(d_nonempty, d_rows, sentinel_row),
                d_nonempty.astype(cp.int32),
            )
            concrete.append((owned, d_rows, d_valid, d_nonempty))
        d_counts = d_counts_with_sentinel[: self.row_count]

        if not self.trusted_singular_rows:
            singular = _host_array(
                cp.stack((cp.all(d_counts <= 1),)),
                dtype=np.bool_,
                strict_disallowed=False,
                surface="vibespatial.api.NativeGeometryComposition.to_geoseries",
                operation="singular_partitioned_certification",
                reason=(
                    "terminal geometry composition multiplicity certified before "
                    "partitioned public export"
                ),
                detail=f"rows={self.row_count}, parts={len(self.parts)}",
            )
            if not bool(singular[0]):
                return None

        selected_codes_with_sentinel = cp.zeros(
            self.row_count + 1,
            dtype=cp.uint64,
        )
        part_codes = []
        high_bit = cp.uint64(1) << cp.uint64(63)
        for part_index, (owned, d_rows, d_valid, d_nonempty) in enumerate(concrete):
            part_count = int(owned.row_count)
            d_lanes = cp.arange(part_count, dtype=cp.uint64)
            d_codes = (cp.uint64(part_index + 1) << cp.uint64(32)) | (
                d_lanes + cp.uint64(1)
            )
            d_codes = cp.where(d_nonempty, d_codes | high_bit, d_codes)
            cp.maximum.at(
                selected_codes_with_sentinel,
                cp.where(d_valid, d_rows, sentinel_row),
                cp.where(d_valid, d_codes, cp.uint64(0)),
            )
            part_codes.append(d_codes)
        selected_codes = selected_codes_with_sentinel[: self.row_count]

        selected_parts = []
        for (owned, d_rows, d_valid, _d_nonempty), d_codes in zip(
            concrete,
            part_codes,
            strict=True,
        ):
            d_selected = d_valid & (selected_codes[d_rows] == d_codes)
            selected_parts.append(
                NativeGeometryCompositionPart(
                    geometry=GeometryNativeResult.from_owned(owned, crs=self.crs).mask_capacity(
                        d_selected,
                        preserve_row_bounds=False,
                    ),
                    output_rows=d_rows,
                )
            )

        result = type(self)(
            parts=tuple(selected_parts),
            row_count=self.row_count,
            crs=self.crs,
            trusted_all_ogc_valid=self.trusted_all_ogc_valid,
            trusted_singular_rows=True,
            trusted_family_domain=self.trusted_family_domain,
        )
        object.__setattr__(self, "_singular_partitioned_cache", result)
        return result

    def _singular_owned_device(self):
        """Physicalize device parts when every logical row has at most one value."""
        cached = self._singular_owned_cache
        if cached is not None:
            return cached
        if self.residency is not Residency.DEVICE:
            return None
        if any(part.collection_position is not None for part in self.parts):
            return None

        import cupy as cp

        from vibespatial.geometry.owned import (
            OwnedGeometryArray,
            build_null_owned_array,
        )

        if self.row_count == 0:
            return build_null_owned_array(0, residency=Residency.DEVICE)

        if self.contiguous_row_partitions:
            owned_parts = [part.geometry.owned for part in self.parts]
            if any(owned is None for owned in owned_parts):
                return None
            result = OwnedGeometryArray.concat(owned_parts)
            if int(result.row_count) != int(self.row_count):
                raise ValueError("contiguous geometry partitions lost logical rows")
            if self.trusted_all_ogc_valid is True:
                result._ensure_device_state(
                    preserve_indexed_view=True,
                ).trusted_all_ogc_valid = True
            object.__setattr__(self, "_singular_owned_cache", result)
            object.__setattr__(
                self,
                "parts",
                (
                    NativeGeometryCompositionPart(
                        geometry=GeometryNativeResult.from_owned(
                            result,
                            crs=self.crs,
                        ),
                        output_rows=cp.arange(self.row_count, dtype=cp.int64),
                    ),
                ),
            )
            return result

        concrete = []
        d_counts = cp.zeros(self.row_count, dtype=cp.int32)
        d_valid_counts = cp.zeros(self.row_count, dtype=cp.int32)
        for part in self.parts:
            owned = part.geometry.owned
            if owned is None:
                return None
            state = owned._ensure_device_state(preserve_indexed_view=True)
            d_valid = cp.asarray(state.validity, dtype=cp.bool_)
            d_nonempty = part.geometry.valid_nonempty_mask_device()
            if d_nonempty is None:
                return None
            d_nonempty = cp.asarray(d_nonempty, dtype=cp.bool_) & d_valid
            d_rows = cp.asarray(part.output_rows, dtype=cp.int64)
            if int(d_valid.size) != int(d_rows.size):
                raise ValueError("composition concrete validity must align with output rows")
            cp.add.at(
                d_counts,
                d_rows[d_nonempty],
                cp.ones_like(d_rows[d_nonempty], dtype=cp.int32),
            )
            cp.add.at(
                d_valid_counts,
                d_rows[d_valid],
                cp.ones_like(d_rows[d_valid], dtype=cp.int32),
            )
            concrete.append((owned, d_rows, d_valid, d_nonempty))

        admission = _host_array(
            cp.stack((cp.all(d_counts <= 1), cp.all(d_valid_counts > 0))),
            dtype=np.bool_,
            strict_disallowed=False,
            surface="vibespatial.api.NativeGeometryComposition.to_geoseries",
            operation="singular_owned_certification",
            reason=(
                "terminal geometry composition multiplicity certified before "
                "device owned physicalization"
            ),
            detail=f"rows={self.row_count}, parts={len(self.parts)}",
        )
        if not bool(admission[0]):
            return None

        full_valid_coverage = bool(admission[1])
        selected_codes = cp.zeros(self.row_count, dtype=cp.uint64)
        part_codes = []
        high_bit = cp.uint64(1) << cp.uint64(63)
        for part_index, (owned, d_rows, d_valid, d_nonempty) in enumerate(concrete):
            part_count = int(owned.row_count)
            d_lanes = cp.arange(part_count, dtype=cp.uint64)
            d_codes = (cp.uint64(part_index + 1) << cp.uint64(32)) | (d_lanes + cp.uint64(1))
            d_codes = cp.where(d_nonempty, d_codes | high_bit, d_codes)
            cp.maximum.at(
                selected_codes,
                d_rows[d_valid],
                d_codes[d_valid],
            )
            part_codes.append(d_codes)

        selected_masks = [
            d_valid & (selected_codes[d_rows] == d_codes)
            for (_owned, d_rows, d_valid, _d_nonempty), d_codes in zip(
                concrete,
                part_codes,
                strict=True,
            )
        ]
        from vibespatial.geometry.owned import (
            device_physicalize_owned_row_selections_exact,
        )

        physical_parts = device_physicalize_owned_row_selections_exact(
            [
                (owned, d_selected)
                for (owned, _d_rows, _d_valid, _d_nonempty), d_selected in zip(
                    concrete,
                    selected_masks,
                    strict=True,
                )
            ],
            reason="native geometry composition exact physicalization allocation packet",
        )

        partitioned_parts = []
        for (
            (_owned, d_rows, _d_valid, _d_nonempty),
            d_selected,
            physical,
        ) in zip(
            concrete,
            selected_masks,
            physical_parts,
            strict=True,
        ):
            if physical is None:
                continue
            d_lanes = cp.flatnonzero(d_selected).astype(cp.int64, copy=False)
            if int(d_lanes.size) == 0:
                continue
            partitioned_parts.append(
                NativeGeometryCompositionPart(
                    geometry=GeometryNativeResult.from_owned(
                        physical._device_indexed_take(
                            d_lanes,
                            assume_unique_indices=True,
                        ),
                        crs=self.crs,
                    ),
                    output_rows=d_rows[d_lanes],
                )
            )
        singular_partitioned = type(self)(
            parts=tuple(partitioned_parts),
            row_count=self.row_count,
            crs=self.crs,
            trusted_all_ogc_valid=self.trusted_all_ogc_valid,
            trusted_singular_rows=True,
            trusted_family_domain=self.trusted_family_domain,
        )
        object.__setattr__(self, "_singular_partitioned_cache", singular_partitioned)

        if full_valid_coverage:
            arrays = []
            d_index_map = cp.zeros(self.row_count, dtype=cp.int64)
            source_offset = 0
        else:
            arrays = [build_null_owned_array(self.row_count, residency=Residency.DEVICE)]
            d_index_map = cp.arange(self.row_count, dtype=cp.int64)
            source_offset = self.row_count
        for (owned, d_rows, _d_valid, _d_nonempty), d_selected, physical in zip(
            concrete,
            selected_masks,
            physical_parts,
            strict=True,
        ):
            if physical is None:
                continue
            part_count = int(owned.row_count)
            d_lanes = cp.arange(part_count, dtype=cp.int64)
            d_destinations = cp.where(
                d_selected,
                d_rows,
                cp.int64(self.row_count) + d_lanes,
            )
            d_extended = cp.concatenate((d_index_map, cp.zeros(part_count, dtype=cp.int64)))
            d_extended[d_destinations] = cp.where(
                d_selected,
                cp.int64(source_offset) + d_lanes,
                cp.int64(0),
            )
            d_index_map = d_extended[: self.row_count]
            arrays.append(physical)
            source_offset += part_count

        root = OwnedGeometryArray.concat(arrays)
        result = OwnedGeometryArray._indexed_view(
            root,
            d_index_map,
            assume_unique_indices=True,
        )
        if self.trusted_all_ogc_valid is True:
            result._ensure_device_state(
                preserve_indexed_view=True,
            ).trusted_all_ogc_valid = True
            result._cached_is_valid_mask = np.ones(self.row_count, dtype=bool)
        object.__setattr__(self, "_singular_owned_cache", result)
        return result

    @classmethod
    def concat(
        cls,
        geometries: list[GeometryNativeResult],
        *,
        crs,
    ) -> NativeGeometryComposition:
        parts: list[NativeGeometryCompositionPart] = []
        contiguous_row_partitions = True
        row_offset = 0
        for geometry in geometries:
            normalized = geometry.with_crs(crs)
            if normalized.composition is None:
                if normalized.residency is Residency.DEVICE:
                    import cupy as xp
                else:
                    xp = np
                output_rows = xp.arange(
                    normalized.row_count,
                    dtype=xp.int64,
                ) + xp.int64(row_offset)
                parts.append(
                    NativeGeometryCompositionPart(
                        geometry=normalized,
                        output_rows=output_rows,
                    )
                )
            else:
                contiguous_row_partitions &= normalized.composition.contiguous_row_partitions
                for part in normalized.composition.parts:
                    if _is_device_array(part.output_rows):
                        import cupy as xp
                    else:
                        xp = np
                    parts.append(
                        NativeGeometryCompositionPart(
                            geometry=part.geometry,
                            output_rows=xp.asarray(
                                part.output_rows,
                                dtype=xp.int64,
                            )
                            + xp.int64(row_offset),
                            collection_position=part.collection_position,
                        )
                    )
            row_offset += normalized.row_count
        family_domains = []
        for geometry in geometries:
            if geometry.composition is not None:
                domain = geometry.composition.trusted_family_domain
            elif geometry.owned is not None:
                state = geometry.owned._ensure_device_state(
                    preserve_indexed_view=True,
                ) if geometry.residency is Residency.DEVICE else None
                domain = (
                    getattr(state, "trusted_family_domain", None)
                    if state is not None
                    else tuple(geometry.owned.families)
                )
                if domain is None and state is not None:
                    domain = tuple(state.families)
            else:
                domain = None
            family_domains.append(domain)
        trusted_family_domain = (
            tuple(
                dict.fromkeys(
                    family
                    for domain in family_domains
                    for family in domain
                )
            )
            if all(domain is not None for domain in family_domains)
            else None
        )
        return cls(
            parts=tuple(parts),
            row_count=row_offset,
            crs=crs,
            trusted_all_ogc_valid=(
                True
                if all(
                    geometry.composition is not None
                    and geometry.composition.trusted_all_ogc_valid is True
                    for geometry in geometries
                )
                else None
            ),
            contiguous_row_partitions=contiguous_row_partitions,
            trusted_singular_rows=all(
                geometry.composition is None
                or geometry.composition.trusted_singular_rows
                for geometry in geometries
            ),
            trusted_family_domain=trusted_family_domain,
        )

    def to_geoseries(self, *, index, name: str) -> GeoSeries:
        """Materialize concrete parts and assemble public rows at export."""
        singular_partitioned = self._singular_partitioned_device()
        if singular_partitioned is not None:
            from vibespatial.api.geoseries import GeoSeries
            from vibespatial.geometry.device_array import DeviceGeometryArray

            return GeoSeries(
                DeviceGeometryArray._from_composition(
                    singular_partitioned,
                    crs=self.crs,
                ),
                index=index,
                name=name,
                crs=self.crs,
                copy=False,
            )
        singular_owned = self._singular_owned_device()
        if singular_owned is not None:
            from vibespatial.io.geoarrow import geoseries_from_owned

            return geoseries_from_owned(
                singular_owned,
                name=name,
                crs=self.crs,
                index=index,
            )

        import shapely
        from shapely.geometry import GeometryCollection

        from vibespatial.api.geoseries import GeoSeries

        row_parts: list[list[Any]] = [[] for _ in range(int(self.row_count))]
        row_ordered_parts: list[list[tuple[int, Any]]] = [[] for _ in range(int(self.row_count))]
        row_empty_fallbacks: list[Any | None] = [None for _ in range(int(self.row_count))]
        for part in self.parts:
            part_rows = _host_array(
                part.output_rows,
                dtype=np.int64,
                strict_disallowed=False,
                surface="vibespatial.api.NativeGeometryComposition.to_geoseries",
                operation="composition_rows_to_host",
                reason="geometry composition row indirection exported to GeoSeries",
                detail=f"parts={part.geometry.row_count}",
            )
            values = np.asarray(
                part.geometry.to_geoseries(
                    index=pd.RangeIndex(part.geometry.row_count),
                    name=name,
                ),
                dtype=object,
            )
            missing = np.asarray(shapely.is_missing(values))
            empty = np.asarray(shapely.is_empty(values))
            for output_row, value, is_missing, is_empty in zip(
                part_rows,
                values,
                missing,
                empty,
                strict=True,
            ):
                row = int(output_row)
                if bool(is_missing):
                    continue
                if part.collection_position is not None:
                    row_ordered_parts[row].append((int(part.collection_position), value))
                    continue
                if bool(is_empty):
                    if row_empty_fallbacks[row] is None:
                        row_empty_fallbacks[row] = value
                    continue
                row_parts[row].append(value)

        assembled = np.empty(int(self.row_count), dtype=object)
        for row, positioned_parts in enumerate(row_ordered_parts):
            if positioned_parts:
                positioned_parts.sort(key=lambda item: item[0])
                assembled[row] = GeometryCollection(
                    [value for _position, value in positioned_parts]
                )
                continue
            parts = row_parts[row]
            if not parts:
                assembled[row] = row_empty_fallbacks[row]
            elif len(parts) == 1:
                assembled[row] = parts[0]
            else:
                assembled[row] = GeometryCollection(parts)
        return GeoSeries(assembled, index=index, name=name, crs=self.crs)


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
                raise ValueError(f"NativeGeometryProvenance {name} length must match row_count")
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
            dict.fromkeys(token for provenance in provenances for token in provenance.source_tokens)
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
    terminal_geodataframe_materializer: Callable[[Any, Any], Any] | None = dataclass_field(
        default=None, repr=False, compare=False
    )
    terminal_geodataframe_materializer_owns_export: bool = dataclass_field(
        default=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attributes",
            NativeAttributeTable.from_value(self.attributes),
        )
        row_count = len(self.attributes)
        if self.geometry.row_count != row_count:
            raise ValueError(
                "primary geometry row count must match attribute row count "
                f"({self.geometry.row_count} != {row_count})"
            )
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
            column
            for column in attr_columns
            if column not in ordered and column not in geometry_names
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
        lazy_public_index: bool = False,
    ) -> NativeAttributeTable:
        """Return attributes indexed for an explicit public export boundary."""
        attributes = NativeAttributeTable.from_value(self.attributes)
        if not include_index or self.index_plan is None:
            return attributes
        public_index = None
        if lazy_public_index:
            from vibespatial.api._native_public_arrays import native_public_index_from_plan

            public_index = native_public_index_from_plan(self.index_plan)
        if public_index is None:
            public_index = self.index_plan.to_public_index(
                surface=surface,
                strict_disallowed=strict_disallowed,
            )
        if _index_equals_without_lazy_native_export(attributes.index, public_index):
            return attributes
        return attributes.with_index(public_index)

    def _record_geodataframe_export_boundary(
        self,
        attributes: NativeAttributeTable,
    ) -> None:
        record_native_export_boundary(
            NativeExportBoundary(
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
            )
        )

    def to_geodataframe(self, *, lazy_public_index: bool = True) -> GeoDataFrame:
        if (
            self.terminal_geodataframe_materializer is not None
            and self.terminal_geodataframe_materializer_owns_export
        ):
            attributes = NativeAttributeTable.from_value(self.attributes)
            self._record_geodataframe_export_boundary(attributes)
            frame = self.terminal_geodataframe_materializer(self, attributes)
            if self.attrs:
                frame.attrs.update(self.attrs)
            return frame

        attributes = self.attributes_for_export(
            surface="vibespatial.api.NativeTabularResult.to_geodataframe",
            include_index=True,
            strict_disallowed=False,
            lazy_public_index=lazy_public_index,
        )
        self._record_geodataframe_export_boundary(attributes)
        if self.terminal_geodataframe_materializer is not None:
            frame = self.terminal_geodataframe_materializer(self, attributes)
            if self.attrs:
                frame.attrs.update(self.attrs)
            return frame
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
            record_native_export_boundary(
                NativeExportBoundary(
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
                )
            )
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
            record_native_export_boundary(
                NativeExportBoundary(
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
                )
            )
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
            record_native_export_boundary(
                NativeExportBoundary(
                    surface="vibespatial.api.NativeTabularResult.to_feather",
                    operation="native_tabular_to_feather",
                    target="feather",
                    reason="native tabular result exported to Feather writer boundary",
                    detail=(
                        f"attribute_columns={len(self.attributes.columns)}, "
                        f"secondary_geometry={len(self.secondary_geometry)}"
                    ),
                    row_count=len(self.attributes),
                )
            )
        _to_feather(
            self,
            path,
            index=index,
            compression=compression,
            schema_version=schema_version,
            **kwargs,
        )

    def take(
        self,
        row_positions,
        *,
        preserve_index: bool = True,
        unique: bool = False,
    ) -> NativeTabularResult:
        from vibespatial.api._native_rowset import NativeRowSet

        rowset = row_positions if isinstance(row_positions, NativeRowSet) else None
        if rowset is not None:
            if rowset.source_row_count is not None and int(rowset.source_row_count) != len(
                self.attributes
            ):
                raise ValueError("NativeRowSet source row count does not match NativeTabularResult")
            if preserve_index and rowset.identity and len(rowset) == len(self.attributes):
                self.geometry.apply_rowset_proofs(rowset)
                return self
            row_positions = rowset.positions
            unique = unique or rowset.unique
        normalized = _normalize_row_selection(row_positions)
        terminal_materializer = self.terminal_geodataframe_materializer
        if terminal_materializer is not None:
            take_materializer = getattr(terminal_materializer, "take", None)
            terminal_materializer = (
                take_materializer(normalized) if callable(take_materializer) else None
            )
        index_plan = self.index_plan
        if preserve_index and index_plan is None:
            from vibespatial.api._native_rowset import NativeIndexPlan

            index_plan = NativeIndexPlan.from_index(self.attributes.index)
        attributes_preserve_index = preserve_index
        if (
            preserve_index
            and _is_device_array(normalized)
            and index_plan is not None
            and index_plan.kind
            in {
                "range",
                "device-labels",
                "host-labels",
                "host-labels-take",
            }
        ):
            attributes_preserve_index = False
        taken_index_plan = (
            None
            if index_plan is None
            else index_plan.take(
                normalized,
                preserve_index=preserve_index,
                unique=unique,
            )
        )
        taken_geometry = self.geometry.take(normalized, unique=unique)
        if rowset is not None:
            taken_geometry.apply_rowset_proofs(rowset)
        return type(self)(
            attributes=self.attributes.take(
                normalized,
                preserve_index=attributes_preserve_index,
            ),
            geometry=taken_geometry,
            geometry_name=self.geometry_name,
            column_order=self.column_order,
            attrs=self.attrs,
            secondary_geometry=tuple(
                NativeGeometryColumn(
                    column.name,
                    column.geometry.take(normalized, unique=unique),
                )
                for column in self.secondary_geometry
            ),
            provenance=(
                self.provenance.take(normalized)
                if self.provenance is not None and hasattr(self.provenance, "take")
                else self.provenance
            ),
            geometry_metadata=(
                None if self.geometry_metadata is None else self.geometry_metadata.take(normalized)
            ),
            index_plan=taken_index_plan,
            terminal_geodataframe_materializer=terminal_materializer,
            terminal_geodataframe_materializer_owns_export=(
                self.terminal_geodataframe_materializer_owns_export
                and terminal_materializer is not None
            ),
        )


@dataclass(frozen=True)
class NativeTabularSelection:
    """Dynamic logical rows over an exact capacity-sized tabular result.

    ``NativeTabularResult`` intentionally retains an exact Python row count.
    This carrier keeps dynamic cardinality in ``NativeDeviceSelection`` until
    a native consumer or explicit terminal export compacts the selected prefix.
    """

    capacity_result: NativeTabularResult
    selection: Any
    public_index_source_plan: Any | None = None
    public_index_source_rows: Any | None = None

    def __post_init__(self) -> None:
        from vibespatial.api._native_rowset import NativeDeviceSelection

        if not isinstance(self.selection, NativeDeviceSelection):
            raise TypeError("NativeTabularSelection requires a NativeDeviceSelection")
        source_row_count = self.selection.source_row_count
        if source_row_count is None or int(source_row_count) != len(
            self.capacity_result.attributes
        ):
            raise ValueError("NativeTabularSelection source_row_count must match capacity result")
        if (self.public_index_source_plan is None) != (self.public_index_source_rows is None):
            raise ValueError("public index source plan and rows must be provided together")
        if self.public_index_source_rows is not None and _row_aligned_size(
            self.public_index_source_rows
        ) != len(self.capacity_result.attributes):
            raise ValueError("public index source rows must align with the capacity result")

    @property
    def capacity(self) -> int:
        return self.selection.capacity

    @property
    def logical_count(self):
        return self.selection.logical_count

    @property
    def provenance(self):
        return self.capacity_result.provenance

    @property
    def terminal_geodataframe_materializer(self):
        return self.capacity_result.terminal_geodataframe_materializer

    @property
    def terminal_geodataframe_materializer_owns_export(self) -> bool:
        return self.capacity_result.terminal_geodataframe_materializer_owns_export

    def sort_selected_by_int64(self, values: Any) -> NativeTabularSelection:
        """Order active rows by a base-aligned device int64 vector."""
        import cupy as cp

        d_keys = self.selection.gather_capacity(
            cp.asarray(values, dtype=cp.int64),
            fill_value=2**63 - 1,
        )
        d_order = cp.argsort(d_keys).astype(cp.int64, copy=False)
        selection = replace(
            self.selection,
            positions=cp.asarray(self.selection.positions, dtype=cp.int64)[d_order],
            ordered=True,
            full_selection_implies_identity=False,
        )
        return replace(self, selection=selection)

    def with_public_index_source(
        self,
        index_plan: Any,
        source_rows: Any,
    ) -> NativeTabularSelection:
        """Attach source-label semantics without materializing dynamic labels."""
        return replace(
            self,
            public_index_source_plan=index_plan,
            public_index_source_rows=source_rows,
        )

    def to_native_tabular_result(
        self,
        *,
        surface: str = "vibespatial.api.NativeTabularSelection.to_native_tabular_result",
        strict_disallowed: bool = True,
    ) -> NativeTabularResult:
        """Compact at an explicit native-consumer or terminal export boundary."""
        rowset = self.selection.compact_rowset(
            surface=surface,
            strict_disallowed=strict_disallowed,
        )
        result = self.capacity_result.take(
            rowset,
            preserve_index=self.public_index_source_plan is None,
            unique=self.selection.unique,
        )
        if self.public_index_source_plan is None:
            return result

        import cupy as cp

        d_public_rows = cp.asarray(
            self.public_index_source_rows,
            dtype=cp.int64,
        )[cp.asarray(rowset.positions, dtype=cp.int64)]
        return replace(
            result,
            index_plan=self.public_index_source_plan.take(
                d_public_rows,
                preserve_index=True,
                unique=self.selection.unique,
                strict_disallowed=False,
            ),
        )

    def physicalize_known_count(
        self,
        row_count: int,
    ) -> NativeTabularResult:
        """Physicalize a selected prefix whose cardinality is already public."""
        from vibespatial.api._native_rowset import NativeRowSet

        row_count = int(row_count)
        if row_count < 0 or row_count > self.capacity:
            raise ValueError("known selection row count must fit within capacity")
        rowset = NativeRowSet.from_positions(
            self.selection.positions[:row_count],
            source_token=self.selection.source_token,
            source_row_count=self.selection.source_row_count,
            ordered=self.selection.ordered,
            unique=self.selection.unique,
            identity=(
                self.selection.full_selection_implies_identity
                and self.selection.source_row_count is not None
                and row_count == int(self.selection.source_row_count)
            ),
            geometry_family_domain=self.selection.geometry_family_domain,
            trusted_all_valid_rows=self.selection.trusted_all_valid_rows,
        )
        result = self.capacity_result.take(
            rowset,
            preserve_index=self.public_index_source_plan is None,
            unique=self.selection.unique,
        )
        if self.public_index_source_plan is None:
            return result

        import cupy as cp

        d_public_rows = cp.asarray(
            self.public_index_source_rows,
            dtype=cp.int64,
        )[cp.asarray(rowset.positions, dtype=cp.int64)]
        return replace(
            result,
            index_plan=self.public_index_source_plan.take(
                d_public_rows,
                preserve_index=True,
                unique=self.selection.unique,
                strict_disallowed=False,
            ),
        )

    def to_geodataframe(self, *, lazy_public_index: bool = True):
        return self.to_native_tabular_result(
            surface="vibespatial.api.NativeTabularSelection.to_geodataframe",
            strict_disallowed=False,
        ).to_geodataframe(lazy_public_index=lazy_public_index)

    def to_arrow(self, **kwargs):
        return self.to_native_tabular_result(
            surface="vibespatial.api.NativeTabularSelection.to_arrow",
            strict_disallowed=False,
        ).to_arrow(**kwargs)

    def to_parquet(self, path, **kwargs) -> None:
        self.to_native_tabular_result(
            surface="vibespatial.api.NativeTabularSelection.to_parquet",
            strict_disallowed=False,
        ).to_parquet(path, **kwargs)

    def to_feather(self, path, **kwargs) -> None:
        self.to_native_tabular_result(
            surface="vibespatial.api.NativeTabularSelection.to_feather",
            strict_disallowed=False,
        ).to_feather(path, **kwargs)


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
    frame = _lazy_public_attribute_frame(
        attributes,
        surface="vibespatial.api.NativeTabularResult.to_geodataframe",
    )
    if frame is None:
        frame = attributes.to_pandas(copy=False)
    geometry_names = [column.name for column in geometry_columns]
    overlap = [name for name in frame.columns if name in geometry_names]
    if overlap:
        raise ValueError("attribute columns must not overlap exported geometry column names")
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
            ordered_frame = pd.DataFrame(index=attributes.index)
            for name in attribute_order:
                ordered_frame[name] = frame[name]
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
            ordered_frame.attrs = _copy_public_frame_attrs(frame)
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
    rebuilt.attrs = _copy_public_frame_attrs(frame)
    return _replace_geometry_column_preserving_backing(
        rebuilt,
        active_geometry.values,
        crs=active_geometry.crs,
    )


__all__ = [
    "GeometryNativeResult",
    "NativeAttributeTable",
    "NativeGeometryComposition",
    "NativeGeometryCompositionPart",
    "NativeGeometryColumn",
    "NativeGeometryProvenance",
    "NativeReadProvenance",
    "NativeTabularSelection",
    "NativeTabularResult",
    "_host_array",
    "_host_row_positions",
    "_materialize_attribute_geometry_frame",
    "_normalize_row_selection",
    "_replace_geometry_column_preserving_backing",
    "_set_active_geometry_name",
    "native_attribute_table_from_arrow_table",
]
