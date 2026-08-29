from __future__ import annotations

import ast
import json
import typing
import warnings
from contextvars import ContextVar
from dataclasses import replace as dataclass_replace
from typing import Any, Literal

import numpy as np
import pandas as pd
import shapely.errors
from pandas import DataFrame, Series
from pandas._libs import lib
from pandas.api.extensions import ExtensionArray
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

import vibespatial.api as geopandas
from vibespatial.api._compat import HAS_PYPROJ, PANDAS_GE_30
from vibespatial.api._decorator import doc
from vibespatial.api.explore import _explore
from vibespatial.api.geo_base import (
    GeoPandasBase,
    _attach_native_expression,
    _is_geometry_like_dtype,
    _native_boolean_rowset_from_public_mask,
    _native_expression_from_public_series,
    _record_native_display_export,
    _record_native_public_export_boundary,
    is_geometry_type,
)
from vibespatial.api.geometry_array import GeometryArray, from_shapely, to_wkb, to_wkt
from vibespatial.api.geoseries import GeoSeries
from vibespatial.overlay.dissolve import (
    evaluate_geopandas_dissolve,
    evaluate_geopandas_lazy_dissolve,
)
from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event
from vibespatial.runtime.residency import Residency

if PANDAS_GE_30:
    from pandas.core.accessor import Accessor
else:
    from pandas.core.accessor import CachedAccessor as Accessor


if typing.TYPE_CHECKING:
    import os
    from collections.abc import Iterable

    import folium
    import sqlalchemy.text
    from pyproj import CRS

    from vibespatial.api.io.arrow import (
        PARQUET_GEOMETRY_ENCODINGS,
        SUPPORTED_VERSIONS_LITERAL,
    )


def _device_geometry_from_shapely(data, crs: Any | None = None):
    """Build a device-backed geometry array from Shapely values when possible."""
    from vibespatial.geometry.device_array import DeviceGeometryArray
    from vibespatial.geometry.owned import from_shapely_geometries

    if isinstance(data, Series):
        values = np.asarray(data)
    else:
        values = np.asarray(data, dtype=object)
    owned = from_shapely_geometries(values.tolist(), residency=Residency.DEVICE)
    return DeviceGeometryArray._from_owned(owned, crs=crs)


def _ensure_geometry_respecting_device_preference(
    data,
    *,
    crs: Any | None = None,
    surface: str,
):
    try:
        return _device_geometry_from_shapely(data, crs=crs)
    except Exception as device_error:
        host_geometry = from_shapely(np.asarray(data, dtype=object), crs=crs)
        record_fallback_event(
            surface=surface,
            requested=ExecutionMode.GPU,
            selected=ExecutionMode.CPU,
            reason="explicit CPU fallback while rebuilding a device-backed geometry column from Shapely values",
            detail=f"{type(device_error).__name__}: {device_error}",
            pipeline="geodataframe",
        )
        return host_geometry


def _current_public_geometry_crs(frame, column_name: Any, fallback: Any):
    if column_name not in frame.columns:
        return fallback
    try:
        series = frame[column_name]
    except Exception:
        return fallback
    if hasattr(series, "crs"):
        return series.crs
    values = getattr(series, "values", None)
    if hasattr(values, "crs"):
        return values.crs
    return fallback


def _public_attribute_frame_for_terminal_export(frame, native_result):
    """Use the public frame as the terminal attribute carrier when it is exact."""
    attributes = native_result.attributes
    attribute_columns = tuple(attributes.columns)
    if len(frame.index) != len(attributes):
        return None
    if any(column not in frame.columns for column in attribute_columns):
        return None
    if not attribute_columns:
        return pd.DataFrame(index=frame.index)
    public_attributes = frame.loc[:, list(attribute_columns)].copy(deep=False)
    if not public_attributes.index.equals(frame.index):
        public_attributes.index = frame.index
    return public_attributes


def _native_tabular_result_with_public_metadata(
    frame,
    native_state,
    *,
    index=None,
    preserve_native_attributes: bool = False,
):
    """Keep private buffers but honor current public metadata at export time."""
    from vibespatial.api._native_result_core import (
        NativeGeometryColumn,
        NativeTabularResult,
    )
    from vibespatial.api._native_rowset import NativeIndexPlan

    result = native_state.to_native_tabular_result()
    attributes = result.attributes
    index_plan = result.index_plan
    if (
        not preserve_native_attributes
        and index is not False
        and getattr(attributes, "device_table", None) is not None
    ):
        public_attributes = _public_attribute_frame_for_terminal_export(frame, result)
        if public_attributes is not None:
            attributes = public_attributes
            index_plan = NativeIndexPlan.from_index(frame.index)
    geometry = result.geometry.with_crs(
        _current_public_geometry_crs(
            frame,
            result.geometry_name,
            result.geometry.crs,
        )
    )
    secondary_geometry = tuple(
        NativeGeometryColumn(
            column.name,
            column.geometry.with_crs(
                _current_public_geometry_crs(
                    frame,
                    column.name,
                    column.geometry.crs,
                )
            ),
        )
        for column in result.secondary_geometry
    )
    attrs = frame.attrs.copy() or None
    if (
        attributes is result.attributes
        and geometry is result.geometry
        and secondary_geometry == result.secondary_geometry
        and attrs == result.attrs
        and index_plan == result.index_plan
    ):
        return result
    return NativeTabularResult(
        attributes=attributes,
        geometry=geometry,
        geometry_name=result.geometry_name,
        column_order=result.column_order,
        attrs=attrs,
        secondary_geometry=secondary_geometry,
        provenance=result.provenance,
        geometry_metadata=result.geometry_metadata,
        index_plan=index_plan,
        terminal_geodataframe_materializer=result.terminal_geodataframe_materializer,
        terminal_geodataframe_materializer_owns_export=(
            result.terminal_geodataframe_materializer_owns_export
        ),
    )


def _ensure_geometry(
    data,
    crs: Any | None = None,
    *,
    prefer_device: bool = False,
    fallback_surface: str = "GeoDataFrame.geometry",
) -> GeoSeries | GeometryArray:
    """
    Ensure the data is of geometry dtype or converted to it.

    If input is a (Geo)Series, output is a GeoSeries, otherwise output
    is GeometryArray.

    If the input is a GeometryDtype with a set CRS, `crs` is ignored.
    """
    if is_geometry_type(data):
        if isinstance(data, Series):
            data = GeoSeries(data)
        if data.crs is None and crs is not None:
            # Avoids caching issues/crs sharing issues
            data = data.copy()
            if isinstance(data, GeometryArray):
                data.crs = crs
            elif hasattr(data, "array"):
                data.array.crs = crs
            else:
                data.crs = crs
        return data
    else:
        if isinstance(data, Series):
            if prefer_device:
                out = _ensure_geometry_respecting_device_preference(
                    data,
                    crs=crs,
                    surface=fallback_surface,
                )
            else:
                out = from_shapely(np.asarray(data), crs=crs)
            return GeoSeries(out, index=data.index, name=data.name)
        else:
            if prefer_device:
                out = _ensure_geometry_respecting_device_preference(
                    data,
                    crs=crs,
                    surface=fallback_surface,
                )
            else:
                out = from_shapely(data, crs=crs)
            return out


def _is_native_state_column_projection_key(key, columns: pd.Index) -> bool:
    """Return True only for explicit column-list projections.

    DataFrame-returning pandas selectors include row filters and slices. Native
    sidecars must not survive those paths until a row-flow contract exists.
    """
    if isinstance(key, (slice, Series, DataFrame)):
        return False
    if isinstance(key, np.ndarray):
        if key.dtype == bool:
            return False
        labels = key.tolist()
    elif isinstance(key, pd.Index):
        if key.dtype == bool:
            return False
        labels = list(key)
    elif pd.api.types.is_list_like(key) and not isinstance(key, (str, bytes)):
        labels = list(key)
        if labels and all(isinstance(label, (bool, np.bool_)) for label in labels):
            return False
    else:
        return False
    return all(label in columns for label in labels)


def _native_boolean_filter_rowset(key, state, *, public_index=None):
    """Return an exact rowset for admitted boolean filters, otherwise decline."""
    if state is None:
        return None
    index_plan = state.index_plan
    if index_plan.kind == "range":
        public_index = index_plan.to_public_index()
    elif index_plan.kind == "host-labels":
        public_index = index_plan.index
    elif index_plan.kind == "device-labels":
        if public_index is None:
            return None
    else:
        return None
    if isinstance(key, Series):
        if public_index is None or not key.index.equals(public_index):
            return None
        native_rowset = _native_boolean_rowset_from_aligned_mask(key, state)
        if native_rowset is not None:
            return native_rowset
        if not pd.api.types.is_bool_dtype(key.dtype):
            return None
        values = key.to_numpy(dtype=bool, na_value=False)
    elif isinstance(key, pd.Index):
        if key.dtype != bool:
            return None
        values = key.to_numpy(dtype=bool, na_value=False)
    elif isinstance(key, np.ndarray):
        if key.dtype != bool:
            return None
        values = np.asarray(key, dtype=bool)
    elif pd.api.types.is_list_like(key) and not isinstance(key, (str, bytes)):
        values = np.asarray(list(key))
        if values.dtype != bool:
            return None
    else:
        return None
    if values.ndim != 1 or values.shape[0] != state.row_count:
        return None

    identity = bool(values.all())
    if identity:
        positions = np.arange(state.row_count, dtype=np.int64)
    else:
        positions = np.flatnonzero(values).astype(np.int64, copy=False)
        positions = _maybe_device_row_positions(positions, state)
    from vibespatial.api._native_rowset import NativeRowSet

    return NativeRowSet.from_positions(
        positions,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
        identity=identity,
    )


def _native_boolean_rowset_from_aligned_mask(key, state):
    rowset = _native_boolean_rowset_from_public_mask(key, state)
    if rowset is None:
        from vibespatial.api._native_public_arrays import (
            native_boolean_rowset_from_mask_array,
        )

        rowset = native_boolean_rowset_from_mask_array(key)
        if rowset is None:
            return None
        if (
            rowset.source_row_count is not None
            and int(rowset.source_row_count) != int(state.row_count)
        ):
            return None

        from vibespatial.api._native_rowset import NativeRowSet

        rowset = NativeRowSet.from_positions(
            rowset.positions,
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=rowset.ordered,
            unique=rowset.unique,
            identity=rowset.identity,
            geometry_family_domain=rowset.geometry_family_domain,
            trusted_all_valid_rows=rowset.trusted_all_valid_rows,
        )
    return _native_rowset_mark_identity_if_full(rowset, state)


def _native_rowset_mark_identity_if_full(rowset, state):
    if (
        rowset is not None
        and not bool(getattr(rowset, "identity", False))
        and bool(getattr(rowset, "ordered", False))
        and bool(getattr(rowset, "unique", False))
        and getattr(rowset, "source_row_count", None) is not None
        and int(rowset.source_row_count) == int(state.row_count)
        and len(rowset) == int(state.row_count)
    ):
        from vibespatial.api._native_rowset import NativeRowSet

        return NativeRowSet.from_positions(
            rowset.positions,
            source_token=rowset.source_token,
            source_row_count=rowset.source_row_count,
            ordered=rowset.ordered,
            unique=rowset.unique,
            identity=True,
            geometry_family_domain=rowset.geometry_family_domain,
            trusted_all_valid_rows=rowset.trusted_all_valid_rows,
        )
    return rowset


def _maybe_device_row_positions(positions, state):
    if _native_state_can_take_device_row_positions(state):
        try:
            from vibespatial.runtime import has_gpu_runtime

            if has_gpu_runtime():
                import cupy as cp

                positions = cp.asarray(positions, dtype=cp.int64)
        except Exception:
            pass
    return positions


def _native_iloc_row_positions(key, row_count: int):
    if isinstance(key, slice):
        positions = np.arange(row_count, dtype=np.int64)[key]
        return positions, True
    if isinstance(key, np.ndarray):
        values = key
    elif isinstance(key, pd.Index):
        values = key.to_numpy()
    elif pd.api.types.is_list_like(key) and not isinstance(key, (str, bytes)):
        values = np.asarray(list(key))
    else:
        return None, False

    if values.ndim != 1:
        return None, False
    if values.dtype == bool:
        if values.shape[0] != row_count:
            return None, False
        positions = np.flatnonzero(values).astype(np.int64, copy=False)
        return positions, True
    if not np.issubdtype(values.dtype, np.integer):
        return None, False

    positions = values.astype(np.int64, copy=False)
    positions = np.where(positions < 0, positions + row_count, positions)
    strictly_increasing = positions.size < 2 or bool(
        np.all(positions[1:] > positions[:-1])
    )
    unique = strictly_increasing or positions.size == np.unique(positions).size
    return positions, bool(unique)


def _native_result_accepts_frame_state(result) -> bool:
    if not isinstance(result, DataFrame):
        return False
    if result.dtypes.map(_is_geometry_like_dtype).sum() <= 0:
        return False
    return bool(result.columns.is_unique)


def _attach_native_state_from_row_positions(
    owner,
    result,
    positions,
    *,
    unique: bool,
    index_override=None,
) -> None:
    if not _native_result_accepts_frame_state(result):
        return

    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return

    if positions is None or len(result) != int(positions.size):
        return

    identity_take = positions.size == state.row_count and bool(
        np.array_equal(positions, np.arange(state.row_count, dtype=np.int64))
    )
    if identity_take:
        taken = state
    else:
        rowset = NativeRowSet.from_positions(
            _maybe_device_row_positions(positions, state),
            source_token=state.lineage_token,
            source_row_count=state.row_count,
            ordered=True,
            unique=unique,
        )
        taken = state.take(rowset, preserve_index=True)

    projected = taken.project_columns(tuple(result.columns))
    if projected is not None:
        if index_override is not None:
            projected = projected.with_index(index_override)
        attach_native_state(result, projected)


def _attach_native_state_after_iloc(owner, key, result) -> None:
    state_row_count = len(owner)
    row_key = key[0] if isinstance(key, tuple) and key else key
    positions, unique = _native_iloc_row_positions(row_key, state_row_count)
    _attach_native_state_from_row_positions(
        owner,
        result,
        positions,
        unique=unique,
    )


def _is_full_slice(key) -> bool:
    return isinstance(key, slice) and key.start is None and key.stop is None and key.step is None


def _take_public_frame_with_native_rowset(
    frame,
    rowset,
    positions: np.ndarray,
    *,
    source_native_state,
    geometry_column,
):
    if (
        bool(getattr(rowset, "identity", False))
        and len(rowset) == int(source_native_state.row_count)
    ):
        source_native_state._apply_rowset_geometry_proofs(
            source_native_state.geometry,
            rowset,
        )
        projected = source_native_state.project_columns(tuple(frame.columns))
        if projected is None:
            return None
        return _shallow_public_frame_with_native_state(
            frame,
            projected,
            geometry_column=geometry_column,
        )

    taken_state = source_native_state.take(
        rowset,
        preserve_index=True,
        index_positions=positions,
    )

    projected = taken_state.project_columns(tuple(frame.columns))
    if projected is None:
        return None

    return _public_frame_from_native_state(
        frame,
        projected,
        geometry_column=geometry_column,
    )


def _native_device_column_policies(attributes, columns):
    """Resolve device column contracts across a composed attribute table."""
    requested = tuple(dict.fromkeys(columns))
    if not requested or not bool(getattr(attributes, "is_device_backed", False)):
        return {}
    parts = getattr(attributes, "parts", None)
    if parts is None:
        return attributes.device_column_policies(requested)

    policies = {}
    remaining = set(requested)
    for part in parts:
        part_columns = tuple(column for column in requested if column in part.columns)
        if not part_columns:
            continue
        part_policies = _native_device_column_policies(part, part_columns)
        if any(column not in part_policies for column in part_columns):
            return {}
        policies.update(part_policies)
        remaining.difference_update(part_columns)
    if remaining:
        return {}
    return {column: policies[column] for column in requested}


def _native_pylibcudf_columns(attributes, columns):
    """Resolve requested device columns without combining parts through Arrow."""
    requested = tuple(columns)
    parts = getattr(attributes, "parts", None)
    if parts is None:
        return attributes.to_pylibcudf_columns(requested)

    columns_by_name = {}
    for part in parts:
        part_columns = tuple(column for column in requested if column in part.columns)
        if not part_columns:
            continue
        resolved = _native_pylibcudf_columns(part, part_columns)
        columns_by_name.update(zip(part_columns, resolved, strict=True))
    if any(column not in columns_by_name for column in requested):
        raise KeyError("requested columns are not present in native device table parts")
    return [columns_by_name[column] for column in requested]


def _native_pylibcudf_missing_normalized_columns(columns, *, stream):
    """Return sort/distinct keys with floating NaNs represented as nulls."""
    import pylibcudf as plc

    normalized = []
    for column in columns:
        if column.type().id() in {plc.types.TypeId.FLOAT32, plc.types.TypeId.FLOAT64}:
            null_mask, null_count = plc.transform.nans_to_nulls(
                column,
                stream=stream,
            )
            column = column.with_mask(null_mask, int(null_count))
        normalized.append(column)
    return normalized


def _native_device_column_storage_bytes(attributes, columns) -> int | None:
    """Return source device-buffer bytes for logical columns without a gather."""
    requested = tuple(dict.fromkeys(columns))
    parts = getattr(attributes, "parts", None)
    if parts is not None:
        total = 0
        remaining = set(requested)
        for part in parts:
            part_columns = tuple(column for column in requested if column in part.columns)
            if not part_columns:
                continue
            part_bytes = _native_device_column_storage_bytes(part, part_columns)
            if part_bytes is None:
                return None
            total += part_bytes
            remaining.difference_update(part_columns)
        return None if remaining else total

    table = getattr(attributes, "device_table", None)
    if table is None or not hasattr(table, "columns"):
        return None
    positions = {name: index for index, name in enumerate(attributes.columns)}
    if any(column not in positions for column in requested):
        return None
    try:
        source_columns = table.columns()
        return sum(
            int(source_columns[positions[column]].device_buffer_size())
            for column in requested
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _native_distinct_required_bytes(attributes, labels, row_count: int) -> int | None:
    """Conservatively bound distinct key copies, row orders, and workspace."""
    source_bytes = _native_device_column_storage_bytes(attributes, labels)
    if source_bytes is None:
        return None
    index_width = 4 if row_count <= np.iinfo(np.int32).max else 8
    # Stable order can retain several key-sized radix/sort buffers. A row-indirected
    # input also needs one key gather before sorting; source_bytes deliberately uses
    # the full physical columns so the estimate stays conservative for such views.
    key_and_workspace_bytes = 5 * max(source_bytes, row_count * len(labels))
    row_order_bytes = row_count * (5 * index_width + 4)
    return key_and_workspace_bytes + row_order_bytes + (1 << 20)


def _native_sort_required_bytes(attributes, labels, row_count: int) -> int | None:
    """Conservatively bound stable-sort key copies, row order, and workspace."""
    source_bytes = _native_device_column_storage_bytes(attributes, labels)
    if source_bytes is None:
        return None
    index_width = 4 if row_count <= np.iinfo(np.int32).max else 8
    key_and_workspace_bytes = 4 * max(source_bytes, row_count * len(labels))
    row_order_bytes = row_count * (4 * index_width)
    return key_and_workspace_bytes + row_order_bytes + (1 << 20)


def _native_column_requires_numpy_public_dtype(attributes, column) -> bool:
    """Whether a lazy device column must expose a NumPy dtype to pandas users."""
    parts = getattr(attributes, "parts", None)
    if parts is not None:
        for part in parts:
            if column in part.columns:
                return _native_column_requires_numpy_public_dtype(part, column)
        return False

    schema = getattr(attributes, "schema_override", None)
    if schema is None or column not in attributes.columns:
        return False
    position = tuple(attributes.columns).index(column)
    metadata = schema.field(position).metadata or {}
    return metadata.get(b"vibespatial:public_dtype") == b"numpy"


def _native_lazy_public_attribute_frame(native_state):
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeNumericExpressionArray,
    )

    index_plan = getattr(native_state, "index_plan", None)
    index = _native_public_index_from_plan(index_plan)
    if index is None:
        return None
    columns: dict[Any, Any] = {}
    attributes = native_state.attributes
    attribute_columns = tuple(attributes.columns)
    numeric_arrays = attributes.numeric_column_arrays(attribute_columns)
    if numeric_arrays is None:
        numeric_columns: tuple[Any, ...] = ()
        if bool(getattr(attributes, "is_device_backed", False)):
            policies = _native_device_column_policies(attributes, attribute_columns)
            numeric_columns = tuple(
                column
                for column in attribute_columns
                if (
                    policy := policies.get(column)
                ) is not None
                and policy.can_compute_numeric
            )
        numeric_arrays = (
            attributes.numeric_column_arrays(numeric_columns)
            if numeric_columns
            else {}
        )
        if numeric_arrays is None:
            numeric_arrays = {}
    if numeric_arrays:
        from vibespatial.api._native_expression import NativeExpression

    for column in attribute_columns:
        values = numeric_arrays.get(column)
        if values is not None:
            columns[column] = NativeNumericExpressionArray(
                NativeExpression(
                    operation=f"attribute.{column}",
                    values=values,
                    source_token=native_state.lineage_token,
                    source_row_count=native_state.row_count,
                    dtype=str(getattr(values, "dtype", "")) or None,
                    precision="source",
                ),
                export_surface="vibespatial.api.GeoDataFrame.__getitem__",
                export_operation="native_attribute_column_to_public_series",
            )
        else:
            columns[column] = NativeAttributeColumnArray(
                attributes,
                column,
                export_surface="vibespatial.api.GeoDataFrame.__getitem__",
                export_operation="native_attribute_column_to_public_series",
                source_token=native_state.lineage_token,
            )
    if not columns:
        return pd.DataFrame(index=index)
    return pd.DataFrame(columns, index=index)


def _native_public_index_from_plan(index_plan):
    from vibespatial.api._native_public_arrays import native_public_index_from_plan

    return native_public_index_from_plan(index_plan)


def _take_public_frame_with_native_state(
    frame,
    rowset,
    *,
    source_native_state,
    geometry_column,
    preserve_index: bool = True,
    index_positions=None,
):
    if (
        preserve_index
        and bool(getattr(rowset, "identity", False))
        and len(rowset) == int(source_native_state.row_count)
    ):
        source_native_state._apply_rowset_geometry_proofs(
            source_native_state.geometry,
            rowset,
        )
        projected = source_native_state.project_columns(tuple(frame.columns))
        if projected is None:
            return None
        return _shallow_public_frame_with_native_state(
            frame,
            projected,
            geometry_column=geometry_column,
        )

    taken_state = source_native_state.take(
        rowset,
        preserve_index=preserve_index,
        index_positions=index_positions,
    )
    projected = taken_state.project_columns(tuple(frame.columns))
    if projected is None:
        return None
    return _public_frame_from_native_state(
        frame,
        projected,
        geometry_column=geometry_column,
    )


def _shallow_public_frame_with_native_state(
    frame,
    native_state,
    *,
    geometry_column,
):
    from vibespatial.api._native_state import attach_native_state

    try:
        result = frame._constructor_from_mgr(frame._mgr, frame._mgr.axes)
        result = result.__finalize__(frame)
    except Exception:
        result = pd.DataFrame.copy(frame, deep=False)
    if type(result) is pd.DataFrame:
        result.__class__ = type(frame)
    if hasattr(result, "_geometry_column_name"):
        result._geometry_column_name = geometry_column
    result.attrs = frame.attrs.copy()
    attach_native_state(result, native_state)
    return result


def _public_frame_columns_are_native_lazy(frame) -> bool:
    from vibespatial.api._native_public_arrays import (
        NativeAttributeColumnArray,
        NativeNumericExpressionArray,
    )

    for column in frame.columns:
        series = pd.DataFrame.__getitem__(frame, column)
        if not isinstance(series, Series):
            return False
        if _is_geometry_like_dtype(getattr(series, "dtype", None)):
            continue
        if not isinstance(
            getattr(series, "array", None),
            (NativeAttributeColumnArray, NativeNumericExpressionArray),
        ):
            return False
    return True


_NATIVE_PUBLIC_METADATA_DEFAULT = object()


def _public_frame_from_native_state(
    frame,
    native_state,
    *,
    geometry_column,
    attrs=_NATIVE_PUBLIC_METADATA_DEFAULT,
    columns_name=_NATIVE_PUBLIC_METADATA_DEFAULT,
):
    from vibespatial.api._native_state import attach_native_state

    attributes = _native_lazy_public_attribute_frame(native_state)
    if attributes is None:
        return None
    geometry = native_state.geometry.to_geoseries(
        index=attributes.index,
        name=native_state.geometry_name,
    )
    result = attributes
    if native_state.geometry_name in result.columns:
        result = result.copy(deep=False)
        result[native_state.geometry_name] = geometry
    else:
        geometry_position = native_state.column_order.index(native_state.geometry_name)
        result.insert(geometry_position, native_state.geometry_name, geometry)
    result.__class__ = type(frame)
    if geometry_column in result:
        result._geometry_column_name = geometry_column
    try:
        finalized = result.__finalize__(frame)
        if finalized is not None:
            result = finalized
    except Exception:
        pass
    result.attrs = (
        frame.attrs.copy()
        if attrs is _NATIVE_PUBLIC_METADATA_DEFAULT
        else dict(attrs)
    )
    result.columns.name = (
        frame.columns.name
        if columns_name is _NATIVE_PUBLIC_METADATA_DEFAULT
        else columns_name
    )
    if dict(getattr(native_state, "attrs", {}) or {}) != result.attrs:
        native_state = dataclass_replace(native_state, attrs=result.attrs.copy())
    attach_native_state(result, native_state)
    return result


def _native_loc_row_positions(key, source_index, result_index):
    """Return exact source row positions for admitted label-based ``.loc`` takes."""
    row_key = key[0] if isinstance(key, tuple) and key else key
    if _is_full_slice(row_key):
        positions = np.arange(len(source_index), dtype=np.int64)
    elif isinstance(row_key, Series):
        if row_key.dtype != bool or not row_key.index.equals(source_index):
            return None, False
        positions = np.flatnonzero(row_key.to_numpy(dtype=bool, na_value=False)).astype(
            np.int64,
            copy=False,
        )
    elif isinstance(row_key, pd.Index):
        if row_key.dtype == bool:
            positions = np.flatnonzero(
                row_key.to_numpy(dtype=bool, na_value=False),
            ).astype(np.int64, copy=False)
        else:
            positions = source_index.get_indexer_for(row_key)
    elif isinstance(row_key, np.ndarray):
        if row_key.ndim != 1:
            return None, False
        if row_key.dtype == bool:
            positions = np.flatnonzero(np.asarray(row_key, dtype=bool)).astype(
                np.int64,
                copy=False,
            )
        else:
            positions = source_index.get_indexer_for(row_key.tolist())
    elif pd.api.types.is_list_like(row_key) and not isinstance(row_key, (str, bytes, tuple)):
        labels = list(row_key)
        if labels and all(isinstance(label, (bool, np.bool_)) for label in labels):
            positions = np.flatnonzero(np.asarray(labels, dtype=bool)).astype(
                np.int64,
                copy=False,
            )
        else:
            positions = source_index.get_indexer_for(labels)
    elif row_key is not None:
        if isinstance(row_key, tuple) and getattr(source_index, "nlevels", 1) > 1:
            return None, False
        positions = source_index.get_indexer_for([row_key])
    else:
        return None, False

    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or np.any(positions < 0):
        return None, False
    if len(result_index) != int(positions.size):
        return None, False
    if not source_index.take(positions).equals(result_index):
        return None, False
    unique = positions.size == np.unique(positions).size
    return positions, bool(unique)


def _attach_native_state_after_loc(owner, key, result) -> None:
    if _native_result_accepts_frame_state(result):
        source_index = getattr(owner, "index", None)
        result_index = getattr(result, "index", None)
        if source_index is not None and result_index is not None:
            positions, unique = _native_loc_row_positions(
                key,
                source_index,
                result_index,
            )
            if positions is not None:
                _attach_native_state_from_row_positions(
                    owner,
                    result,
                    positions,
                    unique=unique,
                )
                return
    _attach_native_state_from_result_index(owner, result)


def _native_loc_from_boolean_mask(owner, key):
    """Serve an aligned native boolean ``.loc`` without exporting its mask."""
    row_key = key
    if isinstance(key, tuple):
        if not key:
            return None
        row_key = key[0]
        column_key = key[1] if len(key) > 1 else slice(None)
        if not _is_full_slice(column_key):
            return None

    try:
        from vibespatial.api._native_state import get_native_state
    except Exception:
        return None

    state = get_native_state(owner)
    if state is None:
        return None
    if isinstance(row_key, Series) and not row_key.index.equals(owner.index):
        return None

    rowset = _native_boolean_rowset_from_aligned_mask(row_key, state)
    if rowset is None:
        return None
    if rowset.is_device and not _native_state_can_take_device_row_positions(state):
        return None
    return _take_public_frame_with_native_state(
        owner,
        rowset,
        source_native_state=state,
        geometry_column=owner._geometry_column_name,
    )


def _native_loc_from_lazy_unique_index(owner, key):
    """Serve ``source.loc[joined.index.unique()]`` from native row positions."""
    row_key = key
    if isinstance(key, tuple):
        if not key:
            return None
        row_key = key[0]
        column_key = key[1] if len(key) > 1 else slice(None)
        if not _is_full_slice(column_key):
            return None

    if not isinstance(row_key, pd.Index):
        return None
    try:
        from vibespatial.api._native_public_arrays import NativeIndexLabelsArray
        from vibespatial.api._native_rowset import NativeRowSet
        from vibespatial.api._native_state import get_native_state
    except Exception:
        return None

    index_array = getattr(row_key, "array", None)
    if not isinstance(index_array, NativeIndexLabelsArray):
        return None
    plan = index_array.index_plan
    positions = getattr(plan, "selection_positions", None)
    if positions is None:
        return None

    state = get_native_state(owner)
    if state is None:
        return None
    if getattr(plan, "selection_source_token", None) != state.lineage_token:
        return None
    if getattr(plan, "selection_source_row_count", None) != state.row_count:
        return None
    if not state.index_plan.admits_unique_label_selection:
        return None
    if hasattr(positions, "__cuda_array_interface__") and not _native_state_can_take_device_row_positions(
        state,
    ):
        return None

    rowset = NativeRowSet.from_positions(
        positions,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
    )
    return _take_public_frame_with_native_state(
        owner,
        rowset,
        source_native_state=state,
        geometry_column=owner._geometry_column_name,
    )


def _ordered_subset_index_positions(source_index, result_index):
    """Return source positions when ``result_index`` is an ordered source subset."""
    if len(result_index) > len(source_index):
        return None
    if len(result_index) == 0:
        return np.empty(0, dtype=np.int64)

    positions = np.empty(len(result_index), dtype=np.int64)
    source_pos = 0
    source_len = len(source_index)
    for result_pos in range(len(result_index)):
        result_label = result_index[result_pos : result_pos + 1]
        while source_pos < source_len:
            if source_index[source_pos : source_pos + 1].equals(result_label):
                positions[result_pos] = source_pos
                source_pos += 1
                break
            source_pos += 1
        else:
            return None

    if not source_index.take(positions).equals(result_index):
        return None
    return positions


def _attach_native_state_from_result_index(
    owner,
    result,
    *,
    allow_ordered_subset: bool = False,
) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    source_index = getattr(owner, "index", None)
    result_index = getattr(result, "index", None)
    if source_index is None or result_index is None:
        return

    if source_index.is_unique and not (
        getattr(source_index, "nlevels", 1) != 1
        or getattr(result_index, "nlevels", 1) != 1
    ):
        positions = source_index.get_indexer(result_index)
        if positions.ndim != 1 or positions.shape[0] != len(result):
            return
        if np.any(positions < 0):
            return
        positions = positions.astype(np.int64, copy=False)
    elif allow_ordered_subset:
        positions = _ordered_subset_index_positions(source_index, result_index)
    else:
        return
    if positions is None:
        return
    _attach_native_state_from_row_positions(
        owner,
        result,
        positions,
        unique=bool(result_index.is_unique),
    )


def _sort_values_with_native_row_position_marker(
    owner,
    by,
    *,
    axis,
    ascending,
    kind,
    na_position,
    ignore_index: bool,
    key,
):
    """Sort once with pandas while carrying exact source row positions."""
    if owner._get_axis_number(axis) != 0:
        return None, None, False

    from vibespatial.api._native_state import get_native_state

    if get_native_state(owner) is None:
        return None, None, False

    marker = object()
    work = owner.copy(deep=False)
    _drop_native_state_from_result(work)
    pd.DataFrame.__setitem__(
        work,
        marker,
        np.arange(len(owner), dtype=np.int64),
    )
    try:
        ordered = pd.DataFrame.sort_values(
            work,
            by=by,
            axis=axis,
            ascending=ascending,
            inplace=False,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
            key=key,
        )
        marker_values = ordered[marker].to_numpy(dtype=np.int64, copy=False)
        result = pd.DataFrame.drop(ordered, columns=[marker])
        if len(result.columns) != len(owner.columns):
            return None, None, False
        result.columns = owner.columns.copy()
    except Exception:
        return None, None, False

    _drop_native_state_from_result(result)
    positions = np.asarray(marker_values, dtype=np.int64)
    if positions.ndim != 1 or int(positions.size) != len(result):
        return None, None, False
    if np.any(positions < 0) or np.any(positions >= len(owner)):
        return None, None, False
    unique = positions.size == np.unique(positions).size
    return result, positions, bool(unique)


def _normalize_sort_columns(by) -> tuple[Any, ...] | None:
    if isinstance(by, (str, bytes)):
        return (by,)
    if pd.api.types.is_list_like(by):
        try:
            columns = tuple(by)
        except TypeError:
            return None
        return columns if columns else None
    return None


def _normalize_sort_ascending(ascending, key_count: int) -> tuple[bool, ...] | None:
    if isinstance(ascending, (bool, np.bool_)):
        return (bool(ascending),) * int(key_count)
    if pd.api.types.is_list_like(ascending):
        try:
            values = tuple(bool(value) for value in ascending)
        except TypeError:
            return None
        if len(values) != int(key_count):
            return None
        return values
    return None


_NATIVE_SORT_CUPY_MAX_ROWS = 4096


def _cupy_sort_key(values, *, ascending: bool, na_position: str):
    import cupy as cp

    dtype = np.dtype(getattr(values, "dtype", np.float64))
    d_values = cp.asarray(values)
    if np.issubdtype(dtype, np.bool_):
        key = d_values.astype(cp.uint8, copy=False)
        return key if ascending else cp.bitwise_xor(key, cp.uint8(1))
    if np.issubdtype(dtype, np.unsignedinteger):
        key = d_values
        return key if ascending else cp.bitwise_not(key)
    if np.issubdtype(dtype, np.signedinteger):
        unsigned_dtype = np.dtype(f"uint{dtype.itemsize * 8}")
        sign_mask = np.array(1 << (dtype.itemsize * 8 - 1), dtype=unsigned_dtype)
        key = d_values.view(unsigned_dtype) ^ cp.asarray(sign_mask, dtype=unsigned_dtype)
        return key if ascending else cp.bitwise_not(key)
    if np.issubdtype(dtype, np.floating):
        key = d_values if ascending else -d_values
        nan_mask = cp.isnan(d_values)
        sentinel = cp.inf if na_position == "last" else -cp.inf
        return cp.where(nan_mask, sentinel, key)
    return None


def _cupy_native_sort_positions(
    arrays: dict[Any, Any],
    sort_columns: tuple[Any, ...],
    ascending_values: tuple[bool, ...],
    *,
    na_position: str,
):
    """Return a device row order for small all-valid numeric native sorts."""
    import cupy as cp

    if not sort_columns:
        return None
    first = cp.asarray(arrays[sort_columns[0]])
    row_count = int(first.size)
    if row_count > _NATIVE_SORT_CUPY_MAX_ROWS:
        return None
    order = cp.arange(row_count, dtype=cp.int64)
    for column, ascending in reversed(
        tuple(zip(sort_columns, ascending_values, strict=True))
    ):
        values = cp.asarray(arrays[column])
        if int(values.size) != row_count:
            return None
        key = _cupy_sort_key(values[order], ascending=ascending, na_position=na_position)
        if key is None:
            return None
        local_order = cp.argsort(key, kind="stable")
        order = order[local_order]
    return order


def _native_sort_values_rowset(
    owner,
    by,
    *,
    ascending,
    kind,
    na_position,
    key,
):
    """Return a native sorted rowset for sortable device-backed attributes."""
    if key is not None or na_position not in {"first", "last"}:
        return None
    sort_columns = _normalize_sort_columns(by)
    if sort_columns is None:
        return None
    ascending_values = _normalize_sort_ascending(ascending, len(sort_columns))
    if ascending_values is None:
        return None

    from vibespatial.api._native_result_core import _pylibcudf_numeric_column_view
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(owner)
    if state is None or not _native_state_can_take_device_row_positions(state):
        return None
    attributes = getattr(state, "attributes", None)
    if attributes is None:
        return None
    attribute_columns = tuple(attributes.columns)
    if any(column not in attribute_columns for column in sort_columns):
        return None
    device_backed = bool(getattr(attributes, "is_device_backed", False))
    if device_backed:
        policies = _native_device_column_policies(attributes, sort_columns)
        if any(
            (policy := policies.get(column)) is None or not policy.can_sort
            for column in sort_columns
        ):
            return None
        required_bytes = _native_sort_required_bytes(
            attributes,
            sort_columns,
            int(state.row_count),
        )
        if required_bytes is None:
            return None
        from vibespatial.cuda._runtime import get_cuda_runtime

        admission = get_cuda_runtime().admit_device_memory(
            stage="tabular-stable-sort",
            required_bytes=required_bytes,
            requested_units=int(state.row_count),
        )
        if not admission.admitted:
            return None

    try:
        import cupy as cp
    except ModuleNotFoundError:
        cp = None
    if cp is not None:
        arrays = attributes.numeric_column_arrays(sort_columns)
        if arrays is not None:
            try:
                sorted_positions = _cupy_native_sort_positions(
                    arrays,
                    sort_columns,
                    ascending_values,
                    na_position=na_position,
                )
            except Exception:
                sorted_positions = None
            if sorted_positions is not None:
                sorted_positions = cp.asarray(sorted_positions, dtype=cp.int64)
                if int(sorted_positions.size) == int(state.row_count):
                    return NativeRowSet.from_positions(
                        sorted_positions,
                        source_token=state.lineage_token,
                        source_row_count=state.row_count,
                        ordered=True,
                        unique=True,
                        identity=False,
                    )

    if not device_backed:
        return None

    try:
        import pylibcudf as plc
        import pylibcudf.sorting as sorting
        from pylibcudf.types import NullOrder, Order
    except ModuleNotFoundError:
        return None

    try:
        from vibespatial.cuda._runtime import pylibcudf_current_stream

        stream = pylibcudf_current_stream(attributes.device_table)
        key_columns = _native_pylibcudf_missing_normalized_columns(
            _native_pylibcudf_columns(attributes, sort_columns),
            stream=stream,
        )
        key_table = plc.Table(key_columns)
        column_order = [
            Order.ASCENDING if is_ascending else Order.DESCENDING
            for is_ascending in ascending_values
        ]
        null_precedence = [
            (
                NullOrder.BEFORE
                if (na_position == "first") == is_ascending
                else NullOrder.AFTER
            )
            for is_ascending in ascending_values
        ]
        sort_fn = (
            sorting.stable_sorted_order
            if kind in {"stable", "mergesort"} or len(sort_columns) > 1
            else sorting.sorted_order
        )
        order_column = sort_fn(
            key_table,
            column_order,
            null_precedence,
            stream=stream,
        )
        sorted_positions = _pylibcudf_numeric_column_view(order_column)
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return None
    if sorted_positions is None:
        return None
    sorted_positions = cp.asarray(sorted_positions, dtype=cp.int64)
    if int(sorted_positions.size) != int(state.row_count):
        return None
    return NativeRowSet.from_positions(
        sorted_positions,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
        identity=False,
    )


def _native_topk_required_bytes(row_count: int, arrow_types) -> int | None:
    """Conservatively bound incremental top-k keys, candidates, and workspace."""
    source_width = 0
    expanded_sort_width = 0
    for arrow_type in arrow_types:
        try:
            width = max((int(arrow_type.bit_width) + 7) // 8, 1)
        except (AttributeError, ValueError):
            return None
        source_width += width
        # Every fixed-width key may need a missing discriminator and cleaned
        # value copy. Admission deliberately assumes the expanded form before
        # inspecting null counts or gathering a row-indirected key.
        expanded_sort_width += 1 + width
    # Iterative selection keeps two position sets, two masks, one selected-row
    # order, gathered keys, and top-k workspace. The final stable sort is over
    # at most the requested/output rows, but charge it at full row count so a
    # keep='all' complete-key tie also remains admitted.
    bytes_per_row = 72 + 2 * source_width + 5 * expanded_sort_width
    return int(row_count) * bytes_per_row + (1 << 20)


def _native_topk_public_dtype_admitted(arrow_type) -> bool:
    """Match pandas nlargest/nsmallest fixed-width dtype admission."""
    import pyarrow as pa

    return bool(
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_boolean(arrow_type)
        or pa.types.is_timestamp(arrow_type)
        or pa.types.is_duration(arrow_type)
    )


def _native_topk_public_series_dtype_admitted(dtype) -> bool:
    """Match pandas SelectN's public Series dtype validation."""
    if getattr(dtype, "name", None) == "vibespatial_attribute":
        # Lazy device columns expose a neutral public extension dtype. Their
        # exact logical dtype is validated separately from the Arrow schema.
        return True
    return bool(
        (
            pd.api.types.is_numeric_dtype(dtype)
            and not pd.api.types.is_complex_dtype(dtype)
        )
        or pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
    )


def _native_topk_request_targets_device_attributes(owner, n, columns, keep) -> bool:
    """Whether a valid public top-k request would otherwise leave the GPU path."""
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or keep not in {"first", "last", "all"}
    ):
        return False
    sort_columns = _normalize_sort_columns(columns)
    if sort_columns is None or len(set(sort_columns)) != len(sort_columns):
        return False

    from vibespatial.api._native_state import get_native_state

    state = get_native_state(owner)
    attributes = getattr(state, "attributes", None)
    if attributes is None or not bool(getattr(attributes, "is_device_backed", False)):
        return False
    attribute_columns = tuple(attributes.columns)
    if any(column not in attribute_columns for column in sort_columns):
        return False
    try:
        public_dtypes = tuple(
            getattr(owner[column], "dtype", None) for column in sort_columns
        )
        if any(
            _is_geometry_like_dtype(dtype)
            or not _native_topk_public_series_dtype_admitted(dtype)
            for dtype in public_dtypes
        ):
            return False
        policies = _native_device_column_policies(attributes, sort_columns)
        schema = attributes.arrow_schema_for_columns(sort_columns)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return all(
        (policy := policies.get(column)) is not None
        and policy.can_sort
        and _native_topk_public_dtype_admitted(field.type)
        for column, field in zip(sort_columns, schema, strict=True)
    )


def _native_topk_rowset(owner, n: int, columns, *, largest: bool, keep: str):
    """Return an exact bounded pylibcudf top-k rowset for sortable columns.

    Physical shape: refine only the boundary-equal span once per key, retain
    strict winners as a bounded selected set, then stable-sort only the output
    rows. Primary-key skew therefore remains linear selection/compaction work;
    it cannot turn into a full-table multi-key sort. Complete-key ties preserve
    pandas ``keep='first'``, ``keep='last'``, and ``keep='all'`` semantics.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        return None
    n = int(n)
    if keep not in {"first", "last", "all"}:
        return None
    sort_columns = _normalize_sort_columns(columns)
    if sort_columns is None:
        return None

    from vibespatial.api._native_result_core import _pylibcudf_numeric_column_view
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(owner)
    if state is None or not _native_state_can_take_device_row_positions(state):
        return None
    row_count = int(state.row_count)
    attributes = getattr(state, "attributes", None)
    if attributes is None or not bool(getattr(attributes, "is_device_backed", False)):
        return None
    attribute_columns = tuple(attributes.columns)
    if (
        len(set(sort_columns)) != len(sort_columns)
        or any(column not in attribute_columns for column in sort_columns)
        or any(
            (dtype := getattr(owner[column], "dtype", None)) is None
            or _is_geometry_like_dtype(dtype)
            or not _native_topk_public_series_dtype_admitted(dtype)
            for column in sort_columns
        )
    ):
        return None
    policies = _native_device_column_policies(attributes, sort_columns)
    if any(
        (policy := policies.get(column)) is None
        or not policy.can_sort
        for column in sort_columns
    ):
        return None

    try:
        import cupy as cp
        import pyarrow as pa
        import pylibcudf as plc
        from pylibcudf.types import NullOrder, Order, TypeId

        from vibespatial.cuda._runtime import (
            get_cuda_runtime,
            pylibcudf_column_from_device,
            pylibcudf_current_stream,
        )
        arrow_schema = attributes.arrow_schema_for_columns(sort_columns)
        arrow_types = tuple(field.type for field in arrow_schema)
    except ModuleNotFoundError:
        return None
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return None
    if not all(_native_topk_public_dtype_admitted(value) for value in arrow_types):
        return None
    if n <= 0 or row_count == 0:
        return NativeRowSet.from_positions(
            cp.empty(0, dtype=cp.int64),
            source_token=state.lineage_token,
            source_row_count=row_count,
            ordered=True,
            unique=True,
            identity=row_count == 0,
        )
    required_bytes = _native_topk_required_bytes(row_count, arrow_types)
    if required_bytes is None:
        return None
    admission = get_cuda_runtime().admit_device_memory(
        stage="tabular-topk",
        required_bytes=required_bytes,
        requested_units=row_count,
    )
    if not admission.admitted:
        return None

    try:
        # Resolving a row-indirected or multipart logical key may gather it, so
        # this must remain strictly after memory admission.
        key_columns = _native_pylibcudf_columns(attributes, sort_columns)
        stream = pylibcudf_current_stream(*key_columns)
        direction = Order.DESCENDING if largest else Order.ASCENDING
        sort_key_columns = []
        sort_orders = []
        rank_columns = []
        missing_columns = []
        for column_index, (column, key_column) in enumerate(
            zip(sort_columns, key_columns, strict=True)
        ):
            policy = policies[column]
            arrow_type = arrow_types[column_index]
            floating = pa.types.is_floating(arrow_type)
            nullable = int(policy.null_count) != 0
            normalized_key = key_column
            if floating:
                normalized_key = _native_pylibcudf_missing_normalized_columns(
                    [key_column],
                    stream=stream,
                )[0]
            if floating or nullable:
                missing_mask = plc.unary.is_null(normalized_key, stream=stream)
                neutral_value = False if pa.types.is_boolean(arrow_type) else 0
                neutral_scalar = plc.Scalar.from_arrow(
                    pa.scalar(neutral_value, type=arrow_type),
                    stream=stream,
                )
                clean_key_column = plc.replace.replace_nulls(
                    normalized_key,
                    neutral_scalar,
                    stream=stream,
                )
                sort_key_columns.extend((missing_mask, clean_key_column))
                sort_orders.extend((Order.ASCENDING, direction))
                rank_column = clean_key_column
                missing_columns.append(missing_mask)
            else:
                clean_key_column = normalized_key
                sort_key_columns.append(clean_key_column)
                sort_orders.append(direction)
                rank_column = normalized_key
                missing_columns.append(None)
            rank_columns.append(rank_column)

        position_dtype = cp.int32 if row_count <= np.iinfo(np.int32).max else cp.int64
        active_positions = pylibcudf_column_from_device(
            cp.arange(row_count, dtype=position_dtype)
        )

        def _reverse_position_table(table):
            count = int(table.columns()[0].size())
            reverse_map = pylibcudf_column_from_device(
                cp.arange(count - 1, -1, -1, dtype=position_dtype)
            )
            return plc.copying.gather(
                table,
                reverse_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )

        if n >= row_count:
            full_order = plc.sorting.stable_sorted_order(
                plc.Table(sort_key_columns),
                sort_orders,
                [NullOrder.AFTER] * len(sort_key_columns),
                stream=stream,
            )
            positions = _pylibcudf_numeric_column_view(full_order)
        else:
            selected_position_tables = []
            remaining = n
            for key_index, rank_column in enumerate(rank_columns):
                active_count = int(active_positions.size())
                final_key_keep_last_reversal = bool(
                    keep == "last"
                    and len(rank_columns) > 1
                    and key_index + 1 == len(rank_columns)
                    and active_count > remaining
                )
                active_map = plc.Table([active_positions])
                missing_column = missing_columns[key_index]
                if missing_column is not None:
                    active_missing = plc.copying.gather(
                        plc.Table([missing_column]),
                        active_positions,
                        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                        stream=stream,
                    ).columns()[0]
                    active_valid = plc.unary.unary_operation(
                        active_missing,
                        plc.unary.UnaryOperator.NOT,
                        stream=stream,
                    )
                    valid_positions = plc.stream_compaction.apply_boolean_mask(
                        active_map,
                        active_valid,
                        stream=stream,
                    ).columns()[0]
                    valid_count = int(valid_positions.size())
                    if valid_count < remaining:
                        if valid_count:
                            valid_table = plc.Table([valid_positions])
                            selected_position_tables.append(
                                _reverse_position_table(valid_table)
                                if final_key_keep_last_reversal
                                else valid_table
                            )
                            remaining -= valid_count
                        active_positions = plc.stream_compaction.apply_boolean_mask(
                            active_map,
                            active_missing,
                            stream=stream,
                        ).columns()[0]
                        if key_index + 1 != len(rank_columns):
                            continue
                        boundary_count = int(active_positions.size())
                        if keep == "all":
                            selected_position_tables.append(
                                plc.Table([active_positions])
                            )
                        else:
                            start = (
                                0
                                if keep == "first"
                                else boundary_count - remaining
                            )
                            chosen = plc.copying.slice(
                                plc.Table([active_positions]),
                                [start, start + remaining],
                                stream=stream,
                            )[0]
                            selected_position_tables.append(
                                _reverse_position_table(chosen)
                                if final_key_keep_last_reversal
                                else chosen
                            )
                        break
                    active_positions = valid_positions
                    active_map = plc.Table([active_positions])
                active_rank = plc.copying.gather(
                    plc.Table([rank_column]),
                    active_positions,
                    plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                    stream=stream,
                ).columns()[0]
                boundary_values = plc.sorting.top_k(
                    active_rank,
                    remaining,
                    direction,
                    stream=stream,
                )
                boundary = plc.reduce.reduce(
                    boundary_values,
                    (
                        plc.aggregation.min()
                        if largest
                        else plc.aggregation.max()
                    ),
                    boundary_values.type(),
                    stream=stream,
                )
                better_operator = (
                    plc.binaryop.BinaryOperator.GREATER
                    if largest
                    else plc.binaryop.BinaryOperator.LESS
                )
                better_mask = plc.binaryop.binary_operation(
                    active_rank,
                    boundary,
                    better_operator,
                    plc.DataType(TypeId.BOOL8),
                    stream=stream,
                )
                equal_mask = plc.binaryop.binary_operation(
                    active_rank,
                    boundary,
                    plc.binaryop.BinaryOperator.EQUAL,
                    plc.DataType(TypeId.BOOL8),
                    stream=stream,
                )
                strict_winners = plc.stream_compaction.apply_boolean_mask(
                    active_map,
                    better_mask,
                    stream=stream,
                )
                strict_count = int(strict_winners.columns()[0].size())
                if strict_count:
                    selected_position_tables.append(
                        _reverse_position_table(strict_winners)
                        if final_key_keep_last_reversal
                        else strict_winners
                    )
                    remaining -= strict_count
                boundary_positions = plc.stream_compaction.apply_boolean_mask(
                    active_map,
                    equal_mask,
                    stream=stream,
                ).columns()[0]

                if key_index + 1 != len(rank_columns):
                    active_positions = boundary_positions
                    continue

                boundary_count = int(boundary_positions.size())
                if keep == "all":
                    selected_position_tables.append(plc.Table([boundary_positions]))
                else:
                    start = 0 if keep == "first" else boundary_count - remaining
                    if start < 0 or start + remaining > boundary_count:
                        raise RuntimeError(
                            "top-k boundary refinement produced an invalid slice: "
                            f"boundary_count={boundary_count}, remaining={remaining}, "
                            f"keep={keep!r}"
                        )
                    chosen = plc.copying.slice(
                        plc.Table([boundary_positions]),
                        [start, start + remaining],
                        stream=stream,
                    )[0]
                    selected_position_tables.append(
                        _reverse_position_table(chosen)
                        if final_key_keep_last_reversal
                        else chosen
                    )

            selected_positions_table = plc.concatenate.concatenate(
                selected_position_tables,
                stream=stream,
            )
            selected_positions_column = selected_positions_table.columns()[0]
            selected_keys = plc.copying.gather(
                plc.Table(sort_key_columns),
                selected_positions_column,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            )
            position_tiebreak_order = None
            if keep in {"first", "all"}:
                position_tiebreak_order = Order.ASCENDING
            elif len(rank_columns) == 1:
                position_tiebreak_order = Order.DESCENDING
            final_sort_columns = list(selected_keys.columns())
            final_sort_orders = list(sort_orders)
            if position_tiebreak_order is not None:
                final_sort_columns.append(selected_positions_column)
                final_sort_orders.append(position_tiebreak_order)
            final_sort_table = plc.Table(final_sort_columns)
            final_order = plc.sorting.stable_sorted_order(
                final_sort_table,
                final_sort_orders,
                [NullOrder.AFTER] * len(final_sort_columns),
                stream=stream,
            )
            ordered_positions = plc.copying.gather(
                plc.Table([selected_positions_column]),
                final_order,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                stream=stream,
            ).columns()[0]
            positions = _pylibcudf_numeric_column_view(ordered_positions)
    except Exception as exc:
        raise RuntimeError("admitted pylibcudf top-k execution failed") from exc
    expected_min = min(max(n, 0), row_count)
    if positions is None or int(positions.size) < expected_min:
        return None
    return NativeRowSet.from_positions(
        cp.asarray(positions, dtype=cp.int64),
        source_token=state.lineage_token,
        source_row_count=row_count,
        ordered=True,
        unique=True,
        identity=False,
    )


def _sort_index_with_native_row_position_marker(
    owner,
    *,
    axis,
    level,
    ascending,
    kind,
    na_position,
    sort_remaining,
    ignore_index: bool,
    key,
):
    """Sort row index with pandas while carrying exact source row positions."""
    if owner._get_axis_number(axis) != 0:
        return None, None, False

    from vibespatial.api._native_state import get_native_state

    if get_native_state(owner) is None:
        return None, None, False

    marker = object()
    work = owner.copy(deep=False)
    _drop_native_state_from_result(work)
    pd.DataFrame.__setitem__(
        work,
        marker,
        np.arange(len(owner), dtype=np.int64),
    )
    try:
        ordered = pd.DataFrame.sort_index(
            work,
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=False,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
            ignore_index=ignore_index,
            key=key,
        )
        marker_values = ordered[marker].to_numpy(dtype=np.int64, copy=False)
        result = pd.DataFrame.drop(ordered, columns=[marker])
        if len(result.columns) != len(owner.columns):
            return None, None, False
        result.columns = owner.columns.copy()
    except Exception:
        return None, None, False

    _drop_native_state_from_result(result)
    positions = np.asarray(marker_values, dtype=np.int64)
    if positions.ndim != 1 or int(positions.size) != len(result):
        return None, None, False
    if np.any(positions < 0) or np.any(positions >= len(owner)):
        return None, None, False
    unique = positions.size == np.unique(positions).size
    return result, positions, bool(unique)


def _attach_native_state_after_column_position_take(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner) or not result.index.equals(owner.index):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    projected = state.project_columns(tuple(result.columns))
    if projected is not None:
        attach_native_state(result, projected)


def _attach_native_state_after_projection_from_state(result, source_state) -> None:
    if source_state is None or not _native_result_accepts_frame_state(result):
        return
    projected = source_state.project_columns(tuple(result.columns))
    if projected is None:
        return

    from vibespatial.api._native_state import attach_native_state

    attach_native_state(result, projected)


def _copy_geometry_series_preserving_owned_backing(
    series: Series,
    *,
    index,
    deep: bool,
) -> GeoSeries:
    """Return an independent geometry Series wrapper without cloning owned buffers."""
    values = series.values
    cached_owned = getattr(values, "cached_owned", None)
    owned = cached_owned() if callable(cached_owned) else getattr(values, "_owned", None)
    composition = getattr(values, "native_composition", None)
    if owned is None and composition is None:
        return series.copy(deep=deep)

    crs = getattr(series, "crs", getattr(values, "crs", None))
    if values.__class__.__name__ == "DeviceGeometryArray":
        from vibespatial.geometry.device_array import DeviceGeometryArray

        copied_values = (
            DeviceGeometryArray._from_owned(
                owned,
                crs=crs,
                provenance=getattr(values, "_provenance", None),
            )
            if owned is not None
            else DeviceGeometryArray._from_composition(
                composition,
                crs=crs,
                provenance=getattr(values, "_provenance", None),
            )
        )
        shapely_cache = getattr(values, "_shapely_cache", None)
        if shapely_cache is not None:
            copied_values._shapely_cache = shapely_cache.copy() if deep else shapely_cache
    else:
        copied_values = GeometryArray.from_owned(owned, crs=crs)
        copied_values._provenance = getattr(values, "_provenance", None)
        copied_values._readonly = False

    result = pd.Series(
        copied_values,
        index=index,
        name=series.name,
        copy=False,
    )
    result.__class__ = GeoSeries
    if getattr(result, "crs", None) is None and crs is not None:
        result.crs = crs
    return result


def _attach_native_state_after_set_geometry(owner, result, source_state, geometry_name) -> None:
    if source_state is None or not _native_result_accepts_frame_state(result):
        return
    if len(result) != source_state.row_count:
        return
    if tuple(result.columns) != source_state.column_order:
        return
    if not result.index.equals(owner.index):
        return
    if getattr(result, "_geometry_column_name", None) != geometry_name:
        return

    from vibespatial.api._native_state import attach_native_state

    updated = source_state.with_active_geometry(geometry_name, crs=getattr(result, "crs", None))
    if updated is not None:
        attach_native_state(result, updated)


def _attach_native_state_after_column_rename(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner) or not result.index.equals(owner.index):
        return
    if len(owner.columns) != len(result.columns):
        return
    if not owner.columns.is_unique or not result.columns.is_unique:
        return

    geometry_name = getattr(result, "_geometry_column_name", None)
    if geometry_name not in result.columns:
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    mapping = {
        old_name: new_name
        for old_name, new_name in zip(
            tuple(owner.columns),
            tuple(result.columns),
            strict=True,
        )
        if old_name != new_name
    }
    renamed = state.rename_columns(mapping)
    if (
        renamed is not None
        and renamed.column_order == tuple(result.columns)
        and renamed.geometry_name == geometry_name
    ):
        attach_native_state(result, renamed)


def _attach_native_state_after_rename(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner):
        return
    if len(owner.columns) != len(result.columns):
        return
    if not owner.columns.is_unique or not result.columns.is_unique:
        return
    if _index_has_geometry_dtype(result.index):
        return

    geometry_name = getattr(result, "_geometry_column_name", None)
    if geometry_name not in result.columns:
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    mapping = {
        old_name: new_name
        for old_name, new_name in zip(
            tuple(owner.columns),
            tuple(result.columns),
            strict=True,
        )
        if old_name != new_name
    }
    renamed = state.rename_columns(mapping)
    if (
        renamed is not None
        and renamed.column_order == tuple(result.columns)
        and renamed.geometry_name == geometry_name
    ):
        attach_native_state(result, renamed.with_index(result.index))


def _attach_native_state_after_reset_index_drop(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner) or tuple(result.columns) != tuple(owner.columns):
        return
    if not isinstance(result.index, pd.RangeIndex):
        return
    expected_index = pd.RangeIndex(len(result))
    if not result.index.equals(expected_index):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is not None:
        attach_native_state(result, state.with_index(expected_index))


def _attach_native_state_after_reset_index_insert(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner):
        return
    index_levels = getattr(owner.index, "nlevels", 1)
    if len(result.columns) != len(owner.columns) + index_levels:
        return
    if tuple(result.columns[index_levels:]) != tuple(owner.columns):
        return
    if not isinstance(result.index, pd.RangeIndex):
        return
    expected_index = pd.RangeIndex(len(result))
    if not result.index.equals(expected_index):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    index_columns = tuple(result.columns[:index_levels])
    if any(_is_geometry_like_dtype(getattr(result[column], "dtype", None)) for column in index_columns):
        return
    updated = state.with_index(expected_index).assign_attributes(
        {
            column: result[column].to_numpy(copy=False)
            for column in index_columns
        },
        column_order=tuple(result.columns),
    )
    if updated is not None:
        attach_native_state(result, updated)


def _attach_native_state_after_reset_index_partial(owner, result, *, drop: bool) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner):
        return
    if _index_has_geometry_dtype(result.index):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    base = state.with_index(result.index)

    if drop:
        if tuple(result.columns) != tuple(owner.columns):
            return
        projected = base.project_columns(tuple(result.columns))
        if projected is not None:
            attach_native_state(result, projected)
        return

    inserted_count = len(result.columns) - len(owner.columns)
    if inserted_count <= 0:
        return
    if tuple(result.columns[inserted_count:]) != tuple(owner.columns):
        return
    inserted_columns = tuple(result.columns[:inserted_count])
    if any(_is_geometry_like_dtype(getattr(result[column], "dtype", None)) for column in inserted_columns):
        return
    updated = base.assign_attributes(
        {
            column: result[column].to_numpy(copy=False)
            for column in inserted_columns
        },
        column_order=tuple(result.columns),
    )
    if updated is not None:
        attach_native_state(result, updated)


def _index_has_geometry_dtype(index) -> bool:
    if isinstance(index, pd.MultiIndex):
        return any(_is_geometry_like_dtype(level.dtype) for level in index.levels)
    return _is_geometry_like_dtype(getattr(index, "dtype", None))


def _attach_native_state_after_set_index(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner):
        return
    if _index_has_geometry_dtype(result.index):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = get_native_state(owner)
    if state is None:
        return
    projected = state.with_index(result.index).project_columns(tuple(result.columns))
    if projected is not None:
        attach_native_state(result, projected)


def _attach_native_state_after_axis_relabel(owner, result) -> bool:
    if not _native_result_accepts_frame_state(result):
        return False
    if len(result) != len(owner):
        return False

    from vibespatial.api._native_state import (
        attach_native_state,
        drop_native_state,
        get_native_state,
    )

    drop_native_state(result)

    if tuple(result.columns) == tuple(owner.columns):
        if _index_has_geometry_dtype(result.index):
            return False

        state = get_native_state(owner)
        if state is None:
            return False
        attach_native_state(result, state.with_index(result.index))
        return True
    elif result.index.equals(owner.index) and len(result.columns) == len(owner.columns):
        _attach_native_state_after_column_rename(owner, result)
        return get_native_state(result) is not None
    return False


def _attach_native_state_after_reindex(owner, result) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if result.index.equals(owner.index):
        _attach_native_state_after_column_position_take(owner, result)
    else:
        _attach_native_state_from_result_index(owner, result)


def _dropna_axis_number(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    axis = kwargs.get("axis", args[0] if args else 0)
    if axis in (0, "index", "rows"):
        return 0
    if axis in (1, "columns"):
        return 1
    return None


def _dropna_ignore_index(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if "ignore_index" in kwargs:
        return bool(kwargs["ignore_index"])
    if len(args) >= 6:
        return bool(args[5])
    return False


def _mixed_geometry_parts_native_tabular_result(
    owned,
    *,
    crs,
    geometry_name: str,
    source_tokens: tuple[str, ...],
    include_point: bool,
    include_lineal: bool,
    include_polygonal: bool,
):
    from vibespatial.api._native_result_core import (
        NativeTabularSelection,
    )
    from vibespatial.api._native_results import _concat_native_tabular_results

    part_results = []
    if include_point:
        from vibespatial.constructive.binary_constructive import (
            point_parts_native_tabular_result as part_builder,
        )

        part_results.append(
            part_builder(
                owned,
                crs=crs,
                geometry_name=geometry_name,
                source_tokens=source_tokens,
            )
        )
    if include_lineal:
        from vibespatial.constructive.binary_constructive import (
            lineal_parts_native_tabular_result as part_builder,
        )

        part_results.append(
            part_builder(
                owned,
                crs=crs,
                geometry_name=geometry_name,
                source_tokens=source_tokens,
            )
        )
    if include_polygonal:
        from vibespatial.constructive.binary_constructive import (
            polygonal_parts_native_tabular_result as part_builder,
        )

        part_results.append(
            part_builder(
                owned,
                crs=crs,
                geometry_name=geometry_name,
                source_tokens=source_tokens,
            )
        )

    if not part_results:
        return None
    combined = _concat_native_tabular_results(
        part_results,
        geometry_name=geometry_name,
        crs=crs,
        ignore_index=True,
    )
    if not isinstance(combined, NativeTabularSelection):
        return None
    provenance = combined.capacity_result.provenance
    source_rows = getattr(provenance, "source_rows", None)
    if provenance is None or source_rows is None:
        return None

    from dataclasses import replace

    capacity_result = replace(
        combined.capacity_result,
        provenance=replace(provenance, operation="mixed_geometry_parts"),
    )
    return replace(
        combined,
        capacity_result=capacity_result,
    ).sort_selected_by_int64(source_rows)


def _shapely_object_values(series) -> np.ndarray:
    values = getattr(series, "values", series)
    data = getattr(values, "_data", None)
    if data is not None:
        return np.asarray(data, dtype=object)
    return np.asarray(values, dtype=object)


def _geometrycollection_parts_native_tabular_result(
    owner,
    result,
    *,
    column,
    exploded_source_rows,
):
    """Lower public GeometryCollection explode parts to a private native frame.

    Public ``explode`` over a GeometryCollection has already crossed a Shapely
    compatibility boundary.  This bridge is a narrow native ingress for the
    exploded single-family parts so immediate geometry consumers can stay on
    device instead of treating row-level mixed parts as permanently host-only.
    """
    from vibespatial.runtime._runtime import has_gpu_runtime

    if not has_gpu_runtime():
        return None
    geometry_name = getattr(result, "_geometry_column_name", None)
    if column != geometry_name:
        return None
    if exploded_source_rows is None or len(result) == 0:
        return None
    if len(exploded_source_rows) != len(result):
        return None
    if any(
        _is_geometry_like_dtype(getattr(result[name], "dtype", None))
        for name in result.columns
        if name != geometry_name
    ):
        return None

    import shapely

    source_values = _shapely_object_values(owner[column])
    source_type_ids = shapely.get_type_id(source_values)
    if not bool(np.any(source_type_ids == 7)):
        return None

    part_values = _shapely_object_values(result[geometry_name])
    part_type_ids = shapely.get_type_id(part_values)
    if bool(np.any(part_type_ids == 7)):
        return None

    try:
        import cupy as cp

        from vibespatial.api._native_metadata import NativeGeometryMetadata
        from vibespatial.api._native_result_core import (
            GeometryNativeResult,
            NativeAttributeTable,
            NativeGeometryProvenance,
            NativeTabularResult,
        )
        from vibespatial.geometry.owned import from_shapely_geometries

        owned = from_shapely_geometries(
            part_values.tolist(),
            residency=Residency.DEVICE,
        )
        state = owned._ensure_device_state()
        source_rows = cp.asarray(exploded_source_rows, dtype=cp.int64)
    except Exception:
        return None

    attr_columns = tuple(name for name in result.columns if name != geometry_name)
    attributes = NativeAttributeTable(
        dataframe=pd.DataFrame(
            result.loc[:, list(attr_columns)].to_numpy(copy=False),
            columns=attr_columns,
            index=pd.RangeIndex(len(result)),
        )
    )
    return NativeTabularResult(
        attributes=attributes,
        geometry=GeometryNativeResult.from_owned(
            owned,
            crs=getattr(owner, "crs", None),
        ),
        geometry_name=geometry_name,
        column_order=tuple(result.columns),
        attrs=dict(getattr(result, "attrs", {}) or {}),
        provenance=NativeGeometryProvenance(
            operation="geometrycollection_parts",
            row_count=len(result),
            source_rows=source_rows,
            part_family_tags=state.tags,
            source_tokens=(f"public_geometrycollection_explode:{id(owner)}",),
        ),
        geometry_metadata=NativeGeometryMetadata.from_cached_owned(owned),
    )


def _attach_native_state_after_geometry_explode(
    owner,
    result,
    *,
    column,
    exploded_source_rows=None,
) -> None:
    if not _native_result_accepts_frame_state(result):
        return

    from vibespatial.api._native_result_core import (
        NativeTabularResult,
        NativeTabularSelection,
    )
    from vibespatial.api._native_state import (
        NativeFrameState,
        attach_native_state,
        get_native_state,
    )
    from vibespatial.geometry.buffers import GeometryFamily

    state = get_native_state(owner)
    parts = None
    if state is not None and column == state.geometry_name and not state.secondary_geometry:
        owned = getattr(state.geometry, "owned", None)
        if owned is not None and getattr(owned, "residency", None) is Residency.DEVICE:
            has_point_rows = owned.family_has_rows(
                GeometryFamily.POINT,
            ) or owned.family_has_rows(GeometryFamily.MULTIPOINT)
            has_lineal_rows = owned.family_has_rows(
                GeometryFamily.LINESTRING,
            ) or owned.family_has_rows(GeometryFamily.MULTILINESTRING)
            has_polygonal_rows = owned.family_has_rows(
                GeometryFamily.POLYGON,
            ) or owned.family_has_rows(GeometryFamily.MULTIPOLYGON)
            family_class_count = sum((has_point_rows, has_lineal_rows, has_polygonal_rows))
            if family_class_count == 1:
                if has_point_rows:
                    from vibespatial.constructive.binary_constructive import (
                        point_parts_native_tabular_result as part_builder,
                    )
                elif has_lineal_rows:
                    from vibespatial.constructive.binary_constructive import (
                        lineal_parts_native_tabular_result as part_builder,
                    )
                else:
                    from vibespatial.constructive.binary_constructive import (
                        polygonal_parts_native_tabular_result as part_builder,
                    )

                parts = part_builder(
                    owned,
                    crs=getattr(owner, "crs", None),
                    geometry_name=state.geometry_name,
                    source_tokens=(state.lineage_token,),
                )
            elif family_class_count > 1:
                parts = _mixed_geometry_parts_native_tabular_result(
                    owned,
                    crs=getattr(owner, "crs", None),
                    geometry_name=state.geometry_name,
                    source_tokens=(state.lineage_token,),
                    include_point=has_point_rows,
                    include_lineal=has_lineal_rows,
                    include_polygonal=has_polygonal_rows,
                )
        if parts is None:
            return
    else:
        parts = _geometrycollection_parts_native_tabular_result(
            owner,
            result,
            column=column,
            exploded_source_rows=exploded_source_rows,
        )
        if parts is None:
            return

    if isinstance(parts, NativeTabularSelection):
        parts = parts.physicalize_known_count(len(result))
    if parts.geometry.row_count != len(result):
        return
    source_rows = getattr(parts.provenance, "source_rows", None)
    if source_rows is None:
        return

    attributes = (
        state.attributes.take(source_rows, preserve_index=False)
        if state is not None
        else parts.attributes
    )
    native_result = NativeTabularResult(
        attributes=attributes,
        geometry=parts.geometry,
        geometry_name=parts.geometry_name,
        column_order=tuple(result.columns),
        attrs=dict(getattr(result, "attrs", {}) or {}),
        provenance=parts.provenance,
        geometry_metadata=parts.geometry_metadata,
    )
    exploded_state = NativeFrameState.from_native_tabular_result(native_result).with_index(
        result.index,
    )
    attach_native_state(result, exploded_state)


def _attach_native_state_after_assign(
    owner,
    result,
    assigned_columns,
    *,
    source_state=None,
    native_expressions=None,
) -> None:
    if not _native_result_accepts_frame_state(result):
        return
    if len(result) != len(owner) or not result.index.equals(owner.index):
        return

    geometry_name = getattr(result, "_geometry_column_name", None)
    if geometry_name not in result.columns:
        return

    assigned = tuple(assigned_columns)
    if any(column not in result.columns for column in assigned):
        return
    if any(column == geometry_name for column in assigned):
        return
    native_expression_columns = dict(native_expressions or {})
    if any(column == geometry_name for column in native_expression_columns):
        return

    regular_assigned = tuple(
        column for column in assigned if column not in native_expression_columns
    )
    if any(
        _is_geometry_like_dtype(getattr(result[column], "dtype", None))
        for column in regular_assigned
    ):
        return

    from vibespatial.api._native_state import attach_native_state, get_native_state

    state = source_state if source_state is not None else get_native_state(owner)
    if state is None:
        return

    assigned_state = state
    if regular_assigned:
        regular_column_order = tuple(
            column for column in result.columns if column not in native_expression_columns
        )
        assigned_state = assigned_state.assign_attributes(
            {column: result[column] for column in regular_assigned},
            column_order=regular_column_order,
        )
        if assigned_state is None:
            return

    if native_expression_columns:
        assigned_state = assigned_state.assign_expression_columns(
            native_expression_columns,
            column_order=tuple(result.columns),
        )

    if assigned_state is not None:
        attach_native_state(result, assigned_state)


def _attach_native_state_after_geometry_column_assign(
    owner,
    result,
    assigned_column,
    *,
    source_state,
) -> bool:
    """Preserve frame state when active geometry is replaced row-for-row."""
    if source_state is None or not _native_result_accepts_frame_state(result):
        return False
    geometry_name = getattr(result, "_geometry_column_name", None)
    if assigned_column != geometry_name:
        return False
    if geometry_name != source_state.geometry_name:
        return False
    if len(result) != source_state.row_count or not result.index.equals(owner.index):
        return False
    if tuple(result.columns) != source_state.column_order:
        return False

    from vibespatial.api._native_result_core import GeometryNativeResult
    from vibespatial.api._native_state import attach_native_state

    geometry = GeometryNativeResult.from_geoseries(result.geometry)
    if geometry.owned is None and geometry.composition is None:
        return False

    attach_native_state(result, source_state.with_geometry_result(geometry))
    return True


def _attach_native_state_after_attribute_only_result(owner, result) -> bool:
    """Attach state after exact pandas operations that only changed attributes."""
    if not _native_result_accepts_frame_state(result):
        return False
    if len(result) != len(owner) or not result.index.equals(owner.index):
        return False
    if tuple(result.columns) != tuple(owner.columns):
        return False

    geometry_name = getattr(owner, "_geometry_column_name", None)
    if geometry_name is None or getattr(result, "_geometry_column_name", None) != geometry_name:
        return False
    if geometry_name not in result.columns:
        return False
    if not _is_geometry_like_dtype(getattr(result[geometry_name], "dtype", None)):
        return False

    source_values = owner[geometry_name].values
    result_values = result[geometry_name].values
    source_cached_owned = getattr(source_values, "cached_owned", None)
    result_cached_owned = getattr(result_values, "cached_owned", None)
    source_owned = (
        source_cached_owned()
        if callable(source_cached_owned)
        else getattr(source_values, "_owned", None)
    )
    result_owned = (
        result_cached_owned()
        if callable(result_cached_owned)
        else getattr(result_values, "_owned", None)
    )
    source_composition = getattr(source_values, "native_composition", None)
    result_composition = getattr(result_values, "native_composition", None)
    if any(
        carrier is not None
        for carrier in (
            source_owned,
            result_owned,
            source_composition,
            result_composition,
        )
    ):
        if (
            result_owned is not source_owned
            or result_composition is not source_composition
        ):
            return False
    else:
        try:
            if not bool(result_values.equals(source_values)):
                return False
        except Exception:
            return False

    assigned_columns = tuple(column for column in result.columns if column != geometry_name)
    from vibespatial.api._native_state import attach_native_state, get_native_state

    source_state = get_native_state(owner)
    if source_state is None:
        return False
    if source_state.geometry_name != geometry_name:
        return False
    assigned_state = source_state.assign_attributes(
        {column: result[column] for column in assigned_columns},
        column_order=tuple(result.columns),
    )
    if assigned_state is None:
        return False
    attach_native_state(result, assigned_state)
    return True


def _native_expression_assignment_public_series(
    name,
    expression,
    *,
    index: pd.Index,
    surface: str,
) -> Series:
    """Build a public column shell for a private expression.

    Numeric expressions stay lazy so assignment remains a native row-aligned
    expression transition. Public array materialization happens only when a
    consumer asks for NumPy/pandas values.
    """
    from vibespatial.api._native_expression import NativeExpression
    from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
    from vibespatial.runtime.materialization import (
        MaterializationBoundary,
        record_materialization_event,
    )

    if not isinstance(expression, NativeExpression):
        raise TypeError("expected a NativeExpression")
    if len(expression) != len(index):
        raise ValueError("NativeExpression assignment length must match GeoDataFrame")

    values = expression.values
    dtype = np.dtype(getattr(values, "dtype", np.float64))
    if np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_):
        return Series(
            NativeNumericExpressionArray(
                expression,
                export_surface=surface,
                export_operation="native_expression_to_public_column",
            ),
            index=index,
            name=name,
        )
    if expression.is_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        device_values = cp.asarray(values)
        record_materialization_event(
            surface=surface,
            boundary=MaterializationBoundary.USER_EXPORT,
            operation="native_expression_to_public_column",
            reason="NativeExpression assigned to a public GeoDataFrame column",
            detail=(
                f"column={name!r}, rows={len(index)}, "
                f"bytes={int(device_values.nbytes)}"
            ),
            d2h_transfer=True,
        )
        public_values = get_cuda_runtime().copy_device_to_host(
            device_values,
            reason=f"{surface}::native_expression_to_public_column",
        )
    else:
        public_values = np.asarray(values)
    return Series(public_values, index=index, name=name)


def _prepare_native_expression_assignments(
    values_by_name: dict[Any, Any],
    *,
    source_state,
    index: pd.Index,
    surface: str,
) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Replace admitted NativeExpression values with public Series for pandas."""
    from vibespatial.api._native_expression import NativeExpression

    public_values = dict(values_by_name)
    expressions: dict[Any, NativeExpression] = {}
    for name, value in values_by_name.items():
        expression = (
            value
            if isinstance(value, NativeExpression)
            else _native_expression_from_public_series(value)
        )
        if not isinstance(expression, NativeExpression):
            continue
        can_preserve = (
            source_state is not None
            and name != source_state.geometry_name
            and (
                expression.source_token is None
                or expression.source_token == source_state.lineage_token
            )
            and (
                expression.source_row_count is None
                or int(expression.source_row_count) == source_state.row_count
            )
            and len(index) == source_state.row_count
        )
        if not can_preserve:
            if isinstance(value, Series):
                public_values[name] = Series(
                    value.array.to_numpy(copy=False),
                    index=value.index,
                    name=value.name,
                    copy=False,
                )
            else:
                public = _native_expression_assignment_public_series(
                    name,
                    expression,
                    index=index,
                    surface=surface,
                )
                public_values[name] = Series(
                    public.array.to_numpy(copy=False),
                    index=index,
                    name=name,
                    copy=False,
                )
            continue
        if isinstance(value, NativeExpression):
            public_values[name] = _native_expression_assignment_public_series(
                name,
                expression,
                index=index,
                surface=surface,
            )
        elif (
            isinstance(value, Series)
            and len(value) == len(index)
            and value.index.equals(index)
        ):
            public_values[name] = value
        expressions[name] = expression
    return public_values, expressions


def _native_setitem_assigned_columns(key, columns) -> tuple[Any, ...] | None:
    try:
        if key in columns:
            return (key,)
    except TypeError:
        pass
    if not pd.api.types.is_list_like(key) or isinstance(key, (str, bytes, slice)):
        return None
    try:
        labels = tuple(key)
    except TypeError:
        return None
    if not labels or len(set(labels)) != len(labels):
        return None
    if all(isinstance(label, (bool, np.bool_)) for label in labels):
        return None
    try:
        if any(label not in columns for label in labels):
            return None
    except TypeError:
        return None
    return labels


def _native_state_can_take_device_row_positions(state) -> bool:
    """Return True when a device rowset will not force a hidden host take."""
    if getattr(getattr(state, "index_plan", None), "kind", None) not in {
        "range",
        "device-labels",
        "host-labels",
        "host-labels-take",
    }:
        return False
    geometry = getattr(state, "geometry", None)
    if (
        getattr(geometry, "owned", None) is None
        and getattr(geometry, "composition", None) is None
    ):
        return False
    for column in getattr(state, "secondary_geometry", ()):
        column_geometry = getattr(column, "geometry", None)
        if (
            getattr(column_geometry, "owned", None) is None
            and getattr(column_geometry, "composition", None) is None
        ):
            return False

    attributes = getattr(state, "attributes", None)
    return any(
        getattr(attributes, attr, None) is not None
        for attr in (
            "device_table",
            "arrow_table",
            "dataframe",
            "loader",
            "parts",
        )
    )


def _drop_native_state_from_result(result):
    """Clear sidecars from broad pandas results outside sanctioned transitions."""
    if result is None:
        return None
    from vibespatial.api._native_state import drop_native_state

    drop_native_state(result)
    return result


_NATIVE_EXPRESSION_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
}
_NATIVE_QUERY_COMPARISONS = {
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
_NATIVE_REVERSED_COMPARISON = {
    ">": "<",
    ">=": "<=",
    "<": ">",
    "<=": ">=",
    "==": "==",
    "!=": "!=",
}
_PANDAS_EXPRESSION_FUNCTIONS = frozenset(
    {
        "abs",
        "arccos",
        "arccosh",
        "arcsin",
        "arcsinh",
        "arctan",
        "arctan2",
        "arctanh",
        "ceil",
        "cos",
        "cosh",
        "exp",
        "expm1",
        "floor",
        "log",
        "log10",
        "log1p",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
_PANDAS_QUERY_FALLBACK_ACTIVE = ContextVar(
    "vibespatial_pandas_query_fallback_active",
    default=False,
)
_PANDAS_JOIN_FALLBACK_ACTIVE = ContextVar(
    "vibespatial_pandas_join_fallback_active",
    default=False,
)


def _native_device_expression_state(owner):
    """Return the device-backed frame state admitted by query/eval."""
    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(owner)
    if state is None:
        return None
    attributes = NativeAttributeTable.from_value(state.attributes)
    if not attributes.is_device_backed:
        return None
    return state


def _native_expression_column(state, name, *, comparison: bool):
    """Resolve one exact all-valid numeric device column and its Arrow type."""
    if not isinstance(name, str) or name == state.geometry_name:
        return None
    attributes = state.attributes
    policies = _native_device_column_policies(attributes, (name,))
    policy = policies.get(name)
    if (
        policy is None
        or not policy.can_compute_numeric
        or int(policy.null_count) != 0
    ):
        return None
    try:
        import pyarrow as pa

        arrow_type = attributes.arrow_schema_for_columns((name,)).field(0).type
    except (AttributeError, ImportError, KeyError, TypeError, ValueError):
        return None
    if comparison and not (
        pa.types.is_integer(arrow_type) or pa.types.is_boolean(arrow_type)
    ):
        # NativeExpression comparisons deliberately treat nonfinite floats as
        # missing.  Until query has a validity-aware float contract, decline
        # rather than changing pandas NaN/inf semantics.
        return None
    try:
        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_result_core import _pylibcudf_numeric_column_view
        from vibespatial.cuda._runtime import pylibcudf_current_stream

        columns = _native_pylibcudf_columns(attributes, (name,))
        pylibcudf_current_stream(*columns)
        values = _pylibcudf_numeric_column_view(columns[0])
    except (AttributeError, ImportError, KeyError, TypeError, ValueError):
        return None
    expression = NativeExpression(
        operation=f"attribute.{name}",
        values=values,
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        dtype=str(getattr(values, "dtype", "")) or None,
        precision="source",
        readiness=state.readiness,
    )
    return expression, arrow_type


def _native_expression_scalar(node):
    if not isinstance(node, ast.Constant):
        return None
    value = node.value
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    return None


def _native_eval_expression_node(node, state):
    """Lower the restricted fixed-width eval grammar to NativeExpression."""
    if isinstance(node, ast.Name):
        resolved = _native_expression_column(state, node.id, comparison=False)
        if resolved is None:
            return None
        import pyarrow as pa

        return resolved[0] if pa.types.is_int64(resolved[1]) else None
    scalar = _native_expression_scalar(node)
    if scalar is not None:
        return (
            scalar
            if isinstance(scalar, int)
            and not isinstance(scalar, bool)
            and np.iinfo(np.int64).min <= scalar <= np.iinfo(np.int64).max
            else None
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _native_eval_expression_node(node.operand, state)
        if operand is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(operand, (bool, int, float)):
            return -operand
        return operand.binary_arithmetic("*", -1)
    if isinstance(node, ast.BinOp):
        symbol = _NATIVE_EXPRESSION_BINOPS.get(type(node.op))
        if symbol not in {"+", "-", "*"}:
            return None
        left = _native_eval_expression_node(node.left, state)
        right = _native_eval_expression_node(node.right, state)
        if left is None or right is None:
            return None
        from vibespatial.api._native_expression import NativeExpression

        if isinstance(left, NativeExpression):
            return left.binary_arithmetic(symbol, right)
        if isinstance(right, NativeExpression):
            return right.binary_arithmetic(symbol, left, reverse=True)
        return None
    return None


def _native_query_scalar_fits_arrow(value, arrow_type) -> bool:
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return isinstance(value, bool)
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    bit_width = int(arrow_type.bit_width)
    if pa.types.is_unsigned_integer(arrow_type):
        return 0 <= value <= (1 << bit_width) - 1
    return -(1 << (bit_width - 1)) <= value <= (1 << (bit_width - 1)) - 1


def _native_query_selection_node(node, state):
    """Lower one exact integer/bool comparison to row-flow selection."""
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or len(node.comparators) != 1
    ):
        return None
    symbol = _NATIVE_QUERY_COMPARISONS.get(type(node.ops[0]))
    if symbol is None:
        return None

    left = (
        _native_expression_column(state, node.left.id, comparison=True)
        if isinstance(node.left, ast.Name)
        else None
    )
    right_node = node.comparators[0]
    right = (
        _native_expression_column(state, right_node.id, comparison=True)
        if isinstance(right_node, ast.Name)
        else None
    )
    left_scalar = _native_expression_scalar(node.left)
    right_scalar = _native_expression_scalar(right_node)

    if left is not None and right is not None:
        return left[0].compare(symbol, right[0])
    if left is not None and right_scalar is not None:
        if not _native_query_scalar_fits_arrow(right_scalar, left[1]):
            return None
        return left[0].compare_scalar_selection(symbol, right_scalar)
    if left_scalar is not None and right is not None:
        if not _native_query_scalar_fits_arrow(left_scalar, right[1]):
            return None
        return right[0].compare_scalar_selection(
            _NATIVE_REVERSED_COMPARISON[symbol],
            left_scalar,
        )
    return None


def _native_device_carrier_bytes(value, *, seen: set[int] | None = None) -> int:
    """Count unique live CUDA buffers reachable from one private carrier."""
    if value is None:
        return 0
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    if hasattr(value, "__cuda_array_interface__"):
        return int(getattr(value, "nbytes", 0))
    if isinstance(value, dict):
        return sum(_native_device_carrier_bytes(item, seen=seen) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return sum(_native_device_carrier_bytes(item, seen=seen) for item in value)
    if type(value).__module__.startswith("vibespatial") and hasattr(value, "__dict__"):
        return sum(
            _native_device_carrier_bytes(item, seen=seen)
            for item in vars(value).values()
        )
    return 0


def _native_expression_transition_bytes(state, *, query: bool) -> int:
    attributes = _native_device_column_storage_bytes(
        state.attributes,
        tuple(state.attributes.columns),
    )
    if attributes is None:
        return 0
    carrier_bytes = _native_device_carrier_bytes(
        (
            state.geometry,
            state.secondary_geometry,
            state.geometry_metadata_cache,
            state.provenance,
            state.index_plan,
        )
    )
    if query:
        # Almost-identity query output is the worst admitted public take: all
        # attributes plus geometry/index row maps can be live beside sources.
        return int(attributes) + carrier_bytes + int(state.row_count) * 32
    physicalize_bytes = _native_attribute_row_view_gather_bytes(state.attributes)
    return physicalize_bytes + int(state.row_count) * 16


def _native_attribute_row_view_gather_bytes(attributes) -> int:
    """Bound the gather needed to physicalize deferred attribute row maps."""
    parts = getattr(attributes, "parts", None)
    if parts is not None:
        return sum(_native_attribute_row_view_gather_bytes(part) for part in parts)
    if getattr(attributes, "row_positions", None) is None:
        return 0
    storage_bytes = _native_device_column_storage_bytes(
        attributes,
        tuple(attributes.columns),
    )
    return 0 if storage_bytes is None else int(storage_bytes)


def _admit_native_public_expression(state, tree, *, stage: str, query: bool) -> bool:
    """Admit row-aligned expression scratch before any device allocation."""
    from vibespatial.cuda._runtime import get_cuda_runtime

    operator_count = sum(
        isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare))
        for node in ast.walk(tree)
    )
    scratch_bytes = int(state.row_count) * (96 + 16 * operator_count)
    required_bytes = (
        scratch_bytes
        + _native_expression_transition_bytes(state, query=query)
        + (1 << 20)
    )
    return get_cuda_runtime().admit_device_memory(
        stage=stage,
        required_bytes=required_bytes,
        requested_units=int(state.row_count),
    ).admitted


def _native_query_result(owner, expr: str, state):
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, TypeError, ValueError):
        return None
    if not _admit_native_public_expression(
        state,
        tree,
        stage="tabular-query-expression",
        query=True,
    ):
        return None
    selection = _native_query_selection_node(tree.body, state)
    if selection is None:
        return None
    from vibespatial.api._native_rowset import NativeDeviceSelection

    rowset = (
        selection.compact_rowset(
            surface="geopandas.geodataframe.query",
            strict_disallowed=False,
        )
        if isinstance(selection, NativeDeviceSelection)
        else selection
    )
    return _take_public_frame_with_native_state(
        owner,
        rowset,
        source_native_state=state,
        geometry_column=owner._geometry_column_name,
        preserve_index=True,
    )


def _native_eval_result(owner, expr: str, state):
    try:
        tree = ast.parse(expr, mode="exec")
    except (SyntaxError, TypeError, ValueError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return None
    target = assignment.targets[0].id
    if target == state.geometry_name:
        return None
    if not _admit_native_public_expression(
        state,
        tree,
        stage="tabular-eval-expression",
        query=False,
    ):
        return None
    expression = _native_eval_expression_node(assignment.value, state)
    from vibespatial.api._native_expression import NativeExpression

    if not isinstance(expression, NativeExpression):
        return None
    column_order = tuple(owner.columns)
    if target not in column_order:
        column_order = (*column_order, target)
    assigned = state.assign_expression_columns(
        {target: expression},
        column_order=column_order,
    )
    if assigned is None:
        return None
    return _public_frame_from_native_state(
        owner,
        assigned,
        geometry_column=owner._geometry_column_name,
    )


def _native_normalize_pandas_expression_syntax(
    expr: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]] | None:
    """Replace pandas-only name syntax so Python's AST can validate grammar."""
    output = []
    position = 0
    backtick_count = 0
    backtick_names = []
    local_names = []
    quote = None
    escaped = False
    while position < len(expr):
        character = expr[position]
        if quote is not None:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif expr.startswith(quote, position):
                if len(quote) > 1:
                    output.extend(quote[1:])
                    position += len(quote) - 1
                quote = None
            position += 1
            continue
        if character in {"'", '"'}:
            quote = character * 3 if expr.startswith(character * 3, position) else character
            output.extend(quote)
            position += len(quote)
            continue
        if character == "#":
            end = expr.find("\n", position)
            if end < 0:
                output.append(expr[position:])
                position = len(expr)
            else:
                output.append(expr[position:end])
                position = end
            continue
        if character == "`":
            end = expr.find("`", position + 1)
            if end < 0:
                return None
            backtick_names.append(expr[position + 1 : end])
            output.append(f"__vibespatial_backtick_{backtick_count}")
            backtick_count += 1
            position = end + 1
            continue
        if character == "@":
            position += 1
            while position < len(expr) and expr[position].isspace():
                position += 1
            if position >= len(expr) or not (
                expr[position].isalpha() or expr[position] == "_"
            ):
                return None
            end = position + 1
            while end < len(expr) and (
                expr[end].isalnum() or expr[end] == "_"
            ):
                end += 1
            local_names.append(expr[position:end])
            output.append(expr[position:end])
            position = end
            continue
        output.append(character)
        position += 1
    if quote is not None:
        return None
    return "".join(output), tuple(backtick_names), tuple(local_names)


def _native_expression_local_name_exists(
    name: str,
    *,
    scope_kwargs,
    scope_locals,
    scope_globals,
) -> bool:
    local_dict = scope_kwargs.get("local_dict")
    global_dict = scope_kwargs.get("global_dict")
    return any(
        mapping is not None and name in mapping
        for mapping in (
            local_dict,
            global_dict,
            scope_locals,
            scope_globals,
        )
    )


def _native_public_expression_is_invalid(
    owner,
    expr: str,
    *,
    assignment: bool,
    scope_kwargs=None,
    scope_locals=None,
    scope_globals=None,
) -> bool:
    """Identify syntax/undefined-name errors that are not CPU fallbacks."""
    mode = "exec" if assignment else "eval"
    try:
        tree = ast.parse(expr, mode=mode)
    except (SyntaxError, TypeError, ValueError):
        if not isinstance(expr, str):
            return True
        normalized_expression = _native_normalize_pandas_expression_syntax(expr)
        if normalized_expression is None:
            return True
        normalized, backtick_names, local_names = normalized_expression
        try:
            ast.parse(normalized, mode=mode)
        except (SyntaxError, TypeError, ValueError):
            return True
        public_names = set(owner.columns) | set(owner.index.names)
        if any(name not in public_names for name in backtick_names):
            return True
        scope_kwargs = {} if scope_kwargs is None else scope_kwargs
        if any(
            not _native_expression_local_name_exists(
                name,
                scope_kwargs=scope_kwargs,
                scope_locals=scope_locals,
                scope_globals=scope_globals,
            )
            for name in local_names
        ):
            return True
        return False
    known = {
        column
        for column in owner.columns
        if isinstance(column, str) and column.isidentifier()
    }
    known.add("index")
    known.update(
        name
        for name in owner.index.names
        if isinstance(name, str) and name.isidentifier()
    )
    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    supported_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _PANDAS_EXPRESSION_FUNCTIONS
    }
    return bool(referenced - known - supported_calls)


def _native_device_table_from_public_frame(frame: pd.DataFrame):
    """Lower an already-public attribute frame for one admitted join."""
    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.api._native_results import (
        _native_attribute_table_from_projected_frames,
    )

    try:
        attributes = _native_attribute_table_from_projected_frames(
            [frame],
            index_override=frame.index,
            storage="device",
        )
    except (ImportError, TypeError, ValueError):
        return None
    if attributes.is_device_backed or len(attributes.columns) == 0:
        return attributes
    try:
        import pylibcudf as plc

        columns = tuple(attributes.columns)
        return NativeAttributeTable(
            device_table=plc.Table(attributes.to_pylibcudf_columns(columns)),
            index_override=frame.index,
            column_override=columns,
            schema_override=attributes.arrow_schema_for_columns(columns),
        )
    except (ImportError, KeyError, TypeError, ValueError):
        return None


def _native_join_output_names(
    left_columns,
    right_columns,
    *,
    shared_keys,
    suffixes,
):
    """Return exact pandas-style column mappings for an admitted equality join."""
    try:
        left_suffix, right_suffix = suffixes
    except (TypeError, ValueError):
        return None
    shared_keys = set(shared_keys)
    overlap = (set(left_columns) - shared_keys) & (set(right_columns) - shared_keys)
    if overlap and not left_suffix and not right_suffix:
        return None

    left_mapping = {
        name: f"{name}{left_suffix}"
        for name in overlap
        if left_suffix is not None
    }
    right_mapping = {
        name: f"{name}{right_suffix}"
        for name in overlap
        if right_suffix is not None
    }
    left_output = tuple(left_mapping.get(name, name) for name in left_columns)
    right_output = tuple(right_mapping.get(name, name) for name in right_columns)
    combined = (*left_output, *right_output)
    if len(set(combined)) != len(combined):
        return None
    return left_mapping, right_mapping, left_output, right_output


_PANDAS_VALID_MERGE_VALIDATIONS = frozenset(
    {
        "1:1",
        "1:m",
        "m:1",
        "m:m",
        "one_to_one",
        "one_to_many",
        "many_to_one",
        "many_to_many",
    }
)


def _native_validate_argument_is_invalid(validate) -> bool:
    if validate is None:
        return False
    try:
        return validate not in _PANDAS_VALID_MERGE_VALIDATIONS
    except TypeError:
        return True


def _native_key_labels(value) -> tuple[Any, ...] | None:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _native_explicit_key_labels(value) -> tuple[Any, ...] | None:
    """Separate pandas label sequences from array-valued merge keys."""
    if isinstance(value, (np.ndarray, pd.Series, pd.Index, ExtensionArray)):
        return None
    labels = _native_key_labels(value)
    if labels is None:
        return None
    try:
        for label in labels:
            hash(label)
    except TypeError:
        return None
    return labels


def _native_is_array_valued_key(value) -> bool:
    return isinstance(value, (np.ndarray, pd.Series, pd.Index, ExtensionArray))


def _native_join_on_key_specs(value) -> tuple[Any, ...]:
    if _native_is_array_valued_key(value):
        return (value,)
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _native_merge_key_count(value) -> int:
    if value is None:
        return 0
    if isinstance(value, (np.ndarray, pd.Series, pd.Index, ExtensionArray)):
        return 1
    labels = _native_key_labels(value)
    return 0 if labels is None else len(labels)


def _native_public_key_exists(frame, key) -> bool:
    try:
        return key in frame.columns or key in frame.index.names
    except (TypeError, ValueError):
        return False


def _native_public_key_is_ambiguous(frame, key) -> bool:
    try:
        return key in frame.columns and key in frame.index.names
    except (TypeError, ValueError):
        return False


def _native_suffixes_are_invalid(suffixes) -> bool:
    if isinstance(suffixes, (str, bytes)):
        return True
    try:
        _left_suffix, _right_suffix = suffixes
    except (TypeError, ValueError):
        return True
    return False


def _native_public_merge_is_invalid(
    owner,
    right,
    *,
    on,
    left_on,
    right_on,
    left_index,
    right_index,
    suffixes,
    validate,
) -> bool:
    """Classify public validation failures that strict-native must not mask."""
    if _native_validate_argument_is_invalid(validate):
        return True
    if not isinstance(right, pd.DataFrame):
        return False
    if _native_suffixes_are_invalid(suffixes):
        return True
    keys = _native_key_labels(on)
    if keys is not None and any(
        not _native_public_key_exists(owner, key)
        or not _native_public_key_exists(right, key)
        or _native_public_key_is_ambiguous(owner, key)
        or _native_public_key_is_ambiguous(right, key)
        for key in keys
    ):
        return True
    if on is not None and left_on is None and right_on is None:
        shared_keys = set(keys or ())
        right_nonkeys = tuple(
            column for column in right.columns if column not in shared_keys
        )
        if _native_join_output_names(
            tuple(owner.columns),
            right_nonkeys,
            shared_keys=shared_keys,
            suffixes=suffixes,
        ) is None:
            return True
    left_keys = _native_explicit_key_labels(left_on)
    if not left_index and left_keys is not None and any(
        not _native_public_key_exists(owner, key)
        or _native_public_key_is_ambiguous(owner, key)
        for key in left_keys
    ):
        return True
    right_keys = _native_explicit_key_labels(right_on)
    if not right_index and right_keys is not None and any(
        not _native_public_key_exists(right, key)
        or _native_public_key_is_ambiguous(right, key)
        for key in right_keys
    ):
        return True
    if on is not None and (left_on is not None or right_on is not None):
        return True
    if on is not None and (left_index or right_index):
        return True
    if left_on is not None and left_index:
        return True
    if right_on is not None and right_index:
        return True
    if left_on is not None and right_on is None and not right_index:
        return True
    if right_on is not None and left_on is None and not left_index:
        return True
    if left_index and not right_index and right_on is None:
        return True
    if right_index and not left_index and left_on is None:
        return True
    if on is None:
        left_key_count = (
            int(owner.index.nlevels)
            if left_index
            else _native_merge_key_count(left_on)
        )
        right_key_count = (
            int(right.index.nlevels)
            if right_index
            else _native_merge_key_count(right_on)
        )
        if left_key_count and right_key_count and left_key_count != right_key_count:
            return True
    if (
        on is None
        and left_on is None
        and right_on is None
        and not left_index
        and not right_index
        and not (set(owner.columns) & set(right.columns))
    ):
        return True
    return False


def _native_join_validate_cardinality_is_invalid(state, other, validate) -> bool:
    if not isinstance(other, pd.DataFrame) or validate is None:
        return False
    left_unique = not bool(getattr(state.index_plan, "has_duplicates", False))
    right_unique = bool(other.index.is_unique)
    if validate in {"one_to_one", "1:1"}:
        return not left_unique or not right_unique
    if validate in {"one_to_many", "1:m"}:
        return not left_unique
    if validate in {"many_to_one", "m:1"}:
        return not right_unique
    return False


def _native_public_join_is_invalid(
    owner,
    other,
    *,
    state,
    on,
    lsuffix,
    rsuffix,
    validate,
) -> bool:
    """Classify index-join validation failures before strict fallback policy."""
    if _native_validate_argument_is_invalid(validate):
        return True
    if on is None and _native_join_validate_cardinality_is_invalid(
        state,
        other,
        validate,
    ):
        return True
    if on is not None:
        keys = _native_explicit_key_labels(on)
        if keys is None:
            key_specs = _native_join_on_key_specs(on)
            for key in key_specs:
                if _native_is_array_valued_key(key):
                    if len(key) != len(owner):
                        return True
                elif (
                    not _native_public_key_exists(owner, key)
                    or _native_public_key_is_ambiguous(owner, key)
                ):
                    return True
            if (
                isinstance(other, pd.DataFrame)
                and len(key_specs) != int(other.index.nlevels)
            ):
                return True
        else:
            if any(
                not _native_public_key_exists(owner, key) for key in keys
            ):
                return True
            if any(
                _native_public_key_is_ambiguous(owner, key) for key in keys
            ):
                return True
            if (
                isinstance(other, pd.DataFrame)
                and len(keys) != int(other.index.nlevels)
            ):
                return True
    if isinstance(other, pd.DataFrame):
        if _native_join_output_names(
            tuple(owner.columns),
            tuple(other.columns),
            shared_keys=(),
            suffixes=(lsuffix, rsuffix),
        ) is None:
            return True
    return False


def _native_frame_has_geometry_dtype(frame: pd.DataFrame) -> bool:
    return any(_is_geometry_like_dtype(dtype) for dtype in frame.dtypes)


def _native_join_columns_are_flat(frame: pd.DataFrame) -> bool:
    return not isinstance(frame.columns, pd.MultiIndex) and not any(
        isinstance(column, tuple) for column in frame.columns
    )


def _native_join_output_metadata(owner, right) -> tuple[dict[Any, Any], Any]:
    attrs = owner.attrs.copy() if owner.attrs == right.attrs else {}
    columns_name = (
        owner.columns.name
        if owner.columns.name == right.columns.name
        else None
    )
    return attrs, columns_name


def _native_join_admitted(
    state,
    right: pd.DataFrame,
    *,
    stage: str,
    output_rows: int,
) -> bool:
    """Admit inputs, relation maps, gathers, and output columns before H2D."""
    left_bytes = _native_device_column_storage_bytes(
        state.attributes,
        tuple(state.attributes.columns),
    )
    if left_bytes is None:
        return False
    try:
        right_bytes = int(right.memory_usage(index=False, deep=True).sum())
    except (AttributeError, TypeError, ValueError):
        return False
    carrier_bytes = _native_device_carrier_bytes(
        (
            state.geometry,
            state.secondary_geometry,
            state.geometry_metadata_cache,
            state.provenance,
            state.index_plan,
        )
    )
    row_view_gather_bytes = _native_attribute_row_view_gather_bytes(
        state.attributes
    )
    row_map_bytes = int(output_rows) * 32
    output_bytes = max(int(left_bytes) + right_bytes, int(output_rows))
    required_bytes = (
        5 * output_bytes
        + 2 * carrier_bytes
        + row_view_gather_bytes
        + row_map_bytes
        + (1 << 20)
    )
    from vibespatial.cuda._runtime import get_cuda_runtime

    return get_cuda_runtime().admit_device_memory(
        stage=stage,
        required_bytes=required_bytes,
        requested_units=int(output_rows),
    ).admitted


def _native_unique_right_inner_merge(
    owner,
    right,
    *,
    on,
    suffixes,
    validate,
):
    """Execute a stable many-to-one inner equality merge on device.

    Physical shape: two key tables -> unique-right equality relation -> stable
    left-row order -> output-shaped gathers -> NativeFrameState.  Requiring a
    unique right key bounds relation cardinality by the left row count.
    """
    state = _native_device_expression_state(owner)
    if state is None or not isinstance(right, pd.DataFrame):
        return None
    if isinstance(right, GeoDataFrame) or not owner.columns.is_unique or not right.columns.is_unique:
        return None
    if not _native_join_columns_are_flat(owner) or not _native_join_columns_are_flat(right):
        return None
    if state.secondary_geometry or _native_frame_has_geometry_dtype(right):
        return None
    if on is None:
        return None
    keys = (on,) if isinstance(on, (str, bytes)) else tuple(on)
    if not keys or len(set(keys)) != len(keys):
        return None
    if any(key == state.geometry_name for key in keys):
        return None
    if any(key not in state.attributes.columns or key not in right.columns for key in keys):
        return None
    if state.geometry_name in right.columns:
        return None
    if validate is not None and validate not in _PANDAS_VALID_MERGE_VALIDATIONS:
        return None
    if not _native_join_admitted(
        state,
        right,
        stage="tabular-unique-right-inner-merge",
        output_rows=state.row_count,
    ):
        return None

    right_attributes = _native_device_table_from_public_frame(right)
    if right_attributes is None or not right_attributes.is_device_backed:
        return None
    from pandas.errors import MergeError

    try:
        import cupy as cp
        import pylibcudf as plc
        import pylibcudf.sorting as sorting

        from vibespatial.api._native_result_core import (
            NativeAttributeTable,
            _pylibcudf_numeric_column_view,
        )
        from vibespatial.api._native_rowset import NativeIndexPlan
        from vibespatial.api._native_state import NativeFrameState
        from vibespatial.cuda._runtime import pylibcudf_current_stream
        from vibespatial.runtime.residency import combined_residency

        left_raw_columns = _native_pylibcudf_columns(state.attributes, keys)
        right_raw_columns = _native_pylibcudf_columns(right_attributes, keys)
        stream = pylibcudf_current_stream(
            *left_raw_columns,
            *right_raw_columns,
        )
        left_key_columns = _native_pylibcudf_missing_normalized_columns(
            left_raw_columns,
            stream=stream,
        )
        right_key_columns = _native_pylibcudf_missing_normalized_columns(
            right_raw_columns,
            stream=stream,
        )
        if any(
            left_column.type() != right_column.type()
            for left_column, right_column in zip(
                left_key_columns,
                right_key_columns,
                strict=True,
            )
        ):
            return None
        left_keys = plc.Table(left_key_columns)
        right_keys = plc.Table(right_key_columns)
        unique_right = plc.stream_compaction.distinct_indices(
            right_keys,
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
            stream=stream,
        )
        if int(unique_right.size()) != len(right):
            if validate in {"many_to_one", "m:1", "one_to_one", "1:1"}:
                relationship = (
                    "one-to-one"
                    if validate in {"one_to_one", "1:1"}
                    else "many-to-one"
                )
                raise MergeError(
                    "Merge keys are not unique in right dataset; "
                    f"not a {relationship} merge"
                )
            return None
        if validate in {"one_to_one", "1:1", "one_to_many", "1:m"}:
            unique_left = plc.stream_compaction.distinct_indices(
                left_keys,
                plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
                stream=stream,
            )
            if int(unique_left.size()) != len(owner):
                relationship = (
                    "one-to-one"
                    if validate in {"one_to_one", "1:1"}
                    else "one-to-many"
                )
                raise MergeError(
                    "Merge keys are not unique in left dataset; "
                    f"not a {relationship} merge"
                )
        left_column, right_column = plc.join.inner_join(
            left_keys,
            right_keys,
            plc.types.NullEquality.EQUAL,
            stream=stream,
        )
        order_column = sorting.stable_sorted_order(
            plc.Table([left_column, right_column]),
            [plc.types.Order.ASCENDING, plc.types.Order.ASCENDING],
            [plc.types.NullOrder.AFTER, plc.types.NullOrder.AFTER],
            stream=stream,
        )
        order = cp.asarray(_pylibcudf_numeric_column_view(order_column), dtype=cp.int64)
        left_positions = cp.asarray(
            _pylibcudf_numeric_column_view(left_column),
            dtype=cp.int64,
        )[order]
        right_positions = cp.asarray(
            _pylibcudf_numeric_column_view(right_column),
            dtype=cp.int64,
        )[order]
    except MergeError:
        raise
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return None

    right_nonkeys = tuple(column for column in right.columns if column not in keys)
    names = _native_join_output_names(
        tuple(state.column_order),
        right_nonkeys,
        shared_keys=keys,
        suffixes=suffixes,
    )
    if names is None:
        return None
    left_mapping, right_mapping, left_output, right_output = names
    geometry_name = left_mapping.get(state.geometry_name, state.geometry_name)
    if geometry_name != state.geometry_name:
        return None

    left_attributes = state.attributes.take(
        left_positions,
        preserve_index=False,
    ).rename_columns(left_mapping)
    right_projected = right_attributes.project_columns(right_nonkeys)
    if right_projected is None:
        return None
    right_projected = right_projected.take(
        right_positions,
        preserve_index=False,
    ).rename_columns(right_mapping)
    output_index = pd.RangeIndex(int(left_positions.size))
    attributes = NativeAttributeTable.combine_columns(
        [left_attributes, right_projected],
        index_override=output_index,
    )
    if attributes is None:
        return None
    geometry = state.geometry.take(
        left_positions,
        unique=False,
        defer_device_metadata=True,
    )
    metadata = (
        None
        if state.geometry_metadata_cache is None
        else state.geometry_metadata_cache.take(left_positions)
    )
    provenance = (
        state.provenance.take(left_positions)
        if state.provenance is not None and hasattr(state.provenance, "take")
        else state.provenance
    )
    output_attrs, output_columns_name = _native_join_output_metadata(owner, right)
    joined_state = NativeFrameState(
        attributes=attributes,
        geometry=geometry,
        geometry_name=geometry_name,
        column_order=(*left_output, *right_output),
        index_plan=NativeIndexPlan.from_index(output_index),
        row_count=int(left_positions.size),
        secondary_geometry=(),
        attrs=output_attrs,
        provenance=provenance,
        geometry_metadata_cache=metadata,
        residency=combined_residency(geometry),
        readiness=state.readiness,
    )
    return _public_frame_from_native_state(
        owner,
        joined_state,
        geometry_column=geometry_name,
        attrs=output_attrs,
        columns_name=output_columns_name,
    )


def _native_exact_index_join(
    owner,
    right,
    *,
    lsuffix,
    rsuffix,
    validate,
):
    """Combine exact unique aligned indexes without constructing a relation."""
    state = _native_device_expression_state(owner)
    if state is None or not isinstance(right, pd.DataFrame):
        return None
    if isinstance(right, GeoDataFrame) or not owner.columns.is_unique or not right.columns.is_unique:
        return None
    if not _native_join_columns_are_flat(owner) or not _native_join_columns_are_flat(right):
        return None
    if state.secondary_geometry or _native_frame_has_geometry_dtype(right):
        return None
    index_plan = state.index_plan
    source_index = getattr(index_plan, "index", None)
    if (
        getattr(index_plan, "kind", None) not in {"range", "host-labels"}
        or not isinstance(source_index, pd.Index)
        or bool(getattr(index_plan, "has_duplicates", False))
        or not right.index.is_unique
        or not source_index.equals(right.index)
    ):
        return None
    if state.geometry_name in right.columns:
        return None
    if validate not in {None, "one_to_one", "1:1", "many_to_one", "m:1"}:
        return None
    if not _native_join_admitted(
        state,
        right,
        stage="tabular-exact-index-join",
        output_rows=state.row_count,
    ):
        return None
    right_attributes = _native_device_table_from_public_frame(right)
    if right_attributes is None or not right_attributes.is_device_backed:
        return None
    names = _native_join_output_names(
        tuple(state.column_order),
        tuple(right.columns),
        shared_keys=(),
        suffixes=(lsuffix, rsuffix),
    )
    if names is None:
        return None
    left_mapping, right_mapping, left_output, right_output = names
    if left_mapping.get(state.geometry_name, state.geometry_name) != state.geometry_name:
        return None

    from vibespatial.api._native_result_core import NativeAttributeTable
    from vibespatial.api._native_state import NativeFrameState

    left_attributes = state.attributes.rename_columns(left_mapping)
    right_attributes = right_attributes.rename_columns(right_mapping)
    attributes = NativeAttributeTable.combine_columns(
        [left_attributes, right_attributes],
        index_override=source_index,
    )
    if attributes is None:
        return None
    output_attrs, output_columns_name = _native_join_output_metadata(owner, right)
    joined_state = NativeFrameState(
        attributes=attributes,
        geometry=state.geometry,
        geometry_name=state.geometry_name,
        column_order=(*left_output, *right_output),
        index_plan=state.index_plan,
        row_count=state.row_count,
        secondary_geometry=state.secondary_geometry,
        attrs=output_attrs,
        provenance=state.provenance,
        geometry_metadata_cache=state.geometry_metadata_cache,
        residency=state.residency,
        readiness=state.readiness,
    )
    return _public_frame_from_native_state(
        owner,
        joined_state,
        geometry_column=state.geometry_name,
        attrs=output_attrs,
        columns_name=output_columns_name,
    )


def _native_drop_duplicates_positions(owner, subset, keep) -> np.ndarray | None:
    """Return source row positions for admitted non-geometry duplicate drops."""
    if subset is None:
        return None

    columns = getattr(owner, "columns", pd.Index([]))
    try:
        if isinstance(subset, (str, bytes)):
            labels = (subset,)
        else:
            labels = tuple(subset)
    except TypeError:
        return None
    if not labels:
        return None
    try:
        if any(label not in columns for label in labels):
            return None
    except TypeError:
        return None
    if any(_is_geometry_like_dtype(getattr(owner[label], "dtype", None)) for label in labels):
        return None

    duplicated = pd.DataFrame.duplicated(owner, subset=subset, keep=keep)
    values = duplicated.to_numpy(dtype=bool, copy=False)
    return np.flatnonzero(~values).astype(np.int64, copy=False)


def _normalize_drop_duplicates_subset(subset, columns) -> tuple[Any, ...] | None:
    if subset is None:
        labels = tuple(columns)
    elif isinstance(subset, (str, bytes)):
        labels = (subset,)
    else:
        try:
            labels = tuple(subset)
        except TypeError:
            return None
    if not labels:
        return None
    try:
        if any(label not in columns for label in labels):
            return None
    except TypeError:
        return None
    return labels


def _native_drop_duplicates_rowset(owner, subset, keep):
    """Return a native rowset for device-backed attribute duplicate filtering.

    Physical shape: stable device row order -> sorted key gather -> distinct row
    positions -> source-ordered ``NativeRowSet``. Null and NaN equality follow
    pandas duplicate semantics; no key or result crosses to host.
    """
    if keep not in {"first", "last", False}:
        return None

    from vibespatial.api._native_result_core import _pylibcudf_numeric_column_view
    from vibespatial.api._native_rowset import NativeRowSet
    from vibespatial.api._native_state import get_native_state

    state = get_native_state(owner)
    if state is None or not _native_state_can_take_device_row_positions(state):
        return None
    labels = _normalize_drop_duplicates_subset(subset, owner.columns)
    if labels is None or any(
        _is_geometry_like_dtype(getattr(owner[label], "dtype", None))
        for label in labels
    ):
        return None
    attributes = getattr(state, "attributes", None)
    if attributes is None:
        return None
    attribute_columns = tuple(attributes.columns)
    if any(label not in attribute_columns for label in labels):
        return None

    if bool(getattr(attributes, "is_device_backed", False)):
        policies = _native_device_column_policies(attributes, labels)
        if any(
            (policy := policies.get(label)) is None or not policy.can_distinct
            for label in labels
        ):
            return None
        required_bytes = _native_distinct_required_bytes(
            attributes,
            labels,
            int(state.row_count),
        )
        if required_bytes is None:
            return None
        from vibespatial.cuda._runtime import get_cuda_runtime

        admission = get_cuda_runtime().admit_device_memory(
            stage="tabular-distinct",
            required_bytes=required_bytes,
            requested_units=int(state.row_count),
        )
        if not admission.admitted:
            return None

    if len(labels) == 1:
        arrays = attributes.numeric_column_arrays(labels)
        if arrays is not None:
            values = arrays.get(labels[0])
            if values is not None:
                dtype = np.dtype(getattr(values, "dtype", np.dtype("O")))
                if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
                    try:
                        import cupy as cp
                    except ModuleNotFoundError:
                        return None
                    d_values = cp.asarray(values)
                    row_count = int(d_values.size)
                    if row_count == 0:
                        unique_positions = cp.asarray([], dtype=cp.int64)
                    elif keep == "first":
                        _unique_values, first_positions = cp.unique(
                            d_values,
                            return_index=True,
                        )
                        unique_positions = cp.sort(first_positions.astype(cp.int64, copy=False))
                    elif keep == "last":
                        _unique_values, reverse_first_positions = cp.unique(
                            d_values[::-1],
                            return_index=True,
                        )
                        unique_positions = cp.sort(
                            row_count
                            - 1
                            - reverse_first_positions.astype(cp.int64, copy=False)
                        )
                    else:
                        _unique_values, first_positions, counts = cp.unique(
                            d_values,
                            return_index=True,
                            return_counts=True,
                        )
                        unique_positions = cp.sort(
                            first_positions.astype(cp.int64, copy=False)[counts == 1]
                        )
                    return NativeRowSet.from_positions(
                        unique_positions.astype(cp.int64, copy=False),
                        source_token=state.lineage_token,
                        source_row_count=state.row_count,
                        ordered=True,
                        unique=True,
                        identity=int(unique_positions.size) == int(state.row_count),
                    )

    if not bool(getattr(attributes, "is_device_backed", False)):
        return None

    try:
        import cupy as cp
        import pylibcudf as plc
        import pylibcudf.sorting as sorting
        import pylibcudf.stream_compaction as stream_compaction
        from pylibcudf.types import NanEquality, NullEquality, NullOrder, Order
    except ModuleNotFoundError:
        return None

    keep_option = {
        "first": stream_compaction.DuplicateKeepOption.KEEP_FIRST,
        "last": stream_compaction.DuplicateKeepOption.KEEP_LAST,
        False: stream_compaction.DuplicateKeepOption.KEEP_NONE,
    }[keep]
    try:
        from vibespatial.cuda._runtime import pylibcudf_current_stream

        stream = pylibcudf_current_stream(attributes.device_table)
        key_columns = _native_pylibcudf_missing_normalized_columns(
            _native_pylibcudf_columns(attributes, labels),
            stream=stream,
        )
        key_table = plc.Table(key_columns)
        sorted_column = sorting.stable_sorted_order(
            key_table,
            [Order.ASCENDING] * len(labels),
            [NullOrder.AFTER] * len(labels),
            stream=stream,
        )
        sorted_positions = _pylibcudf_numeric_column_view(sorted_column)
        if sorted_positions is None:
            return None
        sorted_positions = cp.asarray(sorted_positions, dtype=cp.int64)
        target_dtype = cp.int32 if state.row_count <= np.iinfo(np.int32).max else cp.int64
        from vibespatial.cuda._runtime import pylibcudf_column_from_device

        gather_map = pylibcudf_column_from_device(
            sorted_positions.astype(target_dtype, copy=False)
        )
        sorted_keys = plc.copying.gather(
            key_table,
            gather_map,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            stream=stream,
        )
        unique_local_column = stream_compaction.distinct_indices(
            sorted_keys,
            keep_option,
            NullEquality.EQUAL,
            NanEquality.ALL_EQUAL,
            stream=stream,
        )
        unique_local = _pylibcudf_numeric_column_view(unique_local_column)
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return None
    if unique_local is None:
        return None
    unique_positions = sorted_positions[cp.asarray(unique_local, dtype=cp.int64)]
    unique_positions = unique_positions[cp.argsort(unique_positions)]
    return NativeRowSet.from_positions(
        unique_positions.astype(cp.int64, copy=False),
        source_token=state.lineage_token,
        source_row_count=state.row_count,
        ordered=True,
        unique=True,
        identity=int(unique_positions.size) == int(state.row_count),
    )


def _concat_input_objects(other) -> list[Any]:
    if PANDAS_GE_30:
        return list(getattr(other, "input_objs", ()))
    return list(getattr(other, "objs", ()))


def _attach_native_state_after_concat(result, input_objs: list[Any]) -> bool:
    """Attach native state for exact row-wise concat with admitted index output.

    Physical shape: homogeneous native frame append. Native input carriers are
    `NativeFrameState` objects; native output carrier is a concatenated
    `NativeTabularResult` attached back to the public GeoDataFrame.
    """
    if not input_objs or not isinstance(result, GeoDataFrame):
        return False
    geometry_name = getattr(result, "_geometry_column_name", None)
    if geometry_name is None or geometry_name not in result.columns:
        return False
    range_output = result.index.equals(pd.RangeIndex(len(result)))

    from vibespatial.api._native_results import _concat_native_tabular_results
    from vibespatial.api._native_state import (
        attach_native_state_from_native_tabular_result,
        get_native_state,
    )

    result_columns = tuple(result.columns)
    states = []
    payloads = []
    for obj in input_objs:
        if not isinstance(obj, GeoDataFrame):
            return False
        if tuple(obj.columns) != result_columns:
            return False
        if getattr(obj, "_geometry_column_name", None) != geometry_name:
            return False
        state = get_native_state(obj)
        if state is None:
            return False
        if state.geometry_name != geometry_name or state.column_order != result_columns:
            return False
        states.append(state)
        payloads.append(state.to_native_tabular_result())

    if sum(state.row_count for state in states) != len(result):
        return False

    if range_output:
        ignore_index = True
    else:
        expected_index = input_objs[0].index
        for obj in input_objs[1:]:
            expected_index = expected_index.append(obj.index)
        if not expected_index.equals(result.index):
            return False
        ignore_index = False

    try:
        payload = _concat_native_tabular_results(
            payloads,
            geometry_name=geometry_name,
            crs=getattr(result, "crs", None),
            attrs=result.attrs.copy() or None,
            ignore_index=ignore_index,
        )
    except Exception:
        return False
    if payload.geometry.row_count != len(result):
        return False
    if payload.resolved_column_order != result_columns:
        return False

    attach_native_state_from_native_tabular_result(result, payload)
    return True


class _NativeStateInvalidatingIndexer:
    """Delegate pandas indexer reads while clearing private state on writes."""

    def __init__(self, indexer, owner, *, kind: str) -> None:
        self._indexer = indexer
        self._owner = owner
        self._kind = kind

    def __getitem__(self, key):
        if self._kind == "loc":
            native_result = _native_loc_from_boolean_mask(self._owner, key)
            if native_result is not None:
                return native_result
            native_result = _native_loc_from_lazy_unique_index(self._owner, key)
            if native_result is not None:
                return native_result
        result = self._indexer[key]
        if self._kind == "iloc":
            _attach_native_state_after_iloc(self._owner, key, result)
        elif self._kind == "loc":
            _attach_native_state_after_loc(self._owner, key, result)
        return result

    def __setitem__(self, key, value) -> None:
        from vibespatial.api._native_state import drop_native_state

        drop_native_state(self._owner)
        self._indexer[key] = value

    def __call__(self, *args, **kwargs):
        called = self._indexer(*args, **kwargs)
        if called is self._indexer:
            return self
        return type(self)(called, self._owner, kind=self._kind)

    def __getattr__(self, name):
        return getattr(self._indexer, name)


crs_mismatch_error = (
    "CRS mismatch between CRS of the passed geometries "
    "and 'crs'. Use 'GeoDataFrame.set_crs(crs, "
    "allow_override=True)' to overwrite CRS or "
    "'GeoDataFrame.to_crs(crs)' to reproject geometries. "
)


class GeoDataFrame(GeoPandasBase, DataFrame):
    """A GeoDataFrame object is a pandas.DataFrame that has one or more columns
    containing geometry.

    In addition to the standard DataFrame constructor arguments,
    GeoDataFrame also accepts the following keyword arguments:

    Parameters
    ----------
    crs : value (optional)
        Coordinate Reference System of the geometry objects. Can be anything accepted by
        :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
    geometry : str or array-like (optional)
        Value to use as the active geometry column.
        If str, treated as column name to use. If array-like, it will be
        added as new column named 'geometry' on the GeoDataFrame and set as the
        active geometry column.

        Note that if ``geometry`` is a (Geo)Series with a
        name, the name will not be used, a column named "geometry" will still be
        added. To preserve the name, you can use :meth:`~GeoDataFrame.rename_geometry`
        to update the geometry column name.

    Examples
    --------
    Constructing GeoDataFrame from a dictionary.

    >>> from shapely.geometry import Point
    >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
    >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
    >>> gdf
        col1     geometry
    0  name1  POINT (1 2)
    1  name2  POINT (2 1)

    Notice that the inferred dtype of 'geometry' columns is geometry.

    >>> gdf.dtypes
    col1             str
    geometry    geometry
    dtype: object

    Constructing GeoDataFrame from a pandas DataFrame with a column of WKT geometries:

    >>> import pandas as pd
    >>> d = {'col1': ['name1', 'name2'], 'wkt': ['POINT (1 2)', 'POINT (2 1)']}
    >>> df = pd.DataFrame(d)
    >>> gs = geopandas.GeoSeries.from_wkt(df['wkt'])
    >>> gdf = geopandas.GeoDataFrame(df, geometry=gs, crs="EPSG:4326")
    >>> gdf
        col1          wkt     geometry
    0  name1  POINT (1 2)  POINT (1 2)
    1  name2  POINT (2 1)  POINT (2 1)

    See Also
    --------
    GeoSeries : Series object designed to store shapely geometry objects
    """

    _metadata = ["_geometry_column_name"]

    _internal_names = DataFrame._internal_names + ["geometry"]
    _internal_names_set = set(_internal_names)

    _geometry_column_name = None

    def __repr__(self) -> str:
        _record_native_display_export(
            self,
            surface="vibespatial.api.GeoDataFrame.__repr__",
            operation="geodataframe_repr",
            target="repr",
        )
        return super().__repr__()

    def _repr_html_(self):
        _record_native_display_export(
            self,
            surface="vibespatial.api.GeoDataFrame._repr_html_",
            operation="geodataframe_html_repr",
            target="html-repr",
        )
        return super()._repr_html_()

    def __init__(
        self,
        data=None,
        *args,
        geometry: Any | None = None,
        crs: Any | None = None,
        **kwargs,
    ):
        if (
            kwargs.get("copy") is None
            and isinstance(data, DataFrame)
            and not isinstance(data, GeoDataFrame)
        ):
            kwargs.update(copy=True)

        if data is None and "columns" not in kwargs:
            # pandas will interpret "str" as object dtype for pandas < 3 and
            # as string dtype for pandas >= 3. This ensures we still get string
            # columns when doing GeoDataFrame(geometry=[..])
            kwargs["columns"] = pd.Index([], dtype="str")

        super().__init__(data, *args, **kwargs)

        if isinstance(data, DataFrame) and data.attrs:
            self.attrs = data.attrs

        # set_geometry ensures the geometry data have the proper dtype,
        # but is not called if `geometry=None` ('geometry' column present
        # in the data), so therefore need to ensure it here manually
        # but within a try/except because currently non-geometries are
        # allowed in that case
        # TODO do we want to raise / return normal DataFrame in this case?

        # if gdf passed in and geo_col is set, we use that for geometry
        if geometry is None and isinstance(data, GeoDataFrame):
            self._geometry_column_name = data._geometry_column_name
            if crs is not None and data.crs != crs:
                raise ValueError(crs_mismatch_error)

        if (
            geometry is None
            and self.columns.nlevels == 1
            and "geometry" in self.columns
        ):
            # Check for multiple columns with name "geometry". If there are,
            # self["geometry"] is a gdf and constructor gets recursively recalled
            # by pandas internals trying to access this
            if (self.columns == "geometry").sum() > 1:
                raise ValueError(
                    "GeoDataFrame does not support multiple columns "
                    "using the geometry column name 'geometry'."
                )

            # only if we have actual geometry values -> call set_geometry
            if (
                hasattr(self["geometry"].values, "crs")
                and self["geometry"].values.crs
                and crs
                and not self["geometry"].values.crs == crs
            ):
                raise ValueError(crs_mismatch_error)
            # If "geometry" is potentially coercible to geometry, we try and convert it
            geom_dtype = self["geometry"].dtype
            if (
                geom_dtype == "geometry"
                or getattr(geom_dtype, "name", None) == "device_geometry"
                or geom_dtype == "object"
                # special case for geometry = [], has float dtype
                or (len(self) == 0 and geom_dtype == "float")
                # special case for geometry = [np.nan]
                or ((not self.empty) and self["geometry"].isna().all())
            ):
                try:
                    self["geometry"] = _ensure_geometry(self["geometry"].values, crs)
                except TypeError:
                    pass
                else:
                    # feed through to call set geometry below
                    geometry = "geometry"

        if geometry is not None:
            if (
                hasattr(geometry, "crs")
                and geometry.crs
                and crs
                and not geometry.crs == crs
            ):
                raise ValueError(crs_mismatch_error)

            if isinstance(geometry, pd.Series) and geometry.name not in (
                "geometry",
                None,
            ):
                # __init__ always creates geometry col named "geometry"
                # rename as `set_geometry` respects the given series name
                geometry = geometry.rename("geometry")

            self.set_geometry(geometry, inplace=True, crs=crs)

        if geometry is None and crs:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Supply geometry using the 'geometry=' keyword argument, "
                "or by providing a DataFrame with column name 'geometry'",
            )

    def __setattr__(self, attr, val):
        # have to special case geometry b/c pandas tries to use as column...
        if attr == "geometry":
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)

    def _get_geometry(self) -> GeoSeries:
        if self._geometry_column_name not in self:
            if self._geometry_column_name is None:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    "but the active geometry column to use has not been set. "
                )
            else:
                msg = (
                    "You are calling a geospatial method on the GeoDataFrame, "
                    f"but the active geometry column ('{self._geometry_column_name}') "
                    "is not present. "
                )
            _mask = self.dtypes.map(_is_geometry_like_dtype)
            geo_cols = list(self.columns[np.asarray(_mask, dtype=bool)])
            if len(geo_cols) > 0:
                msg += (
                    f"\nThere are columns with geometry data type ({geo_cols}), and "
                    "you can either set one as the active geometry with "
                    'df.set_geometry("name") or access the column as a '
                    'GeoSeries (df["name"]) and call the method directly on it.'
                )
            else:
                msg += (
                    "\nThere are no existing columns with geometry data type. You can "
                    "add a geometry column as the active geometry column with "
                    "df.set_geometry. "
                )

            raise AttributeError(msg)
        return self[self._geometry_column_name]

    def _set_geometry(self, col):
        if not pd.api.types.is_list_like(col):
            raise ValueError("Must use a list-like to set the geometry property")
        self._persist_old_default_geometry_colname()
        self.set_geometry(col, inplace=True)

    geometry = property(
        fget=_get_geometry, fset=_set_geometry, doc="Geometry data for GeoDataFrame"
    )

    @typing.overload
    def set_geometry(
        self,
        col,
        drop: bool | None = ...,
        inplace: Literal[True] = ...,
        crs: Any | None = ...,
    ) -> None: ...

    @typing.overload
    def set_geometry(
        self,
        col,
        drop: bool | None = ...,
        inplace: Literal[False] = ...,
        crs: Any | None = ...,
    ) -> GeoDataFrame: ...

    def set_geometry(
        self,
        col,
        drop: bool | None = None,
        inplace: bool = False,
        crs: Any | None = None,
    ) -> GeoDataFrame | None:
        """
        Set the GeoDataFrame geometry using either an existing column or
        the specified input. By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : column label or array-like
            An existing column name or values to set as the new geometry column.
            If values (array-like, (Geo)Series) are passed, then if they are named
            (Series) the new geometry column will have the corresponding name,
            otherwise the existing geometry column will be replaced. If there is
            no existing geometry column, the new geometry column will use the
            default name "geometry".
        drop : boolean, default False
            When specifying a named Series or an existing column name for `col`,
            controls if the previous geometry column should be dropped from the
            result. The default of False keeps both the old and new geometry column.

            .. deprecated:: 1.0.0

        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)
        crs : pyproj.CRS, optional
            Coordinate system to use. The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
            If passed, overrides both DataFrame and col's crs.
            Otherwise, tries to get crs from passed col values or DataFrame.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        Passing an array:

        >>> df1 = gdf.set_geometry([Point(0,0), Point(1,1)])
        >>> df1
            col1     geometry
        0  name1  POINT (0 0)
        1  name2  POINT (1 1)

        Using existing column:

        >>> gdf["buffered"] = gdf.buffer(2)
        >>> df2 = gdf.set_geometry("buffered")
        >>> df2.geometry
        0    POLYGON ((3 2, 2.99037 1.80397, 2.96157 1.6098...
        1    POLYGON ((4 1, 3.99037 0.80397, 3.96157 0.6098...
        Name: buffered, dtype: geometry

        Returns
        -------
        GeoDataFrame

        See Also
        --------
        GeoDataFrame.rename_geometry : rename an active geometry column
        """
        from vibespatial.api._native_state import get_native_state

        source_native_state = get_native_state(self)
        if (
            source_native_state is not None
            and isinstance(col, (str, bytes))
            and col in self.columns
            and not drop
            and crs is None
        ):
            updated_state = source_native_state.with_active_geometry(
                col,
                crs=getattr(pd.DataFrame.__getitem__(self, col), "crs", None),
            )
            if updated_state is not None:
                from vibespatial.api._native_state import attach_native_state

                frame = self if inplace else self.copy(deep=False)
                frame._geometry_column_name = col
                attach_native_state(frame, updated_state)
                return None if inplace else frame
        existing_geometry_name = None
        # Most of the code here is taken from DataFrame.set_index()
        if inplace:
            frame = self
        else:
            if PANDAS_GE_30:
                frame = self.copy(deep=False)
            else:
                frame = self.copy()

        geo_column_name = self._geometry_column_name

        if geo_column_name is None:
            geo_column_name = "geometry"
        if isinstance(col, Series | list | np.ndarray | GeometryArray | ExtensionArray):
            if drop:
                msg = (
                    "The `drop` keyword argument is deprecated and has no effect when "
                    "`col` is an array-like value. You should stop passing `drop` to "
                    "`set_geometry` when this is the case."
                )
                warnings.warn(msg, category=FutureWarning, stacklevel=2)
            if isinstance(col, Series) and col.name is not None:
                geo_column_name = col.name

            level = col
        elif hasattr(col, "ndim") and col.ndim > 1:
            raise ValueError("Must pass array with one dimension only.")
        else:  # should be a colname
            try:
                level = frame[col]
            except KeyError as err:
                raise ValueError(f"Unknown column {col}") from err
            if isinstance(level, DataFrame):
                raise ValueError(
                    "GeoDataFrame does not support setting the geometry column where "
                    "the column name is shared by multiple columns."
                )

            given_colname_drop_msg = (
                "The `drop` keyword argument is deprecated and in future the only "
                "supported behaviour will match drop=False. To silence this "
                "warning and adopt the future behaviour, stop providing "
                "`drop` as a keyword to `set_geometry`. To replicate the "
                "`drop=True` behaviour you should update "
                "your code to\n`geo_col_name = gdf.active_geometry_name;"
                " gdf.set_geometry(new_geo_col).drop("
                "columns=geo_col_name).rename_geometry(geo_col_name)`."
            )

            if drop is False:  # specifically False, not falsy i.e. None
                # User supplied False explicitly, but arg is deprecated
                warnings.warn(
                    given_colname_drop_msg,
                    category=FutureWarning,
                    stacklevel=2,
                )
            if drop:
                del frame[col]
                frame.__class__ = GeoDataFrame
                # revert the casting done in __delitem__, keep gdf
                warnings.warn(
                    given_colname_drop_msg,
                    category=FutureWarning,
                    stacklevel=2,
                )
            else:
                # if not dropping, set the active geometry name to the given col name
                geo_column_name = col
            existing_geometry_name = col

        if not crs:
            crs = getattr(level, "crs", None)

        current_geom_dtype = None
        if self._geometry_column_name is not None and self._geometry_column_name in self.columns:
            current_geom_dtype = getattr(self[self._geometry_column_name], "dtype", None)
        prefer_device = getattr(current_geom_dtype, "name", None) == "device_geometry"

        # Check that we are using a listlike of geometries
        level = _ensure_geometry(
            level,
            crs=crs,
            prefer_device=prefer_device,
            fallback_surface="GeoDataFrame.set_geometry",
        )
        # ensure_geometry only sets crs on level if it has crs==None
        if isinstance(level, GeoSeries):
            level.array.crs = crs
        else:
            level.crs = crs
        # update _geometry_column_name prior to assignment
        # to avoid default is None warning
        frame._geometry_column_name = geo_column_name
        frame[geo_column_name] = level
        if existing_geometry_name is not None:
            _attach_native_state_after_set_geometry(
                self,
                frame,
                source_native_state,
                geo_column_name,
            )

        if not inplace:
            return frame

    @typing.overload
    def rename_geometry(
        self,
        col: str,
        inplace: Literal[True] = ...,
    ) -> None: ...

    @typing.overload
    def rename_geometry(
        self,
        col: str,
        inplace: Literal[False] = ...,
    ) -> GeoDataFrame: ...

    def rename_geometry(self, col: str, inplace: bool = False) -> GeoDataFrame | None:
        """Rename the GeoDataFrame geometry column to the specified name.

        By default yields a new object.

        The original geometry column is replaced with the input.

        Parameters
        ----------
        col : new geometry column label
        inplace : boolean, default False
            Modify the GeoDataFrame in place (do not create a new object)

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> df = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> df1 = df.rename_geometry('geom1')
        >>> df1.geometry.name
        'geom1'
        >>> df.rename_geometry('geom1', inplace=True)
        >>> df.geometry.name
        'geom1'


        See Also
        --------
        GeoDataFrame.set_geometry : set the active geometry
        """
        geometry_col = self.geometry.name
        if col in self.columns:
            raise ValueError(f"Column named {col} already exists")
        else:
            if not inplace:
                result = self.rename(columns={geometry_col: col}).set_geometry(
                    col, inplace=inplace
                )
                _attach_native_state_after_column_rename(self, result)
                return result
            self.rename(columns={geometry_col: col}, inplace=inplace)
            self.set_geometry(col, inplace=inplace)

    @property
    def active_geometry_name(self) -> Any:
        """Return the name of the active geometry column.

        Returns a name if a GeoDataFrame has an active geometry column set,
        otherwise returns None. The return type is usually a string, but may be
        an integer, tuple or other hashable, depending on the contents of the
        dataframe columns.

        You can also access the active geometry column using the
        ``.geometry`` property. You can set a GeoSeries to be an active geometry
        using the :meth:`~GeoDataFrame.set_geometry` method.

        Returns
        -------
        str or other index label supported by pandas
            name of an active geometry column or None

        See Also
        --------
        GeoDataFrame.set_geometry : set the active geometry
        """
        return self._geometry_column_name

    @property
    def crs(self) -> CRS:
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns
        -------
        ``pyproj.CRS`` | None
            CRS assigned to an active geometry column

        Examples
        --------
        >>> gdf.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        See Also
        --------
        GeoDataFrame.set_crs : assign CRS
        GeoDataFrame.to_crs : re-project to another CRS

        """
        try:
            return self.geometry.crs
        except AttributeError as err:
            raise AttributeError(
                "The CRS attribute of a GeoDataFrame without an active "
                "geometry column is not defined. Use GeoDataFrame.set_geometry "
                "to set the active geometry column."
            ) from err

    @property
    def gpu_spatial_index(self):
        """GPU-resident Hilbert R-tree spatial index, or None if not built.

        Built automatically when ``read_file(..., build_index=True)`` is used.
        Can also be built manually via
        ``vibespatial.io.gpu_parse.build_spatial_index()``.

        Returns
        -------
        GpuSpatialIndex or None
            The packed Hilbert R-tree spatial index attached to this
            GeoDataFrame, or ``None`` if no index has been built.
        """
        return getattr(self, "_gpu_spatial_index", None)

    @crs.setter
    def crs(self, value) -> None:
        """Set the value of the crs."""
        if self._geometry_column_name is None:
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without a geometry column is not "
                "supported. Use GeoDataFrame.set_geometry to set the active "
                "geometry column.",
            )

        if hasattr(self.geometry.values, "crs"):
            if self.crs is not None:
                warnings.warn(
                    "Overriding the CRS of a GeoDataFrame that already has CRS. "
                    "This unsafe behavior will be deprecated in future versions. "
                    "Use GeoDataFrame.set_crs method instead",
                    stacklevel=2,
                    category=FutureWarning,
                )
            self.geometry.values.crs = value
        else:
            # column called 'geometry' without geometry
            raise ValueError(
                "Assigning CRS to a GeoDataFrame without an active geometry "
                "column is not supported. Use GeoDataFrame.set_geometry to set "
                "the active geometry column.",
            )

    def __setstate__(self, state) -> None:
        # overriding DataFrame method for compat with older pickles (CRS handling)
        crs = None
        if isinstance(state, dict):
            if "crs" in state and "_crs" not in state:
                crs = state.pop("crs", None)
            else:
                crs = state.pop("_crs", None)
            if crs is not None and not HAS_PYPROJ:
                raise ImportError(
                    "Unpickling a GeoDataFrame with CRS requires the 'pyproj' package, "
                    "but it is not installed or does not import correctly. "
                )
            elif crs is not None:
                from pyproj import CRS

                crs = CRS.from_user_input(crs)

        super().__setstate__(state)

        # for some versions that didn't yet have CRS at array level -> crs is set
        # at GeoDataFrame level with '_crs' (and not 'crs'), so without propagating
        # to the GeoSeries/GeometryArray
        try:
            if crs is not None:
                if self.geometry.values.crs is None:
                    self.crs = crs
        except Exception:
            pass

    @classmethod
    def from_dict(
        cls,
        data: dict,
        geometry=None,
        crs: Any | None = None,
        **kwargs,
    ) -> GeoDataFrame:
        """Construct GeoDataFrame from dict of array-like or dicts by
        overriding DataFrame.from_dict method with geometry and crs.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        geometry : str or array (optional)
            If str, column to use as geometry. If array, will be set as 'geometry'
            column on GeoDataFrame.
        crs : str or dict (optional)
            Coordinate reference system to set on the resulting frame.
        kwargs : key-word arguments
            These arguments are passed to DataFrame.from_dict

        Returns
        -------
        GeoDataFrame

        """
        dataframe = DataFrame.from_dict(data, **kwargs)
        return cls(dataframe, geometry=geometry, crs=crs)

    @classmethod
    def from_file(cls, filename: os.PathLike | typing.IO, **kwargs) -> GeoDataFrame:
        """Alternate constructor to create a ``GeoDataFrame`` from a file.

        It is recommended to use :func:`geopandas.read_file` instead.

        Can load a ``GeoDataFrame`` from a file in any format recognized by
        `pyogrio`. See http://pyogrio.readthedocs.io/ for details.

        Parameters
        ----------
        filename : str
            File path or file handle to read from. Depending on which kwargs
            are included, the content of filename may vary. See
            :func:`pyogrio.read_dataframe` for usage details.
        kwargs : key-word arguments
            These arguments are passed to :func:`pyogrio.read_dataframe`, and can be
            used to access multi-layer data, data stored within archives (zip files),
            etc.

        Examples
        --------
        >>> import geodatasets
        >>> path = geodatasets.get_path('nybb')
        >>> gdf = geopandas.GeoDataFrame.from_file(path)
        >>> gdf  # doctest: +SKIP
           BoroCode       BoroName     Shape_Leng    Shape_Area                 \
                          geometry
        0         5  Staten Island  330470.010332  1.623820e+09  MULTIPOLYGON ((\
(970217.022 145643.332, 970227....
        1         4         Queens  896344.047763  3.045213e+09  MULTIPOLYGON ((\
(1029606.077 156073.814, 102957...
        2         3       Brooklyn  741080.523166  1.937479e+09  MULTIPOLYGON ((\
(1021176.479 151374.797, 102100...
        3         1      Manhattan  359299.096471  6.364715e+08  MULTIPOLYGON ((\
(981219.056 188655.316, 980940....
        4         2          Bronx  464392.991824  1.186925e+09  MULTIPOLYGON ((\
(1012821.806 229228.265, 101278...

        The recommended method of reading files is :func:`geopandas.read_file`:

        >>> gdf = geopandas.read_file(path)

        See Also
        --------
        read_file : read file to GeoDataFrame
        GeoDataFrame.to_file : write GeoDataFrame to file

        """
        return geopandas.io.file._read_file(filename, **kwargs)

    @classmethod
    def from_features(
        cls, features, crs: Any | None = None, columns: Iterable[str] | None = None
    ) -> GeoDataFrame:
        """
        Alternate constructor to create GeoDataFrame from an iterable of
        features or a feature collection.

        Parameters
        ----------
        features
            - Iterable of features, where each element must be a feature
              dictionary or implement the __geo_interface__.
            - Feature collection, where the 'features' key contains an
              iterable of features.
            - Object holding a feature collection that implements the
              ``__geo_interface__``.
        crs : str or dict (optional)
            Coordinate reference system to set on the resulting frame.
        columns : list of column names, optional
            Optionally specify the column names to include in the output frame.
            This does not overwrite the property names of the input, but can
            ensure a consistent output format.

        Returns
        -------
        GeoDataFrame

        Notes
        -----
        For more information about the ``__geo_interface__``, see
        https://gist.github.com/sgillies/2217756

        Examples
        --------
        >>> feature_coll = {
        ...     "type": "FeatureCollection",
        ...     "features": [
        ...         {
        ...             "id": "0",
        ...             "type": "Feature",
        ...             "properties": {"col1": "name1"},
        ...             "geometry": {"type": "Point", "coordinates": (1.0, 2.0)},
        ...             "bbox": (1.0, 2.0, 1.0, 2.0),
        ...         },
        ...         {
        ...             "id": "1",
        ...             "type": "Feature",
        ...             "properties": {"col1": "name2"},
        ...             "geometry": {"type": "Point", "coordinates": (2.0, 1.0)},
        ...             "bbox": (2.0, 1.0, 2.0, 1.0),
        ...         },
        ...     ],
        ...     "bbox": (1.0, 1.0, 2.0, 2.0),
        ... }
        >>> df = geopandas.GeoDataFrame.from_features(feature_coll)
        >>> df
              geometry   col1
        0  POINT (1 2)  name1
        1  POINT (2 1)  name2

        """
        # Handle feature collections
        if hasattr(features, "__geo_interface__"):
            fs = features.__geo_interface__
        else:
            fs = features

        if isinstance(fs, dict) and fs.get("type") == "FeatureCollection":
            features_lst = fs["features"]
        else:
            features_lst = features

        rows = []
        for feature in features_lst:
            # load geometry
            if hasattr(feature, "__geo_interface__"):
                feature = feature.__geo_interface__
            row = {
                "geometry": shape(feature["geometry"]) if feature["geometry"] else None
            }
            # load properties
            properties = feature.get("properties") or {}
            row.update(properties)
            rows.append(row)
        return cls(rows, columns=columns, crs=crs)

    @classmethod
    def from_postgis(
        cls,
        sql: str | sqlalchemy.text,
        con,
        geom_col: str = "geom",
        crs: Any | None = None,
        index_col: str | list[str] | None = None,
        coerce_float: bool = True,
        parse_dates: list | dict | None = None,
        params: list | tuple | dict | None = None,
        chunksize: int | None = None,
    ) -> GeoDataFrame:
        """
        Alternate constructor to create a ``GeoDataFrame`` from a sql query
        containing a geometry column in WKB representation.

        Parameters
        ----------
        sql : string
        con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        geom_col : string, default 'geom'
            column name to convert to shapely geometries
        crs : optional
            Coordinate reference system to use for the returned GeoDataFrame
        index_col : string or list of strings, optional, default: None
            Column(s) to set as index(MultiIndex)
        coerce_float : boolean, default True
            Attempt to convert values of non-string, non-numeric objects (like
            decimal.Decimal) to floating point, useful for SQL result sets
        parse_dates : list or dict, default None
            - List of column names to parse as dates.
            - Dict of ``{column_name: format string}`` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of ``{column_name: arg dict}``, where the arg dict
              corresponds to the keyword arguments of
              :func:`pandas.to_datetime`. Especially useful with databases
              without native Datetime support, such as SQLite.
        params : list, tuple or dict, optional, default None
            List of parameters to pass to execute method.
        chunksize : int, default None
            If specified, return an iterator where chunksize is the number
            of rows to include in each chunk.

        Examples
        --------
        PostGIS

        >>> from sqlalchemy import create_engine  # doctest: +SKIP
        >>> db_connection_url = "postgresql://myusername:mypassword@myhost:5432/mydb"
        >>> con = create_engine(db_connection_url)  # doctest: +SKIP
        >>> sql = "SELECT geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)  # doctest: +SKIP

        SpatiaLite

        >>> sql = "SELECT ST_Binary(geom) AS geom, highway FROM roads"
        >>> df = geopandas.GeoDataFrame.from_postgis(sql, con)  # doctest: +SKIP

        The recommended method of reading from PostGIS is
        :func:`geopandas.read_postgis`:

        >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP

        See Also
        --------
        geopandas.read_postgis : read PostGIS database to GeoDataFrame
        """
        df = geopandas.io.sql._read_postgis(
            sql,
            con,
            geom_col=geom_col,
            crs=crs,
            index_col=index_col,
            coerce_float=coerce_float,
            parse_dates=parse_dates,
            params=params,
            chunksize=chunksize,
        )

        return df

    @classmethod
    def from_arrow(
        cls, table, geometry: str | None = None, to_pandas_kwargs: dict | None = None
    ):
        """
        Construct a GeoDataFrame from an Arrow table object based on GeoArrow
        extension types.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This functions accepts any tabular Arrow object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_array__``
        or ``__arrow_c_stream__`` method).

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        .. versionadded:: 1.0

        Parameters
        ----------
        table : pyarrow.Table or Arrow-compatible table
            Any tabular object implementing the Arrow PyCapsule Protocol
            (i.e. has an ``__arrow_c_array__`` or ``__arrow_c_stream__``
            method). This table should have at least one column with a
            geoarrow geometry type.
        geometry : str, default None
            The name of the geometry column to set as the active geometry
            column. If None, the first geometry column found will be used.
        to_pandas_kwargs : dict, optional
            Arguments passed to the `pa.Table.to_pandas` method for non-geometry
            columns. This can be used to control the behavior of the conversion of the
            non-geometry columns to a pandas DataFrame. For example, you can use this
            to control the dtype conversion of the columns. By default, the `to_pandas`
            method is called with no additional arguments.

        Returns
        -------
        GeoDataFrame

        See Also
        --------
        GeoDataFrame.to_arrow
        GeoSeries.from_arrow

        Examples
        --------
        >>> import geoarrow.pyarrow as ga
        >>> import pyarrow as pa
        >>> table = pa.Table.from_arrays([
        ...     ga.as_geoarrow(
        ...     [None, "POLYGON ((0 0, 1 1, 0 1, 0 0))", "LINESTRING (0 0, -1 1, 0 -1)"]
        ...     ),
        ...     pa.array([1, 2, 3]),
        ...     pa.array(["a", "b", "c"]),
        ... ], names=["geometry", "id", "value"])
        >>> gdf = geopandas.GeoDataFrame.from_arrow(table)
        >>> gdf
                                   geometry   id  value
        0                              None    1      a
        1    POLYGON ((0 0, 1 1, 0 1, 0 0))    2      b
        2      LINESTRING (0 0, -1 1, 0 -1)    3      c
        """
        from vibespatial.io.arrow import geodataframe_from_arrow

        return geodataframe_from_arrow(
            table,
            geometry=geometry,
            to_pandas_kwargs=to_pandas_kwargs,
        )

    def to_json(
        self,
        na: Literal["null", "drop", "keep"] = "null",
        show_bbox: bool = False,
        drop_id: bool = False,
        to_wgs84: bool = False,
        **kwargs,
    ) -> str:
        """Return a GeoJSON representation of the ``GeoDataFrame`` as a string.

        Parameters
        ----------
        na : {'null', 'drop', 'keep'}, default 'null'
            Indicates how to output missing (NaN) values in the GeoDataFrame.
            See below.
        show_bbox : bool, optional, default: False
            Include bbox (bounds) in the geojson
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.
        to_wgs84: bool, optional, default: False
            If the CRS is set on the active geometry column it is exported as
            WGS84 (EPSG:4326) to meet the `2016 GeoJSON specification
            <https://tools.ietf.org/html/rfc7946>`_.
            Set to True to force re-projection and set to False to ignore CRS. False by
            default.

        Notes
        -----
        The remaining *kwargs* are passed to json.dumps().

        Missing (NaN) values in the GeoDataFrame can be represented as follows:

        - ``null``: output the missing entries as JSON null.
        - ``drop``: remove the property from the feature. This applies to each
          feature individually so that features may have different properties.
        - ``keep``: output the missing entries as NaN.

        If the GeoDataFrame has a defined CRS, its definition will be included
        in the output unless it is equal to WGS84 (default GeoJSON CRS) or not
        possible to represent in the URN OGC format, or unless ``to_wgs84=True``
        is specified.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:3857")
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        >>> gdf.to_json()
        '{"type": "FeatureCollection", "features": [{"id": "0", "type": "Feature", \
"properties": {"col1": "name1"}, "geometry": {"type": "Point", "coordinates": [1.0,\
 2.0]}}, {"id": "1", "type": "Feature", "properties": {"col1": "name2"}, "geometry"\
: {"type": "Point", "coordinates": [2.0, 1.0]}}], "crs": {"type": "name", "properti\
es": {"name": "urn:ogc:def:crs:EPSG::3857"}}}'

        Alternatively, you can write GeoJSON to file:

        >>> gdf.to_file(path, driver="GeoJSON")  # doctest: +SKIP

        See Also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file

        """
        record_export = bool(kwargs.pop("_record_export", True))
        if record_export:
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_json",
                operation="geodataframe_to_json",
                target="geojson",
                reason="native GeoDataFrame exported to GeoJSON string",
            )
        if to_wgs84:
            if self.crs:
                df = self.to_crs(epsg=4326)
            else:
                raise ValueError(
                    "CRS is not set. Cannot re-project to WGS84 (EPSG:4326)."
                )
        else:
            df = self

        geo = df.to_geo_dict(
            na=na,
            show_bbox=show_bbox,
            drop_id=drop_id,
            _record_export=False,
        )

        # if the geometry is not in WGS84, include CRS in the JSON
        if df.crs is not None and not df.crs.equals("epsg:4326"):
            auth_crsdef = self.crs.to_authority()
            allowed_authorities = ["EDCS", "EPSG", "OGC", "SI", "UCUM"]

            if auth_crsdef is None or auth_crsdef[0] not in allowed_authorities:
                warnings.warn(
                    "GeoDataFrame's CRS is not representable in URN OGC "
                    "format. Resulting JSON will contain no CRS information.",
                    stacklevel=2,
                )
            else:
                authority, code = auth_crsdef
                ogc_crs = f"urn:ogc:def:crs:{authority}::{code}"
                geo["crs"] = {"type": "name", "properties": {"name": ogc_crs}}

        return json.dumps(geo, **kwargs)

    @property
    def __geo_interface__(self) -> dict:
        """Returns a ``GeoDataFrame`` as a python feature collection.

        Implements the `geo_interface`. The returned python data structure
        represents the ``GeoDataFrame`` as a GeoJSON-like
        ``FeatureCollection``.

        This differs from :meth:`to_geo_dict` only in that it is a property with
        default args instead of a method.

        CRS of the dataframe is not passed on to the output, unlike
        :meth:`~GeoDataFrame.to_json()`.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        >>> gdf.__geo_interface__
        {'type': 'FeatureCollection', 'features': [{'id': '0', 'type': 'Feature', \
'properties': {'col1': 'name1'}, 'geometry': {'type': 'Point', 'coordinates': (1.0\
, 2.0)}, 'bbox': (1.0, 2.0, 1.0, 2.0)}, {'id': '1', 'type': 'Feature', 'properties\
	': {'col1': 'name2'}, 'geometry': {'type': 'Point', 'coordinates': (2.0, 1.0)}, 'b\
	box': (2.0, 1.0, 2.0, 1.0)}], 'bbox': (1.0, 1.0, 2.0, 2.0)}
        """
        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.__geo_interface__",
            operation="geodataframe_geo_interface",
            target="geo-interface",
            reason="native GeoDataFrame exported to Python geo interface",
        )
        return self.to_geo_dict(
            na="null",
            show_bbox=True,
            drop_id=False,
            _record_export=False,
        )

    def iterfeatures(
        self,
        na: str = "null",
        show_bbox: bool = False,
        drop_id: bool = False,
        *,
        _record_export: bool = True,
    ) -> typing.Generator[dict]:
        """Return an iterator that yields feature dictionaries that comply with
        __geo_interface__.

        Parameters
        ----------
        na : str, optional
            Options are {'null', 'drop', 'keep'}, default 'null'.
            Indicates how to output missing (NaN) values in the GeoDataFrame

            - null: output the missing entries as JSON null
            - drop: remove the property from the feature. This applies to each feature \
individually so that features may have different properties
            - keep: output the missing entries as NaN

        show_bbox : bool, optional
            Include bbox (bounds) in the geojson. Default False.
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated GeoJSON. Default is False, but may want True
            if the index is just arbitrary row numbers.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        >>> feature = next(gdf.iterfeatures())
        >>> feature
        {'id': '0', 'type': 'Feature', 'properties': {'col1': 'name1'}, 'geometry': {\
'type': 'Point', 'coordinates': (1.0, 2.0)}}
        """
        if _record_export:
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.iterfeatures",
                operation="geodataframe_iterfeatures",
                target="geo-features",
                reason="native GeoDataFrame exported to Python feature dictionaries",
            )
        if na not in ["null", "drop", "keep"]:
            raise ValueError(f"Unknown na method {na}")

        ids = np.asarray(self.index)
        geometries = np.asarray(self.geometry)

        if not self.columns.is_unique:
            raise ValueError("GeoDataFrame cannot contain duplicated column names.")

        properties_cols = self.columns.drop(self._geometry_column_name)

        if len(properties_cols) > 0:
            # convert to object to get python scalars.
            properties_cols = self[properties_cols]
            properties = properties_cols.astype(object)
            na_mask = pd.isna(properties_cols).values

            if na == "null":
                properties[na_mask] = None

            for i, row in enumerate(properties.values):
                geom = geometries[i]

                if na == "drop":
                    na_mask_row = na_mask[i]
                    properties_items = {
                        k: v
                        for k, v, na in zip(properties_cols, row, na_mask_row)
                        if not na
                    }
                else:
                    properties_items = dict(zip(properties_cols, row))

                if drop_id:
                    feature = {}
                else:
                    feature = {"id": str(ids[i])}

                feature["type"] = "Feature"
                feature["properties"] = properties_items
                feature["geometry"] = mapping(geom) if geom else None

                if show_bbox:
                    feature["bbox"] = geom.bounds if geom else None

                yield feature

        else:
            for fid, geom in zip(ids, geometries):
                if drop_id:
                    feature = {}
                else:
                    feature = {"id": str(fid)}

                feature["type"] = "Feature"
                feature["properties"] = {}
                feature["geometry"] = mapping(geom) if geom else None

                if show_bbox:
                    feature["bbox"] = geom.bounds if geom else None

                yield feature

    def to_geo_dict(
        self,
        na: str | None = "null",
        show_bbox: bool = False,
        drop_id: bool = False,
        *,
        _record_export: bool = True,
    ) -> dict:
        """Return a python feature collection representation of the GeoDataFrame
        as a dictionary with a list of features based on the ``__geo_interface__``
        GeoJSON-like specification.

        Parameters
        ----------
        na : str, optional
            Options are {'null', 'drop', 'keep'}, default 'null'.
            Indicates how to output missing (NaN) values in the GeoDataFrame

            - null: output the missing entries as JSON null
            - drop: remove the property from the feature. This applies to each feature \
individually so that features may have different properties
            - keep: output the missing entries as NaN

        show_bbox : bool, optional
            Include bbox (bounds) in the geojson. Default False.
        drop_id : bool, default: False
            Whether to retain the index of the GeoDataFrame as the id property
            in the generated dictionary. Default is False, but may want True
            if the index is just arbitrary row numbers.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        >>> gdf.to_geo_dict()
        {'type': 'FeatureCollection', 'features': [{'id': '0', 'type': 'Feature', '\
properties': {'col1': 'name1'}, 'geometry': {'type': 'Point', 'coordinates': (1.0, \
2.0)}}, {'id': '1', 'type': 'Feature', 'properties': {'col1': 'name2'}, 'geometry':\
 {'type': 'Point', 'coordinates': (2.0, 1.0)}}]}

        See Also
        --------
        GeoDataFrame.to_json : return a GeoDataFrame as a GeoJSON string

        """
        if _record_export:
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_geo_dict",
                operation="geodataframe_to_geo_dict",
                target="geo-dict",
                reason="native GeoDataFrame exported to Python geo dictionary",
            )
        geo = {
            "type": "FeatureCollection",
            "features": list(
                self.iterfeatures(
                    na=na,
                    show_bbox=show_bbox,
                    drop_id=drop_id,
                    _record_export=False,
                )
            ),
        }

        if show_bbox:
            geo["bbox"] = tuple(self.total_bounds.tolist())  # tolist to avoid np dtypes

        return geo

    def to_wkb(self, hex: bool = False, **kwargs) -> pd.DataFrame:
        """
        Encode all geometry columns in the GeoDataFrame to WKB.

        Parameters
        ----------
        hex : bool
            If true, export the WKB as a hexadecimal string.
            The default is to return a binary bytes object.
        kwargs
            Additional keyword args will be passed to
            :func:`shapely.to_wkb`.

        Returns
        -------
        DataFrame
            geometry columns are encoded to WKB
        """
        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.to_wkb",
            operation="geodataframe_to_wkb",
            target="wkb-dataframe",
            reason="native GeoDataFrame exported to WKB DataFrame",
        )
        df = DataFrame(self.copy(deep=not PANDAS_GE_30))

        # Encode all geometry columns to WKB
        for col in df.columns[np.asarray(df.dtypes.map(_is_geometry_like_dtype), dtype=bool)]:
            df[col] = to_wkb(df[col].values, hex=hex, **kwargs)

        return df

    def to_wkt(self, **kwargs) -> pd.DataFrame:
        """
        Encode all geometry columns in the GeoDataFrame to WKT.

        Parameters
        ----------
        kwargs
            Keyword args will be passed to :func:`shapely.to_wkt`.

        Returns
        -------
        DataFrame
            geometry columns are encoded to WKT
        """
        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.to_wkt",
            operation="geodataframe_to_wkt",
            target="wkt-dataframe",
            reason="native GeoDataFrame exported to WKT DataFrame",
        )
        df = DataFrame(self.copy(deep=not PANDAS_GE_30))

        # Encode all geometry columns to WKT
        for col in df.columns[np.asarray(df.dtypes.map(_is_geometry_like_dtype), dtype=bool)]:
            df[col] = to_wkt(df[col].values, **kwargs)

        return df

    def to_arrow(
        self,
        *,
        index: bool | None = None,
        geometry_encoding: PARQUET_GEOMETRY_ENCODINGS = "WKB",
        interleaved: bool = True,
        include_z: bool | None = None,
    ):
        """Encode a GeoDataFrame to GeoArrow format.

        See https://geoarrow.org/ for details on the GeoArrow specification.

        This function returns a generic Arrow data object implementing
        the `Arrow PyCapsule Protocol`_ (i.e. having an ``__arrow_c_stream__``
        method). This object can then be consumed by your Arrow implementation
        of choice that supports this protocol.

        .. _Arrow PyCapsule Protocol: https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

        .. versionadded:: 1.0

        Parameters
        ----------
        index : bool, default None
            If ``True``, always include the dataframe's index(es) as columns
            in the file output.
            If ``False``, the index(es) will not be written to the file.
            If ``None``, the index(ex) will be included as columns in the file
            output except `RangeIndex` which is stored as metadata only.
        geometry_encoding : {'WKB', 'geoarrow' }, default 'WKB'
            The GeoArrow encoding to use for the data conversion.
        interleaved : bool, default True
            Only relevant for 'geoarrow' encoding. If True, the geometries'
            coordinates are interleaved in a single fixed size list array.
            If False, the coordinates are stored as separate arrays in a
            struct type.
        include_z : bool, default None
            Only relevant for 'geoarrow' encoding (for WKB, the dimensionality
            of the individual geometries is preserved).
            If False, return 2D geometries. If True, include the third dimension
            in the output (if a geometry has no third dimension, the z-coordinates
            will be NaN). By default, will infer the dimensionality from the
            input geometries. Note that this inference can be unreliable with
            empty geometries (for a guaranteed result, it is recommended to
            specify the keyword).

        Returns
        -------
        ArrowTable
            A generic Arrow table object with geometry columns encoded to
            GeoArrow.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> data = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(data)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        >>> arrow_table = gdf.to_arrow()
        >>> arrow_table
        <geopandas.io._geoarrow.ArrowTable object at ...>

        The returned data object needs to be consumed by a library implementing
        the Arrow PyCapsule Protocol. For example, wrapping the data as a
        pyarrow.Table (requires pyarrow >= 14.0):

        >>> import pyarrow as pa
        >>> table = pa.table(arrow_table)
        >>> table
        pyarrow.Table
        col1: large_string
        geometry: extension<geoarrow.wkb<WkbType>>
        ----
        col1: [["name1","name2"]]
        geometry: [[0101000000000000000000F03F0000000000000040,\
01010000000000000000000040000000000000F03F]]

        """
        from vibespatial.api._native_state import get_native_state

        native_state = get_native_state(self)
        if native_state is not None:
            native_result = _native_tabular_result_with_public_metadata(
                self,
                native_state,
                index=index,
                preserve_native_attributes=True,
            )
            result = native_result.to_arrow(
                index=index,
                geometry_encoding=geometry_encoding,
                interleaved=interleaved,
                include_z=include_z,
                record_export_boundary=False,
            )
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_arrow",
                operation="geodataframe_to_arrow",
                target="arrow",
                reason="native GeoDataFrame exported to Arrow compatibility object",
            )
            return result

        from vibespatial.io.arrow import geodataframe_to_arrow

        result = geodataframe_to_arrow(
            self,
            index=index,
            geometry_encoding=geometry_encoding,
            interleaved=interleaved,
            include_z=include_z,
            record_export_boundary=False,
            fallback_to_wkb_on_error=False,
        )
        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.to_arrow",
            operation="geodataframe_to_arrow",
            target="arrow",
            reason="native GeoDataFrame exported to Arrow compatibility object",
        )
        return result

    def to_parquet(
        self,
        path: os.PathLike | typing.IO,
        index: bool | None = None,
        compression: str | None = "snappy",
        geometry_encoding: PARQUET_GEOMETRY_ENCODINGS = "WKB",
        write_covering_bbox: bool = False,
        schema_version: SUPPORTED_VERSIONS_LITERAL | None = None,
        **kwargs,
    ) -> None:
        """Write a GeoDataFrame to the Parquet format.

        By default, all geometry columns present are serialized to WKB format
        in the file.

        Requires 'pyarrow'.

        .. versionadded:: 0.8

        Parameters
        ----------
        path : str, path object
        index : bool, default None
            If ``True``, always include the dataframe's index(es) as columns
            in the file output.
            If ``False``, the index(es) will not be written to the file.
            If ``None``, the index(ex) will be included as columns in the file
            output except `RangeIndex` which is stored as metadata only.
        compression : {'snappy', 'gzip', 'brotli', 'lz4', 'zstd', None}, \
default 'snappy'
            Name of the compression to use. Use ``None`` for no compression.
        geometry_encoding : {'WKB', 'geoarrow'}, default 'WKB'
            The encoding to use for the geometry columns. Defaults to "WKB"
            for maximum interoperability. Specify "geoarrow" to use one of the
            native GeoArrow-based single-geometry type encodings.
            Note: the "geoarrow" option is part of the newer GeoParquet 1.1
            specification, should be considered as experimental, and may not
            be supported by all readers.
        write_covering_bbox : bool, default False
            Writes the bounding box column for each row entry with column
            name 'bbox'. Writing a bbox column can be computationally
            expensive, but allows you to specify a `bbox` in :
            func:`read_parquet` for filtered reading.
            Note: this bbox column is part of the newer GeoParquet 1.1
            specification and should be considered as experimental. While
            writing the column is backwards compatible, using it for filtering
            may not be supported by all readers.
        schema_version : {'0.1.0', '0.4.0', '1.0.0', '1.1.0', None}
            GeoParquet specification version; if not provided, will default to
            latest supported stable version (1.0.0).
        kwargs
            Additional keyword arguments passed to :func:`pyarrow.parquet.write_table`.

        Examples
        --------
        >>> gdf.to_parquet('data.parquet')  # doctest: +SKIP

        See Also
        --------
        GeoDataFrame.to_feather : write GeoDataFrame to feather
        GeoDataFrame.to_file : write GeoDataFrame to file
        """
        # Accept engine keyword for compatibility with pandas.DataFrame.to_parquet
        # The only engine currently supported by GeoPandas is pyarrow, so no
        # other engine should be specified.
        engine = kwargs.pop("engine", "auto")
        if engine not in ("auto", "pyarrow"):
            raise ValueError(
                "GeoPandas only supports using pyarrow as the engine for "
                f"to_parquet: {engine!r} passed instead."
            )

        from vibespatial.api._native_state import get_native_state

        native_state = get_native_state(self)
        if native_state is not None:
            native_result = _native_tabular_result_with_public_metadata(
                self,
                native_state,
                index=index,
                preserve_native_attributes=True,
            )
            native_result.to_parquet(
                path,
                compression=compression,
                geometry_encoding=geometry_encoding,
                index=index,
                schema_version=schema_version,
                write_covering_bbox=write_covering_bbox,
                record_export_boundary=False,
                **kwargs,
            )
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_parquet",
                operation="geodataframe_to_parquet",
                target="geoparquet",
                reason="native GeoDataFrame exported to GeoParquet writer",
                d2h_transfer=False,
            )
            return

        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.to_parquet",
            operation="geodataframe_to_parquet",
            target="geoparquet",
            reason="native GeoDataFrame exported to GeoParquet writer",
        )
        from vibespatial.io.arrow import write_geoparquet

        write_geoparquet(
            self,
            path,
            compression=compression,
            geometry_encoding=geometry_encoding,
            index=index,
            schema_version=schema_version,
            write_covering_bbox=write_covering_bbox,
            **kwargs,
        )

    def to_feather(
        self,
        path: os.PathLike,
        index: bool | None = None,
        compression: str | None = None,
        schema_version: SUPPORTED_VERSIONS_LITERAL | None = None,
        **kwargs,
    ):
        """Write a GeoDataFrame to the Feather format.

        Any geometry columns present are serialized to WKB format in the file.

        Requires 'pyarrow' >= 0.17.

        .. versionadded:: 0.8

        Parameters
        ----------
        path : str, path object
        index : bool, default None
            If ``True``, always include the dataframe's index(es) as columns
            in the file output.
            If ``False``, the index(es) will not be written to the file.
            If ``None``, the index(ex) will be included as columns in the file
            output except `RangeIndex` which is stored as metadata only.
        compression : {'zstd', 'lz4', 'uncompressed'}, optional
            Name of the compression to use. Use ``"uncompressed"`` for no
            compression. By default uses LZ4 if available, otherwise uncompressed.
        schema_version : {'0.1.0', '0.4.0', '1.0.0', '1.1.0' None}
            GeoParquet specification version; if not provided will default to
            latest supported stable version (1.0.0).
        kwargs
            Additional keyword arguments passed to
            :func:`pyarrow.feather.write_feather`.

        Examples
        --------
        >>> gdf.to_feather('data.feather')  # doctest: +SKIP

        See Also
        --------
        GeoDataFrame.to_parquet : write GeoDataFrame to parquet
        GeoDataFrame.to_file : write GeoDataFrame to file
        """
        from vibespatial.api._native_state import get_native_state

        native_state = get_native_state(self)
        if native_state is not None:
            native_result = _native_tabular_result_with_public_metadata(
                self,
                native_state,
                index=index,
            )
            native_result.to_feather(
                path,
                index=index,
                compression=compression,
                schema_version=schema_version,
                record_export_boundary=False,
                **kwargs,
            )
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_feather",
                operation="geodataframe_to_feather",
                target="feather",
                reason="native GeoDataFrame exported to Feather writer",
            )
            return

        _record_native_public_export_boundary(
            self,
            surface="vibespatial.api.GeoDataFrame.to_feather",
            operation="geodataframe_to_feather",
            target="feather",
            reason="native GeoDataFrame exported to Feather writer",
        )
        from vibespatial.api.io.arrow import _to_feather

        _to_feather(
            self,
            path,
            index=index,
            compression=compression,
            schema_version=schema_version,
            **kwargs,
        )

    def to_file(
        self,
        filename: os.PathLike | typing.IO,
        driver: str | None = None,
        schema: dict | None = None,
        index: bool | None = None,
        **kwargs,
    ):
        """Write the ``GeoDataFrame`` to a file.

        By default, an ESRI shapefile is written, but any OGR data source
        supported by Pyogrio or Fiona can be written. A dictionary of supported OGR
        providers is available via:

        >>> import pyogrio
        >>> pyogrio.list_drivers()  # doctest: +SKIP

        Parameters
        ----------
        filename : string
            File path or file handle to write to. The path may specify a
            GDAL VSI scheme.
        driver : string, default None
            The OGR format driver used to write the vector file.
            If not specified, it attempts to infer it from the file extension.
            If no extension is specified, it saves ESRI Shapefile to a folder.
        schema : dict, default None
            If specified, the schema dictionary is passed to Fiona to
            better control how the file is written. If None, GeoPandas
            will determine the schema based on each column's dtype.
            Not supported for the "pyogrio" engine.
        index : bool, default None
            If True, write index into one or more columns (for MultiIndex).
            Default None writes the index into one or more columns only if
            the index is named, is a MultiIndex, or has a non-integer data
            type. If False, no index is written.

            .. versionadded:: 0.7
                Previously the index was not written.
        mode : string, default 'w'
            The write mode, 'w' to overwrite the existing file and 'a' to append.
            Not all drivers support appending. The drivers that support appending
            are listed in fiona.supported_drivers or
            https://github.com/Toblerity/Fiona/blob/master/fiona/drvsupport.py
        crs : pyproj.CRS, default None
            If specified, the CRS is passed to Fiona to
            better control how the file is written. If None, GeoPandas
            will determine the crs based on crs df attribute.
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string. The keyword
            is not supported for the "pyogrio" engine.
        engine : str, "pyogrio" or "fiona"
            The underlying library that is used to write the file. Currently, the
            supported options are "pyogrio" and "fiona". Defaults to "pyogrio" if
            installed, otherwise tries "fiona".
        metadata : dict[str, str], default None
            Optional metadata to be stored in the file. Keys and values must be
            strings. Supported only for "GPKG" driver.
        **kwargs :
            Keyword args to be passed to the engine, and can be used to write
            to multi-layer data, store data within archives (zip files), etc.
            In case of the "pyogrio" engine, the keyword arguments are passed to
            `pyogrio.write_dataframe`. In case of the "fiona" engine, the keyword
            arguments are passed to fiona.open`. For more information on possible
            keywords, type: ``import pyogrio; help(pyogrio.write_dataframe)``.

        Notes
        -----
        The format drivers will attempt to detect the encoding of your data, but
        may fail. In this case, the proper encoding can be specified explicitly
        by using the encoding keyword parameter, e.g. ``encoding='utf-8'``.

        See Also
        --------
        GeoSeries.to_file
        GeoDataFrame.to_postgis : write GeoDataFrame to PostGIS database
        GeoDataFrame.to_parquet : write GeoDataFrame to parquet
        GeoDataFrame.to_feather : write GeoDataFrame to feather

        Examples
        --------
        >>> gdf.to_file('dataframe.shp')  # doctest: +SKIP

        >>> gdf.to_file('dataframe.gpkg', driver='GPKG', layer='name')  # doctest: +SKIP

        >>> gdf.to_file('dataframe.geojson', driver='GeoJSON')  # doctest: +SKIP

        With selected drivers you can also append to a file with `mode="a"`:

        >>> gdf.to_file('dataframe.shp', mode="a")  # doctest: +SKIP

        Using the engine-specific keyword arguments it is possible to e.g. create a
        spatialite file with a custom layer name:

        >>> gdf.to_file(
        ...     'dataframe.sqlite', driver='SQLite', spatialite=True, layer='test'
        ... )  # doctest: +SKIP

        """
        from vibespatial.io.file import write_vector_file

        record_export = bool(kwargs.pop("_record_export", True))
        if record_export:
            _record_native_public_export_boundary(
                self,
                surface="vibespatial.api.GeoDataFrame.to_file",
                operation="geodataframe_to_file",
                target="file",
                reason="native GeoDataFrame exported to vector file writer",
            )
        write_vector_file(self, filename, driver, schema, index, **kwargs)

    @typing.overload
    def set_crs(
        self,
        crs: Any | None = ...,
        epsg: int | None = ...,
        inplace: Literal[True] = ...,
        allow_override: bool = ...,
    ) -> None: ...

    @typing.overload
    def set_crs(
        self,
        crs: Any | None = ...,
        epsg: int | None = ...,
        inplace: Literal[False] = ...,
        allow_override: bool = ...,
    ) -> GeoDataFrame: ...

    def set_crs(
        self,
        crs: Any | None = None,
        epsg: int | None = None,
        inplace: bool = False,
        allow_override: bool = False,
    ) -> GeoDataFrame | None:
        """
        Set the Coordinate Reference System (CRS) of the ``GeoDataFrame``.

        If there are multiple geometry columns within the GeoDataFrame, only
        the CRS of the active geometry column is set.

        Pass ``None`` to remove CRS from the active geometry column.

        Notes
        -----
        The underlying geometries are not transformed to this CRS. To
        transform the geometries to a new CRS, use the ``to_crs`` method.

        Parameters
        ----------
        crs : pyproj.CRS | None, optional
            The value can be anything accepted
            by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional
            EPSG code specifying the projection.
        inplace : bool, default False
            If True, the CRS of the GeoDataFrame will be changed in place
            (while still returning the result) instead of making a copy of
            the GeoDataFrame.
        allow_override : bool, default False
            If the the GeoDataFrame already has a CRS, allow to replace the
            existing CRS, even when both are not equal.

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)

        Setting CRS to a GeoDataFrame without one:

        >>> gdf.crs is None
        True

        >>> gdf = gdf.set_crs('epsg:3857')
        >>> gdf.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        Overriding existing CRS:

        >>> gdf = gdf.set_crs(4326, allow_override=True)

        Without ``allow_override=True``, ``set_crs`` returns an error if you try to
        override CRS.

        See Also
        --------
        GeoDataFrame.to_crs : re-project to another CRS

        """
        from vibespatial.api._native_state import attach_native_state, get_native_state

        source_native_state = get_native_state(self)
        if not inplace:
            df = self.copy(deep=not PANDAS_GE_30)
        else:
            df = self
        df.geometry = df.geometry.set_crs(
            crs=crs, epsg=epsg, allow_override=allow_override, inplace=True
        )
        if (
            source_native_state is not None
            and len(df) == source_native_state.row_count
            and tuple(df.columns) == tuple(self.columns)
            and df.index.equals(self.index)
            and getattr(df, "_geometry_column_name", None) == source_native_state.geometry_name
        ):
            attach_native_state(df, source_native_state.with_geometry_crs(df.crs))
        return df

    @typing.overload
    def to_crs(
        self,
        crs: Any | None = ...,
        epsg: int | None = ...,
        inplace: Literal[False] = ...,
    ) -> GeoDataFrame: ...

    @typing.overload
    def to_crs(
        self,
        crs: Any | None = ...,
        epsg: int | None = ...,
        inplace: Literal[True] = ...,
    ) -> None: ...

    def to_crs(
        self,
        crs: Any | None = None,
        epsg: int | None = None,
        inplace: bool = False,
    ) -> GeoDataFrame | None:
        """Transform geometries to a new coordinate reference system.

        Transform all geometries in an active geometry column to a different coordinate
        reference system.  The ``crs`` attribute on the current GeoSeries must
        be set.  Either ``crs`` or ``epsg`` may be specified for output.

        This method will transform all points in all objects. It has no notion
        of projecting entire geometries.  All segments joining points are
        assumed to be lines in the current projection, not geodesics. Objects
        crossing the dateline (or other projection boundary) will have
        undesirable behavior.

        Parameters
        ----------
        crs : pyproj.CRS, optional if `epsg` is specified
            The value can be anything accepted by
            :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
            such as an authority string (eg "EPSG:4326") or a WKT string.
        epsg : int, optional if `crs` is specified
            EPSG code specifying output projection.
        inplace : bool, optional, default: False
            Whether to return a new GeoDataFrame or do the transformation in
            place.

        Returns
        -------
        GeoDataFrame

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {'col1': ['name1', 'name2'], 'geometry': [Point(1, 2), Point(2, 1)]}
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)
        >>> gdf.crs  # doctest: +SKIP
        <Geographic 2D CRS: EPSG:4326>
        Name: WGS 84
        Axis Info [ellipsoidal]:
        - Lat[north]: Geodetic latitude (degree)
        - Lon[east]: Geodetic longitude (degree)
        Area of Use:
        - name: World
        - bounds: (-180.0, -90.0, 180.0, 90.0)
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        >>> gdf = gdf.to_crs(3857)
        >>> gdf
            col1                       geometry
        0  name1  POINT (111319.491 222684.209)
        1  name2  POINT (222638.982 111325.143)
        >>> gdf.crs  # doctest: +SKIP
        <Projected CRS: EPSG:3857>
        Name: WGS 84 / Pseudo-Mercator
        Axis Info [cartesian]:
        - X[east]: Easting (metre)
        - Y[north]: Northing (metre)
        Area of Use:
        - name: World - 85°S to 85°N
        - bounds: (-180.0, -85.06, 180.0, 85.06)
        Coordinate Operation:
        - name: Popular Visualisation Pseudo-Mercator
        - method: Popular Visualisation Pseudo Mercator
        Datum: World Geodetic System 1984
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich

        See Also
        --------
        GeoDataFrame.set_crs : assign CRS without re-projection
        """
        from vibespatial.api._native_state import (
            attach_native_state,
            drop_native_state,
            get_native_state,
        )

        source_native_state = get_native_state(self)
        if inplace:
            df = self
        else:
            df = self.copy(deep=not PANDAS_GE_30)
        geom = df.geometry.to_crs(crs=crs, epsg=epsg)
        if inplace:
            df.geometry = geom
        else:
            from vibespatial.api._native_results import _replace_geometry_column_preserving_backing

            df = _replace_geometry_column_preserving_backing(
                df,
                geom.values,
                crs=geom.crs,
            )
        if (
            source_native_state is not None
            and len(df) == source_native_state.row_count
            and tuple(df.columns) == source_native_state.column_order
            and df.index.equals(self.index)
            and getattr(df, "_geometry_column_name", None) == source_native_state.geometry_name
        ):
            from vibespatial.api._native_result_core import GeometryNativeResult

            geometry = GeometryNativeResult.from_geoseries(geom)
            attach_native_state(df, source_native_state.with_geometry_result(geometry))
        else:
            drop_native_state(df)
        if not inplace:
            return df

    def estimate_utm_crs(self, datum_name: str = "WGS 84") -> CRS:
        """Return the estimated UTM CRS based on the bounds of the dataset.

        .. versionadded:: 0.9

        Parameters
        ----------
        datum_name : str, optional
            The name of the datum to use in the query. Default is WGS 84.

        Returns
        -------
        pyproj.CRS

        Examples
        --------
        >>> import geodatasets
        >>> df = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... )
        >>> df.estimate_utm_crs()  # doctest: +SKIP
        <Derived Projected CRS: EPSG:32616>
        Name: WGS 84 / UTM zone 16N
        Axis Info [cartesian]:
        - E[east]: Easting (metre)
        - N[north]: Northing (metre)
        Area of Use:
        - name: Between 90°W and 84°W, northern hemisphere between equator and 84°N...
        - bounds: (-90.0, 0.0, -84.0, 84.0)
        Coordinate Operation:
        - name: UTM zone 16N
        - method: Transverse Mercator
        Datum: World Geodetic System 1984 ensemble
        - Ellipsoid: WGS 84
        - Prime Meridian: Greenwich
        """
        return self.geometry.estimate_utm_crs(datum_name=datum_name)

    @property
    def loc(self):
        return _NativeStateInvalidatingIndexer(
            pd.DataFrame.loc.fget(self),
            self,
            kind="loc",
        )

    @property
    def iloc(self):
        return _NativeStateInvalidatingIndexer(
            pd.DataFrame.iloc.fget(self),
            self,
            kind="iloc",
        )

    @property
    def at(self):
        return _NativeStateInvalidatingIndexer(
            pd.DataFrame.at.fget(self),
            self,
            kind="at",
        )

    @property
    def iat(self):
        return _NativeStateInvalidatingIndexer(
            pd.DataFrame.iat.fget(self),
            self,
            kind="iat",
        )

    def __getitem__(self, key):
        """
        If the result is a column containing only 'geometry', return a
        GeoSeries. If it's a DataFrame with any columns of GeometryDtype,
        return a GeoDataFrame.
        """
        from vibespatial.api._native_rowset import NativeRowSet
        from vibespatial.api._native_state import (
            attach_native_state,
            get_native_state,
        )

        source_native_state = get_native_state(self)
        geo_col = self._geometry_column_name
        if isinstance(key, NativeRowSet):
            if source_native_state is None:
                return self.take(key.to_host_positions(strict_disallowed=False))
            taken_state = source_native_state.take(key)
            result = taken_state.to_native_tabular_result().to_geodataframe()
            attach_native_state(result, taken_state)
            return result
        if source_native_state is not None:
            rowset = None
            native_mask_rowset = None
            if _is_native_state_column_projection_key(key, self.columns):
                projected = source_native_state.project_columns(tuple(key))
                if projected is not None:
                    result = _public_frame_from_native_state(
                        self,
                        projected,
                        geometry_column=geo_col,
                    )
                    if result is not None:
                        return result
            if isinstance(key, Series) and key.index.equals(self.index):
                rowset = _native_boolean_filter_rowset(
                    key,
                    source_native_state,
                    public_index=self.index,
                )
                native_mask_rowset = _native_boolean_rowset_from_aligned_mask(
                    key,
                    source_native_state,
                )
                if (
                    rowset is None
                    and native_mask_rowset is not None
                    and (
                        not native_mask_rowset.is_device
                        or _native_state_can_take_device_row_positions(
                            source_native_state,
                        )
                    )
                ):
                    rowset = _native_rowset_mark_identity_if_full(
                        native_mask_rowset,
                        source_native_state,
                    )
            elif not isinstance(key, (str, bytes)):
                native_mask_rowset = _native_boolean_rowset_from_aligned_mask(
                    key,
                    source_native_state,
                )
                if (
                    native_mask_rowset is not None
                    and (
                        not native_mask_rowset.is_device
                        or _native_state_can_take_device_row_positions(
                            source_native_state,
                        )
                    )
                ):
                    rowset = _native_rowset_mark_identity_if_full(
                        native_mask_rowset,
                        source_native_state,
                    )
            attributes = getattr(source_native_state, "attributes", None)
            if (
                rowset is not None
                and not bool(getattr(rowset, "identity", False))
                and attributes is not None
            ):
                positions = None
                if (
                    rowset.is_device
                    and _native_state_can_take_device_row_positions(
                        source_native_state,
                    )
                ):
                    index_positions = None
                    if isinstance(key, Series) and (
                        native_mask_rowset is None or not native_mask_rowset.is_device
                    ):
                        mask_values = key.to_numpy(dtype=bool, na_value=False)
                        if (
                            mask_values.ndim == 1
                            and int(mask_values.sum()) == len(rowset)
                        ):
                            index_positions = np.flatnonzero(mask_values).astype(
                                np.int64,
                                copy=False,
                            )
                    result = _take_public_frame_with_native_state(
                        self,
                        rowset,
                        source_native_state=source_native_state,
                        geometry_column=geo_col,
                        index_positions=index_positions,
                    )
                    if result is not None:
                        return result
                if isinstance(key, Series) and not (
                    native_mask_rowset is not None and native_mask_rowset.is_device
                ):
                    mask_values = key.to_numpy(dtype=bool, na_value=False)
                    if (
                        mask_values.ndim == 1
                        and int(mask_values.sum()) == len(rowset)
                    ):
                        positions = np.flatnonzero(mask_values).astype(
                            np.int64,
                            copy=False,
                        )
                if positions is None and native_mask_rowset is None:
                    mask_values = key.to_numpy(dtype=bool, na_value=False)
                    if (
                        mask_values.ndim != 1
                        or int(mask_values.sum()) != len(rowset)
                    ):
                        positions = None
                    else:
                        positions = np.flatnonzero(mask_values).astype(
                            np.int64,
                            copy=False,
                        )
                elif positions is None:
                    positions = rowset.to_host_positions(
                        surface="vibespatial.api.GeoDataFrame.__getitem__",
                        strict_disallowed=False,
                    )
                if positions is not None:
                    result = _take_public_frame_with_native_rowset(
                        self,
                        rowset,
                        positions,
                        source_native_state=source_native_state,
                        geometry_column=geo_col,
                    )
                    if result is not None:
                        return result
            elif rowset is not None and bool(getattr(rowset, "identity", False)):
                positions = np.arange(source_native_state.row_count, dtype=np.int64)
                result = _take_public_frame_with_native_rowset(
                    self,
                    rowset,
                    positions,
                    source_native_state=source_native_state,
                    geometry_column=geo_col,
                )
                if result is not None:
                    return result

        result = super().__getitem__(key)
        # Custom logic to avoid waiting for pandas GH51895
        # result is not geometry dtype for multi-indexes
        if (
            pd.api.types.is_scalar(key)
            and key == ""
            and isinstance(self.columns, pd.MultiIndex)
            and isinstance(result, Series)
            and not is_geometry_type(result)
        ):
            loc = self.columns.get_loc(key)
            # squeeze stops multilevel columns from returning a gdf
            result = self.iloc[:, loc].squeeze(axis="columns")
        if isinstance(result, Series) and _is_geometry_like_dtype(result.dtype):
            result.__class__ = GeoSeries
            if source_native_state is not None and pd.api.types.is_scalar(key):
                selected_state = source_native_state.with_active_geometry(
                    key,
                    crs=getattr(result, "crs", None),
                )
                if (
                    selected_state is not None
                    and len(result) == selected_state.row_count
                    and result.index.equals(self.index)
                ):
                    attach_native_state(result, selected_state)
        elif (
            isinstance(result, Series)
            and source_native_state is not None
            and pd.api.types.is_scalar(key)
            and result.index.equals(self.index)
            and len(result) == source_native_state.row_count
        ):
            from vibespatial.api._native_public_arrays import (
                NativeAttributeColumnArray,
                NativeNumericExpressionArray,
            )

            public_array = getattr(result, "array", None)
            if isinstance(
                public_array,
                NativeNumericExpressionArray,
            ) and _native_column_requires_numpy_public_dtype(
                source_native_state.attributes,
                key,
            ):
                result = Series(
                    public_array.to_numpy(copy=False),
                    index=result.index,
                    name=result.name,
                )
                public_array = result.array
            if isinstance(public_array, NativeAttributeColumnArray):
                # GeoParquet decimal attributes are exposed lazily because they
                # cannot be viewed as a numeric CuPy array until a public cast.
                # Bind that later cast to this frame's lineage so it can compose
                # with geometry-derived expressions without a full host export.
                public_array = public_array.copy()
                public_array.source_token = source_native_state.lineage_token
                result = Series(public_array, index=result.index, name=result.name)
            expression = source_native_state.attribute_expression(key)
            if expression is not None:
                _attach_native_expression(result, expression)
        elif isinstance(result, DataFrame):
            if result.dtypes.map(_is_geometry_like_dtype).sum() > 0:
                result.__class__ = type(self)
                if geo_col in result:
                    result._geometry_column_name = geo_col
                if (
                    source_native_state is not None
                    and _is_native_state_column_projection_key(key, self.columns)
                    and len(result) == source_native_state.row_count
                    and result.index.equals(self.index)
                ):
                    projected = source_native_state.project_columns(tuple(result.columns))
                    if projected is not None:
                        attach_native_state(result, projected)
                elif source_native_state is not None:
                    rowset = _native_boolean_filter_rowset(
                        key,
                        source_native_state,
                        public_index=self.index,
                    )
                    if rowset is not None and len(result) == len(rowset):
                        native_take_rowset = rowset
                        if rowset.is_device and isinstance(key, Series):
                            native_mask_rowset = _native_boolean_rowset_from_aligned_mask(
                                key,
                                source_native_state,
                            )
                            if native_mask_rowset is None or not native_mask_rowset.is_device:
                                mask_values = key.to_numpy(dtype=bool, na_value=False)
                            else:
                                mask_values = None
                            if mask_values is not None and (
                                mask_values.ndim == 1
                                and int(mask_values.sum()) == len(rowset)
                            ):
                                native_take_rowset = NativeRowSet.from_positions(
                                    np.flatnonzero(mask_values).astype(
                                        np.int64,
                                        copy=False,
                                    ),
                                    source_token=rowset.source_token,
                                    source_row_count=rowset.source_row_count,
                                    ordered=rowset.ordered,
                                    unique=rowset.unique,
                                    identity=rowset.identity,
                                    geometry_family_domain=rowset.geometry_family_domain,
                                    trusted_all_valid_rows=rowset.trusted_all_valid_rows,
                                )
                        attach_native_state(
                            result,
                            source_native_state.take(
                                native_take_rowset,
                                preserve_index=True,
                            ),
                        )
                    else:
                        from vibespatial.api._native_state import drop_native_state

                        drop_native_state(result)
            else:
                result.__class__ = DataFrame
        return result

    def _native_geometry_expression(self, operation: str):
        """Return an admitted private geometry expression for internal consumers."""
        from vibespatial.api._native_state import get_native_state

        state = get_native_state(self)
        if state is None:
            return None
        if operation == "area":
            return state.geometry_area_expression()
        if operation == "length":
            return state.geometry_length_expression()
        if operation == "is_valid":
            return state.geometry_validity_expression()
        raise ValueError(f"unsupported native geometry expression {operation!r}")

    def _native_geometry_area_expression(self):
        return self._native_geometry_expression("area")

    def _native_geometry_length_expression(self):
        return self._native_geometry_expression("length")

    def _native_geometry_validity_expression(self):
        return self._native_geometry_expression("is_valid")

    def _native_geometry_distance_expression(self, other):
        """Return admitted private row-aligned distances to another native frame."""
        from vibespatial.api._native_state import get_native_state

        state = get_native_state(self)
        other_state = get_native_state(other)
        if state is None or other_state is None:
            return None
        return state.geometry_distance_expression(other_state)

    def _native_attribute_expression(self, column, *, operation: str | None = None):
        """Return an admitted private attribute expression for internal consumers."""
        from vibespatial.api._native_state import get_native_state

        state = get_native_state(self)
        if state is None:
            return None
        return state.attribute_expression(column, operation=operation)

    def __delitem__(self, key) -> None:
        """If the last geometry column is removed, downcast to a dataframe."""
        from vibespatial.api._native_state import drop_native_state, get_native_state

        source_native_state = get_native_state(self)
        drop_native_state(self)
        super().__delitem__(key)
        if (self.dtypes.map(_is_geometry_like_dtype)).sum() == 0:
            self.__class__ = DataFrame
        else:
            _attach_native_state_after_projection_from_state(self, source_native_state)

    def _persist_old_default_geometry_colname(self) -> None:
        """Persist the default geometry column name of 'geometry' temporarily for
        backwards compatibility.
        """
        # self.columns check required to avoid this warning in __init__
        if self._geometry_column_name is None and "geometry" not in self.columns:
            msg = (
                "You are adding a column named 'geometry' to a GeoDataFrame "
                "constructed without an active geometry column. Currently, "
                "this automatically sets the active geometry column to 'geometry' "
                "but in the future that will no longer happen. Instead, either "
                "provide geometry to the GeoDataFrame constructor "
                "(GeoDataFrame(... geometry=GeoSeries()) or use "
                "`set_geometry('geometry')` "
                "to explicitly set the active geometry column."
            )
            warnings.warn(msg, category=FutureWarning, stacklevel=3)
            self._geometry_column_name = "geometry"

    def __setitem__(self, key, value):
        """Overridden to preserve CRS of GeometryArray.

        Important for cases like
        df['geometry'] = [geom... for geom in df.geometry]
        """
        from vibespatial.api._native_state import drop_native_state, get_native_state

        source_native_state = get_native_state(self)
        native_expressions = {}
        if not pd.api.types.is_list_like(key):
            public_values, native_expressions = _prepare_native_expression_assignments(
                {key: value},
                source_state=source_native_state,
                index=self.index,
                surface="GeoDataFrame.__setitem__",
            )
            value = public_values[key]
        drop_native_state(self)
        if not pd.api.types.is_list_like(key) and (
            key == self._geometry_column_name
            or (key == "geometry" and self._geometry_column_name is None)
        ):
            current_geom_dtype = getattr(self.get(key, None), "dtype", None) if key in self.columns else None
            prefer_device = getattr(current_geom_dtype, "name", None) == "device_geometry"
            if pd.api.types.is_scalar(value) or isinstance(value, BaseGeometry):
                value = [value] * self.shape[0]

            crs = getattr(self, "crs", None)
            # if we don't have a GeoDataFrame yet and there is a column named crs,
            # don't try to use that as a crs
            if isinstance(crs, pd.Series | pd.DataFrame):
                crs = None
            try:
                value = _ensure_geometry(
                    value,
                    crs=crs,
                    prefer_device=prefer_device,
                    fallback_surface="GeoDataFrame.__setitem__",
                )
            except TypeError:
                warnings.warn(
                    "Geometry column does not contain geometry.",
                    stacklevel=2,
                )
            else:
                if key == "geometry":
                    self._persist_old_default_geometry_colname()
        super().__setitem__(key, value)
        assigned_columns = _native_setitem_assigned_columns(key, self.columns)
        if assigned_columns is not None:
            if len(assigned_columns) == 1 and _attach_native_state_after_geometry_column_assign(
                self,
                self,
                assigned_columns[0],
                source_state=source_native_state,
            ):
                return
            _attach_native_state_after_assign(
                self,
                self,
                assigned_columns,
                source_state=source_native_state,
                native_expressions=native_expressions,
            )

    def insert(self, loc: int, column, value, allow_duplicates=lib.no_default) -> None:
        from vibespatial.api._native_state import drop_native_state, get_native_state

        source_native_state = get_native_state(self)
        public_values, native_expressions = _prepare_native_expression_assignments(
            {column: value},
            source_state=source_native_state,
            index=self.index,
            surface="GeoDataFrame.insert",
        )
        value = public_values[column]
        drop_native_state(self)
        super().insert(
            loc,
            column,
            value,
            allow_duplicates=allow_duplicates,
        )
        _attach_native_state_after_assign(
            self,
            self,
            (column,),
            source_state=source_native_state,
            native_expressions=native_expressions,
        )

    def pop(self, item):
        from vibespatial.api._native_state import drop_native_state, get_native_state

        source_native_state = get_native_state(self)
        drop_native_state(self)
        result = super().pop(item)
        if (self.dtypes.map(_is_geometry_like_dtype)).sum() == 0:
            self.__class__ = DataFrame
        else:
            _attach_native_state_after_projection_from_state(self, source_native_state)
        return result

    #
    # Implement pandas methods
    #
    @doc(pd.DataFrame)
    def rename(
        self,
        mapper=None,
        *,
        index=None,
        columns=None,
        axis=None,
        copy=lib.no_default,
        inplace: bool = False,
        level=None,
        errors: str = "ignore",
    ):
        if inplace:
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        result = super().rename(
            mapper=mapper,
            index=index,
            columns=columns,
            axis=axis,
            copy=copy,
            inplace=inplace,
            level=level,
            errors=errors,
        )
        if result is not None:
            _attach_native_state_after_rename(self, result)
        return result

    @doc(pd.DataFrame)
    def drop(
        self,
        labels=None,
        *,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace: bool = False,
        errors: str = "raise",
    ):
        if inplace:
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        result = super().drop(
            labels=labels,
            axis=axis,
            index=index,
            columns=columns,
            level=level,
            inplace=inplace,
            errors=errors,
        )
        if result is None:
            return None

        axis_number = self._get_axis_number(axis)
        row_drop = index is not None or (
            labels is not None and columns is None and axis_number == 0
        )
        column_drop = columns is not None or (
            labels is not None and index is None and axis_number == 1
        )
        if row_drop:
            _attach_native_state_from_result_index(
                self,
                result,
                allow_ordered_subset=True,
            )
        elif column_drop:
            _attach_native_state_after_column_position_take(self, result)
        return result

    @doc(pd.DataFrame)
    def reset_index(
        self,
        level=None,
        *,
        drop: bool = False,
        inplace: bool = False,
        col_level=0,
        col_fill="",
        allow_duplicates=lib.no_default,
        names=None,
    ):
        from vibespatial.api._native_state import (
            attach_native_state,
            drop_native_state,
            get_native_state,
        )

        source_native_state = get_native_state(self)
        if inplace:
            drop_native_state(self)
        elif source_native_state is not None:
            drop_native_state(self)
        try:
            result = super().reset_index(
                level=level,
                drop=drop,
                inplace=inplace,
                col_level=col_level,
                col_fill=col_fill,
                allow_duplicates=allow_duplicates,
                names=names,
            )
        finally:
            if not inplace and source_native_state is not None:
                attach_native_state(self, source_native_state)
        if result is not None and level is None:
            if drop:
                _attach_native_state_after_reset_index_drop(self, result)
            else:
                _attach_native_state_after_reset_index_insert(self, result)
        elif result is not None and level is not None:
            _attach_native_state_after_reset_index_partial(self, result, drop=drop)
        return result

    @doc(pd.DataFrame)
    def set_index(
        self,
        keys,
        *,
        drop: bool = True,
        append: bool = False,
        inplace: bool = False,
        verify_integrity=lib.no_default,
    ):
        if inplace:
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        result = super().set_index(
            keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )
        if result is not None:
            _attach_native_state_after_set_index(self, result)
        return result

    @doc(pd.DataFrame)
    def reindex(
        self,
        labels=None,
        *,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=lib.no_default,
        level=None,
        fill_value=np.nan,
        limit=None,
        tolerance=None,
    ) -> GeoDataFrame:
        result = super().reindex(
            labels=labels,
            index=index,
            columns=columns,
            axis=axis,
            method=method,
            copy=copy,
            level=level,
            fill_value=fill_value,
            limit=limit,
            tolerance=tolerance,
        )
        _attach_native_state_after_reindex(self, result)
        return result

    @doc(pd.DataFrame)
    def reindex_like(
        self,
        other,
        method=None,
        copy=lib.no_default,
        limit=None,
        tolerance=None,
    ) -> GeoDataFrame:
        result = super().reindex_like(
            other,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )
        _attach_native_state_after_reindex(self, result)
        return result

    @doc(pd.DataFrame)
    def filter(self, items=None, like: str | None = None, regex: str | None = None, axis=None):
        result = super().filter(items=items, like=like, regex=regex, axis=axis)
        axis_number = 1 if axis is None else self._get_axis_number(axis)
        if axis_number == 0:
            _attach_native_state_after_reindex(self, result)
        else:
            _attach_native_state_after_column_position_take(self, result)
        return result

    @doc(pd.DataFrame)
    def select_dtypes(self, include=None, exclude=None):
        result = super().select_dtypes(include=include, exclude=exclude)
        _attach_native_state_after_column_position_take(self, result)
        return result

    @doc(pd.DataFrame)
    def query(self, expr: str, *, inplace: bool = False, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        state = _native_device_expression_state(self)
        if state is not None and not inplace and not kwargs:
            native_result = _native_query_result(self, expr, state)
            if native_result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.query",
                    operation="query",
                    implementation="native_device_scalar_expression",
                    reason="row-aligned device predicate lowered to NativeRowSet",
                    detail=f"rows={len(self)}",
                    selected=ExecutionMode.GPU,
                )
                return native_result
        public_invalid = False
        if state is not None:
            import inspect

            caller_frame = inspect.currentframe().f_back
            try:
                public_invalid = _native_public_expression_is_invalid(
                    self,
                    expr,
                    assignment=False,
                    scope_kwargs=kwargs,
                    scope_locals=caller_frame.f_locals,
                    scope_globals=caller_frame.f_globals,
                )
            finally:
                del caller_frame
        if state is not None and not public_invalid:
            record_fallback_event(
                surface="geopandas.geodataframe.query",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native query expression was not admitted or inplace pandas "
                    "mutation was requested"
                ),
                detail=f"rows={len(self)}, inplace={inplace}",
                pipeline="geodataframe",
            )
        if inplace:
            drop_native_state(self)
        query_fallback_token = _PANDAS_QUERY_FALLBACK_ACTIVE.set(True)
        try:
            result = super().query(expr, inplace=inplace, **kwargs)
        finally:
            _PANDAS_QUERY_FALLBACK_ACTIVE.reset(query_fallback_token)
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def eval(self, expr: str, *, inplace: bool = False, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        state = (
            None
            if _PANDAS_QUERY_FALLBACK_ACTIVE.get(False)
            else _native_device_expression_state(self)
        )
        if state is not None and not inplace and not kwargs:
            native_result = _native_eval_result(self, expr, state)
            if native_result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.eval",
                    operation="eval",
                    implementation="native_device_scalar_expression",
                    reason="row-aligned device arithmetic lowered to NativeExpression",
                    detail=f"rows={len(self)}",
                    selected=ExecutionMode.GPU,
                )
                return native_result
        public_invalid = False
        if state is not None:
            import inspect

            caller_frame = inspect.currentframe().f_back
            try:
                public_invalid = _native_public_expression_is_invalid(
                    self,
                    expr,
                    assignment=True,
                    scope_kwargs=kwargs,
                    scope_locals=caller_frame.f_locals,
                    scope_globals=caller_frame.f_globals,
                )
            finally:
                del caller_frame
        if state is not None and not public_invalid:
            record_fallback_event(
                surface="geopandas.geodataframe.eval",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native eval expression was not admitted or inplace pandas "
                    "mutation was requested"
                ),
                detail=f"rows={len(self)}, inplace={inplace}",
                pipeline="geodataframe",
            )
        if inplace:
            drop_native_state(self)
        result = super().eval(expr, inplace=inplace, **kwargs)
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def dropna(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state, get_native_state

        inplace = kwargs.get("inplace", False)
        if inplace:
            drop_native_state(self)
        axis_number = _dropna_axis_number(args, kwargs)
        ignore_index = _dropna_ignore_index(args, kwargs)
        native_ignore_index = (
            not inplace
            and axis_number == 0
            and ignore_index
            and "ignore_index" in kwargs
            and get_native_state(self) is not None
        )
        if native_ignore_index:
            native_kwargs = dict(kwargs)
            native_kwargs["ignore_index"] = False
            result = super().dropna(*args, **native_kwargs)
            if result is None:
                return None
            _attach_native_state_from_result_index(
                self,
                result,
                allow_ordered_subset=True,
            )
            result = result.reset_index(drop=True)
            if get_native_state(result) is None:
                return _drop_native_state_from_result(result)
            return result

        result = super().dropna(*args, **kwargs)
        if result is None:
            return None

        if axis_number == 0 and not ignore_index:
            _attach_native_state_from_result_index(
                self,
                result,
                allow_ordered_subset=True,
            )
        elif axis_number == 1:
            _attach_native_state_after_column_position_take(self, result)

        if get_native_state(result) is None:
            return _drop_native_state_from_result(result)
        return result

    @doc(pd.DataFrame)
    def fillna(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        if kwargs.get("inplace", False):
            drop_native_state(self)
        result = super().fillna(*args, **kwargs)
        if result is not None and _attach_native_state_after_attribute_only_result(self, result):
            return result
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def replace(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        if kwargs.get("inplace", False):
            drop_native_state(self)
        result = super().replace(*args, **kwargs)
        if result is not None and _attach_native_state_after_attribute_only_result(self, result):
            return result
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def where(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        if kwargs.get("inplace", False):
            drop_native_state(self)
        result = super().where(*args, **kwargs)
        if result is not None and _attach_native_state_after_attribute_only_result(self, result):
            return result
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def mask(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        if kwargs.get("inplace", False):
            drop_native_state(self)
        result = super().mask(*args, **kwargs)
        if result is not None and _attach_native_state_after_attribute_only_result(self, result):
            return result
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def rename_axis(self, *args, **kwargs):
        from vibespatial.api._native_state import drop_native_state

        if kwargs.get("inplace", False):
            drop_native_state(self)
        result = super().rename_axis(*args, **kwargs)
        if result is not None and not _attach_native_state_after_axis_relabel(self, result):
            return _drop_native_state_from_result(result)
        return result

    @doc(pd.DataFrame)
    def set_axis(self, *args, **kwargs):
        result = super().set_axis(*args, **kwargs)
        if result is not None and not _attach_native_state_after_axis_relabel(self, result):
            return _drop_native_state_from_result(result)
        return result

    @doc(pd.DataFrame)
    def astype(self, *args, **kwargs):
        result = super().astype(*args, **kwargs)
        if result is not None and _attach_native_state_after_attribute_only_result(self, result):
            return result
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def update(self, *args, **kwargs) -> None:
        from vibespatial.api._native_state import drop_native_state

        drop_native_state(self)
        return super().update(*args, **kwargs)

    @doc(pd.DataFrame)
    def merge(
        self,
        right,
        how: str = "inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes=("_x", "_y"),
        copy=lib.no_default,
        indicator: bool | str = False,
        validate=None,
    ):
        state = (
            None
            if _PANDAS_JOIN_FALLBACK_ACTIVE.get(False)
            else _native_device_expression_state(self)
        )
        public_invalid = state is not None and _native_public_merge_is_invalid(
            self,
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            validate=validate,
        )
        if (
            state is not None
            and not public_invalid
            and how == "inner"
            and left_on is None
            and right_on is None
            and not left_index
            and not right_index
            and not sort
            and not indicator
        ):
            native_result = _native_unique_right_inner_merge(
                self,
                right,
                on=on,
                suffixes=suffixes,
                validate=validate,
            )
            if native_result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.merge",
                    operation="merge",
                    implementation="pylibcudf_unique_right_inner_join",
                    reason=(
                        "bounded many-to-one equality relation consumed as "
                        "NativeFrameState"
                    ),
                    detail=f"left_rows={len(self)}, right_rows={len(right)}",
                    selected=ExecutionMode.GPU,
                )
                return native_result
        if state is not None and not public_invalid:
            record_fallback_event(
                surface="geopandas.geodataframe.merge",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="generic merge shape was not admitted by the bounded native relation path",
                detail=f"left_rows={len(self)}, how={how!r}",
                pipeline="geodataframe",
            )
        result = super().merge(
            right,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def join(
        self,
        other,
        on=None,
        how: str = "left",
        lsuffix: str = "",
        rsuffix: str = "",
        sort: bool = False,
        validate=None,
    ):
        state = _native_device_expression_state(self)
        public_invalid = state is not None and _native_public_join_is_invalid(
            self,
            other,
            state=state,
            on=on,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            validate=validate,
        )
        if (
            state is not None
            and not public_invalid
            and on is None
            and how == "left"
            and not sort
        ):
            native_result = _native_exact_index_join(
                self,
                other,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                validate=validate,
            )
            if native_result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.join",
                    operation="join",
                    implementation="native_exact_index_column_join",
                    reason="unique aligned indexes require no relation materialization",
                    detail=f"rows={len(self)}",
                    selected=ExecutionMode.GPU,
                )
                return native_result
        if state is not None and not public_invalid:
            record_fallback_event(
                surface="geopandas.geodataframe.join",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason="generic index join shape was not admitted by the native aligned path",
                detail=f"left_rows={len(self)}, how={how!r}",
                pipeline="geodataframe",
            )
        join_fallback_token = _PANDAS_JOIN_FALLBACK_ACTIVE.set(True)
        try:
            result = super().join(
                other,
                on=on,
                how=how,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                sort=sort,
                validate=validate,
            )
        finally:
            _PANDAS_JOIN_FALLBACK_ACTIVE.reset(join_fallback_token)
        return _drop_native_state_from_result(result)

    @doc(pd.DataFrame)
    def assign(self, **kwargs) -> GeoDataFrame:
        from vibespatial.api._native_state import get_native_state

        source_native_state = get_native_state(self)
        public_kwargs, native_expressions = _prepare_native_expression_assignments(
            dict(kwargs),
            source_state=source_native_state,
            index=self.index,
            surface="GeoDataFrame.assign",
        )
        result = super().assign(**public_kwargs)
        _attach_native_state_after_assign(
            self,
            result,
            tuple(kwargs),
            source_state=source_native_state,
            native_expressions=native_expressions,
        )
        return result

    def datetime_component(self, column, component: str) -> Series:
        """Extract a calendar component from a datetime attribute.

        This additive public API keeps GeoParquet timestamp columns and the
        resulting integer expression device-resident when a native frame is
        available. Supported components are ``year``, ``month``, ``day``,
        ``weekday``, ``hour``, ``minute``, and ``second``.
        """
        normalized = str(component).lower()
        supported = {
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
            "second",
        }
        if normalized not in supported:
            raise ValueError(
                f"unsupported datetime component {component!r}; "
                f"expected one of {sorted(supported)}"
            )
        if column not in self.columns or column == self._geometry_column_name:
            raise KeyError(column)

        from vibespatial.api._native_expression import NativeExpression
        from vibespatial.api._native_state import get_native_state

        native_rejection_reason = "datetime attribute has no admitted native timestamp payload"
        native_state = get_native_state(self)
        if native_state is not None:
            attributes = native_state.attributes
            policies = attributes.device_column_policies((column,))
            policy = policies.get(column)
            arrow_type = "" if policy is None else policy.arrow_type
            has_timezone = ", tz=" in arrow_type
            utc_timezone = arrow_type.endswith(", tz=UTC]")
            if has_timezone and not utc_timezone:
                native_rejection_reason = (
                    "non-UTC timezone-aware timestamps require pandas local-calendar semantics"
                )
            if (
                policy is not None
                and int(policy.null_count) == 0
                and arrow_type.startswith("timestamp[")
                and (not has_timezone or utc_timezone)
            ):
                try:
                    import pylibcudf as plc

                    from vibespatial.api._native_result_core import (
                        _pylibcudf_numeric_column_view,
                    )
                    from vibespatial.cuda._runtime import pylibcudf_current_stream

                    source = attributes.to_pylibcudf_columns((column,))[0]
                    extracted = plc.datetime.extract_datetime_component(
                        source,
                        getattr(plc.datetime.DatetimeComponent, normalized.upper()),
                        stream=pylibcudf_current_stream(attributes.device_table),
                    )
                    values = _pylibcudf_numeric_column_view(extracted)
                    if values is not None:
                        return _native_expression_assignment_public_series(
                            column,
                            NativeExpression(
                                operation=f"datetime.{normalized}",
                                values=values,
                                source_token=native_state.lineage_token,
                                source_row_count=native_state.row_count,
                                dtype=str(values.dtype),
                                precision="source",
                            ),
                            index=self.index,
                            surface="GeoDataFrame.datetime_component",
                        )
                except (ImportError, AttributeError, NotImplementedError):
                    pass

        from vibespatial.runtime import ExecutionMode
        from vibespatial.runtime.fallbacks import record_fallback_event

        record_fallback_event(
            surface="geopandas.geodataframe.datetime_component",
            reason=native_rejection_reason,
            requested=ExecutionMode.AUTO,
            selected=ExecutionMode.CPU,
        )
        values = pd.to_datetime(pd.DataFrame.__getitem__(self, column))
        return getattr(values.dt, normalized).rename(column)

    @doc(pd.DataFrame)
    def drop_duplicates(
        self,
        subset=None,
        *,
        keep="first",
        inplace: bool = False,
        ignore_index: bool = False,
    ):
        from vibespatial.api._native_state import drop_native_state, get_native_state

        source_native_state = get_native_state(self)
        native_rowset = (
            None
            if inplace or source_native_state is None
            else _native_drop_duplicates_rowset(self, subset, keep)
        )
        if native_rowset is not None:
            result = _take_public_frame_with_native_state(
                self,
                native_rowset,
                source_native_state=source_native_state,
                geometry_column=self._geometry_column_name,
                preserve_index=not ignore_index,
            )
            if result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.drop_duplicates",
                    operation="drop_duplicates",
                    implementation="native_device_distinct_rowset",
                    reason="consume device-backed attribute keys without pandas row assembly",
                    detail=f"rows={len(self)}, keys={len(_normalize_drop_duplicates_subset(subset, self.columns) or ())}",
                    selected=ExecutionMode.GPU,
                )
                return result
        if source_native_state is not None:
            record_fallback_event(
                surface="geopandas.geodataframe.drop_duplicates",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native device distinct was not admitted for the requested "
                    "keys/options or could not assemble the public result"
                ),
                detail=f"rows={len(self)}, keep={keep!r}, inplace={inplace}",
                pipeline="geodataframe",
            )
        positions = (
            None
            if inplace or source_native_state is None
            else _native_drop_duplicates_positions(self, subset, keep)
        )
        if inplace:
            drop_native_state(self)
        result = super().drop_duplicates(
            subset=subset,
            keep=keep,
            inplace=inplace,
            ignore_index=ignore_index,
        )
        if result is None:
            return None
        if positions is None:
            return _drop_native_state_from_result(result)
        if len(result) != int(positions.size):
            return _drop_native_state_from_result(result)

        result_index = result.index if ignore_index else None
        _attach_native_state_from_row_positions(
            self,
            result,
            positions,
            unique=True,
            index_override=result_index,
        )
        return result

    @doc(pd.DataFrame)
    def take(self, indices, axis=0, **kwargs) -> GeoDataFrame:
        axis_number = self._get_axis_number(axis)
        result = super().take(indices, axis=axis, **kwargs)
        if axis_number == 0:
            positions, unique = _native_iloc_row_positions(indices, len(self))
            _attach_native_state_from_row_positions(
                self,
                result,
                positions,
                unique=unique,
            )
        elif axis_number == 1:
            _attach_native_state_after_column_position_take(self, result)
        return result

    @doc(pd.DataFrame)
    def copy(self, deep: bool = True) -> GeoDataFrame:
        from vibespatial.api._native_state import (
            attach_native_state,
            get_native_state,
        )

        source_native_state = get_native_state(self)
        fast_native_copy = (
            bool(deep)
            and source_native_state is not None
            and bool(self.columns.is_unique)
        )
        if fast_native_copy:
            projected = source_native_state.project_columns(tuple(self.columns))
            if projected is not None:
                if _public_frame_columns_are_native_lazy(self):
                    return _shallow_public_frame_with_native_state(
                        self,
                        projected,
                        geometry_column=self._geometry_column_name,
                    )
        copied = super().copy(deep=False if fast_native_copy else deep)
        if type(copied) is pd.DataFrame:
            copied.__class__ = type(self)
            copied._geometry_column_name = self._geometry_column_name
        if fast_native_copy:
            from vibespatial.api._native_public_arrays import NativeNumericExpressionArray

            for column in self.columns:
                source_column = pd.DataFrame.__getitem__(self, column)
                if not isinstance(source_column, Series):
                    fast_native_copy = False
                    break
                if isinstance(getattr(source_column, "array", None), NativeNumericExpressionArray):
                    continue
                if _is_geometry_like_dtype(source_column.dtype):
                    pd.DataFrame.__setitem__(
                        copied,
                        column,
                        _copy_geometry_series_preserving_owned_backing(
                            source_column,
                            index=copied.index,
                            deep=True,
                        ),
                    )
                else:
                    pd.DataFrame.__setitem__(
                        copied,
                        column,
                        source_column.copy(deep=True),
                    )
        if source_native_state is not None:
            attach_native_state(copied, source_native_state)
        return copied

    @doc(pd.DataFrame)
    def sort_values(
        self,
        by,
        *,
        axis=0,
        ascending=True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        ignore_index: bool = False,
        key=None,
    ):
        from vibespatial.api._native_state import get_native_state

        native_state = get_native_state(self)
        native_device_attributes = native_state is not None and bool(
            getattr(native_state.attributes, "is_device_backed", False)
        )
        if inplace:
            if native_device_attributes:
                record_fallback_event(
                    surface="geopandas.geodataframe.sort_values",
                    requested=ExecutionMode.AUTO,
                    selected=ExecutionMode.CPU,
                    reason="in-place sort requires pandas mutation semantics",
                    detail=f"rows={len(self)}, axis={axis!r}",
                    pipeline="geodataframe",
                )
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        axis_number = self._get_axis_number(axis)
        if not inplace and axis_number == 0:
            native_rowset = _native_sort_values_rowset(
                self,
                by,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                key=key,
            )
            if native_state is not None and native_rowset is not None:
                result = _take_public_frame_with_native_state(
                    self,
                    native_rowset,
                    source_native_state=native_state,
                    geometry_column=self._geometry_column_name,
                    preserve_index=not ignore_index,
                )
                if result is not None:
                    record_dispatch_event(
                        surface="geopandas.geodataframe.sort_values",
                        operation="sort_values",
                        implementation="native_device_sorted_rowset",
                        reason="consume device-backed attribute keys without pandas row assembly",
                        detail=f"rows={len(self)}, keys={len(_normalize_sort_columns(by) or ())}",
                        selected=ExecutionMode.GPU,
                    )
                    return result
        if native_device_attributes and not inplace:
            record_fallback_event(
                surface="geopandas.geodataframe.sort_values",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native device stable sort was not admitted for the requested "
                    "keys/options or could not assemble the public result"
                ),
                detail=(
                    f"rows={len(self)}, kind={kind!r}, "
                    f"na_position={na_position!r}"
                ),
                pipeline="geodataframe",
            )
        if not inplace and axis_number == 0 and (
            not self.index.is_unique or getattr(self.index, "nlevels", 1) != 1
        ):
            result, positions, unique = _sort_values_with_native_row_position_marker(
                self,
                by,
                axis=axis,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
            if result is not None and positions is not None:
                _attach_native_state_from_row_positions(
                    self,
                    result,
                    positions,
                    unique=unique,
                    index_override=result.index if ignore_index else None,
                )
                return result

        native_ignore_index = False
        if not inplace and axis_number == 0 and ignore_index:
            from vibespatial.api._native_state import get_native_state

            native_ignore_index = get_native_state(self) is not None

        if native_ignore_index:
            ordered = super().sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
                ignore_index=False,
                key=key,
            )
            _attach_native_state_from_result_index(self, ordered)
            result = ordered.reset_index(drop=True)
        else:
            result = super().sort_values(
                by=by,
                axis=axis,
                ascending=ascending,
                inplace=inplace,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
            if result is not None and axis_number == 0 and not ignore_index:
                _attach_native_state_from_result_index(self, result)
        return result

    @doc(pd.DataFrame)
    def nlargest(self, n: int, columns, keep: str = "first"):
        device_request = _native_topk_request_targets_device_attributes(
            self,
            n,
            columns,
            keep,
        )
        native_rowset = _native_topk_rowset(
            self,
            n,
            columns,
            largest=True,
            keep=keep,
        )
        if native_rowset is not None:
            from vibespatial.api._native_state import get_native_state

            result = _take_public_frame_with_native_state(
                self,
                native_rowset,
                source_native_state=get_native_state(self),
                geometry_column=self._geometry_column_name,
                preserve_index=True,
            )
            if result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.nlargest",
                    operation="nlargest",
                    implementation="native_device_topk_rowset",
                    reason="bounded lexicographic top-k over device-backed attributes",
                    detail=f"rows={len(self)}, n={int(n)}, keep={keep!r}",
                    selected=ExecutionMode.GPU,
                )
                return result
        if device_request:
            record_fallback_event(
                surface="geopandas.geodataframe.nlargest",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native device top-k was not admitted or could not assemble "
                    "the public result"
                ),
                detail=f"rows={len(self)}, n={int(n)}, keep={keep!r}",
                pipeline="geodataframe",
            )
        return super().nlargest(n, columns, keep=keep)

    @doc(pd.DataFrame)
    def nsmallest(self, n: int, columns, keep: str = "first"):
        device_request = _native_topk_request_targets_device_attributes(
            self,
            n,
            columns,
            keep,
        )
        native_rowset = _native_topk_rowset(
            self,
            n,
            columns,
            largest=False,
            keep=keep,
        )
        if native_rowset is not None:
            from vibespatial.api._native_state import get_native_state

            result = _take_public_frame_with_native_state(
                self,
                native_rowset,
                source_native_state=get_native_state(self),
                geometry_column=self._geometry_column_name,
                preserve_index=True,
            )
            if result is not None:
                record_dispatch_event(
                    surface="geopandas.geodataframe.nsmallest",
                    operation="nsmallest",
                    implementation="native_device_topk_rowset",
                    reason="bounded lexicographic top-k over device-backed attributes",
                    detail=f"rows={len(self)}, n={int(n)}, keep={keep!r}",
                    selected=ExecutionMode.GPU,
                )
                return result
        if device_request:
            record_fallback_event(
                surface="geopandas.geodataframe.nsmallest",
                requested=ExecutionMode.AUTO,
                selected=ExecutionMode.CPU,
                reason=(
                    "native device top-k was not admitted or could not assemble "
                    "the public result"
                ),
                detail=f"rows={len(self)}, n={int(n)}, keep={keep!r}",
                pipeline="geodataframe",
            )
        return super().nsmallest(n, columns, keep=keep)

    @doc(pd.DataFrame)
    def sort_index(
        self,
        *,
        axis=0,
        level=None,
        ascending=True,
        inplace: bool = False,
        kind: str = "quicksort",
        na_position: str = "last",
        sort_remaining: bool = True,
        ignore_index: bool = False,
        key=None,
    ):
        if inplace:
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        axis_number = self._get_axis_number(axis)
        if not inplace and axis_number == 0 and (
            not self.index.is_unique or getattr(self.index, "nlevels", 1) != 1
        ):
            result, positions, unique = _sort_index_with_native_row_position_marker(
                self,
                axis=axis,
                level=level,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
                ignore_index=ignore_index,
                key=key,
            )
            if result is not None and positions is not None:
                _attach_native_state_from_row_positions(
                    self,
                    result,
                    positions,
                    unique=unique,
                    index_override=result.index if ignore_index else None,
                )
                return result

        native_ignore_index = False
        if not inplace and axis_number == 0 and ignore_index:
            from vibespatial.api._native_state import get_native_state

            native_ignore_index = get_native_state(self) is not None

        if native_ignore_index:
            ordered = super().sort_index(
                axis=axis,
                level=level,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
                ignore_index=False,
                key=key,
            )
            _attach_native_state_from_result_index(self, ordered)
            result = ordered.reset_index(drop=True)
        else:
            result = super().sort_index(
                axis=axis,
                level=level,
                ascending=ascending,
                inplace=inplace,
                kind=kind,
                na_position=na_position,
                sort_remaining=sort_remaining,
                ignore_index=ignore_index,
                key=key,
            )
            if result is not None and axis_number == 0 and not ignore_index:
                _attach_native_state_from_result_index(self, result)
            elif result is not None and axis_number == 1 and not ignore_index:
                _attach_native_state_after_column_position_take(self, result)
        return result

    @doc(pd.DataFrame)
    def apply(
        self,
        func,
        axis=0,
        raw: bool = False,
        result_type=None,
        args=(),
        **kwargs,
    ):
        result = super().apply(
            func, axis=axis, raw=raw, result_type=result_type, args=args, **kwargs
        )
        # Reconstruct gdf if it was lost by apply
        if (
            isinstance(result, DataFrame)
            and self._geometry_column_name in result.columns
        ):
            # axis=1 apply will split GeometryDType to object, try and cast back
            try:
                result = result.set_geometry(self._geometry_column_name)
            except TypeError:
                pass
            else:
                if self.crs is not None and result.crs is None:
                    result.set_crs(self.crs, inplace=True)
        elif isinstance(result, Series) and result.dtype == "object":
            # Try reconstruct series GeometryDtype if lost by apply
            # If all none and object dtype assert list of nones is more likely
            # intended than list of null geometry.
            if not result.isna().all():
                try:
                    # not enough info about func to preserve CRS
                    result = _ensure_geometry(result)

                except (TypeError, shapely.errors.GeometryTypeError):
                    pass

        return _drop_native_state_from_result(result)

    @classmethod
    def _geodataframe_constructor_with_fallback(
        cls, *args, **kwargs
    ) -> pd.DataFrame | GeoDataFrame:
        """A flexible constructor for GeoDataFrame._constructor, which falls back
        to returning a DataFrame (if a certain operation does not preserve the
        geometry column).
        """
        df = cls(*args, **kwargs)

        geometry_cols_mask = df.dtypes.map(_is_geometry_like_dtype)

        if len(geometry_cols_mask) == 0 or geometry_cols_mask.sum() == 0:
            df = pd.DataFrame(df)

        return df

    @property
    def _constructor(self) -> DataFrame | GeoDataFrame:
        return self._geodataframe_constructor_with_fallback

    def _constructor_from_mgr(self, mgr, axes) -> DataFrame | GeoDataFrame:
        # replicate _geodataframe_constructor_with_fallback behaviour
        # unless safe to skip
        if not any(_is_geometry_like_dtype(block.dtype) for block in mgr.blocks):
            return self._geodataframe_constructor_with_fallback(
                pd.DataFrame._from_mgr(mgr, axes)
            )
        gdf = self._from_mgr(mgr, axes)
        # _from_mgr doesn't preserve metadata (expect __finalize__ to be called)
        # still need to mimic __init__ behaviour with geometry=None
        if (gdf.columns == "geometry").sum() == 1:  # only if "geometry" is single col
            gdf._geometry_column_name = "geometry"
        return gdf

    @property
    def _constructor_sliced(self) -> Series | GeoSeries:
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries.

            Note:

            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """
            srs = pd.Series(*args, **kwargs)
            is_row_proxy = srs.index.is_(self.columns)
            if is_geometry_type(srs) and not is_row_proxy:
                srs = GeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

    def _constructor_sliced_from_mgr(self, mgr, axes) -> Series | GeoSeries:
        is_row_proxy = mgr.index.is_(self.columns)

        if _is_geometry_like_dtype(mgr.blocks[0].dtype) and not is_row_proxy:
            return GeoSeries._from_mgr(mgr, axes)
        return Series._from_mgr(mgr, axes)

    def __finalize__(
        self, other, method: str | None = None, **kwargs
    ) -> GeoDataFrame | GeoSeries:
        """Propagate metadata from other to self."""
        self = super().__finalize__(other, method=method, **kwargs)
        native_state_preserved = False

        # merge operation: using metadata of the left object
        if method == "merge":
            # pandas-dev/pandas#60357 : merge/concat use input_objs
            if PANDAS_GE_30:
                # other is a types.SimpleNameSpace
                left_obj = other.input_objs[0]
            else:
                # other is a _MergeOperation
                left_obj = other.left
            for name in self._metadata:
                object.__setattr__(self, name, getattr(left_obj, name, None))
        # concat operation: using metadata of the first object
        elif method == "concat":
            # pandas-dev/pandas#60357 : merge/concat use input_objs
            input_objs = _concat_input_objects(other)
            first_obj = input_objs[0]
            for name in self._metadata:
                object.__setattr__(self, name, getattr(first_obj, name, None))

            if (
                self.columns.nlevels == 1
                and (self.columns == self._geometry_column_name).sum() > 1
            ) or (
                self.columns.nlevels > 1
                and (
                    self.columns.get_level_values(0) == self._geometry_column_name
                ).sum()
                > 1
            ):
                raise ValueError(
                    "Concat operation has resulted in multiple columns using "
                    f"the geometry column name '{self._geometry_column_name}'.\n"
                    "Please ensure this column from the first DataFrame is not "
                    "repeated."
                )
            native_state_preserved = _attach_native_state_after_concat(
                self,
                input_objs,
            )
        elif method == "unstack":
            # unstack adds multiindex columns and reshapes data.
            # it never makes sense to retain geometry column
            self._geometry_column_name = None
            self._crs = None
        if method in {"merge", "unstack"} or (
            method == "concat" and not native_state_preserved
        ):
            from vibespatial.api._native_state import drop_native_state

            drop_native_state(self)
        return self

    def dissolve(
        self,
        by: str | None = None,
        aggfunc="first",
        as_index: bool = True,
        level=None,
        sort: bool = True,
        observed: bool = False,
        dropna: bool = True,
        method: Literal["unary", "coverage", "disjoint_subset"] = "unary",
        grid_size: float | None = None,
        **kwargs,
    ) -> GeoDataFrame:
        """
        Dissolve geometries within `groupby` into single observation.
        This is accomplished by applying the `union_all` method
        to all geometries within a groupself.

        Observations associated with each `groupby` group will be aggregated
        using the `aggfunc`.

        Parameters
        ----------
        by : str or list-like, default None
            Column(s) whose values define the groups to be dissolved. If None,
            the entire GeoDataFrame is considered as a single group. If a list-like
            object is provided, the values in the list are treated as categorical
            labels, and polygons will be combined based on the equality of
            these categorical labels.
        aggfunc : function or string, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to pandas `groupby.agg` method.
            Accepted combinations are:

            - function
            - string function name
            - list of functions and/or function names, e.g. [np.sum, 'mean']
            - dict of axis labels -> functions, function names or list of such.
        as_index : boolean, default True
            If true, groupby columns become index of result.
        level : int or str or sequence of int or sequence of str, default None
            If the axis is a MultiIndex (hierarchical), group by a
            particular level or levels.
        sort : bool, default True
            Sort group keys. Get better performance by turning this off.
            Note this does not influence the order of observations within
            each group. Groupby preserves the order of rows within each group.
        observed : bool, default False
            This only applies if any of the groupers are Categoricals.
            If True: only show observed values for categorical groupers.
            If False: show all values for categorical groupers.
        dropna : bool, default True
            If True, and if group keys contain NA values, NA values
            together with row/column will be dropped. If False, NA
            values will also be treated as the key in groups.
        method : str (default ``"unary"``)
            The method to use for the union. Options are:

            * ``"unary"``: use the unary union algorithm. This option is the most robust
              but can be slow for large numbers of geometries (default).
            * ``"coverage"``: use the coverage union algorithm. This option is optimized
              for non-overlapping polygons and can be significantly faster than the
              unary union algorithm. However, it can produce invalid geometries if the
              polygons overlap.
            * ``"disjoint_subset:``: use the disjoint subset union algorithm. This
              option is optimized for inputs that can be divided into subsets that do
              not intersect. If there is only one such subset, performance can be
              expected to be worse than ``"unary"``.  Requires Shapely >= 2.1.


        grid_size : float, default None
            When grid size is specified, a fixed-precision space is used to perform the
            union operations. This can be useful when unioning geometries that are not
            perfectly snapped or to avoid geometries not being unioned because of
            `robustness issues <https://libgeos.org/usage/faq/#why-doesnt-a-computed-point-lie-exactly-on-a-line>`_.
            The inputs are first snapped to a grid of the given size. When a line
            segment of a geometry is within tolerance off a vertex of another geometry,
            this vertex will be inserted in the line segment. Finally, the result
            vertices are computed on the same grid. Is only supported for ``method``
            ``"unary"``. If None, the highest precision of the inputs will be used.
            Defaults to None.

            .. versionadded:: 1.1.0
        **kwargs :
            Keyword arguments to be passed to the pandas `DataFrameGroupby.agg` method
            which is used by `dissolve`. In particular, `numeric_only` may be
            supplied, which will be required in pandas 2.0 for certain aggfuncs.

            .. versionadded:: 0.13.0

        Returns
        -------
        GeoDataFrame

        Examples
        --------
        >>> from shapely.geometry import Point
        >>> d = {
        ...     "col1": ["name1", "name2", "name1"],
        ...     "geometry": [Point(1, 2), Point(2, 1), Point(0, 1)],
        ... }
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1     geometry
        0  name1  POINT (1 2)
        1  name2  POINT (2 1)
        2  name1  POINT (0 1)

        >>> dissolved = gdf.dissolve('col1')
        >>> dissolved  # doctest: +SKIP
                                geometry
        col1
        name1  MULTIPOINT ((0 1), (1 2))
        name2                POINT (2 1)

        See Also
        --------
        GeoDataFrame.explode : explode multi-part geometries into single geometries

        """
        collapse_all = by is None and level is None
        if collapse_all:
            by = np.zeros(len(self), dtype="int64")  # type: ignore [assignment]

        aggregated = evaluate_geopandas_dissolve(
            self,
            by=by,
            aggfunc=aggfunc,
            as_index=as_index,
            level=level,
            sort=sort,
            observed=observed,
            dropna=dropna,
            method=method,
            grid_size=grid_size,
            agg_kwargs=kwargs,
        )
        from vibespatial.geometry.device_array import DeviceGeometryArray

        used_device = isinstance(aggregated.geometry.values, DeviceGeometryArray)
        selected = ExecutionMode.GPU if used_device else ExecutionMode.CPU
        record_dispatch_event(
            surface="geopandas.geodataframe.dissolve",
            operation="dissolve",
            implementation="grouped_union_pipeline",
            reason="route grouped geometry unions through the repo-owned grouped dissolve pipeline",
            detail=f"rows={len(self)}, method={method}, sort={sort}, observed={observed}",
            selected=selected,
        )

        return aggregated

    def dissolve_lazy(
        self,
        by: str | None = None,
        aggfunc="first",
        as_index: bool = True,
        level=None,
        sort: bool = True,
        observed: bool = False,
        dropna: bool = True,
        method: Literal["unary", "coverage", "disjoint_subset"] = "unary",
        grid_size: float | None = None,
        **kwargs,
    ):
        """Build a predicate-first dissolve view with on-demand materialization."""
        if by is None and level is None:
            by = np.zeros(len(self), dtype="int64")  # type: ignore [assignment]

        record_dispatch_event(
            surface="geopandas.geodataframe.dissolve_lazy",
            operation="dissolve_lazy",
            implementation="virtual_grouped_dissolve",
            reason="route grouped dissolve into a lazy predicate-first view with deferred union materialization",
            detail=f"rows={len(self)}, method={method}, sort={sort}, observed={observed}",
            selected=ExecutionMode.CPU,
        )
        return evaluate_geopandas_lazy_dissolve(
            self,
            by=by,
            aggfunc=aggfunc,
            as_index=as_index,
            level=level,
            sort=sort,
            observed=observed,
            dropna=dropna,
            method=method,
            grid_size=grid_size,
            agg_kwargs=kwargs,
        )

    # overrides the pandas native explode method to break up features geometrically
    def explode(
        self,
        column: str | None = None,
        ignore_index: bool = False,
        index_parts: bool = False,
        **kwargs,
    ) -> GeoDataFrame | DataFrame:
        """
        Explode multi-part geometries into multiple single geometries.

        Each row containing a multi-part geometry will be split into
        multiple rows with single geometries, thereby increasing the vertical
        size of the GeoDataFrame.

        Parameters
        ----------
        column : string, default None
            Column to explode. In the case of a geometry column, multi-part
            geometries are converted to single-part.
            If None, the active geometry column is used.
        ignore_index : bool, default False
            If True, the resulting index will be labelled 0, 1, …, n - 1,
            ignoring `index_parts`.
        index_parts : boolean, default False
            If True, the resulting index will be a multi-index (original
            index with an additional level indicating the multiple
            geometries: a new zero-based index for each single part geometry
            per multi-part geometry).

        Returns
        -------
        GeoDataFrame
            Exploded geodataframe with each single geometry
            as a separate entry in the geodataframe.

        Examples
        --------
        >>> from shapely.geometry import MultiPoint
        >>> d = {
        ...     "col1": ["name1", "name2"],
        ...     "geometry": [
        ...         MultiPoint([(1, 2), (3, 4)]),
        ...         MultiPoint([(2, 1), (0, 0)]),
        ...     ],
        ... }
        >>> gdf = geopandas.GeoDataFrame(d, crs=4326)
        >>> gdf
            col1               geometry
        0  name1  MULTIPOINT ((1 2), (3 4))
        1  name2  MULTIPOINT ((2 1), (0 0))

        >>> exploded = gdf.explode(index_parts=True)
        >>> exploded
              col1     geometry
        0 0  name1  POINT (1 2)
          1  name1  POINT (3 4)
        1 0  name2  POINT (2 1)
          1  name2  POINT (0 0)

        >>> exploded = gdf.explode(index_parts=False)
        >>> exploded
            col1     geometry
        0  name1  POINT (1 2)
        0  name1  POINT (3 4)
        1  name2  POINT (2 1)
        1  name2  POINT (0 0)

        >>> exploded = gdf.explode(ignore_index=True)
        >>> exploded
            col1     geometry
        0  name1  POINT (1 2)
        1  name1  POINT (3 4)
        2  name2  POINT (2 1)
        3  name2  POINT (0 0)

        See Also
        --------
        GeoDataFrame.dissolve : dissolve geometries into a single observation.

        """
        # If no column is specified then default to the active geometry column
        if column is None:
            column = self.geometry.name
        # If the specified column is not a geometry dtype use pandas explode
        if not _is_geometry_like_dtype(self[column].dtype):
            return super().explode(column, ignore_index=ignore_index, **kwargs)

        exploded_geom = self.geometry.reset_index(drop=True).explode(index_parts=True)
        exploded_source_rows = None
        if isinstance(exploded_geom.index, pd.MultiIndex):
            try:
                exploded_source_rows = np.asarray(
                    exploded_geom.index.get_level_values(0),
                    dtype=np.int64,
                )
            except (TypeError, ValueError):
                exploded_source_rows = None

        df = self.drop(self._geometry_column_name, axis=1).take(
            exploded_geom.index.droplevel(-1)
        )
        df[exploded_geom.name] = exploded_geom.values
        df = df.set_geometry(self._geometry_column_name).__finalize__(self)

        if ignore_index:
            df.reset_index(inplace=True, drop=True)
        elif index_parts:
            # reset to MultiIndex, otherwise df index is only first level of
            # exploded GeoSeries index.
            df = df.set_index(
                exploded_geom.index.droplevel(
                    list(range(exploded_geom.index.nlevels - 1))
                ),
                append=True,
            )

        _attach_native_state_after_geometry_explode(
            self,
            df,
            column=column,
            exploded_source_rows=exploded_source_rows,
        )
        return df

    def to_postgis(
        self,
        name: str,
        con,
        schema: str | None = None,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = False,
        index_label: Iterable[str] | str | None = None,
        chunksize: int | None = None,
        dtype=None,
    ) -> None:
        """
        Upload GeoDataFrame into PostGIS database.

        This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
        Python driver (psycopg or psycopg2) to be installed.

        It is also possible to use :meth:`~GeoDataFrame.to_file` to write to a database.
        Especially for file geodatabases like GeoPackage or SpatiaLite this can be
        easier.

        Parameters
        ----------
        name : str
            Name of the target table.
        con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
            Active connection to the PostGIS database.
        if_exists : {'fail', 'replace', 'append'}, default 'fail'
            How to behave if the table already exists:

            - fail: Raise a ValueError.
            - replace: Drop the table before inserting new values.
            - append: Insert new values to the existing table.
        schema : string, optional
            Specify the schema. If None, use default schema: 'public'.
        index : bool, default False
            Write DataFrame index as a column.
            Uses *index_label* as the column name in the table.
        index_label : string or sequence, default None
            Column label for index column(s).
            If None is given (default) and index is True,
            then the index names are used.
        chunksize : int, optional
            Rows will be written in batches of this size at a time.
            By default, all rows will be written at once.
        dtype : dict of column name to SQL type, default None
            Specifying the datatype for columns.
            The keys should be the column names and the values
            should be the SQLAlchemy types.

        Examples
        --------
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("postgresql://myusername:mypassword@myhost:5432\
/mydatabase")  # doctest: +SKIP
        >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP

        See Also
        --------
        GeoDataFrame.to_file : write GeoDataFrame to file
        read_postgis : read PostGIS database to GeoDataFrame

        """
        geopandas.io.sql._write_postgis(
            self, name, con, schema, if_exists, index, index_label, chunksize, dtype
        )

    plot = Accessor("plot", geopandas.plotting.GeoplotAccessor)

    @doc(_explore)
    def explore(self, *args, **kwargs) -> folium.Map:
        return _explore(self, *args, **kwargs)

    def sjoin(
        self,
        df: GeoDataFrame,
        how: Literal["left", "right", "inner", "outer"] = "inner",
        predicate: str = "intersects",
        lsuffix: str = "left",
        rsuffix: str = "right",
        **kwargs,
    ) -> GeoDataFrame:
        """Spatial join of two GeoDataFrames.

        See the User Guide page :doc:`../../user_guide/mergingdata` for details.

        Parameters
        ----------
        df : GeoDataFrame
        how : string, default 'inner'
            The type of join:

            * 'left': use keys from left_df; retain only left_df geometry column
            * 'right': use keys from right_df; retain only right_df geometry column
            * 'inner': use intersection of keys from both dfs; retain only
              left_df geometry column
            * 'outer': use union of keys from both dfs; retain a single active
              geometry column by preferring left geometries and filling
              unmatched right-only rows from the right geometry column

        predicate : string, default 'intersects'
            Binary predicate. Valid values are determined by the spatial index used.
            You can check the valid values in left_df or right_df as
            ``left_df.sindex.valid_query_predicates`` or
            ``right_df.sindex.valid_query_predicates``

            Available predicates include:

            * ``'intersects'``: True if geometries intersect (boundaries and interiors)
            * ``'within'``: True if left geometry is completely within right geometry
            * ``'contains'``: True if left geometry completely contains right geometry
            * ``'contains_properly'``: True if left geometry contains right geometry
              and their boundaries do not touch
            * ``'overlaps'``: True if geometries overlap but neither contains the other
            * ``'crosses':`` True if geometries cross (interiors intersect but neither
              contains the other, with intersection dimension less than max dimension)
            * ``'touches'``: True if geometries touch at boundaries but interiors don't
            * ``'covers'``: True if left geometry covers right geometry (every point of
              right is a point of left)
            * ``'covered_by'``: True if left geometry is covered by right geometry
            * ``'dwithin'``: True if geometries are within specified distance (requires
              distance parameter)

        lsuffix : string, default 'left'
            Suffix to apply to overlapping column names (left GeoDataFrame).
        rsuffix : string, default 'right'
            Suffix to apply to overlapping column names (right GeoDataFrame).
        distance : number or array_like, optional
            Distance(s) around each input geometry within which to query the tree
            for the 'dwithin' predicate. If array_like, must be
            one-dimesional with length equal to length of left GeoDataFrame.
            Required if ``predicate='dwithin'``.
        on_attribute : string, list or tuple
            Column name(s) to join on as an additional join restriction on top
            of the spatial predicate. These must be found in both DataFrames.
            If set, observations are joined only if the predicate applies
            and values in specified columns match.

        Examples
        --------
        >>> import geodatasets
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_commpop")
        ... )
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... ).to_crs(chicago.crs)

        >>> chicago.head()  # doctest: +SKIP
                 community  ...                                           geometry
        0          DOUGLAS  ...  MULTIPOLYGON (((-87.60914 41.84469, -87.60915 ...
        1          OAKLAND  ...  MULTIPOLYGON (((-87.59215 41.81693, -87.59231 ...
        2      FULLER PARK  ...  MULTIPOLYGON (((-87.62880 41.80189, -87.62879 ...
        3  GRAND BOULEVARD  ...  MULTIPOLYGON (((-87.60671 41.81681, -87.60670 ...
        4          KENWOOD  ...  MULTIPOLYGON (((-87.59215 41.81693, -87.59215 ...

        [5 rows x 9 columns]

        >>> groceries.head()  # doctest: +SKIP
           OBJECTID     Ycoord  ...  Category                           geometry
        0        16  41.973266  ...       NaN  MULTIPOINT ((-87.65661 41.97321))
        1        18  41.696367  ...       NaN  MULTIPOINT ((-87.68136 41.69713))
        2        22  41.868634  ...       NaN  MULTIPOINT ((-87.63918 41.86847))
        3        23  41.877590  ...       new  MULTIPOINT ((-87.65495 41.87783))
        4        27  41.737696  ...       NaN  MULTIPOINT ((-87.62715 41.73623))
        [5 rows x 8 columns]

        >>> groceries_w_communities = groceries.sjoin(chicago)
        >>> groceries_w_communities[["OBJECTID", "community", "geometry"]].head()
           OBJECTID       community                           geometry
        0        16          UPTOWN  MULTIPOINT ((-87.65661 41.97321))
        1        18     MORGAN PARK  MULTIPOINT ((-87.68136 41.69713))
        2        22  NEAR WEST SIDE  MULTIPOINT ((-87.63918 41.86847))
        3        23  NEAR WEST SIDE  MULTIPOINT ((-87.65495 41.87783))
        4        27         CHATHAM  MULTIPOINT ((-87.62715 41.73623))

        Notes
        -----
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.

        See Also
        --------
        GeoDataFrame.sjoin_nearest : nearest neighbor join
        sjoin : equivalent top-level function
        """
        return geopandas.sjoin(
            left_df=self,
            right_df=df,
            how=how,
            predicate=predicate,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            **kwargs,
        )

    def sjoin_nearest(
        self,
        right: GeoDataFrame,
        how: Literal["left", "right", "inner"] = "inner",
        max_distance: float | None = None,
        lsuffix: str = "left",
        rsuffix: str = "right",
        distance_col: str | None = None,
        exclusive: bool = False,
    ) -> GeoDataFrame:
        """
        Spatial join of two GeoDataFrames based on the distance between their
        geometries.

        Results will include multiple output records for a single input record
        where there are multiple equidistant nearest or intersected neighbors.

        See the User Guide page
        https://geopandas.readthedocs.io/en/latest/docs/user_guide/mergingdata.html
        for more details.


        Parameters
        ----------
        right : GeoDataFrame
        how : string, default 'inner'
            The type of join:

            * 'left': use keys from left_df; retain only left_df geometry column
            * 'right': use keys from right_df; retain only right_df geometry column
            * 'inner': use intersection of keys from both dfs; retain only
              left_df geometry column

        max_distance : float, default None
            Maximum distance within which to query for nearest geometry.
            Must be greater than 0.
            The max_distance used to search for nearest items in the tree may have a
            significant impact on performance by reducing the number of input
            geometries that are evaluated for nearest items in the tree.
        lsuffix : string, default 'left'
            Suffix to apply to overlapping column names (left GeoDataFrame).
        rsuffix : string, default 'right'
            Suffix to apply to overlapping column names (right GeoDataFrame).
        distance_col : string, default None
            If set, save the distances computed between matching geometries under a
            column of this name in the joined GeoDataFrame.
        exclusive : bool, optional, default False
            If True, the nearest geometries that are equal to the input geometry
            will not be returned, default False.

        Examples
        --------
        >>> import geodatasets
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... )
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... ).to_crs(groceries.crs)

        >>> chicago.head()  # doctest: +SKIP
           ComAreaID  ...                                           geometry
        0         35  ...  POLYGON ((-87.60914 41.84469, -87.60915 41.844...
        1         36  ...  POLYGON ((-87.59215 41.81693, -87.59231 41.816...
        2         37  ...  POLYGON ((-87.62880 41.80189, -87.62879 41.801...
        3         38  ...  POLYGON ((-87.60671 41.81681, -87.60670 41.816...
        4         39  ...  POLYGON ((-87.59215 41.81693, -87.59215 41.816...
        [5 rows x 87 columns]

        >>> groceries.head()  # doctest: +SKIP
           OBJECTID     Ycoord  ...  Category                           geometry
        0        16  41.973266  ...       NaN  MULTIPOINT ((-87.65661 41.97321))
        1        18  41.696367  ...       NaN  MULTIPOINT ((-87.68136 41.69713))
        2        22  41.868634  ...       NaN  MULTIPOINT ((-87.63918 41.86847))
        3        23  41.877590  ...       new  MULTIPOINT ((-87.65495 41.87783))
        4        27  41.737696  ...       NaN  MULTIPOINT ((-87.62715 41.73623))
        [5 rows x 8 columns]

        >>> groceries_w_communities = groceries.sjoin_nearest(chicago)
        >>> groceries_w_communities[["Chain", "community", "geometry"]].head(2)
                       Chain    community                                geometry
        0     VIET HOA PLAZA       UPTOWN   MULTIPOINT ((1168268.672 1933554.35))
        1  COUNTY FAIR FOODS  MORGAN PARK  MULTIPOINT ((1162302.618 1832900.224))


        To include the distances:

        >>> groceries_w_communities = groceries.sjoin_nearest(chicago, \
distance_col="distances")
        >>> groceries_w_communities[["Chain", "community", \
"distances"]].head(2)
                       Chain    community  distances
        0     VIET HOA PLAZA       UPTOWN        0.0
        1  COUNTY FAIR FOODS  MORGAN PARK        0.0

        In the following example, we get multiple groceries for Uptown because all
        results are equidistant (in this case zero because they intersect).
        In fact, we get 4 results in total:

        >>> chicago_w_groceries = groceries.sjoin_nearest(chicago, \
distance_col="distances", how="right")
        >>> uptown_results = \
chicago_w_groceries[chicago_w_groceries["community"] == "UPTOWN"]
        >>> uptown_results[["Chain", "community"]]
                    Chain community
        30  VIET HOA PLAZA    UPTOWN
        30      JEWEL OSCO    UPTOWN
        30          TARGET    UPTOWN
        30       Mariano's    UPTOWN

        See Also
        --------
        GeoDataFrame.sjoin : binary predicate joins
        sjoin_nearest : equivalent top-level function

        Notes
        -----
        Since this join relies on distances, results will be inaccurate
        if your geometries are in a geographic CRS.

        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.
        """
        return geopandas.sjoin_nearest(
            self,
            right,
            how=how,
            max_distance=max_distance,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            distance_col=distance_col,
            exclusive=exclusive,
        )

    def clip(
        self, mask, keep_geom_type: bool = False, sort: bool = False
    ) -> GeoDataFrame:
        """Clip points, lines, or polygon geometries to the mask extent.

        Both layers must be in the same Coordinate Reference System (CRS).
        The GeoDataFrame will be clipped to the full extent of the ``mask`` object.

        If there are multiple polygons in mask, data from the GeoDataFrame will be
        clipped to the total boundary of all polygons in mask.

        Parameters
        ----------
        mask : GeoDataFrame, GeoSeries, (Multi)Polygon, list-like
            Polygon vector layer used to clip the GeoDataFrame.
            The mask's geometry is dissolved into one geometric feature
            and intersected with GeoDataFrame.
            If the mask is list-like with four elements ``(minx, miny, maxx, maxy)``,
            ``clip`` will use a faster rectangle clipping
            (:meth:`~GeoSeries.clip_by_rect`), possibly leading to slightly different
            results.
        keep_geom_type : boolean, default False
            If True, return only geometries of original type in case of intersection
            resulting in multiple geometry types or GeometryCollections.
            If False, return all resulting geometries (potentially mixed types).
        sort : boolean, default False
            If True, the order of rows in the clipped GeoDataFrame will be preserved at
            small performance cost. If False the order of rows in the clipped
            GeoDataFrame will be random.

        Returns
        -------
        GeoDataFrame
            Vector data (points, lines, polygons) from the GeoDataFrame clipped to
            polygon boundary from mask.

        See Also
        --------
        clip : equivalent top-level function

        Examples
        --------
        Clip points (grocery stores) with polygons (the Near West Side community):

        >>> import geodatasets
        >>> chicago = geopandas.read_file(
        ...     geodatasets.get_path("geoda.chicago_health")
        ... )
        >>> near_west_side = chicago[chicago["community"] == "NEAR WEST SIDE"]
        >>> groceries = geopandas.read_file(
        ...     geodatasets.get_path("geoda.groceries")
        ... ).to_crs(chicago.crs)
        >>> groceries.shape
        (148, 8)

        >>> nws_groceries = groceries.clip(near_west_side)
        >>> nws_groceries.shape
        (7, 8)
        """
        return geopandas.clip(self, mask=mask, keep_geom_type=keep_geom_type, sort=sort)

    def overlay(
        self,
        right: GeoDataFrame,
        how: Literal[
            "intersection", "union", "identity", "symmetric_difference", "difference"
        ] = "intersection",
        keep_geom_type: bool | None = None,
        make_valid: bool = True,
    ):
        """Perform spatial overlay between GeoDataFrames.

        Currently only supports data GeoDataFrames with uniform geometry types,
        i.e. containing only (Multi)Polygons, or only (Multi)Points, or a
        combination of (Multi)LineString and LinearRing shapes.
        Implements several methods that are all effectively subsets of the union.

        See the User Guide page :doc:`../../user_guide/set_operations` for details.

        Parameters
        ----------
        right : GeoDataFrame
        how : string
            Method of spatial overlay: 'intersection', 'union',
            'identity', 'symmetric_difference' or 'difference'.
        keep_geom_type : bool
            If True, return only geometries of the same geometry type the GeoDataFrame
            has, if False, return all resulting geometries. Default is None,
            which will set keep_geom_type to True but warn upon dropping
            geometries.
        make_valid : bool, default True
            If True, any invalid input geometries are corrected with a call to
            make_valid(), if False, a `ValueError` is raised if any input geometries
            are invalid.

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

        >>> df1.overlay(df2, how='union')
           df1_data  df2_data                                           geometry
        0       1.0       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
        1       2.0       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
        2       2.0       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
        3       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
        4       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
        5       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
        6       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

        >>> df1.overlay(df2, how='intersection')
           df1_data  df2_data                             geometry
        0         1         1  POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
        1         2         1  POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
        2         2         2  POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))

        >>> df1.overlay(df2, how='symmetric_difference')
           df1_data  df2_data                                           geometry
        0       1.0       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
        1       2.0       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...
        2       NaN       1.0  MULTIPOLYGON (((2 3, 2 2, 1 2, 1 3, 2 3)), ((3...
        3       NaN       2.0      POLYGON ((3 5, 5 5, 5 3, 4 3, 4 4, 3 4, 3 5))

        >>> df1.overlay(df2, how='difference')
                                                    geometry  df1_data
        0      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))         1
        1  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...         2

        >>> df1.overlay(df2, how='identity')
           df1_data  df2_data                                           geometry
        0         1       1.0                POLYGON ((2 2, 2 1, 1 1, 1 2, 2 2))
        1         2       1.0                POLYGON ((2 2, 2 3, 3 3, 3 2, 2 2))
        2         2       2.0                POLYGON ((4 4, 4 3, 3 3, 3 4, 4 4))
        3         1       NaN      POLYGON ((2 0, 0 0, 0 2, 1 2, 1 1, 2 1, 2 0))
        4         2       NaN  MULTIPOLYGON (((3 4, 3 3, 2 3, 2 4, 3 4)), ((4...

        See Also
        --------
        GeoDataFrame.sjoin : spatial join
        overlay : equivalent top-level function

        Notes
        -----
        Every operation in GeoPandas is planar, i.e. the potential third
        dimension is not taken into account.
        """
        return geopandas.overlay(
            self, right, how=how, keep_geom_type=keep_geom_type, make_valid=make_valid
        )


def _dataframe_set_geometry(
    self,
    col,
    drop: bool | None = None,
    inplace: Literal[False] = False,
    crs: Any | None = None,
) -> GeoDataFrame:
    if inplace:
        raise ValueError(
            "Can't do inplace setting when converting from DataFrame to GeoDataFrame"
        )
    gf = GeoDataFrame(self)
    # this will copy so that BlockManager gets copied
    return gf.set_geometry(col, drop=drop, inplace=False, crs=crs)


DataFrame.set_geometry = _dataframe_set_geometry
