from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype

from vibespatial.api._native_rowset import NativeDeviceSelection, NativeRowSet
from vibespatial.geometry.buffers import GeometryFamily
from vibespatial.geometry.owned import FAMILY_TAGS, NULL_TAG
from vibespatial.runtime.materialization import (
    NativeExportBoundary,
    record_native_export_boundary,
)
from vibespatial.runtime.residency import Residency

_TAG_TO_GEOM_TYPE_NAME: dict[int, str | None] = {
    FAMILY_TAGS[GeometryFamily.POINT]: "Point",
    FAMILY_TAGS[GeometryFamily.LINESTRING]: "LineString",
    FAMILY_TAGS[GeometryFamily.POLYGON]: "Polygon",
    FAMILY_TAGS[GeometryFamily.MULTIPOINT]: "MultiPoint",
    FAMILY_TAGS[GeometryFamily.MULTILINESTRING]: "MultiLineString",
    FAMILY_TAGS[GeometryFamily.MULTIPOLYGON]: "MultiPolygon",
    NULL_TAG: None,
}

_GEOM_TYPE_TO_TAG: dict[str | None, int] = {
    name: tag for tag, name in _TAG_TO_GEOM_TYPE_NAME.items()
}
_TAG_TO_FAMILY: dict[int, GeometryFamily] = {
    FAMILY_TAGS[family]: family for family in GeometryFamily
}


def _is_device_array(values: Any) -> bool:
    return hasattr(values, "__cuda_array_interface__")


def _array_size(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(values)


def _extension_array_getitem(array: ExtensionArray, item, materialize):
    """Keep vector indexing inside the ExtensionArray contract."""
    if isinstance(item, (int, np.integer)):
        return materialize()[int(item)]
    if isinstance(item, slice) and item == slice(None):
        return array.copy()
    positions = np.arange(len(array), dtype=np.intp)[item]
    if np.ndim(positions) == 0:
        return materialize()[int(positions)]
    positions = np.asarray(positions, dtype=np.intp)
    if positions.ndim != 1:
        raise IndexError("native public arrays are one-dimensional")
    return array.take(positions, allow_fill=False)


def _copy_device_to_host(values: Any, *, reason: str):
    from vibespatial.cuda._runtime import get_cuda_runtime

    return get_cuda_runtime().copy_device_to_host(values, reason=reason)


def _unique_first(values: Any):
    """Return values unique by first occurrence plus first-occurrence indices."""
    if _is_device_array(values):
        import cupy as cp

        d_values = cp.asarray(values)
        if int(d_values.size) == 0:
            return d_values, cp.asarray([], dtype=cp.int64)
        _unique_values, first_indices = cp.unique(d_values, return_index=True)
        first_indices = cp.sort(first_indices.astype(cp.int64, copy=False))
        return d_values[first_indices], first_indices

    host_values = np.asarray(values)
    if host_values.size == 0:
        return host_values, np.asarray([], dtype=np.int64)
    _unique_values, first_indices = np.unique(host_values, return_index=True)
    first_indices = np.sort(first_indices.astype(np.int64, copy=False))
    return host_values[first_indices], first_indices


def _unique_grouped_first(values: Any):
    """Deduplicate a grouped vector without allocating a global sort."""
    if _is_device_array(values):
        import cupy as cp

        d_values = cp.asarray(values)
        if int(d_values.size) == 0:
            return d_values, cp.asarray([], dtype=cp.int64)
        first = cp.empty(d_values.size, dtype=cp.bool_)
        first[0] = True
        first[1:] = d_values[1:] != d_values[:-1]
        first_indices = cp.flatnonzero(first).astype(cp.int64, copy=False)
        return d_values[first_indices], first_indices

    host_values = np.asarray(values)
    if host_values.size == 0:
        return host_values, np.asarray([], dtype=np.int64)
    first = np.empty(host_values.size, dtype=bool)
    first[0] = True
    first[1:] = host_values[1:] != host_values[:-1]
    first_indices = np.flatnonzero(first).astype(np.int64, copy=False)
    return host_values[first_indices], first_indices


def _take_optional(values: Any | None, indices: Any) -> Any | None:
    if values is None:
        return None
    if _is_device_array(values) or _is_device_array(indices):
        import cupy as cp

        return cp.asarray(values)[cp.asarray(indices, dtype=cp.int64)]
    return np.asarray(values)[np.asarray(indices, dtype=np.int64)]


def _native_tags_for_state(state) -> Any | None:
    tags = (
        getattr(state.provenance, "part_family_tags", None)
        if getattr(state, "provenance", None) is not None
        else None
    )
    if tags is not None:
        return tags
    geometry = getattr(state, "geometry", None)
    owned = (
        geometry.cached_owned()
        if geometry is not None and hasattr(geometry, "cached_owned")
        else getattr(geometry, "owned", None)
    )
    if owned is None:
        return None
    if getattr(owned, "residency", None) is Residency.DEVICE:
        return owned._ensure_device_state(preserve_indexed_view=True).tags
    return owned.tags


def _requested_tags(values) -> tuple[int, ...] | None:
    try:
        requested = set(values)
    except TypeError:
        return None
    if not requested:
        return ()
    if any(value not in _GEOM_TYPE_TO_TAG for value in requested):
        return None
    return tuple(tag for value, tag in _GEOM_TYPE_TO_TAG.items() if value in requested)


def _family_domain_for_requested_tags(tags: tuple[int, ...]) -> tuple[GeometryFamily, ...]:
    return tuple(dict.fromkeys(_TAG_TO_FAMILY[tag] for tag in tags if tag in _TAG_TO_FAMILY))


def _host_geom_type_values(tags: Any) -> np.ndarray:
    host_tags = (
        np.asarray(
            _copy_device_to_host(
                tags,
                reason="native geometry type tag host export",
            ),
            dtype=np.int8,
        )
        if _is_device_array(tags)
        else np.asarray(tags, dtype=np.int8)
    )
    result = np.empty(host_tags.size, dtype=object)
    result[:] = None
    for tag, name in _TAG_TO_GEOM_TYPE_NAME.items():
        result[host_tags == np.int8(tag)] = name
    return result


def native_geometry_type_array_supported(state) -> bool:
    return _native_tags_for_state(state) is not None


class NativeGeometryTypeDtype(ExtensionDtype):
    type = object
    name = "vibespatial_geometry_type"
    kind = "O"
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return NativeGeometryTypeArray


class NativeBooleanMaskDtype(ExtensionDtype):
    type = np.bool_
    name = "vibespatial_bool"
    kind = "b"
    na_value = False
    _is_boolean = True

    @classmethod
    def construct_array_type(cls):
        return NativeBooleanMaskArray


class NativeNumericExpressionDtype(ExtensionDtype):
    _is_numeric = True

    def __init__(self, numpy_dtype: Any = "float64") -> None:
        self.numpy_dtype = np.dtype(numpy_dtype)

    @property
    def type(self):
        return self.numpy_dtype.type

    @property
    def name(self) -> str:
        return f"vibespatial_{self.numpy_dtype.name}"

    @property
    def kind(self) -> str:
        return self.numpy_dtype.kind

    @property
    def na_value(self):
        return np.nan

    def __eq__(self, other) -> bool:
        if isinstance(other, NativeNumericExpressionDtype):
            return self.numpy_dtype == other.numpy_dtype
        try:
            return self.numpy_dtype == np.dtype(other)
        except TypeError:
            return False

    def __hash__(self) -> int:
        return hash((type(self), self.numpy_dtype.str))

    @classmethod
    def construct_array_type(cls):
        return NativeNumericExpressionArray


class NativeIndexLabelsDtype(ExtensionDtype):
    type = object
    name = "vibespatial_index"
    kind = "O"
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return NativeIndexLabelsArray


class NativeAttributeColumnDtype(ExtensionDtype):
    type = object
    name = "vibespatial_attribute"
    kind = "O"
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return NativeAttributeColumnArray


class NativeIndex(pd.Index):
    """Public Index wrapper whose metadata probes stay on ``NativeIndexPlan``."""

    def equals(self, other) -> bool:
        if self is other:
            return True
        if not isinstance(other, pd.Index):
            try:
                other = pd.Index(other)
            except Exception:
                return False
        if len(self) != len(other) or list(self.names) != list(other.names):
            return False

        values = getattr(self, "_values", None)
        other_values = getattr(other, "_values", None)
        if isinstance(values, NativeIndexLabelsArray) and isinstance(
            other_values,
            NativeIndexLabelsArray,
        ):
            if values.index_plan is other_values.index_plan:
                return True

        self_index = pd.Index(np.asarray(values), name=self.name)
        if isinstance(other_values, NativeIndexLabelsArray):
            other = pd.Index(np.asarray(other_values), name=other.name)
        return bool(self_index.equals(other))

    @property
    def is_unique(self) -> bool:
        values = getattr(self, "_values", None)
        if isinstance(values, NativeIndexLabelsArray):
            return not bool(getattr(values.index_plan, "has_duplicates", False))
        return super().is_unique

    @property
    def has_duplicates(self) -> bool:
        return not self.is_unique


def _native_expression_for_value(value: Any):
    if isinstance(value, NativeNumericExpressionArray):
        return value.expression
    values = getattr(value, "array", None)
    if isinstance(values, NativeNumericExpressionArray):
        return values.expression
    return None


def _numpy_dtype_for_native_expression(expression: Any) -> np.dtype:
    dtype = getattr(expression, "dtype", None)
    if dtype:
        try:
            return np.dtype(dtype)
        except TypeError:
            pass
    values = getattr(expression, "values", None)
    try:
        return np.dtype(getattr(values, "dtype", np.float64))
    except TypeError:
        return np.dtype("float64")


def _operator_symbol(op: Any, *, comparison: bool = False) -> str | None:
    name = getattr(op, "__name__", "")
    if comparison:
        return {
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
            "eq": "==",
            "ne": "!=",
        }.get(name)
    return {
        "add": "+",
        "radd": "+",
        "sub": "-",
        "rsub": "-",
        "mul": "*",
        "rmul": "*",
        "truediv": "/",
        "rtruediv": "/",
        "floordiv": "//",
        "rfloordiv": "//",
        "mod": "%",
        "rmod": "%",
    }.get(name)


def _operator_reverse(op: Any) -> bool:
    return getattr(op, "__name__", "").startswith("r")


@dataclass
class NativeNumericExpressionArray(ExtensionArray):
    expression: Any
    export_surface: str = "vibespatial.api.NativeNumericExpressionArray"
    export_operation: str = "native_numeric_expression_to_public_array"

    @property
    def dtype(self):
        return NativeNumericExpressionDtype(_numpy_dtype_for_native_expression(self.expression))

    def __len__(self) -> int:
        return len(self.expression)

    def _materialize_values(self) -> np.ndarray:
        values = self.expression.values
        dtype = _numpy_dtype_for_native_expression(self.expression)
        if _is_device_array(values):
            import cupy as cp

            d_values = cp.asarray(values)
            record_native_export_boundary(
                NativeExportBoundary(
                    surface=self.export_surface,
                    operation=self.export_operation,
                    target="series",
                    reason="native numeric expression exported to public numeric array",
                    row_count=len(self),
                    byte_count=int(d_values.nbytes),
                    d2h_transfer=True,
                )
            )
            return np.asarray(
                _copy_device_to_host(
                    d_values,
                    reason=f"{self.export_surface}::{self.export_operation}",
                ),
                dtype=dtype,
            )
        return np.asarray(values, dtype=dtype)

    def __getitem__(self, item):
        return _extension_array_getitem(self, item, self._materialize_values)

    def __iter__(self):
        return iter(self._materialize_values())

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def to_numpy(self, dtype=None, copy: bool = False, na_value=np.nan):
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def isna(self) -> np.ndarray:
        values = self._materialize_values()
        if np.issubdtype(values.dtype, np.floating):
            return np.isnan(values)
        return pd.isna(values)

    def equals(self, other: object) -> bool:
        """Compare public numeric values without requiring matching array classes."""
        other_dtype = getattr(other, "dtype", None)
        if other_dtype is None or self.dtype != other_dtype:
            return False
        if len(self) != len(other):
            return False

        other_values = (
            other.expression.values
            if isinstance(other, NativeNumericExpressionArray)
            else other
        )
        values = self.expression.values
        if _is_device_array(values) or _is_device_array(other_values):
            import cupy as cp

            d_values = cp.asarray(values).reshape(-1)
            d_other = cp.asarray(other_values).reshape(-1)
            if d_values.shape != d_other.shape:
                return False
            d_equal = d_values == d_other
            if d_values.dtype.kind in "fc":
                d_equal |= cp.isnan(d_values) & cp.isnan(d_other)
            d_result = cp.all(d_equal).reshape(1)
            record_native_export_boundary(
                NativeExportBoundary(
                    surface=self.export_surface,
                    operation=f"{self.export_operation}_equals",
                    target="scalar",
                    reason="native numeric equality exported to public scalar",
                    row_count=len(self),
                    byte_count=int(d_result.nbytes),
                    d2h_transfer=True,
                )
            )
            return bool(
                np.asarray(
                    _copy_device_to_host(
                        d_result,
                        reason=(
                            f"{self.export_surface}::"
                            f"{self.export_operation}_equals"
                        ),
                    )
                )[0]
            )

        host_values = np.asarray(values).reshape(-1)
        host_other = np.asarray(other_values).reshape(-1)
        if host_values.shape != host_other.shape:
            return False
        equal = host_values == host_other
        if host_values.dtype.kind in "fc":
            equal |= np.isnan(host_values) & np.isnan(host_other)
        return bool(np.all(equal))

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs):
        if name not in {"any", "all", "sum", "prod", "min", "max", "mean"}:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} does not support "
                f"operation '{name}'"
            )

        min_count = int(kwargs.get("min_count", 0) or 0)
        values = self.expression.values
        if _is_device_array(values):
            import cupy as cp

            d_source = cp.asarray(values)
            source_dtype = np.dtype(d_source.dtype)
            if source_dtype.kind in "iub":
                if int(d_source.size) == 0:
                    if name == "sum":
                        result = 0 if min_count == 0 else np.nan
                    elif name == "prod":
                        result = 1 if min_count == 0 else np.nan
                    elif name in {"mean", "min", "max"}:
                        result = np.nan
                    elif name == "any":
                        result = False
                    else:
                        result = True
                    if keepdims:
                        return type(self)._from_sequence([result])
                    return result

                if name == "sum":
                    if int(d_source.size) < min_count:
                        reduced = cp.asarray(np.nan, dtype=cp.float64)
                    else:
                        reduced = cp.sum(
                            d_source,
                            dtype=cp.int64 if source_dtype.kind == "b" else None,
                        )
                elif name == "prod":
                    if int(d_source.size) < min_count:
                        reduced = cp.asarray(np.nan, dtype=cp.float64)
                    else:
                        reduced = cp.prod(
                            d_source,
                            dtype=cp.int64 if source_dtype.kind == "b" else None,
                        )
                elif name == "mean":
                    reduced = cp.mean(d_source, dtype=cp.float64)
                elif name == "min":
                    reduced = cp.min(d_source)
                elif name == "max":
                    reduced = cp.max(d_source)
                elif name == "any":
                    reduced = cp.any(d_source != 0)
                else:
                    reduced = cp.all(d_source != 0)

                record_native_export_boundary(
                    NativeExportBoundary(
                        surface=self.export_surface,
                        operation=f"{self.export_operation}_{name}",
                        target="scalar",
                        reason="native numeric expression reduced to public scalar",
                        row_count=len(self),
                        d2h_transfer=True,
                    )
                )
                result = np.asarray(
                    _copy_device_to_host(
                        cp.asarray(reduced).reshape(1),
                        reason=f"{self.export_surface}::{self.export_operation}_{name}",
                    )
                )[0]
                if name in {"any", "all"} or (
                    source_dtype.kind == "b" and name in {"min", "max"}
                ):
                    result = bool(result)
                elif name == "mean" or isinstance(result, np.floating):
                    result = float(result)
                else:
                    result = int(result)
                if keepdims:
                    return type(self)._from_sequence([result])
                return result

            d_values = d_source.astype(cp.float64, copy=False)
            if int(d_values.size) == 0:
                if name == "sum":
                    result = 0.0 if min_count == 0 else np.nan
                elif name == "prod":
                    result = 1.0 if min_count == 0 else np.nan
                elif name in {"mean", "min", "max"}:
                    result = np.nan
                elif name == "any":
                    result = False
                else:
                    result = True
                if keepdims:
                    return type(self)._from_sequence([result])
                return result
            if skipna:
                d_valid = ~cp.isnan(d_values)
                d_valid_count = cp.sum(d_valid, dtype=cp.int64)
                if name == "sum":
                    reduced = cp.nansum(d_values, dtype=cp.float64)
                    reduced = cp.where(d_valid_count >= min_count, reduced, cp.nan)
                elif name == "prod":
                    reduced = cp.nanprod(d_values, dtype=cp.float64)
                    reduced = cp.where(d_valid_count >= min_count, reduced, cp.nan)
                elif name == "mean":
                    reduced = cp.where(
                        d_valid_count > 0,
                        cp.nansum(d_values, dtype=cp.float64) / d_valid_count,
                        cp.nan,
                    )
                elif name == "min":
                    reduced = cp.min(cp.where(d_valid, d_values, cp.inf))
                    reduced = cp.where(d_valid_count > 0, reduced, cp.nan)
                elif name == "max":
                    reduced = cp.max(cp.where(d_valid, d_values, -cp.inf))
                    reduced = cp.where(d_valid_count > 0, reduced, cp.nan)
                elif name == "any":
                    reduced = cp.any(d_valid & (d_values != 0.0))
                else:
                    reduced = cp.all(cp.where(d_valid, d_values != 0.0, True))
            else:
                if name == "sum":
                    reduced = cp.sum(d_values, dtype=cp.float64)
                elif name == "prod":
                    reduced = cp.prod(d_values, dtype=cp.float64)
                elif name == "mean":
                    reduced = cp.mean(d_values)
                elif name == "min":
                    reduced = cp.min(d_values)
                elif name == "max":
                    reduced = cp.max(d_values)
                elif name == "any":
                    reduced = cp.any(d_values != 0.0)
                else:
                    reduced = cp.all(d_values != 0.0)

            record_native_export_boundary(
                NativeExportBoundary(
                    surface=self.export_surface,
                    operation=f"{self.export_operation}_{name}",
                    target="scalar",
                    reason="native numeric expression reduced to public scalar",
                    row_count=len(self),
                    d2h_transfer=True,
                )
            )
            result = np.asarray(
                _copy_device_to_host(
                    cp.asarray(reduced).reshape(1),
                    reason=f"{self.export_surface}::{self.export_operation}_{name}",
                )
            )[0]
            if name in {"any", "all"}:
                result = bool(result)
            else:
                result = float(result)
        else:
            host_source = np.asarray(values)
            source_dtype = host_source.dtype
            if source_dtype.kind in "iub":
                if host_source.size == 0:
                    if name == "sum":
                        result = 0 if min_count == 0 else np.nan
                    elif name == "prod":
                        result = 1 if min_count == 0 else np.nan
                    elif name in {"mean", "min", "max"}:
                        result = np.nan
                    elif name == "any":
                        result = False
                    else:
                        result = True
                elif name == "sum":
                    result = (
                        int(
                            np.sum(
                                host_source,
                                dtype=(
                                    np.int64 if source_dtype.kind == "b" else None
                                ),
                            )
                        )
                        if host_source.size >= min_count
                        else np.nan
                    )
                elif name == "prod":
                    result = (
                        int(
                            np.prod(
                                host_source,
                                dtype=(
                                    np.int64 if source_dtype.kind == "b" else None
                                ),
                            )
                        )
                        if host_source.size >= min_count
                        else np.nan
                    )
                elif name == "mean":
                    result = float(np.mean(host_source, dtype=np.float64))
                elif name == "min":
                    reduced = np.min(host_source)
                    result = (
                        bool(reduced)
                        if source_dtype.kind == "b"
                        else int(reduced)
                    )
                elif name == "max":
                    reduced = np.max(host_source)
                    result = (
                        bool(reduced)
                        if source_dtype.kind == "b"
                        else int(reduced)
                    )
                elif name == "any":
                    result = bool(np.any(host_source != 0))
                else:
                    result = bool(np.all(host_source != 0))
                if keepdims:
                    return type(self)._from_sequence([result])
                return result

            host_values = host_source.astype(np.float64, copy=False)
            if host_values.size == 0:
                if name == "sum":
                    result = 0.0 if min_count == 0 else np.nan
                elif name == "prod":
                    result = 1.0 if min_count == 0 else np.nan
                elif name in {"mean", "min", "max"}:
                    result = np.nan
                elif name == "any":
                    result = False
                else:
                    result = True
                if keepdims:
                    return type(self)._from_sequence([result])
                return result
            if skipna:
                valid = ~np.isnan(host_values)
                valid_count = int(np.count_nonzero(valid))
                if name == "sum":
                    result = (
                        float(np.nansum(host_values, dtype=np.float64))
                        if valid_count >= min_count
                        else np.nan
                    )
                elif name == "prod":
                    result = (
                        float(np.nanprod(host_values, dtype=np.float64))
                        if valid_count >= min_count
                        else np.nan
                    )
                elif name == "mean":
                    result = (
                        float(np.nansum(host_values, dtype=np.float64) / valid_count)
                        if valid_count > 0
                        else np.nan
                    )
                elif name == "min":
                    result = float(np.min(host_values[valid])) if valid_count else np.nan
                elif name == "max":
                    result = float(np.max(host_values[valid])) if valid_count else np.nan
                elif name == "any":
                    result = bool(np.any(valid & (host_values != 0.0)))
                else:
                    result = bool(np.all(np.where(valid, host_values != 0.0, True)))
            else:
                if name == "sum":
                    result = float(np.sum(host_values, dtype=np.float64))
                elif name == "prod":
                    result = float(np.prod(host_values, dtype=np.float64))
                elif name == "mean":
                    result = float(np.mean(host_values))
                elif name == "min":
                    result = float(np.min(host_values))
                elif name == "max":
                    result = float(np.max(host_values))
                elif name == "any":
                    result = bool(np.any(host_values != 0.0))
                else:
                    result = bool(np.all(host_values != 0.0))

        if keepdims:
            return type(self)._from_sequence([result])
        return result

    def copy(self) -> NativeNumericExpressionArray:
        return type(self)(
            expression=self.expression,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
        )

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        if allow_fill:
            index_values = np.asarray(indices, dtype=np.intp)
            if not np.any(index_values < 0):
                return self.take(index_values, allow_fill=False)
        if not allow_fill:
            values = self.expression.values
            if _is_device_array(values):
                import cupy as cp

                from vibespatial.api._native_expression import NativeExpression

                d_indices = cp.asarray(indices, dtype=cp.int64)
                taken_values = cp.asarray(values)[d_indices]
                expression = NativeExpression(
                    operation=f"{self.expression.operation}.take",
                    values=taken_values,
                    source_token=None,
                    source_row_count=int(d_indices.size),
                    dtype=str(getattr(taken_values, "dtype", "")) or self.expression.dtype,
                    precision=self.expression.precision,
                    null_policy=self.expression.null_policy,
                    readiness=self.expression.readiness,
                )
                return type(self)(
                    expression=expression,
                    export_surface=self.export_surface,
                    export_operation=self.export_operation,
                )
            from vibespatial.api._native_expression import NativeExpression

            host_indices = np.asarray(indices, dtype=np.intp)
            taken_values = np.asarray(values)[host_indices]
            expression = NativeExpression(
                operation=f"{self.expression.operation}.take",
                values=taken_values,
                source_token=None,
                source_row_count=int(taken_values.size),
                dtype=str(getattr(taken_values, "dtype", "")) or self.expression.dtype,
                precision=self.expression.precision,
                null_policy=self.expression.null_policy,
                readiness=self.expression.readiness,
            )
            return type(self)(
                expression=expression,
                export_surface=self.export_surface,
                export_operation=self.export_operation,
            )
        values = self._materialize_values()
        from pandas.api.extensions import take

        taken = take(
            values,
            indices,
            allow_fill=True,
            fill_value=np.nan if fill_value is None else fill_value,
        )
        return type(self)._from_sequence(taken, dtype=values.dtype)

    def _arith_method(self, other, op):
        symbol = _operator_symbol(op)
        if symbol is None:
            return NotImplemented
        other_expression = _native_expression_for_value(other)
        operand = other_expression if other_expression is not None else other
        expression = self.expression.binary_arithmetic(
            symbol,
            operand,
            reverse=_operator_reverse(op),
        )
        if expression is None:
            values = self._materialize_values()
            other_values = (
                other._materialize_values()
                if isinstance(other, NativeNumericExpressionArray)
                else other
            )
            return op(values, other_values)
        return type(self)(
            expression=expression,
            export_surface=self.export_surface,
            export_operation=f"{self.export_operation}_arithmetic",
        )

    def _cmp_method(self, other, op):
        symbol = _operator_symbol(op, comparison=True)
        if symbol is None:
            return NotImplemented
        other_expression = _native_expression_for_value(other)
        operand = other_expression if other_expression is not None else other
        rowset = self.expression.compare(
            symbol,
            operand,
            reverse=_operator_reverse(op),
        )
        if rowset is None:
            values = self._materialize_values()
            other_values = (
                other._materialize_values()
                if isinstance(other, NativeNumericExpressionArray)
                else other
            )
            return op(values, other_values)

        if rowset.is_device:
            import cupy as cp

            mask_values = cp.zeros(len(self), dtype=cp.bool_)
            if len(rowset) > 0:
                mask_values[cp.asarray(rowset.positions, dtype=cp.int64)] = True
        else:
            mask_values = np.zeros(len(self), dtype=bool)
            if len(rowset) > 0:
                mask_values[np.asarray(rowset.positions, dtype=np.int64)] = True
        return NativeBooleanMaskArray(
            row_count=len(self),
            rowset=rowset,
            mask_values=mask_values,
            export_surface=self.export_surface,
            export_operation=f"{self.export_operation}_compare",
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        if isinstance(dtype, NativeNumericExpressionDtype):
            dtype = dtype.numpy_dtype
        values = np.asarray(list(scalars), dtype=dtype)
        if copy:
            values = values.copy()
        from vibespatial.api._native_expression import NativeExpression

        return cls(
            NativeExpression(
                operation="public_sequence",
                values=values,
                source_token=None,
                source_row_count=int(values.size),
                dtype=str(values.dtype),
                precision="source",
            )
        )

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_values() for array in to_concat])
        return cls._from_sequence(values, dtype=values.dtype)


@dataclass
class NativeIndexLabelsArray(ExtensionArray):
    index_plan: Any
    export_surface: str = "vibespatial.api.NativeIndexPlan.to_public_index"
    export_operation: str = "index_plan_to_host"
    _host_cache: np.ndarray | None = None

    @property
    def dtype(self):
        return NativeIndexLabelsDtype()

    def __len__(self) -> int:
        return int(self.index_plan.length)

    def _materialize_values(self) -> np.ndarray:
        if self._host_cache is not None:
            return self._host_cache

        index = getattr(self.index_plan, "index", None)
        if index is not None:
            self._host_cache = np.asarray(index)
            return self._host_cache

        device_labels = getattr(self.index_plan, "device_labels", None)
        if device_labels is not None and _is_device_array(device_labels):
            import cupy as cp

            d_labels = cp.asarray(device_labels)
            record_native_export_boundary(
                NativeExportBoundary(
                    surface=self.export_surface,
                    operation=self.export_operation,
                    target="index",
                    reason="device public index labels were materialized for export",
                    row_count=len(self),
                    byte_count=int(d_labels.nbytes),
                    d2h_transfer=True,
                )
            )
            self._host_cache = np.asarray(
                _copy_device_to_host(
                    d_labels,
                    reason=f"{self.export_surface}::{self.export_operation}",
                )
            )
            return self._host_cache

        source_index = getattr(self.index_plan, "source_index", None)
        take_positions = getattr(self.index_plan, "take_positions", None)
        if source_index is not None and take_positions is not None:
            operation = "index_plan_take_positions_to_host"
            if _is_device_array(take_positions):
                import cupy as cp

                d_positions = cp.asarray(take_positions)
                record_native_export_boundary(
                    NativeExportBoundary(
                        surface="vibespatial.api.NativeIndexPlan.take_public_index",
                        operation=operation,
                        target="index",
                        reason="device row positions were materialized to take host public index labels",
                        row_count=len(self),
                        byte_count=int(d_positions.nbytes),
                        d2h_transfer=True,
                    )
                )
                host_positions = np.asarray(
                    _copy_device_to_host(
                        d_positions,
                        reason=(f"vibespatial.api.NativeIndexPlan.take_public_index::{operation}"),
                    ),
                    dtype=np.int64,
                )
            else:
                host_positions = np.asarray(take_positions, dtype=np.int64)
            self._host_cache = np.asarray(source_index.take(host_positions))
            return self._host_cache

        public_index = self.index_plan.to_public_index(strict_disallowed=False)
        self._host_cache = np.asarray(public_index)
        return self._host_cache

    def __getitem__(self, item):
        return _extension_array_getitem(self, item, self._materialize_values)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def to_numpy(self, dtype=None, copy: bool = False, na_value=None):
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def isna(self) -> np.ndarray:
        return pd.isna(self._materialize_values())

    def copy(self) -> NativeIndexLabelsArray:
        return type(self)(
            index_plan=self.index_plan,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
            _host_cache=None if self._host_cache is None else self._host_cache.copy(),
        )

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        from pandas.api.extensions import take

        values = self._materialize_values()
        taken = take(
            values,
            indices,
            allow_fill=allow_fill,
            fill_value=fill_value,
        )
        return type(self)._from_sequence(taken)

    def unique(self):
        plan = self.index_plan
        selection_positions = getattr(plan, "selection_positions", None)
        if selection_positions is not None:
            unique_fn = (
                _unique_grouped_first
                if getattr(plan, "selection_grouped", False)
                else _unique_first
            )
            unique_selection, first_indices = unique_fn(selection_positions)
            label_plan = plan.take(
                first_indices,
                preserve_index=True,
                unique=True,
            )
            new_plan = replace(
                label_plan,
                selection_source_token=plan.selection_source_token,
                selection_source_row_count=plan.selection_source_row_count,
                selection_positions=unique_selection,
                selection_grouped=True,
                has_duplicates=False,
            )
            return type(self)(
                index_plan=new_plan,
                export_surface=self.export_surface,
                export_operation=self.export_operation,
            )

        device_labels = getattr(plan, "device_labels", None)
        if device_labels is not None and _is_device_array(device_labels):
            unique_labels, first_indices = _unique_first(device_labels)
            new_plan = replace(
                plan,
                length=_array_size(unique_labels),
                index=None,
                device_labels=unique_labels,
                take_positions=_take_optional(
                    getattr(plan, "take_positions", None),
                    first_indices,
                ),
                has_duplicates=False,
            )
            return type(self)(
                index_plan=new_plan,
                export_surface=self.export_surface,
                export_operation=self.export_operation,
            )

        source_index = getattr(plan, "source_index", None)
        take_positions = getattr(plan, "take_positions", None)
        if source_index is not None and take_positions is not None and source_index.is_unique:
            unique_positions, _first_indices = _unique_first(take_positions)
            new_plan = replace(
                plan,
                length=_array_size(unique_positions),
                index=None,
                take_positions=unique_positions,
                has_duplicates=False,
            )
            return type(self)(
                index_plan=new_plan,
                export_surface=self.export_surface,
                export_operation=self.export_operation,
            )

        return pd.array(pd.unique(self._materialize_values()), dtype=object)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        values = np.asarray(list(scalars), dtype=object)
        if copy:
            values = values.copy()
        from vibespatial.api._native_rowset import NativeIndexPlan

        return cls(NativeIndexPlan.from_index(pd.Index(values)))

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values)

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_values() for array in to_concat])
        return cls._from_sequence(values)


@dataclass
class NativeAttributeColumnArray(ExtensionArray):
    table: Any
    column: Any
    export_surface: str = "vibespatial.api.NativeAttributeTable.public_column"
    export_operation: str = "native_attribute_column_to_public_array"
    selection_positions: Any | None = None
    source_token: str | None = None

    @property
    def dtype(self):
        return NativeAttributeColumnDtype()

    def __len__(self) -> int:
        return (
            len(self.table)
            if self.selection_positions is None
            else _array_size(self.selection_positions)
        )

    def _device_selected_table(self):
        if self.selection_positions is None:
            return self.table
        positions = self.selection_positions
        if not _is_device_array(positions):
            host_positions = np.asarray(positions, dtype=np.int64)
            if np.any(host_positions < 0):
                return None
            positions = host_positions
        return self.table.take(positions, preserve_index=False)

    def _materialize_device_temporal_values(self) -> np.ndarray | None:
        table = self._device_selected_table()
        if table is None:
            return None
        policies = table.device_column_policies((self.column,))
        policy = policies.get(self.column)
        if (
            policy is None
            or int(policy.null_count) != 0
            or not (
                policy.arrow_type.startswith("timestamp[")
                or policy.arrow_type.startswith("duration[")
            )
        ):
            return None
        import cupy as cp

        column = table.to_pylibcudf_columns((self.column,))[0]
        values = cp.asarray(column.data()).view(cp.int64)[: int(column.size())]
        arrow_type = policy.arrow_type
        temporal_metadata = arrow_type[
            arrow_type.find("[") + 1 : arrow_type.rfind("]")
        ]
        unit, *metadata = (
            value.strip() for value in temporal_metadata.split(",")
        )
        kind = "datetime64" if arrow_type.startswith("timestamp[") else "timedelta64"
        record_native_export_boundary(
            NativeExportBoundary(
                surface=self.export_surface,
                operation=self.export_operation,
                target="series",
                reason="selected native temporal attribute exported to public array",
                row_count=len(self),
                byte_count=int(values.nbytes),
                d2h_transfer=True,
            )
        )
        host = _copy_device_to_host(
            values,
            reason=f"{self.export_surface}::{self.export_operation}_temporal",
        )
        host = np.asarray(host, dtype=np.int64)
        timezone = next(
            (
                value.removeprefix("tz=")
                for value in metadata
                if value.startswith("tz=")
            ),
            None,
        )
        if kind == "datetime64" and timezone is not None:
            return pd.to_datetime(host, unit=unit, utc=True).tz_convert(timezone).to_numpy()
        return host.view(np.dtype(f"{kind}[{unit}]"))

    def _materialize_values(self) -> np.ndarray:
        temporal = self._materialize_device_temporal_values()
        if temporal is not None:
            return temporal
        table = self._device_selected_table()
        if table is None:
            table = self.table
        projected = table.project_columns((self.column,))
        frame = (table if projected is None else projected).to_pandas(copy=False)
        values = frame[self.column].to_numpy(copy=False)
        values = np.asarray(values)
        if table is not self.table or self.selection_positions is None:
            return values
        positions = np.asarray(self.selection_positions, dtype=np.int64)
        result = np.empty(positions.size, dtype=object)
        result[:] = None
        selected = positions >= 0
        result[selected] = values[positions[selected]]
        return result

    def to_pandas_series(self, *, index=None, name=None) -> pd.Series:
        """Materialize the selected column while preserving its logical dtype."""
        temporal = self._materialize_device_temporal_values()
        if temporal is not None:
            result = pd.Series(temporal, copy=False)
            if index is not None and not result.index.equals(index):
                result.index = index
            result.name = name
            return result
        table = self._device_selected_table()
        if table is None:
            table = self.table
        projected = table.project_columns((self.column,))
        source = (table if projected is None else projected).to_pandas(copy=False)[self.column]
        if table is not self.table or self.selection_positions is None:
            result = source.copy(deep=False)
        else:
            positions = np.asarray(self.selection_positions, dtype=np.int64)
            result = pd.Series(
                source.array.take(
                    positions,
                    allow_fill=True,
                    fill_value=None,
                ),
                copy=False,
            )
        if index is not None and not result.index.equals(index):
            result.index = index
        result.name = name
        return result

    def __getitem__(self, item):
        return _extension_array_getitem(self, item, self._materialize_values)

    def __setitem__(self, key, value) -> None:
        """Detach and mutate this public compatibility column."""
        values = self.to_pandas_series().array.copy()
        values[key] = value

        from vibespatial.api._native_result_core import NativeAttributeTable

        self.table = NativeAttributeTable(
            dataframe=pd.DataFrame(
                {self.column: pd.Series(values, copy=False)},
            ),
        )
        self.selection_positions = None

    def __iter__(self):
        return iter(self._materialize_values())

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def to_numpy(self, dtype=None, copy: bool = False, na_value=None):
        if dtype is not None:
            cast = self.astype(dtype, copy=False)
            if isinstance(cast, NativeNumericExpressionArray):
                return cast.to_numpy(dtype=dtype, copy=copy, na_value=na_value)
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def isna(self) -> np.ndarray:
        return pd.isna(self._materialize_values())

    def astype(self, dtype, copy: bool = True):
        """Cast sortable device attributes without exporting through Arrow."""
        native_numeric_dtype = (
            dtype if isinstance(dtype, NativeNumericExpressionDtype) else None
        )
        cast_dtype = (
            native_numeric_dtype.numpy_dtype
            if native_numeric_dtype is not None
            else dtype
        )
        try:
            target_dtype = np.dtype(cast_dtype)
        except TypeError:
            target_dtype = None
        if target_dtype is not None and (
            np.issubdtype(target_dtype, np.number)
            or np.issubdtype(target_dtype, np.bool_)
        ):
            table = self._device_selected_table()
            if table is None:
                table = self.table
            policies = table.device_column_policies((self.column,))
            policy = policies.get(self.column)
            if policy is not None and policy.null_count == 0:
                if target_dtype == np.dtype(np.int64) and (
                    policy.arrow_type.startswith("timestamp[")
                    or policy.arrow_type.startswith("duration[")
                ):
                    import cupy as cp

                    from vibespatial.api._native_expression import NativeExpression

                    source = table.to_pylibcudf_columns((self.column,))[0]
                    offset = int(source.offset())
                    values = cp.asarray(source.data()).view(cp.int64)[
                        offset : offset + int(source.size())
                    ]
                    return NativeNumericExpressionArray(
                        NativeExpression(
                            operation=f"attribute.{self.column}.astype",
                            values=values,
                            source_token=self.source_token,
                            source_row_count=len(self),
                            dtype=str(target_dtype),
                            precision="source-temporal-unit",
                        ),
                        export_surface=self.export_surface,
                        export_operation=f"{self.export_operation}_astype",
                    )
                try:
                    import pyarrow as pa
                    import pylibcudf as plc

                    from vibespatial.api._native_expression import NativeExpression
                    from vibespatial.api._native_result_core import (
                        _pylibcudf_numeric_column_view,
                    )
                    from vibespatial.cuda._runtime import pylibcudf_current_stream

                    source = table.to_pylibcudf_columns((self.column,))[0]
                    target_type = plc.DataType.from_arrow(pa.from_numpy_dtype(target_dtype))
                    if plc.unary.is_supported_cast(source.type(), target_type):
                        cast = plc.unary.cast(
                            source,
                            target_type,
                            stream=pylibcudf_current_stream(table.device_table),
                        )
                        values = _pylibcudf_numeric_column_view(cast)
                        if values is not None:
                            return NativeNumericExpressionArray(
                                NativeExpression(
                                    operation=f"attribute.{self.column}.astype",
                                    values=values,
                                    source_token=self.source_token,
                                    source_row_count=len(self),
                                    dtype=str(target_dtype),
                                    precision="source",
                                ),
                                export_surface=self.export_surface,
                                export_operation=f"{self.export_operation}_astype",
                            )
                except (ImportError, AttributeError, NotImplementedError):
                    pass
        values = self._materialize_values().astype(cast_dtype, copy=copy)
        if native_numeric_dtype is not None:
            return NativeNumericExpressionArray._from_sequence(
                values,
                dtype=target_dtype,
                copy=False,
            )
        return values

    def _cmp_method(self, other, op):
        other_values = (
            other._materialize_values()
            if isinstance(other, NativeAttributeColumnArray)
            else other
        )
        return op(self._materialize_values(), other_values)

    def __eq__(self, other):
        import operator

        return self._cmp_method(other, operator.eq)

    def __ne__(self, other):
        import operator

        return self._cmp_method(other, operator.ne)

    def copy(self) -> NativeAttributeColumnArray:
        return type(self)(
            table=self.table,
            column=self.column,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
            selection_positions=self.selection_positions,
            source_token=self.source_token,
        )

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        positions = np.asarray(indices, dtype=np.int64)
        source_positions = (
            np.arange(len(self.table), dtype=np.int64)
            if self.selection_positions is None
            else np.asarray(self.selection_positions, dtype=np.int64)
        )
        if not allow_fill:
            return type(self)(
                table=self.table,
                column=self.column,
                export_surface=self.export_surface,
                export_operation=self.export_operation,
                selection_positions=source_positions[positions],
                source_token=self.source_token,
            )
        if fill_value is not None and not pd.isna(fill_value):
            from pandas.api.extensions import take

            taken = take(
                self._materialize_values(),
                positions,
                allow_fill=True,
                fill_value=fill_value,
            )
            return type(self)._from_sequence(taken)
        selected_positions = np.full(positions.size, -1, dtype=np.int64)
        selected = positions >= 0
        selected_positions[selected] = source_positions[positions[selected]]
        return type(self)(
            table=self.table,
            column=self.column,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
            selection_positions=selected_positions,
            source_token=self.source_token,
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        values = np.asarray(list(scalars), dtype=object)
        if copy:
            values = values.copy()
        from vibespatial.api._native_result_core import NativeAttributeTable

        column = "__native_attribute__"
        return cls(
            table=NativeAttributeTable(
                dataframe=pd.DataFrame({column: values}),
            ),
            column=column,
        )

    @classmethod
    def _from_factorized(cls, values, original):
        result = cls._from_sequence(values)
        result.export_surface = original.export_surface
        result.export_operation = original.export_operation
        return result

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_values() for array in to_concat])
        return cls._from_sequence(values)


def native_public_index_from_plan(index_plan) -> pd.Index | None:
    if index_plan is None:
        return None
    if getattr(index_plan, "nlevels", 1) != 1:
        if getattr(index_plan, "index", None) is not None:
            return index_plan.index
        return None
    has_selection_lineage = getattr(index_plan, "selection_positions", None) is not None
    if getattr(index_plan, "index", None) is not None and not has_selection_lineage:
        return index_plan.index
    if has_selection_lineage or getattr(index_plan, "device_labels", None) is not None or (
        getattr(index_plan, "source_index", None) is not None
        and getattr(index_plan, "take_positions", None) is not None
    ):
        index = pd.Index(
            NativeIndexLabelsArray(index_plan),
            name=getattr(index_plan, "name", None),
        )
        index.__class__ = NativeIndex
        return index
    return index_plan.to_public_index(strict_disallowed=False)


@dataclass
class NativeBooleanMaskArray(ExtensionArray):
    row_count: int
    rowset: NativeRowSet | None = None
    selection: NativeDeviceSelection | None = None
    mask_values: Any | None = None
    export_surface: str = "vibespatial.api.NativeBooleanMaskArray"
    export_operation: str = "native_boolean_mask_to_public_array"

    @property
    def dtype(self):
        return NativeBooleanMaskDtype()

    def __len__(self) -> int:
        return int(self.row_count)

    def _materialize_mask(self) -> np.ndarray:
        if self.mask_values is not None:
            values = self.mask_values
            if _is_device_array(values):
                record_native_export_boundary(
                    NativeExportBoundary(
                        surface=self.export_surface,
                        operation=self.export_operation,
                        target="series",
                        reason="native boolean mask exported to public boolean array",
                        row_count=self.row_count,
                        d2h_transfer=True,
                    )
                )
                return np.asarray(
                    _copy_device_to_host(
                        values,
                        reason=f"{self.export_surface}::{self.export_operation}",
                    ),
                    dtype=bool,
                )
            return np.asarray(values, dtype=bool)

        mask = np.zeros(self.row_count, dtype=bool)
        if self.rowset is None or len(self.rowset) == 0:
            return mask
        positions = self.rowset.to_host_positions(strict_disallowed=False)
        mask[np.asarray(positions, dtype=np.int64)] = True
        return mask

    def __getitem__(self, item):
        return _extension_array_getitem(self, item, self._materialize_mask)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._materialize_mask()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def __invert__(self) -> NativeBooleanMaskArray:
        """Negate the mask without crossing the native/public boundary."""
        if self.mask_values is not None:
            values = self.mask_values
            if _is_device_array(values):
                import cupy as cp

                inverted = ~cp.asarray(values, dtype=cp.bool_)
            else:
                inverted = ~np.asarray(values, dtype=bool)
        elif self.rowset is not None and _is_device_array(self.rowset.positions):
            import cupy as cp

            selected = cp.zeros(self.row_count, dtype=cp.bool_)
            selected[cp.asarray(self.rowset.positions, dtype=cp.int64)] = True
            inverted = ~selected
        else:
            inverted = ~self._materialize_mask()
        inverted_selection = None
        if _is_device_array(inverted):
            source = self.selection if self.selection is not None else self.rowset
            inverted_selection = NativeDeviceSelection.from_mask(
                inverted,
                source_token=getattr(source, "source_token", None),
                source_row_count=self.row_count,
                geometry_family_domain=getattr(
                    source,
                    "geometry_family_domain",
                    None,
                ),
                trusted_all_valid_rows=getattr(
                    source,
                    "trusted_all_valid_rows",
                    None,
                ),
            )
        return type(self)(
            row_count=self.row_count,
            selection=inverted_selection,
            mask_values=inverted,
            export_surface=self.export_surface,
            export_operation=f"{self.export_operation}_invert",
        )

    def to_numpy(self, dtype=None, copy: bool = False, na_value=False):
        values = self._materialize_mask()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def _reduce(self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs):
        if name not in {"any", "all", "sum"}:
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} does not support "
                f"operation '{name}'"
            )

        if self.rowset is not None:
            true_count = len(self.rowset)
            if name == "any":
                result = bool(true_count > 0)
            elif name == "all":
                result = bool(true_count == self.row_count)
            else:
                result = int(true_count)
        elif self.mask_values is not None:
            values = self.mask_values
            if _is_device_array(values):
                import cupy as cp

                d_values = cp.asarray(values, dtype=cp.bool_)
                if name == "any":
                    reduced = cp.any(d_values)
                    dtype = bool
                elif name == "all":
                    reduced = cp.all(d_values)
                    dtype = bool
                else:
                    reduced = cp.sum(d_values, dtype=cp.int64)
                    dtype = int
                record_native_export_boundary(
                    NativeExportBoundary(
                        surface=self.export_surface,
                        operation=f"{self.export_operation}_{name}",
                        target="scalar",
                        reason="native boolean mask reduced to public scalar",
                        row_count=self.row_count,
                        d2h_transfer=True,
                    )
                )
                result = np.asarray(
                    _copy_device_to_host(
                        cp.asarray(reduced).reshape(1),
                        reason=(f"{self.export_surface}::{self.export_operation}_{name}"),
                    )
                )[0]
                result = dtype(result)
            else:
                host_values = np.asarray(values, dtype=bool)
                if name == "any":
                    result = bool(np.any(host_values))
                elif name == "all":
                    result = bool(np.all(host_values))
                else:
                    result = int(np.sum(host_values, dtype=np.int64))
        else:
            if name == "any":
                result = False
            elif name == "all":
                result = True
            else:
                result = 0

        if keepdims:
            return type(self)(
                row_count=1,
                mask_values=np.asarray([bool(result)], dtype=bool),
                export_surface=self.export_surface,
                export_operation=self.export_operation,
            )
        return result

    def isna(self) -> np.ndarray:
        return np.zeros(self.row_count, dtype=bool)

    def copy(self) -> NativeBooleanMaskArray:
        return type(self)(
            row_count=self.row_count,
            rowset=self.rowset,
            selection=self.selection,
            mask_values=self.mask_values,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
        )

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        values = self._materialize_mask()
        if allow_fill:
            from pandas.api.extensions import take

            taken = take(
                values,
                indices,
                allow_fill=True,
                fill_value=bool(False if fill_value is None else fill_value),
            )
        else:
            taken = values[np.asarray(indices, dtype=np.intp)]
        return type(self)(
            row_count=len(taken),
            mask_values=np.asarray(taken, dtype=bool),
            export_surface=self.export_surface,
            export_operation=self.export_operation,
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        values = np.asarray(list(scalars), dtype=bool)
        return cls(row_count=int(values.size), mask_values=values.copy() if copy else values)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values)

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_mask() for array in to_concat])
        return cls(row_count=int(values.size), mask_values=values)


@dataclass
class NativeGeometryTypeArray(ExtensionArray):
    state: Any | None
    export_surface: str = "vibespatial.api.GeoSeries.geom_type"
    export_operation: str = "geoseries_geom_type"
    host_values: np.ndarray | None = None

    @property
    def dtype(self):
        return NativeGeometryTypeDtype()

    def __len__(self) -> int:
        if self.host_values is not None:
            return int(self.host_values.size)
        return int(self.state.row_count)

    def _materialize_values(self) -> np.ndarray:
        if self.host_values is not None:
            return self.host_values
        tags = _native_tags_for_state(self.state)
        if tags is None:
            raise TypeError("native geometry type array requires family tags")
        record_native_export_boundary(
            NativeExportBoundary(
                surface=self.export_surface,
                operation=self.export_operation,
                target="series",
                reason="native geometry type values exported to public Series",
                row_count=len(self),
                d2h_transfer=_is_device_array(tags),
            )
        )
        return _host_geom_type_values(tags)

    def __getitem__(self, item):
        return _extension_array_getitem(self, item, self._materialize_values)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def to_numpy(self, dtype=None, copy: bool = False, na_value=None):
        values = self._materialize_values()
        if dtype is not None:
            values = values.astype(dtype, copy=False)
        return values.copy() if copy else values

    def isna(self) -> np.ndarray:
        values = self._materialize_values()
        return pd.isna(values)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ):
        values = self._materialize_values()
        return pd.Series(values, copy=False).value_counts(
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )

    def copy(self) -> NativeGeometryTypeArray:
        return type(self)(
            state=self.state,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
            host_values=None if self.host_values is None else self.host_values.copy(),
        )

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        values = self._materialize_values()
        if allow_fill:
            from pandas.api.extensions import take

            taken = take(
                values,
                indices,
                allow_fill=True,
                fill_value=fill_value,
            )
        else:
            taken = values[np.asarray(indices, dtype=np.intp)]
        return type(self)(
            state=None,
            export_surface=self.export_surface,
            export_operation=self.export_operation,
            host_values=np.asarray(taken, dtype=object),
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        values = np.asarray(list(scalars), dtype=object)
        return cls(
            state=None,
            host_values=values.copy() if copy else values,
        )

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(
            state=None,
            export_surface=original.export_surface,
            export_operation=original.export_operation,
            host_values=np.asarray(values, dtype=object),
        )

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_values() for array in to_concat])
        return cls(state=None, host_values=np.asarray(values, dtype=object))

    def _boolean_result_from_mask(
        self,
        mask_values,
        *,
        operation: str,
        family_domain: tuple[GeometryFamily, ...] | None,
        trusted_all_valid_rows: bool | None,
    ) -> NativeBooleanMaskArray:
        if _is_device_array(mask_values):
            import cupy as cp

            d_mask = cp.asarray(mask_values, dtype=cp.bool_)
            positions = cp.flatnonzero(d_mask).astype(cp.int64, copy=False)
        else:
            mask = np.asarray(mask_values, dtype=bool)
            positions = np.flatnonzero(mask).astype(np.int64, copy=False)
            mask_values = mask
        rowset = NativeRowSet.from_positions(
            positions,
            source_token=self.state.lineage_token,
            source_row_count=self.state.row_count,
            ordered=True,
            unique=True,
            identity=int(_array_size(positions)) == int(self.state.row_count),
            geometry_family_domain=family_domain,
            trusted_all_valid_rows=trusted_all_valid_rows,
        )
        return NativeBooleanMaskArray(
            row_count=self.state.row_count,
            rowset=rowset,
            mask_values=mask_values,
            export_surface=self.export_surface,
            export_operation=operation,
        )

    def _cmp_method(self, other, op):
        symbol = _operator_symbol(op, comparison=True)
        if symbol not in {"==", "!="}:
            return NotImplemented
        if pd.api.types.is_list_like(other) and not isinstance(other, (str, bytes)):
            return op(self._materialize_values(), other)

        if self.state is None:
            return op(self._materialize_values(), other)
        tags = _native_tags_for_state(self.state)
        if tags is None:
            return op(self._materialize_values(), other)

        requested_tags = _requested_tags((other,))
        if requested_tags is None:
            family_domain = None
            trusted_all_valid_rows = None
            if _is_device_array(tags):
                import cupy as cp

                d_mask = cp.zeros(self.state.row_count, dtype=cp.bool_)
                if symbol == "!=":
                    d_mask = ~d_mask
                return self._boolean_result_from_mask(
                    d_mask,
                    operation=f"{self.export_operation}_compare",
                    family_domain=family_domain,
                    trusted_all_valid_rows=trusted_all_valid_rows,
                )
            mask = np.zeros(self.state.row_count, dtype=bool)
            if symbol == "!=":
                mask = ~mask
            return self._boolean_result_from_mask(
                mask,
                operation=f"{self.export_operation}_compare",
                family_domain=family_domain,
                trusted_all_valid_rows=trusted_all_valid_rows,
            )

        if _is_device_array(tags):
            import cupy as cp

            d_tags = cp.asarray(tags, dtype=cp.int8)
            d_mask = cp.zeros(d_tags.shape, dtype=cp.bool_)
            for tag in requested_tags:
                d_mask |= d_tags == np.int8(tag)
            if symbol == "!=":
                d_mask = ~d_mask
            mask_values = d_mask
        else:
            host_tags = np.asarray(tags, dtype=np.int8)
            mask = np.zeros(host_tags.shape, dtype=bool)
            for tag in requested_tags:
                mask |= host_tags == np.int8(tag)
            if symbol == "!=":
                mask = ~mask
            mask_values = mask

        return self._boolean_result_from_mask(
            mask_values,
            operation=f"{self.export_operation}_compare",
            family_domain=(
                _family_domain_for_requested_tags(requested_tags) if symbol == "==" else None
            ),
            trusted_all_valid_rows=(NULL_TAG not in requested_tags if symbol == "==" else None),
        )

    def __eq__(self, other):
        import operator

        return self._cmp_method(other, operator.eq)

    def __ne__(self, other):
        import operator

        return self._cmp_method(other, operator.ne)

    def isin(self, values) -> NativeBooleanMaskArray | np.ndarray:
        requested_tags = _requested_tags(values)
        tags = _native_tags_for_state(self.state)
        if requested_tags is None or tags is None:
            return np.isin(self._materialize_values(), list(values))

        if _is_device_array(tags):
            import cupy as cp

            d_tags = cp.asarray(tags, dtype=cp.int8)
            d_mask = cp.zeros(d_tags.shape, dtype=cp.bool_)
            for tag in requested_tags:
                d_mask |= d_tags == np.int8(tag)
            return self._boolean_result_from_mask(
                d_mask,
                operation=f"{self.export_operation}_isin",
                family_domain=_family_domain_for_requested_tags(requested_tags),
                trusted_all_valid_rows=NULL_TAG not in requested_tags,
            )

        host_tags = np.asarray(tags, dtype=np.int8)
        mask = np.zeros(host_tags.shape, dtype=bool)
        for tag in requested_tags:
            mask |= host_tags == np.int8(tag)
        return self._boolean_result_from_mask(
            mask,
            operation=f"{self.export_operation}_isin",
            family_domain=_family_domain_for_requested_tags(requested_tags),
            trusted_all_valid_rows=NULL_TAG not in requested_tags,
        )

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy: bool = False):
        return pd.array(list(scalars), dtype=object)

    @classmethod
    def _concat_same_type(cls, to_concat):
        values = np.concatenate([array._materialize_values() for array in to_concat])
        return pd.array(values, dtype=object)


def native_boolean_rowset_from_mask_array(mask) -> NativeRowSet | None:
    values = getattr(mask, "array", mask)
    if isinstance(values, NativeBooleanMaskArray):
        if values.rowset is None and values.selection is not None:
            values.rowset = values.selection.compact_rowset(
                surface=f"{values.export_surface}.boolean_filter",
                strict_disallowed=False,
            )
        return values.rowset
    return None


__all__ = [
    "NativeBooleanMaskArray",
    "NativeBooleanMaskDtype",
    "NativeGeometryTypeArray",
    "NativeGeometryTypeDtype",
    "NativeIndexLabelsArray",
    "NativeIndexLabelsDtype",
    "NativeNumericExpressionArray",
    "NativeNumericExpressionDtype",
    "native_boolean_rowset_from_mask_array",
    "native_geometry_type_array_supported",
    "native_public_index_from_plan",
]
