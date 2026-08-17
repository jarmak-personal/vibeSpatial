"""Public numeric reductions that preserve device-backed pandas Series."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vibespatial.api._native_expression import NativeExpression
from vibespatial.api._native_public_arrays import NativeNumericExpressionArray
from vibespatial.api.geo_base import (
    _attach_native_expression,
    _native_expression_from_public_series,
)
from vibespatial.runtime import ExecutionMode, get_requested_mode
from vibespatial.runtime.dispatch import record_dispatch_event


def _count_dtype(dtype: Any) -> np.dtype:
    normalized = np.dtype(dtype)
    if normalized not in {np.dtype(np.uint32), np.dtype(np.uint64)}:
        raise TypeError("dense_count dtype must be uint32 or uint64")
    return normalized


def _public_native_series(
    expression: NativeExpression,
    *,
    index: pd.Index,
    name: Any,
    operation: str,
) -> pd.Series:
    result = pd.Series(
        NativeNumericExpressionArray(
            expression,
            export_surface="vibespatial.api",
            export_operation=operation,
        ),
        index=index,
        name=name,
    )
    _attach_native_expression(result, expression)
    return result


def _device_any(values, *, reason: str) -> bool:
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime

    result = get_cuda_runtime().copy_device_to_host(
        cp.asarray(values, dtype=cp.bool_).reshape(1),
        reason=reason,
    )
    return bool(np.asarray(result, dtype=bool)[0])


def dense_count(
    codes,
    *,
    size: int,
    dtype: Any = np.uint32,
    name: Any = None,
) -> pd.Series:
    """Count non-negative integer codes into a fixed-size dense Series.

    This is the fixed-domain counterpart of ``numpy.bincount``. ``codes`` must
    be one-dimensional and every value must be in ``[0, size)``. Device-backed
    vibeSpatial expressions produce a device-backed real pandas Series.
    """
    group_count = int(size)
    if group_count < 0:
        raise ValueError("dense_count size must be non-negative")
    output_dtype = _count_dtype(dtype)
    expression = _native_expression_from_public_series(codes)
    requested = get_requested_mode()
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and expression is not None
        and expression.is_device
    )
    if use_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        d_codes = cp.asarray(
            expression.values if expression is not None else np.asarray(codes)
        )
        if d_codes.ndim != 1 or d_codes.dtype.kind not in {"i", "u"}:
            raise TypeError("dense_count codes must be one-dimensional integers")
        if int(d_codes.size) > int(np.iinfo(output_dtype).max):
            raise OverflowError("dense_count input exceeds the requested count dtype")
        output_bytes = group_count * int(output_dtype.itemsize)
        index_bytes = (
            0 if d_codes.dtype == cp.dtype(cp.int64) else int(d_codes.size) * 8
        )
        required_bytes = max(
            int(d_codes.size),
            output_bytes + index_bytes,
        )
        runtime = get_cuda_runtime()
        admission = runtime.admit_device_memory(
            stage="tabular-dense-count",
            required_bytes=required_bytes,
            requested_units=group_count,
        )
        if not admission.admitted:
            raise MemoryError(
                "dense_count requires "
                f"{required_bytes} device bytes with "
                f"{admission.remaining_bytes} available"
            )
        invalid = (d_codes < 0) | (d_codes >= group_count)
        if int(d_codes.size) and _device_any(
            cp.any(invalid),
            reason="public dense-count code-domain validation fence",
        ):
            raise ValueError("dense_count codes must be in [0, size)")
        del invalid
        d_indices = d_codes.astype(cp.int64, copy=False)
        d_counts = cp.zeros(group_count, dtype=output_dtype)
        if int(d_codes.size):
            cp.add.at(d_counts, d_indices, 1)
        result_expression = NativeExpression(
            operation="dense_count",
            values=d_counts,
            source_token=None,
            source_row_count=group_count,
            dtype=str(output_dtype),
            precision="exact-integer-count",
        )
        record_dispatch_event(
            surface="vibespatial.api.dense_count",
            operation="dense_count",
            implementation="cupy_dense_scatter_count",
            reason="fixed-domain integer codes reduced into a dense device count vector",
            detail=(
                f"input_rows={int(d_codes.size)}, groups={group_count}, "
                f"dtype={output_dtype.name}"
            ),
            requested=requested,
            selected=ExecutionMode.GPU,
        )
        return _public_native_series(
            result_expression,
            index=pd.RangeIndex(group_count),
            name=name,
            operation="dense_count_to_public_array",
        )

    host_codes = np.asarray(codes)
    if host_codes.ndim != 1 or host_codes.dtype.kind not in {"i", "u"}:
        raise TypeError("dense_count codes must be one-dimensional integers")
    if host_codes.size > int(np.iinfo(output_dtype).max):
        raise OverflowError("dense_count input exceeds the requested count dtype")
    if np.any((host_codes < 0) | (host_codes >= group_count)):
        raise ValueError("dense_count codes must be in [0, size)")
    counts = np.zeros(group_count, dtype=output_dtype)
    if host_codes.size:
        np.add.at(counts, host_codes.astype(np.int64, copy=False), 1)
    record_dispatch_event(
        surface="vibespatial.api.dense_count",
        operation="dense_count",
        implementation="numpy_dense_scatter_count",
        reason="host integer codes reduced into a fixed-domain dense count vector",
        detail=(
            f"input_rows={int(host_codes.size)}, groups={group_count}, "
            f"dtype={output_dtype.name}"
        ),
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return pd.Series(counts, index=pd.RangeIndex(group_count), name=name)


def numeric_take(values, indices, *, name: Any = None) -> pd.Series:
    """Gather numeric values by integer positions into a real pandas Series."""
    value_expression = _native_expression_from_public_series(values)
    index_expression = _native_expression_from_public_series(indices)
    requested = get_requested_mode()
    output_index = (
        indices.index.copy()
        if isinstance(indices, pd.Series)
        else pd.RangeIndex(len(indices))
    )
    output_name = getattr(values, "name", None) if name is None else name
    use_device = requested is ExecutionMode.GPU or (
        requested is not ExecutionMode.CPU
        and value_expression is not None
        and value_expression.is_device
    )
    if use_device:
        import cupy as cp

        from vibespatial.cuda._runtime import get_cuda_runtime

        d_values = cp.asarray(
            value_expression.values
            if value_expression is not None
            else np.asarray(values)
        )
        host_indices = None
        if index_expression is not None and index_expression.is_device:
            d_indices = cp.asarray(index_expression.values)
            source_token = index_expression.source_token
        else:
            host_indices = np.asarray(indices)
            if host_indices.ndim != 1 or host_indices.dtype.kind not in {"i", "u"}:
                raise TypeError("numeric_take indices must be one-dimensional integers")
            if np.any((host_indices < 0) | (host_indices >= int(d_values.size))):
                raise IndexError("numeric_take indices are out of bounds")
            d_indices = host_indices
            source_token = None
        if d_values.ndim != 1:
            raise TypeError("numeric_take values must be one-dimensional")
        if d_indices.ndim != 1 or d_indices.dtype.kind not in {"i", "u"}:
            raise TypeError("numeric_take indices must be one-dimensional integers")
        index_bytes = (
            int(d_indices.size) * 8
            if host_indices is not None or d_indices.dtype != cp.dtype(cp.int64)
            else 0
        )
        output_bytes = int(d_indices.size) * int(d_values.dtype.itemsize)
        required_bytes = max(
            int(d_indices.size),
            output_bytes + index_bytes,
        )
        runtime = get_cuda_runtime()
        admission = runtime.admit_device_memory(
            stage="tabular-numeric-take",
            required_bytes=required_bytes,
            requested_units=int(d_indices.size),
        )
        if not admission.admitted:
            raise MemoryError(
                "numeric_take requires "
                f"{required_bytes} device bytes with "
                f"{admission.remaining_bytes} available"
            )
        if host_indices is not None:
            d_indices = cp.asarray(host_indices, dtype=cp.int64)
        invalid = (d_indices < 0) | (d_indices >= int(d_values.size))
        if int(d_indices.size) and _device_any(
            cp.any(invalid),
            reason="public numeric-take position-domain validation fence",
        ):
            raise IndexError("numeric_take indices are out of bounds")
        del invalid
        d_indices = d_indices.astype(cp.int64, copy=False)
        gathered = d_values[d_indices]
        result_expression = NativeExpression(
            operation="numeric_take",
            values=gathered,
            source_token=source_token,
            source_row_count=int(d_indices.size),
            dtype=str(gathered.dtype),
            precision=(
                value_expression.precision
                if value_expression is not None
                else "host-to-device-numeric-gather"
            ),
            readiness=(
                value_expression.readiness
                if value_expression is not None
                else None
            ),
        )
        record_dispatch_event(
            surface="vibespatial.api.numeric_take",
            operation="numeric_take",
            implementation="cupy_numeric_gather",
            reason="device integer positions gathered a device numeric Series",
            detail=f"source_rows={int(d_values.size)}, output_rows={int(d_indices.size)}",
            requested=requested,
            selected=ExecutionMode.GPU,
        )
        return _public_native_series(
            result_expression,
            index=output_index,
            name=output_name,
            operation="numeric_take_to_public_array",
        )

    host_values = np.asarray(values)
    host_indices = np.asarray(indices)
    if host_values.ndim != 1:
        raise TypeError("numeric_take values must be one-dimensional")
    if host_indices.ndim != 1 or host_indices.dtype.kind not in {"i", "u"}:
        raise TypeError("numeric_take indices must be one-dimensional integers")
    if np.any((host_indices < 0) | (host_indices >= host_values.size)):
        raise IndexError("numeric_take indices are out of bounds")
    result = host_values[host_indices.astype(np.int64, copy=False)]
    record_dispatch_event(
        surface="vibespatial.api.numeric_take",
        operation="numeric_take",
        implementation="numpy_numeric_gather",
        reason="host integer positions gathered a host numeric Series",
        detail=f"source_rows={int(host_values.size)}, output_rows={int(host_indices.size)}",
        requested=requested,
        selected=ExecutionMode.CPU,
    )
    return pd.Series(result, index=output_index, name=output_name)


__all__ = ["dense_count", "numeric_take"]
